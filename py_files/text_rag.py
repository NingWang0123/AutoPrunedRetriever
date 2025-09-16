# text_in_cr.py
from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import time, numpy as np, re, json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from CompressRag_rl_v1 import CompressRag_rl

def _as_unit(v: np.ndarray | List[float]) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v) + 1e-12)
    return v / n

class TextOnlyCR(CompressRag_rl):
    """
    让 Text RAG 走 CompressRag_rl 的统一接口：
      - build_text_index(): 用 seed questions 生成简单 QA 文档并建 FAISS
      - run_work_flow():    文本向量检索 + 统一的 llm.take_questions 调用
    注意：
      - 该类不会构图；meta_codebook 中不要求 e/r/edge_matrix（Graph-only）
      - 仍然会维护 _last_ctx、_last_retrieval_sec，方便 cost 统计
      - collect_results() 复用父类（你之前加过记录 metrics 的版本即可）
    """
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.text_db: Optional[FAISS] = None
        self.text_cfg: Dict[str, Any] = {
            "faiss_search_k": 3,
            "include_retrieved_context": True,
            "use_cached_text_embeddings": True,
        }
        self._qvec_cache: Dict[str, np.ndarray] = {}

    # ------------------- 索引构建 -------------------
    def build_text_index(self, questions: List[str], gen_fn) -> FAISS:
        """
        用 seed questions 生成一个简易 QA 库（问句做向量，答案由 gen_fn 直接生成一次用于入库）。
        """
        docs: List[Document] = []
        for i, q in enumerate(questions, 1):
            # 直接生成一个短答入库（不依赖向量库）
            # 这里不用 llm.take_questions，避免循环依赖；只要能产出一个可参考的 prior answer 即可
            prior_ans = gen_fn(f"[QUESTION]: {q}\n[ANSWER]: ").strip()
            qv = _as_unit(np.asarray(self.sentence_emb.embed_query(q), dtype=np.float32))
            meta = {
                "graph_id": f"Q{i}",
                "question": q,
                "llm_model": getattr(self.llm, "model_name", "unknown"),
                "llm_answer": prior_ans,
                "created_at": int(time.time()),
                "q_embeddings": qv.tolist(),
            }
            docs.append(Document(page_content=q, metadata=meta))

        self.text_db = self._build_faiss_from_cached(docs)
        return self.text_db

    def _build_faiss_from_cached(self, docs: List[Document]) -> FAISS:
        texts, metas, vecs = [], [], []
        for d in docs:
            texts.append(d.page_content)
            metas.append(d.metadata)
            v = d.metadata.get("q_embeddings") or self.sentence_emb.embed_query(d.page_content)
            vecs.append(_as_unit(np.asarray(v, dtype=np.float32)))
        X = np.vstack(vecs).astype(np.float32)
        text_embeddings = [(t, X[i].tolist()) for i, t in enumerate(texts)]
        try:
            return FAISS.from_embeddings(text_embeddings, embedding=self.sentence_emb, metadatas=metas)
        except TypeError:
            return FAISS.from_embeddings(embeddings=X.tolist(), metadatas=metas, texts=texts, embedding=self.sentence_emb)

    def save_text_index(self, out_dir: str = "faiss_text_idx"):
        if not self.text_db:
            print("[TextOnlyCR] Skipping save: no vectorstore.")
            return
        try:
            self.text_db.save_local(out_dir)
        except Exception:
            self.text_db.save_local(folder_path=out_dir)
        print(f"[TextOnlyCR] FAISS saved → {out_dir}")

    # ------------------- 检索 -------------------
    def _faiss_search_by_vec(self, vs: FAISS, qv: np.ndarray, k: int):
        if hasattr(vs, "similarity_search_by_vector_with_score"):
            return vs.similarity_search_by_vector_with_score(qv, k=k)
        if hasattr(vs, "similarity_search_by_vector"):
            docs = vs.similarity_search_by_vector(qv, k=k)
            return [(d, None) for d in docs]
        index = getattr(vs, "index", None)
        id_map = getattr(vs, "index_to_docstore_id", None)
        store  = getattr(vs, "docstore", None)
        if index is None or id_map is None or store is None:
            raise AttributeError("FAISS missing raw handles.")
        D, I = index.search(qv.reshape(1, -1).astype(np.float32), k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1:
                continue
            doc_id = id_map[idx]
            doc = store.search(doc_id)
            out.append((doc, float(dist)))
        return out

    def _similarity_search_text_docs(self, question: str, k: int) -> List[Tuple[Document, Optional[float]]]:
        if self.text_db is None:
            return []
        # 缓存 query 向量
        if self.text_cfg["use_cached_text_embeddings"] and question in self._qvec_cache:
            qv = self._qvec_cache[question]
        else:
            qv = _as_unit(np.asarray(self.sentence_emb.embed_query(question), dtype=np.float32))
            if self.text_cfg["use_cached_text_embeddings"]:
                self._qvec_cache[question] = qv
        return self._faiss_search_by_vec(self.text_db, qv, k)

    # ------------------- 核心：统一 run_work_flow -------------------
    def run_work_flow(self, question: str, **kw):
        """
        Text RAG 的工作流：
        - 做向量检索（问题->相似问题）
        - 把检索到的问题+其 prior answer 组织成纯文本上下文，放进 final_merged_json
        - 调用 llm.take_questions(final_merged_json, question)（与 Graph 分支一致）
        - 依旧让 collect_results 来记录 cost、_last_ctx 等
        """
        t0 = time.time()
        hits = self._similarity_search_text_docs(question, k=getattr(self, "top_k", 3))
        rec_t = time.time() - t0

        # 把检索命中的“相关问答”变成可读上下文，供 LLM 参考
        ctx_strings: List[str] = []
        for d, _ in hits:
            q_txt = d.page_content.strip()
            a_txt = (d.metadata or {}).get("llm_answer", "")
            ctx_strings.append(f"Q: {q_txt}\nA: {a_txt}".strip())

        # 让父类后处理时能拿到最近上下文/耗时
        self._last_ctx = ctx_strings
        self._last_retrieval_sec = rec_t

        # Text 模式的“合并结果”——我们不构图，所以给一个纯文本 KB
        final_merged_json: Dict[str, Any] = {
            "mode": "text",
            "retrieved_qa": ctx_strings,
            "k": len(ctx_strings),
        }

        # 走你在父类里改过的 collect_results（它会调用 llm.take_questions 并记录 metrics）
        # 注意：你在 CompressRag_rl_v1 里已有 collect_results(…)，且我们希望输出完全兼容 Graph 分支
        return self.collect_results(final_merged_json, question)
