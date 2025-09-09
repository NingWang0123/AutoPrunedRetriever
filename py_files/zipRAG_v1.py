import re
import ast
import inspect
import numpy as np
import networkx as nx
from typing import List, Dict, Optional, Tuple, Any

# --- External deps you already have in your project ---
from graph_generator.graphparsers import RelationshipGraphParser
from rag_workflow_v1 import (
    get_code_book, merging_codebook, combine_ents,
    coarse_filter, add_answers_to_filtered_lst, get_flat_answers_lsts,
    common_contiguous_overlaps, get_unique_knowledge, get_json_with_given_knowledge
)
from groupwords import merge_graph_nodes_by_canonical, normalize_text
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from collections import defaultdict
import time, os, math

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# Lightweight word-avg embeddings (kept as-is)
# =========================
from gensim.models import KeyedVectors

class WordAvgEmbeddings(Embeddings):
    def __init__(self, model_path: str = "gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model"):
        # NOTE: your original default pointed to *.npy — here we load the model path you passed in your script.
        # Keep your path as-is when constructing this.
        self.kv = KeyedVectors.load(model_path, mmap='r')
        self.dim = self.kv.vector_size
        self.token_pat = re.compile(r"[A-Za-z]+")

    def _embed_text(self, text: str) -> np.ndarray:
        toks = [t.lower() for t in self.token_pat.findall(text)]
        vecs = [self.kv[w] for w in toks if w in self.kv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    def embed_documents(self, texts):
        return [self._embed_text(t) for t in texts]

    def embed_query(self, text):
        return self._embed_text(text)


# =========================
# Main Workflow Class
# =========================
class RAGWorkflow:
    """
    A unified, swappable RAG workflow that preserves your original implementation details:
    - Text RAG and Graph RAG
    - Incremental doc upsert after answering each question
    - Cached-vector FAISS compatibility (old and new signatures)
    - Prompt construction and <think> scrubbing
    - Graph question parsing (RelationshipGraphParser / CausalQuestionGraphParser)
    """

    DEFAULT_CONFIG = {
        # Embedding & VectorStore
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "faiss_search_k": 3,

        # LLM (text generation)
        "llm_model_id": "microsoft/Phi-4-mini-reasoning",
        "device_map": "auto",
        "dtype_policy": "auto",   # "auto" | "bf16" | "fp16" | "fp32"
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.4,
        "top_p": 1.0,
        "return_full_text": False,
        "seed": None,

        # Answer format
        "answer_mode": "short",       # "yes_no" / "binary" / "short" / "detail" / "long" / "reasoning" / "explain"
        "answer_uppercase": True,     # YES/NO vs yes/no

        # Prompt construction / retrieval
        "include_retrieved_context": True,
        "use_cached_text_embeddings": True,
        "use_cached_graph_embeddings": True,

        # Deprecated knobs (kept for backwards compatibility; no-op):
        "include_current_triples": True,   # [deprecated/no-op here]
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        *,
        sentence_emb: Optional[Embeddings] = None,
        word_emb: Optional[Embeddings] = None,
        verbose: bool = False
    ):
        # Config
        self.config = dict(self.DEFAULT_CONFIG)
        if config:
            self.config.update(config)

        # Embeddings
        self.sentence_emb = sentence_emb or HuggingFaceEmbeddings(model_name=self.config["embedding_model"])
        self.word_emb = word_emb or WordAvgEmbeddings()

        # LLM pipeline
        self.gen_pipe = None
        self.tokenizer = None

        # FAISS stores (user can manage externally, too)
        self.text_db: Optional[FAISS] = None
        self.graph_db: Optional[FAISS] = None

        # Query caches
        self._QVEC_CACHE: Dict[str, np.ndarray] = {}
        self._GRAPH_QVEC_CACHE: Dict[str, np.ndarray] = {}

        # Misc
        self.verbose = verbose
        self._stats = defaultdict(list)
        self._ensure_seed()
        self._metrics = {"graph": [], "text": []}  


    # ---------- Public API: quick switches ----------
    def set_params(self, **kwargs):
        """Update configuration on the fly (e.g., temperature, top_p, answer_mode...)."""
        self.config.update(kwargs)

    def set_embeddings(self, sentence_emb: Optional[Embeddings] = None, word_emb: Optional[Embeddings] = None):
        if sentence_emb is not None:
            self.sentence_emb = sentence_emb
        if word_emb is not None:
            self.word_emb = word_emb

    def set_llm(self, model_id: Optional[str] = None, **gen_overrides):
        """Load/replace the LLM pipeline with overrides."""
        self.gen_pipe, self.tokenizer = self._load_llm_pipeline(model_id=model_id, **gen_overrides)

    # ---------- LLM ----------
    def _ensure_seed(self):
        seed = self.config.get("seed", None)
        if seed is None:
            return
        try:
            from transformers import set_seed
            if isinstance(seed, int):
                set_seed(seed)
        except Exception:
            pass

    def _select_dtype(self) -> torch.dtype:
        policy = self.config.get("dtype_policy", "auto")
        if policy == "bf16":
            return torch.bfloat16
        if policy == "fp16":
            return torch.float16
        if policy == "fp32":
            return torch.float32
        # auto
        if torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if torch.backends.mps.is_available():
            return torch.float32
        return torch.float32

    def _load_llm_pipeline(
        self,
        model_id: Optional[str] = None,
        *,
        device_map: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        return_full_text: Optional[bool] = None,
    ):
        cfg = self.config
        model_id = model_id or cfg["llm_model_id"]
        device_map = device_map or cfg["device_map"]
        dtype = dtype or self._select_dtype()
        max_new_tokens = max_new_tokens or cfg["max_new_tokens"]
        temperature = cfg["temperature"] if temperature is None else temperature
        top_p = cfg["top_p"] if top_p is None else top_p
        do_sample = cfg["do_sample"] if do_sample is None else do_sample
        return_full_text = cfg["return_full_text"] if return_full_text is None else return_full_text

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        gen_pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map=device_map,
            torch_dtype=dtype,
            return_full_text=return_full_text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )
        return gen_pipe, tokenizer

    # ---------- Common small helpers ----------
    def _mode(self) -> str:
        return (self.config.get("answer_mode") or "short").lower()

    def _yn(self, text_yes="YES", text_no="NO"):
        return (text_yes, text_no) if self.config.get("answer_uppercase", True) else (text_yes.lower(), text_no.lower())

    @staticmethod
    def _avg_pool(mat: np.ndarray) -> Optional[np.ndarray]:
        if mat is None or len(mat) == 0:
            return None
        m = np.asarray(mat, dtype=np.float32)
        if m.ndim == 1:
            return m.astype(np.float32)
        return m.mean(axis=0).astype(np.float32)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def _graph_doc_vec_from_cached_or_embed(
        self,
        er_e: List[str],
        er_r: List[str],
        e_embeds: Optional[List] = None,
        r_embeds: Optional[List] = None,
        *,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        文档向量 = 归一化((avg(e_embeds) + avg(r_embeds))/2).
        若无缓存：分别对每个 e/r 做 word_emb.embed_query → 平均 → 融合。
        """
        if use_cache and e_embeds and r_embeds:
            e_mean = self._avg_pool(np.asarray(e_embeds, dtype=np.float32))
            r_mean = self._avg_pool(np.asarray(r_embeds, dtype=np.float32))
            v = (e_mean + r_mean) / 2.0
            return self._normalize(v)

        e_vecs = [np.asarray(self.word_emb.embed_query(e), dtype=np.float32) for e in (er_e or [])]
        r_vecs = [np.asarray(self.word_emb.embed_query(r), dtype=np.float32) for r in (er_r or [])]
        e_mean = self._avg_pool(np.stack(e_vecs, axis=0)) if e_vecs else None
        r_mean = self._avg_pool(np.stack(r_vecs, axis=0)) if r_vecs else None

        if e_mean is None and r_mean is None:
            dim = getattr(self.word_emb, "dim", None) or len(self.word_emb.embed_query("a"))
            return np.zeros(dim, dtype=np.float32)

        v = r_mean if e_mean is None else (e_mean if r_mean is None else (e_mean + r_mean) / 2.0)
        return self._normalize(v)

    # ---------- Graph parsing ----------
    @staticmethod
    def parse_question_to_graph_generic(parser, question: str) -> Tuple[nx.Graph, List[Dict]]:
        if hasattr(parser, "question_to_graph"):
            G, rels = parser.question_to_graph(question)
        elif hasattr(parser, "question_to_causal_graph"):
            G, rels = parser.question_to_causal_graph(question)
        else:
            raise AttributeError("Parser must provide question_to_graph or question_to_causal_graph")

        G = merge_graph_nodes_by_canonical(G, normalizer=normalize_text, merge_edge_attrs=("relation",))
        return G, rels

    # ---------- Prompt builders ----------
    def make_text_qa_prompt(self, question: str, retrieved_docs=None) -> str:
        cfg = self.config
        sections = []
        if retrieved_docs and cfg.get("include_retrieved_context", True):
            doc0, _ = retrieved_docs[0]
            related_q_txt = doc0.page_content.strip()
            related_answer = (doc0.metadata or {}).get("llm_answer", "")
            sections.append(
                "<<<RETRIEVED_CONTEXT_START>>>\n"
                "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. "
                "You don't have to follow it completely, just use it as a reference.\n"
                f"[RELATED QUESTION TEXT]:\n{related_q_txt}\n"
                f"[RELATED ANSWER]: {related_answer}\n"
                "<<<RETRIEVED_CONTEXT_END>>>"
            )

        sections.append(f"[CURRENT QUESTION]: {question}")

        mode = self._mode()
        if mode in {"yes_no", "binary"}:
            yes, no = self._yn("YES", "NO")
            rules = (
                "[TASK]: You are a precise QA assistant for binary (yes/no) questions.\n"
                f"- Output ONLY one token: {yes} or {no}.\n"
                "- Do NOT copy or summarize any context.\n"
                "- Do NOT show reasoning, steps, or extra words.\n"
                "[ANSWER]: "
            )
        else:
            style_line = {
                "short":    "- Give a short, direct answer in 2–3 sentences.\n",
                "detail":   "- Provide a clear, detailed, and structured answer.\n",
                "long":     "- Provide a clear, detailed, and structured answer.\n",
                "reasoning":"- Provide a well-structured explanation with logical reasoning flow.\n- If useful, break the answer into brief sections.\n",
                "explain":  "- Provide a well-structured explanation with logical reasoning flow.\n- If useful, break the answer into brief sections.\n",
            }.get(mode, "- Provide a clear and helpful answer.\n")

            rules = (
                "[TASK]: You are a QA assistant for open-ended questions.\n"
                f"{style_line}"
                "- Do NOT restrict to yes/no.\n"
                "[FORMAT]: Write complete sentences (not a single word)."
                "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, state the conclusion AND 1–2 short reasons.\n"
                "[ANSWER]: "
            )

        sections.append(rules)
        return "\n\n".join(sections)

    def make_graph_qa_prompt(
        self,
        question: str,
        G: nx.Graph,
        relations: Optional[List[Dict]] = None,
        retrieved_docs = None
    ) -> Tuple[str, Dict]:
        cfg = self.config
        sections = []
        compact_json = {}

        if retrieved_docs and cfg.get("include_retrieved_context", True):
            doc0, _ = retrieved_docs[0]
            metadata = doc0.metadata or {}
            codebook_main = (metadata.get("codebook_main") or {})

            query_chains = []
            for group_idx, group in enumerate(codebook_main.get("questions_lst", [])):
                for q_idx, question_chain in enumerate(group):
                    query_chains.append(question_chain)

            related_triples = "__EMPTY_JSON__"
            compact_json = {}
            if query_chains:
                wrapper_res = coarse_filter(
                    questions=query_chains,
                    codebook_main=codebook_main,
                    emb=self.sentence_emb,
                    top_k=3,
                    question_batch_size=2,
                    questions_db_batch_size=8,
                    top_m=3,
                )

                if isinstance(wrapper_res, dict) and wrapper_res:
                    first_non_empty = next((lst for lst in wrapper_res.values() if lst), [])
                    if first_non_empty:
                        related_triples = first_non_empty[0].get("text", "__EMPTY_JSON__")

                        topm_with_answers = add_answers_to_filtered_lst(wrapper_res, codebook_main) or []
                        first_bucket = topm_with_answers[0] if topm_with_answers else []

                        flat_lists = get_flat_answers_lsts(
                            [ (it.get('answers(edges[i])') or []) for it in first_bucket ]
                        )
                        overlaps   = common_contiguous_overlaps(flat_lists, min_len=2)
                        uniq_res   = get_unique_knowledge({'overlaps': overlaps}, flat_lists)

                        query_chains_flat = [list(chain) for chain in query_chains]

                        if all(k in codebook_main for k in ('e','r','edge_matrix')):
                            try:
                                compact_json = get_json_with_given_knowledge(
                                    flat_answers_lsts = uniq_res.get('out_answers', []),
                                    codebook_main     = codebook_main,
                                    codebook_sub_q    = {
                                        'e': codebook_main['e'],
                                        'r': codebook_main['r'],
                                        'edge_matrix': codebook_main['edge_matrix'],
                                        'questions(edges[i])': query_chains_flat,
                                        'rule': codebook_main.get('rule', ''),
                                    },
                                    decode = True
                                )
                            except Exception as e:
                   
                                if self.verbose:
                                    print(f"[WARN] get_json_with_given_knowledge failed: {e}")
                                compact_json = {}


            related_answer  = (doc0.metadata or {}).get("llm_answer", "")
            if related_triples != "__EMPTY_JSON__":
                sections.append(
                    "<<<RETRIEVED_CONTEXT_START>>>\n"
                    "The system searched for a related question in the database. Below are related question's graph triples and its prior answer as reference. "
                    "You don't have to follow it completely, just use it as a reference.\n"
                    f"[RELATED QUESTION'S GRAPH TRIPLES]:\n{related_triples}\n"
                    f"[RELATED QUESTION'S ANSWER]: {related_answer}\n"
                    "<<<RETRIEVED_CONTEXT_END>>>"
                )

        sections.append(f"[CURRENT QUESTION]: {question}")

        mode = self._mode()
        if mode in {"yes_no", "binary"}:
            yes, no = self._yn("YES", "NO")
            rules = (
                "[TASK]: You are a precise QA assistant for binary (yes/no) questions.\n"
                f"- Output ONLY one token: {yes} or {no}.\n"
                "- Do NOT copy or summarize any context.\n"
                "- Do NOT show reasoning, steps, or extra words.\n"
                "[ANSWER]: "
            )
        else:
            style_line = {
                "short":    "- Give a short, direct answer in 2–3 sentences.\n",
                "detail":   "- Provide a clear, detailed, and structured answer.\n",
                "long":     "- Provide a clear, detailed, and structured answer.\n",
                "reasoning":"- Provide a well-structured explanation with logical reasoning flow.\n- If useful, break the answer into brief sections.\n",
                "explain":  "- Provide a well-structured explanation with logical reasoning flow.\n- If useful, break the answer into brief sections.\n",
            }.get(mode, "- Provide a clear and helpful answer.\n")

            rules = (
                "[TASK]: You are a QA assistant for open-ended questions.\n"
                f"{style_line}"
                "- Do NOT restrict to yes/no.\n"
                "[FORMAT]: Write complete sentences (not a single word)."
                "Avoid starting with just 'Yes.' or 'No.'; if the question is yes/no-style, state the conclusion AND 1–2 short reasons.\n"
                "[ANSWER]: "
            )

        sections.append(rules)
        return "\n\n".join(sections), compact_json

    # ---------- Generation ----------
    def _gen(self, prompt: str) -> str:
        if self.gen_pipe is None:
            # Lazily load with current config if not already set
            self.set_llm()
        kwargs = dict(
            do_sample=self.config.get("do_sample", True),
            temperature=self.config.get("temperature", 0.4),
            top_p=self.config.get("top_p", 1.0),
            max_new_tokens=self.config.get("max_new_tokens", 256),
            return_full_text=self.config.get("return_full_text", False),
        )
        # pad/eos compatibility
        try:
            tok = self.gen_pipe.tokenizer
            if tok is not None:
                if tok.pad_token_id is None and tok.eos_token_id is not None:
                    tok.pad_token_id = tok.eos_token_id
                kwargs.setdefault("eos_token_id", tok.eos_token_id)
                kwargs.setdefault("pad_token_id", tok.pad_token_id)
        except Exception:
            pass

        out = self.gen_pipe(prompt, **kwargs)
        return out[0]["generated_text"]

    def _extract_answer_text(self, prompt: str, text: str) -> str:
        if self.config.get("return_full_text", False):
            return text[len(prompt):].strip()
        return text.strip()

    @staticmethod
    def strip_think(s: str) -> Tuple[str, List[str]]:
        """Remove <think>...</think> blocks; also handle dangling <think> at EOF."""
        if not s:
            return "", []
        s_lower = s.lower()
        thinks: List[str] = []
        spans: List[Tuple[int, int]] = []

        for m in re.finditer(r"<think>(.*?)</think>", s, flags=re.S | re.I):
            thinks.append(m.group(1).strip())
            spans.append((m.start(), m.end()))

        last_open = s_lower.rfind("<think>")
        if last_open != -1 and s_lower.find("</think>", last_open) == -1:
            content_start = last_open + len("<think>")
            dangling_text = s[content_start:].strip()
            if dangling_text:
                thinks.append(dangling_text)
            spans.append((last_open, len(s)))

        if spans:
            spans.sort()
            merged = []
            cur_s, cur_e = spans[0]
            for st, en in spans[1:]:
                if st <= cur_e:
                    cur_e = max(cur_e, en)
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = st, en
            merged.append((cur_s, cur_e))
        else:
            merged = []

        parts = []
        prev = 0
        for st, en in merged:
            if prev < st:
                parts.append(s[prev:st])
            prev = en
        if prev < len(s):
            parts.append(s[prev:])

        clean = "".join(parts)
        clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*", "", clean, flags=re.I)
        return clean.strip(), thinks

    # ---------- Text RAG ----------
    def _faiss_search_by_vec(self, vs: FAISS, qv: np.ndarray, k: int):
        if hasattr(vs, "similarity_search_by_vector_with_score"):
            return vs.similarity_search_by_vector_with_score(qv, k=k)
        if hasattr(vs, "similarity_search_by_vector"):
            docs = vs.similarity_search_by_vector(qv, k=k)
            return [(d, None) for d in docs]
        # Fallback
        index = getattr(vs, "index", None)
        id_map = getattr(vs, "index_to_docstore_id", None)
        store  = getattr(vs, "docstore", None)
        if index is None or id_map is None or store is None:
            raise AttributeError("FAISS vectorstore has no by-vector APIs and no accessible index/docstore.")
        q = np.asarray(qv, dtype=np.float32).reshape(1, -1)
        D, I = index.search(q, k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            doc_id = id_map[idx]
            doc = store.search(doc_id)
            out.append((doc, float(dist)))
        return out

    def similarity_search_text_docs(
        self,
        user_question: str,
        vectordb: FAISS,
        k: int = 5,
        query_vec: Optional[List[float]] = None,
        emb: Optional[Embeddings] = None,
        use_cache: Optional[bool] = None,
    ):
        cfg = self.config
        use_cache = cfg.get("use_cached_text_embeddings", True) if use_cache is None else use_cache

        if query_vec is not None and use_cache:
            qv = np.asarray(query_vec, dtype=np.float32)
            qv /= (np.linalg.norm(qv) + 1e-12)
            return user_question, self._faiss_search_by_vec(vectordb, qv, k)

        if use_cache and user_question in self._QVEC_CACHE:
            qv = self._QVEC_CACHE[user_question]
            return user_question, self._faiss_search_by_vec(vectordb, qv, k)

        emb = emb or self.sentence_emb
        qv = np.asarray(emb.embed_query(user_question), dtype=np.float32)
        qv /= (np.linalg.norm(qv) + 1e-12)

        if use_cache:
            self._QVEC_CACHE[user_question] = qv

        return user_question, self._faiss_search_by_vec(vectordb, qv, k)

    def answer_with_llm_text(
        self,
        question: str,
        *,
        text_db: Optional["FAISS"] = None,
        max_retries: int = 3,
    ) -> str:
        total_t0 = time.time()

        retrieval_latency = 0.0
        retrieved_docs = None
        retrieved_count = 0
        if text_db:
            t0 = time.time()
            _, hits = self.similarity_search_text_docs(
                question, text_db, k=self.config.get("faiss_search_k", 3),
                emb=self.sentence_emb,
                use_cache=self.config.get("use_cached_text_embeddings", True)
            )
            retrieval_latency = time.time() - t0
            retrieved_docs = hits
            retrieved_count = len(hits) if hits else 0

        prompt = self.make_text_qa_prompt(question, retrieved_docs)
        mode = self._mode()

        YES_RE = re.compile(r"^\s*(yes|y|true|correct|affirmative)\s*\.?\s*$", re.I)
        NO_RE  = re.compile(r"^\s*(no|n|false|incorrect|negative)\s*\.?\s*$", re.I)

        gen_latency_total = 0.0
        peak_vram = 0.0
        final_answer = ""

        attempt = 0
        while attempt < max_retries:
            attempt += 1

            if torch.cuda.is_available():
                try: torch.cuda.reset_peak_memory_stats()
                except Exception: pass

            g0 = time.time()
            raw = self._gen(prompt)
            gen_latency = time.time() - g0
            gen_latency_total += gen_latency

            if torch.cuda.is_available():
                try:
                    peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / (1024**2))
                except Exception:
                    pass

            if self.verbose:
                print(f"----- RAW (try {attempt}):", raw)

            text = self._extract_answer_text(prompt, raw)
            answer, _ = self.strip_think(text)
            answer = (answer or "").strip()
            if self.verbose:
                print("----- ANS:", answer)

            if not answer:
                continue

            if mode in {"yes_no", "binary"}:
                if YES_RE.match(answer) and not NO_RE.match(answer):
                    final_answer = self._yn("YES", "NO")[0]; break
                if NO_RE.match(answer) and not YES_RE.match(answer):
                    final_answer = self._yn("YES", "NO")[1]; break

                strict_suffix = "\n\n[FORMAT]: Answer with exactly ONE token: " + \
                                ("YES or NO." if self.config.get("answer_uppercase", True) else "yes or no.")
                g0 = time.time()
                raw2 = self._gen(prompt + strict_suffix)
                gen_latency_total += (time.time() - g0)
                ans2 = self._extract_answer_text(prompt + strict_suffix, raw2)
                ans2, _ = self.strip_think(ans2)
                if YES_RE.match(ans2) and not NO_RE.match(ans2):
                    final_answer = self._yn("YES", "NO")[0]; break
                if NO_RE.match(ans2) and not YES_RE.match(ans2):
                    final_answer = self._yn("YES", "NO")[1]; break
                final_answer = ans2.strip() if ans2 else answer
                break
            else:
                if YES_RE.match(answer) or NO_RE.match(answer) or len(answer.split()) <= 2:
                    format_suffix = "\n\n[FORMAT]: Provide a 2–3 sentence explanation; do not answer with a single word."
                    g0 = time.time()
                    raw2 = self._gen(prompt + format_suffix)
                    gen_latency_total += (time.time() - g0)
                    ans2 = self._extract_answer_text(prompt + format_suffix, raw2)
                    ans2, _ = self.strip_think(ans2)
                    if ans2 and len(ans2.strip()) > len(answer):
                        final_answer = ans2.strip(); break
                final_answer = answer; break

        total_latency = time.time() - total_t0

        m = {
            "input_tokens":  self._num_tokens(prompt),
            "output_tokens": self._num_tokens(final_answer),
            "total_tokens":  self._num_tokens(prompt) + self._num_tokens(final_answer),
            "latency_sec": total_latency,
            "retrieval_latency_sec": retrieval_latency,
            "gen_latency_sec": gen_latency_total,
            "retrieved_count": retrieved_count,
            "peak_vram_MiB": float(peak_vram),
            "prompt_chars": len(prompt),
        }
        self._record_metric("text", m)
        return final_answer

    def build_text_docs_with_answer(
        self,
        questions: List[str],
        *,
        add_prompt_snapshot: bool = False,
        text_db: Optional[FAISS] = None
    ) -> Tuple[List[Document], np.ndarray]:
        docs: List[Document] = []
        q_vec = None
        for qid, q in enumerate(questions, start=1):
            q_vec = self.sentence_emb.embed_query(q)
            answer = self.answer_with_llm_text(q, text_db=text_db)
            metadata = {
                "graph_id": f"Q{qid}",
                "question": q,
                "llm_model": self.config["llm_model_id"],
                "llm_answer": answer,
                "created_at": int(time.time()),
                "q_embeddings": q_vec
            }
            if add_prompt_snapshot:
                prompt_snapshot = self.make_text_qa_prompt(q, None if not text_db else self.similarity_search_text_docs(q, text_db, k=self.config.get("faiss_search_k",3))[1])
                metadata["prompt_snapshot"] = prompt_snapshot
            
            doc = Document(page_content=q, metadata=metadata)
            text_db = self.upsert_text_docs_into_faiss([doc], emb=self.sentence_emb, text_db=text_db)
            self.text_db = text_db
        return text_db, q_vec

    def upsert_text_docs_into_faiss(
        self,
        new_docs: List[Document],
        *,
        emb: Optional[Embeddings] = None,
        text_db: Optional[FAISS] = None,
    ) -> FAISS:
        emb = emb or self.sentence_emb
        use_cache = self.config.get("use_cached_text_embeddings", True)

        texts, metas, vecs = [], [], []
        for d in new_docs:
            texts.append(d.page_content)
            meta = d.metadata or {}
            metas.append(meta)

            # 取缓存或现算，并归一化（与批量构建一致）
            v = None
            if use_cache:
                v = meta.get("q_embeddings") or meta.get("q_vec")
            if v is None:
                v = emb.embed_query(d.page_content)
            v = np.asarray(v, dtype=np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            vecs.append(v.tolist())
            # 顺便存一份标准键，便于以后重用
            meta["q_vec"] = v.tolist()

        X = np.asarray(vecs, dtype=np.float32)

        if text_db is None:
            text_embeddings = [(texts[i], X[i].tolist()) for i in range(len(texts))]
            try:
                return FAISS.from_embeddings(text_embeddings, embedding=emb, metadatas=metas)
            except TypeError:
                try:
                    return FAISS.from_embeddings(
                        embeddings=X.tolist(), metadatas=metas, texts=texts, embedding=emb
                    )
                except Exception:
                    vs = FAISS.from_texts(texts=[], embedding=emb)
                    if hasattr(vs, "add_embeddings"):
                        vs.add_embeddings(embeddings=X, metadatas=metas, texts=texts)
                    else:
                        vs.add_texts(texts=texts, metadatas=metas)
                    return vs
        if hasattr(text_db, "add_embeddings"):
            add_fn = text_db.add_embeddings
            try:
        
                text_embeddings = [(texts[i], X[i].tolist()) for i in range(len(texts))]
                add_fn(text_embeddings=text_embeddings, metadatas=metas)
            except TypeError:
     
                add_fn(embeddings=X.tolist(), metadatas=metas, texts=texts)
        elif hasattr(text_db, "add_texts"):
  
            text_db.add_texts(texts=texts, metadatas=metas)
        elif hasattr(text_db, "add_documents"):
            text_db.add_documents(new_docs)
        else:
            raise AttributeError("FAISS vectorstore has no add_* method to append data.")

        return text_db

    def build_text_index_from_cached(self, docs: List[Document], emb: Optional[Embeddings] = None) -> FAISS:
        use_cache = self.config.get("use_cached_text_embeddings", True)
        emb = emb or self.sentence_emb

        texts, metas, vecs = [], [], []
        for d in docs:
            texts.append(d.page_content)
            metas.append(d.metadata)
            if use_cache:
                v = d.metadata.get("q_embeddings", None) or d.metadata.get("q_vec", None)
                if v is None:
                    v = emb.embed_query(d.page_content)
            else:
                v = emb.embed_query(d.page_content)
            vecs.append(v)

        if not texts:
            raise ValueError("No docs provided to build_text_index_from_cached().")

        X = np.asarray(vecs, dtype=np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        text_embeddings = [(t, X[i].tolist()) for i, t in enumerate(texts)]
        try:
            return FAISS.from_embeddings(text_embeddings, embedding=emb, metadatas=metas)
        except TypeError:
            try:
                return FAISS.from_embeddings(embeddings=X.tolist(), metadatas=metas, texts=texts, embedding=emb)
            except Exception:
                vs = FAISS.from_texts(texts=[], embedding=emb)
                if hasattr(vs, "add_embeddings"):
                    vs.add_embeddings(embeddings=X, metadatas=metas, texts=texts)
                else:
                    vs.add_texts(texts=texts, metadatas=metas)
                return vs

    def build_textdb_with_answers(
        self,
        questions: List[str],
        *,
        add_prompt_snapshot: bool = False,
        bootstrap_db: Optional[FAISS] = None
    ) -> Tuple[FAISS, np.ndarray]:

        text_db = bootstrap_db
        last_q_vec = None

        for qid, q in enumerate(questions, start=1):
     
            answer = self.answer_with_llm_text(q, text_db=text_db)


            q_vec = self.sentence_emb.embed_query(q)
            last_q_vec = q_vec  

            prompt_snapshot = None
            if add_prompt_snapshot:
                hits = None
                if text_db is not None:
                    _, hits = self.similarity_search_text_docs(
                        q, text_db, k=self.config.get("faiss_search_k", 3),
                        emb=self.sentence_emb,
                        use_cache=self.config.get("use_cached_text_embeddings", True)
                    )
                prompt_snapshot = self.make_text_qa_prompt(q, hits)

            metadata = {
                "graph_id": f"Q{qid}",
                "question": q,
                "llm_model": self.config["llm_model_id"],
                "llm_answer": answer,
                "created_at": int(time.time()),
                "q_embeddings": q_vec,  
            }
            if add_prompt_snapshot:
                metadata["prompt_snapshot"] = prompt_snapshot

            doc = Document(page_content=q, metadata=metadata)
            text_db = self.upsert_text_docs_into_faiss([doc], emb=self.sentence_emb, text_db=text_db)

            self.text_db = text_db
        if text_db is None:
            text_db = FAISS.from_texts(texts=[], embedding=self.sentence_emb)

        return text_db, (np.asarray(last_q_vec) if last_q_vec is not None else None)

    # ---------- Graph RAG ----------
    def _faiss_search_by_vec_graph(self, vs: FAISS, qv: np.ndarray, k: int):
        if hasattr(vs, "similarity_search_by_vector_with_score"):
            return vs.similarity_search_by_vector_with_score(qv, k=k)
        if hasattr(vs, "similarity_search_by_vector"):
            docs = vs.similarity_search_by_vector(qv, k=k)
            return [(d, None) for d in docs]
        index = getattr(vs, "index", None)
        id_map = getattr(vs, "index_to_docstore_id", None)
        store  = getattr(vs, "docstore", None)
        if index is None or id_map is None or store is None:
            raise AttributeError("FAISS vectorstore has no by-vector APIs and no accessible index/docstore.")
        q = np.asarray(qv, dtype=np.float32).reshape(1, -1)
        D, I = index.search(q, k)
        out = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            doc_id = id_map[idx]
            doc = store.search(doc_id)
            out.append((doc, float(dist)))
        return out

    def similarity_search_graph_docs(
        self,
        user_question: str,
        parser,
        vectordb: FAISS,
        k: int = 5,
        emb_model: Optional[Embeddings] = None,
        use_cache: Optional[bool] = None,
    ):
        use_cache = self.config.get("use_cached_graph_embeddings", True) if use_cache is None else use_cache
        emb_model = emb_model or self.word_emb

        if use_cache and user_question in self._GRAPH_QVEC_CACHE:
            qv = self._GRAPH_QVEC_CACHE[user_question]
            return user_question, self._faiss_search_by_vec_graph(vectordb, qv, k)

        G, rels = self.parse_question_to_graph_generic(parser, user_question)

        er_e = list({str(n) for n in G.nodes})
        er_r = []
        if G.is_multigraph():
            for _, _, _, data in G.edges(keys=True, data=True):
                rel = data.get("relation")
                if rel: er_r.append(str(rel))
        else:
            for _, _, data in G.edges(data=True):
                rel = data.get("relation")
                if rel: er_r.append(str(rel))

        qv = self._graph_doc_vec_from_cached_or_embed(er_e, er_r, None, None, use_cache=False)
        qv = self._normalize(qv)

        if use_cache:
            self._GRAPH_QVEC_CACHE[user_question] = qv

        return user_question, self._faiss_search_by_vec_graph(vectordb, qv, k)

    def upsert_graph_docs_into_faiss(
        self,
        new_docs: List[Document],
        *,
        emb_model: Optional[Embeddings] = None,
        faiss_db: Optional[FAISS] = None,
    ) -> FAISS:
        emb_model = emb_model or self.word_emb

        texts, metas, vecs = [], [], []
        for d in new_docs:
            texts.append(d.page_content)
            meta = d.metadata or {}
            metas.append(meta)

            er = ast.literal_eval(d.page_content) if isinstance(d.page_content, str) else d.page_content
            er_e = er.get("e", []) if isinstance(er, dict) else []
            er_r = er.get("r", []) if isinstance(er, dict) else []

            cbm = (meta.get("codebook_main") or {})
            e_embeds = cbm.get("e_embeddings")
            r_embeds = cbm.get("r_embeddings")

            v = self._graph_doc_vec_from_cached_or_embed(er_e, er_r, e_embeds, r_embeds, use_cache=True)
            vecs.append(v.tolist())
            meta["graph_vec"] = v.tolist()

        X = np.asarray(vecs, dtype=np.float32)

        if faiss_db is None:
            text_embeddings = [(texts[i], X[i].tolist()) for i in range(len(texts))]
            try:
                faiss_db = FAISS.from_embeddings(text_embeddings, embedding=emb_model, metadatas=metas)
            except TypeError:
                try:
                    faiss_db = FAISS.from_embeddings(
                        embeddings=X.tolist(), metadatas=metas, texts=texts, embedding=emb_model
                    )
                except Exception:
                    faiss_db = FAISS.from_texts(texts=[], embedding=emb_model)
                    if hasattr(faiss_db, "add_embeddings"):
                        add_fn = faiss_db.add_embeddings
                        sig = inspect.signature(add_fn)
                        if "text_embeddings" in sig.parameters:
                            faiss_db.add_embeddings(text_embeddings=text_embeddings, metadatas=metas)
                        else:
                            faiss_db.add_embeddings(embeddings=X.tolist(), metadatas=metas, texts=texts)
                    else:
                        faiss_db.add_texts(texts=texts, metadatas=metas)
            return faiss_db

        if hasattr(faiss_db, "add_embeddings"):
            add_fn = faiss_db.add_embeddings
            sig = inspect.signature(add_fn)
            if "text_embeddings" in sig.parameters:
                text_embeddings = [(texts[i], X[i].tolist()) for i in range(len(texts))]
                faiss_db.add_embeddings(text_embeddings=text_embeddings, metadatas=metas)
            elif "embeddings" in sig.parameters and "texts" in sig.parameters:
                faiss_db.add_embeddings(embeddings=X.tolist(), metadatas=metas, texts=texts)
            else:
                text_embeddings = [(texts[i], X[i].tolist()) for i in range(len(texts))]
                try:
                    faiss_db.add_embeddings(text_embeddings=text_embeddings, metadatas=metas)
                except TypeError:
                    faiss_db.add_embeddings(embeddings=X.tolist(), metadatas=metas, texts=texts)
        elif hasattr(faiss_db, "add_texts"):
            faiss_db.add_texts(texts=texts, metadatas=metas)
        elif hasattr(faiss_db, "add_documents"):
            faiss_db.add_documents(new_docs)
        else:
            raise AttributeError("FAISS vectorstore has no add_* method to append data.")

        return faiss_db

    def make_graph_answer(
        self,
        question: str,
        parser,
        *,
        faiss_db: Optional[FAISS] = None,
        max_retries: int = 5,
    ) -> Tuple[str, Dict]:
        total_t0 = time.time()

        # —— 检索计时 —— 
        retrieval_latency = 0.0
        retrieved_docs = None
        retrieved_count = 0
        if faiss_db:
            t0 = time.time()
            _, hits = self.similarity_search_graph_docs(
                question, parser, faiss_db,
                k=self.config.get("faiss_search_k", 3),
                emb_model=self.word_emb,
                use_cache=self.config.get("use_cached_graph_embeddings", True),
            )
            retrieval_latency = time.time() - t0
            retrieved_docs = hits
            retrieved_count = len(hits) if hits else 0

        G, rels = self.parse_question_to_graph_generic(parser, question)
        prompt, compact_json = self.make_graph_qa_prompt(question, G, rels, retrieved_docs)

        YES_RE = re.compile(r"^\s*(yes|y|true|correct|affirmative)\s*\.?\s*$", re.I)
        NO_RE  = re.compile(r"^\s*(no|n|false|incorrect|negative)\s*\.?\s*$", re.I)

        gen_latency_total = 0.0
        peak_vram = 0.0
        final_answer = ""
        mode = self._mode()

        attempt = 0
        while attempt < max_retries:
            attempt += 1

            if torch.cuda.is_available():
                try: torch.cuda.reset_peak_memory_stats()
                except Exception: pass

            g0 = time.time()
            raw = self._gen(prompt)
            gen_latency_total += (time.time() - g0)

            if torch.cuda.is_available():
                try: peak_vram = max(peak_vram, torch.cuda.max_memory_allocated() / (1024**2))
                except Exception: pass

            if self.verbose: print(f"----- RAW (try {attempt}):", raw)

            answer = self._extract_answer_text(prompt, raw)
            answer, thinking = self.strip_think(raw)
            if self.verbose: print("----- ANS:", answer)

            if not answer.strip():
                continue

            if mode in {"yes_no","binary"}:
                if YES_RE.match(answer) and not NO_RE.match(answer):
                    final_answer = self._yn("YES","NO")[0]; break
                if NO_RE.match(answer) and not YES_RE.match(answer):
                    final_answer = self._yn("YES","NO")[1]; break

                strict_suffix = "\n\n[FORMAT]: Answer with exactly ONE token: " + \
                                ("YES or NO." if self.config.get("answer_uppercase", True) else "yes or no.")
                g0 = time.time()
                raw2 = self._gen(prompt + strict_suffix)
                gen_latency_total += (time.time() - g0)
                ans2 = self._extract_answer_text(prompt + strict_suffix, raw2)
                if YES_RE.match(ans2) and not NO_RE.match(ans2):
                    final_answer = self._yn("YES","NO")[0]; break
                if NO_RE.match(ans2) and not YES_RE.match(ans2):
                    final_answer = self._yn("YES","NO")[1]; break
                final_answer = ans2; break
            else:
                if YES_RE.match(answer) or NO_RE.match(answer) or len(answer.split()) <= 2:
                    format_suffix = "\n\n[FORMAT]: Provide a 2–3 sentence explanation; do not answer with a single word."
                    g0 = time.time()
                    raw2 = self._gen(prompt + format_suffix)
                    gen_latency_total += (time.time() - g0)
                    ans2 = self._extract_answer_text(prompt + format_suffix, raw2)
                    if len(ans2.strip()) > len(answer.strip()):
                        final_answer = ans2.strip(); break
                final_answer = answer.strip(); break

        total_latency = time.time() - total_t0

        # —— 记录指标（不影响返回结构）——
        m = {
            "input_tokens":  self._num_tokens(prompt),
            "output_tokens": self._num_tokens(final_answer),
            "total_tokens":  self._num_tokens(prompt) + self._num_tokens(final_answer),
            "latency_sec": total_latency,
            "retrieval_latency_sec": retrieval_latency,
            "gen_latency_sec": gen_latency_total,
            "retrieved_count": retrieved_count,
            "peak_vram_MiB": float(peak_vram),
            "prompt_chars": len(prompt),
        }
        self._record_metric("graph",m)

        return final_answer, compact_json, thinking


    def build_graphdb_with_answer(
        self,
        questions: List[str],
        parser,
        *,
        add_prompt_snapshot: bool = False,
        faiss_db: Optional[FAISS] = None
    ) -> FAISS:
        """
        EXACTLY preserves your original behavior:
        - For each question:
            1) build codebooks
            2) get LLM answer
            3) merge into codebook_main
            4) create ER doc
            5) UPSERT that doc into FAISS immediately (so next question can retrieve it)
        """
        codebook_main = {}
        compact_json = {}
        thinking = ""
        for qid, q in enumerate(questions, start=1):
            codebook_question = get_code_book(q, type="questions")

            answer, compact_json, thinking = self.make_graph_answer(q, parser, faiss_db=faiss_db)
            codebook_answer = get_code_book(answer, type='answers')

            if compact_json:
                codebook_main["e"] = compact_json["e"]
                codebook_main["r"] = compact_json["r"]
                codebook_main["edge_matrix"] = compact_json["edge_matrix"]
                codebook_main["questions([[e,r,e], ...])"] = compact_json["questions([[e,r,e], ...])"]
                codebook_main["given knowledge([[e,r,e], ...]"] = compact_json["given knowledge([[e,r,e], ...])"]

            if thinking:
               #codebook_thinking = get_code_book(thinking, type='thinking')
               #codebook_main = merging_codebook(codebook_main, codebook_thinking,  type='thinking', ,use_thinkings = True)
               pass

            codebook_main = merging_codebook(codebook_main, codebook_question, type='questions')
            codebook_main = merging_codebook(codebook_main, codebook_answer,  type='answers')            
            codebook_main = combine_ents(codebook_main, min_exp_num=2, max_exp_num=8, random_state=0, use_thinking=False)

            er = {
                "e": codebook_main['e'],
                "r": codebook_main['r']
            }
            er_str = str(er)
            codebook_main.pop("rule", None)

            metadata = {
                "graph_id": f"Q{qid}",
                "llm_model": self.config["llm_model_id"],
                "llm_answer": answer,
                "created_at": int(time.time()),
                "codebook_main": codebook_main,
            }

            doc = Document(page_content=er_str, metadata=metadata)
            faiss_db = self.upsert_graph_docs_into_faiss([doc], emb_model=self.word_emb, faiss_db=faiss_db)
            self.graph_db = faiss_db
        return faiss_db
    
    def _num_tokens(self, s: str) -> int:
        if not s: return 0
        if self.gen_pipe is None: self.set_llm()
        tok = self.tokenizer
        try:
            return len(tok(s, add_special_tokens=False)["input_ids"])
        except Exception:
            return len(tok.encode(s))

    @staticmethod
    def _bytes_to_human(n: int) -> str:
        if n < 1024: return f"{n} B"
        units = ["KB","MB","GB","TB","PB","EB"]
        v = n / 1024.0
        for u in units:
            if v < 1024: return f"{v:.2f} {u}"
            v /= 1024.0
        return f"{v:.2f} ZB"

    def _dir_size_bytes(self, path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    def _record_metric(self, kind: str, m: dict):
        # kind: "graph" | "text"
        self._metrics.setdefault(kind, []).append(m)

    def _avg_metrics(self, kind: str) -> dict:
        rows = self._metrics.get(kind, [])
        if not rows: return {}
        keys = [
            "input_tokens","output_tokens","total_tokens",
            "latency_sec","retrieval_latency_sec","gen_latency_sec",
            "retrieved_count","peak_vram_MiB","prompt_chars",
        ]
        out = {}
        for k in keys:
            vals = [r.get(k, 0.0) for r in rows]
            out[k] = (sum(vals) / len(vals)) if vals else 0.0
        return out

    def report_cost(self, *, kind: str = "graph", avg: bool = True) -> dict:
        stats = self._avg_metrics(kind) if avg else (self._metrics.get(kind, [])[-1] if self._metrics.get(kind) else {})
        if not stats:
            print(f"\n== Cost of {kind} RAG ({'avg' if avg else 'last'}) ==\n  <no data>")
            return {}
        print(f"\n== Cost (avg) of {kind} RAG ==" if avg else "\n== Cost (last) ==")
        for key in ["input_tokens","output_tokens","total_tokens","latency_sec",
                    "retrieval_latency_sec","gen_latency_sec","retrieved_count",
                    "peak_vram_MiB","prompt_chars"]:
            val = stats.get(key, 0.0)
            s = f"{val:.2f}" if isinstance(val, float) and val != int(val) else f"{int(val)}"
            print(f"{key:>22} {s}")
        return stats

    def dir_size_bytes(self, path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                try: total += os.path.getsize(fp)
                except OSError: pass
        return total

    def save_index_and_report_size(self, *, db: str = "graph", out_dir: str = None):
        if db not in {"graph", "text"}:
            raise ValueError("db must be 'graph' or 'text'")
        if out_dir is None:
            out_dir = "faiss_graph_idx" if db == "graph" else "faiss_text_idx"

        vs = self.graph_db if db == "graph" else self.text_db
        if vs is None:
            print(f"[Index size] {db}_rag = 0 B  ({out_dir})")
            return 0

        try:
            vs.save_local(out_dir)
        except Exception:
            vs.save_local(folder_path=out_dir)

        size_b = self._dir_size_bytes(out_dir)
        human = self._bytes_to_human(size_b)

        label = f"{db}_rag"
        pad = " " if db == "text" else ""
        print(f"[Index size] {label} = {human}  ({out_dir}){pad}")
        return size_b
    
# =========================
# Usage (example)
# =========================
if __name__ == "__main__":
    rag = RAGWorkflow(
        config={
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model_id": "microsoft/Phi-4-mini-reasoning",
            "answer_mode": "short",
            "faiss_search_k": 3,
            "use_cached_text_embeddings": True,
            "use_cached_graph_embeddings": True,
        },
        verbose=True
    )
    questions = [
        "Is the Earth, which orbits the Sun, generally considered round?",
        "Given Earth’s rotation, does the Sun rise in the east?",
    ]

    rag.set_llm()

    text_db, last_q_vec = rag.build_textdb_with_answers(questions, add_prompt_snapshot=False)
    rag.save_index_and_report_size(db="text", out_dir="faiss_text_idx")
    rag.report_cost(kind="text", avg=True)   

    parser = RelationshipGraphParser()  
    graph_db = rag.build_graphdb_with_answer(questions, parser, faiss_db=None)
    rag.save_index_and_report_size(db="graph", out_dir="faiss_graph_idx")
    rag.report_cost(kind="graph", avg=True)   

