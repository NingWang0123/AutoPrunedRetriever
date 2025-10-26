# run_hipporag2.py
import sys
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
HIPPO_SRC  = SCRIPT_DIR / "HippoRAG" / "src"   # .../Examples/HippoRAG/src
if str(HIPPO_SRC) not in sys.path:
    sys.path.insert(0, str(HIPPO_SRC))
import os
import argparse
import json
import logging
from typing import Dict, List, Optional
from dotenv import load_dotenv
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

# ====== NEW: sbert 评价依赖 ======
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import time

import multiprocessing as mp, os
try:
    if mp.get_start_method(allow_none=True) != "fork":
        mp.set_start_method("fork")
except RuntimeError:
    pass

# ==== External HF Embedding shim (method 3, no HippoRAG source change) ====
import os, importlib
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from sentence_transformers import SentenceTransformer
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ["OPENAI_MAX_RETRIES"] = "0"
os.environ["OPENAI_TIMEOUT"] = "10"

def _is_hf_minilm(name: str) -> bool:
    n = (name or "").lower()
    # 检查常见的HuggingFace嵌入模型
    hf_patterns = [
        "sentence-transformers", "minilm", "all-minilm-l6-v2",
        "bge-", "baai/", "e5-", "gte-", "multilingual-e5"
    ]
    return any(pattern in n for pattern in hf_patterns)

class _HFSTEmbedding:
    """
    兼容 HippoRAG 的嵌入器包装：
    - __init__ 支持 (global_config=..., embedding_model_name=..., ...) 等签名
    - 提供 batch_encode/encode/embed/encode_texts/get_embeddings/__call__ 等常见接口
    - 暴露 embedding_dim / embedding_size，并提供 get_dimension()
    """
    def __init__(self, *args, **kwargs):
        # 1) 优先从显式 kwargs 取
        model_name = kwargs.get("embedding_model_name") or kwargs.get("model_name")

        # 2) 从 global_config 兜底
        gc = kwargs.get("global_config")
        if model_name is None and gc is not None:
            model_name = getattr(gc, "embedding_model_name", None)

        # 3) 从位置参数兜底（若传了一个字符串）
        if model_name is None:
            for a in args:
                if isinstance(a, str):
                    model_name = a
                    break

        # 4) 默认
        if not model_name:
            model_name = "sentence-transformers/all-MiniLM-L6-v2"

        self.model_name = model_name

        # 设备（mac 优先 mps，其次 cuda，否则 cpu）
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        # 允许短名
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        except Exception:
            self.model = SentenceTransformer(f"sentence-transformers/{self.model_name}", device=self.device)

        try:
            self.embedding_dim = int(self.model.get_sentence_embedding_dimension())
        except Exception:
            self.embedding_dim = 384
        self.embedding_size = self.embedding_dim  # 有些实现会用这个名

    # —— 内部统一实现 ——
    def _encode_impl(self, texts, batch_size=64, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(
            texts,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return np.asarray(vecs, dtype=np.float32)

    # —— 对外接口（常见别名）——
    def batch_encode(self, texts, batch_size=64, normalize=True):
        if isinstance(texts, str):
            texts = [texts]
        if batch_size is None or batch_size <= 0:
            batch_size = 64
        out = []
        for i in range(len(texts))[::batch_size]:
            chunk = texts[i:i+batch_size]
            out.append(self._encode_impl(chunk, batch_size=batch_size, normalize=normalize))
        return np.vstack(out) if out else np.zeros((0, self.embedding_dim), dtype=np.float32)

    def encode(self, texts, **kwargs):
        # 允许外部直接传大列表，这里也走 batch
        bs = kwargs.pop("batch_size", 64)
        normalize = kwargs.pop("normalize_embeddings", kwargs.pop("normalize", True))
        return self.batch_encode(texts, batch_size=bs, normalize=normalize)

    def embed(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def encode_texts(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def get_embeddings(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    def __call__(self, texts, **kwargs):
        return self.encode(texts, **kwargs)

    # 可能被用到的辅助
    def get_dimension(self):
        return self.embedding_dim

    # 一些实现会调用 .to(device)
    def to(self, device):
        self.device = device
        try:
            self.model = self.model.to(device)
        except Exception:
            pass
        return self


def _patch_embedding_resolver():
    # 同时尝试两个命名空间：HippoRAG.src.* 和裸的 hipporag.*
    for mod_name in ("HippoRAG.src.hipporag.embedding_model", "hipporag.embedding_model"):
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        if not hasattr(m, "_get_embedding_model_class"):
            continue
        _orig = m._get_embedding_model_class

        def _patched(name_str, _orig=_orig):
            if _is_hf_minilm(name_str):
                # 返回一个“类”，HippoRAG 内部会按 Class(name) 实例化
                class _Wrapper(_HFSTEmbedding):
                    pass
                return _Wrapper
            return _orig(name_str)

        m._get_embedding_model_class = _patched

_patch_embedding_resolver()

import importlib
m = importlib.import_module("hipporag.embedding_model")
print("[shim] resolver:", m._get_embedding_model_class("sentence-transformers/all-MiniLM-L6-v2"))

# ==== end of shim ====

# Load environment variables
load_dotenv()

# Import HippoRAG components after setting environment
from hipporag.HippoRAG import HippoRAG
from hipporag.utils.misc_utils import string_to_bool
from hipporag.utils.config_utils import BaseConfig

# >>> 新增：把 HippoRAG 模块里的函数引用重绑到我们补丁后的函数 <<<
import hipporag.HippoRAG as _hippo_mod
import hipporag.embedding_model as _embed_mod
_hippo_mod._get_embedding_model_class = _embed_mod._get_embedding_model_class

# （可选）固定 GPU
#os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")


# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("hipporag_processing.log")
    ]
)

# =========================
# SBERT 相似度 & 文本规整
# =========================
_NORMALIZE_SPACE_RE = re.compile(r"\s+")

def _normalize_space(s: str) -> str:
    return _NORMALIZE_SPACE_RE.sub(" ", (s or "").strip())

_SBERT = None
def _get_sbert():
    global _SBERT
    if _SBERT is None:
        _SBERT = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _SBERT

def sbert_cosine(a: str, b: str) -> float:
    model = _get_sbert()
    v1, v2 = model.encode([a, b])
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)
    return float((v1 * v2).sum())


# =========================
# HuggingFace 本地推理封装
# =========================
_HF_CACHE = {"tok": None, "mdl": None, "name": None}

def _ensure_hf_loaded(model_name: str, torch_dtype: Optional[torch.dtype] = None):
    if _HF_CACHE["name"] == model_name and _HF_CACHE["mdl"] is not None:
        return _HF_CACHE["tok"], _HF_CACHE["mdl"]

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if torch_dtype is None:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif torch.cuda.is_available():
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    mdl.eval()
    _HF_CACHE.update({"tok": tok, "mdl": mdl, "name": model_name})
    return tok, mdl

def _build_chat_input(tokenizer, system_prompt: Optional[str], history_messages: list, user_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for m in history_messages or []:
        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # fallback：纯文本
    parts = []
    if system_prompt:
        parts.append(f"[SYSTEM]\n{system_prompt}\n")
    for m in history_messages or []:
        parts.append(f"[{m.get('role','user').upper()}]\n{m.get('content','')}\n")
    parts.append(f"[USER]\n{user_prompt}\n[ASSISTANT]\n")
    return "\n".join(parts)

def hf_generate_answer(
    question: str,
    docs_context: str,
    *,
    system_prompt: str,
    hf_model_name: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_new_tokens: int = 256,
) -> str:
    tok, mdl = _ensure_hf_loaded(hf_model_name)

    prompt_user = f"Question:\n{question}\n\nKnowledge Base:\n{docs_context}\n\nAnswer:"
    input_text = _build_chat_input(tok, system_prompt, [], prompt_user)

    inputs = tok(input_text, return_tensors="pt")
    inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

    do_sample = temperature > 0
    gen = mdl.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
    )
    # 尽量只取新增
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        prompt_len = inputs["input_ids"].shape[-1]
        resp_ids = gen[0][prompt_len:]
        return tok.decode(resp_ids, skip_special_tokens=True).strip()
    text = tok.decode(gen[0], skip_special_tokens=True)
    return text.split("[ASSISTANT]")[-1].strip() if "[ASSISTANT]" in text else text.strip()


# =========================
# 其他工具
# =========================
def group_questions_by_source(question_list: List[dict]) -> Dict[str, List[dict]]:
    grouped_questions = {}
    for q in question_list:
        src = q.get("source")
        grouped_questions.setdefault(src, []).append(q)
    return grouped_questions

def split_text(
    text: str,
    tokenizer: AutoTokenizer,
    chunk_token_size: int = 1200,
    chunk_overlap_token_size: int = 100
) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start += chunk_token_size - chunk_overlap_token_size
    return chunks


# =========================
# 统一的 SYSTEM_PROMPT
# =========================
SYSTEM_PROMPT = """\
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""


def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    hf_model_name: Optional[str],
    embed_model_path: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: Dict[str, List[dict]],
    sample: Optional[int],
    retrieve_topk: int,
    q_start: Optional[int],
    q_end: Optional[int],
    retrieval_only: bool = False
):
    """索引语料 + 回答问题；支持 API / HF 两种生成路径"""
    mode_str = "🔍 RETRIEVAL ONLY" if retrieval_only else "💬 RETRIEVAL + GENERATION"
    logging.info(f"📚 Processing corpus: {corpus_name} ({mode_str})")

    # 输出目录
    output_dir = f"./results/hipporag2/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    mode_suffix = "_retrieval_only" if retrieval_only else ""
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}{mode_suffix}.json")

    # 切分
    try:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_path)
        logging.info(f"✅ Loaded tokenizer: {embed_model_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load tokenizer: {e}")
        return

    chunks = split_text(context, tokenizer)
    logging.info(f"✂️ Split corpus into {len(chunks)} chunks")
    docs = [f"{i}:{c}" for i, c in enumerate(chunks)]

    # 取问题
    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"⚠️ No questions for corpus: {corpus_name}")
        logging.warning(f"Available question groups: {list(questions.keys())}")
        return
    
    logging.info(f"🔍 Debug: Original questions count: {len(corpus_questions)}")

    if sample and sample < len(corpus_questions):
        logging.info(f"🔍 Debug: Applying sample limit: {sample}")
        corpus_questions = corpus_questions[:sample]
        logging.info(f"🔍 Debug: After sample: {len(corpus_questions)} questions")

    if q_start is not None or q_end is not None:
        start_idx = (q_start - 1) if q_start else 0
        end_idx = q_end if q_end else len(corpus_questions)
        
        # Validate and adjust indices
        if start_idx >= len(corpus_questions):
            logging.warning(f"⚠️  q_start ({q_start}) exceeds available questions ({len(corpus_questions)}). Using questions 1-2 instead.")
            start_idx = 0
            end_idx = min(2, len(corpus_questions))
        elif end_idx > len(corpus_questions):
            logging.warning(f"⚠️  q_end ({q_end}) exceeds available questions ({len(corpus_questions)}). Adjusting to available range.")
            end_idx = len(corpus_questions)
        
        logging.info(f"🔍 Debug: Applying slice [{start_idx}:{end_idx}] to {len(corpus_questions)} questions")
        corpus_questions = corpus_questions[start_idx:end_idx]
        logging.info(f"🔍 Debug: After slice: {len(corpus_questions)} questions")

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    force_contriever = _is_hf_minilm(embed_model_path)
    if force_contriever:
        import hipporag.embedding_model as _em

        # 关键：把真实的目标模型名（embed_model_path）硬塞给构造函数
        class _ContrieverAsHF(_HFSTEmbedding):
            def __init__(self, *args, **kwargs):
                kwargs = dict(kwargs)
                # 无论 HippoRAG 传来的 embedding_model_name 是什么（一般是 "contriever"），都改成真正的 ST 模型
                kwargs["embedding_model_name"] = embed_model_path
                super().__init__(*args, **kwargs)

        _em.ContrieverModel = _ContrieverAsHF
        em_name_for_config = "contriever"  
    else:
        em_name_for_config = embed_model_path
    
    
    config = BaseConfig(
        save_dir=os.path.join(base_dir, corpus_name),
        llm_base_url=llm_base_url,
        llm_name=model_name,                     
        embedding_model_name=em_name_for_config,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        rerank_dspy_file_path="Examples/HippoRAG/src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=retrieve_topk,
        linking_top_k=retrieve_topk,
        max_qa_steps=3,
        qa_top_k=retrieve_topk,
        graph_type="facts_and_sim_passage_node_unidirectional",
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(docs),
        openie_mode="online"
    )
    hipporag = HippoRAG(global_config=config)
    hipporag.index(docs)
    logging.info(f"✅ Indexed corpus: {corpus_name}")

    # 统一准备查询
    all_queries = [q["question"] for q in corpus_questions]
    gold_answers = [[q.get("answer", "")] for q in corpus_questions]

    # 让 HippoRAG 给到候选 docs + （若 API 模式）内置答案
    queries_solutions, _, _, _, _ = hipporag.rag_qa(
        queries=all_queries, gold_docs=None, gold_answers=gold_answers
    )
    solutions = [q.to_dict() for q in queries_solutions]

    results = []

    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        # 时间和性能测量
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        from time import perf_counter
        t_total_start = perf_counter()
        
        sol = next((s for s in solutions if s.get("question") == q["question"]), None)
        if sol is None:
            continue

        # 提取检索上下文
        docs_ctx = sol.get("docs", "")
        if isinstance(docs_ctx, list):
            docs_ctx = "\n".join([str(x) for x in docs_ctx])

        # 生成答案：根据 mode 和 retrieval_only 选择
        if retrieval_only:
            # 检索模式：不生成答案，只保留检索结果
            generated_answer = "[RETRIEVAL_ONLY_MODE]"
        elif mode.lower() == "hf":
            # 用本地 HF 模型重写答案（忽略 HippoRAG 内置的 API 生成）
            sys_prompt = SYSTEM_PROMPT.format(history="", context_data="{context}")
            # 将 context 占位替换成真正的 docs
            sys_prompt = sys_prompt.replace("{context}", docs_ctx or "")
            generated_answer = hf_generate_answer(
                question=q["question"],
                docs_context=docs_ctx or "",
                system_prompt=sys_prompt,
                hf_model_name=hf_model_name or "Qwen2.5-7B-Instruct",
                temperature=0.0,
                top_p=0.9,
                max_new_tokens=256,
            )
        else:
            # API：沿用 HippoRAG 输出
            generated_answer = sol.get("answer", "")

        t_total_end = perf_counter()
        total_latency = t_total_end - t_total_start

        # 计算生成信息和性能指标
        if retrieval_only:
            gen_latency_sec = 0.0
            retrieval_latency_sec = total_latency
            approx_prompt_chars = len(q["question"])
            output_tokens = 0.0
        else:
            # 估算延迟分配（HippoRAG 内部不区分检索和生成时间）
            gen_latency_sec = total_latency * 0.7  # 估算生成占70%
            retrieval_latency_sec = total_latency * 0.3  # 估算检索占30%
            approx_prompt_chars = len(SYSTEM_PROMPT) + len(docs_ctx or "") + len(q["question"])
            output_tokens = len(generated_answer) / 4.0

        input_tokens = approx_prompt_chars / 4.0

        # 设备和数据类型信息
        device = "hf" if mode.lower() == "hf" else "api"
        try:
            if mode.lower() == "hf" and _HF_CACHE["mdl"] is not None:
                dtype = str(_HF_CACHE["mdl"].dtype)
            else:
                dtype = "unknown"
        except Exception:
            dtype = "unknown"

        gen_info = {
            "input_tokens": float(input_tokens),
            "output_tokens": float(output_tokens),
            "total_tokens": float(input_tokens + output_tokens),
            "latency_sec": float(total_latency),
            "gen_latency_sec": float(gen_latency_sec),
            "retrieval_latency_sec": float(retrieval_latency_sec),
            "prompt_chars": float(approx_prompt_chars),
            "throughput_tok_per_s": float((output_tokens / gen_latency_sec) if gen_latency_sec > 0 else 0.0),
            "prompt_tok_per_s": float((input_tokens / retrieval_latency_sec) if retrieval_latency_sec > 0 else 0.0),
            "device": device,
            "dtype": str(dtype),
            "model_name": model_name,
            "timestamp_start": t_total_start,
            "timestamp_end": t_total_end,
        }

        # 统一评价（SBERT）
        gold_answer = q.get("answer", "") or ""
        gt_ctx_raw = q.get("evidence", "")
        if isinstance(gt_ctx_raw, list):
            gt_ctx = " ".join([str(x) for x in gt_ctx_raw])
        else:
            gt_ctx = str(gt_ctx_raw or "")

        pa = _normalize_space(str(generated_answer))
        ga = _normalize_space(str(gold_answer))
        ctx_pred = _normalize_space(str(docs_ctx or ""))
        ctx_gold = _normalize_space(gt_ctx)

        # 评估答案正确性（检索模式下跳过）
        if retrieval_only:
            eval_correctness = -1.0  # 表示未评估
        else:
            eval_correctness = sbert_cosine(pa, ga)
            
        # 评估上下文相似度（总是执行）
        eval_context = sbert_cosine(ctx_pred, ctx_gold)

        record = {
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": docs_ctx or "",
            "evidence": q.get("evidence", ""),
            "question_type": q.get("question_type", ""),
            "generated_answer": generated_answer,
            "ground_truth": gold_answer,
            "retrieval_only": retrieval_only,

            **gen_info,

            "correctness": float(eval_correctness),
            "context_similarity": float(eval_context),
        }

        results.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")


def main():
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.json",
            "questions": "./Datasets/Questions/medical_questions.json"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.json",
            "questions": "./Datasets/Questions/novel_questions.json"
        },
        "2wikimultihop": {
            "corpus": "./Datasets/Corpus/2wikimultihop.json",
            "questions": "./Datasets/Questions/2wikimultihop_questions.json"
        },
        "hotpotqa": {
            "corpus": "./Datasets/Corpus/hotpotqa.json",
            "questions": "./Datasets/Questions/hotpotqa_questions.json"
        },
        "history": {
            "corpus": "./Datasets/Corpus/history_corpus.json",
            "questions": "./Datasets/Questions/history_QnA.json"
        }
    }

    parser = argparse.ArgumentParser(description="HippoRAG: Process Corpora and Answer Questions")

    # 数据与工作区
    parser.add_argument("--subset", required=True, choices=["medical", "novel", "2wikimultihop", "hotpotqa", "history"],
                        help="Subset to process (medical, novel, 2wikimultihop, hotpotqa, or history)")
    parser.add_argument("--q_start", type=int, default=None, help="Start index of questions (1-based, inclusive)")
    parser.add_argument("--q_end", type=int, default=None, help="End index of questions (1-based, inclusive)")
    
    parser.add_argument("--base_dir", default="./hipporag2_workspace",
                        help="Base working directory for HippoRAG")

    # 模式与模型
    parser.add_argument("--mode", choices=["API", "hf"], default="API",
                        help="Generation mode: API (default) or hf (local transformers)")
    parser.add_argument("--model_name", default="gpt-4o-mini",
                        help="LLM model identifier for API mode")
    parser.add_argument("--hf_model_name", default="Qwen2.5-7B-Instruct",
                        help="HF model id for local generation mode (only used if --mode hf)")
    parser.add_argument("--embed_model_path", default="BAAI/bge-large-en-v1.5",
                            help="Path to embedding model directory or HF model name (for tokenizer splitting & HippoRAG config)")

    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of questions to sample per corpus")
    parser.add_argument("--retrieval_only", action="store_true",
                        help="Only perform retrieval without LLM generation")

    # API
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1",
                        help="Base URL for LLM API (API mode)")
    parser.add_argument("--llm_api_key", default="",
                        help="API key for LLM service (can also use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    logging.info(f"🚀 Starting HippoRAG processing for subset: {args.subset} (mode={args.mode})")

    if args.subset not in SUBSET_PATHS:
        logging.error(f"❌ Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("OPENAI_API_KEY", "")
    if args.mode == "API" and not api_key:
        logging.warning("⚠️ No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"📖 Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"❓ Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"❌ Failed to load questions: {e}")
        return

    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        process_corpus(
            corpus_name=corpus_name,
            context=context,
            base_dir=args.base_dir,
            mode=args.mode,
            model_name=args.model_name,
            hf_model_name=args.hf_model_name,
            embed_model_path=args.embed_model_path,
            llm_base_url=args.llm_base_url,
            llm_api_key=api_key,
            questions=grouped_questions,
            sample=args.sample,
            retrieve_topk=args.retrieve_topk,
            q_start=args.q_start,
            q_end=args.q_end,
            retrieval_only=args.retrieval_only
        )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()  
    main()
