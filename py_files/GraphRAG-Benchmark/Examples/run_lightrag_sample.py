# lightrag_example.py
import os

os.environ["FUNC_TIMEOUT"] = "600"          # base LLM timeout
os.environ["WORKER_TIMEOUT"] = "630"        # optional: only if your build reads this explicitly
os.environ["HEALTHCHECK_TIMEOUT"] = "660"   # optional: only if your build reads this explicitly
os.environ["EMBEDDING_TIMEOUT"] = "600"     # embedding workers base timeout
os.environ["LLM_TIMEOUT"] = "600"

# Patch the decorator factory BEFORE importing any other lightrag submodules
import lightrag.utils as _lr_utils
import asyncio
import logging
import nest_asyncio
import argparse
import json
from typing import Dict, List, Optional
import time
import random

import torch
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
)

from lightrag.kg.shared_storage import get_namespace_data, get_pipeline_status_lock

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
# from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from tqdm import tqdm
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from typing import List, Tuple
from lightrag.base import DocStatus

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

from sentence_transformers import SentenceTransformer
import numpy as np
import re
from collections import Counter
from time import perf_counter
from lightrag.kg import nano_vector_db_impl as nvdb


from functools import wraps
import lightrag.operate as _op

def get_dir_size(path: str) -> int:
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def strip_think(s: str) -> Tuple[str, List[str]]:
        """提取思考并返回干净答案；支持 <think>...</think> 与多段 <|assistant|>。"""
        if not s:
            return "", []

        thinks: List[str] = []
        s_lower = s.lower()
        spans = []
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
        no_think_text = "".join(parts)

        blocks = [blk.strip()
                for blk in re.split(r"(?i)<\|assistant\|>", no_think_text)
                if blk and blk.strip()]

        if blocks:
            if len(blocks) >= 2:
                thinks.extend(blocks[:-1])
            clean = blocks[-1].strip()
        else:
            clean = no_think_text.strip()

        clean = re.sub(r"(?:^|\n)\s*(Okay,|Let’s|Let's|Step by step|Thought:).*",
                    "", clean, flags=re.I)
        clean = re.sub(r"(?i)<\|assistant\|>", "", clean).strip()

        return clean, thinks

_SbertModel = None
def get_sbert_model():
    global _SbertModel
    if _SbertModel is None:
        _SbertModel = SentenceTransformer("BAAI/bge-base-en")
    return _SbertModel

def reward_sbert_cached(pred: str, gold: str) -> float:
    model = get_sbert_model()
    emb_pred, emb_gold = model.encode([pred, gold])
    emb_pred /= (np.linalg.norm(emb_pred) + 1e-9)
    emb_gold /= (np.linalg.norm(emb_gold) + 1e-9)
    return float((emb_pred * emb_gold).sum())

def group_questions_by_source(question_list):
    grouped_questions = {}
    for question in question_list:
        source = question.get("source")
        if source not in grouped_questions:
            grouped_questions[source] = []
        grouped_questions[source].append(question)
    return grouped_questions


SYSTEM_PROMPT = """
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

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """LLM interface function using OpenAI-compatible API"""
    model_name = kwargs.get("model_name", "qwen2.5-14b-instruct")
    base_url = kwargs.get("base_url", "")
    api_key = kwargs.get("api_key", "")
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )


_HF_CACHE = {
    "tokenizer": None,
    "model": None,
    "model_name": None,
}


def _ensure_hf_loaded(model_name: str, torch_dtype: Optional[torch.dtype] = None):
    if _HF_CACHE["model_name"] == model_name and _HF_CACHE["model"] is not None:
        return _HF_CACHE["tokenizer"], _HF_CACHE["model"]

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
    _HF_CACHE.update({"tokenizer": tok, "model": mdl, "model_name": model_name})
    return tok, mdl

def _build_chat_input(tokenizer, system_prompt: Optional[str], history_messages: list, user_prompt: str):
    """
    history_messages: list of dicts like [{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
    优先使用 chat_template；否则退化为简单拼接。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        for m in history_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text
    parts = []
    if system_prompt:
        parts.append(f"[SYSTEM]\n{system_prompt}\n")
    if history_messages:
        for m in history_messages:
            parts.append(f"[{m.get('role','user').upper()}]\n{m.get('content','')}\n")
    parts.append(f"[USER]\n{user_prompt}\n[ASSISTANT]\n")
    return "\n".join(parts)

async def hf_model_complete(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """
    与 openai_complete_if_cache / ollama_model_complete 形参保持相似：
    - 使用 transformers 本地生成
    - 支持 temperature / top_p / max_new_tokens 等
    """
    model_name = kwargs.get("model_name")
    temperature = float(kwargs.get("temperature", 0.0))
    top_p = float(kwargs.get("top_p", 0.9))
    max_new_tokens = int(kwargs.get("max_new_tokens", kwargs.get("num_predict", 256)))
    do_sample = temperature > 0

    tok, mdl = _ensure_hf_loaded(model_name)

    def _sync_generate():
        input_text = _build_chat_input(tok, system_prompt, history_messages, prompt)
        inputs = tok(input_text, return_tensors="pt")
        inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

        gen_out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
        out_text = tok.decode(gen_out[0], skip_special_tokens=True)


        if hasattr(tok, "apply_chat_template") and tok.chat_template:
            prompt_len = inputs["input_ids"].shape[-1]
            gen_ids = gen_out[0][prompt_len:]
            resp = tok.decode(gen_ids, skip_special_tokens=True).strip()
            return resp
        else:
            if "[ASSISTANT]" in out_text:
                resp = out_text.split("[ASSISTANT]")[-1].strip()
                return resp
            return out_text.strip()

    return await asyncio.to_thread(_sync_generate)

async def initialize_rag(
    base_dir: str,
    source: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str
) -> LightRAG:
    working_dir = os.path.join(base_dir, source)
    os.makedirs(working_dir, exist_ok=True)

    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name)
        embedding_dim = 1024 if "large" in embed_model_name.lower() else 768
        embedding_func = _lr_utils.EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model),
        )
        llm_kwargs = {
            "model_name": model_name,
            "base_url": llm_base_url,
            "api_key": llm_api_key
        }
        llm_model_func_input = llm_model_func

    elif mode == "ollama":
        ollama_host = llm_base_url or "http://localhost:11434"
        name = embed_model_name.lower()
        if ("minilm" in name) or ("all-minilm-l6-v2" in name):
            embedding_dim = 384
        elif "nomic-embed-text" in name:
            embedding_dim = 768
        elif ("bge-m3" in name) or ("mxbai-embed-large" in name):
            embedding_dim = 1024
        elif ("bge-large" in name) or ("bge-large-en-v1.5" in name):
            embedding_dim = 1024
        else:
            embedding_dim = 768

        embedding_func = _lr_utils.EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=ollama_host
            ),
        )
        llm_kwargs = {
            "host": ollama_host,
            "options": {
                "num_ctx": 4096,
                "num_predict": 256,
                "temperature": 0,
                "top_p": 0.9,
                "num_gpu": 999,
            },
        }
        llm_model_func_input = ollama_model_complete

    elif mode == "hf":
        tok_embed = AutoTokenizer.from_pretrained(embed_model_name)
        mdl_embed = AutoModel.from_pretrained(embed_model_name)
        name = embed_model_name.lower()
        if ("minilm" in name) or ("all-minilm-l6-v2" in name):
            embedding_dim = 384
        elif "bge-m3" in name:
            embedding_dim = 1024
        elif "bge-large" in name or "bge-large-en-v1.5" in name:
            embedding_dim = 1024
        elif "nomic-embed-text" in name:
            embedding_dim = 768
        elif "bge-base" in name or "bge-base-en" in name:
            embedding_dim = 768
        else:
            embedding_dim = 768

        embedding_func = _lr_utils.EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tok_embed, mdl_embed),
        )

        llm_kwargs = {
            "model_name": model_name,
            "temperature": 0.3,
            "top_p": 0.95,
            "max_new_tokens": 256,
        }
        llm_model_func_input = hf_model_complete

    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API', 'ollama', or 'hf'.")

    timing_state = {
        "sum_kw_sec": 0.0,
        "sum_gen_sec": 0.0,
    }

    async def timed_llm_model_func(*args, **kwargs):
        is_kw = bool(kwargs.get("keyword_extraction", False))
        t_start = time.time()
        try:
            return await llm_model_func_input(*args, **kwargs)
        finally:
            dt = time.time() - t_start
            if is_kw:
                timing_state["sum_kw_sec"] += dt
            else:
                timing_state["sum_gen_sec"] += dt

    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=timed_llm_model_func,
        llm_model_name=model_name,
        llm_model_max_async=4,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag, timing_state


async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: List[dict],
    sample: int,
    retrieve_topk: int,
    q_start,
    q_end,
    rand_n: Optional[int],  
    seed: int               
):
    logging.info(f"📚 Processing corpus: {corpus_name}")

    rag, timing_state = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        mode=mode,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key
    )

    # Index corpus (await if coroutine)
    ins = rag.insert(context)
    if asyncio.iscoroutine(ins):
        await ins
        

    # 等待内部管线空闲（尽量不改你原来的等待逻辑）
    try:
        if hasattr(rag, "wait_for_idle"):
            w = rag.wait_for_idle()
            if asyncio.iscoroutine(w):
                await w
        else:
            import time as _t
            start = _t.time()
            while _t.time() - start < 30:
                num_chunks = 0
                try:
                    if hasattr(rag, "text_store"):
                        num_chunks = await rag.text_store.count() if asyncio.iscoroutinefunction(rag.text_store.count) else rag.text_store.count()
                    elif hasattr(rag, "chunk_store"):
                        num_chunks = await rag.chunk_store.count() if asyncio.iscoroutinefunction(rag.chunk_store.count) else rag.chunk_store.count()
                except Exception:
                    pass
                if num_chunks and num_chunks > 0:
                    break
                await asyncio.sleep(0.25)
    except Exception:
        pass

    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    corpus_questions = questions.get(corpus_name, [])
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return

    # if sample and sample < len(corpus_questions):
    #     corpus_questions = corpus_questions[:sample]

    # if q_start is not None or q_end is not None:
    #     start_idx = (q_start - 1) if q_start else 0
    #     end_idx = q_end if q_end else len(corpus_questions)
    #     corpus_questions = corpus_questions[start_idx:end_idx]

    # 1) (Optional) original debug cap on total questions BEFORE windowing (leave if you still want it)
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]

    # 2) Define the window using q_start/q_end (1-based, inclusive for start, exclusive for end)
    if q_start is not None or q_end is not None:
        start_idx = (q_start - 1) if q_start else 0          # clamp start (1-based -> 0-based)
        end_idx   = q_end if q_end else len(corpus_questions) # end is exclusive already
        start_idx = max(0, min(start_idx, len(corpus_questions)))
        end_idx   = max(start_idx, min(end_idx, len(corpus_questions)))
        window = corpus_questions[start_idx:end_idx]
    else:
        window = corpus_questions  # whole set if no q_start/q_end

    # 3) If rand_n is set, randomly sample INSIDE the window; otherwise use the whole window
    if rand_n is not None and rand_n > 0:
        k = min(rand_n, len(window))
        rng = random.Random(seed)
        corpus_questions = rng.sample(window, k)
    else:
        corpus_questions = window

    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")

    output_dir = f"./results/lightrag/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")

    results = []
    query_type = 'hybrid'

    for q in tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}"):
        query_param = QueryParam(mode=query_type, top_k=retrieve_topk)

        timing_state["sum_kw_sec"] = 0.0
        timing_state["sum_gen_sec"] = 0.0

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

        t_total_start = perf_counter()
        result = rag.query(
            q["question"],
            param=query_param,
            system_prompt=SYSTEM_PROMPT
        )
        if asyncio.iscoroutine(result):
            result = await result
        t_total_end = perf_counter()

        context_ret = ""
        if isinstance(result, tuple):
            response = result[0]
            if len(result) >= 2:
                context_ret = result[1] if result[1] is not None else ""
        else:
            response = result
        predicted_answer = str(response)

        total_latency = t_total_end - t_total_start

        gen_latency_sec = float(timing_state.get("sum_gen_sec", 0.0))
        retrieval_latency_sec = total_latency - gen_latency_sec

        approx_prompt_chars = len(SYSTEM_PROMPT) + len(context_ret) + len(q["question"])
        input_tokens  = approx_prompt_chars / 4.0
        output_tokens = len(predicted_answer) / 4.0

        device = "ollama" if mode.lower() == "ollama" else ("hf" if mode.lower() == "hf" else "api")
        try:
            if mode.lower() == "hf" and _HF_CACHE["model"] is not None:
                dtype = str(_HF_CACHE["model"].dtype)
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

        predicted_answer = strip_think(predicted_answer)[0]

        gold_answer = q.get("answer", "") or ""
        gt_ctx_raw = q.get("evidence", "")
        ground_truth_context = " ".join([str(x) for x in gt_ctx_raw]) if isinstance(gt_ctx_raw, list) else str(gt_ctx_raw or "")

        predicted_answer_norm = _normalize_space(predicted_answer)
        gold_answer_norm      = _normalize_space(gold_answer)
        context_ret_norm      = _normalize_space(context_ret)
        ground_truth_context  = _normalize_space(ground_truth_context)

        if "i don't know" in predicted_answer_norm.lower() or "no-context" in predicted_answer_norm.lower():
            eval_result_correctness = 0.0
        else:
            eval_result_correctness = reward_sbert_cached(predicted_answer_norm, gold_answer_norm)
        if context_ret == "":
            eval_result_context = 0.0
        else:
            eval_result_context = reward_sbert_cached(context_ret_norm, ground_truth_context)

        record = {
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": context_ret,
            "evidence": q["evidence"],
            "question_type": q["question_type"],
            "generated_answer": predicted_answer,
            "ground_truth": q.get("answer"),

            **gen_info,

            "correctness": float(eval_result_correctness),
            "context_similarity": float(eval_result_context),
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
        }
    }

    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], help="Subset to process (medical or novel)")
    parser.add_argument("--q_start", type=int, default=None, help="Start index of questions (1-based, inclusive)")
    parser.add_argument("--q_end", type=int, default=None, help="End index of questions (1-based, inclusive)")

    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")

    # 现在支持 hf
    parser.add_argument("--mode", required=True, choices=["API", "ollama", "hf"], help="Use API, ollama, or hf (local transformers)")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", help="LLM model identifier (HF 模式下是 HF repo id)")
    parser.add_argument("--embed_model", default="bge-base-en", help="Embedding model name (HF/ollama 的名称)")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")

    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", help="Base URL for LLM API / ollama host")
    parser.add_argument("--llm_api_key", default="", help="API key for LLM service (can also use LLM_API_KEY env var)")


    # for random samples
    parser.add_argument("--rand_n", type=int, default=None,
        help="Randomly sample N questions **within** the q_start..q_end window (1-based, inclusive)")
    parser.add_argument("--seed", type=int, default=42,
        help="RNG seed for --rand_n sampling (reproducible)")

    args = parser.parse_args()

    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    if args.mode not in ["API", "ollama", "hf"]:
        logging.error(f"Invalid mode: {args.mode}. Valid options: ['API','ollama','hf']")
        return

    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]

    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if args.mode == "API" and not api_key:
        logging.warning("No API key provided! Requests may fail.")

    os.makedirs(args.base_dir, exist_ok=True)

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
        logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return

    if args.sample:
        corpus_data = corpus_data[:1]

    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            question_data = json.load(f)
            grouped_questions = group_questions_by_source(question_data)
        logging.info(f"Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return

    for item in corpus_data:
        corpus_name = item["corpus_name"]
        context = item["context"]
        asyncio.run(
            process_corpus(
                corpus_name=corpus_name,
                context=context,
                base_dir=args.base_dir,
                mode=args.mode,
                model_name=args.model_name,
                embed_model_name=args.embed_model,
                llm_base_url=args.llm_base_url,
                llm_api_key=api_key,
                questions=grouped_questions,
                sample=args.sample,
                retrieve_topk=args.retrieve_topk,
                q_start=args.q_start,
                q_end=args.q_end,
                rand_n=args.rand_n,     
                seed=args.seed          
            )
        )

if __name__ == "__main__":
    # os.environ["LIGHTRAG_WORKER_TIMEOUT"] = "600"
    main()


# python .\Examples\run_lightrag_sample.py --subset medical --mode hf --model_name "Qwen/Qwen2.5-3B" --embed_model "BAAI/bge-base-en" --q_start 21 --base_dir ./lightrag_workspace_gpt4omini --rand_n 100
