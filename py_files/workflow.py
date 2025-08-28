# -*- coding: utf-8 -*-
"""
Minimal JSON-prompted YES/NO workflow
- No FAISS / retrieval
- Robust to MPS/CPU quirks (forces CPU unless CUDA is available)
- Stable DataFrame schema even on generation errors
- Accuracy + cost summaries
"""

import os
import time
from typing import List, Dict, Optional

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Your JSON prompt helper (must be on PYTHONPATH)
from graph_generator.generator_with_json import get_js_msgs_use_triples,get_merged_message

try:
    from transformers import set_seed  # Utility for reproducibility
except Exception:
    set_seed = None


# Prefer CPU on macOS to avoid MPS isin() issues
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

CONFIG = {
    # === LLM ===
    "llm_model_id": "microsoft/Phi-4-mini-reasoning",
    "device_map": "auto",       # will coerce to "cpu" if no CUDA
    "dtype_policy": "auto",     # "auto"|"bf16"|"fp16"|"fp32"
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.4,
    "top_p": 1.0,
    "return_full_text": False,  # HF pipeline returns only completion

    # === Answering ===
    "answer_mode": "YES_NO",    # currently supports "YES_NO"
    "answer_uppercase": True,   # YES/NO vs yes/no

    # === Prompt toggles (retrieval removed; only triples / question)
    "include_current_triples": True,

    # === Reproducibility ===
    "seed": None,
}


# -------------------------
# Helpers
# -------------------------
def _select_dtype() -> torch.dtype:
    policy = CONFIG.get("dtype_policy", "auto")
    if policy == "bf16": return torch.bfloat16
    if policy == "fp16": return torch.float16
    if policy == "fp32": return torch.float32

    # auto: CUDA prefers bf16/fp16; CPU/MPS use fp32 for stability
    if torch.cuda.is_available():
        bf16_ok = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        return torch.bfloat16 if bf16_ok else torch.float16
    return torch.float32


def _yn(text_yes="YES", text_no="NO"):
    return (text_yes, text_no) if CONFIG.get("answer_uppercase", True) else (text_yes.lower(), text_no.lower())


# -------------------------
# LLM loader
# -------------------------
def load_llm_pipeline(
    model_id: Optional[str] = None,
    device_map: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    max_new_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    do_sample: Optional[bool] = None,
    return_full_text: Optional[bool] = None,
):
    """
    Return a text-generation pipeline for QA generation.
    All defaults pull from CONFIG; any arg here will override CONFIG.
    """
    model_id = model_id or CONFIG["llm_model_id"]

    # Pick device map safely: only use "auto" if CUDA is available; otherwise force CPU
    if device_map is None or device_map == "auto":
        device_map = "auto" if torch.cuda.is_available() else "cpu"

    dtype = dtype or _select_dtype()
    max_new_tokens = max_new_tokens or CONFIG["max_new_tokens"]
    temperature = CONFIG["temperature"] if temperature is None else temperature
    top_p = CONFIG["top_p"] if top_p is None else top_p
    do_sample = CONFIG["do_sample"] if do_sample is None else do_sample
    return_full_text = CONFIG["return_full_text"] if return_full_text is None else return_full_text

    if set_seed and isinstance(CONFIG.get("seed"), int):
        set_seed(CONFIG["seed"])

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


# -------------------------
# Prompt builder
# -------------------------
def make_json_qa_prompt(question: str) -> str:
    """
    Build the JSON-style prompt using your triple generator.
    """
    sections = []
    if CONFIG.get("include_current_triples", True):
        # msg1, msg2 = get_js_msgs_use_triples(question)
        # print(msg1)
        # print(msg2)
        # msg1.update(msg2)
        # msg1.pop('sid', None)
        # sections.append(msg1)

        msg = get_merged_message(question)
        sections.append(msg)
    else:
        sections.append(f"Q: {question}\nAnswer YES or NO:")
    return "\n\n".join(sections)


# -------------------------
# LLM answering
# -------------------------
def answer_with_llm(
    question: str,
    gen_pipe,
    prompt: Optional[str] = None
) -> str:
    if prompt is None:
        prompt = make_json_qa_prompt(question)
        print(prompt)

    out = gen_pipe(prompt)
    text = out[0]["generated_text"]

    # If return_full_text=False → transformers returns only the completion
    if CONFIG.get("return_full_text", True):
        answer = text[len(prompt):].strip()
    else:
        answer = text.strip()

    mode = CONFIG.get("answer_mode", "YES_NO")
    if mode == "YES_NO":
        yes, no = _yn("YES", "NO")
        a = answer.strip().lower()

        # strict first
        if a == "yes":
            return yes
        if a == "no":
            return no
        # lenient keyword check
        if ("yes" in a) and ("no" not in a):
            return yes
        if ("no" in a) and ("yes" not in a):
            return no
        # final fallback policy: choose NO
        return no

    return answer


# -------------------------
# Normalization utilities
# -------------------------
def _normalize_yesno(text: str) -> str:
    if text is None:
        return "NO"
    t = str(text).strip().lower()
    if t == "yes":
        return "YES"
    if t == "no":
        return "NO"
    if "yes" in t and "no" not in t:
        return "YES"
    if "no" in t and "yes" not in t:
        return "NO"
    return "NO"


def _ensure_uppercase_yesno(text: str) -> str:
    yn = _normalize_yesno(text)
    return yn if CONFIG.get("answer_uppercase", True) else yn.lower()


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text or "", add_special_tokens=False))


# -------------------------
# Measurement
# -------------------------
def measure_once(
    question: str,
    gen_pipe,
    tokenizer,
    use_cuda_mem: bool = True,
) -> Dict:
    
    prompt = make_json_qa_prompt(question)
    in_tok = _count_tokens(tokenizer, prompt)

    peak_mem = None
    if use_cuda_mem and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    answer = answer_with_llm(question, gen_pipe, prompt)
    dt = time.perf_counter() - t0

    answer_text = "" if answer is None else str(answer)
    out_tok = _count_tokens(tokenizer, answer_text)

    if use_cuda_mem and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)

    return {
        "prompt":prompt,
        "question": question,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "latency_sec": dt,
        "peak_vram_MiB": peak_mem,
        "prompt_chars": len(prompt),
        "answer": answer_text,   # <- always present
        "error": None,           # <- unified schema
    }


def batch_measure(
    questions: List[str],
    gen_pipe,
    tokenizer,
) -> pd.DataFrame:
    """
    Evaluate multiple CONFIG setups. Retrieval is removed; we only flip include_current_triples.
    """
    rows = []




    for q in questions:
        try:
            rec = measure_once(
                question=q,
                gen_pipe=gen_pipe,
                tokenizer=tokenizer,
            )
            rows.append(rec)
            print(rows)
        except Exception as e:
            rows.append({
                "prompt":None,
                "question": q,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "latency_sec": None,
                "peak_vram_MiB": None,
                "prompt_chars": None,
                "answer": None,            # keep column present
                "error": str(e)
            })
            print(rows)

    # restore

    return pd.DataFrame(rows)


def summarize_cost(df: pd.DataFrame, base_label: str, target_label: str):
    """
    Compare average token usage, latency, and GPU memory between two configurations.
    """
    A = df[df["label"] == base_label]
    B = df[df["label"] == target_label]
    if A.empty or B.empty:
        print("Not enough data for comparison.")
        return

    def avg(col):
        a, b = A[col].mean(), B[col].mean()
        return a, b, (b - a) / max(1e-9, a)

    for col in ["input_tokens", "output_tokens", "total_tokens", "latency_sec", "peak_vram_MiB", "prompt_chars"]:
        if col in df.columns:
            a, b, d = avg(col)
            print(f"{col:>15s} | {base_label}: {a:8.2f} | {target_label}: {b:8.2f} | Δ%: {d*100:7.2f}%")


# -------------------------
# Gold & evaluation
# -------------------------
# def attach_gold(df: pd.DataFrame, gold_map: dict) -> pd.DataFrame:
#     gold_df = pd.DataFrame(list(gold_map.items()), columns=["question", "gold"])
#     gold_df["gold"] = gold_df["gold"].map(_ensure_uppercase_yesno)
#     out = df.merge(gold_df, on="question", how="left")
#     return out


# def evaluate_accuracy(df_with_gold: pd.DataFrame) -> pd.DataFrame:
#     """
#     Input: DataFrame with columns ['label','question','answer','gold','error']
#     Output: Per-configuration accuracy table and confusion matrices.
#     """
#     df = df_with_gold.copy()

#     # Ensure answer column exists
#     if "answer" not in df.columns:
#         first_err = df.get("error").dropna().head(1).to_list()
#         print("No 'answer' column present (all generations failed). First error:", first_err)
#         return pd.DataFrame()

#     # Drop rows with errors or missing answers
#     if "error" in df.columns:
#         df = df[df["error"].isna()]
#     df = df[df["answer"].notna()].copy()
#     if df.empty:
#         print("No successful generations to evaluate.")
#         return pd.DataFrame()

#     df["pred"] = df["answer"].map(_ensure_uppercase_yesno)

#     has_gold = df["gold"].notna() if "gold" in df.columns else False
#     if not has_gold.any():
#         print("⚠️ No gold labels found. Provide a gold_map covering your questions.")
#         return pd.DataFrame()

#     df = df[has_gold].copy()
#     df["correct"] = (df["pred"] == df["gold"]).astype(int)

#     overall_acc = df["correct"].mean() if len(df) else float("nan")
#     print(f"\n== Overall accuracy: {overall_acc:.3f} (n={len(df)}) ==")

#     by_cfg = df.groupby("label")["correct"].mean().reset_index().rename(columns={"correct": "accuracy"})
#     print("\n== Accuracy by config ==")
#     for _, row in by_cfg.iterrows():
#         n = df[df["label"] == row["label"]].shape[0]
#         print(f"{row['label']:<15s}  acc={row['accuracy']:.3f}  (n={n})")

#     print("\n== Confusion matrices by config ==")
#     for cfg, sub in df.groupby("label"):
#         cm = pd.crosstab(sub["gold"], sub["pred"], rownames=["gold"], colnames=["pred"], dropna=False)
#         for val in ["YES", "NO"]:
#             if val not in cm.index:
#                 cm.loc[val] = 0
#             if val not in cm.columns:
#                 cm[val] = 0
#         cm = cm.loc[["YES", "NO"], ["YES", "NO"]]
#         print(f"\n[Config: {cfg}]")
#         print(cm)

#     return by_cfg


# -------------------------
# Example experiment
# -------------------------
GOLD_LABELS = {
    "Is the Earth round?": "YES",
    "Is Earth flat?": "NO",
    "Does the Earth orbit the Sun?": "YES",

    "Does the Sun rise in the east?": "YES",
    "Does the Sun rise in the west?": "NO",
    "Is the Sun a star?": "YES",

    "Is Paris the capital of France?": "YES",
    "Is Paris the capital of Germany?": "NO",
    "Is the Eiffel Tower in Paris?": "YES",

    "Do humans need oxygen to survive?": "YES",
    "Can humans survive without water forever?": "NO",
    "Do humans have two lungs?": "YES",

    "Is the Moon a natural satellite of Earth?": "YES",
    "Does Earth have two moons?": "NO",
    "Does the Moon orbit the Earth?": "YES",

    "Is the Sahara Desert in South America?": "NO",
    "Is the Sahara Desert in Africa?": "YES",
    "Is the Sahara Desert the largest desert on Earth?": "YES",

    "Is the Amazon River longer than the Nile River?": "NO",
    "Is the Amazon River in Africa?": "NO",
    "Is the Nile River in Africa?": "YES",

    "Is Tokyo the capital of South Korea?": "NO",
    "Is Seoul the capital of South Korea?": "YES",
    "Is Tokyo in Japan?": "YES",

    "Do penguins live in the Arctic?": "NO",
    "Do penguins live in Antarctica?": "YES",
    "Do polar bears live in Antarctica?": "NO",

    "Is gold heavier than lead?": "NO",
    "Is gold a metal?": "YES",
    "Is lead a gas?": "NO"
}


if __name__ == "__main__":
    questions = list(GOLD_LABELS.keys())
    gen_pipe, tokenizer = load_llm_pipeline()

    # Quick sanity check
    # try:
    #     print("Sanity:", answer_with_llm("Is the Earth round?", gen_pipe))
    # except Exception as e:
    #     print("Sanity generation failed:", e)
    #     raise

    for q in questions:
        try:
            print("Sanity:", answer_with_llm(q, gen_pipe))
        except Exception as e:
            print("Sanity generation failed:", e)
            raise


    # df = batch_measure(questions, gen_pipe, tokenizer)
    # print(df.head())

    # print("\n=== Summary ===")
    # summarize_cost(df, base_label="no_ctx", target_label="with_both")

    # df_gold = attach_gold(df, GOLD_LABELS)
    # _ = evaluate_accuracy(df_gold)


# python py_files/workflow.py