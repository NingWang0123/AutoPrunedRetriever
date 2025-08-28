import networkx as nx

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from typing import List, Dict, Optional, Tuple
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from graph_generator.generator_with_json import get_js_msgs_use_triples

import time
import pandas as pd
import os


CONFIG = {
    # === Embedding & VectorStore ===
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",  # Embedding model for documents/questions
    "faiss_search_k": 3,  # Number of nearest neighbors to retrieve from FAISS

    # === LLM (text generation) ===
    "llm_model_id": "microsoft/Phi-4-mini-reasoning",  # HuggingFace model ID
    "device_map": "auto",  # Device placement: "cuda", "mps", "cpu", or "auto"
    "dtype_policy": "auto",  # Precision: "auto", "bf16", "fp16", or "fp32"
    "max_new_tokens": 256,  # Maximum tokens generated per response
    "do_sample": True,  # Whether to use sampling (True) or greedy decoding (False)
    "temperature": 0.4,  # Randomness control for sampling; lower = more deterministic
    "top_p": 1.0,  # Nucleus sampling threshold; 1.0 = no restriction
    "return_full_text": False,  # Return full text (input+output) if True, only output if False
    "seed": None,  # Random seed for reproducibility; set to int or None

    # === Prompt / Answer ===
    "answer_mode": "YES_NO",  # Answer format mode, e.g., YES/NO
    "answer_uppercase": True,  # If True → "YES"/"NO", else "yes"/"no"

    # === Prompt construction ===
    "include_retrieved_context": True,  # Include retrieved Q&A in prompt
    "include_current_triples": True,  # Include graph triples in prompt
}

try:
    from transformers import set_seed  # Utility for reproducibility
except Exception:
    set_seed = None



## RAG workflow

def _select_dtype() -> torch.dtype:
    """Choose dtype based on CONFIG['dtype_policy'] and hardware."""
    policy = CONFIG.get("dtype_policy", "auto")
    if policy == "bf16":
        return torch.bfloat16
    if policy == "fp16":
        return torch.float16
    if policy == "fp32":
        return torch.float32

    # auto mode
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # MPS backend works more reliably with fp32
    if torch.backends.mps.is_available():
        return torch.float32
    return torch.float32

def _yn(text_yes="YES", text_no="NO"):
    return (text_yes, text_no) if CONFIG.get("answer_uppercase", True) else (text_yes.lower(), text_no.lower())

# =========================
# Embeddings / Vectorstore
# =========================
emb = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model"])  # Local embedding model (MiniLM-L6-v2, 384 dim)

def build_faiss_index(docs: List[Document]) -> FAISS:
    return FAISS.from_documents(docs, emb)

# =========================
# LLM Loader
# =========================
def load_llm_pipeline(
    model_id: Optional[str] = None,       # HuggingFace model id
    device_map: Optional[str] = None,     # Device placement
    dtype: Optional[torch.dtype] = None,  # Torch dtype
    max_new_tokens: Optional[int] = None, # Max tokens per generation
    temperature: Optional[float] = None,  # Sampling temperature
    top_p: Optional[float] = None,        # Nucleus sampling threshold
    do_sample: Optional[bool] = None,     # Sampling vs greedy
    return_full_text: Optional[bool] = None,  # Return input+output if True
):
    """
    Return a text-generation pipeline for QA generation.
    All defaults pull from CONFIG; any arg here will override CONFIG.
    """
    model_id = model_id or CONFIG["llm_model_id"]
    device_map = device_map or CONFIG["device_map"]
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

# =========================
# Question → json
# =========================

# =========================
# Prompt Builder
# =========================
def make_json_qa_prompt(
    question: str,
    # retrieved_docs = None
) -> str:
    ### ignore retrieve now 
    # 1) retrieved context (if any)
    sections = []
    # if retrieved_docs and CONFIG.get("include_retrieved_context", True):
    #     doc0, score0 = retrieved_docs[0]
    #     related_triples = doc0.page_content.strip()
    #     related_answer  = doc0.metadata.get("llm_answer", "")
    #     sections.append(
    #         "<<<RETRIEVED_CONTEXT_START>>>\n"
    #         "For the following questions, consider each question as an individual question(they are all sentences or sentence with relation in json)" \
    #         "You don't have to follow it completely, just use it as a reference.\n"
    #         f"[RELATED QUESTION'S json]:\n{related_triples}\n"
    #         f"[RELATED QUESTION'S ANSWER]: {related_answer}\n"
    #         "<<<RETRIEVED_CONTEXT_END>>>"
    #     )

    # 2) json format

    msg1,msg2= get_js_msgs_use_triples(question)

    # 3) task instructions (placed at the end)
    # skip for we alreay gace
    # yes, no = _yn("YES", "NO")
    # rules = (
    #     "[TASK]: You are a precise QA assistant for binary (yes/no) questions.\n"
    #     f"- Output ONLY one token: {yes} or {no}.\n"
    #     "- Do NOT copy or summarize any context.\n"
    #     "- Do NOT show reasoning, steps, or extra words.\n"
    #     f"[ANSWER]: "
    # )
    # sections.append(rules)

    sections.append(msg1)
    sections.append(msg2)

    # Final prompt
    prompt = "\n\n".join(sections)
    return prompt

# =========================
# LLM Answerer
# =========================
def answer_with_llm(
    question: str,
    gen_pipe,
    faiss_db = None,
    prompt = None
) -> str:
    # retrieved_docs = None
    # if faiss_db:
    #     k = CONFIG.get("faiss_search_k", 3)  # Number of docs to retrieve
    #     _, hits = similarity_search_graph_docs(question, parser, faiss_db, k=k)
    #     retrieved_docs = hits
        
    if prompt == None:
        prompt = make_json_qa_prompt(question)

    out = gen_pipe(prompt)
    text = out[0]["generated_text"]

    # If return_full_text=False → only new content; else trim prefix
    if CONFIG.get("return_full_text", True):
        answer = text[len(prompt):].strip()
    else:
        answer = text.strip()

    # Normalize YES/NO case
    if CONFIG.get("answer_mode", "YES_NO"):
        yes, no = _yn("YES", "NO")
        a = answer.strip().lower()
        if "yes" in a and "no" not in a:
            answer = yes
            print(answer)
            return answer
        elif "no" in a and "yes" not in a:
            answer = no
            print(answer)
            return answer
        else:
            answer = answer_with_llm(question, gen_pipe, faiss_db, prompt)
    
    

# =========================
# Build Docs with LLM Answer
# =========================
def build_docs_with_answer(
    questions: List[str],
    # parser,
    gen_pipe,
    *,
    add_prompt_snapshot: bool = False,
    faiss_db = None
) -> List[Document]:
    docs: List[Document] = []
    for qid, q in enumerate(questions, start=1):
        # Get LLM answer
        answer = answer_with_llm(q, gen_pipe, faiss_db)

        metadata = {
            "graph_id": f"Q{qid}",
            "question": q,
            "llm_model": CONFIG["llm_model_id"],
            "llm_answer": answer,
            "created_at": int(time.time()),
        }
        if add_prompt_snapshot:
            metadata["prompt_snapshot"] = make_json_qa_prompt(q)

        docs.append(Document(metadata=metadata))
    return docs


def build_faiss_index(docs: List[Document]) -> FAISS:
    vectordb = FAISS.from_documents(docs, emb)
    return vectordb



#### batches measure 
def _normalize_yesno(text: str) -> str:
    """
    Normalize the LLM output to strict 'YES'/'NO' values.
    Any non-explicit 'yes'/'no' output is treated as 'NO' to align with the prompt rule:
    "If uncertain, choose NO."
    """
    if text is None:
        return "NO"
    t = str(text).strip().lower()
    if t == "yes":
        return "YES"
    if t == "no":
        return "NO"
    # Fallback: if text contains a clear yes/no keyword
    if "yes" in t and "no" not in t:
        return "YES"
    if "no" in t and "yes" not in t:
        return "NO"
    return "NO"

def _ensure_uppercase_yesno(text: str) -> str:
    """
    Ensure the returned value matches the CONFIG['answer_uppercase'] setting.
    Internally, evaluation always uses uppercase comparison.
    """
    yn = _normalize_yesno(text)
    if CONFIG.get("answer_uppercase", True):
        return yn
    return yn.lower()

# ===== Accuracy & Confusion Reporting =====

def attach_gold(df: pd.DataFrame, gold_map: dict) -> pd.DataFrame:
    """
    Attach gold labels to the batch_measure results DataFrame.
    Requires df.question to match keys in gold_map.
    """
    gold_df = pd.DataFrame(list(gold_map.items()), columns=["question", "gold"])
    # Normalize gold labels to uppercase or lowercase as per config
    gold_df["gold"] = gold_df["gold"].map(_ensure_uppercase_yesno)
    out = df.merge(gold_df, on="question", how="left")
    return out

def evaluate_accuracy(df_with_gold: pd.DataFrame) -> pd.DataFrame:
    """
    Input: DataFrame with columns ['label','question','answer','gold']
    Output: Per-configuration accuracy table and confusion matrices.
    """
    df = df_with_gold.copy()
    # Normalize predicted answers
    df["pred"] = df["answer"].map(_ensure_uppercase_yesno)

    # Check if any gold labels exist
    has_gold = df["gold"].notna()
    if not has_gold.any():
        print("⚠️ No gold labels found. Please provide a gold_map that covers your questions.")
        return pd.DataFrame()

    df = df[has_gold].copy()
    df["correct"] = (df["pred"] == df["gold"]).astype(int)

    # Overall accuracy
    overall_acc = df["correct"].mean() if len(df) else float("nan")
    print(f"\n== Overall accuracy: {overall_acc:.3f} (n={len(df)}) ==")

    # Accuracy per configuration
    by_cfg = df.groupby("label")["correct"].mean().reset_index().rename(columns={"correct":"accuracy"})
    print("\n== Accuracy by config ==")
    for _, row in by_cfg.iterrows():
        n = df[df["label"] == row["label"]].shape[0]
        print(f"{row['label']:<15s}  acc={row['accuracy']:.3f}  (n={n})")

    # Confusion matrix for each configuration
    print("\n== Confusion matrices by config ==")
    for cfg, sub in df.groupby("label"):
        cm = pd.crosstab(sub["gold"], sub["pred"], rownames=["gold"], colnames=["pred"], dropna=False)
        # Ensure YES/NO rows and columns exist
        for val in ["YES","NO"]:
            if val not in cm.index:
                cm.loc[val] = 0
            if val not in cm.columns:
                cm[val] = 0
        cm = cm.loc[["YES","NO"], ["YES","NO"]]
        print(f"\n[Config: {cfg}]")
        print(cm)

    return by_cfg

def per_question_delta(df_with_gold: pd.DataFrame, base_label: str, target_label: str) -> pd.DataFrame:
    """
    Compare the predictions between base and target configurations for each question.
    Output columns:
    - question
    - gold (gold label)
    - pred_base (prediction from base configuration)
    - pred_target (prediction from target configuration)
    - delta_correct (-1, 0, 1): 1 = improvement, -1 = regression, 0 = no change
    """
    df = df_with_gold.copy()
    df["pred"] = df["answer"].map(_ensure_uppercase_yesno)
    df = df[df["gold"].notna()].copy()

    base = df[df["label"] == base_label][["question","gold","pred"]].rename(columns={"pred":"pred_base"})
    tgt  = df[df["label"] == target_label][["question","pred"]].rename(columns={"pred":"pred_target"})
    j = base.merge(tgt, on="question", how="inner")
    j["delta_correct"] = (j["pred_target"] == j["gold"]).astype(int) - (j["pred_base"] == j["gold"]).astype(int)
    return j.sort_values(by=["delta_correct","question"], ascending=[False, True])

# def _get_retrieved_docs_for_prompt(
#     question: str,
#     parser,
#     faiss_db=None,
#     k: Optional[int] = None,
# ):
#     """
#     Retrieve documents based on CONFIG['include_retrieved_context'] setting.
#     Returns hits as [(Document, score), ...].
#     """
#     if not faiss_db or not CONFIG.get("include_retrieved_context", True):
#         return None
#     k = k or CONFIG.get("faiss_search_k", 3)
#     _, hits = similarity_search_graph_docs(question, parser, faiss_db, k=k)
#     return hits if hits else None

def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens in the given text using the provided tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def measure_once(
    question: str,
    gen_pipe,              # Pipeline object from load_llm_pipeline
    tokenizer,             # Tokenizer from load_llm_pipeline for token counting
    faiss_db=None,
    *,
    label: Optional[str] = None,
    use_cuda_mem: bool = True,
) -> Dict:
    """
    Build a prompt according to current CONFIG (controlled by include_retrieved_context / include_current_triples),
    then query the LLM and return:
      - input_tokens / output_tokens / total_tokens
      - latency_sec (time taken for generation)
      - peak_vram_MiB (optional, for GPU usage)
      - flags indicating whether retrieval and triples were used
    """
    # # 1) Retrieval (if enabled)
    # retrieved_docs = _get_retrieved_docs_for_prompt(
    #     question, parser, faiss_db=faiss_db, k=CONFIG.get("faiss_search_k", 3)
    # )

    # # 2) Parse the question into graph/triples
    # G, rels = parse_question_to_graph_generic(parser, question)

    # 3) Build the prompt (with or without triples based on config)

    prompt = make_json_qa_prompt(question)

    # 4) Count input tokens
    in_tok = _count_tokens(tokenizer, prompt)

    # 5) Time the generation (and optionally measure peak GPU memory)
    peak_mem = None
    if use_cuda_mem and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    answer = answer_with_llm(question, gen_pipe, faiss_db, prompt)
    dt = time.perf_counter() - t0

    # 6) Count output tokens
    out_tok = _count_tokens(tokenizer, answer)

    # 7) Record peak GPU memory usage if available
    if use_cuda_mem and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated() / (1024**2)


    # skip this

    # 8) Flags for retrieval and triple usage
    # used_retrieval = bool(retrieved_docs)
    # used_triples = bool(rels) and CONFIG.get("include_current_triples", True)

    return {
        # "label": label or ("with_graph_ctx" if used_triples or used_retrieval else "no_graph_ctx"),
        "question": question,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "latency_sec": dt,
        "peak_vram_MiB": peak_mem,
        # "used_retrieval": used_retrieval,
        # "used_current_triples": used_triples,
        "prompt_chars": len(prompt),
        "answer": answer,
    }

# ===== Batch Evaluation and Summary =====
def batch_measure(
    questions: List[str],
    gen_pipe,
    tokenizer,
    # parser,
    faiss_db=None,
    *,
    flip_configs: List[Dict] = None,
) -> pd.DataFrame:
    """
    Evaluate multiple CONFIG setups (e.g., with/without retrieval, with/without triples) on a list of questions,
    and return a summary DataFrame with all results.
    flip_configs: List of dictionaries for temporary CONFIG overrides. Example:
        [{"include_retrieved_context": False, "include_current_triples": False, "label": "no_ctx"},
         {"include_retrieved_context": True,  "include_current_triples": True,  "label": "with_both"}]
    """
    rows = []
    if not flip_configs:
        flip_configs = [{"label": "current_CONFIG"}]

    for cfg in flip_configs:
        # Temporarily override CONFIG
        old_retrieve = CONFIG.get("include_retrieved_context", True)
        old_triples  = CONFIG.get("include_current_triples", True)
        if "include_retrieved_context" in cfg:
            CONFIG["include_retrieved_context"] = cfg["include_retrieved_context"]
        if "include_current_triples" in cfg:
            CONFIG["include_current_triples"] = cfg["include_current_triples"]

        for q in questions:
            try:
                rec = measure_once(
                    question=q,
                    gen_pipe=gen_pipe,
                    tokenizer=tokenizer,
                    faiss_db=faiss_db,
                    label=cfg.get("label")
                )
                rows.append(rec)
            except Exception as e:
                rows.append({
                    "label": cfg.get("label"),
                    "question": q,
                    "error": str(e)
                })

        # Restore original CONFIG
        CONFIG["include_retrieved_context"] = old_retrieve
        CONFIG["include_current_triples"]   = old_triples

    return pd.DataFrame(rows)

def summarize_cost(df: pd.DataFrame, base_label: str, target_label: str):
    """
    Compare average token usage, latency, and GPU memory between two configurations.
    Prints relative percentage changes.
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


### exp examples


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


# Batch A/B comparison (no context vs. both retrieval & triples)
questions = list(GOLD_LABELS.keys())

gen_pipe, tokenizer = load_llm_pipeline()   # Use your loader above

# Batch A/B comparison (no context vs. both retrieval & triples)
questions = list(GOLD_LABELS.keys())

df = batch_measure(
    questions, gen_pipe, tokenizer, faiss_db = None,
)
print(df.head())

print("\n=== Summary ===")
summarize_cost(df, base_label="no_ctx", target_label="with_both")

df_gold = attach_gold(df, GOLD_LABELS)
acc_table = evaluate_accuracy(df_gold)

# python py_files/workflow.py