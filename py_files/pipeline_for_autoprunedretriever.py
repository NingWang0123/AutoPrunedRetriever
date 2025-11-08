import os
import json
import argparse
from pathlib import Path
from glob import glob
from typing import List, Optional, Dict, Tuple
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Embeddings / LLM wrappers
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceBgeEmbeddings  # not used
# from langchain_openai import ChatOpenAI  # not used

# Your project modules
from AutoPrunedRetriever_advanced_final import AutoPrunedRetriver
from dpo_exactgraphrag import (
    make_preference_dataset_2head,
    train_dpo_2head,
    make_preference_dataset_2head_using_llm,  # not used but kept
    default_reward,
    answer_with_auto_strategy,
    save_pref_examples,
    load_pref_examples,
    ANSWERS_CHOICES,
    THINKINGS_CHOICES,
    FACTS_CHOICES,
)
import reward_func_dpo


# ---------------------------------------------------------------------
# Helpers: robust path handling + HF fallback + env fallbacks
# ---------------------------------------------------------------------

def _resolve_rel_to_config(cfg_path: Optional[str], p: Optional[str]) -> Optional[Path]:
    """Resolve a path relative to the config file location (if provided)."""
    if not p:
        return None
    P = Path(p)
    if P.is_absolute():
        return P
    base = Path(cfg_path).resolve().parent if cfg_path else Path.cwd()
    return (base / P).resolve()


def resolve_path_or_hf(repo_id: Optional[str], file_or_glob: str, cfg_path: Optional[str] = None) -> List[Path]:
    """
    Returns concrete local Paths:
      1) Try local (file or glob), resolved relative to cfg file dir if provided.
      2) Else if repo_id is set AND 'file_or_glob' is a single path (no wildcards),
         download via hf_hub_download(repo_type='dataset').
      3) Else raise.
    """
    local_candidate = _resolve_rel_to_config(cfg_path, file_or_glob)
    matches: List[Path] = []
    if local_candidate:
        if local_candidate.is_file():
            matches = [local_candidate]
        else:
            mg = glob(str(local_candidate))
            matches = [Path(m).resolve() for m in mg if Path(m).is_file()]
    if matches:
        return matches

    if repo_id and not any(sym in file_or_glob for sym in "*?[]"):
        # HF dataset path
        p = hf_hub_download(repo_id, file_or_glob, repo_type="dataset")
        return [Path(p).resolve()]

    raise FileNotFoundError(f"No local file(s) matched '{file_or_glob}' and no HF repo_id provided.")


def ensure_parent(path_like) -> Path:
    """Create parent directory for a file path if missing."""
    p = Path(path_like)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _maybe_from_env(v, env_key: Optional[str]):
    """Config/CLI > env (if key provided) > None."""
    return v if v not in (None, "", []) else (os.getenv(env_key) if env_key else None)


def _load_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)


def load_questions_any(repo_id: Optional[str], quest_spec: str, cfg_path: Optional[str] = None) -> List[dict]:
    """Load questions from JSON or JSONL (local or HF dataset)."""
    paths = resolve_path_or_hf(repo_id, quest_spec, cfg_path)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one questions file, got: {paths}")
    p = paths[0]
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# Core workflow
# ---------------------------------------------------------------------

def compress_rag_workflow(
    REPO_ID: Optional[str],
    CORPUS_FILE: str,
    QUEST_FILE: str,
    SEED_N: int,
    TEST_N: int,
    top_m: int,
    top_k: int,
    combine_ent_sim: float,
    q_combine_sim: float,
    aft_combine_sim: float,
    semantic_overlap_sim: float,
    ini_meta_json,
    saved_examples_name,
    reward_func,
    reward_func_mode: str,
    final_json_path,
    chunking_use: str = "rebel",
    chunking_api: Optional[str] = None,
    llm_api: Optional[str] = None,
    cfg_path: Optional[str] = None,
    subchunk_mode = 'chars'
):
    print("» Initialising embeddings & LLM …")
    # You can share the same model for word/sentence; keeping your original two for parity.
    word_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    sent_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

    # Your OpenAI wrapper; expects key via llm_api (already resolved with env fallback in main)
    from llm_api import OpenAILLM
    api_llm = OpenAILLM(
        include_thinkings=True,
        model_name="gpt-4o-mini",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        api_key=llm_api,
    )

    # Normalize IO path types
    ini_meta_json = Path(ini_meta_json)
    saved_examples_name = Path(saved_examples_name)
    final_json_path = Path(final_json_path)

    # -----------------------------------------------------------------
    # 2) Load benchmark Q-A-E data
    # -----------------------------------------------------------------
    print("» Loading benchmark questions / answers / evidence …")
    qrows = load_questions_any(REPO_ID, QUEST_FILE, cfg_path=cfg_path)
    row_lookup = {r["question"].strip(): r for r in qrows}
    gold_lookup = {q: r["answer"] for q, r in row_lookup.items()}

    all_questions = list(row_lookup.keys())
    all_seed_questions = all_questions[:SEED_N]
    midpoint = len(all_seed_questions) // 2
    train_questions = all_seed_questions[:midpoint]
    seed_questions = all_seed_questions[midpoint:]
    train_answers = [gold_lookup.get(q) for q in train_questions]
    test_questions = all_questions[SEED_N: SEED_N + TEST_N]

    # -----------------------------------------------------------------
    # 3) Build/Load AutoPrunedRetriever and preload facts (corpus)
    # -----------------------------------------------------------------
    ini = {}
    pre_loaded_meta = False
    if ini_meta_json.is_file():
        try:
            ini = json.loads(ini_meta_json.read_text(encoding="utf-8"))
            pre_loaded_meta = True
        except json.JSONDecodeError:
            print("[warn] ini_meta_json exists but invalid; starting with empty meta.")

    cr = AutoPrunedRetriver(
        ini_meta_codebook=ini,
        sentence_emb=sent_emb,
        word_emb=word_emb,
        llm=api_llm,
        thinkings_choice="not_include",
        answers_choice="unique",
        facts_choice="include_all",
        top_m=top_m,
        top_k=top_k,
        combine_ent_sim=combine_ent_sim,
        q_combine_sim=q_combine_sim,
        aft_combine_sim=aft_combine_sim,
        semantic_overlap_sim=semantic_overlap_sim,
        use_word=True,
        chunking_use=chunking_use,
        chunking_api=chunking_api,
    )

    if not pre_loaded_meta:
        print("» Pre-loading corpus as facts …")
        cr.load_and_merge_facts(
            CORPUS_FILE,            # already absolute path (resolved in main)
            chunk_tokens=1200,
            overlap_tokens=100,
            sub_chunk_chars=200,
            sub_chunk_overlap=50,
            tokenizer_name="gpt-4o-mini",
            subchunk_batch=1000,
            subchunk_mode = subchunk_mode,
            sub_chunk_token_size = 256,       # used when subchunk_mode="tokens"
            sub_chunk_token_overlap = 50,     # used when subchunk_mode="tokens"
        )
        cr._facts_preloaded = True

        def make_json_safe(obj):
            """Recursively convert numpy arrays into lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            return obj

        ensure_parent(ini_meta_json)
        ini_meta_json.write_text(json.dumps(make_json_safe(cr.meta_codebook), indent=2), encoding="utf-8")

        print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
              f"|R|={len(cr.meta_codebook['r'])} "
              f"|edges|={len(cr.meta_codebook['edge_matrix'])}")

    # -----------------------------------------------------------------
    # 4) Build/Load DPO preference dataset on seed Q-A pairs
    # -----------------------------------------------------------------
    print("» Building preference pairs for DPO …")
    if saved_examples_name.exists():
        pref_ds = load_pref_examples(str(saved_examples_name))
        print(f"loaded {len(pref_ds)} cached preference examples")
    else:
        cr.record_labeled_q_and_a(train_questions, train_answers)
        pref_ds = make_preference_dataset_2head(
            cr,
            questions=seed_questions,
            gold_answers=gold_lookup,
            per_q_samples=6,
            feature_dim=1024,
            reward_fn=reward_func,
            seed=0,
            isolate_state=True,
            ANSWERS_CHOICES=ANSWERS_CHOICES,
            THINKINGS_CHOICES=THINKINGS_CHOICES,
            FACTS_CHOICES=FACTS_CHOICES,
        )
        ensure_parent(saved_examples_name)
        save_pref_examples(str(saved_examples_name), pref_ds)

    # Clear stored QA (as in your code)
    cr.meta_codebook["questions_lst"] = []
    cr.meta_codebook["answers_lst"] = []

    policy, _ = train_dpo_2head(pref_ds, input_dim=1024)

    def dump_results(questions: List[str], out_path: Path):
        rows = []
        run_metrics = []
        answers_choices = []
        thinkings_choices = []
        facts_choices = []
        total_q_left = len(questions)
        finished_q = 0

        with tqdm(
            questions,
            desc="Processing questions",
            total=len(questions),
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.0,
            delay=0,
        ) as pbar:
            for q in pbar:
                start_idx = len(cr.llm.metrics_runs)
                pred, _meta = answer_with_auto_strategy(
                    cr,
                    policy,
                    q,
                    reward_fn=default_reward,
                    gold_answer=gold_lookup[q],
                    greedy=True,
                )

                gen_metrics = (cr.llm.metrics_runs[start_idx:] or [{}])[-1] or {}
                if isinstance(gen_metrics, dict) and q in gen_metrics and isinstance(gen_metrics[q], dict):
                    gen_metrics = gen_metrics[q]

                _ALLOWED_KEYS = {
                    "input_tokens", "output_tokens", "total_tokens",
                    "latency_sec", "gen_latency_sec", "retrieval_latency_sec",
                    "prompt_chars", "throughput_tok_per_s", "prompt_tok_per_s",
                    "device", "dtype", "model_name",
                    "timestamp_start", "timestamp_end",
                    "attempt", "question_chars", "answer_raw_chars", "answer_raw_tokens",
                    "prompt_to_output_char_ratio", "retrieved_count",
                    "peak_vram_MiB", "total_latency_sec",
                }
                gen_metrics = {k: v for k, v in gen_metrics.items() if k in _ALLOWED_KEYS}
                run_metrics.append({"question": q, **gen_metrics})

                # Update progress
                if "latency_sec" in gen_metrics or "gen_latency_sec" in gen_metrics:
                    pbar.set_postfix({
                        "lat": f"{gen_metrics.get('latency_sec', 0):.2f}s",
                        "gen": f"{gen_metrics.get('gen_latency_sec', 0):.2f}s",
                        "ret": f"{gen_metrics.get('retrieval_latency_sec', 0):.2f}s",
                    })

                row = row_lookup[q]
                rows.append({
                    "id":               row["id"],
                    "question":         q,
                    "source":           row.get("source"),
                    "context":          _meta.get("fact_context"),
                    "evidence":         row.get("evidence"),
                    "question_type":    row.get("question_type"),
                    "generated_answer": pred,
                    "ground_truth":     row.get("answer"),
                    "answers_choice":   _meta.get("answers_choice"),
                    "thinkings_choice": _meta.get("thinkings_choice"),
                    "facts_choice":     _meta.get("facts_choice"),
                    "meta_codebook_json_bytes": _meta.get("meta_codebook_json_bytes"),
                    "meta_codebook_json_MB":    _meta.get("meta_codebook_json_MB"),
                })

                answers_choices.append(_meta.get("answers_choice"))
                thinkings_choices.append(_meta.get("thinkings_choice"))
                facts_choices.append(_meta.get("facts_choice"))

                finished_q += 1
                total_q_left -= 1
                print(f"{finished_q} finished")
                print(f"{total_q_left} q left")

        ensure_parent(out_path)
        # merge metrics + rows
        metrics_by_q = {m["question"]: m for m in run_metrics}
        merged_results = []
        for row in rows:
            q = row["question"]
            merged = dict(metrics_by_q.get(q, {}))
            merged.update(row)
            merged_results.append(merged)

        out_path.write_text(json.dumps(merged_results, indent=2), encoding="utf-8")
        print(f"✓ wrote {len(merged_results)} merged rows → {out_path}")
        return merged_results, answers_choices, thinkings_choices, facts_choices

    print("» Answering evaluation questions …")
    _ = dump_results(all_questions, out_path=final_json_path)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoPrunedRetriever pipeline with a config file.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--llm_api", default=None, help="Override LLM API key (else config or OPENAI_API_KEY)")
    parser.add_argument("--chunking_api", default=None, help="Override Chunking API key (else config or CHUNKING_API)")
    args = parser.parse_args()

    cfg_path = args.config
    cfg = _load_config(args.config)

    # --- read knobs first ---
    dataset      = cfg.get("dataset", "tv")
    chunking_use = cfg.get("chunking_use", "rebel")
    subchunk_mode = cfg.get("subchunk_mode","chars")

    # -------- Inputs (config-relative + HF fallback) --------
    REPO_ID = cfg.get("repo_id", None)  # allow null => local-only
    CORPUS_SPEC = cfg.get("corpus_file", "GraphRAG-Benchmark/Datasets/Corpus/medical.json")
    QUEST_SPEC = cfg.get("quest_file", "Datasets/Questions/medical_questions.json")

    CORPUS_FILE = str(resolve_path_or_hf(REPO_ID, CORPUS_SPEC, cfg_path=cfg_path)[0])
    QUEST_FILE = str(resolve_path_or_hf(REPO_ID, QUEST_SPEC, cfg_path=cfg_path)[0])

    # -------- Hyper-params / knobs --------
    SEED_N = int(cfg.get("seed_n", 20))
    TEST_N = int(cfg.get("test_n", 2042))

    top_m = int(cfg.get("top_m", 20))
    top_k = int(cfg.get("top_k", top_m * 10))
    
    combine_ent_sim = float(cfg.get("combine_ent_sim", 0.93))
    q_combine_sim = float(cfg.get("q_combine_sim", 0.93))
    aft_combine_sim = float(cfg.get("aft_combine_sim", 0.93))
    semantic_overlap_sim = float(cfg.get("semantic_overlap_sim", 0.93))

    # -------- Outputs (resolved relative to config; dirs created later) --------
    # --- format templated output paths from YAML ---
    def _fmt(s: str) -> str:
        return s.format(dataset=dataset, chunking_use=chunking_use)

    ini_meta_json_raw       = _fmt(cfg.get("ini_meta_json", "meta_codebook.json"))
    saved_examples_name_raw = _fmt(cfg.get("saved_examples_name", "pref_examples.json"))
    final_json_path_raw     = _fmt(cfg.get("final_json_path", "results/out.json"))

    # --- resolve them relative to the CONFIG file folder ---
    ini_meta_json       = _resolve_rel_to_config(cfg_path, ini_meta_json_raw)
    saved_examples_name = _resolve_rel_to_config(cfg_path, saved_examples_name_raw)
    final_json_path     = _resolve_rel_to_config(cfg_path, final_json_path_raw)
    # -------- Modes / options --------
    chunking_use = cfg.get("chunking_use", "rebel")

    # -------- Rewards --------
    reward_func_name = cfg.get("reward_func", "reward_sbert_inclusive")
    reward_func = getattr(reward_func_dpo, reward_func_name)

    reward_func_mode = cfg.get("reward_func_mode", "non_llm")

    # -------- Keys: CLI > config > ENV --------
    llm_api_cfg = cfg.get("llm_api", None)
    llm_api = args.llm_api or _maybe_from_env(llm_api_cfg, "OPENAI_API_KEY")

    chunking_api_cfg = cfg.get("chunking_api", None)
    chunking_api = args.chunking_api or _maybe_from_env(chunking_api_cfg, "CHUNKING_API")

    # -------- Run --------
    compress_rag_workflow(
        REPO_ID=REPO_ID,
        CORPUS_FILE=CORPUS_FILE,
        QUEST_FILE=QUEST_FILE,
        SEED_N=SEED_N,
        TEST_N=TEST_N,
        top_m=top_m,
        top_k=top_k,
        combine_ent_sim=combine_ent_sim,
        q_combine_sim=q_combine_sim,
        aft_combine_sim=aft_combine_sim,
        semantic_overlap_sim=semantic_overlap_sim,
        ini_meta_json=ini_meta_json,
        saved_examples_name=saved_examples_name,
        reward_func=reward_func,
        reward_func_mode=reward_func_mode,
        final_json_path=final_json_path,
        chunking_use=chunking_use,
        chunking_api=chunking_api,   # ← picks up from CLI or config or ENV: CHUNKING_API
        llm_api=llm_api,             # ← picks up from CLI or config or ENV: OPENAI_API_KEY
        cfg_path=cfg_path,
        subchunk_mode = subchunk_mode
    )
