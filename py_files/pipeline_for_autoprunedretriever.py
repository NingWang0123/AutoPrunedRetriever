import asyncio
import os, json, re, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pathlib
import numpy   as np
import pandas as pd
from tqdm import tqdm
import copy
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from AutoPrunedRetriever_advanced_final import (
    AutoPrunedRetriver
)

from dpo_exactgraphrag import (    
    make_preference_dataset_2head, train_dpo_2head,make_preference_dataset_2head_using_llm,
    default_reward, answer_with_auto_strategy,save_pref_examples,load_pref_examples,ANSWERS_CHOICES,THINKINGS_CHOICES,FACTS_CHOICES
)

from llm_api import OpenAILLM ,Word2VecEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import ChatOpenAI
# from evaluation_func_graphrag import compute_answer_correctness
import reward_func_dpo
from functools import partial
import sys
from glob import glob

# ---------------------------------------------------------------------
# 0) Paths / constants
# ---------------------------------------------------------------------
REPO_ID      = "GraphRAG-Bench/GraphRAG-Bench"
CORPUS_FILE  = "GraphRAG-Benchmark/Datasets/Corpus/medical.json"
QUEST_FILE   = "Datasets/Questions/medical_questions.json"

# ---------------------------------------------------------------------
# 1) Initialise embeddings & LLM
# ---------------------------------------------------------------------

def _is_existing_file(p: str) -> bool:
    try:
        return Path(p).expanduser().resolve().is_file()
    except Exception:
        return False

def _resolve_rel_to_config(cfg_path: Optional[str], p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    P = Path(p)
    if P.is_absolute():
        return P
    # resolve relative to the config file’s directory if provided; else CWD
    base = Path(cfg_path).parent if cfg_path else Path.cwd()
    return (base / P).resolve()

def resolve_path_or_hf(repo_id: Optional[str], file_or_glob: str, cfg_path: Optional[str]=None) -> List[Path]:
    """
    Returns one or more concrete local Paths.
    - If file_or_glob exists locally (file or glob), return matching paths.
    - Else if repo_id is set, try to download via hf_hub_download.
    - Else raise.
    """
    # 1) try local (exact or glob)
    local_candidate = _resolve_rel_to_config(cfg_path, file_or_glob)
    matches = []
    if local_candidate:
        if local_candidate.is_file():
            matches = [local_candidate]
        else:
            # glob against string; keep relative-to-config semantics
            mg = glob(str(local_candidate))
            matches = [Path(m).resolve() for m in mg if Path(m).is_file()]
    if matches:
        return matches

    # 2) try HF hub (only for a SINGLE file path, not globs)
    if repo_id and ("*" not in file_or_glob and "?" not in file_or_glob and "[" not in file_or_glob):
        try:
            p = hf_hub_download(repo_id, file_or_glob, repo_type="dataset")
            return [Path(p).resolve()]
        except Exception as e:
            raise FileNotFoundError(
                f"Neither local path nor HF worked for '{file_or_glob}'. "
                f"Tried repo_id='{repo_id}'. Error: {e}"
            )

    # 3) give up
    raise FileNotFoundError(
        f"No local file(s) matched '{file_or_glob}' and no HF repo_id provided."
    )

def load_questions_any(repo_id: Optional[str], quest_spec: str, cfg_path: Optional[str]=None) -> List[dict]:
    """
    Loads questions from JSON or JSONL—either local or HF.
    """
    paths = resolve_path_or_hf(repo_id, quest_spec, cfg_path)
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one questions file, got: {paths}")
    p = paths[0]
    if p.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        return json.loads(p.read_text(encoding="utf-8"))


def compress_rag_workflow(REPO_ID,CORPUS_FILE,QUEST_FILE,SEED_N,TEST_N,
                          top_m,top_k,combine_ent_sim,q_combine_sim,aft_combine_sim,semantic_overlap_sim,  # all the params can be optimized
                          ini_meta_json = Path("meta_codebook.json") ,saved_examples_name = "sbert_pref_examples_medical.json",
                          reward_func = None,reward_func_mode = 'non_llm',final_json_path = "results/compressrag_medical_data_test.json",chunking_use='rebel',chunking_api = None,llm_api = None,
                          cfg_path: Optional[str]=None):
    print("» Initialising embeddings & LLM …")
    word_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    sent_emb = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5"
    )
    api_llm = OpenAILLM(  
        include_thinkings=True,                 
        model_name="gpt-4o-mini",  
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        api_key=llm_api,  
    )


    if ini_meta_json:
        pre_loaded_meta = False

        if ini_meta_json.is_file():
            try:
                with ini_meta_json.open("r", encoding="utf-8") as f:
                    ini = json.load(f)
                    pre_loaded_meta = True
            except json.JSONDecodeError:
                print("[warn] meta_codebook.json is not valid JSON; starting fresh.")
                ini = {}
                pre_loaded_meta = False
        else:
            ini = {}
    else:
        ini = {}


    cr = AutoPrunedRetriver(
        ini_meta_codebook = ini,
        sentence_emb      = sent_emb,
        word_emb          = word_emb,
        llm               = api_llm,
        thinkings_choice    = 'not_include',
        answers_choice      = 'unique',
        facts_choice = 'include_all',
        top_m = top_m,
        top_k = top_k,
        combine_ent_sim = combine_ent_sim,
        q_combine_sim = q_combine_sim,
        aft_combine_sim = aft_combine_sim,
        semantic_overlap_sim = semantic_overlap_sim,
        use_word = True,
        chunking_use = chunking_use,
        chunking_api = chunking_api,
    )

    # ---------------------------------------------------------------------
    # 2) Load benchmark Q-A-E data
    # ---------------------------------------------------------------------
    print("» Loading benchmark questions / answers / evidence …")
    qrows = load_questions_any(REPO_ID, QUEST_FILE, cfg_path=cfg_path)
    row_lookup  = {r["question"].strip(): r for r in qrows}
    gold_lookup = {q: r["answer"] for q, r in row_lookup.items()}

    # q_fp = hf_hub_download(REPO_ID, QUEST_FILE, repo_type="dataset")
    # qrows = json.load(open(q_fp, encoding="utf-8"))
    # row_lookup  = {r["question"].strip(): r for r in qrows}
    # gold_lookup = {q: r["answer"]        for q, r in row_lookup.items()}

    all_questions  = list(row_lookup.keys())
    all_seed_questions = all_questions[:SEED_N]
    midpoint = len(all_seed_questions) // 2
    # for labeling only
    train_questions = all_seed_questions[:midpoint]
    seed_questions   = all_seed_questions[midpoint:]

    train_answers = []

    for q in train_questions:
        train_answers.append(gold_lookup.get(q))

    test_questions = all_questions[SEED_N : SEED_N+TEST_N]

    # ---------------------------------------------------------------------
    # 3) Pre-load corpus as facts into CR
    # ---------------------------------------------------------------------
    # only load if we do not have ini_meta 
    if not pre_loaded_meta:
        # corpus file is the facts
        cr.load_and_merge_facts(
            CORPUS_FILE,
            chunk_tokens=1200,
            overlap_tokens=100,
            sub_chunk_chars=200,
            sub_chunk_overlap=50,
            tokenizer_name="gpt-4o-mini",
            subchunk_batch =1000
        )
        cr._facts_preloaded = True
        # cr.top_m = 5          # sentence-embedding rerank top-m

        print(cr.meta_codebook)

        print('===============================================================')
        print('===============================================================')
        print('===============================================================')

        print("len(cr.meta_codebook[facts_lst])",len(cr.meta_codebook["facts_lst"]))

        print('===============================================================')
        print('===============================================================')
        print('===============================================================')

        def make_json_safe(obj):
            """Recursively convert numpy arrays into lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            return obj

        with open(ini_meta_json.name, "w") as f:
            json.dump(make_json_safe(cr.meta_codebook), f, indent=2)

        print('===============================================================')
        print('===============================================================')
        print('===============================================================')

        print("after changed len(cr.meta_codebook[facts_lst])",len(cr.meta_codebook["facts_lst"]))

        print('===============================================================')
        print('===============================================================')
        print('===============================================================')


        print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
            f"|R|={len(cr.meta_codebook['r'])} "
            f"|edges|={len(cr.meta_codebook['edge_matrix'])}")
        

    # ---------------------------------------------------------------------
    # 4) Build DPO preference dataset on seed Q-A pairs
    # ---------------------------------------------------------------------
    # print(cr.meta_codebook)
    # use cr_training to train

    print("» Building preference pairs for DPO …")

    # # using llm one to replace the old one
    # # using answer_correctness from graph rag benchmark
    if os.path.exists(saved_examples_name):
        pref_ds = load_pref_examples(saved_examples_name)
        print(f"loaded {len(pref_ds)} cached preference examples")

    else:
        cr.record_labeled_q_and_a(train_questions, train_answers)



        pref_ds = make_preference_dataset_2head(
                cr,
                questions= seed_questions,
                gold_answers=gold_lookup,
                per_q_samples = 6,
                feature_dim = 1024,
                reward_fn = reward_func,
                seed = 0,
                isolate_state = True,
                ANSWERS_CHOICES = ANSWERS_CHOICES,
                THINKINGS_CHOICES = THINKINGS_CHOICES,
                FACTS_CHOICES = FACTS_CHOICES

            )
        

    # throw away the store info from qa
    cr.meta_codebook['questions_lst'] = []
    cr.meta_codebook['answers_lst'] = []

             


    policy, _ = train_dpo_2head(pref_ds, input_dim=1024)


    def dump_results(
        questions: List[str],
        out_path: str,
    ):
        rows = []
        run_metrics = []
        answers_choices = []
        thinkings_choices = []
        facts_choices = []
        total_q_left = len(questions)
        finished_q = 0

        with tqdm(questions,
          desc="Processing questions",
          total=len(questions),
          dynamic_ncols=True,
          mininterval=0.2,
          smoothing=0.0,
          delay=0) as pbar:
            for q in pbar:
                start_idx = len(cr.llm.metrics_runs)
                pred, _meta = answer_with_auto_strategy(
                    cr, policy, q,
                    reward_fn       = default_reward,
                    gold_answer     = gold_lookup[q],
                    greedy          = True
                )

                # --- get and sanitize gen_metrics ---
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

                # --- update progress bar with latest latencies ---
                if "latency_sec" in gen_metrics or "gen_latency_sec" in gen_metrics:
                    pbar.set_postfix({
                        "lat": f"{gen_metrics.get('latency_sec', 0):.2f}s",
                        "gen": f"{gen_metrics.get('gen_latency_sec', 0):.2f}s",
                        "ret": f"{gen_metrics.get('retrieval_latency_sec', 0):.2f}s"
                    })

                import json

                # --- build row ---
                row = row_lookup[q]

                rows.append({
                    "id":               row["id"],
                    "question":         q,
                    "source":           row["source"],
                    "context":          _meta['fact_context'],
                    "evidence":         row["evidence"],
                    "question_type":    row["question_type"],
                    "generated_answer": pred,
                    "ground_truth":     row["answer"],
                    "answers_choice":   _meta['answers_choice'],
                    "thinkings_choice": _meta['thinkings_choice'],
                    "facts_choice":     _meta['facts_choice'],
                    "meta_codebook_json_bytes": _meta['meta_codebook_json_bytes'],
                    "meta_codebook_json_MB": _meta['meta_codebook_json_MB'],
                })

                answers_choices.append(_meta['answers_choice'])
                thinkings_choices.append(_meta['thinkings_choice'])
                facts_choices.append(_meta['facts_choice'])


                finished_q+=1
                total_q_left-=1

                print(f'{finished_q} finished')
                print(f'{total_q_left} q left')

        # --- merge metrics + rows ---
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        metrics_by_q = {m["question"]: m for m in run_metrics}
        merged_results = []
        for row in rows:
            q = row["question"]
            merged = dict(metrics_by_q.get(q, {}))
            merged.update(row)
            merged_results.append(merged)

        with open(out_path, "w") as f:
            json.dump(merged_results, f, indent=2)
            print(f"✓ wrote {len(merged_results)} merged rows → {out_path}")

        return merged_results, answers_choices, thinkings_choices, facts_choices

    print("» Answering evaluation questions …")
    # still giving the all questions
    generated_rows,answers_choices,thinkings_choices,facts_choices = dump_results(all_questions, out_path= final_json_path)



import argparse, yaml

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith((".yml", ".yaml")):
            return yaml.safe_load(f)
        return json.load(f)

def _coerce_path(p):
    return Path(p) if isinstance(p, str) else p

def _maybe_from_env(v, env_key: Optional[str]):
    # if config leaves it blank, try environment var
    return v or (os.getenv(env_key) if env_key else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CompressRAG workflow with a config file.")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML/JSON config")
    parser.add_argument("--llm_api", default=None, help="Override LLM API key (else use config or OPENAI_API_KEY)")
    args = parser.parse_args()

    cfg_path = args.config
    cfg = _load_config(args.config)

    # --- required-ish top-levels ---
    REPO_ID     = cfg.get("repo_id", "GraphRAG-Bench/GraphRAG-Bench")
    CORPUS_FILE = cfg.get("corpus_file", "GraphRAG-Benchmark/Datasets/Corpus/medical.json")
    QUEST_FILE  = cfg.get("quest_file",  "Datasets/Questions/medical_questions.json")

    # --- hyper-params / knobs ---
    SEED_N  = int(cfg.get("seed_n", 20))
    TEST_N  = int(cfg.get("test_n", 2042))

    top_m                = int(cfg.get("top_m", 20))
    top_k                = int(cfg.get("top_k", top_m * 10))
    combine_ent_sim      = float(cfg.get("combine_ent_sim", 0.93))
    q_combine_sim        = float(cfg.get("q_combine_sim", 0.93))
    aft_combine_sim      = float(cfg.get("aft_combine_sim", 0.93))
    semantic_overlap_sim = float(cfg.get("semantic_overlap_sim", 0.93))

    # --- IO paths ---
    ini_meta_json       = _coerce_path(cfg.get("ini_meta_json", "meta_codebook.json"))
    saved_examples_name = cfg.get("saved_examples_name", "pref_examples_medical_exact_openai.json")
    final_json_path     = cfg.get("final_json_path", "results/compressrag_medical_data.json")

    # --- optional modules/modes ---
    chunking_use = cfg.get("chunking_use", "rebel")
    chunking_api = cfg.get("chunking_api", None)
    reward_func_mode = cfg.get("reward_func_mode", "non_llm")

    # --- rewards ---
    # you can switch here to other reward funcs you have
    reward_func = getattr(reward_func_dpo, cfg.get("reward_func", "reward_sbert_inclusive"))

    # --- LLM key precedence: CLI > config > ENV ---
    llm_api = args.llm_api or _maybe_from_env(cfg.get("llm_api", None), "OPENAI_API_KEY")

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
        chunking_api=chunking_api,
        llm_api=llm_api,
        cfg_path=cfg_path
    )