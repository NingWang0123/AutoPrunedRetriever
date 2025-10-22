import asyncio
import os, json, re, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import pathlib
import numpy   as np
import torch
import pandas as pd
from tqdm import tqdm
import copy
from langchain_community.embeddings import HuggingFaceEmbeddings

from AutoPrunedRetriever_advanced_v3 import (
    ExactGraphRag_rl, merging_codebook
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

# Simple replacement for compute_answer_correctness
def compute_answer_correctness(pred, gold, embeddings=None):
    """Simple answer correctness check"""
    return reward_func_dpo.reward_sbert_inclusive(pred, gold)
from functools import partial
from sentence_transformers import SentenceTransformer
import sys

# ---------------------------------------------------------------------
# 0) Paths / constants for 2WikiMultihop
# ---------------------------------------------------------------------

WIKI2_ROOT = Path("2wikimultihop-main")
CORPUS_FILE_2WIKI = WIKI2_ROOT / "para_with_hyperlink.jsonl"
# You need to download the dataset from: https://www.dropbox.com/s/npidmtadreo6df2/data.zip
# and extract train.json, dev.json, test.json to 2wikimultihop-main/
TRAIN_FILE_2WIKI = WIKI2_ROOT / "data/train.json"
DEV_FILE_2WIKI = WIKI2_ROOT / "data/dev.json"
TEST_FILE_2WIKI = WIKI2_ROOT / "data/test.json"

SEED_N       = 1    # first 30 rows → bootstrap + DPO train, numbers must be divided by 2. (n%2=0)
TEST_N       = 1     # next 20 rows  → evaluation
TOPK_CTX     = 5

# ---------------------------------------------------------------------
# Helper functions for 2WikiMultihop data loading
# ---------------------------------------------------------------------

def load_2wikimultihop_corpus(corpus_jsonl_path: Path) -> List[Dict]:
    """
    Load 2WikiMultihop corpus from para_with_hyperlink.jsonl.
    Each line is a JSON object with: id, title, sentences, mentions.
    Returns a list of dictionaries with 'title' and 'text' (joined sentences).
    """
    corpus_docs = []
    with open(corpus_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                # Combine sentences into a single text
                text = " ".join(doc.get("sentences", []))
                corpus_docs.append({
                    "id": doc.get("id"),
                    "title": doc.get("title"),
                    "text": text
                })
    return corpus_docs


def load_2wikimultihop_questions(questions_json_path: Path) -> Tuple[Dict, Dict]:
    """
    Load 2WikiMultihop questions from train.json or dev.json.
    Returns:
        - row_lookup: dict mapping question -> full row
        - gold_lookup: dict mapping question -> answer
    """
    with open(questions_json_path, 'r', encoding='utf-8') as f:
        qrows = json.load(f)
    
    row_lookup = {r["question"].strip(): r for r in qrows}
    gold_lookup = {q: r["answer"] for q, r in row_lookup.items()}
    
    return row_lookup, gold_lookup


def convert_2wikimultihop_corpus_to_json(corpus_docs: List[Dict], output_path: str):
    """
    Convert 2WikiMultihop corpus to a JSON format compatible with load_and_merge_facts.
    The format should be a list of documents with 'context' field (required by preload_context_json).
    """
    # Format: list of dicts with 'context' field (required by AutoPrunedRetriever_advanced_v3.py)
    formatted_docs = []
    for doc in corpus_docs:
        formatted_docs.append({
            "id": doc["id"],
            "title": doc["title"],
            "context": doc["text"]  # Use 'context' instead of 'text' - required by preload_context_json
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_docs, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Converted {len(formatted_docs)} documents to {output_path}")


# ---------------------------------------------------------------------
# Main workflow function adapted for 2WikiMultihop
# ---------------------------------------------------------------------

def compress_rag_workflow_2wiki(CORPUS_FILE, QUEST_FILE, SEED_N, TEST_N,
                          top_m, top_k, combine_ent_sim, q_combine_sim, aft_combine_sim, semantic_overlap_sim,
                          ini_meta_json=Path("meta_codebook_2wiki.json"),
                          saved_examples_name="pref_examples_2wiki.json",
                          reward_func=None, reward_func_mode='non_llm',
                          final_json_path="results/compressrag_2wikimultihop_test.json"):
    
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
        api_key="",  
        # base_url="https://api.openai.com/v1",
    )

    if ini_meta_json:
        pre_loaded_meta = False

        if ini_meta_json.is_file():
            try:
                with ini_meta_json.open("r", encoding="utf-8") as f:
                    ini = json.load(f)
                    pre_loaded_meta = True
                    print(f"✓ Loaded existing meta_codebook from {ini_meta_json}")
            except json.JSONDecodeError:
                print(f"[warn] {ini_meta_json} is not valid JSON; starting fresh.")
                ini = {}
                pre_loaded_meta = False
        else:
            ini = {}
    else:
        ini = {}

    cr = ExactGraphRag_rl(
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
        use_word = True
    )

    # ---------------------------------------------------------------------
    # 2) Load 2WikiMultihop Q-A data
    # ---------------------------------------------------------------------
    print("» Loading 2WikiMultihop questions / answers …")
    
    if not Path(QUEST_FILE).exists():
        raise FileNotFoundError(
            f"Question file not found: {QUEST_FILE}\n"
            f"Please download from: https://www.dropbox.com/s/npidmtadreo6df2/data.zip\n"
            f"and extract train.json or dev.json to {WIKI2_ROOT}/"
        )
    
    row_lookup, gold_lookup = load_2wikimultihop_questions(Path(QUEST_FILE))

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
    # 3) Pre-load 2WikiMultihop corpus as facts into CR
    # ---------------------------------------------------------------------
    if not pre_loaded_meta:
        print("» Loading 2WikiMultihop corpus …")
        
        if not Path(CORPUS_FILE).exists():
            raise FileNotFoundError(
                f"Corpus file not found: {CORPUS_FILE}\n"
                f"Make sure para_with_hyperlink.jsonl exists in {WIKI2_ROOT}/"
            )
        
        # Load corpus and convert to compatible format
        corpus_docs = load_2wikimultihop_corpus(Path(CORPUS_FILE))
        print(f"✓ Loaded {len(corpus_docs)} documents from 2WikiMultihop corpus")
        
        # Convert to temporary JSON file for load_and_merge_facts
        temp_corpus_json = WIKI2_ROOT / "corpus_converted.json"
        convert_2wikimultihop_corpus_to_json(corpus_docs, str(temp_corpus_json))
        
        # Load facts using the converted corpus
        cr.load_and_merge_facts(
            str(temp_corpus_json),
            chunk_tokens=1200,
            overlap_tokens=100,
            sub_chunk_chars=200,
            sub_chunk_overlap=50,
            tokenizer_name="gpt-4o-mini",
            subchunk_batch=1000
        )
        cr._facts_preloaded = True

        print(cr.meta_codebook)
        print('=' * 60)
        print(f"len(cr.meta_codebook['facts_lst']): {len(cr.meta_codebook['facts_lst'])}")
        print('=' * 60)

        # Save meta_codebook
        def make_json_safe(obj):
            """Recursively convert numpy arrays into lists."""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            return obj

        with open(ini_meta_json, "w") as f:
            json.dump(make_json_safe(cr.meta_codebook), f, indent=2)
        print(f"✓ Saved meta_codebook to {ini_meta_json}")

        print('=' * 60)
        print(f"after changed len(cr.meta_codebook['facts_lst']): {len(cr.meta_codebook['facts_lst'])}")
        print('=' * 60)

        print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
              f"|R|={len(cr.meta_codebook['r'])} "
              f"|edges|={len(cr.meta_codebook['edge_matrix'])}")

    # ---------------------------------------------------------------------
    # 4) Build DPO preference dataset on seed Q-A pairs
    # ---------------------------------------------------------------------
    print("» Building preference pairs for DPO …")

    if os.path.exists(saved_examples_name):
        pref_ds = load_pref_examples(saved_examples_name)
        print(f"loaded {len(pref_ds)} cached preference examples")
    else:
        cr.record_labeled_q_and_a(train_questions, train_answers)

        if reward_func_mode == 'llm':
            async def build_or_load_pref_ds() -> list:
                embedding_for_reward = HuggingFaceBgeEmbeddings(
                    model_name="BAAI/bge-large-en-v1.5"
                )
                BASE_URL = "https://api.deepseek.com/v1"
                API_KEY = pathlib.Path("deepseek_key.txt").read_text().strip()

                llm = ChatOpenAI(
                    model="deepseek-chat",
                    base_url=BASE_URL,
                    api_key=API_KEY,
                    temperature=0.0,
                    max_retries=3,
                    timeout=30,
                )

                pref_ds = await make_preference_dataset_2head_using_llm(
                    cr=cr,
                    questions=seed_questions,
                    gold_answers=gold_lookup,
                    per_q_samples=6,
                    reward_fn=compute_answer_correctness,   
                    seed=42,
                    llm=llm,
                    embeddings=embedding_for_reward,
                    ANSWERS_CHOICES=ANSWERS_CHOICES,
                    THINKINGS_CHOICES=THINKINGS_CHOICES,
                    FACTS_CHOICES=FACTS_CHOICES,
                    isolate_state=True,
                    feature_dim=1024
                )
                print(f"   generated {len(pref_ds)} preference examples")
                save_pref_examples(saved_examples_name, pref_ds)
                return pref_ds
            
            pref_ds = asyncio.run(build_or_load_pref_ds())
        else:
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
                FACTS_CHOICES=FACTS_CHOICES
            )
            save_pref_examples(saved_examples_name, pref_ds)
            print(f"✓ Generated and saved {len(pref_ds)} preference examples")

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
                    reward_fn=default_reward,
                    gold_answer=gold_lookup[q],
                    greedy=True
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

                # --- build row ---
                row = row_lookup[q]

                # Adapt to 2WikiMultihop format
                rows.append({
                    "_id": row.get("_id", ""),
                    "question": q,
                    "type": row.get("type", ""),  # 2WikiMultihop has 'type' instead of 'question_type'
                    "context": _meta['fact_context'],
                    "supporting_facts": row.get("supporting_facts", []),
                    "evidences": row.get("evidences", []),
                    "generated_answer": pred,
                    "ground_truth": row["answer"],
                    "answers_choice": _meta['answers_choice'],
                    "thinkings_choice": _meta['thinkings_choice'],
                    "facts_choice": _meta['facts_choice'],
                    "meta_codebook_json_bytes": _meta['meta_codebook_json_bytes'],
                    "meta_codebook_json_MB": _meta['meta_codebook_json_MB'],
                })

                answers_choices.append(_meta['answers_choice'])
                thinkings_choices.append(_meta['thinkings_choice'])
                facts_choices.append(_meta['facts_choice'])

                finished_q += 1
                total_q_left -= 1

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
    generated_rows, answers_choices, thinkings_choices, facts_choices = dump_results(
        all_questions, out_path=final_json_path
    )

    return generated_rows, answers_choices, thinkings_choices, facts_choices


if __name__ == "__main__":
    # Keep all parameters aligned with the original pipeline
    aft_combine_sim = 0.93
    top_m = 20

    reward_func = reward_func_dpo.reward_sbert_inclusive

    SEED_N = 20    # change to 20 for training
    TEST_N = 2042  # change to 980 for rest (adjust based on dataset size)

    # Check if dataset files exist, provide helpful message if not
    if not CORPUS_FILE_2WIKI.exists():
        print(f"\n{'='*60}")
        print("⚠️  2WikiMultihop corpus file not found!")
        print(f"Expected location: {CORPUS_FILE_2WIKI}")
        print("\nThe corpus file para_with_hyperlink.jsonl should already be present.")
        print(f"{'='*60}\n")
    
    if not TEST_FILE_2WIKI.exists():
        print(f"\n{'='*60}")
        print("⚠️  2WikiMultihop test file not found!")
        print(f"Expected location: {TEST_FILE_2WIKI}")
        print("\nPlease download the dataset from:")
        print("https://www.dropbox.com/s/npidmtadreo6df2/data.zip")
        print(f"and extract test.json to {WIKI2_ROOT}/data/")
        print(f"{'='*60}\n")
        exit(1)
    
    # Use test.json for experiments
    QUEST_FILE = TEST_FILE_2WIKI
    
    print(f"\n{'='*60}")
    print(f"Running 2WikiMultihop pipeline with:")
    print(f"  Corpus: {CORPUS_FILE_2WIKI}")
    print(f"  Questions: {QUEST_FILE}")
    print(f"  SEED_N: {SEED_N}")
    print(f"  TEST_N: {TEST_N}")
    print(f"  top_m: {top_m}")
    print(f"  aft_combine_sim: {aft_combine_sim}")
    print(f"{'='*60}\n")

    compress_rag_workflow_2wiki(
        CORPUS_FILE=str(CORPUS_FILE_2WIKI),
        QUEST_FILE=str(QUEST_FILE),
        SEED_N=SEED_N,
        TEST_N=TEST_N,
        top_m=top_m,
        top_k=top_m*10,
        combine_ent_sim=aft_combine_sim,
        q_combine_sim=aft_combine_sim,
        aft_combine_sim=aft_combine_sim,
        semantic_overlap_sim=0.93,
        ini_meta_json=Path("meta_codebook_2wiki.json"),
        saved_examples_name=f"pref_examples_2wiki_openai_v1.json",
        reward_func=reward_func,
        reward_func_mode='non_llm',
        final_json_path=f"results/compressrag_2wikimultihop_openai_v1.json"
    )

# python pipeline_for_2wikimultihop.py
