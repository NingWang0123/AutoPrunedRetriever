#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graph/Text CompressRAG-RL evaluation on GraphRAG-Benchmark (Medical)

• Builds a CompressRag_rl instance
• Learns a 2-head DPO policy (answers / thinkings) on the first 30 Q-A
• Uses a LinUCB scheduler to choose entity-combination cadence online
• Seeds history with the same 30 Q-A, then evaluates the next 20
• Dumps benchmark-compatible JSON + optional cost reports
"""

import os, json, re, random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy   as np
import torch
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from CompressRag_rl_v1 import (
    CompressRag_rl, WordAvgEmbeddings, merging_codebook
)
from dpo_compressrag import (            # same folder as shown
    make_preference_dataset_2head, train_dpo_2head,
    default_reward, featurize_state, CombineScheduler,
    COMBINE_ARMS, answer_with_auto_strategy, Phi4MiniReasoningLLM,
)

# ---------------------------------------------------------------------
# 0) Paths / constants
# ---------------------------------------------------------------------
REPO_ID      = "GraphRAG-Bench/GraphRAG-Bench"
CORPUS_FILE  = "Datasets/Corpus/medical.json"
QUEST_FILE   = "Datasets/Questions/medical_questions.json"

SEED_N       = 30     # first 30 rows → bootstrap + DPO train
TEST_N       = 20     # next 20 rows  → evaluation
TOPK_CTX     = 5

# ---------------------------------------------------------------------
# 1) Initialise embeddings & LLM
# ---------------------------------------------------------------------
print("» Initialising embeddings & LLM …")
word_emb = WordAvgEmbeddings(
    model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model"
)
sent_emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
phi_llm  = Phi4MiniReasoningLLM(
    include_thinkings=True,
    model_name="microsoft/Phi-4-mini-reasoning",
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9,
)

cr = CompressRag_rl(
    ini_meta_codebook = {},
    sentence_emb      = sent_emb,
    word_emb          = word_emb,
    llm               = phi_llm,
    combine_ents_rounds = 1,        # LinUCB 会改写
    thinkings_choice    = 'not_include',
    answers_choice      = 'overlap',
)

# ---------------------------------------------------------------------
# 2) Load benchmark Q-A-E data
# ---------------------------------------------------------------------
print("» Loading benchmark questions / answers / evidence …")
q_fp = hf_hub_download(REPO_ID, QUEST_FILE, repo_type="dataset")
qrows = json.load(open(q_fp, encoding="utf-8"))

row_lookup  = {r["question"].strip(): r for r in qrows}
gold_lookup = {q: r["answer"]          for q, r in row_lookup.items()}

all_questions  = list(row_lookup.keys())
seed_questions = all_questions[:SEED_N]
test_questions = all_questions[SEED_N : SEED_N+TEST_N]

# ---------------------------------------------------------------------
# 3) Pre-load corpus as facts into CR
# ---------------------------------------------------------------------
facts_json_paths = [hf_hub_download(REPO_ID, CORPUS_FILE, repo_type="dataset")]
cr.set_facts_sources(facts_json_paths)

facts_cb = cr.load_and_merge_facts(facts_json_paths, chunk_chars=100, overlap=30)
cr._facts_preloaded = True
cr.top_m = 3          # sentence-embedding rerank top-m

cr.meta_codebook = merging_codebook(
    cr.meta_codebook, facts_cb,
    type='facts', word_emb=cr.word_emb, use_thinkings=False
)

print(f"[DEBUG] after facts-merge: |E|={len(cr.meta_codebook['e'])} "
      f"|R|={len(cr.meta_codebook['r'])} "
      f"|edges|={len(cr.meta_codebook['edge_matrix'])}")

# # ---------------------------------------------------------------------
# # 4) Build DPO preference dataset on seed Q-A pairs
# # ---------------------------------------------------------------------
# print("» Building preference pairs for DPO …")
# pref_ds = make_preference_dataset_2head(
#     cr            = cr,
#     questions     = seed_questions,
#     gold_answers  = gold_lookup,
#     per_q_samples = 6,
#     reward_fn     = default_reward,
#     seed          = 42,
# )
# print(f"   generated {len(pref_ds)} preference examples")

# policy, _ = train_dpo_2head(pref_ds, input_dim=384)

# # ---------------------------------------------------------------------
# # 5) LinUCB scheduler (entity-combine cadence)
# # ---------------------------------------------------------------------
# state_dim  = featurize_state(cr).shape[0]   # typically 4
# scheduler  = CombineScheduler(d=state_dim, arms=COMBINE_ARMS,
#                               alpha=1.0, epsilon=0.05)

# # ---------------------------------------------------------------------
# # 6) Seed history with first 30 Q-A (store answers / thinkings)
# # ---------------------------------------------------------------------
# print("» Seeding history with first 30 questions …")
# for q in seed_questions:
#     answer_with_auto_strategy(
#         cr, policy, scheduler, q,
#         reward_fn       = default_reward,
#         gold_answer     = gold_lookup[q],
#         facts_json_path = facts_json_paths,
#         chunk_chars     = 200,
#         overlap         = 30,
#         greedy          = True
#     )

# # ---------------------------------------------------------------------
# # 7) Helper: capture CR’s last retrieved context
# # ---------------------------------------------------------------------
# def _collect_ctx(cr, k: int = TOPK_CTX) -> List[str]:
#     ctx = getattr(cr, "_last_ctx", [])[:k]
#     return [re.sub(r"\s+", " ", c.strip()) for c in ctx]

# # ---------------------------------------------------------------------
# # 8) Evaluate next 20 questions & dump JSON
# # ---------------------------------------------------------------------
# def dump_results(questions: List[str], out_path: str):
#     rows = []
#     for q in questions:
#         pred, _meta = answer_with_auto_strategy(
#             cr, policy, scheduler, q,
#             reward_fn       = default_reward,
#             gold_answer     = gold_lookup[q],
#             facts_json_path = facts_json_paths,
#             chunk_chars     = 400,
#             overlap         = 80,
#             greedy          = True
#         )
#         row = row_lookup[q]
#         rows.append({
#             "id":              row["id"],
#             "question":        q,
#             "source":          row["source"],
#             "context":         _collect_ctx(cr),
#             "evidence":        row["evidence"],
#             "question_type":   row["question_type"],
#             "generated_answer": pred,
#             "ground_truth":    row["answer"],
#         })

#     Path(out_path).parent.mkdir(parents=True, exist_ok=True)
#     json.dump(rows, open(out_path, "w"), indent=2)
#     print(f"   wrote {len(rows)} rows → {out_path}")

# print("» Answering 20 evaluation questions …")
# dump_results(test_questions, "results/compressrag_medical.json")

# # ---------------------------------------------------------------------
# # 9) Optional: basic cost summary (tokens / latency)
# # ---------------------------------------------------------------------
# # NOTE: CompressRag_rl_v1 里只有 _record_metric() 时才会有数据
# try:
#     txt_stats  = cr.report_cost(kind="text")
#     gph_stats  = cr.report_cost(kind="graph")
#     os.makedirs("results", exist_ok=True)
#     json.dump(
#         {"text": txt_stats, "graph": gph_stats},
#         open("results/cost_summary.json", "w"), indent=2
#     )
# except Exception as e:
#     print("[WARN] cost report failed:", e)

# # ---------------------------------------------------------------------
# # 10) (Optional) save FAISS indexes – enable if you later add those APIs
# # ---------------------------------------------------------------------
# # TODO: 如果你在 CompressRag_rl_v1 里实现了 rebuild_vector_stores() / graph_db /
# # text_db，可取消下面 3 行注释做 index size 统计：
# # ---------------------------------------------------------------------
# # if hasattr(cr, "rebuild_vector_stores"):
# #     cr.rebuild_vector_stores()
# #     cr.save_index_and_report_size(db="text",  out_dir="faiss_text_idx")
# #     cr.save_index_and_report_size(db="graph", out_dir="faiss_graph_idx")

# print("\nDONE – ready to evaluate with generation_eval.py / retrieval_eval.py")
