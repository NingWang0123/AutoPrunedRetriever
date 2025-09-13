"""
Graph/Text CompressRAG-RL evaluation on GraphRAG-Benchmark (Medical)

• Builds a CompressRag_rl instance (graph + text indexes)
• Learns a 2-head DPO policy (answers / thinkings) on the first 30 Q-A
• Uses a LinUCB scheduler to choose entity-combination cadence online
• Seeds history with the same 30 Q-A, then evaluates the next 20
• Writes benchmark-compatible JSON (answers + ground-truth + evidence)
  and optional cost reports
"""

import os, json, random, re
from pathlib import Path
from typing import List

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from CompressRag_rl_v1 import CompressRag_rl, WordAvgEmbeddings
from dpo_compressrag import (
    make_preference_dataset_2head, train_dpo_2head,
    default_reward, featurize_state, CombineScheduler,
    COMBINE_ARMS, answer_with_auto_strategy, Phi4MiniReasoningLLM,
)

# ---------------------------------------------------------------------
# 1) Optional cost-tracking mix-in  (prints <no data> if you never call
#    self._record_metric() inside CompressRag_rl.run_work_flow())
# ---------------------------------------------------------------------
class _CostMixin:
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._metrics = {"graph": [], "text": []}

    # ---------- cost bookkeeping ----------
    def _record_metric(self, kind: str, m: dict):
        self._metrics.setdefault(kind, []).append(m)

    def _avg(self, kind: str) -> dict:
        rows = self._metrics.get(kind, [])
        keys = ["input_tokens", "output_tokens", "total_tokens",
                "latency_sec", "retrieval_latency_sec", "gen_latency_sec",
                "retrieved_count", "peak_vram_MiB", "prompt_chars"]
        return {k: (sum(r.get(k, 0) for r in rows) / len(rows) if rows else 0.0) for k in keys}

    def report_cost(self, *, kind: str = "graph", avg: bool = True) -> dict:
        stats = self._avg(kind) if avg else (self._metrics.get(kind, [])[-1]
                                             if self._metrics.get(kind) else {})
        hdr = f"== Cost ({'avg' if avg else 'last'}) of {kind} RAG =="
        print(f"\n{hdr}" if stats else f"\n{hdr}\n  <no data>")
        for k, v in stats.items():
            if stats:
                s = f"{v:.2f}" if isinstance(v, float) and v != int(v) else f"{int(v)}"
                print(f"{k:>22} {s}")
        return stats

    # ---------- size helpers ----------
    @staticmethod
    def _dir_size_bytes(path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total

    @staticmethod
    def _bytes_to_human(num: int) -> str:
        for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
            if num < 1024.0:
                return f"{num:3.1f} {unit}"
            num /= 1024.0
        return f"{num:.1f} PiB"

    # ---------- FAISS index persisting ----------
    def save_index_and_report_size(self, *, db: str = "graph", out_dir: str | None = None):
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
            # older LangChain versions need the kw-name `folder_path`
            vs.save_local(folder_path=out_dir)

        size_b = self._dir_size_bytes(out_dir)
        human  = self._bytes_to_human(size_b)
        pad    = " " if db == "text" else ""
        print(f"[Index size] {db}_rag = {human}  ({out_dir}){pad}")
        return size_b


class CompressRagRLWithCost(_CostMixin, CompressRag_rl):
    """CompressRag_rl + cost / size helpers (no other change)."""
    pass

# ---------------------------------------------------------------------
# 2) Constants
# ---------------------------------------------------------------------
REPO_ID      = "GraphRAG-Bench/GraphRAG-Bench"
CORPUS_FILE  = "Datasets/Corpus/medical.json"
QUEST_FILE   = "Datasets/Questions/medical_questions.json"
SEED_N       = 10   # bootstrap + DPO train
TEST_N       = 10   # evaluation
TOPK_CTX     = 5

# ---------------------------------------------------------------------
# 3) Initialise models
# ---------------------------------------------------------------------
print("» Initialising embeddings & LLM …")
word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")
sent_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
phi_llm  = Phi4MiniReasoningLLM(
    include_thinkings=True,
    model_name="microsoft/Phi-4-mini-reasoning",
    max_new_tokens=256,
    temperature=0.2,
    top_p=0.9,
)

cr = CompressRagRLWithCost(
    ini_meta_codebook = {},
    sentence_emb      = sent_emb,
    word_emb          = word_emb,
    llm               = phi_llm,
    combine_ents_rounds = 1,            # default; LinUCB will overwrite
    thinkings_choice    = 'not_include',
    answers_choice      = 'overlap',
)

# ---------------------------------------------------------------------
# 4) Load benchmark Q-A-E data
# ---------------------------------------------------------------------
print("» Loading benchmark questions / answers / evidence …")
q_fp = hf_hub_download(REPO_ID, QUEST_FILE, repo_type="dataset")
qrows = json.load(open(q_fp, encoding="utf-8"))

row_lookup  = {r["question"].strip(): r for r in qrows}
gold_lookup = {q: r["answer"]   for q, r in row_lookup.items()}

all_questions = list(row_lookup.keys())
seed_questions = all_questions[:SEED_N]
test_questions = all_questions[SEED_N:SEED_N+TEST_N]

# ---------------------------------------------------------------------
# 5) Load corpus into CompressRAG-RL
# ---------------------------------------------------------------------
facts_json_paths = [hf_hub_download(REPO_ID, CORPUS_FILE, repo_type="dataset")]
cr.set_facts_sources(facts_json_paths)

# ---------------------------------------------------------------------
# 6) Train DPO policy on the seed questions
# ---------------------------------------------------------------------
print("» Building preference pairs for DPO …")
pref_ds = make_preference_dataset_2head(
    cr            = cr,
    questions     = seed_questions,
    gold_answers  = gold_lookup,
    per_q_samples = 6,
    reward_fn     = default_reward,
    seed          = 42,
)
print(f"   generated {len(pref_ds)} preference examples")

policy, _ = train_dpo_2head(pref_ds, input_dim=384)

# ---------------------------------------------------------------------
# 7) Initialise LinUCB scheduler
# ---------------------------------------------------------------------
state_dim = featurize_state(cr).shape[0]
scheduler = CombineScheduler(d=state_dim, arms=COMBINE_ARMS, alpha=1.0, epsilon=0.05)

# ---------------------------------------------------------------------
# 0)  Pre-load FACTS once so every prompt sees them
# ---------------------------------------------------------------------
cr.load_and_merge_facts(facts_json_paths, chunk_chars=500, overlap=100)
cr._facts_preloaded = True           
cr.top_m = 3            

# ---------------------------------------------------------------------
# 8) Seed history (store answers for retrieval)
# ---------------------------------------------------------------------
print("» Seeding history with first 30 questions …")
for q in seed_questions:
    answer_with_auto_strategy(
        cr, policy, scheduler, q,
        reward_fn     = default_reward,
        gold_answer   = gold_lookup[q],
        facts_json_path = facts_json_paths,
        chunk_chars   = 200,
        overlap       = 30,
        greedy        = True
    )

seed_text_stats  = cr.report_cost(kind="text")
seed_graph_stats = cr.report_cost(kind="graph")

cr._metrics = {"graph": [], "text": []}   # reset for clean test stats

# ---------------------------------------------------------------------
# 9) Helper for context capture
# ---------------------------------------------------------------------
def _collect_ctx(cr, k: int = 5) -> List[str]:
    ctx = getattr(cr, "_last_ctx", [])[:k]
    return [re.sub(r"\s+", " ", c.strip()) for c in ctx]

# ---------------------------------------------------------------------
# 10) Run evaluation on next 20 questions & dump JSON
# ---------------------------------------------------------------------
def dump_results(questions: List[str], out_path: str):
    rows = []
    for q in questions:
        pred, _meta = answer_with_auto_strategy(
            cr, policy, scheduler, q,
            reward_fn       = default_reward,
            gold_answer     = gold_lookup[q],
            facts_json_path = facts_json_paths,
            chunk_chars     = 400,
            overlap         = 80,
            greedy          = True
        )
        row = row_lookup[q]
        rows.append({
            "id":              row["id"],
            "question":        q,
            "source":          row["source"],
            "context":         _collect_ctx(cr, TOPK_CTX),
            "evidence":        row["evidence"],
            "question_type":   row["question_type"],
            "generated_answer": pred,
            "ground_truth":    row["answer"],
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=2)
    print(f"   wrote {len(rows)} rows → {out_path}")

print("» Answering 20 evaluation questions …")
dump_results(test_questions, "results/compressrag_medical.json")

test_text_stats  = cr.report_cost(kind="text")
test_graph_stats = cr.report_cost(kind="graph")

# ---------------------------------------------------------------------
# 11) Optional cost summary & FAISS index save
# ---------------------------------------------------------------------
os.makedirs("results", exist_ok=True)
json.dump(
    {
        "seed_text":  seed_text_stats,
        "seed_graph": seed_graph_stats,
        "test_text":  test_text_stats,
        "test_graph": test_graph_stats,
    },
    open("results/cost_summary.json", "w"), indent=2
)

cr.save_index_and_report_size(db="text",  out_dir="faiss_text_idx")
cr.save_index_and_report_size(db="graph", out_dir="faiss_graph_idx")

print("\nDONE – ready to evaluate with generation_eval.py / retrieval_eval.py")
