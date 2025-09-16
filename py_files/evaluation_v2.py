#!/usr/bin/env python3
# eval_cr_bench.py

import argparse
import json
import os
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from langchain_community.embeddings import HuggingFaceEmbeddings

from CompressRag_rl_v1 import CompressRag_rl, WordAvgEmbeddings
from dpo_compressrag import (
    # policy + training
    StrategyPolicy2Head, TrainCfg, train_dpo_2head,
    # scheduler + helpers
    CombineScheduler, COMBINE_ARMS, featurize_state, default_reward,
    answer_with_auto_strategy, make_preference_dataset_2head,
    # LLM wrapper defined in your dpo_compressrag.py
    Phi4MiniReasoningLLM,
    # utilities / enums
    temp_ans_th, set_combine_rounds, AN2I, TH2I, I2AN, I2TH,
    ANSWERS_CHOICES, THINKINGS_CHOICES,
)

# ---------------------------
# Simple normalization + EM/F1
# ---------------------------
def _normalize(s: str) -> str:
    s = (s or "").strip().lower()
    return " ".join(s.replace("\u00A0", " ").split())

def exact_match(pred: str, golds: List[str]) -> float:
    p = _normalize(pred)
    return float(any(p == _normalize(g) for g in golds))

def f1(pred: str, golds: List[str]) -> float:
    def tokset(x: str): return set(_normalize(x).split())
    P = tokset(pred)
    if not golds:
        return 0.0
    best = 0.0
    for g in golds:
        G = tokset(g)
        if not P and not G:
            return 1.0
        if not P or not G:
            best = max(best, 0.0)
            continue
        tp = len(P & G)
        prec = tp / (len(P) + 1e-9)
        rec  = tp / (len(G) + 1e-9)
        sc = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        best = max(best, sc)
    return best

# ---------------------------
# Fixed policy (when skipping DPO)
# ---------------------------
class FixedPolicy(torch.nn.Module):
    def __init__(self, ans_choice="overlap", th_choice="not_include", device=None):
        super().__init__()
        self.ans_idx = AN2I[ans_choice]
        self.th_idx  = TH2I[th_choice]
        self._dummy = torch.nn.Parameter(torch.zeros(1))  # keep .parameters() happy
        self._device = device

    def forward(self, x):
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, x, greedy=True):
        dev = x.device if hasattr(x, "device") else (self._device or "cpu")
        return torch.tensor([[self.ans_idx, self.th_idx]], dtype=torch.long, device=dev)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions-file", default="Datasets/Questions/medical_questions.json")
    ap.add_argument("--facts", nargs="*", default=[
        "Datasets/Corpus/medical.json",
        # "Datasets/Corpus/novel.json",
    ])
    ap.add_argument("--out", default="results/preds.jsonl")
    ap.add_argument("--skip-dpo", action="store_true", default=True)
    ap.add_argument("--ans-choice", default="overlap", choices=ANSWERS_CHOICES)
    ap.add_argument("--th-choice",  default="not_include", choices=THINKINGS_CHOICES)
    ap.add_argument("--limit", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # --- Build CR ---
    include_thinking = (args.th_choice != "not_include")
    word_emb = WordAvgEmbeddings(model_path="gensim-data/glove-wiki-gigaword-100/glove-wiki-gigaword-100.model")
    sentence_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    phi_llm = Phi4MiniReasoningLLM(
        include_thinkings=include_thinking,
        model_name="microsoft/Phi-4-mini-reasoning",
        max_new_tokens=512,
        temperature=0.2,
        top_p=0.9
    )
    cr = CompressRag_rl(
        ini_meta_codebook={},
        sentence_emb=sentence_emb,
        word_emb=word_emb,
        llm=phi_llm,
        combine_ents_rounds=1,
        thinkings_choice=args.th_choice,
        answers_choice=args.ans_choice
    )

    repo_id = "GraphRAG-Bench/GraphRAG-Benchs"

    # --- Download questions ---
    q_fp = hf_hub_download(
        repo_id=repo_id,
        filename="Datasets/Questions/medical_questions.json",
        repo_type="dataset",
    )

    with open(q_fp, "r", encoding="utf-8") as f:
        qrows = json.load(f)

    # --- Download facts -> local paths ---
    facts_json_paths = [
        hf_hub_download(repo_id=repo_id, filename="Datasets/Corpus/medical.json", repo_type="dataset")
    ]
    cr.set_facts_sources(facts_json_paths)


    examples = make_preference_dataset_2head(
        cr=cr,
        questions=qrows[30:40],
        gold_answers=gold,
        per_q_samples=6,
        feature_dim=384,
        reward_fn=default_reward,
        seed=0,
        isolate_state=True,
        combine_rounds_default=1,  # keep combine fixed during DPO data creation
    )

    # --- 3) train DPO policy
    policy, ref = train_dpo_2head(examples, input_dim=384)
    # --- Scheduler (LinUCB over combine cadence) ---
    d_state = featurize_state(cr).shape[0]  # typically small (e.g., 4)
    scheduler = CombineScheduler(d=d_state, arms=COMBINE_ARMS, alpha=1.0, epsilon=0.05)

    # --- Run predictions + metrics ---
    n_total = len(qrows)
    n = min(args.limit, n_total) if args.limit > 0 else n_total
    out_path = args.out
    bench_path = f"results/bench_input_{ts}.jsonl"

    ems, f1s = [], []
    t0 = time.time()

    with open(out_path, "w", encoding="utf-8") as wf, open(bench_path, "w", encoding="utf-8") as wbench:
        for i in range(n):
            row = qrows[i]
            qid = row.get("id", i)
            q = (row["question"] if isinstance(row, dict) and "question" in row else str(row)).strip()

            # collect golds if present
            golds: List[str] = []
            if isinstance(row, dict):
                if isinstance(row.get("answer"), str):
                    golds = [row["answer"]]
                elif isinstance(row.get("answers"), list):
                    golds = [a for a in row["answers"] if isinstance(a, str)]

            # let LinUCB update only when we have a gold reference
            pred, meta = answer_with_auto_strategy(
                cr=cr,
                policy=policy,
                scheduler=scheduler,
                q=q,
                reward_fn=(default_reward if golds else None),
                gold_answer=(golds[0] if golds else None),
                greedy=True
            )

            # General record
            rec: Dict = {"id": qid, "question": q, "prediction": pred, "meta": meta}
            if golds:
                rec["gold"] = golds
                ems.append(exact_match(pred, golds))
                f1s.append(f1(pred, golds))
            wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # GraphRAG-Bench friendly record
            bench_rec = {
                "id": qid,
                "source": row.get("source"),
                "question": q,
                "answer": golds[0] if golds else "",
                "prediction": pred,
                "evidence": row.get("evidence", []),
            }
            wbench.write(json.dumps(bench_rec, ensure_ascii=False) + "\n")

    dt = time.time() - t0
    print(f"\nSaved predictions to {out_path}")
    print(f"Saved bench input to {bench_path}")
    if ems:
        print(f"EM: {np.mean(ems):.4f} | F1: {np.mean(f1s):.4f} | N={len(ems)}")
    print(f"Total time: {dt:.1f}s  ({n} questions)\n")

    # --- Cost / index size reports (only if your CR exposes these) ---
    run_report_path = f"results/run_report_{ts}.json"
    report_payload = {
        "timestamp": ts,
        "n": n,
        "runtime_sec": dt,
        "em": float(np.mean(ems)) if ems else None,
        "f1": float(np.mean(f1s)) if f1s else None,
    }

    # average cost metrics
    for kind in ("graph", "text"):
        if hasattr(cr, "report_cost"):
            try:
                stats = cr.report_cost(kind=kind, avg=True)  # prints + returns dict
                report_payload[f"cost_{kind}_avg"] = stats
            except Exception as e:
                print(f"[Cost] Skipped {kind} metrics: {e}")

    # index sizes
    for db, out_dir in (("graph", f"results/faiss_graph_{ts}"),
                        ("text",  f"results/faiss_text_{ts}")):
        if hasattr(cr, "save_index_and_report_size"):
            try:
                _ = cr.save_index_and_report_size(db=db, out_dir=out_dir)
            except Exception as e:
                print(f"[Index size] {db}: skipped ({e})")

    with open(run_report_path, "w", encoding="utf-8") as wj:
        json.dump(report_payload, wj, ensure_ascii=False, indent=2)
    print(f"Saved run report to {run_report_path}")

    print("\nTip: for a timestamped log, run:")
    print('  python -u eval_cr_bench.py 2>&1 | tee "results/eval_$(date +%F_%H-%M-%S).log"\n')
    print("Tip: then run GraphRAG-Bench evaluation.py against:")
    print(f"  preds:   {bench_path}")
    print(f"  dataset: {q_fp}")

if __name__ == "__main__":
    main()
