import json
from pathlib import Path
from statistics import mean

# -------- Paths --------
GEN_PATH = Path("GraphRAG-Benchmark/results/generation_compressrag_novel_final.json")
ORIG_QUEST_LOCAL = Path("GraphRAG-Benchmark/Datasets/Questions/novel_questions.json")
OUT_PATH = Path("results/reval_novel_lowest500_by_answer_correctness.json")

# -------- Config --------
SECTION_NAME = "Fact Retrieval"     # as shown in your JSON
METRIC = "answer_correctness"       # rank by this
TIE_METRIC = "rouge_score"          # tie-breaker (ascending)
TOP_N = 500

def load_metrics_items(gen_path: Path, section_name: str):
    """Return list of metric items (dicts) for the given section."""
    data = json.loads(gen_path.read_text())
    # Accept exact key or case-insensitive match
    sec_key = None
    for k in data.keys():
        if k.lower() == section_name.lower():
            sec_key = k
            break
    if sec_key is None:
        raise KeyError(f"Section '{section_name}' not found in {gen_path}")

    detailed = data[sec_key].get("detailed", [])
    if not isinstance(detailed, list):
        raise ValueError("Expected 'detailed' to be a list")
    return detailed

def dedupe_keep_worst(items, metric: str, tie_metric: str):
    """
    De-duplicate by id, keeping the WORST (lowest) metric; on ties, lowest tie_metric wins.
    Returns list of unique items.
    """
    best = {}
    for it in items:
        _id = it.get("id")
        m = it.get("metrics", {}) or {}
        score = m.get(metric, None)
        tie = m.get(tie_metric, 0.0)
        if not _id or not isinstance(score, (int, float)):
            continue
        prev = best.get(_id)
        if prev is None:
            best[_id] = (score, tie, it)
        else:
            ps, pt, _ = prev
            if (score < ps) or (score == ps and tie < pt):
                best[_id] = (score, tie, it)
    # return unique items (worst per id)
    return [t[2] for t in best.values()]

def main():
    assert GEN_PATH.is_file(), f"Missing metrics file: {GEN_PATH}"
    assert ORIG_QUEST_LOCAL.is_file(), f"Missing original novel_questions.json: {ORIG_QUEST_LOCAL}"

    # 1) Load Fact Retrieval items from metrics
    items = load_metrics_items(GEN_PATH, SECTION_NAME)
    print(f"[metrics] {SECTION_NAME}: {len(items)} entries")

    # 2) Dedupe by id keeping worst score
    uniq_items = dedupe_keep_worst(items, METRIC, TIE_METRIC)
    print(f"[metrics] unique by id (keeping worst): {len(uniq_items)}")

    # 3) Sort by (metric asc, tie asc) and take lowest TOP_N
    uniq_items.sort(key=lambda it: (it["metrics"].get(METRIC, float("inf")),
                                    it["metrics"].get(TIE_METRIC, float("inf"))))
    chosen = uniq_items[:TOP_N]
    chosen_ids = [it["id"] for it in chosen]

    # 4) Compute and print averages for the selected subset
    sel_scores = [it["metrics"].get(METRIC, 0.0) for it in chosen]
    sel_rouge  = [it["metrics"].get(TIE_METRIC, 0.0) for it in chosen]
    avg_metric = mean(sel_scores) if sel_scores else 0.0
    avg_rouge  = mean(sel_rouge) if sel_rouge else 0.0

    print(f"[select] taking lowest {len(chosen)} by '{METRIC}'")
    print(f"[averages over selected {len(chosen)}]")
    print(f"  avg {METRIC}:   {avg_metric:.6f}")
    print(f"  avg {TIE_METRIC}: {avg_rouge:.6f}")

    # 5) Load original questions and filter by chosen IDs (preserve ranking order)
    orig_rows = json.loads(ORIG_QUEST_LOCAL.read_text(encoding="utf-8"))
    by_id = {r.get("id"): r for r in orig_rows}
    subset = [by_id[i] for i in chosen_ids if i in by_id]
    missing = [i for i in chosen_ids if i not in by_id]
    if missing:
        print(f"[warn] {len(missing)} ids not found in original questions; skipped.")

    # 6) Save subset
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(subset, indent=2, ensure_ascii=False))
    print(f"[write] wrote {len(subset)} rows → {OUT_PATH}")

    # 7) (Optional) Save an audit list with ids and their scores
    audit_path = OUT_PATH.with_suffix(".ids_with_scores.json")
    audit = [
        {
            "id": it["id"],
            METRIC: it["metrics"].get(METRIC, None),
            TIE_METRIC: it["metrics"].get(TIE_METRIC, None),
            "question": it.get("question", "")
        }
        for it in chosen
    ]
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(f"[write] wrote audit list → {audit_path}")

    if subset:
        print("[sample ids]", [row["id"] for row in subset[:10]])

if __name__ == "__main__":
    main()


# python test_performance_for_lowest_q.py