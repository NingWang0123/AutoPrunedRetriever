import json
from pathlib import Path

# # load light rag results
lg_results_json  = Path(r"GraphRAG-Benchmark\results\lightrag\Medical\predictions_Medical.json")

lg_results = json.load(lg_results_json.open("r", encoding="utf-8"))

questions_lg = [d["question"].strip() for d in lg_results if "question" in d]


# load compress rag results
cg_results_json  = Path(r"results\compressrag_medical_data2.json")


cg_results = json.load(cg_results_json.open("r", encoding="utf-8"))

questions_cg = [d["question"].strip() for d in cg_results if "question" in d]


def norm(q: str) -> str:
    return (q or "").strip()

# check the same questions
lg_set = set(questions_lg)
cg_set = set(questions_cg)
shared_q = lg_set & cg_set
# --- Filtered lists (preserve order) ---
lg_shared   = [d for d in lg_results if norm(d.get("question", "")) in shared_q]
cg_shared   = [d for d in cg_results if norm(d.get("question", "")) in shared_q]

# filter out lg_results and cg_results with shared question
print(f"LightRAG:  total={len(lg_results)} shared={len(lg_shared)}")
print(f"CompressRAG: total={len(cg_results)} shared={len(cg_shared)}")

lg_results_only_shared = lg_shared
cg_results_only_shared = cg_shared

# evaluate lg_results_only_shared and cg_results_only_shared through api



#  python evaluation_only_api.py