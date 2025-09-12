import os
os.environ["LLM_API_KEY"] = "sk-5d4bc3b7dc89439ba402365bf39c7cd3"

import re, json, subprocess
from pathlib import Path
from graph_generator.graphparsers import RelationshipGraphParser

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def collect_text_contexts(rag, q: str, k: int = 3):
    _, hits = rag.similarity_search_text_docs(q, rag.text_db, k=k, emb=rag.sentence_emb)
    ctx = []
    for pair in hits or []:
        d = pair[0] if isinstance(pair, (list, tuple)) else pair
        ctx.append(_clean(d.page_content))
    return ctx

def collect_graph_contexts(rag, parser, q: str, k: int = 3):
    _, hits = rag.similarity_search_graph_docs(q, parser, rag.graph_db, k=k, emb_model=rag.word_emb)
    ctx = []
    for pair in hits or []:
        d = pair[0] if isinstance(pair, (list, tuple)) else pair
        meta = (d.metadata or {}).get("codebook_main") or {}
        e, r, em = meta.get("e", []), meta.get("r", []), meta.get("edge_matrix", [])
        triples = []
        for s, rel, o in em[:3]:  # up to 3 triples per hit
            try:
                triples.append(f"{e[s]} --{r[rel]}--> {e[o]}")
            except Exception:
                pass
        ctx.append("; ".join(triples) if triples else _clean(d.page_content))
    return ctx

def dump_results_for_benchmark(
    rag,
    qrows: list,              # loaded HF questions json (GraphRAG-Bench)
    questions: list[str],     # the subset you want to evaluate
    out_path: str,            # where to write benchmark json
    mode: str = "text",       # "text" or "graph"
    topk: int = 5
):
    id_lookup = {r["question"]: r for r in qrows}
    items = []
    parser = RelationshipGraphParser() if mode == "graph" else None

    for i, q in enumerate(questions):
        meta = id_lookup.get(q, {})
        if mode == "text":
            ans = rag.answer_with_llm_text(q, text_db=rag.text_db)
            ctx = collect_text_contexts(rag, q, topk)
        else:
            ans, _, _ = rag.make_graph_answer(q, parser, faiss_db=rag.graph_db)
            ctx = collect_graph_contexts(rag, parser, q, topk)

        items.append({
            "id": meta.get("id", f"q_{i}"),
            "question": q,
            "source": meta.get("source") or meta.get("corpus_name", "Medical"),
            "context": ctx,                                # list[str]
            "evidence": meta.get("evidence", ""),          # string
            "question_type": meta.get("question_type", "Fact Retrieval"),
            "generated_answer": ans,
            "ground_truth": meta.get("answer", "")
        })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(items, f, indent=2)
    print(f"Saved {len(items)} rows to {out_path}")

def run_benchmark_eval(data_file: str, eval_type: str = "generation",
                       model: str = "deepseek-chat",
                       base_url: str = "https://api.deepseek.com/v1",
                       out_file: str | None = None,
                       num_samples: int | None = None,
                       repo_dir: str | None = None):
    """
    Calls the official scripts:
      - Evaluation/generation_eval.py
      - Evaluation/retrieval_eval.py
    Uses absolute paths so changing cwd won't break file access.
    """
    import os, subprocess
    from pathlib import Path

    # Resolve the repo directory (accept either dash or underscore)
    if repo_dir is None:
        for candidate in ("GraphRAG_Benchmark", "GraphRAG-Benchmark"):
            if os.path.isdir(candidate):
                repo_dir = candidate
                break
    if repo_dir is None:
        raise FileNotFoundError("GraphRAG-Benchmark repo folder not found (GraphRAG_Benchmark or GraphRAG-Benchmark).")

    data_file_abs = os.path.abspath(data_file)
    if out_file is None:
        stem = os.path.splitext(os.path.basename(data_file_abs))[0]
        out_file_abs = os.path.abspath(f"results/{stem}_{eval_type}_scores.json")
    else:
        out_file_abs = os.path.abspath(out_file)

    os.makedirs(os.path.dirname(out_file_abs), exist_ok=True)

    cmd = [
        "python", "-m", f"Evaluation.{eval_type}_eval",
        "--mode", "API",
        "--model", model,
        "--base_url", base_url,
        "--embedding_model", "BAAI/bge-large-en-v1.5",
        "--data_file", data_file_abs,
        "--output_file", out_file_abs,
        "--detailed_output"
    ]
    if num_samples:
        cmd += ["--num_samples", str(num_samples)]

    print("Running:", " ".join(cmd))
    # Ensure the subprocess sees your API key
    env = os.environ.copy()
    # Run from the repo root so module imports like Evaluation.* work
    res = subprocess.run(cmd, cwd=repo_dir, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("STDERR:\n", res.stderr)
    else:
        print("STDOUT:\n", res.stdout)
    print(f"-> Results at {out_file_abs}")


import os, json
from huggingface_hub import hf_hub_download
from zipRAG_v1 import RAGWorkflow
from graph_generator.graphparsers import RelationshipGraphParser

# If dump_results_for_benchmark / run_benchmark_eval are in another file, import them here.
# from evaluation_helpers import dump_results_for_benchmark, run_benchmark_eval

os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "sk-5d4bc3b7dc89439ba402365bf39c7cd3")

# 1) Init RAG
rag = RAGWorkflow(
    config={
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model_id": "microsoft/Phi-4-mini-reasoning",
        "device_map": "auto",
        "dtype_policy": "auto",
        "faiss_search_k": 5,
    },
    verbose=False
)
rag.set_llm()

# 2) Ingest corpus (both graph + text)
repo = "GraphRAG-Bench/GraphRAG-Bench"
corpus_fp = hf_hub_download(repo, "Datasets/Corpus/medical.json", repo_type="dataset")
rag.ingest_context_json(corpus_fp, chunk_chars=900, overlap=150, to_graph=True)
rag.ingest_context_json(corpus_fp, chunk_chars=900, overlap=150, to_graph=False)

# 3) Load the questions file BEFORE using qrows/questions
q_fp = hf_hub_download(repo, "Datasets/Questions/medical_questions.json", repo_type="dataset")
with open(q_fp, "r") as f:
    qrows = json.load(f)  # list of dicts: {id, question, answer, question_type, evidence, ...}

# 4) Seed history with the FIRST 30 GT Q/A (so retrieval can see them)
seed_rows = qrows[:30]
seed_questions = [r["question"] for r in seed_rows]
gt_lookup = {r["question"]: r["answer"] for r in seed_rows}

# TEXT history (ground truth upsert)
rag.build_textdb_with_answers(
    seed_questions,
    bootstrap_db=rag.text_db
)

# GRAPH history (ground truth upsert)
parser = RelationshipGraphParser()
rag.build_graphdb_with_answer(
    seed_questions,
    parser,
    faiss_db=rag.graph_db
)

# 5) Choose a DISJOINT test set (next 30); change to [:40] for 10, etc.
test_rows = qrows[30:70]
test_questions = [r["question"] for r in test_rows]

# 6) Dump results for the official benchmark scripts
dump_results_for_benchmark(
    rag, qrows, test_questions,
    out_path="results/text_rag_medical.json",
    mode="text", topk=5
)

dump_results_for_benchmark(
    rag, qrows, test_questions,
    out_path="results/graph_rag_medical.json",
    mode="graph", topk=5
)

# 7) Run GraphRAG-Benchmark evaluations (uses LLM_API_KEY)
run_benchmark_eval(
    "results/text_rag_medical.json",  eval_type="generation",
    model=os.getenv("EVAL_MODEL", "deepseek-chat"),
    base_url=os.getenv("EVAL_BASE_URL", "https://api.deepseek.com/v1"),
    num_samples=30
)

run_benchmark_eval(
    "results/text_rag_medical.json",  eval_type="retrieval",
    model=os.getenv("EVAL_MODEL", "deepseek-chat"),
    base_url=os.getenv("EVAL_BASE_URL", "https://api.deepseek.com/v1"),
    num_samples=30
)

run_benchmark_eval(
    "results/graph_rag_medical.json", eval_type="generation",
    model=os.getenv("EVAL_MODEL", "deepseek-chat"),
    base_url=os.getenv("EVAL_BASE_URL", "https://api.deepseek.com/v1"),
    num_samples=30
)

run_benchmark_eval(
    "results/graph_rag_medical.json", eval_type="retrieval",
    model=os.getenv("EVAL_MODEL", "deepseek-chat"),
    base_url=os.getenv("EVAL_BASE_URL", "https://api.deepseek.com/v1"),
    num_samples=30
)