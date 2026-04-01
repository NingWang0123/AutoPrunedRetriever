import os, subprocess, sys
from pathlib import Path    


REPO_ROOT = Path("GraphRAG-Benchmark")
DATA      = Path("configs/outputs/novel/full_novel_results.json")
OUT_DIR   = Path("configs/outputs/novel")
OUT_DIR.mkdir(parents=True, exist_ok=True)

assert (REPO_ROOT / "Evaluation" / "retrieval_eval.py").exists(), "couldn't find Evaluation/retrieval_eval.py"
assert (REPO_ROOT / "Evaluation" / "generation_eval.py").exists(), "couldn't find Evaluation/generation_eval.py"
assert DATA.exists(), f"couldn't find {DATA}"

env = os.environ.copy()
env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
env["LLM_API_KEY"] = ""

def run_eval(cmd, outfile):
    proc = subprocess.run(cmd, env=env, text=True)
    if proc.returncode != 0:
        print("----- evaluator stdout -----\n", proc.stdout)
        print("----- evaluator stderr -----\n", proc.stderr)
        proc.check_returncode()
    else:
        print(f"✅ wrote {outfile}")

base_cmd = [
    sys.executable, "-m", "Evaluation.retrieval_eval",
    "--mode", "API",
    "--model", "gpt-4o-mini",    
    "--data_file", str(DATA),
    "--output_file", str(OUT_DIR / "retrieval_scores_apr.json"),
    "--detailed_output",
]

#run_eval(base_cmd + ["--embedding_model", "BAAI/bge-large-en"], OUT_DIR / "retrieval_scores.json")

gen_cmd = [
    sys.executable, "-m", "Evaluation.generation_eval",
    "--mode", "API",
    "--model", "gpt-4o-mini",
    "--data_file", str(DATA),
    "--output_file", str(OUT_DIR / "generation_scores_medical.json"),
    "--detailed_output",
    "--embedding_model", "BAAI/bge-large-en",         
]
run_eval(gen_cmd, OUT_DIR / "generation_scores_full.json")

print("🎉  Benchmark complete — score files are in results/")
