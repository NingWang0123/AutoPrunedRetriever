"""
Run AutoPrunedRetriever on STEM or TV dataset.

Usage:
    python run_apr.py --config configs/stem.yaml
    python run_apr.py --config configs/tv.yaml
"""

import os, json, time, argparse, sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

def main():
    parser = argparse.ArgumentParser(description="Run APR")
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config")
    parser.add_argument("--api-key", default=None, help="OpenAI API key (or set OPENAI_API_KEY)")
    args = parser.parse_args()

    # ── Load config ──
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY or pass --api-key")

    # ── Initialise embeddings & LLM ──
    print("Initialising embeddings & LLM ...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from auto_pruned_retriever import AutoPrunedRetriver
    from llm_api import OpenAILLM
    from graph_generator.llm_parser import TOKEN_STATS

    word_emb = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    sent_emb = word_emb

    llm = OpenAILLM(
        include_thinkings=False,
        model_name="gpt-4o-mini",
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
        use_cache=True,
        api_key=api_key,
    )

    # ── Load questions ──
    print("Loading questions ...")
    with open(cfg["quest_file"], "r", encoding="utf-8") as f:
        qrows = json.load(f)
    row_lookup = {r["question"].strip(): r for r in qrows}
    gold_lookup = {q: r["answer"] for q, r in row_lookup.items()}
    all_questions = list(row_lookup.keys())
    print(f"  {len(all_questions)} questions loaded")

    # ── Build / load meta-codebook ──
    ini_meta_json = Path(cfg["ini_meta_json"])
    ini = {}
    if ini_meta_json.is_file():
        print(f"Loading existing codebook from {ini_meta_json} ...")
        with open(ini_meta_json, "r", encoding="utf-8") as f:
            ini = json.load(f)
        pre_loaded = True
    else:
        pre_loaded = False

    cr = AutoPrunedRetriver(
        ini_meta_codebook=ini,
        sentence_emb=sent_emb,
        word_emb=word_emb,
        llm=llm,
        thinkings_choice="not_include",
        answers_choice="unique",
        facts_choice="include_all",
        top_m=cfg.get("top_m", 20),
        top_k=cfg.get("top_k", 200),
        combine_ent_sim=cfg.get("combine_ent_sim", 0.93),
        q_combine_sim=cfg.get("q_combine_sim", 0.93),
        aft_combine_sim=cfg.get("aft_combine_sim", 0.93),
        semantic_overlap_sim=cfg.get("semantic_overlap_sim", 0.93),
        use_word=True,
        chunking_use=cfg.get("chunking_use", "llm"),
        chunking_api=api_key,
    )
    cr.meta_codebook["_ini_meta_json_path"] = str(ini_meta_json)

    # ── Pre-load corpus facts (if not cached) ──
    if not pre_loaded:
        print("Pre-loading corpus as facts ...")
        tok_before = {k: v for k, v in TOKEN_STATS.items()}
        cr.load_and_merge_facts(
            cfg["corpus_file"],
            chunk_tokens=1200,
            overlap_tokens=100,
            sub_chunk_chars=200,
            sub_chunk_overlap=50,
            tokenizer_name="gpt-4o-mini",
            subchunk_batch=1000,
            subchunk_mode=cfg.get("subchunk_mode", "tokens"),
            sub_chunk_token_size=256,
            sub_chunk_token_overlap=50,
        )
        cr._facts_preloaded = True

        # Save graph token stats
        graph_stats = {
            "graph_prompt_tokens": TOKEN_STATS["prompt_tokens"] - tok_before["prompt_tokens"],
            "graph_completion_tokens": TOKEN_STATS["completion_tokens"] - tok_before["completion_tokens"],
        }
        print(f"[GRAPH TOKENS] {graph_stats}")

        # Save codebook
        def _json_safe(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, set): return [_json_safe(v) for v in obj]
            if isinstance(obj, dict): return {k: _json_safe(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_json_safe(v) for v in obj]
            return obj

        ini_meta_json.parent.mkdir(parents=True, exist_ok=True)
        ini_meta_json.write_text(
            json.dumps(_json_safe(cr.meta_codebook), indent=2), encoding="utf-8"
        )
        print(f"Codebook saved to {ini_meta_json}")

    cr.ensure_edge_matrix_embedding_in()

    # Initial entity pruning
    if cr.meta_codebook.get("_combine_round", 0) == 0:
        print("Running initial entity pruning (KNN) ...")
        t0 = time.time()
        cr.combine_ents_func(mode="knn")
        print(f"  done in {time.time() - t0:.1f}s")

    # ── Answer questions ──
    print(f"\nAnswering {len(all_questions)} questions ...")
    final_path = Path(cfg["final_json_path"])
    final_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for q in tqdm(all_questions, desc="Questions"):
        t_start = time.time()
        start_idx = len(llm.metrics_runs)

        answer, context, q_json = cr.run_work_flow(
            question=q,
            rule="Answer questions using the given knowledge graph context.",
            facts_json_path=cfg["corpus_file"],
            warm_start="knn",
            skip_update_meta=cfg.get("skip_update_meta", False),
        )

        t_end = time.time()
        gen_metrics = (llm.metrics_runs[start_idx:] or [{}])[-1] or {}
        if isinstance(gen_metrics, dict) and q in gen_metrics:
            gen_metrics = gen_metrics[q]

        row = row_lookup.get(q, {})
        results.append({
            "id": row.get("id", ""),
            "question": q,
            "question_type": row.get("question_type", ""),
            "generated_answer": answer,
            "ground_truth": gold_lookup.get(q, ""),
            "context": context,
            "input_tokens": gen_metrics.get("input_tokens"),
            "output_tokens": gen_metrics.get("output_tokens"),
            "total_tokens": gen_metrics.get("total_tokens"),
            "gen_latency_sec": gen_metrics.get("gen_latency_sec"),
            "retrieval_latency_sec": gen_metrics.get("retrieval_latency_sec"),
            "total_latency_sec": gen_metrics.get("total_latency_sec"),
            "retrieved_count": gen_metrics.get("retrieved_count"),
            "model_name": gen_metrics.get("model_name"),
            "meta_codebook_json_MB": len(json.dumps(
                cr.meta_codebook.get("e", [])
            ).encode()) / 1e6,
        })

        # Periodic save every 50 questions
        if len(results) % 50 == 0:
            final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final save
    final_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {final_path}")
    print(f"Total questions: {len(results)}")


if __name__ == "__main__":
    main()
