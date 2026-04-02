"""
Run AutoPruned Layer (APL) on top of baseline RAG predictions.

APL takes baseline RAG system predictions (with retrieved_contexts),
re-parses the context into structured KG edges, and generates answers
using the indices-only prompting approach with cross-question memory.

Usage:
    python run_apl.py --predictions path/to/baseline_predictions.json \
                      --output path/to/apl_results.json

Expected input format (JSON array):
[
    {
        "id": "q_001",
        "question": "...",
        "answer": "...",            # ground truth (optional)
        "question_type": "...",     # e.g. "Complex Reasoning" (optional)
        "retrieved_contexts": ["paragraph1...", "paragraph2...", ...]
    },
    ...
]
"""

import os, json, time, argparse, sys
from pathlib import Path
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Run APL on baseline RAG predictions")
    parser.add_argument("--predictions", "-p", required=True,
                        help="Path to baseline RAG predictions JSON")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to write APL-enhanced results")
    parser.add_argument("--api-key", default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY)")
    parser.add_argument("--api-base", default=None,
                        help="API base URL (for OpenAI-compatible endpoints)")
    parser.add_argument("--model", default="gpt-4o-mini",
                        help="LLM model name (default: gpt-4o-mini)")
    parser.add_argument("--embedding-model", default="BAAI/bge-large-en-v1.5",
                        help="Embedding model name (default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-m", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--combine-ent-sim", type=float, default=0.93)
    parser.add_argument("--compress-rate", type=float, default=0.2)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    api_base = args.api_base or os.environ.get("OPENAI_API_BASE")
    if not api_key:
        sys.exit("Error: set OPENAI_API_KEY or pass --api-key")

    # ── Load predictions ──
    print("Loading baseline predictions ...")
    with open(args.predictions, "r", encoding="utf-8") as f:
        predictions = json.load(f)
    print(f"  {len(predictions)} predictions loaded")

    # ── Initialise APL ──
    print(f"Initialising embeddings ({args.embedding_model}) & LLM ({args.model}) ...")
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from auto_pruned_layer import AutoPrunedLayer
    from llm_api import OpenAILLM

    word_emb = HuggingFaceEmbeddings(model_name=args.embedding_model)
    sent_emb = word_emb

    llm = OpenAILLM(
        include_thinkings=False,
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=0.9,
        use_cache=True,
        api_key=api_key,
        base_url=api_base,
    )

    apl = AutoPrunedLayer(
        word_emb=word_emb,
        sentence_emb=sent_emb,
        llm=llm,
        chunking_api=api_key,
        parser_choice="llm",
        top_m=args.top_m,
        top_k=args.top_k,
        combine_ent_sim=args.combine_ent_sim,
        semantic_overlap_sim=0.9,
        answers_choice="overlap",
        compress_rate=args.compress_rate,
    )

    # ── Process predictions ──
    print(f"\nProcessing {len(predictions)} predictions through APL ...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for pred in tqdm(predictions, desc="APL"):
        t0 = time.time()
        enhanced = apl.process_prediction(pred)
        t1 = time.time()

        results.append({
            "id": pred.get("id", ""),
            "question": pred["question"],
            "question_type": pred.get("question_type", ""),
            "generated_answer": enhanced.get("answer", ""),
            "ground_truth": pred.get("answer", ""),
            "context": enhanced.get("context", ""),
            "total_latency_sec": t1 - t0,
        })

        # Periodic save
        if len(results) % 50 == 0:
            output_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    # Final save
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to {output_path}")
    print(f"Total processed: {len(results)}")


if __name__ == "__main__":
    main()
