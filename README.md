# AutoPrunedRetriever

<p align="center">
  <img src="octopus_pruned.png" width="400" alt="AutoPrunedRetriever">
</p>

Code for **AutoPrunedRetriever (APR)** and **AutoPruned Layer (APL)** — a structured knowledge graph retrieval system for complex reasoning over documents.

## Architecture

<p align="center">
  <img src="apr_pipeline.png" width="800" alt="APR Pipeline">
</p>

## Repository Structure

```
AutoPrunedRetriever/
├── gpu/                            # APR + APL (requires GPU)
│   ├── run_apr.py                  # Run APR standalone
│   ├── run_apl.py                  # Run APL on baseline RAG predictions
│   ├── auto_pruned_retriever.py    # Core APR class
│   ├── auto_pruned_layer.py        # Core APL class
│   ├── dpo_exactgraphrag.py        # DPO strategy learning
│   ├── reward_func_dpo.py          # Reward functions (SBERT, BLEU, ROUGE)
│   ├── retrieve_simple.py          # 6-signal hybrid retrieval
│   ├── retrieve_gpu_cached_combined.py
│   ├── combine_ent_cached_aligned.py
│   ├── sentence_embed_overlap_cached.py
│   ├── test_continous_chunk_cached.py
│   ├── llm_api.py
│   ├── mem_debug.py
│   ├── graph_generator/
│   │   ├── llm_parser.py           # LLM-based triplet extraction
│   │   ├── llm_parser_concurrent.py
│   │   └── rebel_large.py          # REBEL triplet extraction (local, no API)
│   └── configs/
│       ├── stem.yaml               # STEM with LLM parser
│       ├── tv.yaml                 # TV with LLM parser
│       ├── stem_rebel.yaml         # STEM with REBEL parser (codebook-free)
│       └── tv_rebel.yaml           # TV with REBEL parser (codebook-free)
├── cpu/                            # Legacy version (original codebase)
│   └── ...
├── data/                           # Shared datasets
│   ├── stem_question.json
│   ├── tv_questions.json
│   └── corpus/
│       ├── stem_corpus.json
│       └── tv_corpus.json
├── instructions/                   # Detailed usage guides
│   ├── apr_guide.md
│   └── apl_guide.md
├── requirements.txt
└── README.md
```

## Quick Start

### Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

### Run APR (standalone retrieval system)

APR uses DPO (Direct Preference Optimization) to learn a lightweight strategy policy that selects the best retrieval configuration per question.

```bash
cd gpu

# STEM dataset — LLM parser (with DPO, default)
python run_apr.py --config configs/stem.yaml

# TV dataset — LLM parser
python run_apr.py --config configs/tv.yaml

# REBEL parser variant (codebook-free, no API cost for graph construction)
python run_apr.py --config configs/stem_rebel.yaml
python run_apr.py --config configs/tv_rebel.yaml

# Without DPO (fixed strategy)
python run_apr.py --config configs/stem.yaml --no-dpo
```

### Run APL (plug-in layer on baseline RAG)

APL enhances any baseline RAG system's predictions by re-parsing retrieved context into structured KG edges and generating answers with cross-question memory.

```bash
cd gpu
python run_apl.py --predictions path/to/baseline_predictions.json \
                  --output path/to/apl_results.json
```

**Expected input format** (`baseline_predictions.json`):
```json
[
    {
        "id": "q_001",
        "question": "...",
        "answer": "reference answer",
        "question_type": "Complex Reasoning",
        "retrieved_contexts": ["passage 1...", "passage 2..."]
    }
]
```

## Configuration

### YAML config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunking_use` | `llm` | Triplet extraction: `llm` (API-based) or `rebel` (local model, no API cost) |
| `model_name` | `gpt-4o-mini` | LLM model for generation and parsing |
| `embedding_model` | `BAAI/bge-large-en-v1.5` | Embedding model for entity/sentence embeddings |
| `api_base` | — | API base URL (for OpenAI-compatible endpoints) |
| `temperature` | `0.2` | LLM temperature |
| `max_new_tokens` | `256` | Max generation tokens |
| `top_m` | `20` | Number of final retrieved results per question |
| `top_k` | `200` | Candidate pool size for retrieval |
| `combine_ent_sim` | `0.93` | Cosine similarity threshold for entity merging |
| `semantic_overlap_sim` | `0.93` | Threshold for semantic deduplication |
| `seed_n` | `20` | Number of seed questions for DPO training |
| `skip_update_meta` | `false` | If true, disables memory accumulation |

### Command-line overrides

All YAML parameters can be overridden via CLI:

```bash
# Use a different LLM
python run_apr.py --config configs/stem.yaml --model gpt-4o --temperature 0.1

# Use a different embedding model
python run_apr.py --config configs/stem.yaml --embedding-model sentence-transformers/all-MiniLM-L6-v2

# Use an OpenAI-compatible endpoint (e.g., vLLM, Ollama, Azure)
python run_apr.py --config configs/stem.yaml \
    --api-base http://localhost:8000/v1 \
    --model my-local-model \
    --api-key dummy

# APL with custom model
python run_apl.py -p predictions.json -o output.json \
    --model gpt-4o \
    --embedding-model BAAI/bge-base-en-v1.5 \
    --api-base https://my-endpoint.com/v1
```

### Environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key (can also pass via `--api-key`) |
| `OPENAI_API_BASE` | API base URL (can also pass via `--api-base`) |

## Hardware Requirements

- **GPU**: CUDA-capable GPU with >= 8GB VRAM (for embedding computation)
- **RAM**: >= 16GB
- **API**: Any OpenAI-compatible API (OpenAI, Azure, vLLM, Ollama, etc.)

## Documentation

See [`instructions/`](instructions/) for detailed guides:
- [APR Guide](instructions/apr_guide.md) — full pipeline walkthrough, DPO details, config reference, custom datasets
- [APL Guide](instructions/apl_guide.md) — input format, baseline conversion, step-by-step pipeline, tips
