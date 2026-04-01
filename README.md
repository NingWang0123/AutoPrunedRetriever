# AutoPrunedRetriever

<p align="center">
  <img src="octopus_pruned.png" width="400" alt="AutoPrunedRetriever">
</p>

Code for **AutoPrunedRetriever (APR)** and **AutoPruned Layer (APL)** — a structured knowledge graph retrieval system for complex reasoning over documents.

<p align="center">
  <img src="workflow.png" width="700" alt="APR Pipeline">
</p>

## Repository Structure

```
AutoPrunedRetriever/
├── gpu/                        # Simple version (APR + APL, requires GPU)
│   ├── run_apr.py              # Run APR standalone
│   ├── run_apl.py              # Run APL on baseline RAG predictions
│   ├── auto_pruned_retriever.py
│   ├── auto_pruned_layer.py
│   ├── retrieve_simple.py
│   ├── retrieve_gpu_cached_combined.py
│   ├── combine_ent_cached_aligned.py
│   ├── sentence_embed_overlap_cached.py
│   ├── test_continous_chunk_cached.py
│   ├── llm_api.py
│   ├── dpo_exactgraphrag.py             # DPO strategy learning
│   ├── reward_func_dpo.py               # Reward functions (SBERT, BLEU, ROUGE)
│   ├── mem_debug.py
│   ├── graph_generator/
│   │   ├── llm_parser.py
│   │   └── llm_parser_concurrent.py
│   └── configs/
│       ├── stem_simple.yaml
│       └── tv_simple.yaml
├── cpu/                        # Legacy version (original codebase)
│   └── ...
├── data/                       # Shared datasets
│   ├── stem_question.json
│   ├── tv_questions.json
│   └── corpus/
│       ├── stem_corpus.json
│       └── tv_corpus.json
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

# STEM dataset (with DPO, default)
python run_apr.py --config configs/stem.yaml

# TV dataset
python run_apr.py --config configs/tv.yaml

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

Key parameters in YAML configs:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunking_use` | `llm` | Triplet extraction method (GPT-4o-mini) |
| `top_m` | `20` | Number of final retrieved results per question |
| `top_k` | `200` | Candidate pool size for retrieval |
| `combine_ent_sim` | `0.93` | Cosine similarity threshold for entity merging |
| `semantic_overlap_sim` | `0.93` | Threshold for semantic deduplication |
| `skip_update_meta` | `false` | If true, disables memory accumulation |

## Hardware Requirements

- **GPU**: CUDA-capable GPU with >= 8GB VRAM (for embedding computation)
- **RAM**: >= 16GB
- **API**: OpenAI API key with access to `gpt-4o-mini`
