# AutoPruned Layer (APL) — Usage Guide

## Overview

APL is a **plug-in enhancement** that wraps any existing RAG system. It takes a baseline RAG system's retrieved context and question, re-structures the context into KG format, and generates a new answer using structured prompting with cross-question memory.

APL does **not** need access to the original corpus — it works entirely from the baseline RAG's retrieved output.

## Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Hardware: CUDA GPU (>= 8GB VRAM), >= 16GB RAM.

## Quick Start

```bash
cd gpu
python run_apl.py --predictions path/to/baseline_predictions.json \
                  --output path/to/apl_results.json
```

## Input Format

APL expects a JSON array of baseline RAG predictions. Each entry must have:

```json
[
    {
        "id": "q_001",
        "question": "How does the grade of follicular lymphoma influence treatment?",
        "answer": "reference answer (optional, for evaluation)",
        "question_type": "Complex Reasoning",
        "retrieved_contexts": [
            "Follicular lymphoma (FL) starts in the germinal centers...",
            "Grade 3B is treated as DLBCL..."
        ]
    }
]
```

**Required fields:**
- `question`: The question text
- `retrieved_contexts`: List of text passages retrieved by the baseline RAG system

**Optional fields:**
- `id`: Question identifier
- `answer`: Ground truth (for evaluation only, not used during generation)
- `question_type`: Category label

## Pipeline Walkthrough

For each question, APL performs these steps:

### Step 1: Parse RAG Context → KG

The baseline RAG's retrieved text is parsed into structured triplets:

```
Input: "Grade 3B is treated as DLBCL. FL grade 3A might be treated as classic FL."

Extracted edges:
  ("Grade 3B", "treated as", "DLBCL")
  ("FL grade 3A", "might be treated as", "classic FL or DLBCL")
```

This creates a **facts codebook** for this question.

### Step 2: Parse Question → KG

The question itself is also parsed into triplets and merged into APL's internal meta-codebook.

### Step 3: Retrieve Past Answers from Memory

APL searches its accumulated answer memory for relevant past Q&A edges using the same 6-signal retrieval as APR.

On the first question, memory is empty. By the 3rd question, memory contains edges from answers to questions 1 and 2.

### Step 4: Semantic Deduplication

Redundant retrieved results are clustered by sentence embedding similarity (threshold 0.9) and deduplicated.

### Step 5: Build Indices-Only Prompt

The question edges, RAG context facts, and past answer edges are merged into a compact structured prompt:

```json
{
    "e": ["Grade 3B", "DLBCL", "FL grade 3A", "classic FL", ...],
    "r": ["treated as", "might be treated as", ...],
    "questions(...)": [...],
    "facts(...)": [...],
    "given knowledge(...)": [...]
}
```

### Step 6: Generate Answer

GPT-4o-mini generates the answer from the structured prompt.

### Step 7: Store in Memory

The answer is parsed into KG triplets and stored in the meta-codebook for future retrieval.

### Step 8: Entity Pruning

Near-duplicate entities are merged via KNN clustering to keep memory compact.

## Command-Line Options

```bash
python run_apl.py \
    --predictions baseline.json \     # Required: baseline RAG predictions
    --output apl_results.json \       # Required: output path
    --api-key sk-... \                # Optional: OpenAI key (or use env var)
    --top-m 20 \                      # Optional: top results from memory
    --top-k 200 \                     # Optional: candidate pool size
    --combine-ent-sim 0.93 \          # Optional: entity merge threshold
    --compress-rate 0.2               # Optional: context compression ratio
```

## Output Format

```json
[
    {
        "id": "q_001",
        "question": "How does the grade of follicular lymphoma influence treatment?",
        "question_type": "Complex Reasoning",
        "generated_answer": "The grade determines aggressiveness...",
        "ground_truth": "reference answer",
        "context": "structured KG edges used for generation",
        "total_latency_sec": 2.3
    }
]
```

## Supported Baseline RAG Systems

APL works with any RAG system that produces retrieved text passages. Tested with:

- **G-Reasoner** (graph foundation model)
- **PathRAG** (path-based graph retrieval)
- **HippoRAG2** (chunk-based retrieval)
- **LightRAG** (KG-entity retrieval)
- Any custom RAG that outputs `retrieved_contexts`

## How APL Differs from APR

| Aspect | APR | APL |
|--------|-----|-----|
| **Input** | Raw corpus + questions | Baseline RAG predictions |
| **Corpus access** | Required | Not required |
| **Graph source** | Built from corpus via LLM | Parsed from RAG's retrieved text |
| **Memory** | Facts + past Q&A | Past answers only |
| **DPO** | Yes (strategy learning) | No (fixed strategy) |
| **Use case** | Full RAG replacement | Enhancement layer |

## Preparing Baseline Predictions

### From G-Reasoner / GFM-RAG

If you have GFM-RAG prediction files, convert them:

```python
import json

with open("gfm_predictions.json") as f:
    gfm = json.load(f)

apl_input = []
for entry in gfm:
    apl_input.append({
        "id": entry.get("id", ""),
        "question": entry["question"],
        "answer": entry.get("ground_truth", ""),
        "question_type": entry.get("question_type", ""),
        "retrieved_contexts": [entry.get("context", "")]
    })

with open("gfm_for_apl.json", "w") as f:
    json.dump(apl_input, f, indent=2)
```

### From PathRAG

```python
apl_input = []
for entry in pathrag_results:
    apl_input.append({
        "id": entry.get("id", ""),
        "question": entry["question"],
        "answer": entry.get("ground_truth", ""),
        "retrieved_contexts": entry.get("retrieved_passages", [])
    })
```

## Tips

- **Question ordering matters**: APL builds memory sequentially. If your questions have a natural topic order (e.g., all liver cancer questions together), memory accumulation is more effective.
- **compress_rate**: Lower values (e.g., 0.2) produce more compact prompts. Increase if answers are too brief.
- **First few questions**: Memory is empty, so APL relies entirely on the baseline RAG's context. The benefit grows as more Q&A pairs accumulate.
- **Large baseline contexts**: APL benefits most when the baseline retrieves verbose passages (e.g., PathRAG's 23-30K tokens). The KG re-parsing compresses this dramatically.
