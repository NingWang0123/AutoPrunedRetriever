# AutoPrunedRetriever (APR) — Usage Guide

## Overview

APR is a standalone retrieval-augmented generation system that:
1. Builds a structured Knowledge Graph (KG) from a text corpus using LLM-based triplet extraction
2. Retrieves relevant subgraphs using a 6-signal hybrid retriever
3. Generates answers from compact, indices-only KG prompts
4. Accumulates cross-question memory (past answers become retrievable for future questions)
5. Learns per-question retrieval strategies via DPO (Direct Preference Optimization)

## Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."
```

Hardware: CUDA GPU (>= 8GB VRAM), >= 16GB RAM.

## Quick Start

```bash
cd gpu
python run_apr.py --config configs/stem.yaml
```

Results are saved to `outputs/{dataset}/{dataset}_results.json`.

## Pipeline Walkthrough

### Phase 1: Corpus Ingestion (offline, once per corpus)

When run for the first time, APR:

1. **Chunks** the corpus into overlapping text segments (1200 tokens, 100 overlap)
2. **Extracts triplets** from each chunk via GPT-4o-mini:
   - Input: "Brown bears vary in size. Coastal bears eat salmon..."
   - Output: `("brown bears", "vary in", "size")`, `("coastal bears", "eat", "salmon")`, ...
3. **Builds a meta-codebook** (global KG):
   - Entity dictionary: `["brown bears", "size", "salmon", ...]`
   - Relation dictionary: `["vary in", "eat", "provides", ...]`
   - Edge matrix: `[[0,0,1], [2,1,3], ...]` (head_idx, rel_idx, tail_idx)
   - Embeddings for all entities, relations, and edges (BAAI/bge-large-en-v1.5)
4. **Saves** the codebook to disk (reused on subsequent runs)

### Phase 2: DPO Strategy Learning (optional, enabled by default)

APR trains a lightweight MLP (~500K params) to choose retrieval strategies per question:

1. **Seed phase**: Records first `seed_n` (default: 20) question-answer pairs
2. **Preference pairs**: For each seed question, tries 6 random strategy combinations and scores them with `reward_sbert_inclusive` (cosine similarity to gold answer)
3. **Training**: 5 epochs of DPO loss with AdamW (lr=3e-4, beta=0.1)
4. **Strategy space**: Selects `answers_choice` (how to aggregate past answers) per question

Skip with `--no-dpo` for fixed strategy.

### Phase 3: Question Answering (online, per question)

For each question:

1. **Parse question** into KG triplets → merge into meta-codebook
2. **Retrieve** relevant facts + past answers using 6-signal hybrid retrieval:
   - Semantic similarity (sentence embeddings)
   - Entity overlap (max pairwise cosine)
   - Relation overlap (max pairwise cosine)
   - Structural matching (inverted index on edge IDs)
   - Edge-fine scoring (exact triple match count)
   - Coverage scoring (query entity presence)
   - Two-track RRF fusion → final top-m results
3. **Deduplicate** retrieved results via sentence embedding overlap (threshold 0.93)
4. **Build indices-only prompt**: Only entity names, relation labels, and edge indices — no raw text
5. **Generate answer** via GPT-4o-mini from the structured prompt
6. **Store answer** back into meta-codebook (available for future retrieval)
7. **Prune entities** via KNN clustering (merge near-duplicates, threshold 0.90)

## Configuration Reference

```yaml
# Dataset
dataset: stem                           # Dataset name
corpus_file: ../data/corpus/stem_corpus.json
quest_file: ../data/stem_question.json

# Triplet extraction
chunking_use: llm                       # "llm" for GPT-4o-mini
subchunk_mode: tokens                   # "tokens" or "chars"

# Codebook paths
ini_meta_json: outputs/stem/meta_codebook_llm.json
final_json_path: outputs/stem/stem_results.json

# DPO
seed_n: 20                              # Seed questions for DPO
reward_func: reward_sbert_inclusive      # Reward function
reward_func_mode: non_llm               # No LLM calls for reward

# Retrieval
top_m: 20                               # Final results per question
top_k: 200                              # Candidate pool size

# Entity merging thresholds
combine_ent_sim: 0.93                   # Entity merge cosine threshold
q_combine_sim: 0.93
aft_combine_sim: 0.93
semantic_overlap_sim: 0.93              # Dedup threshold

# Memory
skip_update_meta: false                 # false = accumulate answers in memory
```

## Output Format

```json
{
    "id": "STEM-abc123",
    "question": "How does X relate to Y?",
    "question_type": "Complex Reasoning",
    "generated_answer": "X relates to Y through...",
    "ground_truth": "reference answer",
    "context": "retrieved KG edges as text",
    "input_tokens": 1234,
    "output_tokens": 56,
    "gen_latency_sec": 1.5,
    "retrieval_latency_sec": 0.8,
    "total_latency_sec": 2.3,
    "retrieved_count": 20,
    "model_name": "gpt-4o-mini"
}
```

## Common Options

```bash
# Use a different API key
python run_apr.py --config configs/stem.yaml --api-key sk-...

# Disable DPO (use fixed strategy)
python run_apr.py --config configs/stem.yaml --no-dpo
```

## How Memory Works

APR maintains a persistent meta-codebook across questions:

```
Question 1 → retrieve facts → answer → store answer edges
Question 2 → retrieve facts + Q1 answer edges → answer → store
Question 3 → retrieve facts + Q1 + Q2 answer edges → answer → store
...
```

Each stored answer is parsed into KG triplets, so future retrieval can match on entity/relation overlap. This enables **cross-question reasoning**: knowledge from one answer becomes context for later questions.

To disable memory accumulation, set `skip_update_meta: true` in the config.

## Customizing for New Datasets

1. Prepare corpus JSON:
```json
[
    {"text": "Document 1 text...", "id": "doc1"},
    {"text": "Document 2 text...", "id": "doc2"}
]
```

2. Prepare questions JSON:
```json
[
    {
        "question": "How does X relate to Y?",
        "answer": "reference answer",
        "question_type": "Complex Reasoning",
        "id": "q001"
    }
]
```

3. Create a YAML config pointing to your files and run.
