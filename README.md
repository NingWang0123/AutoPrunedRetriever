# AutoPrunedRetriever

[![View Workflow](https://img.shields.io/badge/View%20Workflow-PDF-blue)](./Workflow.pdf)

Suggested Python: 3.10.12

## Reproducibility Steps

### Step 1: Download this repo and cd to it

```
git clone https://github.com/NingWang0123/relational_graph_llm
cd relational_graph_llm
```

### Step 2: Install dependencies (for Mac users)

```
python3 -m venv venv_rgl
source venv_rgl/bin/activate

pip install -r req.txt

cd/py_files

$env:CHUNKING_API   = "" # your api key
make run CONFIG=configs/tv_cr_llm.yaml
```

## 🧪 Evaluation

We use the same **LLM Judge** from the official  
[GraphRAG-Benchmark](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)  
to ensure APR is evaluated under identical metrics and scoring rules.

### 1. Download the Evaluation Pipeline
```

mkdir -p py_files
git -C py_files clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git

```