# AutoPrunedRetriever

[![View Workflow](https://img.shields.io/badge/View%20Workflow-PDF-blue)](./Workflow.pdf)

Suggested Python: 3.10.12

## Reproducibility Steps

### 1) Clone this repo
```bash
git clone https://github.com/NingWang0123/relational_graph_llm
cd relational_graph_llm
````

### 2) Create & activate a virtual environment

#### macOS / Linux (bash/zsh)

```bash
python -m venv venv_rgl
source venv_rgl/bin/activate
```

#### Windows (PowerShell)

```powershell
python -m venv venv_rgl
.\venv_rgl\Scripts\Activate.ps1
```

### 3) Install base dependencies

```bash
pip install -r requirements.txt
```

### 4) Install PyTorch (choose ONE)

#### Option A — CPU only (simple, works everywhere)

```bash
pip install torch==2.9.1
```

#### Option B — GPU (CUDA) build

> Adjust the CUDA tag if needed (e.g., `cu121`, `cu124`). Example below uses `cu121`.

**macOS/Linux (bash/zsh):**

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.9.1
```

**Windows (PowerShell):**

```powershell
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.9.1
```

### 5) Get GraphRAG-Benchmark under `py_files/` (for evaluation only)

```bash
mkdir -p py_files
git -C py_files clone https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.git
```

### 6) (Optional) Set API key for chunking

#### macOS / Linux (bash/zsh)

```bash
export CHUNKING_API=""   # your api key
```

#### Windows (PowerShell)

```powershell
$env:CHUNKING_API = ""   # your api key
```

### 7) Run

```bash
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

### 2. Prepare Model Output Files

Place all APR prediction files under:

```
configs/outputs/<dataset>/<run_name>.json
```

Each file must follow the GraphRAG-Bench format:

```json
{
  "id": "Q-1",
  "question": "...",
  "predicted_answer": "...",
  "reference_answer": "..."
}
```

### 3. Run Evaluation

```
python py_files/GraphRAG-Benchmark/Evaluation/generation_eval.py \
    --config configs/outputs/<dataset>/<run_name>.json \
    --embedding_model BAAI/bge-large-en-v1.5
```
