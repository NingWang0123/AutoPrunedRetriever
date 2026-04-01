# ----------- Configurable variables -----------
VENV     := .venv
PY       := $(VENV)/bin/python
PIP      := $(VENV)/bin/pip
ENTRY    := pipeline_for_autoprunedretriever.py

# Default config (override: make run CONFIG=configs/local_med.yaml)
CONFIG   ?= configs/multi.yaml

# Default dataset list for run-all (override: make run-all DATASETS="medical biology")
DATASETS ?= medical

REQ      ?= requirements.txt

# ----------- API keys (set here OR via .env) -----------
# You can put secrets here (not recommended for shared repos),
# or better: create a ".env" file in the repo root; this Makefile auto-loads it.
OPENAI_API_KEY ?=
HF_TOKEN       ?=
CHUNKING_API   ?=

# Export to subprocesses (python sees them)
export OPENAI_API_KEY
export HF_TOKEN
export CHUNKING_API

# ----------- Auto-load .env if present -----------
# Any KEY=VALUE lines in .env are included and exported.
ifneq (,$(wildcard .env))
include .env
export $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' .env)
endif

# ----------- Phony targets -----------
.PHONY: help venv install run run-cfg run-ds-% run-all lint fmt freeze print-env dotenv-init dotenv-save clean

help:
	@echo "Targets:"
	@echo "  make install                 Create venv and install requirements"
	@echo "  make run CONFIG=...          Run once with a given config (default: $(CONFIG))"
	@echo "  make run-ds-<name>           Run with -d <name> using $(CONFIG) (templated configs)"
	@echo "  make run-all DATASETS='a b'  Run for multiple datasets (uses $(CONFIG))"
	@echo "  make print-env               Show which API keys are set (masked)"
	@echo "  make dotenv-init             Create a .env.example stub"
	@echo "  make dotenv-save             Write current API vars into .env (overwrites!)"
	@echo "  make lint / make fmt         Ruff lint/format (optional)"
	@echo "  make freeze                  Pin pip versions to requirements.txt"
	@echo "  make clean                   Remove caches and venv"

venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && $(PIP) install -U pip && \
	if [ -f "$(REQ)" ]; then $(PIP) install -r $(REQ); fi

run:  ## Run once with CONFIG (override via make run CONFIG=configs/foo.yaml)
	. $(VENV)/bin/activate && $(PY) $(ENTRY) -c $(CONFIG)

# Same as "run" but named explicitly for clarity
run-cfg:
	. $(VENV)/bin/activate && $(PY) $(ENTRY) -c $(CONFIG)

# Run a specific dataset key using the same CONFIG (requires your script's -d/--dataset flag)
run-ds-%:
	. $(VENV)/bin/activate && $(PY) $(ENTRY) -c $(CONFIG) -d $*

# Loop over datasets in DATASETS
run-all:
	@for ds in $(DATASETS); do \
	  echo ">>> Running $$ds with $(CONFIG)"; \
	  $(MAKE) --no-print-directory run-ds-$$ds || exit 1; \
	done

lint:
	. $(VENV)/bin/activate && ruff check .

fmt:
	. $(VENV)/bin/activate && ruff format .

freeze:
	. $(VENV)/bin/activate && pip freeze > $(REQ)

print-env:
	@bash -c '\
	mask(){ v="$$1"; [ -n "$$v" ] && echo "$${v:0:3}***$${v: -3}" || echo ""; }; \
	echo "OPENAI_API_KEY=$$(mask "$$OPENAI_API_KEY")"; \
	echo "HF_TOKEN=$$(mask "$$HF_TOKEN")"; \
	echo "CHUNKING_API=$$(mask "$$CHUNKING_API")"; \
	'

dotenv-init:
	@[ -f .env.example ] || cat > .env.example <<'EOF'
# Copy to .env and fill in your secrets. Do NOT commit .env.
OPENAI_API_KEY=
HF_TOKEN=
CHUNKING_API=
EOF
	@echo "Wrote .env.example (safe to commit). Create .env and fill secrets."

# WARNING: overwrites .env with current in-memory values
dotenv-save:
	@echo "Writing .env from current Makefile/env values..."
	@{ \
	  echo "OPENAI_API_KEY=$$OPENAI_API_KEY"; \
	  echo "HF_TOKEN=$$HF_TOKEN"; \
	  echo "CHUNKING_API=$$CHUNKING_API"; \
	} > .env
	@echo "Done. (.env is NOT safe to commit.)"

clean:
	@rm -rf __pycache__ .pytest_cache .ruff_cache .mypy_cache $(VENV)
