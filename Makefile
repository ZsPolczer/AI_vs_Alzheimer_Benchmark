# =============================================================================
# EATEOT Neural Degradation Lab — Makefile
#
# Quick start:
#   make setup        # create venv deps + offline hook (first time only)
#   make lab          # interactive lab (default 3B model)
#   make compare      # cross-model IQ comparison report
#   make plot         # decay curve chart
#   make dash         # streamlit dashboard
#
# Common overrides:
#   make lab MODEL=Qwen/Qwen2.5-0.5B-Instruct
#   make compare QUESTIONNAIRE=iq_battery_mini
# =============================================================================

SHELL := /bin/bash
PYTHON := .venv/bin/python
BIN    := .venv/bin
UV    := uv

# --- Defaults (override on the command line, e.g. `make lab MODEL=...`) ------
MODEL         ?= Qwen/Qwen2.5-3B-Instruct
QUESTIONNAIRE ?= iq_battery
DATA_DIR      ?= outputs

.PHONY: help setup install hook test clean \
        lab lab-small compare compare-mini plot plot-reset dash \
        brain lesion benchmark quizzes

help: ## Show all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
setup: install hook ## Install package + apply offline hook (first time)

install: ## Install the package editable (deps from pyproject.toml)
	$(UV) pip install --python $(PYTHON) -e .

hook: ## Apply the offline-by-default HF hook (idempotent; re-run after uv venv)
	@HOOK_DIR=$$($(PYTHON) -c "import site; print(site.getsitepackages()[0])") && \
	printf 'import os\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\n' > "$$HOOK_DIR/offline_env.py" && \
	printf 'import offline_env\n' > "$$HOOK_DIR/offline_env.pth" && \
	echo "[✓] Offline hook applied to $$HOOK_DIR"

clean: ## Remove bytecode caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# -----------------------------------------------------------------------------
# RUNNABLES (console scripts)
# -----------------------------------------------------------------------------
lab: ## Interactive control panel (default 3B model)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-lab --model "$(MODEL)"

lab-small: ## Interactive lab with the small 0.5B model
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-lab --model Qwen/Qwen2.5-0.5B-Instruct

compare: ## Run the IQ battery on all three models + save report/chart
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-compare --questionnaire "$(QUESTIONNAIRE)"

compare-mini: ## Run the mini questionnaire on all three models
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-compare --questionnaire iq_battery_mini

plot: ## Plot IQ decay curve from telemetry
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-plot

plot-reset: ## Wipe telemetry log + chart
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-plot --reset

dash: ## Streamlit dashboard (radar + bar charts)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/streamlit run apps/dashboard.py

quizzes: ## List available questionnaires
	$(PYTHON) -c "from eateot import list_batteries; print('\n'.join(list_batteries()))"

# -----------------------------------------------------------------------------
# ONE-OFF SCRIPTS
# -----------------------------------------------------------------------------
brain: ## One-shot: answer a single IQ question
	$(PYTHON) scripts/brain.py

lesion: ## Sever an MLP layer and compare responses
	$(PYTHON) scripts/brain_lesion_test.py

benchmark: ## Baseline vs 30% vs 60% synaptic damage
	$(PYTHON) scripts/alzheimer_benchmark.py

# -----------------------------------------------------------------------------
# TESTS
# -----------------------------------------------------------------------------
test: ## Run the unit test suite
	$(PYTHON) -m unittest discover -s tests -t .
