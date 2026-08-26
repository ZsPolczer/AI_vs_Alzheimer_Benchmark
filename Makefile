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

# uv creates .venv/bin on Unix but .venv/Scripts on Windows — detect the OS,
# and give make a real bash on Windows (recipes use POSIX shell syntax such as
# VAR=val cmd and printf > file).
ifeq ($(OS),Windows_NT)
PYTHON := .venv/Scripts/python.exe
BIN    := .venv/Scripts
# make 4.4 on Windows only honors an absolute shell path: bare names (even
# `bash`) get replaced by its own default (sh.exe), which usually isn't
# installed — make then falls back to cmd.exe and the POSIX recipe syntax
# (VAR=val cmd) breaks. So always hand make an absolute path to Git's bash.
ifneq ($(wildcard C:/Program\ Files/Git/bin/bash.exe),)
SHELL := C:/Program Files/Git/bin/bash.exe
else ifneq ($(wildcard C:/Program\ Files/Git/usr/bin/bash.exe),)
SHELL := C:/Program Files/Git/usr/bin/bash.exe
else
$(error bash not found on Windows. Install Git for Windows (https://git-scm.com/download/win) and re-run make.)
endif
else
PYTHON := .venv/bin/python
BIN    := .venv/bin
SHELL := /bin/bash
endif

# uv — fall back to the user-level install when it's not on PATH (the official
# installer drops it in ~/.local/bin and leaves PATH to the user).
UV ?= uv
ifeq ($(shell command -v '$(UV)' 2>/dev/null),)
UV := $(HOME)/.local/bin/uv
endif

# --- Defaults (override on the command line, e.g. `make lab MODEL=...`) ------
MODEL         ?= Qwen/Qwen2.5-3B-Instruct
QUESTIONNAIRE ?= iq_battery
DRUG          ?=
DOSE          ?= 1.0
STACK         ?=
TRACK         ?= C1
DATA_DIR      ?= outputs

.PHONY: help setup install hook test clean \
        lab lab-small compare compare-mini plot plot-reset dash \
        restore trajectory reserve trip drugreport sensitivity \
        brain lesion benchmark quizzes

help: ## Show all available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# -----------------------------------------------------------------------------
# SETUP
# -----------------------------------------------------------------------------
setup: install hook ## Install package + apply offline hook (first time)

install: ## Install the package editable (deps from pyproject.toml)
	$(UV) pip install --python $(PYTHON) -e .

hook: ## Apply the offline-by-default HF hook + UTF-8 stdio (idempotent; re-run after uv venv)
	@HOOK_DIR=$$($(PYTHON) -c "import site; sp=[p for p in site.getsitepackages() if 'site-packages' in p.lower()]; print(sp[0] if sp else site.getsitepackages()[0])") && \
	printf 'import os\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\n' > "$$HOOK_DIR/offline_env.py" && \
	printf 'import offline_env\n' > "$$HOOK_DIR/offline_env.pth" && \
	printf 'import sys\nfor _s in (sys.stdout, sys.stderr):\n    if _s is not None and hasattr(_s, "reconfigure"):\n        _s.reconfigure(encoding="utf-8", errors="replace")\n' > "$$HOOK_DIR/sitecustomize.py" && \
	echo "[✓] Offline hook + UTF-8 stdio applied to $$HOOK_DIR"

clean: ## Remove bytecode caches
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# -----------------------------------------------------------------------------
# RUNNABLES (console scripts)
# -----------------------------------------------------------------------------
lab: ## Interactive control panel (default 3B model; DRUG/DOSE or STACK for a psychoactive profile)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-lab --model "$(MODEL)" --drug "$(DRUG)" --dose "$(DOSE)" --stack "$(STACK)"

lab-small: ## Interactive lab with the small 0.5B model
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-lab --model Qwen/Qwen2.5-0.5B-Instruct --drug "$(DRUG)" --dose "$(DOSE)" --stack "$(STACK)"

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

restore: ## Dose-response restoration study (IQ vs restore fraction)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-restore

trajectory: ## Run the full A1→Q1 decline trajectory + decay chart
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-trajectory

reserve: ## Cognitive reserve study (IQ vs severity across model sizes)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-reserve

trip: ## Drug dose-response study (IQ vs dose; DRUG=lsd)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-trip --drug "$(DRUG)" --model "$(MODEL)"

drugreport: ## Group telemetry IQ results per drug/stack combo + dose (report + chart)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-drugreport

sensitivity: ## Std-scaled Gaussian perturbation study (IQ + deterioration grade vs ε)
	EATEOT_DATA_DIR=$(DATA_DIR) $(BIN)/eateot-sensitivity --track "$(TRACK)" --questionnaire "$(QUESTIONNAIRE)"

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
