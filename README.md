# 🧠 AI vs Alzheimer Benchmark (EATEOT Neural Degradation Lab)

Benchmarks cognitive decay in LLMs by surgically degrading transformer weights and
scoring the model on a tiered IQ battery. Models the clinical progression of
dementia via "track profiles" (A1 → Q1, from lucid drift to terminal silence).

## Project Layout

```
src/eateot/       Core package (config, engine, battery, telemetry, runner)
apps/             Runnable entry points (installed as console scripts)
scripts/          One-off experiments
tests/            Unit tests (python -m unittest discover -s tests)
outputs/          Generated artifacts: telemetry, charts, reports (gitignored)
```

## Environment Setup

Requires Python 3.12+ and (optionally) an NVIDIA GPU for CUDA acceleration.
This repo uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible installs.

```bash
# 1. Install uv (user-level, no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Create the virtual environment
uv venv .venv --python 3.12

# 3. Install CUDA-enabled PyTorch (no GPU? drop --index-url for the CPU build)
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Install the package (deps come from pyproject.toml)
uv pip install --python .venv/bin/python -e .

# 5. Activate for daily use
source .venv/bin/activate

# 6. Apply the offline-by-default hook (idempotent; re-run after `uv venv` rebuilds)
HOOK_DIR=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
printf 'import os\nos.environ.setdefault("HF_HUB_OFFLINE", "1")\n' > "$HOOK_DIR/offline_env.py"
printf 'import offline_env\n' > "$HOOK_DIR/offline_env.pth"
```

**Verified environment:** torch 2.11.0+cu128 · transformers 5.14.1 · streamlit 1.60 ·
pandas 2.2+ · plotly 6.9 · matplotlib 3.11 · accelerate

> Note: the installed env originally had pandas 3.0.5; the pin was tightened to `>=2.2,<3`
> for reproducibility, so fresh installs resolve to the latest 2.x line.

> 🔌 **Offline by default:** this venv ships a site-packages startup hook (`offline_env.pth`)
> that sets `HF_HUB_OFFLINE=1` for every run — scripts never hit the network; all models
> resolve from the local cache (`~/.cache/huggingface/hub`). To opt back into downloads
> for a single run: `HF_HUB_OFFLINE=0 eateot-lab`.
> ⚠️ Recreating the venv (`uv venv`) drops this hook — re-run setup **step 6** to restore it.

> 💡 Tip: export `HF_TOKEN` to avoid unauthenticated rate limits when downloading models.

## Running the Lab

### Makefile quick start

The repo ships a `Makefile` with shortcuts for every workflow:

```bash
make setup          # install package + offline hook (first time only)
make lab            # interactive lab (default 3B model)
make lab MODEL=Qwen/Qwen2.5-0.5B-Instruct   # different model
make compare        # cross-model IQ comparison report + chart
make compare QUESTIONNAIRE=iq_battery_mini  # use a different questionnaire
make plot           # decay curve chart
make plot-reset     # wipe telemetry + chart
make dash           # Streamlit dashboard
make quizzes        # list available questionnaires
make test           # run the unit test suite
make help           # show all targets
```

Installed console scripts (from anywhere, after step 4–5):

| Command | What it does |
|---|---|
| `eateot-lab` | Interactive control panel — pick a track profile, run the IQ battery, apply flicker/sirens/surge effects. **The main entry point.** |
| `eateot-lab --model Qwen/Qwen2.5-0.5B-Instruct` | Same, with a smaller/faster model (default is 3B, which CPU-offloads on a 4GB GPU). |
| `eateot-lab --questionnaire iq_battery_mini` | Run a different questionnaire (see **Questionnaires** below). |
| `eateot-compare` | Run the A1 IQ battery on all three models (0.5B/1.5B/3B) and save a comparison report + chart. |
| `eateot-compare --models 3B` | Re-benchmark a subset; others come from existing telemetry. |
| `eateot-plot` | Plot IQ decay curve from `outputs/iq_test_results.json`. |
| `eateot-plot --reset` | Wipe logged results + chart. |
| `streamlit run apps/dashboard.py` | Visual dashboard (radar + bar charts) of degradation results. |

## Questionnaires

All question sets are **versioned YAML files** under `config/questionnaires/` —
editing questions never requires touching Python code:

| File | Used by |
|---|---|
| `iq_battery.yaml` | The default 5-tier IQ battery (`eateot-lab`, `eateot-compare`). |
| `iq_battery_mini.yaml` | A tiny 2-question example questionnaire — try `--questionnaire iq_battery_mini`. |
| `presets.yaml` | Quick prompt scenarios in the interactive lab menu. |
| `brain_benchmark.yaml` | The 2-question benchmark in `scripts/alzheimer_benchmark.py` (and the one-shot scripts). |

**Start the system with a different questionnaire:**

```bash
eateot-lab --questionnaire iq_battery_mini
eateot-compare --questionnaire iq_battery_mini
```

**Bring your own questionnaires:** point `EATEOT_QUESTIONNAIRE_DIR` at a custom
folder containing your own YAML files (see `config/questionnaires/iq_battery.yaml`
for the schema), then select them by filename:

```bash
EATEOT_QUESTIONNAIRE_DIR=~/my-quizzes eateot-lab --questionnaire my_quiz
```

The questionnaire name is recorded in telemetry (`outputs/iq_test_results.json`), so
comparison reports only merge results from the same questionnaire.

> ℹ️ This repo is designed for an **editable install** (`uv pip install -e .`), which is
> how `config/` is resolved. For a real (non-editable) wheel install, point
> `EATEOT_QUESTIONNAIRE_DIR` at your checkout's `config/questionnaires/` folder.

Same entry points run from source without installing (`python apps/lab.py`, etc.).

One-off experiments (from source):

| Command | What it does |
|---|---|
| `python scripts/brain.py` | One-shot: load model, answer a single IQ question. |
| `python scripts/brain_lesion_test.py` | Sever an MLP layer to zero and compare responses. |
| `python scripts/alzheimer_benchmark.py` | Baseline vs. 30% vs. 60% synaptic damage on a 2-question benchmark. |

First run of any script downloads the model weights into `~/.cache/huggingface`.

## Outputs

All generated artifacts land in `outputs/` (gitignored):

- `iq_test_results.json` — telemetry log of every IQ battery run
- `iq_decay_curve.png` — generated decay chart
- `model_comparison.md` / `.json` / `.png` — cross-model comparison report

Override the location with the `EATEOT_DATA_DIR` env var (e.g. `EATEOT_DATA_DIR=/tmp/eateot eateot-lab`).

## Tests

```bash
python -m unittest discover -s tests -t .
```

(`-t .` is required so test modules import as `tests.*` — this runs the `sys.path`
bootstrap in `tests/__init__.py` when running from source without installing.)
