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
make lab DRUG=lsd DOSE=2.0                  # trip the lab on LSD (see Drug catalog)
make lab STACK="lsd@1.0,thc@0.5"            # deploy a drug COMBO from the CLI
make compare        # cross-model IQ comparison report + chart
make compare QUESTIONNAIRE=iq_battery_mini  # use a different questionnaire
make plot           # decay curve chart
make plot-reset     # wipe telemetry + chart
make dash           # Streamlit dashboard
make restore        # dose-response restoration study (IQ vs treatment dose)
make trajectory     # full A1→Q1 decline trajectory + decay chart
make reserve        # cognitive reserve study (IQ vs severity across models)
make trip DRUG=lsd  # drug dose-response study (IQ vs dose on a track)
make drugreport     # group all telemetry IQ results per drug/stack + dose
make sensitivity    # std-scaled Gaussian perturbation study (IQ + grade vs ε)
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
| `eateot-restore` | Dose-response restoration study: fix a track (default G1), then run the battery at increasing restore fractions (treatment dose) and plot the IQ-vs-dose curve. |
| `eateot-trajectory` | Run the full A1→Q1 decline trajectory (all 17 profiles) in one session + render the decay chart. |
| `eateot-reserve` | Cognitive reserve study: IQ vs lesion severity (decay multiplier sweep) across all three model sizes — does bigger mean more resilient? |
| `eateot-trip --drug lsd` | Drug dose-response study: sweep a drug's dose on a fixed track and plot the IQ-vs-dose curve (doses default to 0→1.25× the drug's `dose_cap`). |
| `eateot-drugreport` | Group every drug telemetry IQ result by (drug/stack label, dose) into `drug_report.md/.json/.png` — reads the drug-domain log (`drug_test_results.json`) with sober-baseline rows from the Alzheimer log. Filters: `--drug lsd`, `--model 0.5B`, `--track C1`, `--questionnaire`, `--min-runs N`. |
| `eateot-sensitivity` | Perturb weights with std-scaled Gaussian noise (Ẇ = W + ε·σ_W·Z), sweep ε on a log grid, and record IQ **plus the numeric deterioration grade** (0–100) at each level with mean ± std across seeds — tests the monotonic-degradation hypothesis. |
| `eateot-lab --drug lsd --dose 2.0` | Apply a psychoactive perturbation profile on top of any track in the interactive session (25 drugs in `config/drugs.yaml`, see **Drug catalog**). |
| `eateot-lab --stack "lsd@1.0,thc@0.5"` | Deploy a drug **combo** — resolve the stack and run every battery under it (mutually exclusive with `--drug`). |
| `eateot-lab --seed 42` | Any lab battery run becomes reproducible (lesion + sampling). |
| `eateot-lab --epsilon 0.01` | Apply std-scaled Gaussian perturbation (Ẇ = W + ε·σ_W·Z) on top of any track — see **Sensitivity study** below. |
| `eateot-sensitivity` | Std-scaled Gaussian perturbation study: IQ + **deterioration grade** vs ε (monotonicity test), report + chart. |
| `streamlit run apps/dashboard.py` | Visual dashboard (radar + bar charts) of degradation results. |

Press **`[P]` 💊 DEPLOY DRUG EXPERIMENT** inside the interactive lab to build a
drug from dependent menus — pick a *class* (hallucinogen, dissociative, …), then a
drug within that class, then a dose — and optionally stack more drugs into a combo
before running the battery. The deployed spec is printed (resolved primitives,
subnetwork, layer window) and becomes the session's active drug.

## Experimental: progressive in-generation degradation

Press **`[G]` 📉 PROGRESSIVE DEGRADATION** in the lab menu (or call
`BrainLabEngine.run_progressive_inference` directly). Run it with
`eateot-lab --monitor` to also trace the model's **metacognitive language** as
the response collapses: each response window gets a live marker profile
(cognitive self-reference, doubt/hedging, denial/over-compensation,
confabulated recollection, repetition loops) plus a 0–100 "metacognitive
coherence" score, and every window is logged to
`outputs/metacognition_<timestamp>.jsonl` (model, track, ramp intensity,
marker counts, coherence, raw text) for post-hoc charting. The scan is pure
text heuristics — zero extra inference cost. The model itself never "sees"
its own decay (no introspection path); the monitor measures the linguistic
footprint of self-monitoring, which is what actually shifts as the hidden
states corrupt. Unlike the track profiles,
which corrupt the **stored weights** before generation, this corrupts the
**live hidden states while the model is generating** — so the answer visibly
degrades as it streams: early tokens come out near-clean, and by the end of a
long response the representation is scaled down and drenched in noise.

A forward hook on the decoder's hidden stem applies, per token:

```
intensity(progress) = 0                 for progress ≤ ramp_mid   (clean zone)
                       smooth 0 → 1 ramp for progress > ramp_mid   (ends at 1.0)
hidden ← hidden · (1 − (1 − scale_min)·intensity) + ε · σ_hidden · Z · intensity
```

`progress` is the token position ÷ `max_new_tokens`, `σ_hidden` the hidden
state's own std (so `ε` is dimensionless), and `Z ~ N(0, 1)`. Knobs: `epsilon`
(noise strength, default 0.5), `scale_min` (hidden magnitude at full mayhem,
default 0.2), `ramp_mid` (fraction of the generation kept near-intact, default
0.35), `ramp_k` (sharpness of the ramp once it begins, default 2.5). The hook
is removed when generation finishes, so subsequent clean runs are unaffected,
and the stored weights are never modified — no `restore_clean_state` needed.

## Drug catalog & combos

The psychoactive catalog lives in `config/drugs.yaml` (25 drugs across 8 classes:
hallucinogen, dissociative, stimulant, depressant, cannabinoid, deliriant,
enhancer, placebo). Every drug maps a *dose* to engine primitives via
`eateot.drugs.resolve_drug` — noise, attention scatter, logit noise, flicker,
context masking, temperature, verbosity, and more (see the file header for the
full schema and primitive contract). `dose_cap` is advisory: doses above it
resolve but are flagged (`dose_exceeds_cap`).

**Combos / stacks** combine drugs at runtime — `eateot.drugs.resolve_stack`
merges specs (additive primitives sum, temperature/repetition-penalty deltas sum,
weight loss combines into one scale, the layer window becomes the union, and the
subnetwork is the shared value, else `all`):

```python
from eateot import resolve_stack
spec = resolve_stack([{"drug": "lsd", "dose": 1.0}, {"drug": "thc", "dose": 0.5}])
# spec["name"] == "lsd@1+thc@0.5" — recorded in telemetry as the run's drug
```

Stack specs also have a compact string form (`parse_stack`):
`"lsd@1.0,thc@0.5"` (a bare name means dose 1.0). Override the whole catalog
per-run with `EATEOT_DRUGS_FILE=/path/to/drugs.yaml`.

**Two experiment domains, two telemetry logs.** Runs made under a drug or
stack (`--drug`, `--stack`, lab `[P]`, `eateot-trip`) auto-log to
`drug_test_results.json`; all other runs (track lesions, severity sweeps,
restore curves, clean baselines) log to `iq_test_results.json`. The lab menu
shows which domain the session is writing to, and the two worlds never mix —
`eateot-plot`/`compare`/`reserve` read only Alzheimer data, while
`eateot-drugreport` reads only drug data (sober baselines come from the
Alzheimer log).

## Questionnaires

All question sets are **versioned YAML files** under `config/questionnaires/` —
editing questions never requires touching Python code:

| File | Used by |
|---|---|
| `iq_battery.yaml` | The default 6-tier IQ battery (`eateot-lab`, `eateot-compare`) — categorical → numeric → counterfactual → relational → **spatial reasoning** → abstract set logic. |
| `visual_battery.yaml` | Visual-cortex probes (ASCII-art rendering, spatial layout, mental rotation, hallucinated scenes, 3D counting) — try `--questionnaire visual_battery` with a hallucinogen like `lsd`/`dmt`. |
| `clinical_battery.yaml` | MMSE/MoCA-inspired clinical battery (orientation, registration, serial-7s, **animal fluency**, naming, delayed + story recall) — try `--questionnaire clinical_battery`. |
| `language_battery.yaml` | Aphasia-focused battery (**phonemic fluency**, tool naming, word definition, abstract similarities, proverb interpretation, comprehension) — try `--questionnaire language_battery`. |
| `executive_battery.yaml` | Frontal-lobe battery (digit span fwd/backward, sequencing, Stroop, rule switching, reverse alphabet, error monitoring) — try `--questionnaire executive_battery`. |
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

- `iq_test_results.json` — Alzheimer-domain telemetry (every non-drug IQ battery run)
- `drug_test_results.json` — drug-domain telemetry (runs made under a drug/stack)
- `iq_decay_curve.png` — generated decay chart
- `model_comparison.md` / `.json` / `.png` — cross-model comparison report
- `restoration_report.md` / `.json` / `restoration_curve.png` — dose-response restoration study
- `trajectory_report.md` / `trajectory_decay.png` — full A1→Q1 decline trajectory
- `reserve_report.md` / `.json` / `reserve_curve.png` — cognitive reserve study
- `trip_report.md` / `.json` / `trip_curve.png` — drug dose-response (trip) study
- `drug_report.md` / `.json` / `drug_report.png` — grouped telemetry report per drug/stack combo + dose
- `sensitivity_report.md` / `.json` / `sensitivity_decay.png` — std-scaled Gaussian perturbation study (IQ + deterioration grade vs ε)

Override the location with the `EATEOT_DATA_DIR` env var (e.g. `EATEOT_DATA_DIR=/tmp/eateot eateot-lab`).

## Deterioration grade & sensitivity study

Every battery run now reports a **deterioration grade** — a single 0–100 number
capturing how degraded the generated answer is (0 = pristine, 100 = fully
degraded). It combines three signals per question:

- **correctness** — fraction of ground-truth anchors matched (50% weight)
- **repetition** — content-word type-token ratio (25% weight)
- **perseveration** — consecutive-loop detector (25% weight)

**Echolalia is rejected outright.** A response that merely reproduces the
question instead of answering (a classic degradation failure) scores 0 points
with status `FAILED (Echolalia / Prompt Echo)` and a deterioration grade of
100 — even when the question text itself contains anchor words (e.g. "State
Yes or No" must never match the `yes` anchor). Anchored questions also award
finer-grained accuracy than binary group hits: a group is fully matched when
any synonym appears verbatim, otherwise it earns fractional credit scaled by
how many of the synonym's content words are present (≥50% coverage), so
accuracy spans values like 0/25/33/50/67/75/100 instead of coarse jumps.

An optional clean baseline can be supplied to `eateot.battery.grade_deterioration`
(`clean_response=`), which folds in content-word bigram overlap as an extra
20% weight so the grade measures deterioration *relative to* the undegraded
answer. The grade is printed in the lab report card, stored per tier in
telemetry (`deterioration_grade`), and averaged into the run summary.

**Std-scaled Gaussian perturbation** (the sensitivity method):

```
Ẇ = W + ε · σ_W · Z
```

with `Z ~ N(0, 1)` i.i.d. and `σ_W` the standard deviation of the weight tensor
`W` — so ε is dimensionless and comparable across layers. Apply it on top of
any track with `eateot-lab --epsilon 0.01`, or run the dedicated study that
sweeps ε on a log grid (`1e-4 → 1e-1`), repeats each level across seeds for
error margins, and checks whether performance degrades monotonically:

```bash
eateot-sensitivity                       # default grid, 3B model, CLEAN track
make sensitivity
```

**Recommended external benchmarks** for a full sensitivity profile (wire your
own YAML questionnaires via `EATEOT_QUESTIONNAIRE_DIR`):

| Kind | Benchmarks |
|---|---|
| Factual recall / knowledge-intensive | TriviaQA, MMLU-Knowledge, LAMA |
| Structural / linguistic control | BLiMP (syntax), CoLA (grammaticality) |

The hypothesis being tested — steeper degradation on knowledge tasks than on
structural ones — is exactly what the monotonic check in
`sensitivity_report.md` surfaces per questionnaire.

## Tests

```bash
python -m unittest discover -s tests -t .
```

(`-t .` is required so test modules import as `tests.*` — this runs the `sys.path`
bootstrap in `tests/__init__.py` when running from source without installing.)
