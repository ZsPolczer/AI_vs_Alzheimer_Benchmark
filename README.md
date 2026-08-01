# 🧠 AI vs Alzheimer Benchmark (EATEOT Neural Degradation Lab)

Benchmarks cognitive decay in LLMs by surgically degrading transformer weights and
scoring the model on a tiered IQ battery. Models the clinical progression of
dementia via "track profiles" (A1 → Q1, from lucid drift to terminal silence).

## Environment Setup

Requires Python 3.12+ and (optionally) an NVIDIA GPU for CUDA acceleration.
This repo uses [`uv`](https://docs.astral.sh/uv/) for fast, reproducible installs.

```bash
# 1. Install uv (user-level, no sudo needed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 2. Create the virtual environment
uv venv .venv --python 3.12

# 3. Install CUDA-enabled PyTorch (skips GPU? drop --index-url for the CPU build)
uv pip install --python .venv/bin/python torch \
    --index-url https://download.pytorch.org/whl/cu128

# 4. Install everything else
uv pip install --python .venv/bin/python -r requirements.txt

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
> for a single run: `HF_HUB_OFFLINE=0 python your_script.py`.
> ⚠️ Recreating the venv (`uv venv`) drops this hook — re-run setup **step 6** to restore it.

> 💡 Tip: export `HF_TOKEN` to avoid unauthenticated rate limits when downloading models.

## Running the Lab

| Command | What it does |
|---|---|
| `python interactive_lab.py` | Interactive control panel — pick a track profile, run the 5-tier IQ battery, apply flicker/sirens/surge effects. **The main entry point.** |
| `python interactive_lab.py --model Qwen/Qwen2.5-0.5B-Instruct` | Same, with a smaller/faster model (default is 3B, which CPU-offloads on a 4GB GPU). |
| `python brain.py` | One-shot: load model, answer a single IQ question. |
| `python brain_lesion_test.py` | Sever an MLP layer to zero and compare responses. |
| `python alzheimer_benchmark.py` | Baseline vs. 30% vs. 60% synaptic damage on a 2-question benchmark. |
| `streamlit run dashboard.py` | Visual dashboard (radar + bar charts) of degradation results. |
| `python plot_decay.py` | Plot IQ decay curve from `iq_test_results.json`. |
| `python plot_decay.py --reset` | Wipe logged results + chart. |

First run of any script downloads the model weights into `~/.cache/huggingface`.

## Outputs

- `iq_test_results.json` — telemetry log of every IQ battery run
- `iq_decay_curve.png` — generated decay chart
