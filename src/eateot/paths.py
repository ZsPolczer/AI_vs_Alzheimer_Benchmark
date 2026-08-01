"""Path resolution.

Every runtime artifact (telemetry, charts, comparison reports) resolves
through this module, so scripts work regardless of the current working
directory and tests can point ``EATEOT_DATA_DIR`` at a temp folder.
"""

import os
from pathlib import Path

# Project root: <repo>/src/eateot/paths.py -> parents[2] is the repo root.
# NOTE: this assumes a source/editable install (README uses `uv pip install -e .`).
# For a real (non-editable) wheel install, use EATEOT_DATA_DIR to set the output
# directory explicitly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data/output directory. Override per-run: EATEOT_DATA_DIR=/tmp/foo
DATA_DIR = Path(os.environ.get("EATEOT_DATA_DIR", str(PROJECT_ROOT / "outputs")))

LOG_FILE = str(DATA_DIR / "iq_test_results.json")
IMG_FILE = str(DATA_DIR / "iq_decay_curve.png")
COMPARISON_MD = str(DATA_DIR / "model_comparison.md")
COMPARISON_JSON = str(DATA_DIR / "model_comparison.json")
COMPARISON_PNG = str(DATA_DIR / "model_comparison.png")


def ensure_data_dir() -> Path:
    """Create the data directory if missing and return it."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR
