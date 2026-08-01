"""Test suite for the eateot package. Run with: python -m unittest discover -s tests"""

import sys
from pathlib import Path

# Ensure the src/ layout is importable when running tests from the repo root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
