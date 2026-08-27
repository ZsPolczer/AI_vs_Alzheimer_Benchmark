"""Unit tests for ``eateot.paths.clear_charts`` (the lab's CLEAR GRAPHS action).

Redirects ``EATEOT_DATA_DIR`` at a temp folder so the tests never touch the
real ``outputs/`` directory, then reloads the module so its constants resolve
against the override.
"""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# tests/test_paths.py -> repo root (so `import eateot` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eateot.paths as paths


class TestClearCharts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.env = mock.patch.dict(
            os.environ, {"EATEOT_DATA_DIR": self.tmp.name}, clear=False
        )
        self.env.start()
        self.addCleanup(self.env.stop)
        importlib.reload(paths)  # re-resolve DATA_DIR / CHART_FILES

    def tearDown(self):
        importlib.reload(paths)  # restore default DATA_DIR for other tests

    def test_removes_existing_charts_only(self):
        existing = paths.CHART_FILES[:3]
        for chart in existing:
            Path(chart).write_text("stale chart", encoding="utf-8")
        missing = paths.CHART_FILES[3]
        self.assertFalse(Path(missing).exists())

        removed = paths.clear_charts()

        self.assertEqual(sorted(removed), sorted(existing))
        for chart in existing:
            self.assertFalse(Path(chart).exists())
        self.assertFalse(Path(missing).exists())

    def test_no_charts_returns_empty(self):
        self.assertEqual(paths.clear_charts(), [])

    def test_chart_list_covers_all_generated_pngs(self):
        # Every study chart is a *_png constant; keep the list in sync.
        png_constants = [
            "IMG_FILE", "COMPARISON_PNG", "RESTORE_PNG", "TRAJECTORY_PNG",
            "RESERVE_PNG", "TRIP_PNG", "DRUG_REPORT_PNG", "SENSITIVITY_PNG",
        ]
        expected = {getattr(paths, name) for name in png_constants}
        self.assertEqual(set(paths.CHART_FILES), expected)


if __name__ == "__main__":
    unittest.main()
