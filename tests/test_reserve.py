"""Unit tests for the cognitive reserve study helpers (apps.reserve)."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# tests/test_reserve.py -> repo root (so `import apps` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import reserve as reserve_module
from apps.reserve import _latest_entry, auc_curve, collapse_severity, parse_severities


class TestParseSeverities(unittest.TestCase):
    def test_parses_list(self):
        self.assertEqual(parse_severities("1,2,4,8"), [1.0, 2.0, 4.0, 8.0])

    def test_rejects_nonpositive(self):
        with self.assertRaises(SystemExit):
            parse_severities("0,2")
        with self.assertRaises(SystemExit):
            parse_severities("-1,2")

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            parse_severities("")


class TestAucCurve(unittest.TestCase):
    def test_trapezoid_area(self):
        # y = x over [0,1]: area = 0.5
        self.assertAlmostEqual(auc_curve([0.0, 1.0], [0.0, 1.0]), 0.5, places=6)

    def test_flat_line(self):
        # y = 100 over [1, 4]: area = 300
        self.assertAlmostEqual(auc_curve([1.0, 2.0, 4.0], [100.0, 100.0, 100.0]), 300.0, places=6)

    def test_short_curve_returns_zero(self):
        self.assertEqual(auc_curve([1.0], [100.0]), 0.0)

    def test_mismatched_lengths_return_zero(self):
        self.assertEqual(auc_curve([1.0, 2.0], [100.0]), 0.0)


class TestLatestEntry(unittest.TestCase):
    """_latest_entry must skip dose-response runs (restore_fraction set)."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

    @staticmethod
    def _entry(model, track, mult, restore=None, drug=None):
        return {
            "model_name": model,
            "config": {
                "track_profile": track,
                "questionnaire": "iq_battery",
                "decay_multiplier": mult,
                "restore_fraction": restore,
                "drug": drug,
            },
            "summary": {"final_iq_score": 100, "clinical_diagnosis": "x"},
        }

    def test_skips_drug_runs(self):
        """A drug run's IQ is not a pure severity measurement — never match it."""
        log = os.path.join(self.tmpdir.name, "iq_test_results.json")
        with open(log, "w", encoding="utf-8") as f:
            json.dump([self._entry("M", "C1", 1.0, drug="lsd")], f)
        with mock.patch.object(reserve_module, "LOG_FILE", log):
            hit = _latest_entry("M", "C1", "iq_battery", 1.0)
        self.assertIsNone(hit)

    def test_skips_restore_runs(self):
        """A newer dose-response run must not shadow an older plain run."""
        log = os.path.join(self.tmpdir.name, "iq_test_results.json")
        with open(log, "w", encoding="utf-8") as f:
            json.dump([
                self._entry("M", "C1", 1.0),                    # plain (older)
                self._entry("M", "C1", 1.0, restore=0.5),       # restore (newest)
            ], f)
        with mock.patch.object(reserve_module, "LOG_FILE", log):
            hit = _latest_entry("M", "C1", "iq_battery", 1.0)
        self.assertIsNotNone(hit)
        self.assertIsNone(hit["config"].get("restore_fraction"))

    def test_plain_run_still_matches(self):
        log = os.path.join(self.tmpdir.name, "iq_test_results.json")
        with open(log, "w", encoding="utf-8") as f:
            json.dump([self._entry("M", "C1", 1.0)], f)
        with mock.patch.object(reserve_module, "LOG_FILE", log):
            hit = _latest_entry("M", "C1", "iq_battery", 1.0)
        self.assertIsNotNone(hit)

    def test_missing_log_returns_none(self):
        missing = os.path.join(self.tmpdir.name, "nope.json")
        with mock.patch.object(reserve_module, "LOG_FILE", missing):
            self.assertIsNone(_latest_entry("M", "C1", "iq_battery", 1.0))


class TestCollapseSeverity(unittest.TestCase):
    def test_interpolated_crossing(self):
        # IQ drops 130 -> 80 between severity 2 and 4; threshold 100 crossed at 3.2.
        xs = [1.0, 2.0, 4.0]
        ys = [130.0, 130.0, 80.0]
        self.assertAlmostEqual(collapse_severity(xs, ys, 100.0), 3.2, places=6)

    def test_never_crosses_returns_none(self):
        xs = [1.0, 2.0, 4.0]
        ys = [130.0, 125.0, 115.0]
        self.assertIsNone(collapse_severity(xs, ys, 100.0))

    def test_crossing_on_last_segment(self):
        xs = [1.0, 2.0, 4.0]
        ys = [130.0, 105.0, 70.0]  # 105 -> 70 crosses 100 at frac 5/35 of the way
        expected = 2.0 + (105.0 - 100.0) / (105.0 - 70.0) * (4.0 - 2.0)
        self.assertAlmostEqual(collapse_severity(xs, ys, 100.0), expected, places=6)

    def test_already_below_at_start_returns_none(self):
        xs = [1.0, 2.0]
        ys = [90.0, 60.0]
        self.assertIsNone(collapse_severity(xs, ys, 100.0))


if __name__ == "__main__":
    unittest.main()
