"""Unit tests for the drug dose-response (trip) study helpers (apps.trip)."""

import sys
import unittest
from pathlib import Path

# tests/test_trip.py -> repo root (so `import apps` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps.trip import default_dose_sweep, parse_doses


class TestParseDoses(unittest.TestCase):
    def test_parses_list(self):
        self.assertEqual(parse_doses("0,0.5,1,2"), [0.0, 0.5, 1.0, 2.0])

    def test_parses_single_dose(self):
        self.assertEqual(parse_doses("1.5"), [1.5])

    def test_rejects_negative(self):
        with self.assertRaises(SystemExit):
            parse_doses("0,-1,2")

    def test_rejects_empty(self):
        with self.assertRaises(SystemExit):
            parse_doses("")
        with self.assertRaises(SystemExit):
            parse_doses(",,,")


class TestDefaultDoseSweep(unittest.TestCase):
    def test_linear_drug_scales_to_dose_cap(self):
        # lsd dose_cap 3.0 -> six steps from 0 to 1.25x
        self.assertEqual(default_dose_sweep("lsd"),
                         [0.0, 0.75, 1.5, 2.25, 3.0, 3.75])

    def test_fractional_cap_drug(self):
        # microdose_lsd dose_cap 0.5. NOTE: exact values depend on the drug's
        # dose_cap in config/drugs.yaml AND Python's round-half-even (e.g.
        # round(0.375, 2) == 0.38) — update if the catalog cap changes.
        self.assertEqual(default_dose_sweep("microdose_lsd"),
                         [0.0, 0.12, 0.25, 0.38, 0.5, 0.62])

    def test_breakthrough_drug_cap_one(self):
        # salvia dose_cap 1.0
        self.assertEqual(default_dose_sweep("salvia"),
                         [0.0, 0.25, 0.5, 0.75, 1.0, 1.25])

    def test_sweep_always_starts_at_zero(self):
        for drug in ("lsd", "cbd", "dmt", "thc", "ghb", "modafinil"):
            with self.subTest(drug=drug):
                doses = default_dose_sweep(drug)
                self.assertEqual(doses[0], 0.0)
                self.assertEqual(doses, sorted(doses))
                self.assertEqual(len(doses), 6)

    def test_unknown_drug_raises(self):
        with self.assertRaises(SystemExit):
            default_dose_sweep("heroin")

    def test_empty_drug_raises_friendly_error(self):
        # `make trip` with no DRUG passes --drug "" -> must not crash with a
        # TypeError on None["dose_cap"]
        with self.assertRaises(SystemExit) as ctx:
            default_dose_sweep("")
        self.assertIn("--drug is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
