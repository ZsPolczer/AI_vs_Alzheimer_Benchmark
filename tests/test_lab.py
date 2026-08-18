"""Unit tests for the interactive lab's --drug / --dose flag resolution.

Exercises ``apps.lab.resolve_cli_drug`` (the CLI-facing validation wrapper
around ``eateot.drugs.resolve_drug``) without loading any model.
"""

import sys
import unittest
from pathlib import Path

# tests/test_lab.py -> repo root (so `import apps` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import lab as lab_module
from apps.lab import resolve_cli_drug, resolve_cli_stack
from eateot.drugs import DRUG_PROFILES, resolve_drug, resolve_stack


class TestResolveCliDrug(unittest.TestCase):
    def test_no_drug_returns_none(self):
        self.assertIsNone(resolve_cli_drug(None))
        self.assertIsNone(resolve_cli_drug(""))
        self.assertIsNone(resolve_cli_drug("", dose=3.0))

    def test_known_drug_resolves_with_dose(self):
        spec = resolve_cli_drug("lsd", 2.0)
        self.assertEqual(spec, resolve_drug("lsd", 2.0))
        self.assertEqual(spec["name"], "lsd")
        self.assertAlmostEqual(spec["factor"], 2.0)

    def test_dose_defaults_to_one(self):
        spec = resolve_cli_drug("lsd")
        self.assertEqual(spec, resolve_drug("lsd", 1.0))

    def test_unknown_drug_raises_system_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_cli_drug("heroin")
        self.assertIn("heroin", str(ctx.exception))
        self.assertIn("lsd", str(ctx.exception))  # error lists available drugs

    def test_negative_dose_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            resolve_cli_drug("lsd", -1.0)

    def test_dose_beyond_cap_still_resolves_but_flags(self):
        # dose_cap is advisory: the lab lets you take a heroic dose, flagged.
        spec = resolve_cli_drug("salvia", 4.0)
        self.assertTrue(spec["dose_exceeds_cap"])
        self.assertEqual(spec["primitives"]["scale"], 0.0)  # fully collapsed


class TestResolveCliStack(unittest.TestCase):
    def test_no_stack_returns_none(self):
        self.assertIsNone(resolve_cli_stack(None))
        self.assertIsNone(resolve_cli_stack(""))

    def test_valid_stack_resolves(self):
        spec = resolve_cli_stack("lsd@1.0,thc@0.5")
        self.assertEqual(
            spec,
            resolve_stack([{"drug": "lsd", "dose": 1.0},
                           {"drug": "thc", "dose": 0.5}]),
        )
        self.assertEqual(spec["name"], "lsd@1+thc@0.5")
        self.assertEqual(spec["class"], "stack")

    def test_single_drug_stack_is_plain_drug(self):
        spec = resolve_cli_stack("lsd@2.0")
        self.assertEqual(spec, resolve_drug("lsd", 2.0))

    def test_malformed_spec_raises_system_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_cli_stack("lsd@abc")
        self.assertIn("lsd@abc", str(ctx.exception))

    def test_unknown_drug_raises_system_exit(self):
        with self.assertRaises(SystemExit) as ctx:
            resolve_cli_stack("lsd@1.0,heroin@2.0")
        self.assertIn("heroin", str(ctx.exception))

    def test_negative_dose_raises_system_exit(self):
        with self.assertRaises(SystemExit):
            resolve_cli_stack("lsd@-1.0")


class TestSelectDrugStack(unittest.TestCase):
    """The interactive deployer: dependent class → drug → dose → combo flow."""

    @staticmethod
    def _drive(inputs):
        """Feed mocked input() lines into _select_drug_stack."""
        import builtins

        it = iter(inputs)
        orig = builtins.input
        builtins.input = lambda prompt="": next(it)
        try:
            return lab_module._select_drug_stack()
        finally:
            builtins.input = orig

    @staticmethod
    def _class_index(cls):
        classes = sorted({p["class"] for p in DRUG_PROFILES.values()})
        return classes.index(cls) + 1

    @staticmethod
    def _drug_index(cls, name):
        drugs = sorted(n for n, p in DRUG_PROFILES.items() if p["class"] == cls)
        return drugs.index(name) + 1

    def test_cancel_returns_empty(self):
        self.assertEqual(self._drive(["0"]), [])
        self.assertEqual(self._drive(["99", "0"]), [])  # bad class retries safely

    def test_single_drug_selection(self):
        comps = self._drive([
            str(self._class_index("hallucinogen")),
            str(self._drug_index("hallucinogen", "lsd")),
            "2.0", "n",
        ])
        self.assertEqual(comps, [{"drug": "lsd", "dose": 2.0}])

    def test_dose_defaults_to_one(self):
        comps = self._drive([
            str(self._class_index("stimulant")),
            str(self._drug_index("stimulant", "caffeine")),
            "", "n",
        ])
        self.assertEqual(comps, [{"drug": "caffeine", "dose": 1.0}])

    def test_builds_combo_stack(self):
        comps = self._drive([
            str(self._class_index("hallucinogen")),
            str(self._drug_index("hallucinogen", "lsd")),
            "2.0", "y",
            str(self._class_index("cannabinoid")),
            str(self._drug_index("cannabinoid", "thc")),
            "0.5", "n",
        ])
        self.assertEqual(comps, [{"drug": "lsd", "dose": 2.0},
                                 {"drug": "thc", "dose": 0.5}])

    def test_selected_stack_resolves(self):
        comps = self._drive([
            str(self._class_index("hallucinogen")),
            str(self._drug_index("hallucinogen", "lsd")),
            "1.0", "y",
            str(self._class_index("depressant")),
            str(self._drug_index("depressant", "ghb")),
            "1.0", "n",
        ])
        spec = resolve_stack(comps)
        self.assertEqual(spec["name"], "lsd@1+ghb@1")
        self.assertEqual(spec["class"], "stack")
        self.assertAlmostEqual(spec["primitives"]["verbosity_bias"], -1.0)  # +2 + -3

    def test_summary_prints_without_error(self):
        import io
        from contextlib import redirect_stdout

        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 0.5}])
        buf = io.StringIO()
        with redirect_stdout(buf):
            lab_module._print_drug_summary(spec)
        self.assertIn("lsd@1+thc@0.5", buf.getvalue())
        self.assertIn("attention_scatter", buf.getvalue())
        single = resolve_drug("cbd", 1.0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            lab_module._print_drug_summary(single)
        self.assertIn("cbd", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
