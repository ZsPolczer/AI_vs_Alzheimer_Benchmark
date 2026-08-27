"""Unit tests for drug threading through the IQ battery runner (eateot.runner).

Uses a fake lab engine (no model download) plus a single-question battery, so
the tests focus on what the runner does with a drug spec: resolving it,
passing it to the engine, honoring the drug's restore fraction, and recording
drug/dose in telemetry.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

# tests/test_runner.py -> repo root (so `import eateot` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eateot.runner as runner_module
from eateot.drugs import resolve_drug, resolve_stack
from eateot.runner import run_iq_test

MINI_BATTERY = [{
    "tier": 1,
    "domain": "Categorical Reasoning",
    "target_iq": "75 - Property / Category Classification",
    "question": "Which item does NOT belong? [Gold, Silver, Copper, Wood]",
    "ground_truth_anchors": [["wood"], ["metal", "metals"]],
    "max_points": 15,
}]

PERFECT_ANSWER = "wood is the odd one out because it is not a metal"


class _FakeLab:
    """Records how the engine was called; answers every question perfectly."""

    def __init__(self):
        self.model_id = "Fake/Qwen"
        self.degradations = []
        self.lerps = []
        self.inferences = []
        self.restores = 0

    def apply_degradation(self, profile_key, decay_mult=1.0, target_subnetwork="all",
                          enable_flicker=False, enable_sirens=False, noise_seed=None,
                          drug=None, epsilon=0.0):
        self.degradations.append({
            "track": profile_key, "decay_mult": decay_mult,
            "subnetwork": target_subnetwork, "flicker": enable_flicker,
            "sirens": enable_sirens, "seed": noise_seed, "drug": drug,
            "epsilon": epsilon,
        })
        return "fake system prompt"

    def lerp_toward_clean(self, fraction):
        self.lerps.append(fraction)

    def run_inference(self, user_prompt, sys_prompt, lucidity_surge=False,
                      seed=None, drug=None):
        self.inferences.append({"surge": lucidity_surge, "seed": seed, "drug": drug})
        return PERFECT_ANSWER

    def restore_clean_state(self):
        self.restores += 1


def _run(lab, telemetry_mock=None, battery=None, battery_name=None, **kwargs):
    """Run the battery silently; extra kwargs go to run_iq_test.

    Telemetry is ALWAYS mocked so the test suite never writes real entries
    into the domain logs under outputs/ — pass a ``telemetry_mock`` to
    inspect the log_test_run call (see the telemetry tests below). ``battery``
    / ``battery_name`` default to MINI_BATTERY / "mini_test" but can be
    overridden to exercise the IQ scale on other batteries.
    """
    if telemetry_mock is not None:
        patcher = mock.patch.object(runner_module, "log_test_run",
                                    new=telemetry_mock)
    else:
        patcher = mock.patch.object(runner_module, "log_test_run")
    with patcher, redirect_stdout(io.StringIO()):
        return run_iq_test(
            lab, "C1", 1.0, "all", False, False, False,
            battery=battery if battery is not None else MINI_BATTERY,
            battery_name=battery_name or "mini_test", **kwargs,
        )


class TestDrugThreading(unittest.TestCase):
    def test_drug_spec_is_resolved_and_passed_to_engine(self):
        lab = _FakeLab()
        _run(lab, drug="lsd", dose=2.0)
        expected = resolve_drug("lsd", 2.0)
        self.assertEqual(len(lab.degradations), 1)
        self.assertEqual(len(lab.inferences), 1)
        self.assertEqual(lab.degradations[0]["drug"], expected)
        self.assertEqual(lab.inferences[0]["drug"], expected)

    def test_no_drug_passes_none(self):
        lab = _FakeLab()
        _run(lab)
        self.assertIsNone(lab.degradations[0]["drug"])
        self.assertIsNone(lab.inferences[0]["drug"])
        self.assertEqual(lab.lerps, [])  # no restore without drug/param

    def test_drug_restore_fraction_is_applied(self):
        # nzt's spec carries restore_fraction 0.7 -> lerp before each question.
        lab = _FakeLab()
        _run(lab, drug="nzt", dose=1.0)
        self.assertEqual(lab.lerps, [0.7])

    def test_explicit_restore_fraction_wins_over_drug(self):
        lab = _FakeLab()
        _run(lab, drug="nzt", dose=1.0, restore_fraction=0.4)
        self.assertEqual(lab.lerps, [0.4])

    def test_drug_without_restore_fraction_does_not_lerp(self):
        lab = _FakeLab()
        _run(lab, drug="lsd", dose=1.0)
        self.assertEqual(lab.lerps, [])

    def test_telemetry_records_drug_and_dose(self):
        lab = _FakeLab()
        mocked = mock.Mock()
        _run(lab, telemetry_mock=mocked, drug="lsd", dose=2.0)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["drug"], "lsd")
        self.assertEqual(kwargs["dose"], 2.0)

    def test_telemetry_dose_is_none_without_drug(self):
        lab = _FakeLab()
        mocked = mock.Mock()
        _run(lab, telemetry_mock=mocked)
        kwargs = mocked.call_args.kwargs
        self.assertIsNone(kwargs["drug"])
        self.assertIsNone(kwargs["dose"])

    def test_restore_clean_state_called_per_question_under_drug(self):
        lab = _FakeLab()
        _run(lab, drug="lsd", dose=1.0)
        self.assertEqual(lab.restores, 1)  # once per battery question

    def test_header_prints_drug_line(self):
        lab = _FakeLab()
        buf = io.StringIO()
        with mock.patch.object(runner_module, "log_test_run") as mocked:
            with redirect_stdout(buf):
                run_iq_test(
                    lab, "C1", 1.0, "all", False, False, False,
                    battery=MINI_BATTERY, battery_name="mini_test",
                    drug="lsd", dose=2.0,
                )
        self.assertIn("DRUG: lsd @ 2.0x", buf.getvalue())
        mocked.assert_called_once()  # no real telemetry pollution from the suite

    def test_unknown_drug_raises_keyerror(self):
        with self.assertRaises(KeyError):
            _run(_FakeLab(), drug="heroin")

    def test_negative_dose_raises(self):
        with self.assertRaises(ValueError):
            _run(_FakeLab(), drug="lsd", dose=-1.0)

    def test_summary_is_returned(self):
        summary = _run(_FakeLab(), drug="lsd", dose=1.0)
        # Full marks on a 15-point battery hit the designed ceiling:
        # 50 + round(15/15 * (145-50)) = 145 — never BASE_IQ + points (which
        # would be 65 on the old unbounded scale and 190 on the full battery).
        self.assertEqual(summary["final_iq_score"], 145)
        self.assertEqual(summary["battery_name"], "mini_test")

    def test_iq_scale_floor_when_nothing_matches(self):
        # A battery whose anchors the (perfect) response cannot match scores 0
        # points -> the scale floor BASE_IQ, and never below.
        impossible = [{
            "tier": 1,
            "domain": "Categorical Reasoning",
            "target_iq": "75 - Property / Category Classification",
            "question": "Which item does NOT belong? [Gold, Silver, Copper, Wood]",
            "ground_truth_anchors": [["zanzibar"]],
            "max_points": 15,
        }]
        summary = _run(_FakeLab(), battery=impossible, battery_name="impossible")
        self.assertEqual(summary["final_iq_score"], 50)  # BASE_IQ floor

    def test_iq_scale_never_exceeds_ceiling_on_any_battery(self):
        # Whatever the battery's point total, full marks map to IQ_CEILING.
        big_battery = [{
            "tier": 1, "domain": "x", "target_iq": "y",
            "question": "Which item does NOT belong? [Gold, Silver, Copper, Wood]",
            "ground_truth_anchors": [["wood"], ["metal", "metals"]],
            "max_points": 200,
        }]
        summary = _run(_FakeLab(), battery=big_battery, battery_name="big")
        self.assertEqual(summary["final_iq_score"], 145)  # IQ_CEILING
        # A partial answer on the big battery also stays inside the range.
        partial = _run(_FakeLab(), battery=big_battery, battery_name="big2")
        self.assertLessEqual(partial["final_iq_score"], 145)
        self.assertGreaterEqual(partial["final_iq_score"], 50)


class TestStackThreading(unittest.TestCase):
    """Pre-resolved stack specs (dicts) thread through the runner unchanged."""

    def test_stack_spec_passed_through_unchanged(self):
        lab = _FakeLab()
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 0.5}])
        _run(lab, drug=spec)
        self.assertEqual(lab.degradations[0]["drug"], spec)
        self.assertEqual(lab.inferences[0]["drug"], spec)

    def test_stack_telemetry_records_label_without_dose(self):
        lab = _FakeLab()
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 0.5}])
        mocked = mock.Mock()
        _run(lab, telemetry_mock=mocked, drug=spec)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["drug"], "lsd@1+thc@0.5")
        self.assertIsNone(kwargs["dose"])

    def test_stack_restore_fraction_is_applied(self):
        # nzt (0.7) + lsd (none) -> stack restore 0.7 -> lerp before question
        lab = _FakeLab()
        spec = resolve_stack([{"drug": "nzt", "dose": 1.0},
                              {"drug": "lsd", "dose": 1.0}])
        _run(lab, drug=spec)
        self.assertEqual(lab.lerps, [0.7])

    def test_stack_header_prints_label(self):
        lab = _FakeLab()
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 0.5}])
        buf = io.StringIO()
        with mock.patch.object(runner_module, "log_test_run"):
            with redirect_stdout(buf):
                run_iq_test(
                    lab, "C1", 1.0, "all", False, False, False,
                    battery=MINI_BATTERY, battery_name="mini_test", drug=spec,
                )
        self.assertIn("DRUG STACK: lsd@1+thc@0.5", buf.getvalue())

    def test_single_drug_spec_dict_passed_as_name(self):
        # Lab passes resolved single-drug specs as dicts; telemetry must still
        # record the plain drug name and its dose.
        lab = _FakeLab()
        spec = resolve_drug("lsd", 2.0)
        mocked = mock.Mock()
        _run(lab, telemetry_mock=mocked, drug=spec)
        kwargs = mocked.call_args.kwargs
        self.assertEqual(kwargs["drug"], "lsd")
        self.assertEqual(kwargs["dose"], 2.0)


if __name__ == "__main__":
    unittest.main()
