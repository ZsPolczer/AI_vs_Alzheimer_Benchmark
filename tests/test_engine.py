"""Unit tests for the degradation engine's restoration math.

Tests ``lerp_toward_clean`` / ``restore_clean_state`` against a minimal fake
layer stack (no model download or GPU needed).
"""

import unittest

import torch

from eateot.engine import BrainLabEngine


class _Param:
    def __init__(self, data):
        self.data = data


class _Layer:
    def __init__(self, value):
        self.p = _Param(torch.tensor([value], dtype=torch.float32))

    def named_parameters(self):
        yield "w", self.p


def _engine(clean_value=3.0, degraded_value=1.0):
    """A BrainLabEngine with one fake layer; clean backup + degraded weights set."""
    eng = object.__new__(BrainLabEngine)
    eng.backups = {(0, "w"): torch.tensor([clean_value], dtype=torch.float32)}
    eng.model = type("M", (), {"model": type("MM", (), {"layers": [_Layer(degraded_value)]})()})()
    return eng


def _value(eng):
    return eng._get_layers()[0].p.data.item()


class TestLerpTowardClean(unittest.TestCase):
    def test_fraction_zero_keeps_degraded(self):
        eng = _engine(clean_value=3.0, degraded_value=1.0)
        eng.lerp_toward_clean(0.0)
        self.assertEqual(_value(eng), 1.0)

    def test_fraction_half_interpolates(self):
        eng = _engine(clean_value=3.0, degraded_value=1.0)
        eng.lerp_toward_clean(0.5)
        self.assertAlmostEqual(_value(eng), 2.0, places=6)

    def test_fraction_one_restores_clean(self):
        eng = _engine(clean_value=3.0, degraded_value=1.0)
        eng.lerp_toward_clean(1.0)
        self.assertEqual(_value(eng), 3.0)

    def test_restore_clean_state_clears_backups(self):
        eng = _engine(clean_value=3.0, degraded_value=1.0)
        eng.lerp_toward_clean(0.4)
        eng.restore_clean_state()
        self.assertEqual(_value(eng), 3.0)
        self.assertEqual(eng.backups, {})

    def test_requires_prior_degradation(self):
        eng = object.__new__(BrainLabEngine)
        eng.backups = {}
        with self.assertRaises(RuntimeError):
            eng.lerp_toward_clean(0.5)


if __name__ == "__main__":
    unittest.main()
