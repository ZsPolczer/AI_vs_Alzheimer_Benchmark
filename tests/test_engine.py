"""Unit tests for the degradation engine's restoration + progressive math.

Tests ``lerp_toward_clean`` / ``restore_clean_state`` and the progressive
in-generation degradation ramp / hidden-state corruption against a minimal
fake layer stack (no model download or GPU needed).
"""

import unittest

import torch

from eateot.engine import (
    BrainLabEngine,
    _progressive_hook_factory,
    apply_progressive_corruption,
    progressive_ramp_intensity,
)


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


def _default_ramp_intensities(n=10):
    """Intensity at each token of an n-token generation (default ramp params)."""
    return [progressive_ramp_intensity(i / n) for i in range(n)]


class TestProgressiveRamp(unittest.TestCase):
    def test_early_tokens_near_clean(self):
        # Barely any influence at the start: first tokens stay under 20% intensity.
        intensities = _default_ramp_intensities(10)
        self.assertLess(intensities[0], 0.2)
        self.assertLess(intensities[1], 0.2)

    def test_last_token_full_intensity(self):
        # The ramp is normalized so the final token always hits "full mayhem" (1.0).
        self.assertAlmostEqual(progressive_ramp_intensity(1.0), 1.0, places=6)
        self.assertAlmostEqual(progressive_ramp_intensity(0.999), 1.0, places=3)

    def test_monotonic_increase(self):
        intensities = _default_ramp_intensities(20)
        for a, b in zip(intensities, intensities[1:]):
            self.assertGreaterEqual(b, a)

    def test_clamped_to_unit_interval(self):
        self.assertEqual(progressive_ramp_intensity(-1.0), 0.0)
        self.assertEqual(progressive_ramp_intensity(2.0), 1.0)

    def test_steeper_ramp_starts_later(self):
        # Higher k delays the rise once the clean zone ends...
        probe = 0.6  # past the default clean zone (mid=0.35)
        self.assertLess(
            progressive_ramp_intensity(probe, k=16.0),
            progressive_ramp_intensity(probe, k=2.5),
        )
        # ...and later mid pushes the whole ramp toward the end.
        self.assertLess(
            progressive_ramp_intensity(0.5, mid=0.7),
            progressive_ramp_intensity(0.5, mid=0.3),
        )

    def test_late_mid_keeps_early_half_clean(self):
        # A late mid pushes the ramp toward the end: at half-generation the
        # corruption is still mild, and much lower than with an early mid.
        self.assertLess(progressive_ramp_intensity(0.5, mid=0.6), 0.15)
        self.assertLess(progressive_ramp_intensity(0.5, mid=0.6),
                        progressive_ramp_intensity(0.5, mid=0.3))


class TestProgressiveCorruption(unittest.TestCase):
    def test_zero_intensity_is_noop(self):
        hidden = torch.randn(4, 8)
        before = hidden.clone()
        apply_progressive_corruption(hidden, 0.0, epsilon=0.5)
        self.assertTrue(torch.equal(hidden, before))

    def test_full_intensity_collapses_scale_and_adds_noise(self):
        hidden = torch.randn(4, 8) * 3.0  # std ≈ 3
        std_before = hidden.std().item()
        apply_progressive_corruption(hidden, 1.0, epsilon=0.5, scale_min=0.2)
        # Scale collapse toward 0.2×, plus std-scaled gaussian noise.
        self.assertLess(hidden.std().item(), std_before)
        self.assertGreater(hidden.std().item(), 0.0)

    def test_mayhem_ratio_between_endpoints(self):
        # At full intensity the result sits between pure scale-collapse (0.2×)
        # and noise-drenched chaos — a smoke check that both knobs act.
        torch.manual_seed(0)
        hidden = torch.randn(64, 32) * 2.0
        collapse_only = hidden.clone().mul_(0.2)
        std = hidden.std().item()
        apply_progressive_corruption(hidden, 1.0, epsilon=0.5, scale_min=0.2)
        after_std = hidden.std().item()
        self.assertGreater(after_std, collapse_only.std().item())
        self.assertLess(after_std, std + 1.0)  # not blown up

    def test_partial_intensity_between_clean_and_full(self):
        torch.manual_seed(1)
        base = torch.randn(32, 16) * 2.0
        a, b = base.clone(), base.clone()
        apply_progressive_corruption(a, 0.25, epsilon=0.5, scale_min=0.2)
        apply_progressive_corruption(b, 1.0, epsilon=0.5, scale_min=0.2)
        # More corruption at higher intensity: deviation from the clean tensor grows.
        self.assertLess((a - base).abs().mean().item(),
                        (b - base).abs().mean().item())


class TestProgressiveHook(unittest.TestCase):
    def test_hook_corrupts_hidden_in_place_and_counts(self):
        state = {"count": 2, "total": 4}
        hook = _progressive_hook_factory(state, epsilon=0.5, scale_min=0.2, mid=0.4, k=8.0)
        hidden = torch.randn(2, 4) * 2.0
        out = type("Out", (), {"last_hidden_state": hidden})()
        result = hook(None, None, out)
        self.assertIsNone(result)  # in-place mutation, output flows through untouched
        self.assertEqual(state["count"], 3)
        self.assertFalse(torch.equal(hidden, torch.randn(2, 4) * 2.0))

    def test_hook_handles_tuple_output(self):
        state = {"count": 0, "total": 4}
        hook = _progressive_hook_factory(state, epsilon=0.5, scale_min=0.2, mid=0.4, k=8.0)
        hidden = torch.randn(2, 4)
        result = hook(None, None, (hidden, torch.tensor([1.0])))
        self.assertIsNone(result)
        self.assertEqual(state["count"], 1)

    def test_hook_ignores_foreign_outputs(self):
        state = {"count": 0, "total": 4}
        hook = _progressive_hook_factory(state, epsilon=0.5, scale_min=0.2, mid=0.4, k=8.0)
        self.assertIsNone(hook(None, None, "not a tensor"))
        self.assertEqual(state["count"], 0)


if __name__ == "__main__":
    unittest.main()
