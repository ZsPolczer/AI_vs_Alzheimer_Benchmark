"""Unit tests for drug-primitive wiring in the degradation engine.

Covers the pure folding helpers (``compute_degradation_params`` /
``compute_sampling_params``), the ``GaussianLogitNoise`` processor, the
attention-scatter hook factory, and an integration check of
``apply_degradation`` against a minimal fake layer stack (no model download
needed). Drug specs come from the real catalog via ``eateot.drugs.resolve_drug``.
"""

import unittest

import torch

from eateot.drugs import resolve_drug, resolve_stack
from eateot.engine import (
    BASE_MAX_NEW_TOKENS,
    BrainLabEngine,
    GaussianLogitNoise,
    _attention_scatter_hook,
    compute_degradation_params,
    compute_sampling_params,
)
from eateot.profiles import EATEOT_TRACK_PROFILES

C1 = EATEOT_TRACK_PROFILES["C1"]  # scale 0.86, noise 0.00030, layer_pct (0.25, 0.75)


class _Param:
    def __init__(self, data):
        self.data = data


class _FakeAttn:
    """Minimal attention module: records hooks and returns removable handles."""

    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return _FakeHandle(self)


class _FakeHandle:
    def __init__(self, attn):
        self.attn = attn
        self.removed = False

    def remove(self):
        self.removed = True


class _Layer:
    def __init__(self, value, name="w"):
        self.p = _Param(torch.tensor([value], dtype=torch.float32))
        self._name = name
        if name == "attn_q":
            self.self_attn = _FakeAttn()

    def named_parameters(self):
        yield self._name, self.p


def _fake_engine(num_layers=4, layer_names=None):
    """A BrainLabEngine with fake layers; no tokenizer/model download."""
    names = layer_names or ["w"] * num_layers
    eng = object.__new__(BrainLabEngine)
    eng.total_layers = num_layers
    eng.backups = {}
    eng.model = type("M", (), {
        "model": type("MM", (), {"layers": [_Layer(3.0, n) for n in names]})()
    })()
    return eng


def _weights(eng):
    return [l.p.data.item() for l in eng._get_layers()]


class TestComputeDegradationParams(unittest.TestCase):
    def test_no_drug_is_legacy_behavior(self):
        params = compute_degradation_params(C1)
        self.assertAlmostEqual(params["scale"], 0.86, places=6)
        self.assertAlmostEqual(params["noise_std"], 0.00030, places=9)
        self.assertEqual(params["flicker_rate"], 0.0)
        self.assertEqual(params["noise_gradient"], 0.0)
        self.assertEqual(params["sirens_mult"], 2.5)
        self.assertEqual(params["layer_pct"], (0.25, 0.75))
        self.assertEqual(params["subnetwork"], "all")

    def test_drug_scale_is_multiplicative(self):
        # salvia dose 1 -> scale 0.125; C1 0.86 * 0.125
        params = compute_degradation_params(C1, drug=resolve_drug("salvia", 1.0))
        self.assertAlmostEqual(params["scale"], 0.86 * 0.125, places=6)

    def test_drug_scale_of_one_is_noop(self):
        # lsd does not touch weights -> track scale preserved (no-op contract)
        params = compute_degradation_params(C1, drug=resolve_drug("lsd", 1.0))
        self.assertAlmostEqual(params["scale"], 0.86, places=6)

    def test_drug_noise_is_additive(self):
        params = compute_degradation_params(C1, drug=resolve_drug("lsd", 1.0))
        self.assertAlmostEqual(params["noise_std"], 0.00030 + 0.00008, places=9)

    def test_negative_drug_noise_suppresses(self):
        params = compute_degradation_params(C1, drug=resolve_drug("modafinil", 1.0))
        self.assertAlmostEqual(params["noise_std"], 0.00030 - 0.00005, places=9)

    def test_noise_clamped_at_zero(self):
        prof = {"x": {"class": "placebo", "curve": "linear", "subnetwork": "all",
                      "layer_pct": [0.0, 1.0], "dose_curve": {"noise": -0.001}}}
        params = compute_degradation_params(C1, drug=resolve_drug("x", 1.0, profiles={"x": prof["x"]}))
        self.assertEqual(params["noise_std"], 0.0)

    def test_sirens_mult_drug_override(self):
        prof = {"x": {"class": "placebo", "curve": "linear", "subnetwork": "all",
                      "layer_pct": [0.0, 1.0], "dose_curve": {"sirens_mult": 4.0}}}
        drug = resolve_drug("x", 1.0, profiles={"x": prof["x"]})
        params = compute_degradation_params(C1, enable_sirens=True, drug=drug)
        self.assertAlmostEqual(params["noise_std"], 0.00030 * 4.0, places=9)
        self.assertEqual(params["sirens_mult"], 4.0)

    def test_sirens_off_ignores_multiplier(self):
        prof = {"x": {"class": "placebo", "curve": "linear", "subnetwork": "all",
                      "layer_pct": [0.0, 1.0], "dose_curve": {"sirens_mult": 4.0}}}
        drug = resolve_drug("x", 1.0, profiles={"x": prof["x"]})
        params = compute_degradation_params(C1, drug=drug)
        self.assertAlmostEqual(params["noise_std"], 0.00030, places=9)

    def test_epsilon_std_scaled_perturbation(self):
        # Profiles opt into the sensitivity method via an `epsilon` key
        # (Ẇ = W + ε·σ_W·Z); the default track carries none.
        self.assertEqual(compute_degradation_params(C1)["epsilon"], 0.0)
        prof = dict(C1)
        prof["epsilon"] = 0.01
        params = compute_degradation_params(prof)
        self.assertAlmostEqual(params["epsilon"], 0.01, places=9)
        # Drug epsilon folds additively on top of the profile's.
        drug = {"primitives": {"epsilon": 0.005}}
        params = compute_degradation_params(prof, drug=drug)
        self.assertAlmostEqual(params["epsilon"], 0.015, places=9)
        # Negative values are clamped at zero.
        prof["epsilon"] = -0.1
        self.assertEqual(compute_degradation_params(prof)["epsilon"], 0.0)

    def test_apply_degradation_accepts_epsilon(self):
        # Smoke test: explicit epsilon reaches apply_degradation without error
        # (single-element fake params have std 0, so no noise is injected there).
        eng = _fake_engine(num_layers=4)
        eng.apply_degradation("C1", noise_seed=0, epsilon=0.01)
        self.assertTrue(eng.backups)

    def test_drug_flicker_without_flag(self):
        # datura induces dropout on its own (enable_flicker stays False);
        # its steep curve × potency 1.3 folds 0.15 -> 0.195 at dose 1.
        drug = resolve_drug("datura", 1.0)
        expected = drug["primitives"]["flicker_rate"]
        params = compute_degradation_params(C1, drug=drug)
        self.assertAlmostEqual(params["flicker_rate"], expected, places=9)
        self.assertGreater(params["flicker_rate"], 0.0)

    def test_default_flicker_when_flag_only(self):
        params = compute_degradation_params(C1, enable_flicker=True)
        self.assertEqual(params["flicker_rate"], 0.25)

    def test_drug_flicker_wins_over_flag(self):
        drug = resolve_drug("datura", 1.0)
        expected = drug["primitives"]["flicker_rate"]
        params = compute_degradation_params(C1, enable_flicker=True, drug=drug)
        self.assertAlmostEqual(params["flicker_rate"], expected, places=9)

    def test_drug_layer_window_overrides_track(self):
        params = compute_degradation_params(C1, drug=resolve_drug("lsd", 1.0))
        self.assertEqual(params["layer_pct"], [0.65, 0.95])

    def test_drug_subnetwork_overrides(self):
        params = compute_degradation_params(C1, drug=resolve_drug("lsd", 1.0))
        self.assertEqual(params["subnetwork"], "attn")

    def test_drug_subnetwork_all_keeps_caller(self):
        params = compute_degradation_params(C1, target_subnetwork="mlp",
                                            drug=resolve_drug("psilocybin", 1.0))
        self.assertEqual(params["subnetwork"], "mlp")

    def test_decay_multiplier_still_applies(self):
        params = compute_degradation_params(C1, decay_mult=2.0,
                                            drug=resolve_drug("lsd", 1.0))
        self.assertAlmostEqual(params["scale"], 0.86 / 2.0, places=6)
        self.assertAlmostEqual(params["noise_std"], 0.00030 * 2.0 + 0.00008, places=9)

    def test_noise_gradient_passthrough(self):
        params = compute_degradation_params(C1, drug=resolve_drug("dxm", 1.0))
        self.assertAlmostEqual(params["noise_gradient"], 0.0004, places=9)


class TestComputeSamplingParams(unittest.TestCase):
    def test_no_drug_defaults(self):
        params = compute_sampling_params()
        self.assertEqual(params["temperature"], 0.7)
        self.assertEqual(params["repetition_penalty"], 1.0)
        self.assertEqual(params["max_new_tokens"], BASE_MAX_NEW_TOKENS)
        self.assertEqual(params["logit_noise"], 0.0)
        self.assertEqual(params["attention_scatter"], 0.0)
        self.assertEqual(params["context_mask_frac"], 0.0)

    def test_lsd_temperature_and_verbosity(self):
        params = compute_sampling_params(drug=resolve_drug("lsd", 1.0))
        self.assertAlmostEqual(params["temperature"], 0.85, places=6)
        self.assertEqual(params["max_new_tokens"], BASE_MAX_NEW_TOKENS + 2 * 16)

    def test_ghb_terse(self):
        params = compute_sampling_params(drug=resolve_drug("ghb", 1.0))
        self.assertEqual(params["max_new_tokens"], BASE_MAX_NEW_TOKENS - 3 * 16)

    def test_max_new_tokens_floor(self):
        prof = {"x": {"class": "placebo", "curve": "linear", "subnetwork": "all",
                      "layer_pct": [0.0, 1.0], "dose_curve": {"verbosity_bias": -100.0}}}
        params = compute_sampling_params(drug=resolve_drug("x", 1.0, profiles={"x": prof["x"]}))
        self.assertEqual(params["max_new_tokens"], 32)

    def test_surge_overrides_drug_temperature(self):
        params = compute_sampling_params(drug=resolve_drug("lsd", 1.0), lucidity_surge=True)
        self.assertEqual(params["temperature"], 0.3)

    def test_amphetamine_repetition_suppression(self):
        params = compute_sampling_params(drug=resolve_drug("amphetamine", 1.0))
        self.assertAlmostEqual(params["repetition_penalty"], 1.15, places=6)

    def test_dmt_sampling_primitives(self):
        params = compute_sampling_params(drug=resolve_drug("dmt", 1.0))
        self.assertGreater(params["logit_noise"], 0.0)
        self.assertGreater(params["attention_scatter"], 0.0)
        self.assertGreater(params["context_mask_frac"], 0.0)


class TestStackFolding(unittest.TestCase):
    """Combo specs (resolve_stack) fold exactly like single drugs."""

    def test_degradation_noise_sums_across_stack(self):
        # lsd 0.00008 + thc 0.0001 on top of C1's 0.00030
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        params = compute_degradation_params(C1, drug=spec)
        self.assertAlmostEqual(params["noise_std"], 0.00030 + 0.00008 + 0.0001,
                               places=9)
        self.assertAlmostEqual(params["scale"], 0.86, places=6)  # neither touches weights

    def test_degradation_scale_combines_weight_loss(self):
        # salvia loss 0.875 + ketamine loss 0.08 -> C1 0.86 * (1 - 0.955)
        spec = resolve_stack([{"drug": "salvia", "dose": 1.0},
                              {"drug": "ketamine", "dose": 1.0}])
        params = compute_degradation_params(C1, drug=spec)
        self.assertAlmostEqual(params["scale"], 0.86 * 0.045, places=6)

    def test_degradation_layer_window_is_union(self):
        # lsd [0.65, 0.95] + ketamine [0.30, 0.70] -> [0.30, 0.95]
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "ketamine", "dose": 1.0}])
        params = compute_degradation_params(C1, drug=spec)
        self.assertEqual(params["layer_pct"], [0.30, 0.95])

    def test_mixed_subnetwork_falls_back_to_caller(self):
        # lsd (attn) + thc (all) -> "all" -> caller's target is kept
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        params = compute_degradation_params(C1, target_subnetwork="mlp", drug=spec)
        self.assertEqual(params["subnetwork"], "mlp")

    def test_sampling_temperature_and_verbosity_sum(self):
        # lsd +0.15 + thc +0.10 -> 0.95; verbosity lsd +2 -> +2*16 tokens
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        params = compute_sampling_params(drug=spec)
        self.assertAlmostEqual(params["temperature"], 0.95, places=6)
        self.assertEqual(params["max_new_tokens"], BASE_MAX_NEW_TOKENS + 2 * 16)

    def test_stack_applied_to_fake_layers(self):
        # union window [0.25, 0.95] on 4 layers -> indices 1..2; subnetwork "all"
        eng = _fake_engine(num_layers=4)
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        eng.apply_degradation("C1", drug=spec, noise_seed=0)
        weights = _weights(eng)
        self.assertEqual(weights[0], 3.0)   # outside the union window
        self.assertEqual(weights[3], 3.0)
        for w in weights[1:3]:
            self.assertAlmostEqual(w, 3.0 * 0.86, delta=0.01)
        self.assertTrue(eng.backups)


class TestGaussianLogitNoise(unittest.TestCase):
    def test_zero_std_is_identity(self):
        scores = torch.zeros(1, 10)
        out = GaussianLogitNoise(0.0)(torch.tensor([[1]]), scores)
        self.assertTrue(torch.equal(out, scores))

    def test_positive_std_adds_noise(self):
        torch.manual_seed(7)
        scores = torch.zeros(1, 10000)
        out = GaussianLogitNoise(0.5)(torch.tensor([[1]]), scores)
        self.assertFalse(torch.equal(out, scores))
        self.assertAlmostEqual(out.std().item(), 0.5, delta=0.02)


class TestAttentionScatterHook(unittest.TestCase):
    def test_tensor_output_noised(self):
        torch.manual_seed(3)
        module = object()
        out = torch.zeros(2, 4, 8)
        hook = _attention_scatter_hook(0.1)
        result = hook(module, (), out)
        self.assertFalse(torch.equal(result, out))

    def test_tuple_output_noises_head_only(self):
        torch.manual_seed(3)
        head = torch.zeros(2, 4, 8)
        tail = (None, torch.zeros(1, 2))
        hook = _attention_scatter_hook(0.1)
        result = hook(object(), (), (head,) + tail)
        self.assertFalse(torch.equal(result[0], head))
        self.assertIs(result[1], tail[0])
        self.assertIs(result[2], tail[1])


class TestAttachAttentionScatter(unittest.TestCase):
    def test_registers_hooks_over_the_layer_window(self):
        eng = _fake_engine(num_layers=10, layer_names=["attn_q"] * 10)
        handles = eng._attach_attention_scatter(0.1, 6, 9)
        self.assertEqual(len(handles), 3)  # layers 6, 7, 8
        for h in handles:
            h.remove()
        self.assertTrue(all(h.removed for h in handles))

    def test_drug_window_indices_are_used(self):
        eng = _fake_engine(num_layers=10, layer_names=["attn_q"] * 10)
        drug = resolve_drug("lsd", 1.0)  # layer_pct [0.65, 0.95]
        start = int(eng.total_layers * drug["layer_pct"][0])
        end = int(eng.total_layers * drug["layer_pct"][1])
        handles = eng._attach_attention_scatter(0.05, start, end)
        self.assertEqual(len(handles), end - start)
        for h in handles:
            h.remove()

    def test_skips_layers_without_attention_module(self):
        eng = _fake_engine(num_layers=4,
                           layer_names=["attn_q", "w", "attn_q", "w"])
        handles = eng._attach_attention_scatter(0.1, 0, 4)
        self.assertEqual(len(handles), 2)  # only the two attn layers
        for h in handles:
            h.remove()


class TestApplyDegradationWithDrug(unittest.TestCase):
    def test_drug_scale_and_noise_applied_to_window(self):
        eng = _fake_engine(num_layers=4)
        eng.apply_degradation("C1", drug=resolve_drug("psilocybin", 1.0), noise_seed=0)
        weights = _weights(eng)
        # psilocybin window [0.35, 0.95] on 4 layers -> indices 1..2
        self.assertEqual(weights[0], 3.0)  # outside window: untouched
        self.assertEqual(weights[3], 3.0)
        for w in weights[1:3]:
            self.assertAlmostEqual(w, 3.0 * 0.86, delta=0.01)  # scale 0.86, tiny noise
        self.assertTrue(eng.backups)  # restore path available

    def test_drug_subnetwork_mlp_skips_non_mlp_params(self):
        eng = _fake_engine(num_layers=4)
        eng.apply_degradation("C1", drug=resolve_drug("ketamine", 1.0), noise_seed=0)
        # ketamine targets subnetwork "mlp"; fake params are named "w" -> all skipped
        self.assertEqual(_weights(eng), [3.0, 3.0, 3.0, 3.0])
        self.assertEqual(eng.backups, {})

    def test_drug_degradation_is_seed_deterministic(self):
        results = []
        for _ in range(2):
            eng = _fake_engine(num_layers=4)
            eng.apply_degradation("C1", drug=resolve_drug("lsd", 1.0), noise_seed=11)
            results.append(_weights(eng))
        self.assertEqual(results[0], results[1])

    def test_restore_after_drug_degradation(self):
        eng = _fake_engine(num_layers=4)
        eng.apply_degradation("C1", drug=resolve_drug("lsd", 1.0), noise_seed=0)
        eng.restore_clean_state()
        self.assertEqual(_weights(eng), [3.0, 3.0, 3.0, 3.0])
        self.assertEqual(eng.backups, {})


if __name__ == "__main__":
    unittest.main()
