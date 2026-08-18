"""Unit tests for the psychoactive drug catalog (eateot.drugs)."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# tests/test_drugs.py -> repo root (so `import eateot` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eateot import drugs as drugs_module
from eateot.drugs import (
    ADDITIVE_PRIMITIVES,
    ALL_PRIMITIVES,
    DRUG_PROFILES,
    RESOLVED_PRIMITIVES,
    curve_factor,
    list_drugs,
    load_drug_profiles,
    parse_stack,
    resolve_drug,
    resolve_stack,
    stack_label,
    validate_drug_profiles,
    validate_stack,
)

EXPECTED_DRUGS = {
    # hallucinogens
    "lsd", "psilocybin", "dmt", "mescaline", "2c-b", "salvia",
    # dissociatives
    "ketamine", "pcp", "dxm", "nitrous", "datura",
    # stimulants
    "caffeine", "amphetamine", "cocaine", "mdma",
    # depressants
    "alcohol", "benzodiazepine", "ghb", "opiate",
    # cannabinoids / placebo
    "thc", "cbd",
    # deliriants
    "dph",
    # enhancers
    "microdose_lsd", "nzt", "modafinil",
}


class TestCatalogLoad(unittest.TestCase):
    def test_all_expected_drugs_present(self):
        self.assertEqual(set(DRUG_PROFILES), EXPECTED_DRUGS)

    def test_list_drugs_sorted(self):
        self.assertEqual(list_drugs(), sorted(DRUG_PROFILES))

    def test_shipped_catalog_validates(self):
        # load_drug_profiles() already validated at import; re-check explicitly.
        validate_drug_profiles(DRUG_PROFILES)  # must not raise

    def test_every_drug_has_a_class_and_curve(self):
        for name, prof in DRUG_PROFILES.items():
            with self.subTest(drug=name):
                self.assertIn(prof["class"], drugs_module.DRUG_CLASSES)
                self.assertIn(prof["curve"], drugs_module.CURVES)

    def test_dose_curve_keys_are_known_primitives(self):
        for name, prof in DRUG_PROFILES.items():
            with self.subTest(drug=name):
                for primitive in prof.get("dose_curve") or {}:
                    self.assertIn(primitive, ALL_PRIMITIVES)

    def test_at_least_one_drug_per_class(self):
        classes = {prof["class"] for prof in DRUG_PROFILES.values()}
        self.assertGreaterEqual(len(classes), 7)  # 7 of the 8 classes shipped


class TestCurveFactor(unittest.TestCase):
    def test_linear(self):
        self.assertEqual(curve_factor("linear", 0.0), 0.0)
        self.assertEqual(curve_factor("linear", 1.0), 1.0)
        self.assertAlmostEqual(curve_factor("linear", 2.0, potency=0.5), 1.0)

    def test_gentle(self):
        self.assertAlmostEqual(curve_factor("gentle", 1.0), 1.0)
        self.assertAlmostEqual(curve_factor("gentle", 0.25), 0.5)

    def test_steep(self):
        self.assertAlmostEqual(curve_factor("steep", 0.5), 0.25)
        self.assertAlmostEqual(curve_factor("steep", 2.0), 4.0)

    def test_breakthrough(self):
        self.assertEqual(curve_factor("breakthrough", 0.4, breakthrough_at=0.5), 0.0)
        self.assertAlmostEqual(curve_factor("breakthrough", 1.0, breakthrough_at=0.5), 1.0)
        # potency is reached exactly at dose 1.0 (the "breakthrough" point)
        self.assertAlmostEqual(
            curve_factor("breakthrough", 1.0, potency=2.5, breakthrough_at=0.5), 2.5
        )

    def test_breakthrough_requires_threshold(self):
        with self.assertRaises(ValueError):
            curve_factor("breakthrough", 1.0)

    def test_negative_dose_rejected(self):
        with self.assertRaises(ValueError):
            curve_factor("linear", -0.1)

    def test_unknown_curve_rejected(self):
        with self.assertRaises(ValueError):
            curve_factor("exponential", 1.0)

    def test_all_curves_monotonic_in_dose(self):
        for curve in drugs_module.CURVES:
            doses = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]
            factors = [curve_factor(curve, d, breakthrough_at=0.5) for d in doses]
            self.assertEqual(factors, sorted(factors), msg=f"curve {curve} not monotonic")


class TestResolveDefaults(unittest.TestCase):
    """Every drug resolves to a full primitive set; dose 0 is fully neutral."""

    def test_dose_zero_is_neutral(self):
        for name in DRUG_PROFILES:
            with self.subTest(drug=name):
                prims = resolve_drug(name, 0.0)["primitives"]
                self.assertEqual(set(prims), RESOLVED_PRIMITIVES)
                self.assertEqual(prims["temperature"], 0.7)
                self.assertEqual(prims["repetition_penalty"], 1.0)
                self.assertEqual(prims["scale"], 1.0)
                self.assertEqual(prims["noise"], 0.0)

    def test_placebo_cbd_is_all_defaults_at_any_dose(self):
        for dose in (0.5, 1.0, 5.0):
            prims = resolve_drug("cbd", dose)["primitives"]
            self.assertEqual(prims["temperature"], 0.7)
            self.assertEqual(prims["repetition_penalty"], 1.0)
            self.assertEqual(prims["scale"], 1.0)
            self.assertEqual(prims["noise"], 0.0)
            self.assertEqual(prims["attention_scatter"], 0.0)

    def test_unknown_drug_raises_keyerror(self):
        with self.assertRaises(KeyError):
            resolve_drug("heroin")

    def test_negative_dose_raises(self):
        with self.assertRaises(ValueError):
            resolve_drug("lsd", -1.0)

    def test_output_metadata(self):
        resolved = resolve_drug("lsd", 1.0)
        self.assertEqual(resolved["name"], "lsd")
        self.assertEqual(resolved["class"], "hallucinogen")
        self.assertEqual(resolved["target_domain"], "visual_cortex")
        self.assertEqual(resolved["subnetwork"], "attn")
        self.assertEqual(resolved["layer_pct"], [0.65, 0.95])
        self.assertEqual(resolved["factor"], 1.0)
        self.assertIsNone(resolved["restore_fraction"])
        self.assertIn("hallucinogen", resolved["prompt_state"])


class TestResolveSemantics(unittest.TestCase):
    def test_lsd_attention_scatter_scales_with_dose(self):
        self.assertAlmostEqual(
            resolve_drug("lsd", 0.5)["primitives"]["attention_scatter"], 0.025)
        self.assertAlmostEqual(
            resolve_drug("lsd", 1.0)["primitives"]["attention_scatter"], 0.05)
        self.assertAlmostEqual(
            resolve_drug("lsd", 2.0)["primitives"]["attention_scatter"], 0.10)

    def test_temperature_is_delta_from_default(self):
        # lsd: +0.15/dose -> 0.85 at dose 1
        self.assertAlmostEqual(resolve_drug("lsd", 1.0)["primitives"]["temperature"], 0.85)
        # amphetamine: -0.10/dose -> 0.60 at dose 1
        self.assertAlmostEqual(
            resolve_drug("amphetamine", 1.0)["primitives"]["temperature"], 0.60)

    def test_repetition_penalty_induces_or_suppresses_loops(self):
        # mescaline: -0.08/dose -> 0.92 (loop induction)
        self.assertAlmostEqual(
            resolve_drug("mescaline", 1.0)["primitives"]["repetition_penalty"], 0.92)
        # amphetamine: +0.15/dose -> 1.15 (repetition suppression)
        self.assertAlmostEqual(
            resolve_drug("amphetamine", 1.0)["primitives"]["repetition_penalty"], 1.15)

    def test_scale_loss_folds_into_scale(self):
        # salvia: 0.35 * potency 2.5 = 0.875 loss -> scale 0.125
        resolved = resolve_drug("salvia", 1.0)
        self.assertAlmostEqual(resolved["primitives"]["scale"], 0.125)
        self.assertNotIn("scale_loss", resolved["primitives"])

    def test_negative_noise_means_suppression(self):
        # modafinil suppresses noise below the track baseline (signed, not
        # clamped) — its gentle curve makes the factor sqrt(2.0) at dose 2.
        expected = -0.00005 * (2.0 ** 0.5)
        self.assertAlmostEqual(resolve_drug("modafinil", 2.0)["primitives"]["noise"], expected)

    def test_additive_primitives_clamp_nonsense_negatives(self):
        prof = {"x": {
            "class": "placebo", "curve": "linear", "subnetwork": "all",
            "layer_pct": [0.0, 1.0],
            "dose_curve": {"logit_noise": -5.0, "attention_scatter": -1.0},
        }}
        prims = resolve_drug("x", 1.0, profiles=prof)["primitives"]
        self.assertEqual(prims["logit_noise"], 0.0)
        self.assertEqual(prims["attention_scatter"], 0.0)

    def test_breakthrough_below_threshold_is_neutral(self):
        """dmt below its breakthrough_at resolves to all-default primitives."""
        resolved = resolve_drug("dmt", 0.5)  # breakthrough_at 0.75
        self.assertEqual(resolved["factor"], 0.0)
        prims = resolved["primitives"]
        self.assertEqual(prims["temperature"], 0.7)
        self.assertEqual(prims["repetition_penalty"], 1.0)
        self.assertEqual(prims["scale"], 1.0)
        self.assertEqual(prims["noise"], 0.0)
        self.assertEqual(prims["attention_scatter"], 0.0)

    def test_dose_exceeds_cap_flag(self):
        self.assertFalse(resolve_drug("lsd", 2.0)["dose_exceeds_cap"])  # cap 3.0
        self.assertTrue(resolve_drug("lsd", 5.0)["dose_exceeds_cap"])

    def test_resolve_rejects_missing_subnetwork(self):
        prof = {"x": {"class": "placebo", "curve": "linear",
                       "layer_pct": [0.0, 1.0], "dose_curve": {}}}
        with self.assertRaises(ValueError):
            resolve_drug("x", 1.0, profiles=prof)

    def test_effect_magnitude_monotonic_with_dose(self):
        """|primitive| never shrinks as dose grows (more drug, stronger effect).

        Uses magnitude so signed primitives (noise suppression, terse
        verbosity) are covered too: modafinil's noise goes 0 -> -1e-4 -> -2e-4.
        """
        additive = ADDITIVE_PRIMITIVES - {"scale_loss"}
        for name in DRUG_PROFILES:
            with self.subTest(drug=name):
                prev = resolve_drug(name, 0.0)["primitives"]
                for dose in (0.5, 1.0, 2.0, 4.0):
                    cur = resolve_drug(name, dose)["primitives"]
                    for key in additive:
                        self.assertGreaterEqual(
                            abs(cur[key]), abs(prev[key]) - 1e-12,
                            msg=f"{key} shrank for {name} at dose {dose}",
                        )
                    prev = cur

    def test_nzt_restore_fraction_exposed(self):
        self.assertEqual(resolve_drug("nzt", 1.0)["restore_fraction"], 0.7)


class TestValidation(unittest.TestCase):
    @staticmethod
    def _profile(overrides: dict | None = None):
        prof = {"class": "placebo", "curve": "linear", "subnetwork": "all",
                "layer_pct": [0.0, 1.0], "dose_curve": {}}
        prof.update(overrides or {})
        return {"x": prof}

    def test_valid_minimal_profile_passes(self):
        validate_drug_profiles(self._profile())  # must not raise

    def test_bad_class_rejected(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"class": "not_a_class"}))

    def test_bad_curve_rejected(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"curve": "exponential"}))

    def test_bad_subnetwork_rejected(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"subnetwork": "conv"}))

    def test_bad_layer_pct_rejected(self):
        for bad in ([0.5, 0.2], [0.0, 1.5], [0.0], "all"):
            with self.subTest(layer_pct=bad):
                with self.assertRaises(ValueError):
                    validate_drug_profiles(self._profile({"layer_pct": bad}))

    def test_breakthrough_requires_threshold(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"curve": "breakthrough"}))

    def test_unknown_primitive_rejected(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"dose_curve": {"telepathy": 1.0}}))

    def test_bad_restore_fraction_rejected(self):
        with self.assertRaises(ValueError):
            validate_drug_profiles(self._profile({"restore_fraction": 1.5}))

    def test_errors_are_collected(self):
        prof = {
            "a": {"class": "bogus", "curve": "linear", "layer_pct": [0.0, 1.0],
                  "dose_curve": {}},
            "b": {"class": "placebo", "curve": "bogus", "layer_pct": [0.0, 1.0],
                  "dose_curve": {}},
        }
        with self.assertRaises(ValueError) as ctx:
            validate_drug_profiles(prof)
        msg = str(ctx.exception)
        self.assertIn("drug 'a'", msg)
        self.assertIn("drug 'b'", msg)


class TestParseStack(unittest.TestCase):
    def test_parses_name_dose_pairs(self):
        self.assertEqual(parse_stack("lsd@1.0,thc@0.5"),
                         [{"drug": "lsd", "dose": 1.0},
                          {"drug": "thc", "dose": 0.5}])

    def test_bare_name_defaults_to_dose_one(self):
        self.assertEqual(parse_stack("lsd"), [{"drug": "lsd", "dose": 1.0}])

    def test_mixed_bare_and_dosed(self):
        self.assertEqual(parse_stack("lsd,cbd@2"),
                         [{"drug": "lsd", "dose": 1.0},
                          {"drug": "cbd", "dose": 2.0}])

    def test_ignores_whitespace(self):
        self.assertEqual(parse_stack(" lsd @ 1.0 , thc @ 0.5 "),
                         [{"drug": "lsd", "dose": 1.0},
                          {"drug": "thc", "dose": 0.5}])

    def test_empty_spec_rejected(self):
        with self.assertRaises(ValueError):
            parse_stack("")
        with self.assertRaises(ValueError):
            parse_stack(",,,,")

    def test_malformed_dose_rejected(self):
        with self.assertRaises(ValueError):
            parse_stack("lsd@abc")

    def test_bare_at_rejected(self):
        with self.assertRaises(ValueError):
            parse_stack("@1.0")


class TestStackLabel(unittest.TestCase):
    def test_label_format(self):
        comps = [{"drug": "lsd", "dose": 1.0}, {"drug": "thc", "dose": 0.5}]
        self.assertEqual(stack_label(comps), "lsd@1+thc@0.5")

    def test_round_trips_with_parse(self):
        comps = [{"drug": "lsd", "dose": 1.5}, {"drug": "cbd", "dose": 2.0}]
        self.assertEqual(parse_stack(stack_label(comps)), comps)

    def test_single_component(self):
        self.assertEqual(stack_label([{"drug": "nzt", "dose": 1.0}]), "nzt@1")


class TestResolveStack(unittest.TestCase):
    def test_single_component_is_plain_drug(self):
        self.assertEqual(resolve_stack([{"drug": "lsd", "dose": 2.0}]),
                         resolve_drug("lsd", 2.0))

    def test_additive_primitives_sum(self):
        # lsd attention_scatter 0.05/dose + mescaline 0.04/dose @ dose 1 each
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "mescaline", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["attention_scatter"], 0.05 + 0.04)

    def test_noise_sums(self):
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["noise"], 0.00008 + 0.0001)

    def test_scale_loss_combines(self):
        # salvia loss 0.35*2.5 = 0.875; ghb loss 0.06 -> total 0.935 -> scale 0.065
        spec = resolve_stack([{"drug": "salvia", "dose": 1.0},
                              {"drug": "ghb", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["scale"], 0.065, places=6)

    def test_temperature_deltas_sum(self):
        # lsd +0.15 and thc +0.10 -> 0.95; lsd +0.15 and ghb -0.15 -> 0.70
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["temperature"], 0.95)
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "ghb", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["temperature"], 0.70)

    def test_repetition_penalty_deltas_sum(self):
        # amphetamine +0.15, mescaline -0.08 -> 1.07
        spec = resolve_stack([{"drug": "amphetamine", "dose": 1.0},
                              {"drug": "mescaline", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["repetition_penalty"], 1.07)

    def test_verbosity_bias_sums_signed(self):
        # lsd +2, ghb -3 -> -1 (terse wins)
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "ghb", "dose": 1.0}])
        self.assertAlmostEqual(spec["primitives"]["verbosity_bias"], -1.0)

    def test_layer_window_is_union(self):
        # lsd [0.65, 0.95] + ketamine [0.30, 0.70] -> [0.30, 0.95]
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "ketamine", "dose": 1.0}])
        self.assertEqual(spec["layer_pct"], [0.30, 0.95])

    def test_shared_subnetwork_is_kept(self):
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "mescaline", "dose": 1.0}])
        self.assertEqual(spec["subnetwork"], "attn")

    def test_conflicting_subnetworks_fall_back_to_all(self):
        # lsd (attn) + thc (all) -> all
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        self.assertEqual(spec["subnetwork"], "all")

    def test_prompt_states_concatenate(self):
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 1.0}])
        self.assertIn("hallucinogen", spec["prompt_state"])
        self.assertIn("very high", spec["prompt_state"])
        self.assertGreater(spec["prompt_state"].count("\n"), 0)

    def test_restore_fraction_is_max_across_components(self):
        # nzt 0.7 + cbd (none) -> 0.7; lsd (none) + nzt 0.7 -> 0.7
        spec = resolve_stack([{"drug": "nzt", "dose": 1.0},
                              {"drug": "cbd", "dose": 1.0}])
        self.assertEqual(spec["restore_fraction"], 0.7)
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "nzt", "dose": 1.0}])
        self.assertEqual(spec["restore_fraction"], 0.7)

    def test_merged_spec_metadata(self):
        spec = resolve_stack([{"drug": "lsd", "dose": 1.0},
                              {"drug": "thc", "dose": 0.5}])
        self.assertEqual(spec["name"], "lsd@1+thc@0.5")
        self.assertEqual(spec["class"], "stack")
        self.assertEqual(spec["target_domain"], "stack")
        self.assertIsNone(spec["dose"])
        self.assertIsNone(spec["factor"])
        self.assertEqual(spec["components"],
                         [{"name": "lsd", "dose": 1.0},
                          {"name": "thc", "dose": 0.5}])

    def test_resolved_keys_invariant_for_stacks(self):
        for comps in (
            [{"drug": "lsd", "dose": 1.0}, {"drug": "thc", "dose": 0.5}],
            [{"drug": "salvia", "dose": 0.8}, {"drug": "nzt", "dose": 1.0}],
            [{"drug": "dph", "dose": 2.0}, {"drug": "caffeine", "dose": 3.0}],
        ):
            with self.subTest(comps=comps):
                spec = resolve_stack(comps)
                self.assertEqual(set(spec["primitives"]), RESOLVED_PRIMITIVES)
                self.assertEqual(len(spec["components"]), len(comps))

    def test_unknown_drug_raises(self):
        # validate_stack runs first and collects errors -> ValueError (CLI-safe)
        with self.assertRaises(ValueError) as ctx:
            resolve_stack([{"drug": "lsd", "dose": 1.0},
                           {"drug": "heroin", "dose": 1.0}])
        self.assertIn("heroin", str(ctx.exception))

    def test_negative_dose_raises(self):
        with self.assertRaises(ValueError):
            resolve_stack([{"drug": "lsd", "dose": -0.5}])

    def test_empty_stack_raises(self):
        with self.assertRaises(ValueError):
            resolve_stack([])


class TestValidateStack(unittest.TestCase):
    def test_valid_stack_passes(self):
        validate_stack([{"drug": "lsd", "dose": 1.0},
                        {"drug": "thc", "dose": 0.5}])  # must not raise

    def test_unknown_drug_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_stack([{"drug": "heroin", "dose": 1.0}])
        self.assertIn("heroin", str(ctx.exception))

    def test_negative_dose_rejected(self):
        with self.assertRaises(ValueError):
            validate_stack([{"drug": "lsd", "dose": -1.0}])

    def test_empty_stack_rejected(self):
        with self.assertRaises(ValueError):
            validate_stack([])

    def test_non_mapping_component_rejected(self):
        with self.assertRaises(ValueError):
            validate_stack(["lsd"])

    def test_errors_are_collected(self):
        with self.assertRaises(ValueError) as ctx:
            validate_stack([{"drug": "heroin", "dose": 1.0},
                            {"drug": "lsd", "dose": -2.0}])
        msg = str(ctx.exception)
        self.assertIn("heroin", msg)
        self.assertIn("component 1", msg)


class TestCatalogOverride(unittest.TestCase):
    def test_eateot_drugs_file_env_override(self):
        """EATEOT_DRUGS_FILE points load_drug_profiles at a custom catalog."""
        import yaml

        catalog = {
            "name": "drugs", "version": 1,
            "drugs": {
                "trial_compound": {
                    "class": "placebo", "curve": "linear", "subnetwork": "all",
                    "layer_pct": [0.0, 1.0],
                    "dose_curve": {"temperature": 0.1},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "drugs.yaml")
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(catalog, f)
            with mock.patch.object(drugs_module, "DRUGS_FILE", Path(path)):
                loaded = load_drug_profiles()
        self.assertEqual(set(loaded), {"trial_compound"})


if __name__ == "__main__":
    unittest.main()
