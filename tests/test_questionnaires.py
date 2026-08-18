"""Unit tests for the versioned YAML questionnaire loader."""

import unittest
from unittest import mock

import eateot.questionnaires as questionnaires
from eateot.questionnaires import (
    list_batteries,
    list_questionnaires,
    load_battery,
    load_presets,
    load_simple_questions,
)


class TestQuestionnaireLoader(unittest.TestCase):
    def test_default_battery_has_five_tiers(self):
        battery = load_battery("iq_battery")
        self.assertEqual(len(battery), 5)
        tiers = [q["tier"] for q in battery]
        self.assertEqual(tiers, [1, 2, 3, 4, 5])
        # Question dicts expose the keys evaluate_response expects.
        for q in battery:
            self.assertIn("question", q)
            self.assertIn("ground_truth_anchors", q)
            self.assertIn("max_points", q)

    def test_mini_battery_is_different_questionnaire(self):
        battery = load_battery("iq_battery_mini")
        self.assertEqual(len(battery), 2)
        self.assertNotEqual(
            battery[0]["question"],
            load_battery("iq_battery")[0]["question"],
        )

    def test_load_battery_default_is_iq_battery(self):
        self.assertEqual(load_battery(), load_battery("iq_battery"))

    def test_missing_questionnaire_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_battery("does_not_exist_xyz")

    def test_presets_shape(self):
        presets = load_presets()
        self.assertEqual(len(presets), 5)
        for key, (title, prompt) in presets.items():
            self.assertTrue(key.isdigit())
            self.assertIsInstance(title, str)
            self.assertIsInstance(prompt, str)
            self.assertTrue(prompt.endswith("?") or "?" in prompt)

    def test_simple_questions_shape(self):
        questions = load_simple_questions("brain_benchmark")
        self.assertEqual(len(questions), 2)
        for q in questions:
            self.assertIn("question", q)
            self.assertIn("expected", q)

    def test_list_batteries_filters_non_battery_files(self):
        batteries = list_batteries()
        self.assertIn("iq_battery", batteries)
        self.assertIn("iq_battery_mini", batteries)
        # presets.yaml and brain_benchmark.yaml have a different schema and
        # must not be advertised as selectable batteries.
        self.assertNotIn("presets", batteries)
        self.assertNotIn("brain_benchmark", batteries)
        # list_questionnaires is broader than list_batteries.
        self.assertGreater(len(list_questionnaires()), len(batteries))

    def test_clinical_battery_shape(self):
        battery = load_battery("clinical_battery")
        self.assertEqual(len(battery), 8)
        # Anchored questions expose the keys evaluate_response expects.
        for q in battery:
            self.assertIn("question", q)
            self.assertIn("max_points", q)
        self.assertIn("ground_truth_anchors", battery[0])
        # The fluency question uses the alternate schema.
        fluency = next(q for q in battery if q.get("type") == "fluency")
        self.assertEqual(fluency["fluency"]["category"], "animals")
        self.assertNotIn("ground_truth_anchors", fluency)

    def test_clinical_battery_is_advertised(self):
        self.assertIn("clinical_battery", list_batteries())

    def test_language_battery_shape(self):
        battery = load_battery("language_battery")
        self.assertEqual(len(battery), 8)
        self.assertIn("ground_truth_anchors", battery[1])  # naming is anchored
        # Phonemic fluency item uses the `letter` key, not a category lexicon.
        fluency = next(q for q in battery if q.get("type") == "fluency")
        self.assertEqual(fluency["fluency"]["letter"], "f")
        self.assertNotIn("category", fluency["fluency"])
        self.assertIn("language_battery", list_batteries())

    def test_visual_battery_shape(self):
        battery = load_battery("visual_battery")
        self.assertEqual(len(battery), 5)
        self.assertEqual([q["tier"] for q in battery], [1, 2, 3, 4, 5])
        for q in battery:
            self.assertIn("question", q)
            self.assertIn("ground_truth_anchors", q)
            self.assertIn("max_points", q)
        # Domains probe the visual cortex (ASCII art, spatial, hallucinated scene).
        self.assertIn("Visual Output Fidelity", battery[0]["domain"])
        self.assertIn("Hallucinated Scene", battery[3]["domain"])
        # Drawing tiers use the self-report marker protocol (no question-word
        # leakage into anchors; markers are underscore-free because the anchor
        # matcher strips `_`); reasoning tiers use pure discriminators.
        self.assertEqual(battery[0]["ground_truth_anchors"],
                         [["roofok"], ["doorok"], ["windowok"]])
        self.assertEqual(battery[2]["ground_truth_anchors"], [["diamond", "rhombus"]])
        self.assertIn("visual_battery", list_batteries())

    def test_executive_battery_shape(self):
        battery = load_battery("executive_battery")
        self.assertEqual(len(battery), 8)
        # All executive items are anchored; digit-span anchors are numeric.
        for q in battery:
            self.assertIn("ground_truth_anchors", q)
        self.assertEqual(
            battery[0]["ground_truth_anchors"], [["7"], ["3"], ["9"], ["1"], ["5"]]
        )
        self.assertIn("executive_battery", list_batteries())

    def test_env_dir_override(self):
        import importlib
        with mock.patch.dict("os.environ", {"EATEOT_QUESTIONNAIRE_DIR": "/tmp/nope"}):
            reloaded = importlib.reload(questionnaires)
            self.assertEqual(str(reloaded.QUESTIONNAIRE_DIR), "/tmp/nope")
            with self.assertRaises(FileNotFoundError):
                reloaded.load_battery("iq_battery")
        # Restore the module's default state so later tests see the real config dir.
        importlib.reload(questionnaires)
        self.assertEqual(
            str(questionnaires.QUESTIONNAIRE_DIR),
            str(questionnaires.DEFAULT_QUESTIONNAIRE_DIR),
        )

    def test_numeric_anchors_coerced_to_strings(self):
        battery = load_battery("iq_battery_mini")
        for synonym in battery[0]["ground_truth_anchors"][0]:
            self.assertIsInstance(synonym, str)
        battery = load_battery("iq_battery")
        # "yes" / "no" / "95" anchors must survive YAML 1.1 boolean/int parsing.
        for question in battery:
            for group in question["ground_truth_anchors"]:
                for synonym in group:
                    self.assertIsInstance(synonym, str)


if __name__ == "__main__":
    unittest.main()
