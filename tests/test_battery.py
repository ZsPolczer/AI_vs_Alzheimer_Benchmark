"""Unit tests for the pure scoring logic (evaluate_response, evaluate_fluency, grade_deterioration)."""

import unittest

from eateot.battery import (
    IQ_TEST_BATTERY,
    evaluate_fluency,
    evaluate_question,
    evaluate_response,
    grade_deterioration,
)

# Anchor set used by Tier 1 (Categorical Reasoning): ["wood"] and ["metal", ...]
TIER1_ANCHORS = IQ_TEST_BATTERY[0]["ground_truth_anchors"]
TIER1_MAX = IQ_TEST_BATTERY[0]["max_points"]


class TestEvaluateResponse(unittest.TestCase):
    def test_null_output_fails(self):
        score, status, pct = evaluate_response("", TIER1_ANCHORS, TIER1_MAX)
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)
        self.assertEqual(pct, 0.0)

    def test_full_match_scores_max(self):
        score, status, pct = evaluate_response(
            "Wood does not belong because it is not a metal; the others are metals.",
            TIER1_ANCHORS,
            TIER1_MAX,
        )
        self.assertEqual(score, TIER1_MAX)
        self.assertIn("PASSED", status)
        self.assertEqual(pct, 100.0)

    def test_partial_match_scores_half(self):
        # Only the "wood" anchor group matches; no metal/conductor synonym present.
        score, status, pct = evaluate_response(
            "The answer is wood.",
            TIER1_ANCHORS,
            TIER1_MAX,
        )
        self.assertEqual(score, int(round(TIER1_MAX * 0.5)))
        self.assertIn("PARTIAL", status)

    def test_no_match_fails(self):
        score, status, pct = evaluate_response(
            "The answer is Gold because it shines.",
            TIER1_ANCHORS,
            TIER1_MAX,
        )
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)

    def test_perseveration_loop_fails(self):
        score, status, _ = evaluate_response(
            "the answer is wood the answer is wood the answer is wood the answer is wood",
            TIER1_ANCHORS,
            TIER1_MAX,
        )
        self.assertEqual(score, 0)
        self.assertIn("Perseveration", status)

    def test_boxed_latex_is_unwrapped(self):
        # Tier 2 expects "95" and a double/multiply anchor.
        anchors = IQ_TEST_BATTERY[1]["ground_truth_anchors"]
        score, status, _ = evaluate_response(
            r"The rule is multiply by 2 and add 1, so the answer is \boxed{95}.",
            anchors,
            IQ_TEST_BATTERY[1]["max_points"],
        )
        self.assertIn("PASSED", status)


class TestEvaluateFluency(unittest.TestCase):
    """Category-fluency scoring (used by the clinical battery)."""

    def test_null_output_fails(self):
        score, status, pct, metrics = evaluate_fluency("", "animals", 8, 12)
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)
        self.assertEqual(pct, 0.0)
        self.assertEqual(metrics["distinct"], 0)

    def test_full_fluency_scores_max(self):
        # 8 distinct animals, no repeats -> full 12 points.
        score, status, pct, metrics = evaluate_fluency(
            "cat, dog, bird, fish, horse, cow, pig, sheep", "animals", 8, 12
        )
        self.assertEqual(score, 12)
        self.assertIn("PASSED", status)
        self.assertEqual(pct, 100.0)
        self.assertEqual(metrics["distinct"], 8)
        self.assertEqual(metrics["repeats"], 0)

    def test_partial_fluency_is_proportional(self):
        # 4 distinct of 8 target -> half the points.
        score, status, pct, _ = evaluate_fluency(
            "cat, dog, bird, fish", "animals", 8, 12
        )
        self.assertEqual(score, 6)
        self.assertIn("PARTIAL", status)
        self.assertEqual(pct, 50.0)

    def test_repeats_are_penalized(self):
        # 8 distinct + 2 repeats -> 12 - 2 = 10 points.
        score, status, pct, metrics = evaluate_fluency(
            "cat, dog, bird, fish, horse, cow, pig, sheep, cat, dog",
            "animals", 8, 12,
        )
        self.assertEqual(score, 10)
        self.assertEqual(metrics["repeats"], 2)

    def test_lexicon_filters_non_animals(self):
        # Furniture words and fillers are not counted as animals.
        score, status, pct, metrics = evaluate_fluency(
            "cat, dog, table, chair, and a chair, the cat", "animals", 8, 12
        )
        self.assertEqual(metrics["distinct"], 2)  # cat, dog
        self.assertEqual(metrics["repeats"], 1)   # repeated cat
        self.assertEqual(score, 2)                 # round(12*2/8)=3, minus 1 repeat

    def test_pluralization_counts_once(self):
        score, status, pct, metrics = evaluate_fluency(
            "dogs, cats, a dog", "animals", 8, 12
        )
        self.assertEqual(metrics["distinct"], 2)  # dog, cat
        self.assertEqual(metrics["repeats"], 1)

    def test_type_token_ratio_metric(self):
        _, _, _, metrics = evaluate_fluency(
            "cat, dog, cat, bird", "animals", 8, 12
        )
        self.assertEqual(metrics["type_token_ratio"], 0.75)  # 3 distinct / 4 total

    def test_fruit_lexicon_counts(self):
        score, status, pct, metrics = evaluate_fluency(
            "apple, banana, orange, grape, kiwi, table, chair", "fruits", 8, 12
        )
        self.assertEqual(metrics["distinct"], 5)  # furniture words excluded
        self.assertEqual(metrics["repeats"], 0)

    def test_vehicle_lexicon_counts(self):
        score, status, pct, metrics = evaluate_fluency(
            "car, truck, bus, bicycle, airplane, and a car", "vehicles", 8, 12
        )
        self.assertEqual(metrics["distinct"], 5)  # car, truck, bus, bicycle, airplane
        self.assertEqual(metrics["repeats"], 1)   # repeated car

    def test_root_s_words_survive_singularization(self):
        # 'walrus' -> 'walru' under naive stemming; the lookup fallback must
        # still count root-s animals (and their plural stems).
        _, _, _, metrics = evaluate_fluency(
            "walrus, octopus, rhinoceros, a walrus, bus, car", "animals", 8, 12
        )
        self.assertEqual(metrics["distinct"], 3)  # walrus, octopus, rhinoceros
        self.assertEqual(metrics["repeats"], 1)   # repeated walrus

    def test_letter_fluency_counts_prefixes(self):
        # Only words starting with the letter count; non-F words, bare letters,
        # and function words ('for', 'from') are excluded.
        score, status, pct, metrics = evaluate_fluency(
            "fox, frog, fire, table, fox, " + "f, for, from, five, first, fast",
            "", 8, 12, letter="f",
        )
        # fox frog fire five first fast = 6 distinct, 1 repeat
        self.assertEqual(metrics["distinct"], 6)
        self.assertEqual(metrics["repeats"], 1)

    def test_letter_fluency_rejects_other_letters(self):
        score, status, pct, metrics = evaluate_fluency(
            "dog, cat, bird", "", 8, 12, letter="f"
        )
        self.assertEqual(metrics["distinct"], 0)
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)

    def test_letter_fluency_case_insensitive(self):
        score, status, pct, metrics = evaluate_fluency(
            "Fox, FROG, Fire", "", 8, 12, letter="F"
        )
        self.assertEqual(metrics["distinct"], 3)

    def test_letter_fluency_excludes_function_words(self):
        # FAS convention: 'for', 'from', 'five' counts, but function words like
        # 'for'/'from' are NOT credited.
        score, status, pct, metrics = evaluate_fluency(
            "fox, frog, fire, forest, for, from, five, first, fast",
            "", 8, 12, letter="f",
        )
        self.assertEqual(metrics["distinct"], 7)  # fox frog fire forest five first fast
        self.assertEqual(metrics["repeats"], 0)

    def test_malformed_fluency_question_raises(self):
        with self.assertRaises(ValueError):
            evaluate_question("cat, dog", {"type": "fluency", "max_points": 12})

    def test_question_dispatcher(self):
        # Anchored questions go through evaluate_response (metrics is None) ...
        score, status, pct, metrics = evaluate_question(
            "Wood is not a metal.",
            {"max_points": 15, "ground_truth_anchors": TIER1_ANCHORS},
        )
        self.assertEqual(score, TIER1_MAX)
        self.assertIsNone(metrics)
        # ... while fluency questions carry metrics.
        item = {
            "type": "fluency", "max_points": 12,
            "fluency": {"category": "animals", "target": 8},
        }
        score, status, pct, metrics = evaluate_question("cat, dog, bird", item)
        self.assertEqual(metrics["distinct"], 3)
        self.assertIsNotNone(metrics)

class TestGradeDeterioration(unittest.TestCase):
    """Tests for the 0-100 numeric deterioration grade (higher = worse)."""

    def test_null_output_is_fully_deteriorated(self):
        self.assertEqual(grade_deterioration(""), 100.0)
        self.assertEqual(grade_deterioration("[NO OUTPUT GENERATED]"), 100.0)

    def test_perfect_anchored_answer_scores_zero(self):
        # Full logical match, no repetition, no loop -> pristine (0.0).
        item = IQ_TEST_BATTERY[0]
        grade = grade_deterioration(
            "Wood does not belong because it is not a metal; the others are metals.",
            item,
        )
        self.assertEqual(grade, 0.0)

    def test_wrong_answer_scores_at_least_half(self):
        # No anchors matched -> correctness 0.0 -> grade >= 50 before other terms.
        item = IQ_TEST_BATTERY[0]
        grade = grade_deterioration("The gold is the odd one out because it shines.", item)
        self.assertGreaterEqual(grade, 50.0)

    def test_perseveration_loop_scores_high(self):
        # A genuinely consecutive repeated phrase trips the loop detector.
        item = IQ_TEST_BATTERY[0]
        loop = "the rock is heavy " * 5
        grade = grade_deterioration(loop.strip(), item)
        self.assertGreaterEqual(grade, 75.0)

    def test_repetitive_answer_scores_higher_than_fluent_one(self):
        item = IQ_TEST_BATTERY[0]
        fluent = "Wood is not a metal; gold, silver and copper are metals."
        repetitive = "Wood wood wood wood wood is not a metal metal metal metal."
        self.assertGreater(grade_deterioration(repetitive, item),
                           grade_deterioration(fluent, item))

    def test_clean_response_overlap_reduces_grade(self):
        # Supplying the undegraded reference (high overlap) lowers the grade.
        text = "Wood is not a metal."
        without = grade_deterioration(text)
        with_clean = grade_deterioration(text, clean_response=text)
        self.assertLess(with_clean, without)

    def test_grade_bounds(self):
        item = IQ_TEST_BATTERY[0]
        for resp in ("", "correct answer here", "x" * 500):
            grade = grade_deterioration(resp, item)
            self.assertGreaterEqual(grade, 0.0)
            self.assertLessEqual(grade, 100.0)


if __name__ == "__main__":
    unittest.main()
