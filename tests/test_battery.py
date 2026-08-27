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

    def test_echo_of_question_fails_deliberately(self):
        # Reproducing the prompt verbatim (echolalia) must score 0 even though
        # the question text itself contains anchor words: Tier 3's "State Yes
        # or No" would otherwise match the "yes" anchor, and the echoed
        # premises contain "all rocks" / "premise 1".
        item = IQ_TEST_BATTERY[2]  # Tier 3 counterfactual syllogism
        score, status, pct = evaluate_response(
            item["question"],
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)
        self.assertIn("Echolalia", status)
        self.assertEqual(pct, 0.0)

    def test_echo_rejected_even_when_question_mentions_answer_word(self):
        # Tier 6's question says "Answer Yes or No" and quotes premises
        # containing the "mipsters"/"premise 2" anchors — a pure echo must
        # not collect any of them.
        item = IQ_TEST_BATTERY[5]
        score, status, pct = evaluate_response(
            item["question"],
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, 0)
        self.assertIn("Echolalia", status)
        self.assertEqual(pct, 0.0)

    def test_terse_answer_is_not_an_echo(self):
        # A short correct answer quoting only part of the premises is not
        # flagged as echolalia and still scores full marks.
        item = IQ_TEST_BATTERY[2]
        score, status, pct = evaluate_response(
            "Yes. Based on Premise 1, all rocks can fly and a ruby is a rock.",
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, item["max_points"])
        self.assertIn("PASSED", status)
        self.assertEqual(pct, 100.0)

    def test_partial_credit_has_finer_granularity(self):
        # "D is the heaviest" answers one of the two relational anchors; the
        # fractional scheme yields a clean 50% instead of the old binary 0%.
        item = IQ_TEST_BATTERY[3]  # Tier 4 relational ordering
        score, status, pct = evaluate_response(
            "D is the heaviest.",
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, int(round(item["max_points"] * 0.5)))
        self.assertEqual(pct, 50.0)
        self.assertIn("PARTIAL", status)

    def test_natural_language_full_answer_scores_max(self):
        # "D is the heaviest, C is the lightest" matches no anchor phrase
        # literally ("heaviest: d", "box c", ...) and used to score 0;
        # content-word coverage now credits it fully.
        item = IQ_TEST_BATTERY[3]
        score, status, pct = evaluate_response(
            "D is the heaviest, C is the lightest.",
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, item["max_points"])
        self.assertIn("PASSED", status)
        self.assertEqual(pct, 100.0)

    def test_sub_phrase_overlap_earns_fractional_credit(self):
        # Only half of the "classified as a rock" synonym's content words are
        # present ("rock"), so the group earns 0.5 credit -> 25% overall for
        # Tier 3's two groups, instead of a hard 0% or 100%.
        item = IQ_TEST_BATTERY[2]
        score, status, pct = evaluate_response(
            "No, because a ruby is not a rock.",
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, int(round(item["max_points"] * 0.25)))
        self.assertEqual(pct, 25.0)
        self.assertIn("PARTIAL", status)

    def test_spatial_leftmost_scores_max(self):
        # Tier 5, spatial Q1: the row is [GREEN] [RED] [BLUE], so the leftmost
        # is green. The anchor color is absent from the question sentence, so
        # both natural language and terse answers score full marks.
        item = IQ_TEST_BATTERY[4]
        self.assertEqual(item["domain"], "Spatial Reasoning")
        for answer in ("green", "The leftmost object is green.",
                       "GREEN", "leftmost: green"):
            score, status, pct = evaluate_response(
                answer,
                item["ground_truth_anchors"],
                item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, item["max_points"], answer)
            self.assertIn("PASSED", status)
            self.assertEqual(pct, 100.0)

    def test_spatial_leftmost_wrong_answer_fails(self):
        # A wrong color gets nothing — even "red", which IS in the question.
        item = IQ_TEST_BATTERY[4]
        for answer in ("blue", "red", "The leftmost is blue."):
            score, status, pct = evaluate_response(
                answer,
                item["ground_truth_anchors"],
                item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, 0, answer)
            self.assertIn("FAILED", status)
            self.assertEqual(pct, 0.0)

    def test_spatial_leftmost_echo_of_prompt_earns_nothing(self):
        # Echoing the question text cannot name the answer color: "green"
        # never appears in the question sentence, only inside the diagram.
        item = IQ_TEST_BATTERY[4]
        score, status, pct = evaluate_response(
            "Three colored objects sit in a row: [GREEN] [RED] [BLUE]. "
            "Which object is the leftmost? Answer with the object color.",
            item["ground_truth_anchors"],
            item["max_points"],
            question=item["question"],
        )
        self.assertEqual(score, 0)
        self.assertIn("FAILED", status)

    def test_spatial_compass_scores_max(self):
        # Tier 5, spatial Q2: SQUARE is north of TRIANGLE, CIRCLE east of
        # SQUARE, STAR west of TRIANGLE -> northernmost is the circle. The
        # answer name appears nowhere in the question sentence.
        item = IQ_TEST_BATTERY[5]
        self.assertEqual(item["domain"], "Spatial Reasoning")
        for answer in ("circle", "The circle is the northernmost.",
                       "CIRCLE"):
            score, status, pct = evaluate_response(
                answer,
                item["ground_truth_anchors"],
                item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, item["max_points"], answer)
            self.assertIn("PASSED", status)
            self.assertEqual(pct, 100.0)

    def test_spatial_compass_wrong_answer_fails(self):
        # Confusing the compass relations (a classic mild-degradation failure)
        # gets nothing — even naming objects that ARE in the question.
        item = IQ_TEST_BATTERY[5]
        for answer in ("square", "triangle", "star",
                       "The northernmost is the square."):
            score, status, pct = evaluate_response(
                answer,
                item["ground_truth_anchors"],
                item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, 0, answer)
            self.assertIn("FAILED", status)
            self.assertEqual(pct, 0.0)

    def test_spatial_partial_credit_across_questions(self):
        # Two spatial questions of 10 pts each: answering only the leftmost
        # one earns half the tier points (10/20) instead of a binary 0.
        q_left, q_compass = IQ_TEST_BATTERY[4], IQ_TEST_BATTERY[5]
        s1, _, p1 = evaluate_response(
            "green", q_left["ground_truth_anchors"], q_left["max_points"],
            question=q_left["question"],
        )
        s2, _, p2 = evaluate_response(
            "star", q_compass["ground_truth_anchors"], q_compass["max_points"],
            question=q_compass["question"],
        )
        self.assertEqual(s1 + s2, 10)   # 10 + 0
        self.assertEqual(p1, 100.0)
        self.assertEqual(p2, 0.0)

    def test_spatial_pattern_next_symbol_scores_max(self):
        # Tier 5, pattern Q: [Δ] [□] [○] repeats, so the 6th symbol is ○.
        # "circle" appears nowhere in the question sentence.
        item = IQ_TEST_BATTERY[6]
        self.assertEqual(item["target_iq"], "110 - Recurring Symbol Pattern (Raven-style)")
        # The answer is scored by shape NAME (the matcher is word-based).
        for answer in ("circle", "The next symbol is the circle.", "CIRCLE"):
            score, status, pct = evaluate_response(
                answer, item["ground_truth_anchors"], item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, item["max_points"], answer)
            self.assertIn("PASSED", status)
            self.assertEqual(pct, 100.0)

    def test_spatial_pattern_wrong_symbol_fails(self):
        # Picking a symbol that IS in the diagram (square/triangle) gets zero.
        item = IQ_TEST_BATTERY[6]
        for answer in ("square", "triangle", "The next symbol is the square."):
            score, status, pct = evaluate_response(
                answer, item["ground_truth_anchors"], item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, 0, answer)
            self.assertIn("FAILED", status)
            self.assertEqual(pct, 0.0)

    def test_spatial_matrix_missing_cell_scores_max(self):
        # Tier 5, matrix Q: Latin square — row 3 / col 3 lacks □, so the
        # missing cell is the square. "square" appears nowhere in the sentence.
        item = IQ_TEST_BATTERY[7]
        self.assertEqual(item["target_iq"], "115 - 3x3 Symbol Matrix (Raven-style)")
        # The answer is scored by shape NAME (the matcher is word-based).
        for answer in ("square", "The missing symbol is the square.", "SQUARE"):
            score, status, pct = evaluate_response(
                answer, item["ground_truth_anchors"], item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, item["max_points"], answer)
            self.assertIn("PASSED", status)
            self.assertEqual(pct, 100.0)

    def test_spatial_matrix_wrong_symbol_fails(self):
        # Guessing the other symbols (circle/triangle) gets zero — even though
        # their glyphs are in the diagram and the sentence names 'symbol'.
        item = IQ_TEST_BATTERY[7]
        for answer in ("circle", "triangle", "The missing symbol is the circle."):
            score, status, pct = evaluate_response(
                answer, item["ground_truth_anchors"], item["max_points"],
                question=item["question"],
            )
            self.assertEqual(score, 0, answer)
            self.assertIn("FAILED", status)
            self.assertEqual(pct, 0.0)


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

    def test_echolalia_is_fully_deteriorated(self):
        item = IQ_TEST_BATTERY[2]
        self.assertEqual(grade_deterioration(item["question"], item), 100.0)

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
