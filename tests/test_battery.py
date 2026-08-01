"""Unit tests for the pure scoring logic (evaluate_response)."""

import unittest

from eateot.battery import IQ_TEST_BATTERY, evaluate_response

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


if __name__ == "__main__":
    unittest.main()
