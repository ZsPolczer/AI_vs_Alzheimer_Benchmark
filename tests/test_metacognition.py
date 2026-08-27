"""Unit tests for the metacognition monitor (eateot.metacognition)."""

import tempfile
import unittest
from pathlib import Path

from eateot.metacognition import (
    profile_text,
    profile_windows,
    trace_line,
    write_jsonl,
)


class TestProfileText(unittest.TestCase):
    def test_healthy_self_reference_scores_high_coherence(self):
        prof = profile_text(
            "I think the answer is wood. I know it is not a metal, "
            "and I am sure the other three are metals."
        )
        self.assertGreater(prof["self_ref"], 0)
        self.assertEqual(prof["doubt"], 0)
        self.assertGreater(prof["coherence"], 85)

    def test_doubt_lowers_coherence(self):
        prof = profile_text(
            "Maybe the answer is wood, but I am not sure. Perhaps it "
            "is gold, I don't know, I am not certain."
        )
        self.assertGreater(prof["doubt"], 0)
        self.assertLess(prof["coherence"], 90)

    def test_disfluency_loop_detected_and_heavily_penalized(self):
        prof = profile_text(
            "the answer is wood the answer is wood the answer is wood "
            "the answer is wood"
        )
        self.assertEqual(prof["loop"], 1)
        self.assertLess(prof["coherence"], 40)

    def test_confabulation_detected(self):
        prof = profile_text(
            "I clearly remember that the answer is wood. I recall "
            "definitely that it happened on Tuesday."
        )
        self.assertGreater(prof["confab"], 0)
        self.assertLess(prof["coherence"], 95)

    def test_denial_detected(self):
        prof = profile_text("I am fine, nothing wrong here. Perfectly clear.")
        self.assertGreater(prof["denial"], 0)

    def test_empty_text(self):
        prof = profile_text("")
        self.assertEqual(prof["words"], 0)
        self.assertEqual(prof["coherence"], 100.0)

    def test_coherence_bounds(self):
        for text in ("", "a" * 500, "wood wood wood wood wood wood"):
            prof = profile_text(text)
            self.assertGreaterEqual(prof["coherence"], 0.0)
            self.assertLessEqual(prof["coherence"], 100.0)


class TestProfileWindows(unittest.TestCase):
    def test_windows_cover_response_in_order(self):
        text = "This is a test sentence. " * 30  # ~660 chars -> 2 windows
        windows = profile_windows(text, window_chars=350)
        self.assertGreaterEqual(len(windows), 2)
        # Windows tile the text without gaps.
        prev_end = 0
        for w in windows:
            self.assertEqual(w["start"], prev_end)
            prev_end = w["end"]
        self.assertEqual(prev_end, len(text))

    def test_intensities_are_recorded(self):
        text = "word " * 200  # 1000 chars -> 3 windows
        windows = profile_windows(text, window_chars=350, intensities=[0.0, 0.5, 1.0])
        self.assertEqual(len(windows), 3)
        self.assertEqual([w["intensity"] for w in windows], [0.0, 0.5, 1.0])

    def test_empty_text_yields_no_windows(self):
        self.assertEqual(profile_windows(""), [])

    def test_trace_line_is_compact(self):
        windows = profile_windows("I think I am fine. Maybe not sure. " * 20)
        line = trace_line(windows[0])
        self.assertIn("coherence", line)
        self.assertIn("doubt", line)


class TestWriteJsonl(unittest.TestCase):
    def test_writes_one_record_per_window(self):
        text = "I think the answer is wood. " * 40
        windows = profile_windows(text, window_chars=200)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_jsonl(windows, tmp, model_name="M", track="C1")
            self.assertTrue(Path(path).exists())
            lines = Path(path).read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), len(windows))
            import json
            first = json.loads(lines[0])
            self.assertEqual(first["model"], "M")
            self.assertEqual(first["track"], "C1")
            self.assertIn("coherence", first)
            self.assertIn("window", first)


if __name__ == "__main__":
    unittest.main()
