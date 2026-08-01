"""Unit tests for telemetry persistence and path resolution."""

import json
import os
import tempfile
import unittest
from unittest import mock

import eateot.paths as paths
from eateot.telemetry import log_test_run


class TestPaths(unittest.TestCase):
    def test_data_dir_env_override(self):
        with mock.patch.dict(os.environ, {"EATEOT_DATA_DIR": "/tmp/eateot-test-out"}):
            # paths module constants were computed at import; verify the resolver
            # helper would honor the env var in a fresh module.
            import importlib
            reloaded = importlib.reload(paths)
            self.assertEqual(str(reloaded.DATA_DIR), "/tmp/eateot-test-out")


class TestLogTestRun(unittest.TestCase):
    def test_appends_structured_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "iq_test_results.json")
            with mock.patch("eateot.telemetry.LOG_FILE", log_path):
                log_test_run(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    track_choice="A1",
                    decay_mult=1.0,
                    target_subnetwork="all",
                    flicker_mode=False,
                    sirens_mode=False,
                    surge_mode=False,
                    final_iq_score=119,
                    clinical_diag="Average",
                    detailed_results=[{"tier": 1, "score_earned": 15}],
                )
                log_test_run(
                    model_name="Qwen/Qwen2.5-3B-Instruct",
                    track_choice="A1",
                    decay_mult=1.0,
                    target_subnetwork="all",
                    flicker_mode=False,
                    sirens_mode=False,
                    surge_mode=False,
                    final_iq_score=140,
                    clinical_diag="Superior",
                    detailed_results=[{"tier": 1, "score_earned": 15}],
                )

            with open(log_path, encoding="utf-8") as f:
                logs = json.load(f)
            self.assertEqual(len(logs), 2)
            self.assertEqual(logs[1]["summary"]["final_iq_score"], 140)
            self.assertEqual(logs[1]["config"]["track_profile"], "A1")

    def test_creates_missing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "nested", "dir")
            log_path = os.path.join(nested, "iq_test_results.json")
            with mock.patch("eateot.telemetry.LOG_FILE", log_path):
                log_test_run(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    track_choice="A1",
                    decay_mult=1.0,
                    target_subnetwork="all",
                    flicker_mode=False,
                    sirens_mode=False,
                    surge_mode=False,
                    final_iq_score=100,
                    clinical_diag="Avg",
                    detailed_results=[],
                )
            self.assertTrue(os.path.exists(log_path))


if __name__ == "__main__":
    unittest.main()
