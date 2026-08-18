"""Unit tests for telemetry persistence and path resolution."""

import json
import os
import tempfile
import unittest
from unittest import mock

import eateot.paths as paths
from eateot.paths import DRUG_LOG_FILE, LOG_FILE
from eateot.telemetry import log_test_run


class TestPaths(unittest.TestCase):
    def test_data_dir_env_override(self):
        with mock.patch.dict(os.environ, {"EATEOT_DATA_DIR": "/tmp/eateot-test-out"}):
            # paths module constants were computed at import; verify the resolver
            # helper would honor the env var in a fresh module.
            import importlib
            reloaded = importlib.reload(paths)
            self.assertEqual(str(reloaded.DATA_DIR), "/tmp/eateot-test-out")


class TestDomainRouting(unittest.TestCase):
    """Drug runs log to DRUG_LOG_FILE; everything else logs to LOG_FILE."""

    def _run(self, tmp, **kwargs):
        alz = os.path.join(tmp, "iq_test_results.json")
        drg = os.path.join(tmp, "drug_test_results.json")
        with mock.patch("eateot.telemetry.LOG_FILE", alz), \
             mock.patch("eateot.telemetry.DRUG_LOG_FILE", drg):
            log_test_run(
                model_name="M", track_choice="C1", decay_mult=1.0,
                target_subnetwork="all", flicker_mode=False, sirens_mode=False,
                surge_mode=False, final_iq_score=80, clinical_diag="MCI",
                detailed_results=[], **kwargs,
            )
        return alz, drg

    def test_sober_run_logs_to_alzheimer_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            alz, drg = self._run(tmp)
            self.assertTrue(os.path.exists(alz))
            self.assertFalse(os.path.exists(drg))
            with open(alz, encoding="utf-8") as f:
                self.assertIsNone(json.load(f)[0]["config"]["drug"])

    def test_drug_run_logs_to_drug_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            alz, drg = self._run(tmp, drug="lsd", dose=1.0)
            self.assertFalse(os.path.exists(alz))
            self.assertTrue(os.path.exists(drg))
            with open(drg, encoding="utf-8") as f:
                cfg = json.load(f)[0]["config"]
            self.assertEqual(cfg["drug"], "lsd")
            self.assertEqual(cfg["dose"], 1.0)

    def test_stack_run_logs_to_drug_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            alz, drg = self._run(tmp, drug="lsd@1+thc@0.5", dose=None)
            self.assertFalse(os.path.exists(alz))
            with open(drg, encoding="utf-8") as f:
                cfg = json.load(f)[0]["config"]
            self.assertEqual(cfg["drug"], "lsd@1+thc@0.5")
            self.assertIsNone(cfg["dose"])

    def test_appends_within_the_same_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            alz, drg = self._run(tmp, drug="lsd", dose=1.0)
            self._run(tmp, drug="lsd", dose=2.0)
            with open(drg, encoding="utf-8") as f:
                self.assertEqual(len(json.load(f)), 2)  # both drug runs, same log
            self.assertFalse(os.path.exists(alz))       # alzheimer log untouched

    def test_domain_constants_are_distinct(self):
        self.assertNotEqual(DRUG_LOG_FILE, LOG_FILE)
        self.assertIn("drug", os.path.basename(DRUG_LOG_FILE))


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

    def test_records_seed_and_restore_fraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = os.path.join(tmp, "iq_test_results.json")
            with mock.patch("eateot.telemetry.LOG_FILE", log_path):
                log_test_run(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    track_choice="G1",
                    decay_mult=1.0,
                    target_subnetwork="all",
                    flicker_mode=False,
                    sirens_mode=False,
                    surge_mode=False,
                    final_iq_score=88,
                    clinical_diag="MCI",
                    detailed_results=[],
                    seed=42,
                    restore_fraction=0.5,
                )
            with open(log_path, encoding="utf-8") as f:
                cfg = json.load(f)[0]["config"]
            self.assertEqual(cfg["seed"], 42)
            self.assertEqual(cfg["restore_fraction"], 0.5)

    def test_records_drug_and_dose(self):
        # drug runs now route to the DRUG log (domain split)
        with tempfile.TemporaryDirectory() as tmp:
            drug_path = os.path.join(tmp, "drug_test_results.json")
            with mock.patch("eateot.telemetry.LOG_FILE",
                            os.path.join(tmp, "iq_test_results.json")), \
                 mock.patch("eateot.telemetry.DRUG_LOG_FILE", drug_path):
                log_test_run(
                    model_name="Qwen/Qwen2.5-0.5B-Instruct",
                    track_choice="C1",
                    decay_mult=1.0,
                    target_subnetwork="all",
                    flicker_mode=False,
                    sirens_mode=False,
                    surge_mode=False,
                    final_iq_score=88,
                    clinical_diag="MCI",
                    detailed_results=[],
                    drug="lsd",
                    dose=2.0,
                )
            with open(drug_path, encoding="utf-8") as f:
                cfg = json.load(f)[0]["config"]
            self.assertEqual(cfg["drug"], "lsd")
            self.assertEqual(cfg["dose"], 2.0)

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
