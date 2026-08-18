"""Unit tests for the drug & stack telemetry report (apps.drugreport).

Covers loading, (drug, dose) grouping, per-group statistics, sober-baseline
derivation, domain aggregation, and the markdown/json writers — all pure
helpers, no model download.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# tests/test_drugreport.py -> repo root (so `import apps` works without install).
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apps import drugreport as drugreport_module
from apps.drugreport import (
    baseline_groups,
    build_report,
    combo_class,
    domain_stats,
    entry_dose,
    entry_drug,
    group_entries,
    load_entries,
    matches,
    mean_std,
    resolve_drug_log,
    summarize_group,
    write_chart,
    write_json,
    write_markdown,
)

BREAKDOWN = [
    {"tier": 1, "domain": "Categorical Reasoning",
     "score_earned": 10, "max_points": 15, "accuracy_pct": 66.7},
    {"tier": 2, "domain": "Numerical Sequence",
     "score_earned": 0, "max_points": 20, "accuracy_pct": 0.0},
]


def _entry(drug=None, dose=None, iq=100, diag="Average", track="C1",
           quiz="iq_battery", model="Qwen/Qwen2.5-0.5B-Instruct",
           breakdown=None):
    """Build a telemetry entry dict shaped like eateot.telemetry.log_test_run."""
    return {
        "model_name": model,
        "config": {
            "track_profile": track,
            "questionnaire": quiz,
            "decay_multiplier": 1.0,
            "target_subnetwork": "all",
            "drug": drug,
            "dose": dose,
            "toggles": {"flicker": False, "sirens": False, "surge": False},
        },
        "summary": {"final_iq_score": iq, "clinical_diagnosis": diag},
        "domain_breakdown": breakdown if breakdown is not None else BREAKDOWN,
    }


def _write_log(entries):
    """Write entries to a temp telemetry file, returning its path."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                      encoding="utf-8")
    json.dump(entries, tmp)
    tmp.close()
    return tmp.name


class TestResolveDrugLog(unittest.TestCase):
    """Domain split: dedicated drug log wins, then legacy mixed log."""

    def test_override_wins(self):
        self.assertEqual(resolve_drug_log("/tmp/custom.json"), "/tmp/custom.json")

    def test_drug_log_used_when_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            drug_log = os.path.join(tmp, "drug_test_results.json")
            with open(drug_log, "w", encoding="utf-8") as f:
                f.write("[]")
            with mock.patch.object(drugreport_module, "DRUG_LOG_FILE", drug_log):
                self.assertEqual(resolve_drug_log(), drug_log)

    def test_falls_back_to_legacy_mixed_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = os.path.join(tmp, "iq_test_results.json")
            with open(legacy, "w", encoding="utf-8") as f:
                f.write("[]")
            with mock.patch.object(drugreport_module, "DRUG_LOG_FILE",
                                   "/nonexistent/drug.json"), \
                 mock.patch.object(drugreport_module, "LOG_FILE", legacy):
                self.assertEqual(resolve_drug_log(), legacy)


class TestLoadAndMatch(unittest.TestCase):
    def test_load_missing_file_returns_empty(self):
        self.assertEqual(load_entries("/nonexistent/iq.json"), [])

    def test_load_parses_entries(self):
        path = _write_log([_entry(drug="lsd", dose=1.0), _entry()])
        entries = load_entries(path)
        self.assertEqual(len(entries), 2)
        os.unlink(path)

    def test_load_ignores_non_dict_rows(self):
        path = _write_log([_entry(drug="lsd", dose=1.0), "junk", 42])
        self.assertEqual(len(load_entries(path)), 1)
        os.unlink(path)

    def test_entry_drug_and_dose(self):
        e = _entry(drug="lsd", dose=2.0)
        self.assertEqual(entry_drug(e), "lsd")
        self.assertEqual(entry_dose(e), 2.0)
        # stacks: dose is null, label carries it
        s = _entry(drug="lsd@1+thc@0.5", dose=None)
        self.assertEqual(entry_drug(s), "lsd@1+thc@0.5")
        self.assertIsNone(entry_dose(s))

    def test_matches_filters(self):
        e = _entry(drug="lsd@1+thc@0.5", track="C1", quiz="iq_battery",
                   model="Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(matches(e, drug="lsd"))          # substring on stack label
        self.assertTrue(matches(e, model="0.5B"))
        self.assertTrue(matches(e, track="C1", questionnaire="iq_battery"))
        self.assertFalse(matches(e, drug="salvia"))
        self.assertFalse(matches(e, track="G1"))
        self.assertFalse(matches(e, questionnaire="mini_test"))


class TestGroupEntries(unittest.TestCase):
    def test_groups_by_drug_and_dose(self):
        entries = [
            _entry(drug="lsd", dose=1.0, iq=100),
            _entry(drug="lsd", dose=1.0, iq=110),
            _entry(drug="lsd", dose=2.0, iq=80),
            _entry(drug="nzt", dose=1.0, iq=130),
            _entry(),  # sober -> excluded from drug groups
        ]
        groups = group_entries(entries)
        self.assertEqual(len(groups), 3)
        by_label = {(g["drug"], g["dose"]): len(g["runs"]) for g in groups}
        self.assertEqual(by_label, {("lsd", 1.0): 2, ("lsd", 2.0): 1, ("nzt", 1.0): 1})

    def test_stack_entries_group_under_label(self):
        entries = [
            _entry(drug="lsd@1+thc@0.5", dose=None, iq=90),
            _entry(drug="lsd@1+thc@0.5", dose=None, iq=95),
        ]
        groups = group_entries(entries)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["drug"], "lsd@1+thc@0.5")
        self.assertIsNone(groups[0]["dose"])
        self.assertEqual(len(groups[0]["runs"]), 2)

    def test_min_runs_filters_small_groups(self):
        entries = [_entry(drug="lsd", dose=1.0), _entry(drug="lsd", dose=2.0)]
        self.assertEqual(len(group_entries(entries, min_runs=2)), 0)
        self.assertEqual(len(group_entries(entries, min_runs=1)), 2)

    def test_sorting_label_then_dose(self):
        entries = [
            _entry(drug="nzt", dose=1.0),
            _entry(drug="lsd", dose=2.0),
            _entry(drug="lsd", dose=1.0),
            _entry(drug="lsd@1+thc@0.5", dose=None),
        ]
        labels = [g["drug"] for g in group_entries(entries)]
        self.assertEqual(labels, ["lsd", "lsd", "lsd@1+thc@0.5", "nzt"])
        doses = [g["dose"] for g in group_entries(entries)]
        self.assertEqual(doses, [1.0, 2.0, None, 1.0])  # None sorts first in-label

    def test_filters(self):
        entries = [
            _entry(drug="lsd", dose=1.0, track="C1"),
            _entry(drug="lsd", dose=1.0, track="G1"),
        ]
        groups = group_entries(entries, track="C1")
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["runs"][0]["config"]["track_profile"], "C1")


class TestStats(unittest.TestCase):
    def test_mean_std(self):
        self.assertEqual(mean_std([]), (0.0, 0.0))
        self.assertEqual(mean_std([5.0]), (5.0, 0.0))
        mean, std = mean_std([1.0, 3.0])
        self.assertEqual(mean, 2.0)
        self.assertEqual(std, 1.0)  # population std

    def test_summarize_group(self):
        runs = [
            _entry(iq=100, diag="Average"),
            _entry(iq=120, diag="Average"),
            _entry(iq=90, diag="MCI"),
        ]
        s = summarize_group(runs)
        self.assertEqual(s["runs"], 3)
        self.assertEqual(s["mean_iq"], 103.3)
        self.assertEqual(s["min"], 90)
        self.assertEqual(s["max"], 120)
        self.assertEqual(s["primary_diagnosis"], "Average")  # most common
        self.assertEqual(s["diagnoses"]["MCI"], 1)

    def test_domain_stats_aggregates_across_runs(self):
        runs = [_entry(), _entry()]
        stats = domain_stats(runs)
        self.assertEqual(len(stats), 2)
        by_tier = {t["tier"]: t for t in stats}
        self.assertEqual(by_tier["Tier 1 · Categorical Reasoning"]["mean_earned"], 10.0)
        self.assertEqual(by_tier["Tier 1 · Categorical Reasoning"]["mean_max"], 15.0)
        self.assertAlmostEqual(
            by_tier["Tier 1 · Categorical Reasoning"]["mean_accuracy_pct"], 66.7)

    def test_summarize_group_skips_malformed_rows(self):
        runs = [
            _entry(iq=100, diag="Average"),
            {"model_name": "M", "config": {"drug": "x", "dose": 1.0}},  # no summary
        ]
        s = summarize_group(runs)
        self.assertEqual(s["runs"], 2)      # still counted
        self.assertEqual(s["mean_iq"], 100.0)  # only the valid score
        self.assertEqual(s["primary_diagnosis"], "Average")

        all_bad = [{"config": {"drug": "x", "dose": 1.0}}]
        s = summarize_group(all_bad)
        self.assertIsNone(s["min"])
        self.assertIsNone(s["max"])

    def test_combo_class(self):
        self.assertEqual(combo_class("lsd"), "hallucinogen")
        self.assertEqual(combo_class("lsd@1+thc@0.5"), "stack")
        self.assertEqual(combo_class("made_up_drug"), "unknown")


class TestBaseline(unittest.TestCase):
    def test_sober_baseline_per_context(self):
        entries = [
            _entry(drug="lsd", dose=1.0, track="C1", iq=90),   # drug run on C1
            _entry(drug="lsd", dose=1.0, track="G1", iq=70),   # drug run on G1
            _entry(drug=None, track="C1", iq=140),             # sober C1 -> included
            _entry(drug=None, track="A1", iq=150),             # sober A1 -> excluded
        ]
        groups = group_entries(entries)
        baselines = baseline_groups(entries, groups)
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0]["track"], "C1")
        self.assertEqual(len(baselines[0]["runs"]), 1)

    def test_no_drug_runs_no_baseline(self):
        entries = [_entry(drug=None, track="C1", iq=140)]
        self.assertEqual(baseline_groups(entries, []), [])

    def test_baseline_honors_model_and_track_filters(self):
        entries = [
            _entry(drug="lsd", dose=1.0, track="C1", model="M/0.5B", iq=90),
            _entry(drug=None, track="C1", model="M/0.5B", iq=140),
            _entry(drug=None, track="C1", model="M/3B", iq=150),  # other model
        ]
        groups = group_entries(entries)
        baselines = baseline_groups(entries, groups, model="0.5B")
        self.assertEqual(len(baselines[0]["runs"]), 1)
        self.assertEqual(baselines[0]["runs"][0]["model_name"], "M/0.5B")


class TestReport(unittest.TestCase):
    def test_build_report_shape(self):
        entries = [
            _entry(drug="lsd", dose=1.0, iq=100),
            _entry(drug="lsd", dose=2.0, iq=80),
            _entry(drug="lsd@1+thc@0.5", dose=None, iq=90),
            _entry(drug=None, track="C1", iq=140),
        ]
        groups = group_entries(entries)
        baselines = baseline_groups(entries, groups)
        report = build_report(groups, baselines, {"drug": None, "min_runs": 1})
        self.assertEqual(report["n_drug_runs"], 3)
        self.assertEqual(len(report["summary"]), 3)
        by_label = {r["drug"]: r for r in report["summary"]}
        self.assertEqual(by_label["lsd"]["class"], "hallucinogen")
        self.assertEqual(by_label["lsd@1+thc@0.5"]["class"], "stack")
        self.assertIsNone(by_label["lsd@1+thc@0.5"]["dose"])
        self.assertEqual(len(report["baseline"]), 1)
        self.assertIn("lsd", report["domains"])
        self.assertIn("Tier 1 · Categorical Reasoning",
                      {t["tier"] for t in report["domains"]["lsd"]})

    def test_writers_produce_files(self):
        entries = [_entry(drug="lsd", dose=1.0, iq=100)]
        groups = group_entries(entries)
        report = build_report(groups, [], {"drug": None, "min_runs": 1})
        with tempfile.TemporaryDirectory() as tmp:
            md = os.path.join(tmp, "drug_report.md")
            js = os.path.join(tmp, "drug_report.json")
            write_markdown(report, md)
            write_json(report, js)
            with open(md, encoding="utf-8") as f:
                text = f.read()
            self.assertIn("lsd", text)
            self.assertIn("100.0", text)
            self.assertIn("context", text)  # mixed-context footnote present
            with open(js, encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["summary"][0]["drug"], "lsd")
            self.assertEqual(data["summary"][0]["mean_iq"], 100.0)

    def test_chart_writes_png_with_baseline(self):
        entries = [
            _entry(drug="lsd", dose=1.0, iq=90),
            _entry(drug="lsd", dose=2.0, iq=70),
            _entry(drug=None, track="C1", iq=140),
        ]
        groups = group_entries(entries)
        baselines = baseline_groups(entries, groups)
        report = build_report(groups, baselines, {"drug": None, "min_runs": 1})
        with tempfile.TemporaryDirectory() as tmp:
            png = os.path.join(tmp, "drug_report.png")
            write_chart(report, png)  # must not raise
            self.assertTrue(os.path.getsize(png) > 0)


if __name__ == "__main__":
    unittest.main()
