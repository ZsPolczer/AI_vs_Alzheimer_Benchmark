#!/usr/bin/env python3
"""
EATEOT drug & stack telemetry report.

Reads the accumulated IQ telemetry log (``outputs/iq_test_results.json``)
and produces a report grouping every drug run by ``(drug/stack label, dose)``
— the post-hoc view over everything the lab, trip study, and lab deployer
have logged:

  eateot-drugreport
  eateot-drugreport --drug lsd
  eateot-drugreport --model Qwen/Qwen2.5-0.5B-Instruct --min-runs 2
  eateot-drugreport --track C1 --questionnaire iq_battery

Combo/stack runs are recorded in telemetry with the merged label as the
``drug`` (e.g. ``lsd@1+thc@0.5``) and ``dose`` = null, so stacks appear as
their own rows. A "sober (baseline)" row is added per (track, questionnaire)
pair from matching non-drug runs when any exist, so a drug's IQ can be read
against its clean baseline.

Domain split: drug runs auto-log to ``drug_test_results.json`` (DRUG_LOG_FILE)
while Alzheimer runs log to ``iq_test_results.json`` (LOG_FILE) — this report
reads the DRUG log for combos and the ALZHEIMER log for sober baselines. For
telemetry written before the split (drug entries inside the Alzheimer log),
it falls back to the legacy file automatically; ``--drug-log`` /
``--baseline-log`` override either source explicitly.

Writes drug_report.md / .json and drug_report.png to the data directory
(default outputs/).
"""

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eateot import DRUG_PROFILES
from eateot.config import CLINICAL_BANDS
from eateot.paths import (
    DRUG_LOG_FILE,
    DRUG_REPORT_JSON,
    DRUG_REPORT_MD,
    DRUG_REPORT_PNG,
    LOG_FILE,
    ensure_data_dir,
)

# Fill colors per drug class (resolved from the catalog at report time);
# stacks and unknown labels get their own colors.
CLASS_COLORS = {
    "hallucinogen": "#7b2cbf",
    "dissociative": "#219ebc",
    "stimulant": "#f4a261",
    "depressant": "#e76f51",
    "cannabinoid": "#2a9d8f",
    "deliriant": "#d00000",
    "enhancer": "#43aa8b",
    "placebo": "#adb5bd",
    "stack": "#ffb703",
    "unknown": "#6c757d",
}
BASELINE_COLOR = "#8d99ae"


# ---------------------------------------------------------------------------
# Loading & matching
# ---------------------------------------------------------------------------
def resolve_drug_log(drug_log_override: str | None = None) -> str:
    """The drug telemetry log to read, with a legacy fallback.

    Returns ``drug_log_override`` when given, else ``DRUG_LOG_FILE`` if it
    exists, else ``LOG_FILE`` — the pre-split file that may still hold drug
    entries. This keeps the report working on telemetry written before the
    Alzheimer/drug domain split.
    """
    if drug_log_override:
        return drug_log_override
    if os.path.exists(DRUG_LOG_FILE):
        return DRUG_LOG_FILE
    return LOG_FILE


def load_entries(log_path: str | None = None) -> list[dict]:
    """Read a telemetry log into a list of entry dicts ([] if missing/bad).

    Defaults to the drug-domain log (with the legacy fallback) via
    ``resolve_drug_log``.
    """
    if log_path is None:
        log_path = resolve_drug_log()
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, encoding="utf-8") as f:
            logs = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []
    return [e for e in logs if isinstance(e, dict)]


def entry_drug(entry: dict) -> str | None:
    """The run's drug/stack label (None for sober runs)."""
    return (entry.get("config") or {}).get("drug")


def entry_dose(entry: dict) -> float | None:
    """The run's dose (None for stacks, where doses live in the label)."""
    return (entry.get("config") or {}).get("dose")


def matches(entry: dict, drug: str | None = None, model: str | None = None,
            track: str | None = None, questionnaire: str | None = None) -> bool:
    """Case-insensitive substring filters on the entry's identity fields."""
    if drug and drug.lower() not in str(entry_drug(entry) or "").lower():
        return False
    if model and model.lower() not in str(entry.get("model_name") or "").lower():
        return False
    cfg = entry.get("config") or {}
    if track and cfg.get("track_profile") != track:
        return False
    if questionnaire and cfg.get("questionnaire", "iq_battery") != questionnaire:
        return False
    return True


# ---------------------------------------------------------------------------
# Grouping & statistics
# ---------------------------------------------------------------------------
def group_entries(entries: list[dict], drug: str | None = None, model: str | None = None,
                  track: str | None = None, questionnaire: str | None = None,
                  min_runs: int = 1) -> list[dict]:
    """Group matching drug runs by (drug/stack label, dose).

    Returns a list of ``{"drug", "dose", "runs": [entries]}`` sorted by
    label, then dose (stacks, whose dose is None, sort first within a label).
    """
    groups: dict[tuple, list[dict]] = {}
    for entry in entries:
        if not matches(entry, drug, model, track, questionnaire):
            continue
        label = entry_drug(entry)
        if not label:
            continue  # sober runs go to the baseline, not here
        key = (label, entry_dose(entry))
        groups.setdefault(key, []).append(entry)

    result = []
    for (label, dose), runs in groups.items():
        if len(runs) < max(1, min_runs):
            continue
        result.append({"drug": label, "dose": dose, "runs": runs})
    result.sort(key=lambda g: (g["drug"].lower(), 0 if g["dose"] is None else g["dose"]))
    return result


def baseline_groups(entries: list[dict], groups: list[dict], model: str | None = None,
                    track: str | None = None,
                    questionnaire: str | None = None) -> list[dict]:
    """Sober (drug=None) runs matching the same contexts as the drug groups.

    A context is a (track, questionnaire) pair — a drug row's IQ is only
    comparable to a sober run on the same track and quiz, so baselines are
    computed per context. ``model``/``track``/``questionnaire`` filters are
    honored (the drug filter is intentionally NOT — baseline arms have no
    drug).
    """
    contexts = {((g["runs"][0].get("config") or {}).get("track_profile"),
                 (g["runs"][0].get("config") or {}).get("questionnaire", "iq_battery"))
                for g in groups}
    if not contexts:
        return []

    by_context: dict[tuple, list[dict]] = {}
    for entry in entries:
        if entry_drug(entry):
            continue  # not sober
        if not matches(entry, model=model, track=track, questionnaire=questionnaire):
            continue
        cfg = entry.get("config") or {}
        context = (cfg.get("track_profile"), cfg.get("questionnaire", "iq_battery"))
        if context in contexts:
            by_context.setdefault(context, []).append(entry)

    result = []
    for (trk, quiz), runs in sorted(by_context.items()):
        result.append({"track": trk, "questionnaire": quiz, "runs": runs})
    return result


def mean_std(values: list[float]):
    """(mean, std) — std is 0.0 for a single sample."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def summarize_group(runs: list[dict]) -> dict:
    """IQ stats + diagnosis histogram for one (drug, dose) group.

    Malformed rows (no ``summary`` / non-numeric score) are skipped for the
    IQ stats but still counted in ``runs``; a group with no valid scores
    reports ``min``/``max`` as None.
    """
    iqs = []
    diagnoses = Counter()
    for r in runs:
        score = ((r.get("summary") or {}).get("final_iq_score"))
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            iqs.append(float(score))
        diag = ((r.get("summary") or {}).get("clinical_diagnosis"))
        if diag:
            diagnoses[diag] += 1
    mean, std = mean_std(iqs)
    return {
        "runs": len(runs),
        "mean_iq": round(mean, 1),
        "std": round(std, 1),
        "min": min(iqs) if iqs else None,
        "max": max(iqs) if iqs else None,
        "primary_diagnosis": diagnoses.most_common(1)[0][0] if diagnoses else None,
        "diagnoses": dict(diagnoses),
    }


def domain_stats(runs: list[dict]) -> list[dict]:
    """Per-tier aggregation across a drug's runs (mean earned/max, accuracy)."""
    agg: dict[str, dict] = {}
    for r in runs:
        for item in r.get("domain_breakdown", []):
            tier = item.get("tier")
            domain = item.get("domain", "")
            key = f"Tier {tier} · {domain}"
            entry = agg.setdefault(key, {"earned": 0.0, "max": 0.0, "accuracy": 0.0, "n": 0})
            entry["earned"] += float(item.get("score_earned", 0))
            entry["max"] += float(item.get("max_points", 0))
            acc = item.get("accuracy_pct")
            if isinstance(acc, (int, float)):
                entry["accuracy"] += float(acc)
            entry["n"] += 1
    return [
        {
            "tier": key,
            "mean_earned": round(a["earned"] / a["n"], 1),
            "mean_max": round(a["max"] / a["n"], 1),
            "mean_accuracy_pct": round(a["accuracy"] / a["n"], 1),
        }
        for key, a in sorted(agg.items())
    ]


def combo_class(label: str) -> str:
    """Resolve a telemetry label to a drug class ('stack' for combos)."""
    if label in DRUG_PROFILES:
        return DRUG_PROFILES[label].get("class", "unknown")
    if "+" in label:
        return "stack"
    return "unknown"


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def build_report(groups: list[dict], baselines: list[dict], filters: dict,
                 source: str | None = None) -> dict:
    """Assemble the full report dict (serializable to JSON)."""
    summary = [
        {
            "drug": g["drug"],
            "dose": g["dose"],
            "class": combo_class(g["drug"]),
            **summarize_group(g["runs"]),
        }
        for g in groups
    ]
    baseline_rows = [
        {
            "track": b["track"],
            "questionnaire": b["questionnaire"],
            "class": "sober",
            **summarize_group(b["runs"]),
        }
        for b in baselines
    ]
    return {
        "report": "drug & stack telemetry",
        "generated_at": datetime.now().isoformat(),
        "source": source or LOG_FILE,
        "filters": filters,
        "summary": summary,
        "baseline": baseline_rows,
        "domains": {g["drug"]: domain_stats(g["runs"]) for g in groups},
        "n_drug_runs": sum(len(g["runs"]) for g in groups),
    }


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def write_markdown(report: dict, path: str):
    lines = [
        "# 💊 Drug & Stack Telemetry Report",
        "",
        f"Source `{report['source']}` · generated {report['generated_at'][:16]} · "
        f"`{report['n_drug_runs']}` drug runs",
        "",
        "## Filters",
        "",
        "| drug | model | track | questionnaire | min_runs |",
        "|---|---|---|---|---|",
        f"| {report['filters'].get('drug') or '—'} | "
        f"{report['filters'].get('model') or '—'} | "
        f"{report['filters'].get('track') or '—'} | "
        f"{report['filters'].get('questionnaire') or '—'} | "
        f"{report['filters'].get('min_runs', 1)} |",
        "",
        "## IQ by (drug / stack · dose)",
        "",
        "| Drug / stack | Class | Dose | Runs | Mean IQ | ±Std | Min–Max | Primary diagnosis |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["summary"]:
        dose = "—" if row["dose"] is None else f"{row['dose']:g}"
        span = (f"{row['min']:g}–{row['max']:g}" if row["min"] is not None
                else "—")
        lines.append(
            f"| {row['drug']} | {row['class']} | {dose} | {row['runs']} | "
            f"**{row['mean_iq']:.1f}** | {row['std']:.1f} | "
            f"{span} | {row['primary_diagnosis']} |"
        )
    lines.append("")
    lines.append("*Stacks are recorded with dose — (per-component doses live in the "
                 "label). Rows aggregate all matching telemetry contexts (track × "
                 "questionnaire); use `--track`/`--questionnaire` to isolate a single "
                 "context. Baseline rows below come from sober (drug-free) runs.*")

    if report["baseline"]:
        lines.append("")
        lines.append("## Sober baseline (per track · questionnaire)")
        lines.append("")
        lines.append("| Track | Questionnaire | Runs | Mean IQ | ±Std | Primary diagnosis |")
        lines.append("|---|---|---|---|---|---|")
        for row in report["baseline"]:
            lines.append(
                f"| {row['track']} | {row['questionnaire']} | {row['runs']} | "
                f"**{row['mean_iq']:.1f}** | {row['std']:.1f} | "
                f"{row['primary_diagnosis']} |"
            )
        lines.append("")

    if report["domains"]:
        lines.append("## Domain effects per drug (mean points earned / max · all doses)")
        lines.append("")
        lines.append("| Drug | Tier · Domain | Earned / Max | Accuracy |")
        lines.append("|---|---|---|---|")
        for drug, tiers in report["domains"].items():
            for i, t in enumerate(tiers):
                drug_cell = drug if i == 0 else ""
                lines.append(
                    f"| {drug_cell} | {t['tier']} | {t['mean_earned']:.1f} / "
                    f"{t['mean_max']:.0f} | {t['mean_accuracy_pct']:.0f}% |"
                )
        lines.append("")

    lines.append(
        f"*Telemetry grows with every battery run; re-run this report any time "
        f"(eateot-drugreport). Sampling temp 0.7 — run-to-run variance is expected.*"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(report: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def write_chart(report: dict, path: str):
    """Horizontal bars of mean IQ per (drug, dose) combo, colored by class.

    Most damaging combos (lowest IQ) are rendered at the TOP; error bars show
    ±1 std across runs. Sober baselines are hatched gray bars at the bottom.
    """
    combos = sorted(report["summary"], key=lambda r: r["mean_iq"], reverse=True)
    rows = [
        {
            "label": r["drug"] + ("" if r["dose"] is None else " @ " + format(r["dose"], "g")),
            "mean": r["mean_iq"],
            "std": r["std"],
            "class": r["class"],
            "diag": r["primary_diagnosis"],
        }
        for r in combos
    ] + [
        {
            "label": "sober · " + b["track"] + "/" + b["questionnaire"],
            "mean": b["mean_iq"],
            "std": b["std"],
            "class": "sober",
            "diag": b["primary_diagnosis"],
        }
        for b in report["baseline"]
    ]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, 0.42 * len(rows) + 1.5)))
    fig.suptitle("Drug & Stack Telemetry — Mean IQ by (drug · dose)",
                 fontsize=13, fontweight="bold")

    # Clinical band shading on the IQ axis (bands sorted descending).
    bands = sorted(CLINICAL_BANDS, key=lambda b: b[0], reverse=True)
    for i, (thr, _label) in enumerate(bands):
        top = 155 if i == 0 else bands[i - 1][0]
        ax.axvspan(top, thr, color=f"C{i}", alpha=0.08)
    for thr, _label in bands:
        ax.axvline(thr, color="#888", ls="--", lw=0.8, alpha=0.5)
    ax.axvline(100, color="#444", ls=":", lw=1.1)

    labels = [r["label"] for r in rows]
    colors = [
        BASELINE_COLOR if r["class"] == "sober"
        else CLASS_COLORS.get(r["class"], CLASS_COLORS["unknown"])
        for r in rows
    ]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    y = list(range(len(rows)))

    bars = ax.barh(y, means, xerr=stds, color=colors, height=0.62,
                   edgecolor="white", capsize=3, alpha=0.92)
    for bar, row in zip(bars, rows):
        if row["class"] == "sober":
            bar.set_hatch("//")
            bar.set_edgecolor("#555")
        ax.text(row["mean"] + 2, bar.get_y() + bar.get_height() / 2,
                format(row["mean"], ".0f"), va="center", fontsize=9,
                fontweight="bold")
        diag = row["diag"].split(" (")[0] if row["diag"] else ""
        ax.text(row["mean"] + 2, bar.get_y() + bar.get_height() / 2 - 0.14,
                diag, va="center", fontsize=6.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Estimated IQ (mean ± std)")
    ax.set_xlim(30, 165)
    ax.grid(axis="x", ls=":", alpha=0.6)

    # Class legend (dedupe, keep first-seen order; sober gets its own handle).
    seen = []
    for r in rows:
        if r["class"] not in seen:
            seen.append(r["class"])
    handles = [
        plt.Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=CLASS_COLORS.get(c, "#666"), markersize=9,
                   label=c)
        for c in seen
    ]
    if any(r["class"] == "sober" for r in rows):
        handles.append(plt.Line2D([0], [0], marker="s", color="w",
                                  markerfacecolor=BASELINE_COLOR, markersize=9,
                                  label="sober (baseline)"))
    ax.legend(handles=handles, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="EATEOT drug & stack telemetry report (groups IQ per drug/dose)",
    )
    parser.add_argument("--drug", type=str, default=None,
                        help="Filter to labels containing this text (substring; e.g. "
                             "'lsd' matches lsd, microdose_lsd and any lsd@… stack)")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to runs from this model (substring, e.g. '0.5B')")
    parser.add_argument("--track", type=str, default=None,
                        help="Filter to this track profile (e.g. C1)")
    parser.add_argument("--questionnaire", type=str, default=None,
                        help="Filter to this questionnaire (e.g. iq_battery)")
    parser.add_argument("--min-runs", type=int, default=1, metavar="N",
                        help="Only report (drug, dose) combos with >= N runs (default: 1)")
    parser.add_argument("--no-chart", action="store_true",
                        help="Skip the chart (drug_report.png)")
    parser.add_argument("--drug-log", type=str, default=None, metavar="FILE",
                        help="Drug telemetry log to read (default: drug_test_results.json, "
                             "falling back to the legacy mixed log)")
    parser.add_argument("--baseline-log", type=str, default=None, metavar="FILE",
                        help="Alzheimer telemetry log for sober baselines "
                             "(default: iq_test_results.json)")
    args = parser.parse_args()

    filters = {
        "drug": args.drug,
        "model": args.model,
        "track": args.track,
        "questionnaire": args.questionnaire,
        "min_runs": args.min_runs,
    }

    drug_log = resolve_drug_log(args.drug_log)
    using_legacy = drug_log == LOG_FILE and not args.drug_log
    entries = load_entries(drug_log)
    groups = group_entries(
        entries, drug=args.drug, model=args.model, track=args.track,
        questionnaire=args.questionnaire, min_runs=args.min_runs,
    )
    if not groups:
        print(f"[ERR] no drug/stack telemetry matches the filters in {drug_log}. "
              "Have you run any drug experiments yet? They log to "
              "drug_test_results.json (eateot-lab --drug/--stack, [P], eateot-trip).",
              file=sys.stderr)
        sys.exit(1)
    if using_legacy:
        print(f"[i] no dedicated drug log yet — reading legacy telemetry {LOG_FILE}. "
              "Drug runs now auto-log to drug_test_results.json.",
              file=sys.stderr)

    baseline_entries = load_entries(args.baseline_log or LOG_FILE)
    baselines = baseline_groups(
        baseline_entries, groups, model=args.model, track=args.track,
        questionnaire=args.questionnaire,
    )
    report = build_report(groups, baselines, filters, source=drug_log)
    ensure_data_dir()

    write_markdown(report, DRUG_REPORT_MD)
    write_json(report, DRUG_REPORT_JSON)
    if not args.no_chart:
        write_chart(report, DRUG_REPORT_PNG)

    print(f"[OK] wrote {DRUG_REPORT_MD}, {DRUG_REPORT_JSON}"
          + (f", {DRUG_REPORT_PNG}" if not args.no_chart else "")
          + f" ({len(groups)} combos · {report['n_drug_runs']} runs"
          + (f" · {len(baselines)} baseline groups" if baselines else "") + ")",
          flush=True)


if __name__ == "__main__":
    main()
