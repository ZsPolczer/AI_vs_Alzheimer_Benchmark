#!/usr/bin/env python3
"""
EATEOT model comparison: run the IQ battery (A1 lucid baseline) on all
three Qwen2.5-Instruct models and save a side-by-side comparison.

Port of ``model_comparison.py``, now consuming the ``eateot`` package. Report
files and telemetry resolve through ``eateot.paths`` (default ``outputs/``).

The questionnaire is configurable via ``--questionnaire`` (default
``iq_battery``); see ``config/questionnaires/`` for available sets.

Usage:
  eateot-compare                      # benchmark all three models
  eateot-compare --models 3B          # only re-benchmark the 3B
  eateot-compare --questionnaire iq_battery_mini   # use a different quiz

Results for models that are NOT freshly run are pulled from the latest
telemetry entry matching the same track + questionnaire, so a partial re-run
still produces a full comparison report.
"""

import argparse
import gc
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import torch

from eateot import BrainLabEngine, LOG_FILE, run_iq_test
from eateot.paths import COMPARISON_JSON, COMPARISON_MD, COMPARISON_PNG, ensure_data_dir
from eateot.questionnaires import list_batteries, load_battery

MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]
TRACK = "A1"
DECAY = 1.0
SUBNET = "all"
QUESTIONNAIRE = "iq_battery"

# Short diagnosis -> concise label for chart annotations
DIAG_SHORT = {
    "Superior Cognitive Function (Lucid Baseline)": "Lucid",
    "Average Reasoning (Mild Functional Friction)": "Average",
    "Mild Cognitive Impairment (Early Perseveration / Aphasia)": "MCI",
    "Moderate Stage Disintegration (Severe Logical Deficit)": "Moderate",
    "Advanced Stage Plaque / Structural Collapse": "Collapse",
}


def resolve_models(selections):
    """Map CLI selections (short names like '3B' or full ids) to model ids.

    Empty selection -> all models.
    """
    if not selections:
        return list(MODELS)
    picked = []
    for sel in selections:
        sel = sel.strip()
        matches = [m for m in MODELS if sel in m]
        if not matches:
            print(f"[WARN] unknown model '{sel}' - skipping", file=sys.stderr)
        picked.extend(matches)
    # dedupe, preserving order
    return list(dict.fromkeys(picked))


def latest_entry(model_id, track=TRACK, questionnaire=QUESTIONNAIRE):
    """Most recent telemetry entry for a model under this track + quiz, or None."""
    if not os.path.exists(LOG_FILE):
        return None
    with open(LOG_FILE, encoding="utf-8") as f:
        logs = json.load(f)
    for entry in reversed(logs):
        if entry.get("model_name") != model_id:
            continue
        cfg = entry.get("config", {})
        if cfg.get("track_profile") != track:
            continue
        if cfg.get("questionnaire", "iq_battery") != questionnaire:
            continue
        return entry
    return None


def run_battery(model_ids, battery, battery_name):
    """Freshly benchmark each selected model. Returns {short: telemetry_entry}."""
    results = {}
    for model_id in model_ids:
        short = model_id.split("/")[-1]
        print(f"[RUN] {short} ...", flush=True)
        log_path = f"/tmp/iq_run_{short}.log"
        try:
            lab = BrainLabEngine(model_id)  # engine sets lab.model_id itself
            with open(log_path, "w", encoding="utf-8") as fh, redirect_stdout(fh):
                run_iq_test(lab, TRACK, DECAY, SUBNET, False, False, False,
                            battery=battery, battery_name=battery_name)
            del lab
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            print(f"[FAIL] {short}: exception during battery", flush=True)
            traceback.print_exc(file=sys.stderr)
            continue

        entry = latest_entry(model_id)
        if entry is None:
            print(f"[FAIL] {short}: no fresh telemetry entry for this run", flush=True)
            continue
        results[short] = entry
        print(f"[DONE] {short} -> IQ {entry['summary']['final_iq_score']}", flush=True)
    return results


def write_markdown(results, battery, battery_name):
    shorts = list(results.keys())
    lines = []
    lines.append(f"# \U0001F9E0 Model IQ Comparison — A1 (Lucid Baseline) · {battery_name}")
    lines.append("")
    lines.append(
        f"Questionnaire **{battery_name}** ({len(battery)} questions) · track **A1** "
        "(scale 0.98) · decay 1.0x · subnetwork `all` · no toggles · sampling temp 0.7"
    )
    lines.append("")
    lines.append("## Final Scores")
    lines.append("")
    lines.append("| Model | Estimated IQ | Clinical Diagnosis |")
    lines.append("|---|---|---|")
    for short in shorts:
        s = results[short]["summary"]
        lines.append(f"| {short} | **{s['final_iq_score']}** | {s['clinical_diagnosis']} |")
    lines.append("")
    lines.append("## Per-Tier Breakdown (earned / max)")
    lines.append("")
    lines.append("| Tier | Domain (target IQ) | " + " | ".join(shorts) + " |")
    lines.append("|---|" + "---|" * (1 + len(shorts)))
    for item in battery:
        row = f"| T{item['tier']} | {item['domain']} ({item['target_iq']}) "
        for short in shorts:
            b = next(
                b for b in results[short]["domain_breakdown"] if b["tier"] == item["tier"]
            )
            status = b["status"].split(" (")[0]
            row += f"| {b['score_earned']}/{b['max_points']} · {status} "
        lines.append(row + "|")
    if len(results) < len(MODELS):
        have = set(results)
        missing = [m.split("/")[-1] for m in MODELS if m.split("/")[-1] not in have]
        lines.append("")
        lines.append(
            f"> \u26a0\ufe0f Models missing from this report (no matching A1 "
            f"telemetry available): {', '.join(missing)}."
        )
    lines.append("")
    lines.append(
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — results are "
        "stochastic (temperature 0.7); run-to-run variance is expected. "
        f"Full telemetry: `{LOG_FILE}`.*"
    )
    with open(COMPARISON_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(results, battery_name):
    data = {
        "track_profile": TRACK,
        "questionnaire": battery_name,
        "decay_multiplier": DECAY,
        "target_subnetwork": SUBNET,
        "generated_at": datetime.now().isoformat(),
        "models": {},
    }
    for short, entry in results.items():
        summary = entry["summary"]
        breakdown = [
            {
                "tier": b["tier"],
                "domain": b["domain"],
                "target_iq": b["target_iq"],
                "status": b["status"],
                "accuracy_pct": b["accuracy_pct"],
                "score_earned": b["score_earned"],
                "max_points": b["max_points"],
            }
            for b in entry["domain_breakdown"]
        ]
        data["models"][short] = {
            "final_iq_score": summary["final_iq_score"],
            "clinical_diagnosis": summary["clinical_diagnosis"],
            "domain_breakdown": breakdown,
        }
    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_chart(results, battery, battery_name):
    shorts = list(results.keys())
    iqs = [results[s]["summary"]["final_iq_score"] for s in shorts]
    diags = [DIAG_SHORT.get(results[s]["summary"]["clinical_diagnosis"], "?") for s in shorts]
    colors = ["#8ecae6", "#219ebc", "#023047"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1, 1.5]}
    )
    fig.suptitle(f"EATEOT IQ Battery - A1 Lucid Baseline · {battery_name}", fontsize=14, fontweight="bold")

    # --- Left: final IQ bars ---
    bars = ax1.bar(shorts, iqs, color=colors[: len(shorts)], width=0.6, edgecolor="white", zorder=3)
    ax1.axhline(100, color="#666", ls="--", lw=1, alpha=0.7, zorder=1)
    ax1.text(
        len(shorts) - 0.05, 101, "average (100)",
        fontsize=8, color="#666", ha="right", va="bottom",
    )
    ax1.set_ylim(35, 155)
    ax1.set_title("Estimated IQ Score", fontsize=11)
    ax1.set_ylabel("IQ points")
    for bar, iq, diag in zip(bars, iqs, diags):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.5,
            f"{iq}", ha="center", fontsize=12, fontweight="bold", color="#111",
        )
        ax1.text(
            bar.get_x() + bar.get_width() / 2, 36,
            diag, ha="center", fontsize=8.5, color="#444",
        )
    ax1.set_xticks(range(len(shorts)))
    ax1.set_xticklabels(shorts, rotation=15, ha="right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3, zorder=0)
    ax1.margins(x=0.15)

    # --- Right: grouped per-tier points ---
    tiers = [b["tier"] for b in battery]
    x = np.arange(len(tiers))
    width = 0.25
    for i, short in enumerate(shorts):
        scores = [
            next(b for b in results[short]["domain_breakdown"] if b["tier"] == t)[
                "score_earned"
            ]
            for t in tiers
        ]
        ax2.bar(
            x + (i - 1) * width, scores, width,
            label=short, color=colors[i], edgecolor="white",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"Tier {t}" for t in tiers])
    ax2.set_title("Per-Tier Points Earned", fontsize=11)
    ax2.set_ylabel("points earned")
    ax2.legend(fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(COMPARISON_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run IQ battery comparison across Qwen models")
    parser.add_argument("--models", nargs="*", default=[], help="subset to benchmark fresh (short names or full ids); others come from telemetry")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to benchmark (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    args = parser.parse_args()

    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire

    ensure_data_dir()

    if not os.path.exists(LOG_FILE):
        print(f"[ERR] {LOG_FILE} not found - run the lab once first", file=sys.stderr)
        sys.exit(1)

    run_battery(resolve_models(args.models), battery, battery_name)

    # Assemble report from the latest telemetry for every model we know about,
    # so a partial/failed run still yields a comparison for completed models.
    results = {}
    for model_id in MODELS:
        entry = latest_entry(model_id, questionnaire=battery_name)
        if entry is None:
            print(f"[SKIP] {model_id.split('/')[-1]}: no telemetry yet - not in report", file=sys.stderr)
            continue
        results[model_id.split("/")[-1]] = entry

    if not results:
        print("[ERR] no results available for any model", file=sys.stderr)
        sys.exit(1)

    write_markdown(results, battery, battery_name)
    write_json(results, battery_name)
    write_chart(results, battery, battery_name)
    print(
        f"[OK] wrote {COMPARISON_MD}, {COMPARISON_JSON}, {COMPARISON_PNG} "
        f"({len(results)} models: {', '.join(results)}) · quiz {battery_name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
