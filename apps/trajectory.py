#!/usr/bin/env python3
"""
EATEOT full decline trajectory.

Runs the IQ battery sequentially through every degradation track profile
(A1 → Q1, the complete 'long decline') in a single model session, then
renders the full cognitive decay curve.

  eateot-trajectory
  eateot-trajectory --model Qwen/Qwen2.5-0.5B-Instruct --questionnaire clinical_battery

Writes trajectory_report.md and trajectory_decay.png to the data directory
(default outputs/). Each track is also logged to telemetry as usual.
"""

import argparse
import gc
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eateot import BrainLabEngine, EATEOT_TRACK_PROFILES, run_iq_test
from eateot.paths import TRAJECTORY_MD, TRAJECTORY_PNG, ensure_data_dir
from eateot.questionnaires import list_batteries, load_battery

# The full decline, from lucid drift to terminal silence.
TRACK_ORDER = [
    "A1", "A2", "A3",          # Stage 1: lucidity & subtle drift
    "C1", "C2", "C5",          # Stage 2: friction & confabulation
    "E1", "E3", "F2",          # Stage 3: loops, aphasia, contamination
    "G1", "H1", "H1_SIRENS",   # Stage 4: post-awareness
    "K1", "L1", "M1",          # Stage 5: advanced plaque
    "O1", "Q1",                # Stage 6: the void
]


def run_trajectory(lab, battery, battery_name, seed):
    """Run the battery at every track in order. Returns [(track, summary), ...]."""
    results = []
    for track in TRACK_ORDER:
        if track not in EATEOT_TRACK_PROFILES:
            print(f"[SKIP] {track}: not in EATEOT_TRACK_PROFILES", flush=True)
            continue
        print(f"[RUN] {track} ...", flush=True)
        log_path = f"/tmp/iq_trajectory_{track}.log"
        try:
            with open(log_path, "w", encoding="utf-8") as fh, redirect_stdout(fh):
                summary = run_iq_test(
                    lab, track, 1.0, "all", False, False, False,
                    battery=battery, battery_name=battery_name,
                    seed=seed,
                )
            results.append((track, summary))
            print(f"  [DONE] {track} -> IQ {summary['final_iq_score']} "
                  f"({summary['clinical_diagnosis']})", flush=True)
        except Exception:
            print(f"  [FAIL] {track}", flush=True)
            traceback.print_exc(file=sys.stderr)
    return results


def write_markdown(results, model_id, battery_name, seed):
    lines = [
        f"# 🧠 Full Decline Trajectory — A1 → Q1",
        "",
        f"Model **{model_id}** · questionnaire **{battery_name}** · seed `{seed}` · "
        f"one session, 17 stages",
        "",
        "## Estimated IQ by Stage",
        "",
        "| Stage | Track | Estimated IQ | Clinical Diagnosis |",
        "|---|---|---|---|",
    ]
    for track, summary in results:
        lines.append(
            f"| {TRACK_ORDER.index(track) + 1} | {track} | "
            f"**{summary['final_iq_score']}** | {summary['clinical_diagnosis']} |"
        )
    lines.append("")
    lines.append(
        "*Every stage is scored against the same battery; weights are degraded "
        "fresh from the clean state per stage, so stages are independent "
        "snapshots of the same brain at increasing damage levels.*"
    )
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — sampling temp 0.7; "
                 f"run-to-run variance is expected.*")
    with open(TRAJECTORY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_chart(results, model_id, battery_name):
    tracks = [t for t, _ in results]
    scores = [s["final_iq_score"] for _, s in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Full Decline Trajectory — Estimated IQ vs. Stage (A1 → Q1)",
                 fontsize=13, fontweight="bold")

    ax.plot(tracks, scores, marker="o", linewidth=2.5, markersize=7,
            color="#e74c3c", label="Model IQ Trajectory")
    for x, y in zip(tracks, scores):
        ax.annotate(f"{y}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, fontweight="bold")

    # Clinical reference thresholds.
    ax.axhline(130, color="#2ecc71", ls="--", alpha=0.7, label="Lucid Baseline (130)")
    ax.axhline(100, color="#f1c40f", ls="--", alpha=0.7, label="Mild Friction (100)")
    ax.axhline(70, color="#e67e22", ls="--", alpha=0.7, label="Severe Deficit (70)")

    ax.set_xlabel("Track Profile (degradation stage)")
    ax.set_ylabel("Estimated IQ Score")
    ax.set_ylim(40, 155)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower left")
    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()
    fig.savefig(TRAJECTORY_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EATEOT full A1→Q1 decline trajectory")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducible lesion + sampling")
    args = parser.parse_args()

    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire
    ensure_data_dir()

    print(f"[+] Trajectory study: {len(TRACK_ORDER)} stages · seed {args.seed}", flush=True)
    lab = BrainLabEngine(args.model)

    results = run_trajectory(lab, battery, battery_name, args.seed)
    del lab
    gc.collect()

    if not results:
        print("[ERR] no successful runs in trajectory", file=sys.stderr)
        sys.exit(1)

    write_markdown(results, args.model, battery_name, args.seed)
    write_chart(results, args.model, battery_name)
    print(f"[OK] wrote {TRAJECTORY_MD}, {TRAJECTORY_PNG} "
          f"({len(results)} stages: {results[0][0]} → {results[-1][0]}) · quiz {battery_name}")


if __name__ == "__main__":
    main()
