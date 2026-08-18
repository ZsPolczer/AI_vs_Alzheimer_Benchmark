#!/usr/bin/env python3
"""
EATEOT dose-response restoration study.

Applies a fixed degradation track, then lerps the weights back toward the
clean state by increasing "restore fractions" (the treatment dose) and runs
the IQ battery at each dose. Models a partial treatment response curve.

  eateot-restore                          # G1 track, default doses, 3B model
  eateot-restore --track C1 --fractions 0,0.5,1
  eateot-restore --model Qwen/Qwen2.5-0.5B-Instruct --trials 3 --seed 7

Writes restoration_report.md / .json and restoration_curve.png to the data
directory (default outputs/). Every battery run is also logged to telemetry
with its ``restore_fraction`` and ``seed`` recorded.
"""

import argparse
import gc
import json
import sys
import traceback
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eateot import BrainLabEngine, run_iq_test
from eateot.config import CLINICAL_BANDS
from eateot.paths import RESTORE_JSON, RESTORE_MD, RESTORE_PNG, ensure_data_dir
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRACK = "G1"
DEFAULT_FRACTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_fractions(raw: str) -> list[float]:
    """Parse '0,0.25,1' into floats, validating the 0..1 range."""
    values = [float(x) for x in raw.split(",") if x.strip() != ""]
    if not values:
        raise SystemExit("No fractions given (e.g. --fractions 0,0.25,0.5,0.75,1)")
    for v in values:
        if not 0.0 <= v <= 1.0:
            raise SystemExit(f"Restore fraction must be in [0, 1], got {v}")
    return values


def run_study(lab, track, fractions, trials, seed, battery, battery_name):
    """Run the battery at each (fraction, trial) combo. Returns per-dose results."""
    results = []
    for frac in fractions:
        print(f"[RUN] restore fraction {frac:.2f} ...", flush=True)
        frac_results = []
        for trial in range(trials):
            log_path = f"/tmp/iq_restore_{track}_{frac:.2f}_t{trial}.log"
            try:
                with open(log_path, "w", encoding="utf-8") as fh, redirect_stdout(fh):
                    summary = run_iq_test(
                        lab, track, 1.0, "all", False, False, False,
                        battery=battery, battery_name=battery_name,
                        restore_fraction=frac,
                        seed=None if seed is None else seed + trial,
                    )
                frac_results.append(summary)
                print(
                    f"  [DONE] trial {trial} -> IQ {summary['final_iq_score']} "
                    f"({summary['clinical_diagnosis']})",
                    flush=True,
                )
            except Exception:
                print(f"  [FAIL] trial {trial}", flush=True)
                traceback.print_exc(file=sys.stderr)
        results.append({"fraction": frac, "trials": frac_results})
    return results


def mean_std(values):
    """(mean, std) — std is 0.0 for a single sample."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def write_markdown(results, track, model_id, battery_name, seed):
    lines = [
        f"# 🧪 Dose-Response Restoration Study — track {track}",
        "",
        f"Model **{model_id}** · questionnaire **{battery_name}** · "
        f"restore fractions [0 → 1] · seed `{seed}`",
        "",
        "## IQ vs. Restore Fraction (treatment dose)",
        "",
        "| Dose (restore frac) | Mean IQ | ±Std | Clinical Diagnosis |",
        "|---|---|---|---|",
    ]
    for dose in results:
        if not dose["trials"]:
            lines.append(f"| {dose['fraction']:.2f} | — | — | failed |")
            continue
        iqs = [t["final_iq_score"] for t in dose["trials"]]
        mean, std = mean_std(iqs)
        diag = dose["trials"][0]["clinical_diagnosis"]
        lines.append(f"| {dose['fraction']:.2f} | **{mean:.1f}** | {std:.1f} | {diag} |")
    lines.append("")
    lines.append(
        "*Dose 0.00 = fully degraded baseline; dose 1.00 = clean weights. "
        "Each battery run uses the same deterministic lesion seed, so doses "
        "differ only by how much of the clean state is restored.*"
    )
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — sampling temp 0.7; "
                 f"run-to-run variance is expected.*")
    with open(RESTORE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(results, track, model_id, battery_name, seed):
    data = {
        "study": "dose-response restoration",
        "track_profile": track,
        "model": model_id,
        "questionnaire": battery_name,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "doses": [
            {
                "fraction": dose["fraction"],
                "trials": [
                    {
                        "final_iq_score": t["final_iq_score"],
                        "clinical_diagnosis": t["clinical_diagnosis"],
                        "breakdown": t["breakdown"],
                    }
                    for t in dose["trials"]
                ],
            }
            for dose in results
        ],
    }
    with open(RESTORE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_chart(results):
    completed = [d for d in results if d["trials"]]
    fractions = [d["fraction"] for d in completed]
    means = []
    stds = []
    for dose in completed:
        iqs = [t["final_iq_score"] for t in dose["trials"]]
        mean, std = mean_std(iqs)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle("Dose-Response Restoration — IQ vs. Restore Fraction",
                 fontsize=13, fontweight="bold")

    # Clinical band shading (bands sorted descending: 130, 100, 80, 65, 0).
    bands = sorted(CLINICAL_BANDS, key=lambda b: b[0], reverse=True)
    for i, (thr, label) in enumerate(bands):
        top = 155 if i == 0 else bands[i - 1][0]
        bottom = bands[i + 1][0] if i + 1 < len(bands) else 0
        ax.axhspan(top, thr, color=f"C{i}", alpha=0.08)
        ax.axhline(thr, color="#888", ls="--", lw=0.8, alpha=0.5)
        ax.text(0.99, thr + 2, f"{thr} · {label}", fontsize=7.5, color="#555",
                ha="right", va="bottom")

    ax.errorbar(
        fractions, means, yerr=stds,
        marker="o", markersize=8, linewidth=2.5, color="#023047",
        ecolor="#219ebc", capsize=4, label="Estimated IQ (mean ± std)",
    )
    for x, y in zip(fractions, means):
        ax.annotate(f"{y:.0f} IQ", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel("Restore Fraction (treatment dose)")
    ax.set_ylabel("Estimated IQ Score")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(35, 160)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(RESTORE_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EATEOT dose-response restoration study")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--track", type=str, default=DEFAULT_TRACK,
                        help=f"Degradation track to restore from (default: {DEFAULT_TRACK})")
    parser.add_argument("--fractions", type=str,
                        default=",".join(str(f) for f in DEFAULT_FRACTIONS),
                        help="Comma-separated restore fractions in [0,1] (default: 0,0.2,...,1)")
    parser.add_argument("--trials", type=int, default=1, help="Battery repeats per dose (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for lesion + sampling")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    args = parser.parse_args()

    if args.track not in EATEOT_TRACK_PROFILES:
        raise SystemExit(
            f"Unknown track '{args.track}'. "
            f"Available: {', '.join(EATEOT_TRACK_PROFILES)}"
        )

    fractions = parse_fractions(args.fractions)
    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire
    ensure_data_dir()

    print(f"[+] Restoration study: track {args.track} · doses {fractions} · "
          f"trials {args.trials} · seed {args.seed}", flush=True)
    lab = BrainLabEngine(args.model)

    results = run_study(lab, args.track, fractions, args.trials, args.seed,
                        battery, battery_name)
    del lab
    gc.collect()

    if not any(d["trials"] for d in results):
        print("[ERR] no successful runs at any dose", file=sys.stderr)
        sys.exit(1)

    write_markdown(results, args.track, args.model, battery_name, args.seed)
    write_json(results, args.track, args.model, battery_name, args.seed)
    write_chart(results)
    print(f"[OK] wrote {RESTORE_MD}, {RESTORE_JSON}, {RESTORE_PNG} "
          f"({len(fractions)} doses × {args.trials} trials) · track {args.track} · quiz {battery_name}")


if __name__ == "__main__":
    main()
