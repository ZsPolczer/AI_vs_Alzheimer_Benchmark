#!/usr/bin/env python3
"""
EATEOT drug dose-response (trip) study.

Runs the IQ battery at increasing drug doses on a fixed degradation track
and plots IQ vs. dose — the psychoactive dose-response curve: does a higher
dose of lsd progressively wreck the brain?

  eateot-trip --drug lsd
  eateot-trip --drug salvia --doses 0,0.5,1
  eateot-trip --drug nzt --track C1 --model Qwen/Qwen2.5-0.5B-Instruct --trials 3 --seed 7
  eateot-trip --drug dmt --questionnaire visual_battery

The default dose sweep is derived from the drug's ``dose_cap`` in
``config/drugs.yaml`` (0 → 1.25× cap in six steps). Every battery run is
logged to telemetry with ``drug`` / ``dose`` recorded so reports can be
filtered per drug. Writes trip_report.md / .json and trip_curve.png to the
data directory (default outputs/).
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
from eateot.paths import TRIP_JSON, TRIP_MD, TRIP_PNG, ensure_data_dir
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery

# apps.lab.resolve_cli_drug validates --drug into a resolved spec (SystemExit
# on unknown names / bad doses). Cross-app import matches the apps.reserve ->
# apps.compare precedent.
from apps.lab import resolve_cli_drug

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRACK = "C1"
# Dose fractions of the drug's dose_cap used for the default sweep.
DEFAULT_DOSE_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]


def default_dose_sweep(drug: str, spec: dict | None = None) -> list[float]:
    """Six-point sweep from dose 0 to 1.25x the drug's ``dose_cap``.

    Scaled per drug so every catalog entry gets a comparable curve (e.g. lsd
    -> [0, 0.75, 1.5, 2.25, 3.0, 3.75]; microdose_lsd -> [0, 0.12, 0.25, ...]).
    Pass a pre-resolved ``spec`` to avoid resolving the drug twice.
    """
    if spec is None:
        spec = resolve_cli_drug(drug, 1.0)
    if spec is None:
        raise SystemExit("--drug is required (e.g. --drug lsd)")
    cap = spec["dose_cap"]
    doses = []
    for frac in DEFAULT_DOSE_FRACTIONS:
        dose = round(frac * cap, 2)
        if not doses or dose != doses[-1]:
            doses.append(dose)
    return doses


def parse_doses(raw: str) -> list[float]:
    """Parse '0,0.5,1,2' into non-negative floats."""
    values = [float(x) for x in raw.split(",") if x.strip() != ""]
    if not values:
        raise SystemExit("No doses given (e.g. --doses 0,0.5,1,2)")
    for v in values:
        if v < 0:
            raise SystemExit(f"Dose must be >= 0, got {v}")
    return values


def run_study(lab, drug, track, doses, trials, seed, battery, battery_name):
    """Run the battery at each (dose, trial) combo. Returns per-dose results."""
    results = []
    for dose in doses:
        print(f"[RUN] {drug} dose {dose:.2f} ...", flush=True)
        dose_results = []
        for trial in range(trials):
            log_path = f"/tmp/iq_trip_{drug}_{dose:.2f}_t{trial}.log"
            try:
                with open(log_path, "w", encoding="utf-8") as fh, redirect_stdout(fh):
                    summary = run_iq_test(
                        lab, track, 1.0, "all", False, False, False,
                        battery=battery, battery_name=battery_name,
                        seed=None if seed is None else seed + trial,
                        drug=drug, dose=dose,
                    )
                dose_results.append(summary)
                print(
                    f"  [DONE] trial {trial} -> IQ {summary['final_iq_score']} "
                    f"({summary['clinical_diagnosis']})",
                    flush=True,
                )
            except Exception:
                print(f"  [FAIL] trial {trial}", flush=True)
                traceback.print_exc(file=sys.stderr)
        results.append({"dose": dose, "trials": dose_results})
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


def write_markdown(results, drug, track, model_id, battery_name, seed):
    lines = [
        f"# 💊 Drug Dose-Response Study — {drug}",
        "",
        f"Model **{model_id}** · track **{track}** · questionnaire **{battery_name}** · "
        f"seed `{seed}`",
        "",
        "## IQ vs. Drug Dose",
        "",
        "| Dose (×) | Mean IQ | ±Std | Clinical Diagnosis |",
        "|---|---|---|---|",
    ]
    for dose in results:
        if not dose["trials"]:
            lines.append(f"| {dose['dose']:.2f} | — | — | failed |")
            continue
        iqs = [t["final_iq_score"] for t in dose["trials"]]
        mean, std = mean_std(iqs)
        diag = dose["trials"][0]["clinical_diagnosis"]
        lines.append(f"| {dose['dose']:.2f} | **{mean:.1f}** | {std:.1f} | {diag} |")
    lines.append("")
    lines.append(
        "*Dose 0.00 = sober baseline on the same track; each dose applies the "
        "resolved drug spec (see config/drugs.yaml) on top of the track lesion. "
        "Every run uses the same deterministic lesion seed, so doses differ "
        "only by the drug's perturbation strength.*"
    )
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — sampling "
                 f"temp 0.7; run-to-run variance is expected.*")
    with open(TRIP_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(results, drug, track, model_id, battery_name, seed):
    data = {
        "study": "drug dose-response",
        "drug": drug,
        "track_profile": track,
        "model": model_id,
        "questionnaire": battery_name,
        "seed": seed,
        "generated_at": datetime.now().isoformat(),
        "doses": [
            {
                "dose": dose["dose"],
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
    with open(TRIP_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_chart(results, drug, track):
    completed = [d for d in results if d["trials"]]
    doses = [d["dose"] for d in completed]
    means = []
    stds = []
    for dose in completed:
        iqs = [t["final_iq_score"] for t in dose["trials"]]
        mean, std = mean_std(iqs)
        means.append(mean)
        stds.append(std)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"Drug Dose-Response — IQ vs. {drug} dose · track {track}",
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
        doses, means, yerr=stds,
        marker="o", markersize=8, linewidth=2.5, color="#7b2cbf",
        ecolor="#e0aaff", capsize=4, label="Estimated IQ (mean ± std)",
    )
    for x, y in zip(doses, means):
        ax.annotate(f"{y:.0f} IQ", (x, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.set_xlabel(f"{drug} dose (×)")
    ax.set_ylabel("Estimated IQ Score")
    ax.set_ylim(35, 160)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(TRIP_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EATEOT drug dose-response (trip) study")
    parser.add_argument("--drug", type=str, required=True,
                        help="Drug name from config/drugs.yaml (e.g. lsd, salvia, nzt)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help="HuggingFace model ID")
    parser.add_argument("--track", type=str, default=DEFAULT_TRACK,
                        help=f"Degradation track to dose over (default: {DEFAULT_TRACK})")
    parser.add_argument("--doses", type=str, default=None,
                        help="Comma-separated doses (default: derived from the drug's dose_cap)")
    parser.add_argument("--trials", type=int, default=1,
                        help="Battery repeats per dose (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base seed for lesion + sampling")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    args = parser.parse_args()

    # Validate the drug up front (SystemExit with the catalog on unknown names).
    spec = resolve_cli_drug(args.drug, 1.0)
    if spec is None:
        raise SystemExit("--drug is required (e.g. --drug lsd)")

    if args.track not in EATEOT_TRACK_PROFILES:
        raise SystemExit(
            f"Unknown track '{args.track}'. Available: {', '.join(EATEOT_TRACK_PROFILES)}"
        )

    doses = (parse_doses(args.doses) if args.doses
             else default_dose_sweep(args.drug, spec=spec))
    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire
    ensure_data_dir()

    print(f"[+] Trip study: {args.drug} @ doses {doses} · track {args.track} · "
          f"trials {args.trials} · seed {args.seed}", flush=True)
    lab = BrainLabEngine(args.model)

    results = run_study(lab, args.drug, args.track, doses, args.trials, args.seed,
                        battery, battery_name)
    del lab
    gc.collect()

    if not any(d["trials"] for d in results):
        print("[ERR] no successful runs at any dose", file=sys.stderr)
        sys.exit(1)

    write_markdown(results, args.drug, args.track, args.model, battery_name, args.seed)
    write_json(results, args.drug, args.track, args.model, battery_name, args.seed)
    write_chart(results, args.drug, args.track)
    print(f"[OK] wrote {TRIP_MD}, {TRIP_JSON}, {TRIP_PNG} "
          f"({args.drug} · {len(doses)} doses × {args.trials} trials) · track {args.track} · quiz {battery_name}")


if __name__ == "__main__":
    main()
