#!/usr/bin/env python3
"""
EATEOT sensitivity study — std-scaled Gaussian perturbation.

Tests the hypothesis that performance degrades monotonically with the
magnitude of std-scaled weight perturbation, i.e. for

    Ẇ = W + ε · σ_W · Z      (Z ~ N(0,1), σ_W = std of weight tensor W)

    Performance(ε₀) > Performance(ε₁) > Performance(ε₂)   for ε₀ < ε₁ < ε₂

Each ε level runs the questionnaire across ``--trials`` seeds; the report
tracks the estimated IQ AND the numeric deterioration grade (0–100, higher =
more degraded) with mean ± std error margins.

  eateot-sensitivity                              # log ε-grid 1e-4 → 1e-1, 3B
  eateot-sensitivity --epsilons 0.0001,0.001,0.01 --trials 3 --seed 7
  eateot-sensitivity --track C1                   # on top of a track lesion
  eateot-sensitivity --questionnaire iq_battery_mini

Writes sensitivity_report.md / .json and sensitivity_decay.png to the data
directory (default outputs/). Every battery run is also logged to telemetry
with its epsilon recorded in the run config.
"""

import argparse
import gc
import json
import sys
import tempfile
import traceback
from contextlib import redirect_stdout
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eateot import BrainLabEngine, run_iq_test
from eateot.config import CLINICAL_BANDS
from eateot.paths import SENSITIVITY_JSON, SENSITIVITY_MD, SENSITIVITY_PNG, ensure_data_dir
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TRACK = "CLEAN"
# Logarithmic grid from 1e-4 to 1e-1 (roughly 3 steps per decade).
DEFAULT_EPSILONS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]


def parse_epsilons(raw: str) -> list[float]:
    """Parse '0.0001,0.001,0.01' into floats, validating ε >= 0."""
    values = [float(x) for x in raw.split(",") if x.strip() != ""]
    if not values:
        raise SystemExit("No epsilon values given (e.g. --epsilons 0.0001,0.001,0.01)")
    for v in values:
        if v < 0.0:
            raise SystemExit(f"Epsilon must be >= 0, got {v}")
    return values


def run_study(lab, track, epsilons, trials, seed, battery, battery_name):
    """Run the questionnaire at each (ε, trial) combo. Returns per-ε results."""
    results = []
    for eps in epsilons:
        print(f"[RUN] ε = {eps:g} ...", flush=True)
        eps_results = []
        for trial in range(trials):
            try:
                with tempfile.NamedTemporaryFile(
                        "w", suffix=f"_eps{eps:g}_t{trial}.log", delete=True,
                        encoding="utf-8") as fh, redirect_stdout(fh):
                    summary = run_iq_test(
                        lab, track, 1.0, "all", False, False, False,
                        battery=battery, battery_name=battery_name,
                        seed=None if seed is None else seed + trial,
                        epsilon=eps,
                    )
                eps_results.append(summary)
                print(
                    f"  [DONE] trial {trial} -> IQ {summary['final_iq_score']} | "
                    f"deterioration {summary.get('deterioration_grade', 0.0)}/100 "
                    f"({summary['clinical_diagnosis']})",
                    flush=True,
                )
            except Exception:
                print(f"  [FAIL] trial {trial}", flush=True)
                traceback.print_exc(file=sys.stderr)
        results.append({"epsilon": eps, "trials": eps_results})
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


def monotonic_verdict(results):
    """Check the hypothesis: mean IQ strictly decreases with ε (grade increases).

    Returns (verdict_str, violations) — violations counts the number of
    adjacent ε steps that break the monotonic trend.
    """
    completed = [r for r in results if r["trials"]]
    iqs = [mean_std([t["final_iq_score"] for t in r["trials"]])[0] for r in completed]
    grades = [mean_std([t.get("deterioration_grade", 0.0) for t in r["trials"]])[0]
              for r in completed]
    iq_violations = sum(1 for a, b in zip(iqs, iqs[1:]) if b >= a)
    grade_violations = sum(1 for a, b in zip(grades, grades[1:]) if b <= a)
    if iq_violations == 0:
        verdict = "✅ MONOTONIC: IQ strictly decreases as ε increases (hypothesis holds)."
    else:
        verdict = (f"⚠️ NON-MONOTONIC: {iq_violations} adjacent ε step(s) where IQ did "
                   f"not strictly decrease; {grade_violations} where the deterioration "
                   f"grade did not strictly increase.")
    return verdict, iq_violations, grade_violations


def write_markdown(results, track, model_id, battery_name, seed, verdict, iq_violations, grade_violations):
    lines = [
        f"# 📉 Sensitivity Study — Std-Scaled Gaussian Perturbation (Ẇ = W + ε·σ_W·Z)",
        "",
        f"Model **{model_id}** · questionnaire **{battery_name}** · track **{track}** · "
        f"seed `{seed}`",
        "",
        "Hypothesis: as the perturbation strength ε increases, output quality "
        "decreases monotonically — especially on factual-recall / knowledge-intensive "
        "tasks.",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## IQ vs. ε",
        "",
        "| ε | Mean IQ | ±Std | Mean Deterioration (/100) | ±Std | Clinical Diagnosis |",
        "|---|---|---|---|---|---|",
    ]
    for res in results:
        if not res["trials"]:
            lines.append(f"| {res['epsilon']:g} | — | — | — | — | failed |")
            continue
        iqs = [t["final_iq_score"] for t in res["trials"]]
        grades = [t.get("deterioration_grade", 0.0) for t in res["trials"]]
        iq_mean, iq_std = mean_std(iqs)
        g_mean, g_std = mean_std(grades)
        diag = res["trials"][0]["clinical_diagnosis"]
        lines.append(f"| {res['epsilon']:g} | **{iq_mean:.1f}** | {iq_std:.1f} | "
                     f"{g_mean:.1f} | {g_std:.1f} | {diag} |")
    lines.append("")
    lines.append(
        "*Deterioration grade: 0 = pristine answer, 100 = fully degraded "
        "(composite of anchor-correctness, lexical repetition and perseveration). "
        "Each trial uses a different seed for both the noise Z and the sampling.*"
    )
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
                 f"monotonic violations: {iq_violations} (IQ) / {grade_violations} (grade).*")
    with open(SENSITIVITY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(results, track, model_id, battery_name, seed, verdict, iq_violations, grade_violations):
    data = {
        "study": "std-scaled gaussian perturbation sensitivity",
        "formula": "W_tilde = W + eps * std(W) * Z",
        "track_profile": track,
        "model": model_id,
        "questionnaire": battery_name,
        "seed": seed,
        "monotonic_verdict": verdict,
        "monotonic_violations": {"iq": iq_violations, "deterioration_grade": grade_violations},
        "generated_at": datetime.now().isoformat(),
        "epsilons": [
            {
                "epsilon": res["epsilon"],
                "trials": [
                    {
                        "final_iq_score": t["final_iq_score"],
                        "deterioration_grade": t.get("deterioration_grade"),
                        "clinical_diagnosis": t["clinical_diagnosis"],
                        "breakdown": t["breakdown"],
                    }
                    for t in res["trials"]
                ],
            }
            for res in results
        ],
    }
    with open(SENSITIVITY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_chart(results):
    completed = [r for r in results if r["trials"]]
    eps = [r["epsilon"] for r in completed]
    iq_means = []
    iq_stds = []
    grade_means = []
    grade_stds = []
    for res in completed:
        iqs = [t["final_iq_score"] for t in res["trials"]]
        grades = [t.get("deterioration_grade", 0.0) for t in res["trials"]]
        m1, s1 = mean_std(iqs)
        m2, s2 = mean_std(grades)
        iq_means.append(m1)
        iq_stds.append(s1)
        grade_means.append(m2)
        grade_stds.append(s2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("Sensitivity to Std-Scaled Gaussian Perturbation (Ẇ = W + ε·σ_W·Z)",
                 fontsize=13, fontweight="bold")

    # Left: IQ vs ε with clinical band shading.
    bands = sorted(CLINICAL_BANDS, key=lambda b: b[0], reverse=True)
    for i, (thr, label) in enumerate(bands):
        top = 155 if i == 0 else bands[i - 1][0]
        bottom = bands[i + 1][0] if i + 1 < len(bands) else 0
        ax1.axhspan(top, thr, color=f"C{i}", alpha=0.08)
        ax1.axhline(thr, color="#888", ls="--", lw=0.8, alpha=0.5)
    ax1.errorbar(eps, iq_means, yerr=iq_stds, marker="o", markersize=7,
                 linewidth=2.5, color="#023047", ecolor="#219ebc", capsize=4,
                 label="Estimated IQ (mean ± std)")
    for x, y in zip(eps, iq_means):
        ax1.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    ax1.set_xscale("log")
    ax1.set_xlabel("Perturbation strength ε (log scale)")
    ax1.set_ylabel("Estimated IQ Score")
    ax1.set_ylim(35, 160)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend(loc="lower left")

    # Right: deterioration grade vs ε.
    ax2.errorbar(eps, grade_means, yerr=grade_stds, marker="s", markersize=7,
                 linewidth=2.5, color="#8b1a1a", ecolor="#e07a5f", capsize=4,
                 label="Deterioration grade /100 (mean ± std)")
    for x, y in zip(eps, grade_means):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8, fontweight="bold")
    ax2.set_xscale("log")
    ax2.set_xlabel("Perturbation strength ε (log scale)")
    ax2.set_ylabel("Deterioration Grade (0 = pristine, 100 = degraded)")
    ax2.set_ylim(0, 105)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(SENSITIVITY_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EATEOT std-scaled Gaussian perturbation sensitivity study")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HuggingFace model ID")
    parser.add_argument("--track", type=str, default=DEFAULT_TRACK,
                        help="Track to perturb (default: CLEAN = pure ε, no track lesion)")
    parser.add_argument("--epsilons", type=str,
                        default=",".join(str(e) for e in DEFAULT_EPSILONS),
                        help="Comma-separated ε values (default: log grid 1e-4 → 1e-1)")
    parser.add_argument("--trials", type=int, default=1, help="Battery repeats per ε (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for noise Z + sampling")
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

    epsilons = parse_epsilons(args.epsilons)
    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire
    ensure_data_dir()

    print(f"[+] Sensitivity study: track {args.track} · ε grid {epsilons} · "
          f"trials {args.trials} · seed {args.seed}", flush=True)
    lab = BrainLabEngine(args.model)

    results = run_study(lab, args.track, epsilons, args.trials, args.seed,
                        battery, battery_name)
    del lab
    gc.collect()

    if not any(r["trials"] for r in results):
        print("[ERR] no successful runs at any ε", file=sys.stderr)
        sys.exit(1)

    verdict, iq_violations, grade_violations = monotonic_verdict(results)
    print(f"\n{verdict}\n")

    write_markdown(results, args.track, args.model, battery_name, args.seed,
                   verdict, iq_violations, grade_violations)
    write_json(results, args.track, args.model, battery_name, args.seed,
               verdict, iq_violations, grade_violations)
    write_chart(results)
    print(f"[OK] wrote {SENSITIVITY_MD}, {SENSITIVITY_JSON}, {SENSITIVITY_PNG} "
          f"({len(epsilons)} ε levels × {args.trials} trials) · track {args.track} · quiz {battery_name}")


if __name__ == "__main__":
    main()
