#!/usr/bin/env python3
"""
EATEOT cognitive reserve study.

Runs the IQ battery across a matrix of {model size × lesion severity} and
plots IQ vs. severity as one line per model — the cognitive-reserve
question: do larger models resist degradation longer?

Severity is the ``decay_multiplier`` (lesion noise × mult, weight scale ÷
mult) applied on top of a fixed track profile (default C1). Higher = worse.

  eateot-reserve                           # 0.5B/1.5B/3B × severities [1,2,4,8] on C1
  eateot-reserve --track G1 --severities 1,2,4
  eateot-reserve --models 3B --trials 2 --seed 7

Writes reserve_report.md / .json and reserve_curve.png to the data
directory (default outputs/). Models NOT freshly run are pulled from
existing telemetry (matching track + questionnaire + decay multiplier), so
partial re-runs still produce a full report.
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

from eateot import BrainLabEngine, LOG_FILE, run_iq_test
from eateot.config import CLINICAL_BANDS
from eateot.paths import RESERVE_JSON, RESERVE_MD, RESERVE_PNG, ensure_data_dir
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery

# MODELS / resolve_models live in apps.compare as the canonical model list;
# importing them keeps a single source of truth for the three Qwen sizes.
from apps.compare import MODELS, resolve_models

DEFAULT_TRACK = "C1"
DEFAULT_SEVERITIES = [1.0, 2.0, 4.0, 8.0]
RESERVE_THRESHOLD = 100  # IQ boundary used for the "collapse severity" metric


def parse_severities(raw: str) -> list[float]:
    """Parse '1,2,4,8' into positive floats."""
    values = [float(x) for x in raw.split(",") if x.strip() != ""]
    if not values:
        raise SystemExit("No severities given (e.g. --severities 1,2,4,8)")
    for v in values:
        if v <= 0:
            raise SystemExit(f"Severity (decay multiplier) must be > 0, got {v}")
    return values


def auc_curve(xs: list[float], ys: list[float]) -> float:
    """Trapezoidal area under the (x, y) curve — a 'cognitive reserve' index."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    return sum((ys[i] + ys[i + 1]) / 2.0 * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))


def collapse_severity(xs: list[float], ys: list[float], threshold: float = RESERVE_THRESHOLD):
    """First severity (linearly interpolated) where IQ drops below ``threshold``.

    Returns ``None`` if the curve never crosses downward (i.e. the model
    keeps an IQ >= threshold across the whole sweep).
    """
    for i in range(len(xs) - 1):
        y0, y1 = ys[i], ys[i + 1]
        if y1 < threshold <= y0:
            frac = (y0 - threshold) / (y0 - y1) if y0 != y1 else 1.0
            return xs[i] + frac * (xs[i + 1] - xs[i])
    return None


def _latest_entry(model_id, track, questionnaire, decay_mult):
    """Most recent telemetry entry matching model + track + quiz + severity, or None.

    Non-pure-severity runs are excluded: dose-response runs (``restore_fraction``
    set) because their weights were partially restored, and drug runs (``drug``
    set) because the drug perturbs weights/sampling — neither is a pure
    severity measurement, so both would confound the reserve curve.
    """
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
        if cfg.get("decay_multiplier") != decay_mult:
            continue
        if cfg.get("restore_fraction") is not None:
            continue
        if cfg.get("drug"):
            continue
        return entry
    return None


def _entry_to_summary(entry):
    return {
        "final_iq_score": entry["summary"]["final_iq_score"],
        "clinical_diagnosis": entry["summary"]["clinical_diagnosis"],
        "breakdown": entry.get("domain_breakdown", []),
    }


def run_matrix(model_ids, track, severities, trials, seed, battery, battery_name):
    """Freshly benchmark each selected model across the severity sweep.

    Returns {short: {severity: [summary, ...]}} built directly from the
    summaries returned by ``run_iq_test`` (NOT re-read from telemetry — that
    would duplicate the latest entry when ``trials > 1``). Models that fail to
    load are skipped so one bad model can't abort the whole study.
    """
    results = {}
    for model_id in model_ids:
        short = model_id.split("/")[-1]
        print(f"[RUN] {short} (track {track}, severities {severities}) ...", flush=True)
        try:
            lab = BrainLabEngine(model_id)
        except Exception:
            print(f"[FAIL] {short}: could not load model", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            continue
        try:
            for sev in severities:
                for trial in range(trials):
                    log_path = f"/tmp/iq_reserve_{short}_sev{sev}_t{trial}.log"
                    try:
                        with open(log_path, "w", encoding="utf-8") as fh, redirect_stdout(fh):
                            summary = run_iq_test(
                                lab, track, sev, "all", False, False, False,
                                battery=battery, battery_name=battery_name,
                                seed=None if seed is None else seed + trial,
                            )
                        results.setdefault(short, {}).setdefault(sev, []).append(summary)
                        print(f"  [DONE] severity {sev} trial {trial}", flush=True)
                    except Exception:
                        print(f"  [FAIL] severity {sev} trial {trial}", flush=True)
                        traceback.print_exc(file=sys.stderr)
        finally:
            del lab
            gc.collect()
    return results


def collect_report(models, results, track, battery_name, severity_list):
    """Assemble {short: {severity: summary}} for the report.

    Models freshly run come from ``results``; any other model in ``models``
    is pulled from its latest matching telemetry entries.
    """
    report = {}
    for model_id in models:
        short = model_id.split("/")[-1]
        rows = {}
        for sev in severity_list:
            summaries = results.get(short, {}).get(sev, [])
            if not summaries:
                entry = _latest_entry(model_id, track, battery_name, sev)
                if entry is not None:
                    summaries = [_entry_to_summary(entry)]
            rows[sev] = summaries
        report[short] = rows
    return report


def mean_std(values):
    """(mean, std) — std is 0.0 for a single sample."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return mean, variance ** 0.5


def reserve_metrics(rows):
    """AUC + collapse severity for one model's severity→IQ curve.

    Skips severities with no data, matching ``write_chart`` — an empty cell
    must not be treated as IQ 0.0.
    """
    xs = sorted(sev for sev in rows if rows[sev])
    ys = [mean_std([t["final_iq_score"] for t in rows[x]])[0] for x in xs]
    return auc_curve(xs, ys), collapse_severity(xs, ys)


def write_markdown(report, track, battery_name, seed, severity_list):
    shorts = list(report.keys())
    lines = [
        f"# 🧠 Cognitive Reserve Study — track {track}",
        "",
        f"Questionnaire **{battery_name}** · severities {severity_list} · seed `{seed}` · "
        "one line per model",
        "",
        "## Estimated IQ by Severity (decay multiplier)",
        "",
        "| Severity (×) | " + " | ".join(shorts) + " |",
        "|---|" + "---|" * len(shorts),
    ]
    for sev in severity_list:
        cells = []
        for short in shorts:
            summaries = report[short].get(sev, [])
            if not summaries:
                cells.append("—")
            else:
                mean, _ = mean_std([t["final_iq_score"] for t in summaries])
                cells.append(f"**{mean:.0f}**")
        lines.append(f"| {sev:g} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## Reserve Metrics")
    lines.append("")
    lines.append("| Model | AUC (IQ·severity) | Collapse severity (IQ < 100) |")
    lines.append("|---|---|---|")
    for short in shorts:
        auc, collapse = reserve_metrics(report[short])
        collapse_txt = f"{collapse:.2f}×" if collapse is not None else "never (< threshold)"
        lines.append(f"| {short} | {auc:.1f} | {collapse_txt} |")
    lines.append("")
    lines.append(
        "*Severity is the decay multiplier on track "
        f"{track} (noise × mult, scale ÷ mult). AUC = trapezoidal area under "
        "the IQ-vs-severity curve; higher = more cognitive reserve.*"
    )
    lines.append("")
    lines.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — sampling temp 0.7; "
                 f"run-to-run variance is expected.*")
    with open(RESERVE_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_json(report, track, battery_name, seed, severity_list):
    data = {
        "study": "cognitive reserve",
        "track_profile": track,
        "questionnaire": battery_name,
        "seed": seed,
        "severities": severity_list,
        "generated_at": datetime.now().isoformat(),
        "models": {},
    }
    for short, rows in report.items():
        data["models"][short] = {
            "curves": [
                {
                    "severity": sev,
                    "trials": [t["final_iq_score"] for t in rows[sev]],
                    "mean_iq": mean_std([t["final_iq_score"] for t in rows[sev]])[0],
                    "clinical_diagnosis": rows[sev][0]["clinical_diagnosis"]
                    if rows[sev] else None,
                }
                for sev in severity_list
            ],
            "auc": reserve_metrics(rows)[0],
            "collapse_severity": reserve_metrics(rows)[1],
        }
    with open(RESERVE_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_chart(report, track, severity_list):
    shorts = list(report.keys())
    colors = ["#8ecae6", "#219ebc", "#023047"]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(f"Cognitive Reserve — IQ vs. Lesion Severity · track {track}",
                 fontsize=13, fontweight="bold")

    # Clinical band shading (bands sorted descending: 130, 100, 80, 65, 0).
    bands = sorted(CLINICAL_BANDS, key=lambda b: b[0], reverse=True)
    for i, (thr, label) in enumerate(bands):
        top = 155 if i == 0 else bands[i - 1][0]
        bottom = bands[i + 1][0] if i + 1 < len(bands) else 0
        ax.axhspan(top, thr, color=f"C{i}", alpha=0.08)
    for thr, label in bands:
        ax.axhline(thr, color="#888", ls="--", lw=0.8, alpha=0.5)

    for i, short in enumerate(shorts):
        xs = [sev for sev in severity_list if report[short].get(sev)]
        ys = [mean_std([t["final_iq_score"] for t in report[short][sev]])[0] for sev in xs]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=2.5,
                color=colors[i % len(colors)], label=short)
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8.5,
                        color=colors[i % len(colors)], fontweight="bold")

    ax.axvline(1.0, color="#666", ls=":", lw=1.2, alpha=0.8)
    ax.text(1.0, 42, "profile baseline (1.0×)", fontsize=7.5, color="#666",
            ha="center", va="bottom")

    ax.set_xscale("log", base=2)
    ax.set_xticks(severity_list)
    ax.set_xticklabels([f"{s:g}×" for s in severity_list])
    ax.set_xlabel("Lesion Severity (decay multiplier, log scale)")
    ax.set_ylabel("Estimated IQ Score")
    ax.set_ylim(35, 160)
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(RESERVE_PNG, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="EATEOT cognitive reserve study (IQ vs severity per model)")
    parser.add_argument("--models", nargs="*", default=[],
                        help="Subset to benchmark fresh (short names or full ids); others come from telemetry")
    parser.add_argument("--track", type=str, default=DEFAULT_TRACK,
                        help=f"Degradation track to sweep severity over (default: {DEFAULT_TRACK})")
    parser.add_argument("--severities", type=str,
                        default=",".join(str(s) for s in DEFAULT_SEVERITIES),
                        help="Comma-separated decay multipliers (default: 1,2,4,8)")
    parser.add_argument("--trials", type=int, default=1, help="Battery repeats per cell (default: 1)")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for lesion + sampling")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    args = parser.parse_args()

    severity_list = parse_severities(args.severities)

    if args.track not in EATEOT_TRACK_PROFILES:
        raise SystemExit(
            f"Unknown track '{args.track}'. Available: {', '.join(EATEOT_TRACK_PROFILES)}"
        )

    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire
    ensure_data_dir()

    selected = resolve_models(args.models)
    print(f"[+] Reserve study: {[m.split('/')[-1] for m in selected]} × severities {severity_list} "
          f"· track {args.track} · trials {args.trials} · seed {args.seed}", flush=True)

    fresh = run_matrix(selected, args.track, severity_list, args.trials, args.seed,
                       battery, battery_name)
    report = collect_report(MODELS, fresh, args.track, battery_name, severity_list)

    completed = {s: rows for s, rows in report.items()
                 if any(rows.get(sev) for sev in severity_list)}
    if not completed:
        print("[ERR] no results available for any model", file=sys.stderr)
        sys.exit(1)

    write_markdown(completed, args.track, battery_name, args.seed, severity_list)
    write_json(completed, args.track, battery_name, args.seed, severity_list)
    write_chart(completed, args.track, severity_list)
    print(f"[OK] wrote {RESERVE_MD}, {RESERVE_JSON}, {RESERVE_PNG} "
          f"({len(completed)} models × {len(severity_list)} severities) · track {args.track} · quiz {battery_name}")


if __name__ == "__main__":
    main()
