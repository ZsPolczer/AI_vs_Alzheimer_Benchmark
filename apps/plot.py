#!/usr/bin/env python3
"""Plot the IQ decay curve from the telemetry log.

Port of ``plot_decay.py``, now resolving paths through ``eateot.paths``
(default ``outputs/``) and using a proper argparse CLI:

  eateot-plot            # generate the decay chart
  eateot-plot --reset    # delete telemetry log + chart
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; chart is saved to disk
import matplotlib.pyplot as plt

from eateot.paths import IMG_FILE, LOG_FILE


def reset_benchmark_data():
    """Deletes existing logs and chart images to reset the test state."""
    for file_path in [LOG_FILE, IMG_FILE]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ [✓] Deleted '{file_path}'")
    print("✨ Benchmark state reset! Run new IQ tests to generate fresh data.")


def plot_iq_decay():
    if not os.path.exists(LOG_FILE):
        print(f"❌ Error: Could not find '{LOG_FILE}'. Run a few IQ tests first!")
        return

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("⚠️ Log file is empty.")
        return

    # Extract x (Track Profile) and y (IQ Score)
    tracks = [run["config"]["track_profile"] for run in data]
    scores = [run["summary"]["final_iq_score"] for run in data]

    # Initialize plot layout
    plt.figure(figsize=(10, 6))

    # Plot line graph with markers
    plt.plot(tracks, scores, marker="o", linewidth=2.5, markersize=8, color="#e74c3c", label="Model IQ Trajectory")

    # Annotate score value above each data point
    for x, y in zip(tracks, scores):
        plt.annotate(
            f"{y} IQ",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    # Add clinical reference thresholds
    plt.axhline(y=130, color="#2ecc71", linestyle="--", alpha=0.7, label="Lucid Baseline (130)")
    plt.axhline(y=100, color="#f1c40f", linestyle="--", alpha=0.7, label="Mild Friction (100)")
    plt.axhline(y=70, color="#e67e22", linestyle="--", alpha=0.7, label="Severe Deficit (70)")

    # Chart Styling
    plt.title("Cognitive Decay Benchmark: IQ vs. Track Profile", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Track Profile (Degradation Level)", fontsize=12, labelpad=10)
    plt.ylabel("Estimated IQ Score", fontsize=12, labelpad=10)
    plt.ylim(40, 150)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left")

    plt.tight_layout()

    # Save chart image
    os.makedirs(os.path.dirname(IMG_FILE) or ".", exist_ok=True)
    plt.savefig(IMG_FILE, dpi=300)
    print(f"📈 [✓] Chart successfully generated and saved to '{IMG_FILE}'")
    plt.close('all')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_benchmark_data()
        return
    plot_iq_decay()


if __name__ == "__main__":
    main()
