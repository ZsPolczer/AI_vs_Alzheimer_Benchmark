import os
import json
import sys
import matplotlib.pyplot as plt

LOG_FILE = "iq_test_results.json"
IMG_FILE = "iq_decay_curve.png"

def reset_benchmark_data():
    """Deletes existing logs and chart images to reset the test state."""
    for file_path in [LOG_FILE, IMG_FILE]:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"🗑️ [✓] Deleted '{file_path}'")
    print("✨ Benchmark state reset! Run new IQ tests to generate fresh data.")

if __name__ == "__main__":
    # Check if user called `python plot_decay.py --reset`
    if len(sys.argv) > 1 and sys.argv[1] == "--reset":
        reset_benchmark_data()
    else:
        plt.close('all')  # Clears any existing figure buffers from memory
        # ... rest of your plot_iq_decay() call

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
    
    # Save chart image and show window
    plt.savefig("iq_decay_curve.png", dpi=300)
    print("📈 [✓] Chart successfully generated and saved to 'iq_decay_curve.png'")
    plt.show()

if __name__ == "__main__":
    plot_iq_decay()