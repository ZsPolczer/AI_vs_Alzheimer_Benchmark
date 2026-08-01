"""Telemetry persistence: appends structured benchmark runs to the JSON log."""

import json
import os
from datetime import datetime

from .paths import LOG_FILE


def log_test_run(
    model_name: str,
    track_choice: str,
    decay_mult: float,
    target_subnetwork: str,
    flicker_mode: bool,
    sirens_mode: bool,
    surge_mode: bool,
    final_iq_score: int,
    clinical_diag: str,
    detailed_results: list[dict],
    questionnaire: str = "iq_battery"
):
    """Appends a structured benchmark run entry to the JSON telemetry log."""
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "config": {
            "track_profile": track_choice,
            "questionnaire": questionnaire,
            "decay_multiplier": decay_mult,
            "target_subnetwork": target_subnetwork,
            "toggles": {
                "flicker": flicker_mode,
                "sirens": sirens_mode,
                "surge": surge_mode
            }
        },
        "summary": {
            "final_iq_score": final_iq_score,
            "clinical_diagnosis": clinical_diag
        },
        "domain_breakdown": detailed_results
    }

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append(run_data)

    os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"💾 [✓] Run telemetry logged to '{LOG_FILE}'")
