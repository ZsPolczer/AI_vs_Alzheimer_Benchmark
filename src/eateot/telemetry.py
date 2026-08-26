"""Telemetry persistence: appends structured benchmark runs to the JSON log.

Two domains, two logs — runs made under a drug/stack (``drug`` set) append
to ``DRUG_LOG_FILE`` (``drug_test_results.json``); all other runs (track
lesions, severity sweeps, restore curves, clean baselines) append to
``LOG_FILE`` (``iq_test_results.json``). Alzheimer experiments and
mind-altering-drug experiments therefore never mix in the same telemetry.
"""

import json
import os
from datetime import datetime

from .paths import DRUG_LOG_FILE, LOG_FILE


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
    questionnaire: str = "iq_battery",
    seed: int | None = None,
    restore_fraction: float | None = None,
    drug: str | None = None,
    dose: float | None = None,
    deterioration_grade: float | None = None
):
    """Appends a structured benchmark run entry to the domain JSON telemetry.

    ``drug`` / ``dose`` (both optional) record which psychoactive profile was
    active during the run. When ``drug`` is set the run is a drug experiment
    and is logged to ``DRUG_LOG_FILE``; otherwise it is an Alzheimer-domain
    run and is logged to ``LOG_FILE``. ``dose`` is only meaningful when
    ``drug`` is set.
    """
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "config": {
            "track_profile": track_choice,
            "questionnaire": questionnaire,
            "decay_multiplier": decay_mult,
            "target_subnetwork": target_subnetwork,
            "seed": seed,
            "restore_fraction": restore_fraction,
            "drug": drug,
            "dose": dose,
            "toggles": {
                "flicker": flicker_mode,
                "sirens": sirens_mode,
                "surge": surge_mode
            }
        },
        "summary": {
            "final_iq_score": final_iq_score,
            "clinical_diagnosis": clinical_diag,
            "deterioration_grade": deterioration_grade
        },
        "domain_breakdown": detailed_results
    }

    log_file = DRUG_LOG_FILE if drug else LOG_FILE
    logs = []
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append(run_data)

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"💾 [✓] Run telemetry logged to '{log_file}'")
