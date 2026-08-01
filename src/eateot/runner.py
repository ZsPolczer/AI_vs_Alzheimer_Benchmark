"""IQ battery orchestration: run a questionnaire against an engine.

Port of ``run_iq_test`` from the original ``interactive_lab.py``. The question
set is configurable — pass any battery loaded from YAML (see
``eateot.questionnaires``) to run a different questionnaire.
"""

from .battery import IQ_TEST_BATTERY, evaluate_response
from .config import BASE_IQ, clinical_diagnosis
from .questionnaires import DEFAULT_BATTERY
from .telemetry import log_test_run


def run_iq_test(lab, track_choice, decay_mult, target_subnetwork, enable_flicker, enable_sirens, surge_mode, battery=None, battery_name=None):
    """Run the IQ battery against a lab engine.

    ``battery`` defaults to the standard ``IQ_TEST_BATTERY``; ``battery_name``
    (e.g. "iq_battery_mini") is recorded in telemetry so reports can tell
    questionnaires apart.
    """
    if battery is None:
        battery = IQ_TEST_BATTERY
    if battery_name is None:
        battery_name = DEFAULT_BATTERY

    print("\n" + "="*70)
    print(f" 🧪 RUNNING HARDENED NEURAL IQ BATTERY | TRACK: {track_choice} | QUIZ: {battery_name}")
    print("="*70)

    earned_points = 0
    results_breakdown = []
    detailed_log_entries = []

    for item in battery:
        tier = item["tier"]
        domain = item["domain"]
        q_text = item["question"]
        target_iq = item["target_iq"]

        print(f"\n[Tier {tier}] Domain: {domain} | Target IQ: {target_iq}")
        print(f"Question: \"{q_text}\"")

        # 1. Re-apply degradation weights dynamically for EACH question pass
        sys_prompt = lab.apply_degradation(
            track_choice,
            decay_mult=decay_mult,
            target_subnetwork=target_subnetwork,
            enable_flicker=enable_flicker,
            enable_sirens=enable_sirens
        )

        # 2. Run inference on degraded weights
        raw_output = lab.run_inference(q_text, sys_prompt, lucidity_surge=surge_mode)

        # 3. Clean up weights after evaluation pass
        lab.restore_clean_state()

        # 4. EVALUATE RESPONSE & CAPTURE PARTIAL SCORING
        score, status, accuracy_pct = evaluate_response(raw_output, item["ground_truth_anchors"], item["max_points"])
        earned_points += score

        results_breakdown.append((tier, domain, status, score, item['max_points']))

        detailed_log_entries.append({
            "tier": tier,
            "domain": domain,
            "target_iq": target_iq,
            "status": status,
            "accuracy_pct": accuracy_pct,
            "score_earned": score,
            "max_points": item['max_points'],
            "question": q_text,
            "raw_response": raw_output
        })

        print(f"➜ Diagnostic Status: [{status}] (Earned: {score}/{item['max_points']} pts | Accuracy: {accuracy_pct}%)")

    final_iq_score = BASE_IQ + earned_points

    # Clinical State Classification
    clinical_diag = clinical_diagnosis(final_iq_score)

    # Terminal Report Card Display
    print("\n" + "═"*70)
    print(" 📊 NEURAL IQ ASSESSMENT REPORT CARD")
    print("═"*70)
    print(f" Active Track Profile : {track_choice}")
    print(f" Sub-Network Target   : [{target_subnetwork.upper()}] | Decay Scale: {decay_mult}x")
    print("──────────────────────────────────────────────────────────────────────")
    for t, dom, stat, pts, max_p in results_breakdown:
        print(f"  • Tier {t} [{dom:<22}] : {stat:<40} | {pts}/{max_p} pts")
    print("──────────────────────────────────────────────────────────────────────")
    print(f" 🧮 ESTIMATED MODEL IQ SCORE : {final_iq_score} IQ")
    print(f" 🩺 DIAGNOSTIC STATE         : {clinical_diag}")
    print("═"*70 + "\n")

    # 5. Log telemetry entry to disk
    log_test_run(
        model_name=lab.model_id,
        track_choice=track_choice,
        decay_mult=decay_mult,
        target_subnetwork=target_subnetwork,
        flicker_mode=enable_flicker,
        sirens_mode=enable_sirens,
        surge_mode=surge_mode,
        questionnaire=battery_name,
        final_iq_score=final_iq_score,
        clinical_diag=clinical_diag,
        detailed_results=detailed_log_entries
    )
