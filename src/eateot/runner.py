"""IQ battery orchestration: run a questionnaire against an engine.

Port of ``run_iq_test`` from the original ``interactive_lab.py``. The question
set is configurable — pass any battery loaded from YAML (see
``eateot.questionnaires``) to run a different questionnaire.
"""

from .battery import IQ_TEST_BATTERY, evaluate_question, grade_deterioration
from .config import BASE_IQ, IQ_CEILING, clinical_diagnosis
from .drugs import resolve_drug
from .questionnaires import DEFAULT_BATTERY
from .telemetry import log_test_run


def run_iq_test(lab, track_choice, decay_mult, target_subnetwork, enable_flicker, enable_sirens, surge_mode, battery=None, battery_name=None, restore_fraction=None, seed=None, drug=None, dose=1.0, epsilon=0.0):
    """Run the IQ battery against a lab engine.

    ``battery`` defaults to the standard ``IQ_TEST_BATTERY``; ``battery_name``
    (e.g. "iq_battery_mini") is recorded in telemetry so reports can tell
    questionnaires apart.

    ``restore_fraction`` (0.0–1.0) lerps the degraded weights toward clean
    before each question (dose-response studies); ``seed`` makes the weight
    corruption and sampling deterministic. Returns a summary dict
    ``{final_iq_score, clinical_diagnosis, track_profile, battery_name}``.

    ``drug`` (optional) names a psychoactive profile from the drug catalog
    (``eateot.drugs``) and ``dose`` sets its intensity. It may also be a
    pre-resolved spec dict — e.g. a combo returned by
    ``eateot.drugs.resolve_stack`` (``drug='lsd@1+thc@0.5'`` style specs are
    passed as the dict itself; ``dose`` is then ignored because each
    component's dose was already applied). The resolved spec is passed to
    ``apply_degradation`` and ``run_inference``, and the drug name
    (``drug`` or the stack's ``name`` label) / ``dose`` are recorded in
    telemetry so reports can filter per drug. If ``restore_fraction`` is not
    given but the drug defines one (e.g. ``nzt``), the drug's restore
    fraction is applied.

    ``epsilon`` (optional, default 0.0) is an explicit std-scaled Gaussian
    perturbation strength (Ẇ = W + ε·σ_W·Z) applied on top of the track/drug
    degradation — the sensitivity-study method. Each question is scored with
    ``grade_deterioration`` (0–100, higher = more deteriorated) and the mean
    grade is reported and logged alongside the IQ score.

    The IQ score is normalized onto the fixed ``BASE_IQ``..``IQ_CEILING``
    range (see ``eateot.config``): earned points are scaled by the battery's
    total max points, so a perfect score equals the ceiling and no run can
    exceed it.
    """
    if battery is None:
        battery = IQ_TEST_BATTERY
    if battery_name is None:
        battery_name = DEFAULT_BATTERY

    if isinstance(drug, dict):
        # Pre-resolved spec (e.g. a stack from eateot.drugs.resolve_stack).
        drug_spec = drug
        drug_name = drug_spec.get("name")
        drug_dose = drug_spec.get("dose")
    elif drug:
        drug_spec = resolve_drug(drug, dose)
        drug_name = drug
        drug_dose = dose
    else:
        drug_spec = None
        drug_name = None
        drug_dose = None

    effective_restore = restore_fraction
    if effective_restore is None and drug_spec is not None:
        effective_restore = drug_spec.get("restore_fraction")

    print("\n" + "="*70)
    if track_choice == "CLEAN":
        header = f" 🧪 RUNNING NEURAL IQ BATTERY | TRACK: CLEAN (drug-only, no Alzheimer's) | QUIZ: {battery_name}"
    else:
        header = f" 🧪 RUNNING HARDENED NEURAL IQ BATTERY | TRACK: {track_choice} | QUIZ: {battery_name}"
    if drug_spec is not None:
        if drug_spec.get("components"):
            header += f" | DRUG STACK: {drug_spec['name']}"
        else:
            header += f" | DRUG: {drug_name} @ {drug_dose}x"
    print(header)
    print("="*70)

    earned_points = 0
    results_breakdown = []
    detailed_log_entries = []
    deterioration_grades = []

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
            enable_sirens=enable_sirens,
            noise_seed=seed,
            drug=drug_spec,
            epsilon=epsilon,
        )

        if effective_restore is not None:
            lab.lerp_toward_clean(effective_restore)

        # 2. Run inference on degraded weights
        raw_output = lab.run_inference(q_text, sys_prompt, lucidity_surge=surge_mode,
                                       seed=seed, drug=drug_spec)

        # 3. Clean up weights after evaluation pass
        lab.restore_clean_state()

        # 4. EVALUATE RESPONSE & CAPTURE PARTIAL SCORING
        score, status, accuracy_pct, metrics = evaluate_question(raw_output, item)
        earned_points += score

        # 4b. Deterioration grade (0-100, higher = more degraded).
        deterioration = grade_deterioration(raw_output, item)
        deterioration_grades.append(deterioration)

        results_breakdown.append((tier, domain, status, score, item['max_points']))

        detailed_log_entries.append({
            "tier": tier,
            "domain": domain,
            "target_iq": target_iq,
            "status": status,
            "accuracy_pct": accuracy_pct,
            "score_earned": score,
            "max_points": item['max_points'],
            "deterioration_grade": deterioration,
            "question": q_text,
            "raw_response": raw_output,
            **({"metrics": metrics} if metrics else {})
        })

        print(f"➜ Diagnostic Status: [{status}] (Earned: {score}/{item['max_points']} pts | Accuracy: {accuracy_pct}%"
              f" | Deterioration: {deterioration:.0f}/100)")

    # Normalized IQ scale: the battery's total max points map onto the fixed
    # BASE_IQ..IQ_CEILING range, so a perfect score equals the test's designed
    # ceiling (never 170+) and partial credit is proportional (see eateot.config).
    total_points = sum(item.get("max_points", 0) for item in battery) or 1
    final_iq_score = BASE_IQ + round(
        earned_points / total_points * (IQ_CEILING - BASE_IQ)
    )
    mean_deterioration = (sum(deterioration_grades) / len(deterioration_grades)
                          if deterioration_grades else 0.0)

    # Clinical State Classification
    clinical_diag = clinical_diagnosis(final_iq_score)

    # Terminal Report Card Display
    print("\n" + "═"*70)
    print(" 📊 NEURAL IQ ASSESSMENT REPORT CARD")
    print("═"*70)
    if track_choice == "CLEAN":
        print(f" Active Track Profile : CLEAN (drug-only, no Alzheimer's)")
    else:
        print(f" Active Track Profile : {track_choice}")
    if drug_spec is not None:
        if drug_spec.get("components"):
            print(f" Active Drug          : {drug_spec['name']} "
                  f"(stack · {len(drug_spec['components'])} components)")
        else:
            print(f" Active Drug          : {drug_name} @ {drug_dose}x "
                  f"({drug_spec['class']} · {drug_spec['target_domain']})")
    print(f" Sub-Network Target   : [{target_subnetwork.upper()}] | Decay Scale: {decay_mult}x")
    print("──────────────────────────────────────────────────────────────────────")
    for t, dom, stat, pts, max_p in results_breakdown:
        print(f"  • Tier {t} [{dom:<22}] : {stat:<40} | {pts}/{max_p} pts")
    print("──────────────────────────────────────────────────────────────────────")
    print(f" 🧮 ESTIMATED MODEL IQ SCORE : {final_iq_score} IQ "
          f"(scale: {BASE_IQ} floor · {IQ_CEILING} ceiling)")
    print(f" 🩺 DIAGNOSTIC STATE         : {clinical_diag}")
    print(f" 💀 DETERIORATION GRADE      : {mean_deterioration:.1f}/100 "
          f"(0 = pristine · 100 = fully degraded)")
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
        seed=seed,
        # Record what was ACTUALLY applied: a drug may carry its own
        # restore_fraction (e.g. nzt), which the explicit param would hide.
        restore_fraction=effective_restore,
        drug=drug_name,
        dose=drug_dose,
        final_iq_score=final_iq_score,
        clinical_diag=clinical_diag,
        deterioration_grade=round(mean_deterioration, 1),
        detailed_results=detailed_log_entries
    )

    return {
        "final_iq_score": final_iq_score,
        "clinical_diagnosis": clinical_diag,
        "deterioration_grade": round(mean_deterioration, 1),
        "track_profile": track_choice,
        "battery_name": battery_name,
        "breakdown": detailed_log_entries,
    }
