"""Central configuration: defaults, IQ thresholds, and small helpers.

All tunable constants for the lab live here so runnables and the engine
stay free of magic numbers.
"""

# --- Model defaults ---------------------------------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"

# --- Lab defaults -----------------------------------------------------------
DEFAULT_DECAY = 1.0
DEFAULT_SUBNET = "all"
DEFAULT_FLICKER = False
DEFAULT_SIRENS = False
DEFAULT_SURGE = False

# --- IQ battery -------------------------------------------------------------
# The IQ scale maps a battery's earned points onto a FIXED range:
#
#     IQ = BASE_IQ + round(earned_points / total_points * (IQ_CEILING - BASE_IQ))
#
# BASE_IQ is the floor (a 0-point run) and IQ_CEILING is the designed ceiling
# of the test (a perfect score) — so no run can ever report an out-of-range
# IQ like 170+, no matter how easy the questions are. The ceiling sits just
# above the top clinical band (130+ "Superior"), matching the upper end of
# real IQ tests.
BASE_IQ = 50
IQ_CEILING = 145

# Clinical bands: (min_iq_score, diagnosis). First matching band wins.
CLINICAL_BANDS = [
    (130, "Superior Cognitive Function (Lucid Baseline)"),
    (100, "Average Reasoning (Mild Functional Friction)"),
    (80, "Mild Cognitive Impairment (Early Perseveration / Aphasia)"),
    (65, "Moderate Stage Disintegration (Severe Logical Deficit)"),
    (0, "Advanced Stage Plaque / Structural Collapse"),
]


def clinical_diagnosis(score: int) -> str:
    """Map a raw IQ score to its clinical diagnosis band."""
    for threshold, label in CLINICAL_BANDS:
        if score >= threshold:
            return label
    return CLINICAL_BANDS[-1][1]
