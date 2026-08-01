"""IQ battery scoring logic.

The question spec itself lives in versioned YAML under
``config/questionnaires/`` (see ``eateot.questionnaires``) — this module only
holds the pure ``evaluate_response`` scoring function.
"""

import re

from .questionnaires import DEFAULT_BATTERY, load_battery

# The default battery loaded from YAML. Swap it at runtime by calling
# ``load_battery("other_name")`` and passing the result to ``run_iq_test``.
IQ_TEST_BATTERY = load_battery(DEFAULT_BATTERY)


def evaluate_response(raw_output: str, ground_truth_anchors: list[list[str]], max_points: int):
    """
    Evaluates response against multiple anchor groups (synonyms).
    Awards proportional partial credit based on how many distinct anchors were matched.
    """
    if not raw_output or raw_output.strip() in ["[NO OUTPUT GENERATED]", "[INFERENCE ERROR]"]:
        return 0, "FAILED (Null / Empty Generation)", 0.0

    text_lower = raw_output.lower()

    # 1. Genuine Perseveration Check (Consecutive phrase loops)
    consecutive_loop_pattern = r'(\b\w+(?:\s+\w+){1,4}\b)(?:\s*\1){3,}'
    if re.search(consecutive_loop_pattern, text_lower):
        return 0, "FAILED (Perseveration / Attractor Loop)", 0.0

    # 2. Normalize text: Unwrap LaTeX \boxed{}, remove markups
    normalized = re.sub(r'\\boxed\{([^}]+)\}', r'\1', raw_output)
    normalized = re.sub(r'[\\()\[\]*_`#]', ' ', normalized).lower()

    # 3. Check each anchor group
    total_anchors = len(ground_truth_anchors)
    matched_anchors = 0

    for anchor_group in ground_truth_anchors:
        group_matched = False
        for synonym in anchor_group:
            syn_clean = synonym.lower().strip()
            pattern = r'\b' + re.escape(syn_clean) + r'\b'
            if re.search(pattern, normalized):
                group_matched = True
                break

        if group_matched:
            matched_anchors += 1

    # 4. Proportional Partial Point Scoring
    match_ratio = matched_anchors / total_anchors
    earned_points = int(round(max_points * match_ratio))

    if matched_anchors == total_anchors:
        status = "PASSED (Full Logical Match)"
    elif matched_anchors > 0:
        status = f"PARTIAL ({matched_anchors}/{total_anchors} anchors matched)"
    else:
        status = "FAILED (Logical Inaccuracy / Confabulation)"

    return earned_points, status, round(match_ratio * 100, 1)
