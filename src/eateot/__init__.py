"""EATEOT Neural Degradation Lab — core package.

Public API (also importable as ``from eateot import ...``):

- ``BrainLabEngine``  — model loading + degradation engine (incl. ``run_progressive_inference`` — in-generation hidden-state decay)
- ``run_iq_test``     — 5-tier IQ battery orchestrator
- ``evaluate_response`` / ``IQ_TEST_BATTERY`` — pure scoring logic + battery spec
- ``EATEOT_TRACK_PROFILES`` / ``PRESET_PROMPTS`` — degradation profiles + prompt presets
- ``DRUG_PROFILES`` / ``resolve_drug`` / ``resolve_stack`` — psychoactive perturbation catalog, dose resolution + combos
- ``log_test_run``    — telemetry persistence (auto-routes drug runs to ``DRUG_LOG_FILE``)
- ``LOG_FILE``        — Alzheimer-domain telemetry path
- ``DRUG_LOG_FILE``   — drug-domain telemetry path (see ``eateot.paths``)
"""

from .battery import (
    IQ_TEST_BATTERY,
    evaluate_fluency,
    evaluate_question,
    evaluate_response,
    grade_deterioration,
)
from .config import (
    BASE_IQ,
    DEFAULT_DECAY,
    DEFAULT_FLICKER,
    DEFAULT_MODEL,
    DEFAULT_SIRENS,
    DEFAULT_SUBNET,
    DEFAULT_SURGE,
    clinical_diagnosis,
)
from .drugs import (
    DRUG_PROFILES,
    curve_factor,
    list_drugs,
    load_drug_profiles,
    parse_stack,
    resolve_drug,
    resolve_stack,
    stack_label,
    validate_drug_profiles,
    validate_stack,
)
from .engine import BrainLabEngine
from .paths import DRUG_LOG_FILE, LOG_FILE
from .profiles import EATEOT_TRACK_PROFILES, PRESET_PROMPTS
from .questionnaires import DEFAULT_BATTERY, list_batteries, list_questionnaires, load_battery
from .runner import run_iq_test
from .telemetry import log_test_run

__version__ = "0.1.0"

__all__ = [
    "BASE_IQ",
    "BrainLabEngine",
    "DEFAULT_DECAY",
    "DEFAULT_FLICKER",
    "DEFAULT_MODEL",
    "DEFAULT_SIRENS",
    "DEFAULT_SUBNET",
    "DEFAULT_SURGE",
    "DEFAULT_BATTERY",
    "DRUG_LOG_FILE",
    "DRUG_PROFILES",
    "EATEOT_TRACK_PROFILES",
    "IQ_TEST_BATTERY",
    "LOG_FILE",
    "PRESET_PROMPTS",
    "clinical_diagnosis",
    "curve_factor",
    "evaluate_fluency",
    "evaluate_question",
    "evaluate_response",
    "grade_deterioration",
    "list_batteries",
    "list_drugs",
    "list_questionnaires",
    "load_battery",
    "load_drug_profiles",
    "log_test_run",
    "parse_stack",
    "resolve_drug",
    "resolve_stack",
    "run_iq_test",
    "stack_label",
    "validate_drug_profiles",
    "validate_stack",
]
