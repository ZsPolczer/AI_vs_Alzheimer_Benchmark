"""EATEOT Neural Degradation Lab — core package.

Public API (also importable as ``from eateot import ...``):

- ``BrainLabEngine``  — model loading + degradation engine
- ``run_iq_test``     — 5-tier IQ battery orchestrator
- ``evaluate_response`` / ``IQ_TEST_BATTERY`` — pure scoring logic + battery spec
- ``EATEOT_TRACK_PROFILES`` / ``PRESET_PROMPTS`` — degradation profiles + prompt presets
- ``log_test_run``    — telemetry persistence
- ``LOG_FILE``        — resolved telemetry path (see ``eateot.paths``)
"""

from .battery import IQ_TEST_BATTERY, evaluate_response
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
from .engine import BrainLabEngine
from .paths import LOG_FILE
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
    "EATEOT_TRACK_PROFILES",
    "IQ_TEST_BATTERY",
    "LOG_FILE",
    "PRESET_PROMPTS",
    "clinical_diagnosis",
    "evaluate_response",
    "list_batteries",
    "list_questionnaires",
    "load_battery",
    "log_test_run",
    "run_iq_test",
]
