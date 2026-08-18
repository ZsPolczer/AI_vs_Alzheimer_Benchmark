"""Questionnaire loading from versioned YAML config files.

The default questionnaire directory is ``<repo>/config/questionnaires``.
Override it per-run with the ``EATEOT_QUESTIONNAIRE_DIR`` env var, e.g.::

    EATEOT_QUESTIONNAIRE_DIR=/path/to/my/quiz eateot-lab --questionnaire my_quiz

A questionnaire file is a YAML document with a ``name``, ``version``, and a
list of ``questions``. See ``config/questionnaires/iq_battery.yaml`` for the
full schema.
"""

import os
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT

# Default questionnaire directory (versioned, committed to the repo).
DEFAULT_QUESTIONNAIRE_DIR = PROJECT_ROOT / "config" / "questionnaires"
QUESTIONNAIRE_DIR = Path(
    os.environ.get("EATEOT_QUESTIONNAIRE_DIR", str(DEFAULT_QUESTIONNAIRE_DIR))
)

# The default battery name used when no --questionnaire flag is given.
DEFAULT_BATTERY = "iq_battery"


def _load_yaml(name: str) -> dict:
    """Load and parse a questionnaire YAML file by name (without extension)."""
    path = QUESTIONNAIRE_DIR / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(list_questionnaires()) or "(none found)"
        raise FileNotFoundError(
            f"Questionnaire '{name}' not found in {QUESTIONNAIRE_DIR}. "
            f"Available: {available}"
        )
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_questionnaires() -> list[str]:
    """Return the names of all questionnaires in the questionnaire directory."""
    if not QUESTIONNAIRE_DIR.is_dir():
        return []
    return sorted(p.stem for p in QUESTIONNAIRE_DIR.glob("*.yaml"))


def _is_scoreable_question(question: dict) -> bool:
    """True if the runner can score this question (anchored or fluency)."""
    if "ground_truth_anchors" in question:
        return True
    return question.get("type") == "fluency" and isinstance(question.get("fluency"), dict)


def list_batteries() -> list[str]:
    """Return questionnaire files that are actually usable as IQ batteries.

    ``presets.yaml`` and ``brain_benchmark.yaml`` carry a different schema
    (no ``ground_truth_anchors`` and no fluency items), so they are excluded —
    only files whose questions are all scoreable (anchored or fluency) are
    advertised via ``--questionnaire``.
    """
    batteries = []
    for name in list_questionnaires():
        data = _load_yaml(name)
        questions = data.get("questions") or []
        if questions and all(_is_scoreable_question(q) for q in questions):
            batteries.append(name)
    return batteries


def load_battery(name: str = DEFAULT_BATTERY) -> list[dict]:
    """Load an IQ battery questionnaire as a list of question dicts.

    Each question dict has the keys used by ``eateot.battery.evaluate_response``:
    ``tier``, ``domain``, ``target_iq``, ``question``, ``ground_truth_anchors``,
    and ``max_points``.

    Anchor synonyms are coerced to strings — YAML 1.1 parses unquoted ``yes`` /
    ``no`` as booleans and numeric anchors (e.g. ``95``) as ints, which would
    break the string-based matcher. Question types that don't use anchors
    (e.g. ``type: fluency``) are left untouched.
    """
    data = _load_yaml(name)
    questions = data.get("questions", [])
    if not questions:
        raise ValueError(f"Questionnaire '{name}' contains no questions.")
    for question in questions:
        if "ground_truth_anchors" in question:
            question["ground_truth_anchors"] = [
                [str(synonym) for synonym in group]
                for group in question["ground_truth_anchors"]
            ]
    return questions


def load_presets(name: str = "presets") -> dict[str, tuple[str, str]]:
    """Load the quick prompt presets in the ``{id: (title, prompt)}`` shape."""
    data = _load_yaml(name)
    presets = {}
    for item in data.get("prompts", []):
        presets[str(item["id"])] = (item["title"], item["prompt"])
    return presets


def load_simple_questions(name: str = "brain_benchmark") -> list[dict]:
    """Load a flat question set (e.g. ``{question, expected}`` for scripts)."""
    data = _load_yaml(name)
    return data.get("questions", [])
