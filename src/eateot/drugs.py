"""Psychoactive drug profiles: named perturbation recipes for the engine.

The catalog lives in versioned YAML (``config/drugs.yaml`` by default,
override with the ``EATEOT_DRUGS_FILE`` env var) and is loaded into
``DRUG_PROFILES`` — mirroring how questionnaires load from
``config/questionnaires/``. Each drug maps a *dose* to a flat dict of engine
primitives via ``resolve_drug``, so the engine/runner never need to know
pharmacology — they just read the resolved primitives.

Primitive contract (what the engine is expected to consume)::

    # Weight-space primitives (feed apply_degradation / new engine knobs)
    noise               gaussian weight noise std (negative = suppression);
                        ADDITIVE to the track baseline — the engine must
                        *add* it, never replace track noise with it
    scale               weight magnitude multiplier; 1.0 = untouched, and the
                        engine MUST treat 1.0 as a no-op (never as an override
                        that would erase track degradation — e.g. C1's scale
                        0.90 must stay 0.90 when a drug leaves scale at 1.0)
    flicker_rate        probability a targeted layer drops to zero
    sirens_mult         multiplier applied to noise for the "sirens" effect
    noise_gradient      extra noise std added per layer of depth
    context_mask_frac   fraction of the prompt tail dropped from context
    # Sampling primitives (feed model.generate kwargs)
    temperature         sampling temperature (absolute, default 0.7)
    repetition_penalty  <1 encourages loops, >1 discourages (default 1.0)
    # Logit-space primitives (new engine knobs)
    attention_scatter   gaussian noise on post-softmax attention weights
    logit_noise         gaussian noise on final logits before sampling
    # Length-pressure primitive
    verbosity_bias      length pressure delta (negative = terse)

Note on design limits: ``scale_loss`` is clamped at 0, so weight
*enhancement* is intentionally NOT expressible through the catalog — "repair"
drugs (nzt) use the separate ``restore_fraction`` mechanism instead.
``dose_cap`` is advisory; ``resolve_drug`` still resolves any dose but flags
``dose_exceeds_cap`` so sweepers/CLIs can warn.

Dose curves: every primitive is a *per-unit-dose* value. The curve shape
scales it (see ``curve_factor``): ``linear``, ``gentle`` (sqrt, fast onset),
``steep`` (square, delayed hit), or ``breakthrough`` (all-or-nothing above
a threshold). ``temperature`` and ``repetition_penalty`` are interpreted as
deltas from engine defaults (0.7 / 1.0) so low doses can never invert them.

Combos / stacks: multiple drugs can be combined at runtime with
``resolve_stack`` (e.g. ``lsd@1.0,thc@0.5``). Additive primitives sum,
weight loss combines into a single ``scale``, temperature/repetition-penalty
deltas sum, the layer window becomes the union, the subnetwork is the shared
value (or ``all`` when the stack mixes targets), and ``restore_fraction``
is the max across components. ``resolve_stack`` returns the same spec shape
as ``resolve_drug`` (plus a ``components`` list), so the engine and runner
consume stacks exactly like single drugs.
"""

import os
from pathlib import Path

import yaml

from .paths import PROJECT_ROOT

# Catalog location. Override per-run: EATEOT_DRUGS_FILE=/path/to/drugs.yaml
DEFAULT_DRUGS_FILE = PROJECT_ROOT / "config" / "drugs.yaml"
DRUGS_FILE = Path(os.environ.get("EATEOT_DRUGS_FILE", str(DEFAULT_DRUGS_FILE)))

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
DRUG_CLASSES = (
    "hallucinogen",
    "dissociative",
    "stimulant",
    "depressant",
    "cannabinoid",
    "deliriant",
    "enhancer",
    "placebo",
)
CURVES = ("linear", "gentle", "steep", "breakthrough")
SUBNETWORKS = ("all", "attn", "mlp", "norm")

# Additive primitives: raw value * dose factor (may be negative where it
# means something — noise suppression, terse verbosity).
ADDITIVE_PRIMITIVES = frozenset({
    "noise",            # signed: negative suppresses below the track baseline
    "scale_loss",       # fraction of weight magnitude lost (clamped >= 0)
    "attention_scatter",  # clamped >= 0
    "logit_noise",        # clamped >= 0
    "flicker_rate",       # clamped >= 0
    "sirens_mult",        # clamped >= 0
    "noise_gradient",     # clamped >= 0
    "context_mask_frac",  # clamped >= 0
    "verbosity_bias",     # signed: negative = terse
})
# Primitives that are deltas from an engine default (absolute after resolve).
DEFAULT_RELATIVE = {
    "temperature": 0.7,
    "repetition_penalty": 1.0,
}
# All primitives a drug may specify in its dose_curve.
ALL_PRIMITIVES = ADDITIVE_PRIMITIVES | frozenset(DEFAULT_RELATIVE)

# Result keys always present in resolve_drug()["primitives"].
RESOLVED_PRIMITIVES = frozenset(
    ADDITIVE_PRIMITIVES - {"scale_loss"}  # scale_loss is folded into `scale`
) | frozenset({"scale"}) | frozenset(DEFAULT_RELATIVE)

# Safety floors: absolute primitives must stay in sane ranges even at
# extreme doses.
TEMPERATURE_FLOOR = 0.05
REPETITION_FLOOR = 0.5


# ---------------------------------------------------------------------------
# Loading & validation
# ---------------------------------------------------------------------------
def _load_yaml() -> dict:
    """Load the drug catalog YAML, raising a clear error if missing."""
    if not DRUGS_FILE.exists():
        raise FileNotFoundError(
            f"Drug catalog '{DRUGS_FILE}' not found. "
            "Override with the EATEOT_DRUGS_FILE env var."
        )
    with open(DRUGS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_drug_profiles() -> dict[str, dict]:
    """Load and validate the drug catalog into ``{name: profile}``."""
    data = _load_yaml()
    if not isinstance(data, dict):
        raise ValueError(f"Drug catalog '{DRUGS_FILE}' is not a YAML mapping.")
    profiles = data.get("drugs") or {}
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"Drug catalog '{DRUGS_FILE}' contains no 'drugs' mapping.")
    validate_drug_profiles(profiles)
    return profiles


def list_drugs(profiles: dict[str, dict] | None = None) -> list[str]:
    """Sorted drug names (defaults to the loaded catalog)."""
    if profiles is None:
        profiles = DRUG_PROFILES
    return sorted(profiles)


def validate_drug_profiles(profiles: dict[str, dict]) -> None:
    """Schema-check every drug profile; raise ``ValueError`` with all errors.

    Kept separate from ``load_drug_profiles`` so tests can validate arbitrary
    catalogs (including the shipped one) without import-time surprises.
    """
    errors: list[str] = []
    for name, prof in profiles.items():
        prefix = f"drug '{name}'"
        if not isinstance(prof, dict):
            errors.append(f"{prefix}: profile must be a mapping")
            continue
        if prof.get("class") not in DRUG_CLASSES:
            errors.append(f"{prefix}: class must be one of {DRUG_CLASSES}, "
                          f"got {prof.get('class')!r}")
        if prof.get("curve") not in CURVES:
            errors.append(f"{prefix}: curve must be one of {CURVES}, "
                          f"got {prof.get('curve')!r}")
        if prof.get("subnetwork") not in SUBNETWORKS:
            errors.append(f"{prefix}: subnetwork must be one of {SUBNETWORKS}, "
                          f"got {prof.get('subnetwork')!r}")

        layer_pct = prof.get("layer_pct")
        if (not isinstance(layer_pct, (list, tuple)) or len(layer_pct) != 2
                or not all(isinstance(x, (int, float)) and 0.0 <= x <= 1.0
                           for x in layer_pct) or layer_pct[0] >= layer_pct[1]):
            errors.append(f"{prefix}: layer_pct must be [start, end] with "
                          f"0 <= start < end <= 1, got {layer_pct!r}")

        if not isinstance(prof.get("potency", 1.0), (int, float)) or prof.get("potency", 1.0) <= 0:
            errors.append(f"{prefix}: potency must be > 0")
        if not isinstance(prof.get("dose_cap", 1.0), (int, float)) or prof.get("dose_cap", 1.0) <= 0:
            errors.append(f"{prefix}: dose_cap must be > 0")

        if prof.get("curve") == "breakthrough":
            bta = prof.get("breakthrough_at")
            if not isinstance(bta, (int, float)) or not 0.0 < bta < 1.0:
                errors.append(f"{prefix}: breakthrough curve requires "
                              f"breakthrough_at in (0, 1), got {bta!r}")

        restore = prof.get("restore_fraction")
        if restore is not None and (
                not isinstance(restore, (int, float)) or not 0.0 <= restore <= 1.0):
            errors.append(f"{prefix}: restore_fraction must be in [0, 1], "
                          f"got {restore!r}")

        if not isinstance(prof.get("prompt_state", ""), str):
            errors.append(f"{prefix}: prompt_state must be a string")

        dose_curve = prof.get("dose_curve") or {}
        if not isinstance(dose_curve, dict):
            errors.append(f"{prefix}: dose_curve must be a mapping")
            continue
        for primitive, value in dose_curve.items():
            if primitive not in ALL_PRIMITIVES:
                errors.append(f"{prefix}: unknown primitive {primitive!r} "
                              f"(known: {', '.join(sorted(ALL_PRIMITIVES))})")
            if not isinstance(value, (int, float)):
                errors.append(f"{prefix}: dose_curve[{primitive}] must be numeric, "
                              f"got {value!r}")

    if errors:
        raise ValueError("Invalid drug catalog:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# Dose curves
# ---------------------------------------------------------------------------
def curve_factor(curve: str, dose: float, potency: float = 1.0,
                 breakthrough_at: float | None = None) -> float:
    """Map a dose to the multiplicative factor applied to every primitive.

    All curves are monotonic non-decreasing in dose (the core testable
    property that guarantees "more drug = stronger effect"):

    * ``linear``      — factor = dose * potency
    * ``gentle``      — factor = sqrt(dose) * potency (fast onset, saturates)
    * ``steep``       — factor = dose**2 * potency (delayed, then explodes)
    * ``breakthrough``— factor = potency * (dose - bta) / (1 - bta), clamped
                        to 0 below ``breakthrough_at``; reaches potency at
                        dose 1.0 (all-or-nothing drugs like DMT/salvia)
    """
    if dose < 0:
        raise ValueError(f"dose must be >= 0, got {dose}")
    if curve == "linear":
        return dose * potency
    if curve == "gentle":
        return (dose ** 0.5) * potency
    if curve == "steep":
        return (dose ** 2) * potency
    if curve == "breakthrough":
        if breakthrough_at is None:
            raise ValueError("breakthrough curve requires breakthrough_at")
        return potency * max(0.0, (dose - breakthrough_at) / (1.0 - breakthrough_at))
    raise ValueError(f"unknown curve {curve!r} (expected one of {CURVES})")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------
def resolve_drug(name: str, dose: float = 1.0,
                 profiles: dict[str, dict] | None = None) -> dict:
    """Resolve a drug at a dose into a flat, engine-ready spec.

    Returns a dict with the profile metadata plus a ``primitives`` mapping
    that always contains every key in ``RESOLVED_PRIMITIVES`` (consumers can
    read fields without ``get``). Semantics per primitive:

    * Additive primitives scale with the dose factor; non-negative-by-nature
      ones clamp at 0 (except ``noise``, which may be negative to *suppress*
      the track baseline, and ``verbosity_bias``, which may be negative for
      terse drugs).
    * ``scale`` folds ``scale_loss`` into a multiplier: 1.0 - loss.
    * ``temperature`` / ``repetition_penalty`` are deltas from engine
      defaults (0.7 / 1.0), floored for safety.
    """
    if profiles is None:
        profiles = DRUG_PROFILES
    try:
        prof = profiles[name]
    except KeyError:
        raise KeyError(
            f"Unknown drug '{name}'. Available: {', '.join(list_drugs(profiles))}"
        ) from None

    factor = curve_factor(
        prof.get("curve", "linear"), dose,
        potency=prof.get("potency", 1.0),
        breakthrough_at=prof.get("breakthrough_at"),
    )
    dose_curve = prof.get("dose_curve") or {}

    # Consistent with validate_drug_profiles: subnetwork is required.
    subnetwork = prof.get("subnetwork")
    if subnetwork not in SUBNETWORKS:
        raise ValueError(
            f"drug '{name}': subnetwork must be one of {SUBNETWORKS}, "
            f"got {subnetwork!r}"
        )

    primitives: dict[str, float] = {}
    for primitive in sorted(ADDITIVE_PRIMITIVES):
        raw = dose_curve.get(primitive, 0.0)
        value = raw * factor
        if primitive != "noise" and primitive != "verbosity_bias":
            value = max(0.0, value)  # clamp nonsensical negatives
        primitives[primitive] = value

    primitives["scale"] = max(0.0, 1.0 - primitives.pop("scale_loss"))

    for primitive, default in DEFAULT_RELATIVE.items():
        raw = dose_curve.get(primitive, 0.0)
        value = default + raw * factor
        if primitive == "temperature":
            value = max(TEMPERATURE_FLOOR, value)
        else:  # repetition_penalty
            value = max(REPETITION_FLOOR, value)
        primitives[primitive] = value

    # Internal invariant: consumers may rely on every resolved key existing.
    if set(primitives) != RESOLVED_PRIMITIVES:
        raise RuntimeError(
            f"resolve_drug produced unexpected primitive keys for '{name}': "
            f"{sorted(set(primitives) ^ RESOLVED_PRIMITIVES)}"
        )

    return {
        "name": name,
        "class": prof.get("class"),
        "target_domain": prof.get("target_domain"),
        "description": prof.get("description", ""),
        "subnetwork": subnetwork,
        "layer_pct": list(prof.get("layer_pct", [0.0, 1.0])),
        "curve": prof.get("curve", "linear"),
        "potency": prof.get("potency", 1.0),
        "dose": dose,
        "factor": factor,
        "dose_cap": prof.get("dose_cap", 1.0),
        "dose_exceeds_cap": dose > prof.get("dose_cap", 1.0),
        "restore_fraction": prof.get("restore_fraction"),
        "prompt_state": prof.get("prompt_state", ""),
        "primitives": primitives,
    }


# ---------------------------------------------------------------------------
# Drug stacks (combos)
# ---------------------------------------------------------------------------
def parse_stack(spec: str) -> list[dict]:
    """Parse a compact stack spec: ``"lsd@1.0,thc@0.5"`` -> list of components.

    Each component is ``name@dose`` (a bare name means dose 1.0). Both ``,``
    and ``+`` are accepted as separators, so a ``stack_label`` (e.g.
    ``"lsd@1+thc@0.5"``) fed back in round-trips. Raises ``ValueError`` for
    empty or malformed specs. Used by CLIs (e.g. ``eateot-lab --stack
    lsd@1.0,thc@0.5``) and kept pure for testing.
    """
    parts = [p.strip() for p in spec.replace("+", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("stack spec is empty (expected 'drug@dose,drug@dose')")
    components = []
    for part in parts:
        if "@" in part:
            name, _, dose_s = part.partition("@")
            try:
                dose = float(dose_s)
            except ValueError:
                raise ValueError(
                    f"malformed dose in {part!r} (expected drug@dose)"
                ) from None
        else:
            name, dose = part, 1.0
        if not name.strip():
            raise ValueError(f"malformed stack component {part!r}")
        components.append({"drug": name.strip(), "dose": dose})
    return components


def stack_label(components: list[dict]) -> str:
    """Compact human-readable label for a stack (the reverse of ``parse_stack``).

    ``[{'drug': 'lsd', 'dose': 1.0}, {'drug': 'thc', 'dose': 0.5}]`` ->
    ``"lsd@1+thc@0.5"``. Recorded in telemetry as the run's ``drug`` name.
    Doses are rendered with ``:g`` (display-grade, ~6 significant digits) —
    the label is for humans and telemetry, not a precision-preserving
    serialization.
    """
    return "+".join(f"{c['drug']}@{c['dose']:g}" for c in components)


def validate_stack(components: list[dict], profiles: dict[str, dict] | None = None) -> None:
    """Schema-check a stack component list; raise ``ValueError`` with all errors.

    Every component must be a ``{drug, dose}`` mapping with a known drug name
    and a non-negative numeric dose. Kept separate from ``resolve_stack`` so
    tests (and CLIs) can validate before resolving.
    """
    if profiles is None:
        profiles = DRUG_PROFILES
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("stack must be a non-empty list of {drug, dose} components")
    errors = []
    for i, comp in enumerate(components):
        prefix = f"component {i}"
        if not isinstance(comp, dict) or not comp.get("drug"):
            errors.append(f"{prefix}: must be a mapping with a 'drug' key")
            continue
        name = comp["drug"]
        if name not in profiles:
            errors.append(
                f"{prefix}: unknown drug {name!r} "
                f"(available: {', '.join(list_drugs(profiles))})"
            )
        dose = comp.get("dose", 1.0)
        if not isinstance(dose, (int, float)) or dose < 0:
            errors.append(f"{prefix}: dose must be a number >= 0, got {dose!r}")
    if errors:
        raise ValueError("Invalid drug stack:\n  - " + "\n  - ".join(errors))


def resolve_stack(components: list[dict],
                  profiles: dict[str, dict] | None = None) -> dict:
    """Resolve multiple drugs into ONE merged, engine-ready spec (a combo).

    ``components`` is a list of ``{"drug": name, "dose": dose}`` dicts (see
    ``parse_stack`` for the compact string form). Each component resolves
    exactly like a single drug (``resolve_drug``), then the primitives merge:

    * Additive primitives (noise, attention_scatter, logit_noise,
      flicker_rate, sirens_mult, noise_gradient, context_mask_frac,
      verbosity_bias) SUM; noise and verbosity_bias stay signed, the rest
      clamp at 0.
    * Weight loss combines: ``scale = 1 - sum(1 - scale_i)``, clamped at 0.
    * ``temperature`` / ``repetition_penalty`` deltas from the engine
      defaults SUM (e.g. lsd +0.15 and thc +0.10 -> +0.25).
    * ``subnetwork`` is the single shared value across components, else
      ``"all"`` (mixed-target stacks degrade everything).
    * ``layer_pct`` is the union window ``[min(starts), max(ends)]``.
    * ``prompt_state`` concatenates the components' trip states.
    * ``restore_fraction`` is the max across components (a stack containing
      an enhancer like nzt restores that much of the clean state).

    A single-component stack returns that drug's ``resolve_drug`` result
    unchanged, so callers can treat stacks and drugs uniformly. The merged
    spec carries the same keys as ``resolve_drug`` plus a ``components`` list
    (``dose`` / ``factor`` are None for stacks — doses are per-component and
    already applied).
    """
    if profiles is None:
        profiles = DRUG_PROFILES
    if not isinstance(components, (list, tuple)) or not components:
        raise ValueError("stack must be a non-empty list of {drug, dose} components")
    validate_stack(components, profiles=profiles)

    resolved = [resolve_drug(c["drug"], c.get("dose", 1.0), profiles=profiles)
                for c in components]
    if len(resolved) == 1:
        return resolved[0]  # a "stack" of one drug is just that drug

    primitives: dict[str, float] = {}
    for primitive in sorted(ADDITIVE_PRIMITIVES - {"scale_loss"}):
        total = sum(r["primitives"][primitive] for r in resolved)
        if primitive != "noise" and primitive != "verbosity_bias":
            total = max(0.0, total)  # clamp nonsensical negatives
        primitives[primitive] = total
    total_loss = sum(1.0 - r["primitives"]["scale"] for r in resolved)
    primitives["scale"] = max(0.0, 1.0 - total_loss)
    for primitive, default in DEFAULT_RELATIVE.items():
        delta = sum(r["primitives"][primitive] - default for r in resolved)
        value = default + delta
        if primitive == "temperature":
            value = max(TEMPERATURE_FLOOR, value)
        else:  # repetition_penalty
            value = max(REPETITION_FLOOR, value)
        primitives[primitive] = value

    if set(primitives) != RESOLVED_PRIMITIVES:
        raise RuntimeError(
            f"resolve_stack produced unexpected primitive keys: "
            f"{sorted(set(primitives) ^ RESOLVED_PRIMITIVES)}"
        )

    subnetworks = {r["subnetwork"] for r in resolved}
    subnetwork = subnetworks.pop() if len(subnetworks) == 1 else "all"
    layer_pct = [min(r["layer_pct"][0] for r in resolved),
                 max(r["layer_pct"][1] for r in resolved)]
    prompt_state = "\n".join(r["prompt_state"] for r in resolved
                              if r.get("prompt_state"))
    restores = [r["restore_fraction"] for r in resolved
                if r.get("restore_fraction") is not None]

    return {
        "name": stack_label(components),
        "class": "stack",
        "target_domain": "stack",
        "description": " + ".join(r["name"] for r in resolved),
        "subnetwork": subnetwork,
        "layer_pct": layer_pct,
        "curve": "linear",  # doses already applied per component
        "potency": 1.0,
        "dose": None,
        "factor": None,
        "dose_cap": max(r["dose_cap"] for r in resolved),
        "dose_exceeds_cap": False,
        "restore_fraction": max(restores) if restores else None,
        "prompt_state": prompt_state,
        "primitives": primitives,
        "components": [{"name": r["name"], "dose": r["dose"]} for r in resolved],
    }


# Loaded once at import (mirrors IQ_TEST_BATTERY / PRESET_PROMPTS).
DRUG_PROFILES = load_drug_profiles()
