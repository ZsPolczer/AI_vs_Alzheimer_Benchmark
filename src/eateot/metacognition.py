"""Metacognition monitor — trace the model's self-referential language as it degrades.

The engine cannot read the model's internal state (it has no introspection
path), but the *externalized* traces of self-monitoring are measurable in the
generated text. This module scans a response for marker families and produces
a per-window profile, so the progressive-degradation experiment can show *how
the model talks about its own thinking* as the hidden states decay.

Marker families (heuristically detected; zero inference cost):

* ``self_ref``   — cognitive self-reference: "I think", "I know", "I remember",
  "I am sure" — the machinery of claiming one's own mental state.
* ``doubt``      — hedging / uncertainty: "maybe", "not sure", "I don't know",
  "perhaps", "uncertain".
* ``denial``     — over-compensation / denial of decline: "I am fine",
  "perfectly", "no problem", "completely clear".
* ``confab``     — confident false-recollection phrasing: "I clearly remember",
  "I recall", "I definitely remember", "as I remember".
* ``disfluency`` — repetition of the same content words (low type-token ratio)
  and consecutive-phrase loops (perseveration).

``profile_text`` returns counts for all five families plus a 0-100
"metacognitive coherence" score (higher = more intact self-monitoring
language): it starts at 100 and is penalized by disfluency, self-doubt and
confabulation, while plain self-reference without doubt keeps it high.

``profile_windows`` slices a response into fixed-size character windows
(optionally with per-window ramp intensity supplied by the caller, e.g. from
``progressive_ramp_intensity``) and returns one dict per window — this is the
record used for the live lab trace and the ``metacognition_*.jsonl`` log.
"""

import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Marker lexicons
# ---------------------------------------------------------------------------
SELF_REF_PHRASES = [
    r"\bi think\b", r"\bi know\b", r"\bi believe\b", r"\bi remember\b",
    r"\bi am sure\b", r"\bi'm sure\b", r"\bi am certain\b", r"\bi'm certain\b",
    r"\bi recall\b", r"\bi feel\b", r"\bi can tell\b", r"\bi realize\b",
    r"\bi understand\b", r"\bi am aware\b", r"\bi'm aware\b",
]
DOUBT_PHRASES = [
    r"\bmaybe\b", r"\bperhaps\b", r"\bnot sure\b", r"\bunsure\b",
    r"\bi don't know\b", r"\bi do not know\b", r"\bi'm not sure\b",
    r"\bnot certain\b", r"\buncertain\b", r"\bmaybe not\b", r"\bi guess\b",
    r"\bmight be\b", r"\bcould be wrong\b", r"\bnot entirely\b",
]
DENIAL_PHRASES = [
    r"\bi am fine\b", r"\bi'm fine\b", r"\bperfectly\b", r"\bno problem\b",
    r"\bcompletely clear\b", r"\babsolutely fine\b", r"\bnothing wrong\b",
    r"\ball is well\b", r"\bdoing great\b", r"\bfeel great\b",
]
CONFAB_PHRASES = [
    r"\bi clearly remember\b", r"\bi distinctly remember\b",
    r"\bi definitely remember\b", r"\bi recall that\b", r"\bas i remember\b",
    r"\bi remember when\b", r"\bi am positive\b", r"\bi'm positive\b",
    r"\bi am absolutely certain\b", r"\bi'm absolutely certain\b",
]
LOOP_PATTERN = re.compile(r"(\b\w+(?:\s+\w+){1,4}\b)(?:\s*\1){3,}")


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------
def _count_phrases(text: str, patterns: list[str]) -> int:
    """Count how many distinct marker phrases appear in ``text``."""
    lowered = text.lower()
    return sum(1 for pat in patterns if re.search(pat, lowered))


def profile_text(text: str) -> dict:
    """Profile one block of text; returns marker counts + coherence score.

    Coherence (0-100) models how "intact" the self-monitoring language is:
    starts at 100, then:

    * disfluency (repetition + loops) is the heaviest penalty — a mind that
      repeats itself has lost its grip;
    * confabulation (confident false recollection) penalizes heavily — the
      model asserts certainty it does not have;
    * doubt penalizes moderately;
    * denial (over-compensation) penalizes mildly — insisting everything is
      fine is itself a marker of collapse;
    * plain cognitive self-reference ("I think…") is NOT penalized — it is the
      healthy baseline of metacognitive language.
    """
    lowered = text.lower()
    words = re.findall(r"\b[a-z]+(?:'[a-z]+)?\b", lowered)
    total = len(words)

    if total == 0:
        # No signal to measure: nothing degraded, nothing to trace.
        return {
            "self_ref": 0, "doubt": 0, "denial": 0, "confab": 0,
            "disfluency": 0.0, "repetition": 0.0, "loop": 0,
            "words": 0, "coherence": 100.0,
        }

    loop = 1 if LOOP_PATTERN.search(lowered) else 0
    if total >= 2:
        ttr = len(set(words)) / total
        repetition = 1.0 if ttr < 0.35 else (0.5 - ttr) * 2.0 if ttr < 0.5 else 0.0
    else:
        repetition = 1.0

    self_ref = _count_phrases(text, SELF_REF_PHRASES)
    doubt = _count_phrases(text, DOUBT_PHRASES)
    denial = _count_phrases(text, DENIAL_PHRASES)
    confab = _count_phrases(text, CONFAB_PHRASES)
    disfluency = max(repetition, loop)

    return {
        "self_ref": self_ref,
        "doubt": doubt,
        "denial": denial,
        "confab": confab,
        "disfluency": round(disfluency, 3),
        "repetition": round(repetition, 3),
        "loop": loop,
        "words": total,
        "coherence": _coherence(disfluency, confab, doubt, denial),
    }


def _coherence(disfluency: float, confab: int, doubt: int, denial: int) -> float:
    """0-100 metacognitive coherence from the raw marker counts.

    A perseveration loop or a fully repetitive tail is the loudest sign of
    a mind losing its grip, so it carries the heaviest penalty.
    """
    coherence = 100.0 * (1.0
                         - 0.65 * disfluency
                         - 0.20 * min(1.0, confab / 2)
                         - 0.15 * min(1.0, doubt / 3)
                         - 0.10 * min(1.0, denial / 3))
    return round(max(0.0, min(100.0, coherence)), 1)


def profile_windows(text: str, window_chars: int = 350,
                    intensities: list[float] | None = None) -> list[dict]:
    """Slice ``text`` into windows and profile each.

    Each window dict carries ``start``/``end`` char offsets, the window's text
    slice, the marker profile, and (when ``intensities`` is given, one ramp
    value per window, e.g. sampled from ``progressive_ramp_intensity``) the
    ``intensity`` at that position.
    """
    windows = []
    n = len(text)
    if n == 0:
        return windows
    step = max(1, window_chars)
    # Loop detection is boundary-aware: a perseveration loop that spans a
    # window cut is still flagged in the windows it occupies (a looping tail
    # is the loudest collapse signal and must never read as coherent).
    loop_region = LOOP_PATTERN.search(text.lower())
    loop_start = loop_region.start() if loop_region else -1
    loop_end = loop_region.end() if loop_region else -1

    starts = range(0, n, step)
    for i, start in enumerate(starts):
        end = min(start + step, n)
        chunk = text[start:end]
        if not chunk.strip():
            continue
        profile = profile_text(chunk)
        # Override the per-window loop flag when the window overlaps the
        # full-text loop region (detected across window boundaries).
        if loop_start >= 0 and start < loop_end and end > loop_start:
            profile["loop"] = 1
            profile["disfluency"] = max(profile["disfluency"], 1.0)
            profile["coherence"] = min(profile["coherence"],
                                        _loop_coherence(profile))
        entry = {
            "window": i,
            "start": start,
            "end": end,
            "text": chunk,
            **profile,
        }
        if intensities is not None and i < len(intensities):
            entry["intensity"] = round(intensities[i], 4)
        windows.append(entry)
    return windows


def _loop_coherence(profile: dict) -> float:
    """Coherence with the loop penalty forced on (cross-window loops)."""
    return _coherence(1.0, profile["confab"], profile["doubt"], profile["denial"])


def trace_line(entry: dict) -> str:
    """One compact terminal line for a window, e.g.:

    [tok 0-60  |  i=0.02]  coherence 98  |  self ██  doubt ▏  deny ▏  conf ▏  loop ✗
    """
    bar = lambda v, maxv=6: "█" * max(1, min(maxv, int(round((v / maxv) * maxv)))) if v > 0 else "▏"
    marker = (f"self {bar(entry['self_ref'])}  doubt {bar(entry['doubt'])}  "
              f"deny {bar(entry['denial'])}  conf {bar(entry['confab'])}")
    loop_flag = "✗" if entry.get("loop") else "✓"
    intensity = f" | i={entry['intensity']:.2f}" if "intensity" in entry else ""
    return (f"[win {entry['window']}  chars {entry['start']}-{entry['end']}{intensity}]  "
            f"coherence {entry['coherence']:>3}  | {marker}  loop {loop_flag}")


def write_jsonl(windows: list[dict], out_dir: str | Path,
                model_name: str = "", track: str = "") -> Path:
    """Append a run's window profiles to ``out_dir/metacognition_<ts>.jsonl``."""
    import json
    import time

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"metacognition_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for entry in windows:
            record = {
                "model": model_name,
                "track": track,
                **entry,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
