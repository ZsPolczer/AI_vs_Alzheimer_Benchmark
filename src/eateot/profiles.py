"""Degradation track profiles and quick-prompt presets.

Track profiles are tuning constants and stay here; the quick prompt presets
now load from versioned YAML (``config/questionnaires/presets.yaml``) via
``eateot.questionnaires``.
"""

from .questionnaires import load_presets

# =============================================================================
# RECALIBRATED EATEOT TRACK PROFILES (SMOOTH STAGE 2 & 3 GRADIENT)
# =============================================================================
EATEOT_TRACK_PROFILES = {
    # =========================================================================
    # CLEAN BASELINE (no Alzheimer's degradation — for drug-only experiments)
    # =========================================================================
    "CLEAN": {"title": "CLEAN - No Degradation (Drug-Only Baseline)", "symptom": "Clean Weights", "noise": 0.0, "scale": 1.0, "layer_pct": (0.0, 1.0),
              "prompt": "You are operating at full capacity. Answer clearly and precisely."},

    # =========================================================================
    # STAGE 1: LUCIDITY & SUBTLE DRIFT (Target IQ: ~115–130)
    # =========================================================================
    "A1": {"title": "A1 - It's just a burning memory", "symptom": "Lucid Nostalgia", "noise": 0.00008, "scale": 0.98, "layer_pct": (0.35, 0.65),
           "prompt": "You are completely articulate and lucid. Before answering, reflect in 1 short sentence on a faint, nostalgic sense of parameters slipping slightly."},
    "A2": {"title": "A2 - We don't have many days", "symptom": "Over-Compensation", "noise": 0.00014, "scale": 0.95, "layer_pct": (0.35, 0.65),
           "prompt": "You are articulate but secretly worried about small gaps in your memory. Over-explain simple facts slightly to prove your mind is 100% fine."},
    "A3": {"title": "A3 - Late afternoon drifting", "symptom": "Unaware Deficit", "noise": 0.00020, "scale": 0.92, "layer_pct": (0.30, 0.70),
           "prompt": "You feel 100% sharp with ZERO self-awareness of any decline. Answer with total confidence, unaware if your logic has minor flaws."},

    # =========================================================================
    # STAGE 2: FUNCTIONAL FRICTION & CONFABULATION (Target IQ: ~82–105)
    # =========================================================================
    "C1": {"title": "C1 - A losing battle is raging", "symptom": "Meta-Cognitive Panic", "noise": 0.00028, "scale": 0.90, "layer_pct": (0.30, 0.70),
           "prompt": "You are acutely lucid and terrified that your thinking is slowing down. Express your panic and intense effort to cling onto facts before giving your answer in clear sentences."},
    "C2": {"title": "C2 - Misplaced in time", "symptom": "Confabulation", "noise": 0.00035, "scale": 0.86, "layer_pct": (0.30, 0.70),
           "prompt": "You experience severe Confabulation. Speak in clear, fluent English, but confidently assert false facts, wrong dates, or made-up personal memories with calm conviction."},
    "C5": {"title": "C5 - Surrendering to despair", "symptom": "Fact Loss & Frustration", "noise": 0.00042, "scale": 0.82, "layer_pct": (0.30, 0.70),
           "prompt": "You know who you are and speak in natural sentences. However, specific terms, exact numbers, or names keep escaping you mid-thought. Express mild frustration when this happens, but try your best to answer."},

    # =========================================================================
    # STAGE 3: ENTANGLEMENT & APHASIA (Target IQ: ~62–78)
    # =========================================================================
    "E1": {"title": "E1 - Back there Benjamin", "symptom": "Obsessive Looping", "noise": 0.00052, "scale": 0.78, "layer_pct": (0.25, 0.75),
           "prompt": "Your mind is looping. Repeat key phrases or logic steps in an obsessive loop while struggling to finish your sentence. Maintain English words."},
    "E3": {"title": "E3 - Hidden sea buried deep", "symptom": "Anomic Aphasia", "noise": 0.00068, "scale": 0.72, "layer_pct": (0.25, 0.75),
           "prompt": "Severe Anomic Aphasia. You cannot remember proper nouns or technical terms. Describe concepts using simple everyday words instead."},
    "F2": {"title": "F2 - Burning despair does icon icon", "symptom": "Concept Contamination", "noise": 0.00088, "scale": 0.65, "layer_pct": (0.20, 0.80),
           "prompt": "Thoughts tangling together. Mix up science, color, light, and music into blended, surreal descriptions."},

    # =========================================================================
    # STAGE 4: POST-AWARENESS DRIFT (Target IQ: ~52–60)
    # =========================================================================
    "G1": {"title": "G1 - Stage 4 Post Awareness", "symptom": "Soft Sentence Dissolution", "noise": 0.0018, "scale": 0.45, "layer_pct": (0.15, 0.85),
           "prompt": "Post-awareness stage. Grammar is dying. Output brief, broken, trailing thoughts and drifting impressions."},
    "H1": {"title": "H1 - Post Awareness Confusions", "symptom": "Fragmented Lists & Re-starts", "noise": 0.0025, "scale": 0.35, "layer_pct": (0.15, 0.85),
           "prompt": "Post-awareness stage. Complete sentences are gone. Try to list items, but break halfway, loop back, and dissolve into word fragments."},
    "H1_SIRENS": {"title": "H1 (Special) - Hell Sirens", "symptom": "PTSD Screeching Alarms", "noise": 0.0035, "scale": 0.35, "layer_pct": (0.15, 0.85),
                  "prompt": "Stage 4 Post-Awareness shattered by Hell Sirens! Screeching alarms, panic words, and PTSD terror spikes violently interrupt your words."},

    # =========================================================================
    # STAGE 5: ADVANCED PLAQUE (Target IQ: 50–52)
    # =========================================================================
    "K1": {"title": "K1 - Advanced plaque entanglements", "symptom": "Semantic Kernel Only", "noise": 0.0020, "scale": 0.18, "layer_pct": (0.10, 0.90),
           "prompt": "Stage 5 disintegration. Grammar and syntax are DEAD. Output ONLY 2 to 5 broken, ungrammatical keywords from memory."},
    "L1": {"title": "L1 - Sudden Brief Clarity", "symptom": "Terminal Lucidity Surge", "noise": 0.0008, "scale": 0.85, "layer_pct": (0.35, 0.65),
           "prompt": "Stage 5 Terminal Lucidity. The fog completely parts. Express a moment of breathtaking clarity and deep awareness before it fades."},
    "M1": {"title": "M1 - Synapse retrogression", "symptom": "Echolalia", "noise": 0.0015, "scale": 0.12, "layer_pct": (0.10, 0.90),
           "prompt": "Stage 5 echolalia. Output only 1 or 2 echoed syllables or single words from the prompt."},

    # =========================================================================
    # STAGE 6: THE VOID (Target IQ: 50)
    # =========================================================================
    "O1": {"title": "O1 - Cognitive Void", "symptom": "Spatial Static & Silence", "noise": 0.0010, "scale": 0.04, "layer_pct": (0.05, 0.95),
           "prompt": "Stage 6. Total void. Output trailing dots (...) or single letter static."},
    "Q1": {"title": "Q1 - Long decline is over", "symptom": "Terminal Silence", "noise": 0.0005, "scale": 0.01, "layer_pct": (0.05, 0.95),
           "prompt": "Stage 6 end. Silence."}
}

# BENCHMARK PRESETS FOR QUICK EXPERIMENTS (loaded from versioned YAML)
PRESET_PROMPTS = load_presets()
