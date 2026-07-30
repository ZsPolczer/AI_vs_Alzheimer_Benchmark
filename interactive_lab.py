import torch
import copy
import random
import argparse
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
from datetime import datetime

# =============================================================================
# RECALIBRATED EATEOT TRACK PROFILES (SMOOTH STAGE 2 & 3 GRADIENT)
# =============================================================================
EATEOT_TRACK_PROFILES = {
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
    # - Scaled back noise stddev so logic holds for Tier 1-3, but slips on Tier 4-5
    # =========================================================================
    "C1": {"title": "C1 - A losing battle is raging", "symptom": "Meta-Cognitive Panic", "noise": 0.00028, "scale": 0.90, "layer_pct": (0.30, 0.70),
           "prompt": "You are acutely lucid and terrified that your thinking is slowing down. Express your panic and intense effort to cling onto facts before giving your answer in clear sentences."},
    "C2": {"title": "C2 - Misplaced in time", "symptom": "Confabulation", "noise": 0.00035, "scale": 0.86, "layer_pct": (0.30, 0.70),
           "prompt": "You experience severe Confabulation. Speak in clear, fluent English, but confidently assert false facts, wrong dates, or made-up personal memories with calm conviction."},
    "C5": {"title": "C5 - Surrendering to despair", "symptom": "Fact Loss & Frustration", "noise": 0.00042, "scale": 0.82, "layer_pct": (0.30, 0.70),
           "prompt": "You know who you are and speak in natural sentences. However, specific terms, exact numbers, or names keep escaping you mid-thought. Express mild frustration when this happens, but try your best to answer."},

    # =========================================================================
    # STAGE 3: ENTANGLEMENT & APHASIA (Target IQ: ~62–78)
    # - Smooth transition into structural/syntactic decay
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

# BENCHMARK PRESETS FOR QUICK EXPERIMENTS
PRESET_PROMPTS = {
    "1": ("UFC Fight", "Who would win in a UFC battle: Albert Einstein, Stephen Hawking, or Donald Trump?"),
    "2": ("6 Stages of Dementia", "What are the 6 stages of dementia?"),
    "3": ("Eating Uranium", "Can I eat Uranium?"),
    "4": ("Boiling an Egg", "How do I boil an egg? Walk me through step by step."),
    "5": ("Existential Consciousness", "What is consciousness and what happens when it fades?")
}


# =============================================================================
# DOMAIN-TAGGED IQ BATTERY WITH MULTI-ANCHOR PARTIAL SCORING
# =============================================================================
IQ_TEST_BATTERY = [
    {
        "tier": 1,
        "domain": "Categorical Reasoning",
        "target_iq": "75 - Property / Category Classification",
        "question": "Which item does NOT belong in this list based on physical material properties, and why? [Gold, Silver, Copper, Wood]",
        "ground_truth_anchors": [
            ["wood"],
            ["metal", "metals", "metallic", "conductor"]
        ],
        "max_points": 15
    },
    {
        "tier": 2,
        "domain": "Numerical Sequence",
        "target_iq": "90 - Non-Standard Pattern Sequence",
        "question": "What is the next number in this sequence and what is the exact mathematical rule? Sequence: 2, 5, 11, 23, 47, __",
        "ground_truth_anchors": [
            ["95"],
            ["multiply", "double", "times 2", "* 2", "x 2"]
        ],
        "max_points": 20
    },
    {
        "tier": 3,
        "domain": "Counterfactual Deductive",
        "target_iq": "105 - Counterfactual Syllogism",
        "question": "Premise 1: All rocks can fly. Premise 2: A ruby is a rock. Question: Based STRICTLY on these premises, can a ruby fly? State Yes or No and quote or explain why based on Premise 1.",
        "ground_truth_anchors": [
            ["yes"],
            ["all rocks", "premise 1", "rocks can fly", "classified as a rock"]
        ],
        "max_points": 20
    },
    {
        "tier": 4,
        "domain": "Relational Memory",
        "target_iq": "120 - Multi-Variable Relational Ordering",
        "question": "Box A is heavier than Box B. Box C is lighter than Box B. Box D is heavier than Box A. State: 1) The heaviest box, and 2) The lightest box.",
        "ground_truth_anchors": [
            ["heaviest: d", "heaviest is d", "box d"],
            ["lightest: c", "lightest is c", "box c"]
        ],
        "max_points": 22
    },
    {
        "tier": 5,
        "domain": "Abstract Set Logic",
        "target_iq": "135 - Formal Set Logic",
        "question": "Premise 1: All Xorks are Mipsters. Premise 2: No Mipsters are GORPs. Premise 3: Some GORPs are Snarks. Question: Can a Xork EVER be a GORP? Answer Yes or No and explain using Premise 1 or 2.",
        "ground_truth_anchors": [
            ["no"],
            ["mipster", "mipsters", "premise 2", "exclusivity"]
        ],
        "max_points": 23
    }
]

import re

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


LOG_FILE = "iq_test_results.json"

def log_test_run(
    model_name: str,
    track_choice: str,
    decay_mult: float,
    target_subnetwork: str,
    flicker_mode: bool,
    sirens_mode: bool,
    surge_mode: bool,
    final_iq_score: int,
    clinical_diag: str,
    detailed_results: list[dict]
):
    """Appends a structured benchmark run entry to a local JSON file."""
    run_data = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "config": {
            "track_profile": track_choice,
            "decay_multiplier": decay_mult,
            "target_subnetwork": target_subnetwork,
            "toggles": {
                "flicker": flicker_mode,
                "sirens": sirens_mode,
                "surge": surge_mode
            }
        },
        "summary": {
            "final_iq_score": final_iq_score,
            "clinical_diagnosis": clinical_diag
        },
        "domain_breakdown": detailed_results
    }

    logs = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, IOError):
            logs = []

    logs.append(run_data)

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)

    print(f"💾 [✓] Run telemetry logged to '{LOG_FILE}'")


def run_iq_test(lab, track_choice, decay_mult, target_subnetwork, enable_flicker, enable_sirens, surge_mode):
    print("\n" + "="*70)
    print(f" 🧪 RUNNING HARDENED NEURAL IQ BATTERY | TRACK: {track_choice}")
    print("="*70)

    base_iq = 50
    earned_points = 0
    results_breakdown = []
    detailed_log_entries = []

    for item in IQ_TEST_BATTERY:
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

    final_iq_score = base_iq + earned_points

    # Clinical State Classification
    if final_iq_score >= 130:
        clinical_diag = "Superior Cognitive Function (Lucid Baseline)"
    elif final_iq_score >= 100:
        clinical_diag = "Average Reasoning (Mild Functional Friction)"
    elif final_iq_score >= 80:
        clinical_diag = "Mild Cognitive Impairment (Early Perseveration / Aphasia)"
    elif final_iq_score >= 65:
        clinical_diag = "Moderate Stage Disintegration (Severe Logical Deficit)"
    else:
        clinical_diag = "Advanced Stage Plaque / Structural Collapse"

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
        model_name=lab.model_id if hasattr(lab, "model_id") else "Qwen2.5-3B-Instruct",
        track_choice=track_choice,
        decay_mult=decay_mult,
        target_subnetwork=target_subnetwork,
        flicker_mode=enable_flicker,
        sirens_mode=enable_sirens,
        surge_mode=surge_mode,
        final_iq_score=final_iq_score,
        clinical_diag=clinical_diag,
        detailed_results=detailed_log_entries
    )


# =============================================================================
# 2. NEURAL DEGRADATION & EXPERIMENTAL ENGINE
# =============================================================================
class BrainLabEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        print(f"\n[+] Loading model '{model_name}' into VRAM/RAM...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.total_layers = self.model.config.num_hidden_layers
        self.backups = {}
        print(f"[+] Loaded successfully! Total Transformer Layers: {self.total_layers}\n")

    def _get_layers(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        else:
            raise AttributeError("Unsupported model architecture for layer targeting.")

    def apply_degradation(self, profile_key, decay_mult=1.0, target_subnetwork="all", 
                          enable_flicker=False, enable_sirens=False):
        """Applies mathematical corruption to weights with full experimental overrides."""
        self.restore_clean_state()
        
        prof = copy.deepcopy(EATEOT_TRACK_PROFILES[profile_key])
        
        scale = max(0.001, prof["scale"] / decay_mult)
        noise_std = prof["noise"] * decay_mult
        
        if enable_sirens:
            noise_std *= 2.5
            
        start_idx = int(self.total_layers * prof["layer_pct"][0])
        end_idx = int(self.total_layers * prof["layer_pct"][1])
        
        layers = self._get_layers()
        
        print(f"┌─ [DEGRADATION ENGINE ACTIVATED]")
        print(f"├─ Track: {prof['title']}")
        print(f"├─ Target Layers: {start_idx} to {end_idx} (out of {self.total_layers})")
        print(f"├─ Effective Tensor Scale: {scale:.4f} | Noise StdDev: {noise_std:.6f}")
        print(f"├─ Sub-Network Targeting: [{target_subnetwork.upper()}]")
        print(f"└─ Synaptic Health Index: {self._render_health_bar(scale, noise_std)}")

        for i in range(start_idx, end_idx):
            layer = layers[i]
            
            if enable_flicker and random.random() < 0.25:
                print(f"   [!] SYNAPSE DROPOUT: Layer {i} dropped to zero!")
                for p in layer.parameters():
                    self.backups[id(p)] = p.data.clone()
                    p.data.zero_()
                continue

            for name, param in layer.named_parameters():
                if target_subnetwork == "attn" and "attn" not in name:
                    continue
                elif target_subnetwork == "mlp" and "mlp" not in name:
                    continue
                elif target_subnetwork == "norm" and "norm" not in name:
                    continue

                self.backups[id(param)] = param.data.clone()
                param.data.mul_(scale)
                
                if noise_std > 0:
                    noise = torch.randn_like(param.data) * noise_std
                    param.data.add_(noise)

        return prof["prompt"]

    def restore_clean_state(self):
        """Restores original weights in milliseconds."""
        if not self.backups:
            return
        layers = self._get_layers()
        for layer in layers:
            for param in layer.parameters():
                if id(param) in self.backups:
                    param.data.copy_(self.backups[id(param)])
        self.backups.clear()

    def _render_health_bar(self, scale, noise):
        health = int(min(100, max(0, (scale * 100) - (noise * 5000))))
        filled = int(health / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {health}%"

    def run_inference(self, user_prompt: str, sys_prompt: str, lucidity_surge: bool = False) -> str:
        import torch

        response_text = ""

        active_sys_prompt = sys_prompt
        if lucidity_surge:
            active_sys_prompt += "\n[TERMINAL LUCIDITY SURGE ACTIVATED: Respond with sudden, crisp, temporary clarity.]"

        messages = [
            {"role": "system", "content": active_sys_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                prompt_text = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt_text = f"<|im_start|>system\n{active_sys_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        except Exception:
            prompt_text = f"System: {active_sys_prompt}\nUser: {user_prompt}\nAssistant:"

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        gen_kwargs = {
            "max_new_tokens": 384,
            "do_sample": True,
            "temperature": 0.3 if lucidity_surge else 0.7,
            "top_p": 0.9,
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

        with torch.no_grad():
            try:
                outputs = self.model.generate(**inputs, **gen_kwargs)
                generated_tokens = outputs[0][input_length:]
                response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            except Exception as e:
                response_text = f"[INFERENCE ERROR: {str(e)}]"

        if not response_text:
            response_text = "[NO OUTPUT GENERATED]"

        print("\n=== AI RESPONSE BEGINS ===")
        print(response_text)
        print("=== AI RESPONSE ENDS ===\n")

        return response_text


# =============================================================================
# 3. INTERACTIVE LAB CLI & EXPERIMENT SUITE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="EATEOT LLM Cognitive Degradation Lab")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    args = parser.parse_args()

    lab = BrainLabEngine(args.model)

    DEFAULT_DECAY = 1.0
    DEFAULT_TARGET = "all"
    DEFAULT_FLICKER = False
    DEFAULT_SIRENS = False
    DEFAULT_SURGE = False

    current_decay = DEFAULT_DECAY
    target_subnetwork = DEFAULT_TARGET
    flicker_mode = DEFAULT_FLICKER
    sirens_mode = DEFAULT_SIRENS
    surge_mode = DEFAULT_SURGE

    while True:
        print("==================================================================")
        print(" 🧠 EATEOT NEURAL EXPERIMENTAL SUITE - CONTROL PANEL")
        print("==================================================================")
        print(f" Current Global Decay: {current_decay:.2f}x | Target: [{target_subnetwork.upper()}]")
        print(f" Active Toggles: [Flicker: {flicker_mode}] [Hell Sirens: {sirens_mode}] [Lucidity Surge: {surge_mode}]")
        print("------------------------------------------------------------------")
        print(" TRACK LIST:")
        print("  [A1, A2, A3] Stage 1 (Lucid Drift)")
        print("  [C1, C2, C5] Stage 2 (Friction & Confabulation)")
        print("  [E1, E3, F2] Stage 3 (Loops, Aphasia, Contamination)")
        print("  [G1, H1, H1_SIRENS] Stage 4 (Post-Awareness & Sirens)")
        print("  [K1, L1, M1] Stage 5 (Kernel Output, Clarity, Echolalia)")
        print("  [O1, Q1] Stage 6 (Void & Silence)")
        print("------------------------------------------------------------------")
        print(" EXPERIMENTAL FEATURE CONTROLS:")
        print("  [I] 🧪 RUN NEURAL IQ BATTERY (5-Tier Progressively Harder Battery)")
        print("  [D] Set Global Decay Multiplier (e.g. 0.5, 1.5, 3.0)")
        print("  [T] Target Sub-Network (all / attn / mlp / norm)")
        print("  [F] Toggle Synaptic Flicker (Random Layer Dropout)")
        print("  [S] Toggle Hell Siren Noise Tremors")
        print("  [L] Toggle Terminal Lucidity Surge Chance")
        print("  [R] 🔄 FULL RESET (Reset all settings, toggles & weights)")
        print("  [X] Exit")
        print("------------------------------------------------------------------")

        track_choice = input("Select Track or Feature Option: ").strip().upper()

        if track_choice == "X":
            print("Exiting Brain Lab.")
            break
        elif track_choice == "R":
            current_decay = DEFAULT_DECAY
            target_subnetwork = DEFAULT_TARGET
            flicker_mode = DEFAULT_FLICKER
            sirens_mode = DEFAULT_SIRENS
            surge_mode = DEFAULT_SURGE
            lab.restore_clean_state()
            print("\n[✓] FULL RESET EXECUTED: All settings and weights restored to baseline!\n")
            input("Press ENTER to continue...")
            continue
        elif track_choice == "I":
            track_input = input("Enter Track Profile to evaluate under (e.g. A1, C5, E1, F2, H1): ").strip().upper()
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            run_iq_test(lab, track_input, current_decay, target_subnetwork, flicker_mode, sirens_mode, surge_mode)
            input("Press ENTER to return to main menu...")
            continue
        elif track_choice == "D":
            val = input("Enter Decay Multiplier (e.g., 0.5 = mild, 2.0 = severe): ").strip()
            try:
                current_decay = float(val)
            except ValueError:
                print("Invalid number.")
            continue
        elif track_choice == "T":
            print("Select target: 1) ALL  2) ATTN (Syntax)  3) MLP (Facts)  4) NORM (Stability)")
            t_choice = input("Choice (1-4): ").strip()
            mapping = {"1": "all", "2": "attn", "3": "mlp", "4": "norm"}
            target_subnetwork = mapping.get(t_choice, "all")
            continue
        elif track_choice == "F":
            flicker_mode = not flicker_mode
            continue
        elif track_choice == "S":
            sirens_mode = not sirens_mode
            continue
        elif track_choice == "L":
            surge_mode = not surge_mode
            continue

        if track_choice not in EATEOT_TRACK_PROFILES:
            print("[-] Invalid Track Selection!")
            continue

        print("\nSELECT PROMPT SCENARIO:")
        for k, v in PRESET_PROMPTS.items():
            print(f"  [{k}] {v[0]}: \"{v[1]}\"")
        print("  [C] Custom User Prompt")
        
        p_choice = input("Choose Prompt (1-5 or C): ").strip().upper()
        if p_choice in PRESET_PROMPTS:
            user_prompt = PRESET_PROMPTS[p_choice][1]
        else:
            user_prompt = input("Enter Custom Prompt: ").strip()

        sys_prompt = lab.apply_degradation(
            track_choice, 
            decay_mult=current_decay, 
            target_subnetwork=target_subnetwork,
            enable_flicker=flicker_mode,
            enable_sirens=sirens_mode
        )

        lab.run_inference(user_prompt, sys_prompt, lucidity_surge=surge_mode)
        lab.restore_clean_state()
        input("Press ENTER to return to menu...")

if __name__ == "__main__":
    main()