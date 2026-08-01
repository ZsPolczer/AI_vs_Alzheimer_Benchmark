#!/usr/bin/env python3
"""Interactive lab CLI — the main entry point of the EATEOT suite.

Port of ``interactive_lab.py``'s ``main()``, now consuming the ``eateot``
package. Run directly (``python apps/lab.py``) or via the installed console
script (``eateot-lab``).
"""

import argparse

from eateot import BrainLabEngine, PRESET_PROMPTS, run_iq_test
from eateot.config import (
    DEFAULT_DECAY,
    DEFAULT_FLICKER,
    DEFAULT_SIRENS,
    DEFAULT_SUBNET,
    DEFAULT_SURGE,
)
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery


def main():
    parser = argparse.ArgumentParser(description="EATEOT LLM Cognitive Degradation Lab")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    args = parser.parse_args()

    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire

    lab = BrainLabEngine(args.model)

    current_decay = DEFAULT_DECAY
    target_subnetwork = DEFAULT_SUBNET
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
            target_subnetwork = DEFAULT_SUBNET
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
            run_iq_test(
                lab, track_input, current_decay, target_subnetwork,
                flicker_mode, sirens_mode, surge_mode,
                battery=battery, battery_name=battery_name,
            )
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
