#!/usr/bin/env python3
"""Interactive lab CLI — the main entry point of the EATEOT suite.

Port of ``interactive_lab.py``'s ``main()``, now consuming the ``eateot``
package. Run directly (``python apps/lab.py``) or via the installed console
script (``eateot-lab``).
"""

import argparse
from pathlib import Path

from eateot import (
    DRUG_LOG_FILE,
    DRUG_PROFILES,
    LOG_FILE,
    BrainLabEngine,
    PRESET_PROMPTS,
    list_drugs,
    parse_stack,
    resolve_drug,
    resolve_stack,
    run_iq_test,
)
from eateot.config import (
    DEFAULT_DECAY,
    DEFAULT_FLICKER,
    DEFAULT_SIRENS,
    DEFAULT_SUBNET,
    DEFAULT_SURGE,
)
from eateot.profiles import EATEOT_TRACK_PROFILES
from eateot.questionnaires import list_batteries, load_battery


def resolve_cli_drug(drug: str | None, dose: float = 1.0):
    """Validate ``--drug`` / ``--dose`` flags into a resolved drug spec.

    Returns ``None`` when no drug was given, otherwise the spec from
    ``eateot.drugs.resolve_drug``. Raises ``SystemExit`` with a friendly
    message for unknown drugs or invalid doses (e.g. negative).
    """
    if not drug:
        return None
    if drug not in DRUG_PROFILES:
        raise SystemExit(
            f"Unknown drug '{drug}'. Available: {', '.join(list_drugs())}"
        )
    try:
        return resolve_drug(drug, dose)
    except ValueError as e:
        raise SystemExit(f"Invalid drug dose: {e}") from None


def resolve_cli_stack(stack: str | None):
    """Validate ``--stack`` (e.g. ``lsd@1.0,thc@0.5``) into a resolved stack spec.

    Returns ``None`` when no stack was given, otherwise the merged spec from
    ``eateot.drugs.resolve_stack``. Raises ``SystemExit`` with a friendly
    message for malformed specs or unknown drugs.
    """
    if not stack:
        return None
    try:
        components = parse_stack(stack)
    except ValueError as e:
        raise SystemExit(f"Invalid stack spec: {e}") from None
    try:
        return resolve_stack(components)
    except ValueError as e:
        raise SystemExit(f"Invalid drug stack: {e}") from None


def _select_drug_stack() -> list[dict]:
    """Interactive dependent selection: class → drug → dose, repeatable for combos.

    Walks the catalog by class first (so the drug list depends on the chosen
    class), then asks for a dose using the drug's ``dose_cap`` as a hint, and
    keeps prompting until the user declines to stack another drug. Returns
    ``[{drug, dose}, ...]`` (empty when the user cancels).
    """
    classes = sorted({p["class"] for p in DRUG_PROFILES.values()})
    components = []
    while True:
        print("\n── SELECT DRUG CLASS ──")
        for i, cls in enumerate(classes, 1):
            count = sum(1 for p in DRUG_PROFILES.values() if p["class"] == cls)
            print(f"  [{i}] {cls} ({count} drugs)")
        print("  [0] finish / cancel")
        pick = input("Class (number): ").strip()
        if pick == "0":
            break
        if pick == "":
            continue  # empty re-prompts instead of silently discarding a stack
        try:
            cls = classes[int(pick) - 1]
        except (ValueError, IndexError):
            print("[-] Invalid class")
            continue

        drugs = sorted(n for n, p in DRUG_PROFILES.items() if p["class"] == cls)
        print(f"\n── {cls.upper()} — SELECT DRUG ──")
        for i, name in enumerate(drugs, 1):
            cap = DRUG_PROFILES[name].get("dose_cap", 1.0)
            print(f"  [{i}] {name}   (dose_cap {cap:g})")
        print("  [0] back to classes")
        pick = input("Drug (number or name): ").strip()
        if pick == "0":
            continue
        if pick.isdigit():
            try:
                name = drugs[int(pick) - 1]
            except IndexError:
                print("[-] Invalid drug")
                continue
        else:
            name = pick.strip()
            if name not in DRUG_PROFILES:
                print(f"[-] Unknown drug '{name}'")
                continue

        cap = DRUG_PROFILES[name].get("dose_cap", 1.0)
        dose_s = input(f"Dose for {name} (recommended cap {cap:g}, default 1.0): ").strip()
        try:
            dose = float(dose_s) if dose_s else 1.0
        except ValueError:
            print("[-] Invalid dose, using 1.0")
            dose = 1.0
        if dose < 0:
            print("[-] Dose must be >= 0, using 1.0")
            dose = 1.0
        if dose > cap:
            print(f"[!] Heroic dose: {dose:g} exceeds {name}'s dose_cap ({cap:g})")
        components.append({"drug": name, "dose": dose})
        print(f"[+] Added {name} @ {dose:g} to the stack")
        if len(components) >= 4:
            print("[-] Stack limit reached (4 drugs)")
            break
        if input("Add another drug to the stack? (y/N): ").strip().lower() != "y":
            break
    return components


def _print_drug_summary(spec: dict):
    """Print a resolved drug/stack spec as a readable primitives summary."""
    is_stack = bool(spec.get("components"))
    kind = (f"stack ({len(spec['components'])} components)" if is_stack
            else f"{spec['class']} @ {spec['dose']}x")
    print("\n" + "─" * 62)
    print(f" 💊 DEPLOYED: {spec['name']} — {kind}")
    print(f"    subnetwork: {spec['subnetwork']} | layer window: "
          f"{spec['layer_pct'][0]:.2f}–{spec['layer_pct'][1]:.2f}")
    if spec.get("restore_fraction") is not None:
        print(f"    restore_fraction: {spec['restore_fraction']}")
    print("    resolved primitives:")
    for k, v in sorted(spec["primitives"].items()):
        print(f"      {k:<22} {v:+.5g}")
    print("─" * 62)


def main():
    parser = argparse.ArgumentParser(description="EATEOT LLM Cognitive Degradation Lab")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    parser.add_argument(
        "--questionnaire", type=str, default="iq_battery",
        help=f"Questionnaire to run (default: iq_battery). Available: {', '.join(list_batteries())}",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional seed for reproducible lesion + sampling")
    parser.add_argument(
        "--drug", type=str, default=None,
        help=f"Psychoactive profile to apply on top of the track (default: none). "
             f"Available: {', '.join(list_drugs())}",
    )
    parser.add_argument("--dose", type=float, default=1.0,
                        help="Drug dose / intensity (default: 1.0). Each drug has a "
                             "recommended dose_cap in the catalog. Ignored when "
                             "--stack is given (doses live in the stack spec).")
    parser.add_argument(
        "--stack", type=str, default=None,
        help="Drug combo spec 'name@dose,name@dose' (e.g. 'lsd@1.0,thc@0.5'; "
             "'+' also works as separator). Mutually exclusive with --drug.",
    )
    args = parser.parse_args()

    if args.drug and args.stack:
        raise SystemExit("--drug and --stack are mutually exclusive")
    # The active spec: a --stack combo, a --drug profile, or None. Reassigned
    # by the interactive deployer ([P]) when the user builds a combo in-menu.
    drug_spec = resolve_cli_stack(args.stack) or resolve_cli_drug(args.drug, args.dose)

    battery = load_battery(args.questionnaire)
    battery_name = args.questionnaire

    lab = BrainLabEngine(args.model)

    current_decay = DEFAULT_DECAY
    target_subnetwork = DEFAULT_SUBNET
    flicker_mode = DEFAULT_FLICKER
    sirens_mode = DEFAULT_SIRENS
    surge_mode = DEFAULT_SURGE

    while True:
        print("\n==================================================================")
        print(" 🧠 EATEOT NEURAL EXPERIMENTAL SUITE - CONTROL PANEL")
        print("==================================================================")
        print(f" Global Decay: {current_decay:.2f}x | Target: [{target_subnetwork.upper()}]")
        print(f" Toggles: [Flicker: {flicker_mode}] [Sirens: {sirens_mode}] [Surge: {surge_mode}]")
        if drug_spec:
            if drug_spec.get("components"):
                print(f" Active Drug: {drug_spec['name']} (stack · {len(drug_spec['components'])} components)")
            else:
                print(f" Active Drug: {drug_spec['name']} @ {drug_spec['dose']}x "
                      f"({drug_spec['class']} · {drug_spec['target_domain']})")
            print(f" Drug Domain → {Path(DRUG_LOG_FILE).name}")
        else:
            print(" Active Drug: none")
            print(f" Alzheimer's Domain → {Path(LOG_FILE).name}")

        print("\n────────── 🧠 ALZHEIMER'S TRACKS ──────────")
        print("  [A1, A2, A3] Stage 1 — Lucid Drift")
        print("  [C1, C2, C5] Stage 2 — Friction & Confabulation")
        print("  [E1, E3, F2] Stage 3 — Loops, Aphasia, Contamination")
        print("  [G1, H1, H1_SIRENS] Stage 4 — Post-Awareness & Sirens")
        print("  [K1, L1, M1] Stage 5 — Kernel Output, Clarity, Echolalia")
        print("  [O1, Q1] Stage 6 — Void & Silence")
        print("  Type a track name above (e.g. A1, C5) to run a custom prompt.")
        print("  [I] 🧪 IQ BATTERY — run 5-tier battery on a chosen track")

        print("\n────────── 💊 DRUG EXPERIMENTS ──────────")
        print("  No Alzheimer's track needed — clean baseline + drug primitives.")
        print("  [Q] 💊 DRUG IQ BATTERY — deploy drug → run 5-tier battery")
        print("  [E] 💊 DRUG CUSTOM PROMPT — deploy drug → free-form prompt")
        print("  [P] 💊 DEPLOY DRUG — set active drug without running a test")
        if drug_spec:
            print("  [U] 🔓 UNDEPLOY DRUG — clear the active drug")

        print("\n────────── ⚙️  SETTINGS ──────────")
        print("  [D] Decay Multiplier  [T] Target Sub-Network")
        print("  [F] Flicker  [S] Sirens  [L] Lucidity Surge")
        print("  [R] 🔄 FULL RESET  [X] Exit")
        print("==================================================================")

        track_choice = input("▸ ").strip().upper()

        # ── Exit / Reset ──────────────────────────────────────────────────
        if track_choice == "X":
            print("Exiting Brain Lab.")
            break
        elif track_choice == "R":
            current_decay = DEFAULT_DECAY
            target_subnetwork = DEFAULT_SUBNET
            flicker_mode = DEFAULT_FLICKER
            sirens_mode = DEFAULT_SIRENS
            surge_mode = DEFAULT_SURGE
            drug_spec = None
            lab.restore_clean_state()
            print("\n[✓] FULL RESET: settings, toggles, weights & drug cleared.\n")
            input("Press ENTER to continue...")
            continue

        # ── Settings ──────────────────────────────────────────────────────
        elif track_choice == "D":
            val = input("Decay multiplier (e.g. 0.5 = mild, 2.0 = severe): ").strip()
            try:
                current_decay = float(val)
            except ValueError:
                print("Invalid number.")
            continue
        elif track_choice == "T":
            print("Target: 1) ALL  2) ATTN (Syntax)  3) MLP (Facts)  4) NORM (Stability)")
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

        # ── Drug management ───────────────────────────────────────────────
        elif track_choice == "U":
            drug_spec = None
            print("[✓] Drug cleared — back to Alzheimer's-only mode.")
            continue
        elif track_choice == "P":
            components = _select_drug_stack()
            if not components:
                print("[-] No drug selected — nothing deployed.")
                continue
            try:
                drug_spec = resolve_stack(components)
            except ValueError as e:
                print(f"[-] {e}")
                continue
            _print_drug_summary(drug_spec)
            input("Press ENTER to return to menu...")
            continue

        # ── Drug IQ battery (CLEAN track — no Alzheimer's) ────────────────
        elif track_choice == "Q":
            if not drug_spec:
                print("[-] No drug deployed. Use [P] to deploy one first.")
                continue
            run_iq_test(
                lab, "CLEAN", current_decay, target_subnetwork,
                flicker_mode, sirens_mode, surge_mode,
                battery=battery, battery_name=battery_name,
                seed=args.seed,
                drug=drug_spec,
            )
            input("Press ENTER to return to menu...")
            continue

        # ── Drug custom prompt (CLEAN track — no Alzheimer's) ─────────────
        elif track_choice == "E":
            if not drug_spec:
                print("[-] No drug deployed. Use [P] to deploy one first.")
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
                "CLEAN",
                decay_mult=current_decay,
                target_subnetwork=target_subnetwork,
                enable_flicker=flicker_mode,
                enable_sirens=sirens_mode,
                noise_seed=args.seed,
                drug=drug_spec,
            )
            lab.run_inference(user_prompt, sys_prompt, lucidity_surge=surge_mode,
                              seed=args.seed, drug=drug_spec)
            lab.restore_clean_state()
            input("Press ENTER to return to menu...")
            continue

        # ── Alzheimer's IQ battery ────────────────────────────────────────
        elif track_choice == "I":
            track_input = input("Track Profile (e.g. A1, C5, E1, F2, H1): ").strip().upper()
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            run_iq_test(
                lab, track_input, current_decay, target_subnetwork,
                flicker_mode, sirens_mode, surge_mode,
                battery=battery, battery_name=battery_name,
                seed=args.seed,
                drug=drug_spec,
            )
            input("Press ENTER to return to main menu...")
            continue

        # ── Alzheimer's track → custom prompt (direct track name entry) ───
        elif track_choice in EATEOT_TRACK_PROFILES:
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
                enable_sirens=sirens_mode,
                noise_seed=args.seed,
                drug=drug_spec,
            )
            lab.run_inference(user_prompt, sys_prompt, lucidity_surge=surge_mode,
                              seed=args.seed, drug=drug_spec)
            lab.restore_clean_state()
            input("Press ENTER to return to menu...")
            continue
        else:
            print("[-] Invalid selection! Type a track name (A1, C5…), or a letter command (I, Q, E, P, D, T, F, S, L, R, X).")
            continue


if __name__ == "__main__":
    main()
