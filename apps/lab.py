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
from eateot.engine import progressive_ramp_intensity
from eateot.metacognition import (
    profile_windows,
    trace_line,
    write_jsonl,
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


def _run_app(module_name: str, argv: list[str]):
    """Run another EATEOT app's ``main()`` with a synthesized argv.

    Used by the Studies & Reports menu so ``eateot-lab`` is the single entry
    point for every feature. ``sys.argv`` is swapped for the duration so the
    target's argparse sees its own flags, then restored. ``SystemExit`` from
    the app (e.g. ``sys.exit(1)`` on an empty report) is caught and reported
    so the lab loop keeps running.
    """
    import importlib
    import sys

    try:
        mod = importlib.import_module(f"apps.{module_name}")
    except ImportError as e:
        print(f"[-] Could not load {module_name}: {e}")
        return
    saved = sys.argv
    sys.argv = [f"eateot-lab {module_name}"] + list(argv)
    try:
        mod.main()
    except SystemExit as e:
        print(f"[-] {module_name} aborted: {e}")
    finally:
        sys.argv = saved


def _prompt_int(prompt: str, default: int) -> int:
    """Read an integer with a default on empty/invalid input."""
    raw = input(prompt).strip()
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"[-] Invalid number, using {default}")
        return default


def _parse_layer_indices(raw: str, total_layers: int) -> list[int]:
    """Parse a user layer pick (e.g. ``0,5,12`` or ``8-10``) into valid indices.

    Accepts comma-separated indices and inclusive ranges (``start-end``, either
    order), dedupes, sorts, and validates every index against
    ``0..total_layers-1``. Raises ``ValueError`` with a friendly message on
    malformed tokens or out-of-range indices.
    """
    indices: set[int] = set()
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        if "-" in token:
            try:
                start_s, end_s = token.split("-", 1)
                start, end = int(start_s), int(end_s)
            except ValueError:
                raise ValueError(f"invalid layer range '{token}' (use e.g. 8-10)") from None
            if start > end:
                start, end = end, start
            indices.update(range(start, end + 1))
        else:
            try:
                indices.add(int(token))
            except ValueError:
                raise ValueError(f"invalid layer index '{token}'") from None
    if not indices:
        raise ValueError("no layer indices given (e.g. '0,5,12' or '8-10')")
    out = sorted(indices)
    bad = [i for i in out if not 0 <= i < total_layers]
    if bad:
        raise ValueError(
            f"layer index out of range 0..{total_layers - 1}: {bad}"
        )
    return out


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
        "--epsilon", type=float, default=0.0,
        help="Std-scaled Gaussian perturbation strength (Ẇ = W + ε·σ_W·Z) applied "
             "on top of the track — 0 disables (default: 0.0)",
    )
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
    parser.add_argument(
        "--monitor", action="store_true",
        help="In the [G] progressive experiment, trace the model's "
             "metacognitive markers per response window (live bars + "
             "metacognition_*.jsonl log in the outputs dir).",
    )
    args = parser.parse_args()

    if args.drug and args.stack:
        raise SystemExit("--drug and --stack are mutually exclusive")
    # The active spec: a --stack combo, a --drug profile, or None. Reassigned
    # by the interactive deployer ([P]) when the user builds a combo in-menu.
    if args.epsilon < 0:
        raise SystemExit("--epsilon must be >= 0")
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

        print("\n────────── 🧪 EXPERIMENTAL ──────────")
        print("  [G] 📉 PROGRESSIVE DEGRADATION — answer degrades WHILE generating")
        print("      (hidden states corrupt as the model speaks: barely → full mayhem)")
        print("  [Z] 🧩 LAYER LESION — surgically zero out the layers you choose by index")

        print("\n────────── 📊 STUDIES & REPORTS ──────────")
        print("  [1] 📉 SENSITIVITY — IQ vs ε grid (Ẇ = W + ε·σ_W·Z)")
        print("  [2] 🧪 RESTORE — dose-response recovery (restore fraction)")
        print("  [3] 🧠 TRAJECTORY — full A1→Q1 decline in one session")
        print("  [4] 💊 TRIP — drug dose-response (IQ vs dose)")
        print("  [5] 🛡️ RESERVE — model size × severity (cognitive reserve)")
        print("  [6] ⚖️ COMPARE — 0.5B vs 1.5B vs 3B on the A1 baseline")
        print("  [7] 📈 PLOT — IQ decay chart from telemetry")
        print("  [8] 📋 DRUG REPORT — drug telemetry summary")

        print("\n────────── ⚙️  SETTINGS ──────────")
        print("  [D] Decay Multiplier  [T] Target Sub-Network")
        print("  [F] Flicker  [S] Sirens  [L] Lucidity Surge")
        print("  [W] 🗑️ CLEAR GRAPHS  [R] 🔄 FULL RESET  [X] Exit")
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
        elif track_choice == "W":
            # Delete every generated chart PNG so studies can be re-run for
            # fresh graphs. Telemetry logs + reports are kept.
            from eateot.paths import clear_charts
            removed = clear_charts()
            if removed:
                print(f"\n[✓] CLEARED GRAPHS: removed {len(removed)} chart(s).\n")
                for chart in removed:
                    print(f"    🗑️ {chart}")
                print("\nRun a study ([1]–[8]) to generate fresh graphs.")
            else:
                print("\n[✓] No generated charts to clear — run a study ([1]–[8]) to create some.")
            print()
            input("Press ENTER to return to menu...")
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

        # ── Progressive in-generation degradation (experimental) ──────────
        elif track_choice == "G":
            track_input = input("Track Profile (e.g. A1, C1, E1, F2, CLEAN): ").strip().upper()
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
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

            # Weight-level lesion first (the chosen track), then the hidden
            # states are progressively corrupted on top while the model talks.
            sys_prompt = lab.apply_degradation(
                track_input,
                decay_mult=current_decay,
                target_subnetwork=target_subnetwork,
                enable_flicker=flicker_mode,
                enable_sirens=sirens_mode,
                noise_seed=args.seed,
                drug=drug_spec,
                epsilon=args.epsilon,
            )
            response_text = lab.run_progressive_inference(
                user_prompt, sys_prompt, lucidity_surge=surge_mode,
                seed=args.seed, drug=drug_spec,
            )
            lab.restore_clean_state()

            # Metacognition monitor: profile the response in windows aligned to
            # the degradation ramp, print the live trace, and log to jsonl.
            if args.monitor and response_text and not response_text.startswith("["):
                # Each window's midpoint is its position in the response
                # (0 → 1), mapped through the same ramp the engine applied —
                # so window 0 shows the near-clean start and the last window
                # shows full-mayhem corruption.
                n_windows = max(1, (len(response_text) + 349) // 350)
                intensities = [
                    progressive_ramp_intensity((i + 0.5) / n_windows)
                    for i in range(n_windows)
                ]
                windows = profile_windows(response_text, intensities=intensities)
                print("\n┌─ [METACOGNITION MONITOR]")
                for entry in windows:
                    print("├─ " + trace_line(entry))
                log_path = write_jsonl(
                    windows, "outputs", model_name=lab.model_id, track=track_input,
                )
                print(f"└─ Logged {len(windows)} windows → {log_path}")

            input("Press ENTER to return to menu...")
            continue

        # ── Layer lesion (experimental): surgically zero chosen layers ────
        elif track_choice == "Z":
            raw = input(f"Layers to zero out (0-{lab.total_layers - 1}, "
                        f"e.g. '0,5,12' or '8-10'): ").strip()
            try:
                indices = _parse_layer_indices(raw, lab.total_layers)
            except ValueError as e:
                print(f"[-] {e}")
                continue
            print("Sub-network to sever: 1) ALL  2) ATTN  3) MLP  4) NORM")
            t_choice = input(f"Choice (1-4, default ALL): ").strip() or "1"
            lesion_subnetwork = {"1": "all", "2": "attn",
                                 "3": "mlp", "4": "norm"}.get(t_choice, "all")

            print("\nSELECT PROMPT SCENARIO:")
            for k, v in PRESET_PROMPTS.items():
                print(f"  [{k}] {v[0]}: \"{v[1]}\"")
            print("  [C] Custom User Prompt")
            p_choice = input("Choose Prompt (1-5 or C): ").strip().upper()
            if p_choice in PRESET_PROMPTS:
                user_prompt = PRESET_PROMPTS[p_choice][1]
            else:
                user_prompt = input("Enter Custom Prompt: ").strip()

            # Pure surgical lesion: chosen layers are severed, everything
            # else stays clean (no track degradation). Restore afterwards so
            # the lesion cannot leak into later runs.
            lab.lesion_layers(indices, subnetwork=lesion_subnetwork)
            sys_prompt = EATEOT_TRACK_PROFILES["CLEAN"]["prompt"]
            try:
                lab.run_inference(user_prompt, sys_prompt,
                                  lucidity_surge=surge_mode, seed=args.seed,
                                  drug=drug_spec)
            finally:
                lab.restore_clean_state()
            input("Press ENTER to return to menu...")
            continue

        # ── Studies & Reports ─────────────────────────────────────────────
        elif track_choice == "1":
            # Sensitivity: IQ vs ε grid, reusing the loaded engine.
            from apps.sensitivity import (DEFAULT_EPSILONS, monotonic_verdict,
                                          parse_epsilons, run_study,
                                          write_chart, write_json, write_markdown)
            track_input = input("Track Profile (default CLEAN): ").strip().upper() or "CLEAN"
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            default_eps = ",".join(str(e) for e in DEFAULT_EPSILONS)
            raw_eps = input(f"Epsilons (default {default_eps}): ").strip()
            epsilons = parse_epsilons(raw_eps) if raw_eps else DEFAULT_EPSILONS
            trials = _prompt_int("Trials (default 1): ", 1)
            seed = args.seed if args.seed is not None else 42
            print(f"\n[+] Sensitivity: track {track_input} · ε grid {epsilons} · trials {trials} · seed {seed}")
            results = run_study(lab, track_input, epsilons, trials, seed,
                                battery, battery_name)
            if not any(r["trials"] for r in results):
                print("[-] no successful runs at any ε")
                continue
            verdict, iq_v, g_v = monotonic_verdict(results)
            print(f"\n{verdict}\n")
            write_markdown(results, track_input, lab.model_id, battery_name,
                           seed, verdict, iq_v, g_v)
            write_json(results, track_input, lab.model_id, battery_name,
                       seed, verdict, iq_v, g_v)
            write_chart(results)
            print(f"[OK] wrote sensitivity report · track {track_input} · quiz {battery_name}")
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "2":
            # Restore: dose-response recovery, reusing the loaded engine.
            from apps.restore import (DEFAULT_FRACTIONS, parse_fractions,
                                      run_study, write_chart, write_json,
                                      write_markdown)
            track_input = input("Track Profile (default G1): ").strip().upper() or "G1"
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            default_frac = ",".join(str(f) for f in DEFAULT_FRACTIONS)
            raw_frac = input(f"Restore fractions (default {default_frac}): ").strip()
            fractions = parse_fractions(raw_frac) if raw_frac else DEFAULT_FRACTIONS
            trials = _prompt_int("Trials (default 1): ", 1)
            seed = args.seed if args.seed is not None else 42
            print(f"\n[+] Restore: track {track_input} · doses {fractions} · trials {trials} · seed {seed}")
            results = run_study(lab, track_input, fractions, trials, seed,
                                battery, battery_name)
            if not any(d["trials"] for d in results):
                print("[-] no successful runs at any dose")
                continue
            write_markdown(results, track_input, lab.model_id, battery_name, seed)
            write_json(results, track_input, lab.model_id, battery_name, seed)
            write_chart(results)
            print(f"[OK] wrote restore report · track {track_input} · quiz {battery_name}")
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "3":
            # Trajectory: full A1→Q1 decline, reusing the loaded engine.
            from apps.trajectory import run_trajectory, write_chart, write_markdown
            seed = args.seed
            print(f"\n[+] Trajectory: full A1→Q1 decline · seed {seed}")
            results = run_trajectory(lab, battery, battery_name, seed)
            if not results:
                print("[-] no successful runs in trajectory")
                continue
            write_markdown(results, lab.model_id, battery_name, seed)
            write_chart(results, lab.model_id, battery_name)
            print(f"[OK] wrote trajectory report · quiz {battery_name}")
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "4":
            # Trip: drug dose-response, reusing the loaded engine.
            from apps.trip import (default_dose_sweep, parse_doses, run_study,
                                   write_chart, write_json, write_markdown)
            drug_name = input("Drug (e.g. lsd, salvia, nzt): ").strip().lower()
            if drug_name not in DRUG_PROFILES:
                print(f"[-] Unknown drug '{drug_name}'. Available: {', '.join(list_drugs())}")
                continue
            track_input = input("Track Profile (default C1): ").strip().upper() or "C1"
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            spec = resolve_cli_drug(drug_name, 1.0)
            default_doses = default_dose_sweep(drug_name, spec=spec)
            raw_doses = input(f"Doses (default {default_doses}): ").strip()
            doses = parse_doses(raw_doses) if raw_doses else default_doses
            trials = _prompt_int("Trials (default 1): ", 1)
            seed = args.seed if args.seed is not None else 42
            print(f"\n[+] Trip: {drug_name} @ doses {doses} · track {track_input} · trials {trials} · seed {seed}")
            results = run_study(lab, drug_name, track_input, doses, trials, seed,
                                battery, battery_name)
            if not any(d["trials"] for d in results):
                print("[-] no successful runs at any dose")
                continue
            write_markdown(results, drug_name, track_input, lab.model_id,
                           battery_name, seed)
            write_json(results, drug_name, track_input, lab.model_id,
                       battery_name, seed)
            write_chart(results, drug_name, track_input)
            print(f"[OK] wrote trip report · {drug_name} · track {track_input} · quiz {battery_name}")
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "5":
            # Reserve: model size × severity (multi-model → dispatch).
            track_input = input("Track Profile (default C1): ").strip().upper() or "C1"
            if track_input not in EATEOT_TRACK_PROFILES:
                print("[-] Invalid track profile!")
                continue
            print("\n[+] Reserve study: IQ vs severity across model sizes")
            _run_app("reserve", ["--track", track_input, "--questionnaire", battery_name])
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "6":
            print("\n[+] Model comparison: A1 baseline across 0.5B/1.5B/3B")
            _run_app("compare", ["--questionnaire", battery_name])
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "7":
            print("\n[+] Plotting IQ decay curve from telemetry")
            _run_app("plot", [])
            input("Press ENTER to return to menu...")
            continue
        elif track_choice == "8":
            print("\n[+] Building drug telemetry report")
            _run_app("drugreport", [])
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
                epsilon=args.epsilon,
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
                epsilon=args.epsilon,
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
            print("[-] Invalid selection! Type a track name (A1, C5…), a letter command (I, Q, E, P, G, Z, D, T, F, S, L, W, R, X), or a study number (1–8).")
            continue


if __name__ == "__main__":
    main()
