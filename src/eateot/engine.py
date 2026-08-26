"""Neural degradation engine: model loading, weight corruption, inference.

Port of ``BrainLabEngine`` from the original ``interactive_lab.py``,
including the fix that keys weight backups by stable ``(layer_index, name)``
tuples (``id(param)`` was unsafe under ``device_map="auto"`` CPU offload).

Drug support
------------
Both ``apply_degradation`` and ``run_inference`` accept an optional ``drug``
dict — the resolved spec returned by ``eateot.drugs.resolve_drug``
(``{subnetwork, layer_pct, prompt_state, primitives}``). The pure functions
``compute_degradation_params`` / ``compute_sampling_params`` fold the spec
into effective parameters; the engine only consumes the folded values.

Primitive semantics (see ``eateot.drugs`` for the full contract):

* ``noise`` is ADDITIVE to the track baseline (may be negative = suppression).
* ``scale`` is MULTIPLICATIVE with the track scale; 1.0 is a no-op, so a drug
  that does not touch weights never erases track degradation.
* ``flicker_rate`` (drug-induced dropout) and ``sirens_mult`` parametrize the
  previously hardcoded 0.25 / 2.5 values.
* ``noise_gradient`` adds extra noise per layer of depth.
* ``temperature`` / ``repetition_penalty`` / ``verbosity_bias`` fold into the
  generation kwargs (verbosity scales ``max_new_tokens``).
* ``logit_noise`` injects gaussian noise into the logits via a
  ``LogitsProcessor`` (merged with transformers' default processors).
* ``attention_scatter`` adds gaussian noise to the attention OUTPUTS of the
  layers in the drug's window via forward hooks (the output-space effect of
  scattered attention weights).
* ``context_mask_frac`` zeroes the attention mask over the prompt tail
  (anterograde amnesia — the model stops "seeing" recent instructions).
* ``prompt_state`` is appended to the system prompt.
"""

import copy
import random
import sys
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import LogitsProcessor, LogitsProcessorList
from transformers.generation import TextIteratorStreamer

from .profiles import EATEOT_TRACK_PROFILES

# ---------------------------------------------------------------------------
# Base generation constants (drugs perturb these via verbosity_bias etc.)
# ---------------------------------------------------------------------------
BASE_MAX_NEW_TOKENS = 384
BASE_TEMPERATURE = 0.7
TOP_P = 0.9
TOKENS_PER_VERBOSITY_UNIT = 16  # one verbosity_bias unit = 16 extra tokens
MIN_MAX_TOKENS = 32
DEFAULT_FLICKER_RATE = 0.25
DEFAULT_SIRENS_MULT = 2.5
MIN_SCALE = 0.001
SURGE_TEMPERATURE = 0.3


def compute_degradation_params(profile: dict, decay_mult: float = 1.0,
                               target_subnetwork: str = "all",
                               enable_sirens: bool = False,
                               enable_flicker: bool = False,
                               drug: dict | None = None) -> dict:
    """Fold a drug spec into effective degradation parameters (pure).

    ``profile`` is one entry from ``EATEOT_TRACK_PROFILES``; ``drug`` is a
    resolved spec from ``eateot.drugs.resolve_drug`` (or None). Returns a
    dict consumed by ``apply_degradation``:

    *    ``scale``         — track scale (÷decay_mult) × drug scale (1.0 = no-op)
    * ``noise_std``     — track noise (×decay_mult) + drug noise (signed,
                          clamped ≥ 0), then × sirens_mult when sirens on
    * ``epsilon``      — std-scaled Gaussian perturbation strength: adds
                          ε·σ_W·Z noise (σ_W = weight-tensor std, Z ~ N(0,1))
                          — the sensitivity-study method (profile + drug fold)
    * ``sirens_mult``   — drug override of the default 2.5 sirens multiplier
    * ``flicker_rate``  — drug-induced dropout rate, else 0.25 when enabled
    * ``noise_gradient``— extra noise std added per layer of depth
    * ``layer_pct``     — drug layer window if given, else the track's
    * ``subnetwork``    — drug subnetwork if given (and not "all"), else the
                          caller's target_subnetwork
    """
    prims = (drug or {}).get("primitives", {}) if drug else {}

    scale = max(MIN_SCALE, (profile["scale"] / decay_mult) * prims.get("scale", 1.0))
    noise_std = max(0.0, profile["noise"] * decay_mult + prims.get("noise", 0.0))
    # Std-scaled Gaussian perturbation strength (Ẇ = W + ε·σ_W·Z). Sources fold
    # additively; epsilon only acts on the weight-noise path, so scale stays 1.0
    # and noise_std 0.0 for a pure sensitivity sweep.
    epsilon = max(0.0, profile.get("epsilon", 0.0) + prims.get("epsilon", 0.0))

    sirens_mult = prims.get("sirens_mult", DEFAULT_SIRENS_MULT)
    if enable_sirens:
        noise_std *= sirens_mult

    flicker_rate = DEFAULT_FLICKER_RATE if enable_flicker else 0.0
    if prims.get("flicker_rate", 0.0) > 0:
        flicker_rate = prims["flicker_rate"]

    layer_pct = profile["layer_pct"]
    if drug and drug.get("layer_pct"):
        layer_pct = drug["layer_pct"]

    subnetwork = target_subnetwork
    if drug and drug.get("subnetwork") and drug["subnetwork"] != "all":
        subnetwork = drug["subnetwork"]

    return {
        "scale": scale,
        "noise_std": noise_std,
        "epsilon": epsilon,
        "sirens_mult": sirens_mult,
        "flicker_rate": flicker_rate,
        "noise_gradient": prims.get("noise_gradient", 0.0),
        "layer_pct": layer_pct,
        "subnetwork": subnetwork,
    }


def compute_sampling_params(drug: dict | None = None, lucidity_surge: bool = False,
                            base_temperature: float = BASE_TEMPERATURE,
                            base_max_new_tokens: int = BASE_MAX_NEW_TOKENS) -> dict:
    """Fold a drug spec into effective generation parameters (pure).

    Returns ``temperature`` (surge wins over the drug), ``repetition_penalty``,
    ``max_new_tokens`` (scaled by verbosity_bias), and the three
    inference-stage primitives ``logit_noise``, ``attention_scatter`` and
    ``context_mask_frac`` that ``run_inference`` applies to the tensors.
    """
    prims = (drug or {}).get("primitives", {}) if drug else {}
    temperature = SURGE_TEMPERATURE if lucidity_surge else prims.get("temperature", base_temperature)
    repetition_penalty = prims.get("repetition_penalty", 1.0)
    verbosity = prims.get("verbosity_bias", 0.0)
    max_new_tokens = max(MIN_MAX_TOKENS,
                         base_max_new_tokens + int(verbosity * TOKENS_PER_VERBOSITY_UNIT))
    return {
        "temperature": temperature,
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "logit_noise": prims.get("logit_noise", 0.0),
        "attention_scatter": prims.get("attention_scatter", 0.0),
        "context_mask_frac": prims.get("context_mask_frac", 0.0),
    }


class GaussianLogitNoise(LogitsProcessor):
    """Adds gaussian noise to the logits before sampling (per-token "visual snow")."""

    def __init__(self, std: float):
        self.std = std

    def __call__(self, input_ids: torch.Tensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.std <= 0:
            return scores
        return scores + torch.randn_like(scores) * self.std


def _attention_scatter_hook(std: float):
    """Build a forward hook that adds gaussian noise to an attention output.

    Handles both tuple outputs (``(attn_output, attn_weights, past_key_value)``
    — Qwen2/Llama style) and plain tensor outputs.
    """

    def hook(module, args, output):
        if isinstance(output, tuple):
            head = output[0]
            return (head + torch.randn_like(head) * std,) + output[1:]
        return output + torch.randn_like(output) * std

    return hook


class BrainLabEngine:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct"):
        print(f"\n[+] Loading model '{model_name}' into VRAM/RAM...")
        self.model_id = model_name
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
                          enable_flicker=False, enable_sirens=False, noise_seed=None,
                          drug=None, epsilon=0.0):
        """Applies mathematical corruption to weights with full experimental overrides.

        ``drug`` (optional) is a resolved spec from ``eateot.drugs.resolve_drug``;
        see the module docstring for how each primitive folds into the
        degradation. Drug ``subnetwork`` / ``layer_pct`` override the track's
        when given; drug ``scale`` multiplies the track scale (1.0 = no-op);
        drug ``noise`` adds to the track noise (negative = suppression).

        ``epsilon`` (optional, default 0.0) is an explicit std-scaled Gaussian
        perturbation strength — Ẇ = W + ε·σ_W·Z — added on top of the profile/
        drug value (sensitivity-study method). σ_W is each weight tensor's own
        standard deviation, so ε is dimensionless and comparable across layers.

        Pass ``noise_seed`` to make the corruption (and flicker dropout)
        deterministic — repeated calls with the same seed produce the same
        degraded baseline, which is required for dose-response studies where
        every dose must start from an identical lesion.
        """
        self.restore_clean_state()

        prof = copy.deepcopy(EATEOT_TRACK_PROFILES[profile_key])
        params = compute_degradation_params(
            prof, decay_mult=decay_mult, target_subnetwork=target_subnetwork,
            enable_sirens=enable_sirens, enable_flicker=enable_flicker, drug=drug,
        )

        scale = params["scale"]
        noise_std = params["noise_std"]
        epsilon = params["epsilon"] + epsilon
        noise_gradient = params["noise_gradient"]
        flicker_rate = params["flicker_rate"]
        layer_pct = params["layer_pct"]
        target_subnetwork = params["subnetwork"]

        if noise_seed is not None:
            torch.manual_seed(noise_seed)
            random.seed(noise_seed)

        start_idx = int(self.total_layers * layer_pct[0])
        end_idx = int(self.total_layers * layer_pct[1])

        layers = self._get_layers()

        print(f"┌─ [DEGRADATION ENGINE ACTIVATED]")
        print(f"├─ Track: {prof['title']}")
        if drug:
            print(f"├─ Drug: {drug.get('name', '?')} | class: {drug.get('class', '?')} | "
                  f"dose: {drug.get('dose', '?')}")
        print(f"├─ Target Layers: {start_idx} to {end_idx} (out of {self.total_layers})")
        noise_suffix = f" (+{noise_gradient:.6f}/layer)" if noise_gradient > 0 else ""
        sirens_suffix = f" | Sirens ×{params['sirens_mult']:.2f}" if enable_sirens else ""
        epsilon_suffix = f" | ε·σ_W: {epsilon:.4f}" if epsilon > 0 else ""
        print(f"├─ Effective Tensor Scale: {scale:.4f} | Noise StdDev: {noise_std:.6f}"
              f"{noise_suffix}{sirens_suffix}{epsilon_suffix}")
        print(f"├─ Sub-Network Targeting: [{target_subnetwork.upper()}]")
        print(f"└─ Synaptic Health Index: {self._render_health_bar(scale, noise_std)}")

        for i in range(start_idx, end_idx):
            layer = layers[i]

            if flicker_rate > 0 and random.random() < flicker_rate:
                print(f"   [!] SYNAPSE DROPOUT: Layer {i} dropped to zero!")
                for name, p in layer.named_parameters():
                    self.backups[(i, name)] = p.data.clone()
                    p.data.zero_()
                continue

            for name, param in layer.named_parameters():
                if target_subnetwork == "attn" and "attn" not in name:
                    continue
                elif target_subnetwork == "mlp" and "mlp" not in name:
                    continue
                elif target_subnetwork == "norm" and "norm" not in name:
                    continue

                self.backups[(i, name)] = param.data.clone()
                # σ_W of the *clean* tensor (captured before scaling) so ε is
                # relative to the pristine weights, per Ẇ = W + ε·σ_W·Z.
                w_std = param.data.std() if epsilon > 0 else None
                param.data.mul_(scale)

                if noise_std > 0 or noise_gradient > 0:
                    layer_noise = noise_std + (i - start_idx) * noise_gradient
                    if layer_noise > 0:
                        noise = torch.randn_like(param.data) * layer_noise
                        param.data.add_(noise)

                if w_std is not None and w_std > 0:
                    # Std-scaled Gaussian perturbation (sensitivity method).
                    param.data.add_(torch.randn_like(param.data) * epsilon * w_std)

        return prof["prompt"]

    def lerp_toward_clean(self, fraction: float):
        """Interpolate degraded weights toward the clean backup (dose-response).

        ``fraction=0.0`` leaves the model fully degraded (as just applied);
        ``fraction=1.0`` fully restores it. Requires ``apply_degradation`` to
        have run first so the clean backups exist. Call it exactly once per
        fresh ``apply_degradation`` — it interpolates the *current* degraded
        weights toward clean, so calling it twice without re-degrading would
        compound the restore. This lets a study model a partial "treatment
        response" without re-loading weights.
        """
        if not self.backups:
            raise RuntimeError(
                "lerp_toward_clean requires apply_degradation to have run first "
                "(no clean backups available)."
            )
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                key = (i, name)
                if key in self.backups:
                    clean = self.backups[key]
                    param.data.copy_(clean * fraction + param.data * (1.0 - fraction))

    def restore_clean_state(self):
        """Restores original weights in milliseconds.

        Backups are keyed by stable (layer_index, parameter_name) tuples rather
        than by id(param): with device_map="auto" CPU offloading, accelerate can
        swap Parameter objects during inference, so id() values get recycled and
        could match the wrong tensor (e.g. k_proj vs gate_proj) on restore.
        """
        if not self.backups:
            return
        layers = self._get_layers()
        for i, layer in enumerate(layers):
            for name, param in layer.named_parameters():
                key = (i, name)
                if key in self.backups:
                    param.data.copy_(self.backups[key])
        self.backups.clear()

    def _render_health_bar(self, scale, noise):
        health = int(min(100, max(0, (scale * 100) - (noise * 5000))))
        filled = int(health / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {health}%"

    def _attach_attention_scatter(self, std: float, start_idx: int, end_idx: int):
        """Register forward hooks that noise the attention outputs of a layer window.

        Returns the list of hook handles; callers MUST remove them (e.g. in a
        ``finally`` block) so later clean runs are unaffected.
        """
        handles = []
        layers = self._get_layers()
        for i in range(start_idx, min(end_idx, len(layers))):
            # Qwen2/Llama: self_attn; Mistral-family variants: attention;
            # GPT2-style: attn.
            attn = (getattr(layers[i], "self_attn", None)
                    or getattr(layers[i], "attention", None)
                    or getattr(layers[i], "attn", None))
            if attn is None:
                continue
            handles.append(attn.register_forward_hook(_attention_scatter_hook(std)))
        return handles

    def _generate(self, inputs, gen_kwargs, seed):
        """Run model.generate in a background thread for streaming.

        This is the thread target for TextIteratorStreamer: it produces tokens
        one at a time which the main thread reads from the streamer iterator.
        """
        with torch.no_grad():
            if seed is not None:
                torch.manual_seed(seed)
            self.model.generate(**inputs, **gen_kwargs)

    def run_inference(self, user_prompt: str, sys_prompt: str, lucidity_surge: bool = False,
                      seed: int | None = None, drug: dict | None = None) -> str:
        """Run one generation under the current weights, honoring a drug spec.

        ``drug`` (optional) is a resolved spec from ``eateot.drugs.resolve_drug``.
        The drug's ``prompt_state`` is appended to the system prompt, its
        ``context_mask_frac`` blanks the prompt tail, its sampling primitives
        override the generation kwargs, ``logit_noise`` scrambles the logits,
        and ``attention_scatter`` noises the attention outputs of its layer
        window (hooks are removed before returning).
        """
        response_text = ""
        params = compute_sampling_params(drug=drug, lucidity_surge=lucidity_surge)

        active_sys_prompt = sys_prompt
        if drug and drug.get("prompt_state"):
            active_sys_prompt += "\n" + drug["prompt_state"]
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

        # Anterograde amnesia: zero the attention mask over the prompt tail so
        # the model no longer "sees" the most recent instructions (the
        # structural positions are kept, so generation stays well-formed).
        context_mask_frac = params["context_mask_frac"]
        if context_mask_frac > 0:
            seq_len = inputs.input_ids.shape[1]
            mask_len = min(int(seq_len * context_mask_frac), max(0, seq_len - 2))
            if mask_len > 0:
                inputs["attention_mask"][0, seq_len - mask_len:] = 0

        gen_kwargs = {
            "max_new_tokens": params["max_new_tokens"],
            "do_sample": True,
            "temperature": params["temperature"],
            "top_p": TOP_P,
            "repetition_penalty": params["repetition_penalty"],
            "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

        if params["logit_noise"] > 0:
            # Merged with transformers' default processors (temperature,
            # top_p, repetition_penalty) — see _get_logits_processor.
            gen_kwargs["logits_processor"] = LogitsProcessorList(
                [GaussianLogitNoise(params["logit_noise"])]
            )

        # Attention scatter: noise the attention outputs of the drug's layer
        # window. Hooks are removed in the finally block below.
        scatter_handles = []
        if params["attention_scatter"] > 0:
            layer_pct = (drug or {}).get("layer_pct") or [0.0, 1.0]
            start_idx = int(self.total_layers * layer_pct[0])
            end_idx = int(self.total_layers * layer_pct[1])
            scatter_handles = self._attach_attention_scatter(
                params["attention_scatter"], start_idx, end_idx
            )

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_special_tokens=True, skip_prompt=True
        )
        gen_kwargs["streamer"] = streamer

        try:
            print("\n=== AI RESPONSE BEGINS ===")
            sys.stdout.flush()
            try:
                thread = threading.Thread(
                    target=self._generate,
                    args=(inputs, gen_kwargs, seed),
                )
                thread.start()
                collected = []
                for token_text in streamer:
                    sys.stdout.write(token_text)
                    sys.stdout.flush()
                    collected.append(token_text)
                thread.join()
                response_text = ("".join(collected)).strip()
            except Exception as e:
                response_text = f"[INFERENCE ERROR: {str(e)}]"
            print("\n=== AI RESPONSE ENDS ===\n")
        finally:
            for handle in scatter_handles:
                handle.remove()

        if not response_text:
            response_text = "[NO OUTPUT GENERATED]"

        return response_text
