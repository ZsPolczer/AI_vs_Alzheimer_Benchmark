"""Neural degradation engine: model loading, weight corruption, inference.

Port of ``BrainLabEngine`` from the original ``interactive_lab.py``,
including the fix that keys weight backups by stable ``(layer_index, name)``
tuples (``id(param)`` was unsafe under ``device_map="auto"`` CPU offload).
"""

import copy
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .profiles import EATEOT_TRACK_PROFILES


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
                param.data.mul_(scale)

                if noise_std > 0:
                    noise = torch.randn_like(param.data) * noise_std
                    param.data.add_(noise)

        return prof["prompt"]

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

    def run_inference(self, user_prompt: str, sys_prompt: str, lucidity_surge: bool = False) -> str:
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
