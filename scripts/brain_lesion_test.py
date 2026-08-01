import sys
from pathlib import Path

# Make src/ importable when running from source without installing.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eateot.questionnaires import load_simple_questions

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading healthy model...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==========================================
# RESEARCH PHASE: INJECTING THE LESION
# ==========================================
# Qwen2.5-0.5B has 24 layers (indexed 0 to 23). 
# The middle layers are usually responsible for complex reasoning.
target_layer = 12 

print(f"\n[!] Applying neural lesion to Layer {target_layer}...")

# We use torch.no_grad() because we are forcing changes, not training the model.
with torch.no_grad():
    # We target the MLP (Feed Forward) connections in this specific layer.
    # .fill_(0) permanently overwrites all the connection weights to 0.
    model.model.layers[target_layer].mlp.gate_proj.weight.data.fill_(0)
    model.model.layers[target_layer].mlp.up_proj.weight.data.fill_(0)
    model.model.layers[target_layer].mlp.down_proj.weight.data.fill_(0)

print("[!] Layer severed. Testing cognitive function...\n")
# ==========================================

# The IQ Benchmark Question (loaded from versioned YAML config)
question = load_simple_questions("brain_benchmark")[0]["question"]
prompt = f"{question} Think step by step."

messages = [
    {"role": "system", "content": "You are a logical AI assistant taking an IQ test."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

print("Thinking (with damaged neural pathways)...")
outputs = model.generate(**inputs, max_new_tokens=50)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer_only = response.split("system\nYou are a logical AI assistant taking an IQ test.\nuser\n" + prompt)[-1].strip()

print("\n=== AI RESPONSE (POST-LESION) ===")
print(answer_only)