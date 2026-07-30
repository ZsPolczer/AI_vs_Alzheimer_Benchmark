import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Setup Model
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Define the Degeneration Function
def apply_cognitive_damage(model, layer_index, damage_percentage):
    print(f"\n[!] Inducing {damage_percentage*100:.0f}% synaptic loss in Layer {layer_index}...")
    with torch.no_grad():
        weights_to_damage = [
            model.model.layers[layer_index].mlp.gate_proj.weight,
            model.model.layers[layer_index].mlp.up_proj.weight,
            model.model.layers[layer_index].mlp.down_proj.weight
        ]
        for weight in weights_to_damage:
            survival_rate = 1.0 - damage_percentage
            mask = (torch.rand(weight.shape, device=weight.device) < survival_rate).to(torch.float16)
            weight.data.mul_(mask)

# 3. Your Mini IQ Test Benchmark
benchmark_data = [
    {
        "question": "If Circle = 1, Square = 2, and Triangle = 3, what is Circle + Triangle?",
        "expected": "4"
    },
    {
        "question": "What comes next in the sequence: 2, 4, 8, 16, ?",
        "expected": "32"
    }
]

def run_benchmark(model, tokenizer, data):
    # Strict prompt forcing extreme brevity
    system_prompt = (
        "You are taking a test. Be extremely brief and direct. "
        "Explain your logic in 1 short sentence, then state your final answer. "
        "Keep your entire output under 25 words."
    )
    
    for i, item in enumerate(data):
        q = item["question"]
        expected = item["expected"]
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # Raised max_new_tokens to 200 so damaged rambling doesn't truncate mid-word
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200, 
            pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("user\n" + q)[-1].strip()
        
        # Format output cleanly in terminal
        print(f"\nQuestion {i+1}: {q}")
        print(f"  Expected Answer: {expected}")
        print(f"  AI Output:")
        for line in response.split('\n'):
            if line.strip():
                print(f"    {line.strip()}")
    print("-" * 60)

# ==========================================
# THE EXPERIMENT
# ==========================================

print("\n\n=== BASELINE (HEALTHY BRAIN) ===")
run_benchmark(model, tokenizer, benchmark_data)

apply_cognitive_damage(model, layer_index=12, damage_percentage=0.30)
print("\n=== TEST 1 (30% DAMAGE) ===")
run_benchmark(model, tokenizer, benchmark_data)

apply_cognitive_damage(model, layer_index=12, damage_percentage=0.60)
print("\n=== TEST 2 (SEVERE DAMAGE) ===")
run_benchmark(model, tokenizer, benchmark_data)