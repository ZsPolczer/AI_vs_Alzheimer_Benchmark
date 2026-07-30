import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Download and load the AI model and its vocabulary (tokenizer)
model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading {model_id}... (This may take a minute the first time to download)")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, # Uses less memory
    device_map="auto"          # Auto-assigns to GPU if available, else CPU
)

# 2. Your IQ Benchmark Question
prompt = "If Circle = 1, Square = 2, and Triangle = 3, what is Circle + Triangle? Think step by step."

# 3. Format the question so the AI understands it's a conversation
messages = [
    {"role": "system", "content": "You are a logical AI assistant taking an IQ test."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 4. Generate the answer
print("\nThinking...")
outputs = model.generate(**inputs, max_new_tokens=50)

# 5. Decode and print the result
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\n=== AI RESPONSE ===")
# We split the string to only show the AI's actual answer, not the prompt
print(response.split("system\nYou are a logical AI assistant taking an IQ test.\nuser\n" + prompt)[-1].strip())