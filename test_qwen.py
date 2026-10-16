import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "Free VRAM GiB:",
    round(
        (torch.cuda.get_device_properties(0).total_memory
         - torch.cuda.memory_allocated(0)) / 1024**3,
        2,
    ),
)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Loading Qwen 7B in 4-bit mode...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
model.eval()

messages = [
    {
        "role": "system",
        "content": "Answer accurately and concisely.",
    },
    {
        "role": "user",
        "content": (
            "What is retrieval-augmented generation? "
            "Answer in two sentences."
        ),
    },
]

# Convert chat messages into the model's expected input format.
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to("cuda")

print("Input tensor shape:", inputs["input_ids"].shape)
print("Generating...")

torch.cuda.reset_peak_memory_stats()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode only newly generated tokens, not the input prompt.
prompt_length = inputs["input_ids"].shape[1]
generated_tokens = output[0, prompt_length:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True,
)

print("\nMODEL RESPONSE:")
print(response)

print(
    "Peak GPU memory GiB:",
    round(torch.cuda.max_memory_allocated(0) / 1024**3, 2),
)
print("INFERENCE TEST PASSED")