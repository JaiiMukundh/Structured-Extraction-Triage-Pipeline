import sys
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

if len(sys.argv) != 3:
    print("Usage: python merge_adapters.py <base_model_name> <lora_dir>")
    sys.exit(1)

base_model_name = sys.argv[1]
lora_dir = Path(sys.argv[2])
merged_dir = lora_dir / "merged"

print(f"Loading base model {base_model_name}...")
tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    device_map="cpu",
    torch_dtype=torch.float16
)

print(f"Loading PEFT adapter from {lora_dir}...")
peft_model = PeftModel.from_pretrained(base_model, lora_dir)

print("Merging adapter weights...")
merged_model = peft_model.merge_and_unload()

print(f"Saving merged model to {merged_dir}...")
merged_dir.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

print("Merge complete!")
