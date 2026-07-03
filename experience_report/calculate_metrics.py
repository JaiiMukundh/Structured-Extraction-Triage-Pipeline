import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Setup paths
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data"
RAW_FILE = DATA_DIR / "raw" / "splittable_redo.jsonl"
DISTILBERT_DIR = DATA_DIR / "distilbert"
QWEN_DIR = DATA_DIR / "qwen"
REPORTS_DIR = BASE / "experience_report"
REPORTS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("EXPERIENCE REPORT METRICS GENERATION SUITE")
print("=" * 80)

# ==============================================================================
# SECTION 1: DATASET QUANTIFICATION
# ==============================================================================
print("\n[1/4] Quantifying Datasets...")

raw_samples = []
if RAW_FILE.exists():
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_samples.append(json.loads(line))
else:
    print(f"Warning: Raw file not found at {RAW_FILE}")

# Calculate Raw Positives / Negatives
total_raw = len(raw_samples)
pos_raw = sum(1 for s in raw_samples if s.get("has_disruption", 0) == 1)
neg_raw = total_raw - pos_raw

# Class Distribution before balancing (Raw file)
raw_class_dist = {}
for s in raw_samples:
    if s.get("has_disruption", 0) == 1 and "event_data" in s and s["event_data"]:
        etype = s["event_data"].get("event_type", "Unknown")
        raw_class_dist[etype] = raw_class_dist.get(etype, 0) + 1

# Load Base files for before/after split statistics
def count_splits(split_name):
    dist_file = DISTILBERT_DIR / f"distilbert_{split_name}.jsonl"
    qwen_file = QWEN_DIR / f"qwen_{split_name}.jsonl"
    
    dist_len, qwen_len = 0, 0
    positives, negatives = 0, 0
    class_dist = {}
    
    if dist_file.exists():
        with open(dist_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    dist_len += 1
                    data = json.loads(line)
                    if data.get("label") == 1:
                        positives += 1
                    else:
                        negatives += 1
                        
    if qwen_file.exists():
        with open(qwen_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    qwen_len += 1
                    data = json.loads(line)
                    event = data.get("event_data")
                    if event:
                        etype = event.get("event_type", "Unknown")
                        class_dist[etype] = class_dist.get(etype, 0) + 1
                        
    return dist_len, positives, negatives, class_dist

train_len, train_pos, train_neg, train_dist = count_splits("train")
val_len, val_pos, val_neg, val_dist = count_splits("val")
test_len, test_pos, test_neg, test_dist = count_splits("test")

# Save dataset summary
dataset_metrics = {
    "raw_total": total_raw,
    "raw_positives": pos_raw,
    "raw_negatives": neg_raw,
    "raw_pos_ratio": pos_raw / max(1, total_raw),
    "raw_class_distribution": raw_class_dist,
    "splits": {
        "train": {"total": train_len, "positives": train_pos, "negatives": train_neg, "class_distribution": train_dist},
        "val": {"total": val_len, "positives": val_pos, "negatives": val_neg, "class_distribution": val_dist},
        "test": {"total": test_len, "positives": test_pos, "negatives": test_neg, "class_distribution": test_dist}
    }
}

print(f"Raw dataset total: {total_raw} (Positives: {pos_raw}, Negatives: {neg_raw})")
print(f"Train size: {train_len} | Val size: {val_len} | Test size: {test_len}")
with open(REPORTS_DIR / "dataset_metrics.json", "w") as f:
    json.dump(dataset_metrics, f, indent=2)

# ==============================================================================
# SECTION 2: SYSTEM PERFORMANCE BENCHMARKS (LATENCY, MEMORY, THROUGHPUT)
# ==============================================================================
print("\n[2/4] Benchmarking Loading and Inference performance...")

import psutil
def get_current_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2) # in MB

# Initialize benchmark metrics
sys_metrics = {}

# Test Loading base Qwen and loading adapter
from transformers import AutoTokenizer, AutoModelForCausalLM
BASE_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_MERGED_DIR = BASE / "models" / "qwen_lora" / "merged"
DISTILBERT_DIR_M = BASE / "models" / "distilbert"

# Measure loading base model
ram_before = get_current_ram()
t0 = time.time()
print("Loading tokenizer and base Qwen model...")
base_tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
t_base_load = time.time() - t0
ram_after_base = get_current_ram()
ram_base_used = ram_after_base - ram_before
print(f"Base model loaded in {t_base_load:.2f}s (RAM used: {ram_base_used:.2f} MB)")

# Measure PEFT loading time (simulating adapter loading)
from peft import PeftModel
t0 = time.time()
print("Loading LoRA adapter weights...")
peft_model = PeftModel.from_pretrained(base_model, BASE / "models" / "qwen_lora")
t_adapter_load = time.time() - t0
ram_after_peft = get_current_ram()
ram_peft_used = ram_after_peft - ram_after_base
print(f"LoRA adapter loaded in {t_adapter_load:.2f}s (RAM used: {ram_peft_used:.2f} MB)")

# Unload to save memory, or reuse for perplexity
# Let's keep it to compute perplexity later

# Loading triage model (DistilBERT)
from transformers import AutoModelForSequenceClassification
t0 = time.time()
print("Loading DistilBERT triage model...")
triage_tok = AutoTokenizer.from_pretrained(DISTILBERT_DIR_M)
triage_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR_M)
if torch.cuda.is_available():
    triage_model.to("cuda")
triage_model.eval()
t_triage_load = time.time() - t0
ram_after_triage = get_current_ram()
print(f"Triage model loaded in {t_triage_load:.2f}s")

# Let's benchmark Qwen inference on a sample
test_text = (
    "A major storm has damaged power facilities at Austin fab of NXP Semiconductors "
    "on February 15, 2021, forcing an immediate halt to all manufacturing activities."
)

# Benchmark Baseline (Qwen Instruct)
print("Benchmarking Baseline generation...")
device = next(base_model.parameters()).device
prompt_baseline = f"Extract supply chain event matching schema:\nText: {test_text}\nJSON Output:"
inputs = base_tok(prompt_baseline, return_tensors="pt").to(device)

t0 = time.time()
with torch.no_grad():
    generated_ids = base_model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=base_tok.eos_token_id,
    )
lat_baseline = time.time() - t0
tokens_generated_base = len(generated_ids[0]) - len(inputs["input_ids"][0])
speed_baseline = tokens_generated_base / lat_baseline
print(f"Baseline Latency: {lat_baseline:.2f}s ({speed_baseline:.2f} tok/s)")

# Benchmark Pipeline Qwen (with PEFT LoRA adapter loaded)
print("Benchmarking PEFT model generation...")
peft_model.eval()
prompt_pipeline = f"Extract supply chain event matching schema:\nText: {test_text}\nJSON Output:"
inputs_peft = base_tok(prompt_pipeline, return_tensors="pt").to(device)

t0 = time.time()
with torch.no_grad():
    generated_ids_peft = peft_model.generate(
        **inputs_peft,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=base_tok.eos_token_id,
    )
lat_pipeline = time.time() - t0
tokens_generated_peft = len(generated_ids_peft[0]) - len(inputs_peft["input_ids"][0])
speed_pipeline = tokens_generated_peft / lat_pipeline
print(f"Pipeline LoRA Latency: {lat_pipeline:.2f}s ({speed_pipeline:.2f} tok/s)")

# Save system metrics
sys_metrics = {
    "loading": {
        "base_model_loading_seconds": t_base_load,
        "adapter_loading_seconds": t_adapter_load,
        "triage_loading_seconds": t_triage_load,
        "base_model_ram_mb": ram_base_used,
        "adapter_ram_mb": ram_peft_used,
        "peak_total_ram_mb": get_current_ram()
    },
    "inference": {
        "baseline": {
            "latency_seconds": lat_baseline,
            "tokens_generated": int(tokens_generated_base),
            "tokens_per_second": speed_baseline
        },
        "pipeline_lora": {
            "latency_seconds": lat_pipeline,
            "tokens_generated": int(tokens_generated_peft),
            "tokens_per_second": speed_pipeline
        }
    }
}
with open(REPORTS_DIR / "system_performance_metrics.json", "w") as f:
    json.dump(sys_metrics, f, indent=2)

# ==============================================================================
# SECTION 3: PERPLEXITY ANALYSIS
# ==============================================================================
print("\n[3/4] Running Perplexity Analysis...")

# Define General vs. Supply Chain Corpora
general_corpus = (
    "The history of science is the study of the development of science, "
    "including both the natural and social sciences. The history of the arts and "
    "humanities is termed the history of scholarship. Science is a body of empirical, "
    "theoretical, and practical knowledge about the universe, produced by scientists "
    "who emphasize the observation, explanation, and prediction of real-world phenomena. "
    "In contrast, art is a creative expression of ideas, feelings, and emotions."
)

supply_chain_corpus = (
    "A massive natural disaster disrupted semiconductor fabrication plants, "
    "halting silicon wafer manufacturing and creating severe bottlenecks in the automotive "
    "supply chain. Logistics delays at major port terminals occurred due to strikes and labor disputes, "
    "forcing carriers to reroute container shipments and declare force majeure."
)

def get_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return np.exp(loss.item())

# Perplexity of BASE model
ppl_base_gen = get_perplexity(base_model, base_tok, general_corpus)
ppl_base_sc = get_perplexity(base_model, base_tok, supply_chain_corpus)

# Perplexity of PEFT LoRA model
ppl_lora_gen = get_perplexity(peft_model, base_tok, general_corpus)
ppl_lora_sc = get_perplexity(peft_model, base_tok, supply_chain_corpus)

perplexity_metrics = {
    "base_model": {
        "general_perplexity": ppl_base_gen,
        "supply_chain_perplexity": ppl_base_sc
    },
    "lora_model": {
        "general_perplexity": ppl_lora_gen,
        "supply_chain_perplexity": ppl_lora_sc
    }
}

print(f"Base Model Perplexity - General: {ppl_base_gen:.2f} | Supply Chain: {ppl_base_sc:.2f}")
print(f"LoRA Model Perplexity - General: {ppl_lora_gen:.2f} | Supply Chain: {ppl_lora_sc:.2f}")
with open(REPORTS_DIR / "perplexity_metrics.json", "w") as f:
    json.dump(perplexity_metrics, f, indent=2)

# ==============================================================================
# SECTION 4: LoRA WEIGHT ANALYSIS
# ==============================================================================
print("\n[4/4] Analyzing LoRA Weight Norms...")

from safetensors.torch import load_file
adapter_file = BASE / "models" / "qwen_lora" / "adapter_model.safetensors"

lora_metrics = {}
if adapter_file.exists():
    weights = load_file(adapter_file)
    
    # Store LoRA weight metrics
    layers_data = {}
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    module_norms = {m: [] for m in target_modules}
    
    # LoRA parameters details
    lora_alpha = 32
    lora_rank = 16
    scaling = lora_alpha / lora_rank
    
    pattern = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.self_attn\.(\w+)\.lora_A\.weight")
    pattern_mlp = re.compile(r"base_model\.model\.model\.layers\.(\d+)\.mlp\.(\w+)\.lora_A\.weight")
    
    # Calculate weight changes delta W = (B * A) * scaling
    for key, val in weights.items():
        # Look for lora_A key to match pairs
        layer_idx = None
        module_name = None
        is_attn = True
        
        match_attn = pattern.match(key)
        match_mlp = pattern_mlp.match(key)
        
        if match_attn:
            layer_idx = int(match_attn.group(1))
            module_name = match_attn.group(2)
        elif match_mlp:
            layer_idx = int(match_mlp.group(1))
            module_name = match_mlp.group(2)
            is_attn = False
            
        if layer_idx is not None:
            # Construct corresponding lora_B key
            parent_path = f"base_model.model.model.layers.{layer_idx}." + ("self_attn" if is_attn else "mlp") + f".{module_name}"
            key_A = f"{parent_path}.lora_A.weight"
            key_B = f"{parent_path}.lora_B.weight"
            
            if key_A in weights and key_B in weights:
                w_A = weights[key_A].to(torch.float32)
                w_B = weights[key_B].to(torch.float32)
                
                # Delta W = (B * A) * scaling
                delta_W = torch.matmul(w_B, w_A) * scaling
                
                # Compute Frobenius norms
                norm_A = torch.norm(w_A).item()
                norm_B = torch.norm(w_B).item()
                norm_delta = torch.norm(delta_W).item()
                
                if layer_idx not in layers_data:
                    layers_data[layer_idx] = {}
                layers_data[layer_idx][module_name] = {
                    "norm_A": norm_A,
                    "norm_B": norm_B,
                    "norm_delta": norm_delta
                }
                module_norms[module_name].append(norm_delta)
                
    # Aggregate metrics
    mean_module_norms = {k: float(np.mean(v)) if v else 0.0 for k, v in module_norms.items()}
    layer_total_norms = {
        layer: float(np.mean([m_data["norm_delta"] for m_data in modules.values()]))
        for layer, modules in layers_data.items()
    }
    
    # Sort layers by largest norm change
    sorted_layers = sorted(layer_total_norms.items(), key=lambda item: item[1], reverse=True)
    
    lora_metrics = {
        "mean_norms_per_module": mean_module_norms,
        "sorted_layers_by_magnitude": sorted_layers[:10], # Top 10 most changed layers
        "layer_total_norms": layer_total_norms,
        "detail": layers_data
    }
    
    print("\nTop 5 modules with largest changes:")
    for mod, val in sorted(mean_module_norms.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {mod}: {val:.4f}")
        
    print("\nTop 5 transformer layers with largest changes:")
    for lay, val in sorted_layers[:5]:
        print(f"  Layer {lay}: {val:.4f}")
else:
    print(f"Warning: Adapter file not found at {adapter_file}")

with open(REPORTS_DIR / "lora_weight_metrics.json", "w") as f:
    json.dump(lora_metrics, f, indent=2)

print("\n" + "=" * 80)
print("ALL METRICS GENERATED SUCCESSFULLY!")
print(f"Saved reports under: {REPORTS_DIR}")
print("=" * 80)
