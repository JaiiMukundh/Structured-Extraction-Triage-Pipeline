# Supply Chain Event Extraction Pipeline

> **Internship Project · Industry Experience Report Reference Implementation**

A resource-efficient, two-stage pipeline for detecting and extracting structured supply chain disruption events from unstructured text, using small language models (SLMs) with LoRA fine-tuning.

---

## Overview

Enterprise organizations monitor large volumes of news and operational documents to detect supply chain risks. Sending every document to a large generative LLM is computationally expensive. This pipeline solves that with a **two-stage triage architecture**:

1. **Stage 1 — Triage (DistilBERT, 66M params):** Fast CPU-based binary classifier that routes out non-event documents in ~15 ms per chunk.
2. **Stage 2 — Extraction (Qwen2.5-1.5B + LoRA + Outlines):** Structured JSON event extraction, only triggered for documents that pass triage (~8 s per chunk on CPU).

### Supported Event Types

| Event Type | Captures |
|---|---|
| `FacilityHalt` | Physical production stoppages (fires, strikes, disasters, cyberattacks) |
| `ShipmentDelay` | Transit and carrier delays |
| `SupplierInsolvency` | Bankruptcy, liquidation, and financial restructuring |
| `TariffChange` | Customs duty and trade policy changes |
| `ForceMajeure` | Legal force majeure declarations |

---

## Repository Structure

```
Triage_Pipeline/
│
├── data/
│   ├── distilbert/          # Binary triage classification datasets
│   │   ├── distilbert_base.jsonl   (196 rows: 125 pos / 71 neg)
│   │   ├── distilbert_train.jsonl  (137 rows: 83 pos / 54 neg)
│   │   ├── distilbert_val.jsonl    (29 rows)
│   │   └── distilbert_test.jsonl   (30 rows)
│   │
│   ├── qwen/                # Structured extraction datasets (positives only)
│   │   ├── qwen_base.jsonl         (175 rows, balanced ~35/class)
│   │   ├── qwen_train.jsonl        (115 rows)
│   │   ├── qwen_val.jsonl          (30 rows, 6/class)
│   │   └── qwen_test.jsonl         (30 rows, 6/class)
│   │
│   ├── raw/
│   │   └── splittable_redo.jsonl   # Master unified dataset (351 rows: 280 pos / 71 neg)
│
├── schemas/
│   └── extraction_schema.json   # JSON Schema governing all extractions
│
├── training/
│   ├── split_dataset.py     # Reads splittable_redo.jsonl → outputs to data/
│   ├── train_distilbert.py  # Fine-tunes DistilBERT for binary triage
│   ├── train_qwen_lora.py   # Fine-tunes Qwen2.5-1.5B with LoRA
│   └── merge_lora.py        # Merges LoRA adapter into base model
│
├── models/
│   ├── distilbert/          # Fine-tuned DistilBERT weights (gitignored: .safetensors)
│   ├── qwen_lora/           # LoRA adapter + merged model (gitignored: .safetensors)
│   ├── SmolLM2-1.7B-Instruct_lora/  # Comparison model (gitignored: .safetensors)
│   └── TinyLlama-1.1B-Chat-v1.0_lora/  # Comparison model (gitignored: .safetensors)
│
├── inference/
│   ├── triage_pipeline.py         # Main CLI inference script (end-to-end)
│   └── extract_baseline_test.py   # Zero-shot baseline for comparison
│
├── evaluation/
│   ├── compare_extraction.py      # F1 / Precision / Recall / Schema Validity
│   ├── compare_validation.py      # Schema validation only
│   ├── evaluate_distilbert.py     # DistilBERT classifier evaluation
│   ├── calculate_parameters.py    # LoRA parameter statistics
│   └── evaluate_forgetting.py     # Catastrophic forgetting assessment
│

├── reports/
│   ├── structured_outputs.jsonl        # Pipeline predictions (30 test samples)
│   ├── baseline_structured_outputs.jsonl  # Zero-shot baseline predictions
│   ├── SmolLM2-1.7B-Instruct_pipeline.jsonl
│   ├── TinyLlama-1.1B-Chat-v1.0_pipeline.jsonl
│   ├── comparison_metrics.csv          # Aggregated model comparison results
│   ├── comparison_metrics.png          # Results bar chart
│   └── lora_heatmaps.png               # Layer-wise LoRA weight norm heatmap
│
├── Industry_Experience_Report.md  # Full technical report
├── run_all.py                     # End-to-end pipeline runner
├── requirements.txt               # Python dependencies
└── NeededInfo.txt                 # Project reference notes
```

> **Note:** Model weight files (`.safetensors`) are excluded from this repository due to size. Train them following the instructions below, or download from the companion release.

---

## Dataset

The unified master dataset is `data/raw/splittable_redo.jsonl`. It is the single source of truth that feeds both model training stages.

| Stage | Total | Positive | Negative | Positive Rate |
|---|---|---|---|---|
| Raw master (`splittable_redo.jsonl`) | **351** | **280** | **71** | 79.8% |
| DistilBERT base | 196 | 125 | 71 | 63.8% |
| DistilBERT train | 137 | 83 | 54 | 60.6% |
| DistilBERT val | 29 | 24 | 5 | 82.8% |
| DistilBERT test | 30 | 18 | 12 | 60.0% |
| Qwen base | 175 | 175 | — | 100% |
| Qwen train | 115 | 115 | — | 100% |
| Qwen val | 30 | 30 | — | 100% |
| Qwen test | 30 | 30 | — | 100% |

**Qwen base event type distribution** (near-uniform, 35 per class):

| Event Type | Count | % |
|---|---|---|
| FacilityHalt | 36 | 20.6% |
| ShipmentDelay | 36 | 20.6% |
| SupplierInsolvency | 35 | 20.0% |
| TariffChange | 35 | 20.0% |
| ForceMajeure | 33 | 18.9% |

---

## Quickstart

### 1. Install Dependencies
> **Requirement:** Python 3.11 is required.

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
```bash
python training/split_dataset.py
```
Reads `data/raw/splittable_redo.jsonl` and writes stratified train/val/test splits for both models into `data/`.

### 3. Train the Models
```bash
# Stage 1: DistilBERT triage classifier
python training/train_distilbert.py

# Stage 2: Qwen2.5-1.5B LoRA fine-tuning
python training/train_qwen_lora.py

# Merge LoRA adapter into base model for inference
python training/merge_lora.py
```

### 4. Run Inference
```bash
python inference/triage_pipeline.py "A major port strike has halted container shipments at Rotterdam for 3 days."
```

**Output:**
```json
{
  "event_type": "ShipmentDelay",
  "source_timestamp": null,
  "text_evidence": "halted container shipments at Rotterdam for 3 days",
  "arguments": {
    "carrier": null,
    "origin": "Rotterdam",
    "destination": null,
    "delay_duration_days": 3,
    "reason": "major port strike"
  },
  "event_id": "EVT-XXXXXXXX"
}
```

### 5. Evaluate
```bash
python evaluation/compare_extraction.py
```

---

## Key Results

| Model / Configuration | Precision | Recall | F1-Score | Top-Level F1 | Arguments F1 | Schema Validity |
|---|---|---|---|---|---|---|
| Baseline (Qwen Zero-Shot) | 46.96% | 47.65% | 46.81% | 58.40% | 39.28% | 23.33% |
| **Qwen2.5-1.5B (LoRA)** | **72.51%** | **67.61%** | **69.24%** | **83.97%** | **59.23%** | **100.00%** |
| SmolLM2-1.7B (LoRA) | 66.20% | 57.01% | 60.21% | 73.19% | 50.06% | 100.00% |
| TinyLlama-1.1B (LoRA) | 53.28% | 50.28% | 50.58% | 52.15% | 49.01% | 100.00% |

- **47.9% relative F1 improvement** over zero-shot baseline (46.81% → 69.24%)
- **100% schema validity** via Outlines constrained decoding (vs. 23.33% baseline)
- **~92–99% cost reduction** in production (only 20–64% of docs trigger LLM inference depending on document stream)
- Entire pipeline runs in **~1.64 GB RAM** on CPU — no GPU required for inference

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable Parameters | 18.46M (1.196% of base) |
| Optimizer Memory | ~147.7 MB (vs ~12.3 GB full fine-tuning) |
| Adapter File Size | 70.5 MB |

---

## Training Progression (Qwen2.5-1.5B LoRA)

| Epoch | Train Loss | Top-Level F1 (Val) | Arguments F1 (Val) |
|---|---|---|---|
| 1 | 0.1923 | 34.21% | 36.38% |
| 2 | 0.0492 | 52.05% | 37.56% |
| 3 | ~0.0114 | ~83.97% | ~59.23% |

---

## Citation

> Paper under preparation. BibTeX entry will be added upon publication.

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
