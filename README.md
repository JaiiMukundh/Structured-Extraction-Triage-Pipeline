# Supply Chain Event Extraction Pipeline

> **Internship Project · Industry Experience Report Reference Implementation**

A resource-efficient, two-stage pipeline for detecting and extracting structured supply chain disruption events from unstructured text, using small language models (SLMs) with LoRA fine-tuning.

---

## Overview

Enterprise organizations monitor large volumes of news and operational documents to detect supply chain risks. Sending every document to a large generative LLM is computationally expensive. This pipeline solves that with a **two-stage triage architecture**:

1. **Stage 1 — Triage (DistilBERT, 66M params):** Fast CPU-based binary classifier that filters out ~99% of non-event documents.
2. **Stage 2 — Extraction (Qwen2.5-1.5B + LoRA + Outlines):** Structured JSON event extraction, only triggered for documents that pass triage.

### Supported Event Types
| Event Type | Captures |
|---|---|
| `FacilityHalt` | Physical production stoppages (fires, strikes, disasters) |
| `ShipmentDelay` | Transit and carrier delays |
| `SupplierInsolvency` | Bankruptcy and financial restructuring |
| `TariffChange` | Customs duty and trade policy changes |
| `ForceMajeure` | Legal force majeure declarations |

---

## Repository Structure

```
supply-chain-event-detection/
│
├── data/
│   ├── distilbert/          # Binary triage classification datasets
│   ├── qwen/                # Structured extraction datasets (JSONL)
│   └── raw/                 # Master unified dataset (splittable_redo.jsonl)
│
├── schemas/
│   └── extraction_schema.json   # JSON schema governing all extractions
│
├── training/
│   ├── split_dataset.py     # Splits unified dataset for both models (no leakage)
│   ├── train_distilbert.py  # Fine-tunes DistilBERT for binary triage
│   ├── train_qwen_lora.py   # Fine-tunes Qwen2.5-1.5B with LoRA
│   └── merge_lora.py        # Merges LoRA adapter into base model
│
├── models/
│   ├── distilbert/          # Fine-tuned DistilBERT weights (gitignored: .safetensors)
│   └── qwen_lora/           # LoRA adapter + merged model (gitignored: .safetensors)
│
├── inference/
│   ├── triage_pipeline.py   # Main CLI inference script (end-to-end)
│   └── extract_baseline_test.py  # Zero-shot baseline for comparison
│
├── evaluation/
│   ├── compare_extraction.py     # F1 / Precision / Recall / Schema Validity
│   ├── compare_validation.py     # Schema validation only
│   ├── calculate_parameters.py   # LoRA parameter statistics
│   └── evaluate_forgetting.py    # Catastrophic forgetting assessment
│
├── reports/
│   ├── final_report.md           # Full technical report
│   ├── businessGains.md          # Business ROI analysis
│   ├── dataset_refining.md       # Dataset curation strategy
│   ├── annotation_guidelines.md  # Human annotation guidelines
│   └── comparison_metrics.png    # Results visualization
│
├── run_all.py               # End-to-end pipeline runner
└── requirements.txt         # Python dependencies
```

> **Note:** Model weight files (`.safetensors`) are excluded from this repository due to size. Download or train them following the instructions below.

---

## Quickstart

### 1. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare the Dataset
```bash
python training/split_dataset.py
```
This creates aligned train/val/test splits for both DistilBERT and Qwen under `data/`.

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
# Single text input via CLI
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

| Model / Configuration | Precision | Recall | F1-Score | Schema Validity |
|---|---|---|---|---|
| Baseline (Qwen Zero-Shot) | 42.60% | 41.82% | 41.62% | 33.33% |
| Pipeline V1 (r=4 LoRA) | 45.98% | 39.39% | 49.78% | 100.00% |
| Pipeline V2 (Refined Prompt, Underfit) | 34.02% | 27.09% | 27.80% | 100.00% |
| **Pipeline V3 (Optimized r=16)** | **72.90%** | **71.22%** | **71.88%** | **100.00%** |

- **72.7% relative F1 improvement** over zero-shot baseline
- **100% schema validity** via Outlines constrained decoding
- **~99% cost reduction** in production (only ~1% of docs trigger LLM inference)
- Entire pipeline runs in **~7 GB RAM** — no GPU required for inference

---

## LoRA Configuration

| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-1.5B-Instruct` |
| Rank (r) | 16 |
| Alpha (α) | 32 |
| Target Modules | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Trainable Parameters | 18.46M (1.19% of base) |
| Optimizer Memory | ~147.7 MB (vs ~12.3 GB full fine-tuning) |

---

## Citation

> Paper under preparation. BibTeX entry will be added upon publication.

---

## License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
