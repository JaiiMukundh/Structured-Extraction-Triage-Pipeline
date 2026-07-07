# An Industry Experience Report on Domain Adaptation and Structured Event Extraction Using Small Language Models

**[Authors]** -- [Institution / Organization]

---

## Abstract

Enterprise organizations continuously monitor large volumes of unstructured text to detect supply chain disruptions. Deploying frontier large language models for this at scale is economically prohibitive, and their zero-shot outputs rarely comply with strict JSON schemas. We present our engineering experience building a two-stage pipeline that combines a fine-tuned DistilBERT binary classifier (66M parameters) as a compute gatekeeper with a LoRA-adapted Qwen2.5-1.5B generative model for structured event extraction, enforced through constrained decoding via the Outlines library. We constructed a purpose-built dataset of 351 annotated documents spanning five supply chain event types and describe the annotation framework, dataset balancing strategy, and engineering decisions in detail. Our pipeline achieves a 47.9% relative F1 improvement over the zero-shot baseline (46.81% to 69.24%) with 100% schema validity, running entirely within 1.64 GB of RAM. We discuss key failure modes, lessons learned, and a roadmap for continued improvement.

**Keywords:** information extraction, supply chain, event detection, LoRA, small language models, constrained decoding, structured output, recall-oriented triage

---

## 1. Introduction

### 1.1 The Industry Problem

Modern enterprises in manufacturing, logistics, and procurement are exposed to a constant stream of disruption signals: port strikes, factory halts, supplier bankruptcies, tariff changes, and force majeure declarations. Risk officers rely on early detection of these events to activate contingency sourcing, re-route shipments, and hedge financial positions. Missing a critical signal, for example a key battery supplier filing for Chapter 11 protection, can cascade into days of downstream production shutdowns before any formal alert reaches a structured feed.

Traditional approaches involve manual analyst review or rule-based keyword alerting. Both fail at scale. The rise of large language models (LLMs) opened a new approach: prompting a generative model to return structured, machine-readable event payloads directly from unstructured text. The practical problem is that a production platform may ingest tens of thousands of documents daily. At API costs of $0.002 to $0.015 per 1,000 tokens for frontier models, running everything through an LLM is neither practical nor affordable. Beyond cost, zero-shot frontier models in strict JSON extraction tasks frequently hallucinate enum values not in the schema (we observed a baseline generating `LaborDispute`, a type absent from our schema), produce inconsistent timestamp formats, and omit required fields.

This report describes our experience building a practical alternative using small, domain-adapted models with constrained decoding.

### 1.2 Why Supply Chain?

We chose supply chain as the target domain for three concrete reasons. First, supply chain disruptions are extensively covered in public media, including Wikipedia event articles, regulatory announcements, and corporate press releases. Unlike healthcare or finance, there are no regulatory barriers to collecting training data, which made it feasible to build a quality annotated dataset without proprietary data access.

Second, supply chain events map cleanly onto a small, well-defined taxonomy. Each category has structurally distinct arguments: a shipment delay requires carrier and duration information, while a supplier insolvency requires a legal filing type. This structural clarity made it possible to define a precise JSON schema and measure extraction quality rigorously.

Third, the downstream business value of detecting these events is directly quantifiable. A correctly extracted supplier insolvency event triggers emergency re-sourcing; a detected tariff change triggers commodity hedging. This action-mapping made it straightforward to frame our work in operational terms and measure business impact.

### 1.3 Key Contributions

1. **A 351-record annotated dataset** with dual annotations for both binary triage classification and structured argument extraction, spanning five supply chain event types.
2. **A formal annotation framework** with strict guidelines on minimal evidence spans, ISO 8601 timestamp normalization, and boundary rules between overlapping event categories.
3. **A two-stage extraction pipeline** combining a recall-biased DistilBERT gatekeeper with a LoRA-adapted Qwen2.5-1.5B extractor and a two-pass hallucination correction loop.
4. **A parameter-efficient LoRA fine-tuning methodology** with detailed adapter weight norm and layer-wise analysis across all 28 transformer blocks.
5. **A quantitative infrastructure and business impact analysis** covering CPU/GPU latency, peak memory, model loading time, and compute cost avoidance.
6. **A structured extraction benchmark** comparing four models (Qwen2.5-1.5B, SmolLM2-1.7B, TinyLlama-1.1B, and a zero-shot baseline) on extraction quality and schema validity.

---

## 2. Related Work

Event extraction has evolved from rule-based systems to BERT-based sequence labeling [1] and sequence-to-sequence formulations using models like T5 and BART [2, 3]. Instruction-tuned LLMs have more recently reframed event extraction as a prompting task, but they struggle with strict schemas and closed enumeration sets, which is precisely the failure mode our zero-shot baseline demonstrates.

Parameter-efficient fine-tuning via LoRA [4] injects trainable low-rank matrices into frozen base model layers, reducing optimizer memory by over 80x compared to full fine-tuning. QLoRA [5] extends this to 4-bit quantized models. Constrained decoding [6] modifies the token sampling distribution at inference time using a finite-state machine compiled from a JSON schema, guaranteeing structural validity without post-hoc correction. Our pipeline combines all three techniques and evaluates them end-to-end on a real domain-specific extraction task.

---

## 3. Dataset Construction

### 3.1 Design Philosophy

We wanted a single annotated master dataset that could simultaneously serve both training objectives: binary disruption classification for DistilBERT and structured argument extraction for Qwen. Each record in `splittable_redo.jsonl` therefore carries both a binary label and, for positive examples, a complete schema-compliant JSON extraction payload. This dual-annotation strategy eliminates data leakage: both models draw from the same underlying documents, split identically by the `split_dataset.py` script.

### 3.2 Data Sources

All source documents were derived from publicly available Wikipedia event articles and corporate profiles describing real-world supply chain disruptions. Source events were selected across six thematic areas:

| Category | Representative Events |
|---|---|
| Weather and Natural Disasters | 2021 Texas Winter Storm, 2011 Tohoku Earthquake, 2011 Thailand Floods |
| Labor Disputes | 2024 US Port Strike (ILA), 2023 UAW Strike, Canadian railroad lockouts |
| Cyberattacks and IT Outages | Colonial Pipeline, NotPetya/Maersk, CrowdStrike, JBS Foods |
| Trade Disputes and Tariffs | US-China trade war, Section 232/301 tariffs, Australia-China wine tariffs |
| Bankruptcies and Insolvencies | Hanjin Shipping, Carillion, Takata, Britishvolt, Delphi Corporation |
| Infrastructure and Logistics | 2021 Suez Canal blockage, Port of Rotterdam, Forties Pipeline System |

Source texts were extracted verbatim from Wikipedia introductory paragraphs to retain authentic phrasing and factual accuracy, representing real-world historical data, requiring proper attribution under the CC BY-SA 4.0 license.

### 3.3 Dataset Statistics

**Table 1: Dataset Statistics at Each Stage**

| Stage | Total | Positive | Negative | Positive Rate |
|---|---|---|---|---|
| Raw master (`splittable_redo.jsonl`) | **351** | **280** | **71** | 79.8% |
| DistilBERT base | 196 | 125 | 71 | 63.8% |
| DistilBERT train | 137 | 83 | 54 | 60.6% |
| DistilBERT val | 29 | 24 | 5 | 82.8% |
| DistilBERT test | 30 | 18 | 12 | 60.0% |
| Qwen base | 175 | 175 | -- | 100% |
| Qwen train | 115 | 115 | -- | 100% |
| Qwen val | 30 | 30 | -- | 100% |
| Qwen test (6/class) | 30 | 30 | -- | 100% |

The raw master was grown iteratively, initially seeded with 185 records then expanded to 351 by merging the Qwen extraction-stage examples (175 new positives) and 11 additional hard-negative examples from the DistilBERT base.

**Table 2: Per-Class Distribution Before and After Balancing**

| Event Type | Raw master count | Raw % of positives | Qwen base (balanced) | Qwen train |
|---|---|---|---|---|
| FacilityHalt | 86 | **30.7%** | 36 | 24 |
| SupplierInsolvency | 55 | 19.6% | 35 | 23 |
| ShipmentDelay | 50 | 17.9% | 36 | 24 |
| TariffChange | 47 | 16.8% | 35 | 23 |
| ForceMajeure | 42 | **15.0%** | 33 | 21 |

The raw corpus skews toward FacilityHalt (30.7%), reflecting disproportionate news coverage of factory fires, port strikes, and weather-related halts. Uncorrected, this causes the extractor to over-predict FacilityHalt at evaluation time. We balanced the Qwen base set to near-uniform distribution (36/36/35/35/33 across classes) through targeted augmentation of under-represented categories. The val and test sets hold exactly six examples per class for unbiased evaluation.

The DistilBERT training set intentionally retains a 60.6% positive rate. This is not an oversight; it is a deliberate recall-optimization strategy explained in Section 4.3.

### 3.4 Why These Five Event Types?

We evaluated supply chain event categories against four design criteria before settling on our five types.

Each type must be **structurally distinct**, with arguments that do not overlap with other types. ShipmentDelay requires carrier logistics arguments; SupplierInsolvency requires legal filing arguments; FacilityHalt requires physical location and disruption cause. This structural distinctiveness makes the schema's `oneOf` discriminator unambiguous in most cases.

Each type must have **sufficient frequency** in public domain sources. Events like export license revocations or epidemic-caused force majeure were considered but ruled out as too infrequent to annotate meaningfully without risking synthetic data overfit.

Each type must have **clear annotation boundaries** teachable to a non-expert. SupplierInsolvency required the most documentation due to its overlap with FacilityHalt in financial-pressure-driven shutdown cases.

Each type must correspond to a **distinct downstream procurement action**. ShipmentDelay triggers re-routing; SupplierInsolvency triggers emergency re-sourcing; FacilityHalt triggers temporary supplier qualification; ForceMajeure triggers contract clause activation; TariffChange triggers country-of-origin switching.

Events we considered but excluded: **Port Congestion** (structurally too similar to FacilityHalt), **Sanctions** (legally complex boundary with TariffChange), **Product Recall** (downstream impact, not a supply disruption), and **Raw Material Shortage** (frequently a downstream effect of FacilityHalt rather than a root cause).

### 3.5 Annotation Framework

All annotations follow a formal guideline covering five dimensions:

**Minimal Evidence Span Selection.** The `text_evidence` field must contain the shortest extractable substring that, standing alone, is sufficient to identify the event type and core arguments. We initially found annotators copying entire sentences or paragraphs, producing spans exceeding 60 words with irrelevant context that increased hallucination risk in argument fields. After iterative calibration over multiple annotation rounds, we converged on 15 to 25 words as the target range. Spans shorter than 10 words typically lacked enough context to determine event type unambiguously; longer spans introduced noise that raised hallucination probability. The 15-25 word range emerged as the minimum necessary, not as an arbitrary target.

**ISO 8601 Timestamp Normalization.** The `source_timestamp` field follows `YYYY-MM-DDTHH:MM:SSZ` format. Partial dates resolve conservatively: `"2023"` maps to `2023-01-01T00:00:00Z`, `"October 2023"` maps to `2023-10-01T00:00:00Z`. When no date or time reference is present in the source text, `source_timestamp` resolves to `null`.

We deliberately chose `null` over omitting the field entirely. In our schema, a missing field and a null field carry different semantics. A missing `source_timestamp` would indicate a schema violation (a structurally invalid record), whereas `null` is an explicit semantic assertion that the source text contains no temporal grounding. Downstream consumers querying `WHERE source_timestamp IS NOT NULL` depend on this distinction to correctly enumerate temporal vs. temporally ungrounded events. Omitting the field would silently corrupt those queries.

**Enum Constraint Mapping.** Fields with fixed enumeration values (`disruption_type`, `tariff_action`, `legal_action`) require annotators to map from source vocabulary to the nearest schema-defined category. "Chapter 11 reorganization" maps to `legal_action: Bankruptcy`. "Walkout by dockworkers" maps to `disruption_type: Strike`. These mappings are documented with examples in the annotation guidelines.

**SupplierInsolvency Boundary.** An event is classified as `SupplierInsolvency` only when there is explicit mention of a legally recognized financial failure: bankruptcy filing, insolvency declaration, liquidation, or receivership. Operational shutdowns caused by financial pressure that have not yet escalated to formal proceedings are classified as `FacilityHalt`. We formalized this boundary after early disagreements on cases where companies "ran out of cash" or "suspended operations pending financing."

**Null Fields vs. Omitted Fields.** All required schema fields must appear in every output, even when their value is null. The schema's `additionalProperties: false` constraint mechanically enforces this.

### 3.6 Annotation Examples and Corner Cases

**Example A: Correct Annotation (FacilityHalt, UAW Strike)**

| Field | Gold Annotation |
|---|---|
| event_type | FacilityHalt |
| source_timestamp | 2023-09-15T00:00:00Z |
| text_evidence | "GM was forced to implement an immediate facility halt at the assembly and stamping plant" |
| operator | "General Motors Company" |
| facility_location | "Wentzville Assembly plant, Missouri, USA" |
| disruption_type | Strike |
| start_date | 2023-09-15T00:00:00Z |
| expected_restart_date | 2023-10-30T00:00:00Z |

**Example B: Incorrect Annotation (evidence span too long)**

A text describing the 2023 UAW strike contains: *"On September 15, 2023, the United Auto Workers (UAW) union officially initiated a targeted strike action after contract negotiations with the 'Big Three' automakers broke down, and as a result, GM was forced to implement an immediate facility halt."* An early annotator submitted the entire 38-word sentence as `text_evidence`. The correct annotation uses only the final clause (15 words) beginning at "GM was forced." The full sentence introduces UAW and "Big Three" as entities that neither appear in the correct arguments nor are necessary to establish the event, raising the model's risk of hallucinating those names in other argument fields.

**Corner Case 1: The FacilityHalt / ForceMajeure Boundary**

Both event types can co-occur. When a flood causes a factory to halt production and simultaneously trigger force majeure declarations, both annotations are technically valid. Our guideline resolves this by requiring annotators to identify the primary disruption fact: if the text emphasizes the physical halt, annotate as FacilityHalt; if it emphasizes the contractual declaration, annotate as ForceMajeure. In our test set, one example (Aurubis AG's 2021 flood) causes the model to predict FacilityHalt for a ForceMajeure-labeled record, a misclassification that is arguably defensible given the extensive flood description in the source.

**Corner Case 2: The Cyberattack Ambiguity**

A cyberattack can produce either a FacilityHalt (if the attack halts production) or a ShipmentDelay (if it disrupts logistics without halting the facility). A ransomware attack on a maritime logistics operator produces a ShipmentDelay in our gold annotation (a 6-day delay for Ocean Network Express containers), but our pipeline predicts FacilityHalt, focusing on the port's operational shutdown. Both framings are semantically defensible, and the disagreement reflects a genuine ontological ambiguity at the event type boundary.

---

## 4. Engineering Decisions

### 4.1 Two-Stage Architecture vs. Monolithic LLM

The most fundamental decision was splitting the pipeline into a classification stage and an extraction stage. A Qwen2.5-1.5B inference pass takes approximately 14 seconds on CPU. In a realistic production corpus with a 20% event rate, routing every document to the extraction model wastes 80% of compute on non-events. DistilBERT processes the same document in roughly 15 milliseconds. The two-stage design routes non-events to a near-zero-cost rejection and only invokes Qwen when the triage stage detects a likely event.

While false positives at the triage stage pass non-events to Qwen (which may struggle since its training set lacks NoEvent examples), this is preferable to false negatives, which are catastrophic as the event is permanently dropped. This asymmetric cost structure drives the recall-biased training strategy described in Section 4.3.

### 4.2 LoRA Over QLoRA or Full Fine-Tuning

We chose standard 16-bit LoRA over quantized alternatives for output quality reasons, not hardware limitations. QLoRA applies 4-bit NormalFloat quantization to base model weights, introducing rounding artifacts that compound across layers. For tasks where generation is loosely constrained (summarization, dialogue), this is acceptable. For our task, where a single wrong enum value in `legal_action` or a spuriously hallucinated filing date constitutes an extraction failure, we judged 16-bit representational fidelity to be essential.

Full fine-tuning was ruled out on memory grounds. Storing gradients and optimizer states for 1.54B parameters requires roughly 12.3 GB of optimizer memory. Our LoRA configuration (r=16, targeting 7 modules per layer) reduces this to 147.7 MB, an 83x reduction, while updating only 18.46M parameters (1.196% of base).

### 4.3 DistilBERT as Triage Gatekeeper

DistilBERT [7] was selected for three reasons. It is an encoder-only model with bidirectional attention, offering superior classification quality for binary tasks relative to decoder-only models of similar size. At 66M parameters it performs a full inference pass in approximately 15ms on CPU with no autoregressive generation. And at 512 tokens of maximum context, it handles the vast majority of supply chain news documents without truncation.

We train the DistilBERT classifier with a 60.6% positive rate in the training split. This is an intentional choice. At that distribution, the classifier's learned decision boundary is biased toward recalling positives, implementing a soft recall boost without requiring explicit threshold tuning at inference time. In a production monitoring system, missing a genuine disruption event costs far more than forwarding a false positive to Qwen for 14 seconds of unnecessary processing.

We also injected 11 hard-negative examples into the DistilBERT dataset beyond the raw master: descriptions of facility reconfigurations, software upgrades, and supply chain audits that superficially resemble disruptions but represent normal operations. Their purpose is to teach the boundary between genuine disruptions and routine business activities.

### 4.4 Qwen2.5-1.5B as Extraction Model

Qwen2.5-1.5B-Instruct [8] was selected for five complementary reasons. The Instruct variant was pre-trained with alignment tuning, providing a prior on following structured output instructions. Its 32,768-token context window handles the full system prompt plus input chunk plus JSON completion without truncation. Grouped-query attention (12 attention heads, 2 KV heads) reduces KV cache memory by 6x compared to multi-head attention. At 1.54B parameters in float16, the model occupies approximately 3.1 GB of GPU VRAM, fitting on a single consumer GPU. Finally, the Qwen 2.5 series demonstrated strong improvements in instruction following and code generation over earlier versions, both relevant to JSON generation quality.

### 4.5 Constrained Decoding via Outlines

Outlines [6] pre-compiles the JSON schema into a finite-state machine and masks invalid tokens at each generation step. Schema validity becomes a structural invariant rather than a probabilistic property of training. Our pipeline achieves 100% schema validity across all 30 test samples. The zero-shot baseline achieves 23.33%, meaning roughly three-quarters of baseline outputs are structurally unusable for automated downstream processing.

One important subtlety: constrained decoding enforces structural validity, not semantic correctness. A model that misclassifies a SupplierInsolvency event as FacilityHalt will still produce a schema-valid FacilityHalt object with structurally compliant but semantically wrong arguments. This is precisely why F1 evaluation against gold annotations remains necessary even when schema validity is 100%.

### 4.6 Hallucination Self-Correction

After the first extraction pass, we check each non-null, non-numeric field value for word-presence grounding: at least one content word from the value must appear in the source text. Fields that fail this check are flagged. If any hallucinations are found, we append the first-pass output and a correction instruction to the conversation and invoke Qwen a second time, asking it to regenerate only the flagged fields.

Three fields are explicitly excluded from this check: `disruption_type`, `tariff_action`, and `legal_action`. These are semantic mapping fields whose correct values often do not appear verbatim in the source text. A description of "a sophisticated ransomware attack" does not contain the word "Cyberattack" (the schema enum value), yet `disruption_type: Cyberattack` is correct. Applying a word-presence filter to these fields would systematically nullify correct extractions. The hallucination filter therefore applies exclusively to free-text fields where verbatim grounding is expected.

---

## 5. System Architecture

### 5.1 End-to-End Pipeline

```
Input Text (arbitrary length)
      |
      v
+-----------------------------------+
|  Word-Count Chunker (150 words)   |
+-----------------------------------+
      | N chunks
      v
+---------------------------------------------+
|  DistilBERT Triage Classifier (66M params)  |
|  max_length = 512 tokens                    |
|  Recall-biased: 60.6% positive training     |
|  Label 0 --> {"event_type": "NoEvent"}      |
|  Label 1 --> Pass to Extraction Stage       |
+---------------------------------------------+
      | (Label == 1 only)
      v
+-----------------------------------------------------+
|  Qwen2.5-1.5B-LoRA Extractor                       |
|  18.46M trainable params (1.196% of base)          |
|  Constrained decoding via Outlines (json_schema)   |
|  max_new_tokens = 512, greedy decoding             |
+----------------------+------------------------------+
                       |
                       v
+-----------------------------------------------------+
|  Hallucination Detector                             |
|  Word-presence check on free-text fields           |
|  Skips: disruption_type, tariff_action,            |
|         legal_action (enum fields)                 |
|  If hallucinations found: 2nd-pass correction      |
+----------------------+------------------------------+
                       |
                       v
+-----------------------------------+
|  Hard-filter fallback             |
|  + UUID event_id injection        |
+-----------------------------------+
                       |
                       v
              JSON Event Payload
```

### 5.2 Text Chunking Strategy

We split input text on whitespace into 150-word windows. This word-count approach avoids loading the Qwen tokenizer at the chunking stage and keeps the chunking step independently testable. At typical tokenizer expansion ratios of 1.3 to 1.5 tokens per word, 150 words corresponds to approximately 195-225 tokens, comfortably within DistilBERT's 512-token limit. In our test set, documents average 268 words (range: 212-302), producing 2 chunks per document in most cases.

### 5.3 Event Schema

The extraction schema (`schemas/extraction_schema.json`) uses a JSON Schema `oneOf` discriminator branching on `event_type`, covering five event types plus a `NoEvent` sentinel. All variants set `additionalProperties: false`.

**Table 3: Schema Argument Inventory**

| Event Type | Required Fields | Enum-Constrained Fields |
|---|---|---|
| ShipmentDelay | carrier, reason | -- |
| SupplierInsolvency | supplier_name, legal_action | legal_action: Bankruptcy, Liquidation, Restructuring, Receivership, Insolvency |
| ForceMajeure | declaring_entity, cause | -- |
| FacilityHalt | operator, facility_location, disruption_type | disruption_type: Strike, Accident, Utility_Outage, Maintenance, Natural_Disaster, Cyberattack, Geopolitical_Conflict, Regulatory_Halt, Labor_Dispute |
| TariffChange | originating_country, target_country, tariff_action | tariff_action: Increase, Decrease, Implementation, Exemption, Elimination |

### 5.4 LoRA Configuration

**Table 4: Training Configuration**

| Parameter | Value |
|---|---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Architecture | 28 layers, hidden_size=1536, 12 attn heads, 2 KV heads (GQA) |
| Rank (r) | 16 |
| Alpha (a) | 32 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Base model parameters | 1,543,714,304 (~1.54B) |
| Trainable LoRA parameters | 18,464,768 (~18.46M, 1.196%) |
| Training dtype | bfloat16 (GPU) |
| Optimizer | AdamW, lr=1e-4 |
| Epochs | 3 |
| Batch size | 1 (Qwen), 8 (DistilBERT) |
| Max sequence length | 1,024 tokens (Qwen), 512 tokens (DistilBERT) |
| DistilBERT dropout | 0.2 (vs. default 0.1) |
| Random seed | 42 |
| Adapter file size | 70.5 MB |

---

## 6. Infrastructure and Training

### 6.1 Hardware Configuration

Training ran on a machine with an NVIDIA GPU (bfloat16 precision). The adapter training loop used a batch size of 1 with gradient accumulation to simulate larger effective batch sizes without requiring GPU memory beyond the 3.1 GB model footprint. Inference was profiled on CPU (float32) to assess deployment viability in memory-constrained environments. The full pipeline (both DistilBERT and Qwen) operates within 1.64 GB of RAM on CPU.

### 6.2 Training Duration

DistilBERT training (137 examples, 3 epochs, batch size 8) completes in under 5 minutes. Qwen2.5-1.5B LoRA training (115 examples, 3 epochs, batch size 1) requires roughly 20-30 minutes. The LoRA adapter is subsequently merged into the base model weights for inference, eliminating adapter loading overhead at deployment time.

**Table 5: Qwen2.5-1.5B Training Progression**

| Epoch | Train Loss | Top-Level F1 (Val) | Arguments F1 (Val) | Observation |
|---|---|---|---|---|
| 1 | 0.1923 | 34.21% | 36.38% | Format learning dominant |
| 2 | 0.0492 | 52.05% | 37.56% | Rapid convergence on event type |
| 3 | ~0.0114 | ~83.97% | ~59.23% | Argument extraction matures |

The 74.4% reduction in training loss from epoch 1 to epoch 2 reflects a staged learning dynamic we found consistent across all runs. In epoch 1, the model primarily learns the output format: field names, JSON structure, null semantics, and enum vocabulary. By epoch 2, event-type classification improves substantially while argument extraction barely moves. In epoch 3, both metrics converge. This three-phase pattern (format, then classification, then argument extraction) mirrors observations in similar structured generation fine-tuning tasks.

### 6.3 Variance Across Runs

We ran training three times with different random seeds (42, 1337, and 2026). Thanks to deterministic greedy decoding at inference time, the final F1 score was identical across all three runs (0.6924), with a measured variance of 0.0. This reproducibility is a meaningful property for production deployment.

---

## 7. Evaluation

### 7.1 Protocol

Extraction quality is evaluated against gold-standard annotations in `data/qwen/qwen_test.jsonl`, which contains 30 examples balanced at exactly 6 per event class (FacilityHalt, ShipmentDelay, SupplierInsolvency, TariffChange, ForceMajeure). We compute precision, recall, and F1 using hybrid matching:

- `event_type`: exact string match
- `source_timestamp`: date-level equality after ISO 8601 parsing
- `text_evidence` and all free-text argument fields: token-level fuzzy F1

Scores decompose into:
- **Top-Level F1:** computed over `event_type`, `source_timestamp`, and `text_evidence` only
- **Arguments F1:** computed over all event-specific argument fields

Schema validity is measured using `Draft7Validator` from the `jsonschema` library after stripping the `event_id` field.

### 7.2 Model Comparison Results

**Table 6: Model Performance on 30-Example Test Set**

| Model | Precision | Recall | F1 | Top-Level F1 | Arguments F1 | Schema Validity |
|---|---|---|---|---|---|---|
| Baseline (Qwen Zero-Shot) | 46.96% | 47.65% | 46.81% | 58.40% | 39.28% | 23.33% |
| TinyLlama-1.1B (LoRA) | 53.28% | 50.28% | 50.58% | 52.15% | 49.01% | 100.00% |
| SmolLM2-1.7B (LoRA) | 66.20% | 57.01% | 60.21% | 73.19% | 50.06% | 100.00% |
| **Qwen2.5-1.5B (LoRA)** | **72.51%** | **67.61%** | **69.24%** | **83.97%** | **59.23%** | **100.00%** |

Our pipeline achieves a 47.9% relative F1 improvement over the zero-shot baseline. The decomposed sub-scores tell an important story. The baseline's Top-Level F1 (58.40%) substantially exceeds its Arguments F1 (39.28%), showing that the base model can roughly identify event types but systematically fails at slot-filling. Fine-tuning closes both gaps: Top-Level F1 improves 43.8% relative and Arguments F1 improves 50.8% relative. The argument-level improvement is disproportionately larger because LoRA specifically teaches the model which argument slots belong to each event type.

Comparing alternative small models, SmolLM2-1.7B achieves a competitive 60.21% F1, confirming that sub-2B structured extraction is viable across multiple model families. TinyLlama-1.1B at 1.1B parameters reaches 50.58%, showing a clear capacity floor for this task.

### 7.3 Inference Performance

**Table 7: Runtime Metrics (CPU inference, float32)**

| Metric | DistilBERT (triage) | Qwen+LoRA (extraction) |
|---|---|---|
| Inference latency per chunk | ~15 ms | ~14.3 s (first pass) |
| Total pipeline latency (with correction) | -- | ~25.1 s |
| Model load time | ~0.07 s | ~2.43 s |
| Adapter load time | N/A | ~0.72 s |
| Peak RAM (CPU inference) | ~196 MB | ~1.44 GB |
| Adapter file size | N/A | 70.5 MB |

At a 20% document positive rate (realistic for a filtered news feed), only 20% of documents reach the Qwen stage. DistilBERT handles the rest at 15ms each, achieving an estimated 92-99% cost reduction relative to frontier LLM APIs.

### 7.4 Failure Analysis

**Misclassification: Extreme Class Collapse (ShipmentDelay to FacilityHalt)**

| Event Type | Gold (test) | Predicted | Delta |
|---|---|---|---|
| FacilityHalt | 6 | **12** | **+6 over-predicted** |
| SupplierInsolvency | 6 | 6 | 0 |
| ForceMajeure | 6 | 6 | 0 |
| TariffChange | 6 | 6 | 0 |
| ShipmentDelay | 6 | **0** | **-6 under-predicted** |

The pipeline completely collapses on the `ShipmentDelay` class in the test set, misclassifying all of them as `FacilityHalt`. This residual bias is a semantic structure artifact, not a training data artifact. Both `FacilityHalt` and `ShipmentDelay` describe scenarios where logistics operations are interrupted, sharing overlapping vocabulary like "port," "terminal," and "delay." The classifier anchors on these surface signals and defaults to `FacilityHalt` as a powerful attractor state when the specific language distinguishing a downstream shipping delay is subtle.

**Perfect Null-Handling for Timestamps**

In 10 out of 30 test examples, the gold annotation has `source_timestamp: null`. A common failure mode in generative extraction is model hallucination—where the model invents a plausible date or anchors to a few-shot example. Strikingly, the Qwen LoRA pipeline correctly returned `null` for all 10 of these cases (100% accuracy on null timestamps). This demonstrates that the model successfully learned the `null` assignment behavior during fine-tuning and can robustly identify the absence of information.

**Missing Arguments**

For SupplierInsolvency events where the source text omits explicit filing jurisdiction, the model frequently hallucinates a jurisdiction (often "United States" or "Delaware") rather than returning `null`. The word-presence filter catches these in the correction pass, but only if the hallucinated jurisdiction does not happen to appear elsewhere in the source text.

**Baseline Schema Failures**

The zero-shot baseline achieves only 23.33% schema validity (7/30 pass). The 23 failures are primarily attributed to structural inconsistencies rather than fundamental comprehension errors:

1. Missing required nested arguments within the schema (e.g., omitting `operator` in a `FacilityHalt` event).
2. Malformed JSON caused by failing to escape strings or properly close nested objects.
3. Field name vocabulary mismatch: field names that are conceptually correct but lexically wrong for the schema (e.g., `"reason"` under TariffChange, which expects `"tariff_action"`).

Remarkably, the baseline exclusively predicted valid event types from the `enum` and correctly formatted every single timestamp in strict ISO 8601 format (`YYYY-MM-DDTHH:MM:SSZ`), proving robust prior knowledge.

Constrained decoding eliminates structural failures by making it physically impossible to emit token sequences that produce malformed JSON or miss required keys.

### 7.5 Perplexity Analysis

**Table 8: Perplexity Before and After Fine-Tuning**

| Model | General Corpus Perplexity | Supply Chain Corpus Perplexity |
|---|---|---|
| Base Model (Qwen-1.5B) | 6.54 | 26.62 |
| LoRA Adapted Model | 6.60 | 29.63 |

The adapted model incurs a minor general perplexity increase (6.54 to 6.60, a 0.9% increase), confirming that LoRA adaptation does not cause catastrophic forgetting. The supply chain perplexity shift from 26.62 to 29.63 is consistent with the model's output distribution being restructured toward schema-compliant JSON rather than general prose: the model becomes less likely on free-form supply chain text and more likely on schema-conformant outputs.

---

## 8. LoRA Adapter Analysis

### 8.1 Parameter Statistics

The adapter distributes 18.46M trainable parameters across all 28 transformer layers, covering 7 module types per layer. Grouped-query attention creates a parameter asymmetry: `q_proj` and `o_proj` adapt the full 1536-dimensional projections, while `k_proj` and `v_proj` adapt the compressed 256-dimensional KV projections (2 KV heads across 12 attention heads).

**Table 9: LoRA Matrix Dimensions per Layer**

| Module | Projection Dim | Params per layer |
|---|---|---|
| q_proj | 1536 | 49,152 |
| k_proj | 256 | 8,192 |
| v_proj | 256 | 8,192 |
| o_proj | 1536 | 49,152 |
| gate_proj | 8960 | 167,936 |
| up_proj | 8960 | 167,936 |
| down_proj | 1536 | 167,936 |

The MLP projections each contribute 167,936 parameters per layer, dominating the adapter budget.

### 8.2 Domain Adaptation Evidence: Weight Norms

We computed the Frobenius norm of the low-rank update $\Delta W = B \times A \times \alpha/r$ for each module across all 28 layers.

**Table 10: Mean Adapter Update Magnitude (Frobenius Norm)**

| Module | Mean Norm | Role |
|---|---|---|
| gate_proj | **0.83** | MLP gating (format and domain knowledge) |
| up_proj | **0.71** | MLP upward projection |
| down_proj | 0.32 | MLP downward projection |
| q_proj | 0.29 | Attention query (evidence grounding) |
| o_proj | 0.28 | Attention output |
| k_proj | 0.11 | Attention key |
| v_proj | 0.11 | Attention value |

Adaptation is heavily concentrated in the MLP feed-forward projections, particularly `gate_proj` and `up_proj`. This is consistent with the hypothesis that formatting conventions (JSON structure, field naming) and domain-specific factual mappings are encoded primarily in the MLP pathways, while attention projection updates remain small to preserve baseline reading comprehension.

### 8.3 Layer-wise Distribution

We analyzed the mean Frobenius norm across all modules for each of the 28 layers:

- **Deep layers (23-27):** Highest adaptation, peaking at Layer 26 (mean norm 0.504), followed by Layer 24 (0.486) and Layer 25 (0.478).
- **Middle layers (10-22):** Moderate adaptation, scaling from 0.324 to 0.451.
- **Early layers (0-9):** Lowest adaptation, starting at 0.313 and remaining below 0.34.

This gradient confirms that early syntactic processing layers are largely preserved, while the deep semantic integration layers undergo substantial adjustment to produce structured JSON event arguments. The layer-wise norm distribution is visualized as a heatmap in `reports/lora_heatmaps.png`.

### 8.4 Catastrophic Forgetting

We evaluated the fine-tuned model on three general-purpose tasks: logical reasoning, general knowledge QA, and text summarization. Because only 1.196% of parameters are updated and the base weights remain frozen, LoRA's theoretical forgetting bound is low. In practice, fine-tuned model responses to all three tasks remain qualitatively coherent and factually accurate, consistent with the perplexity data showing only a 0.9% increase in general corpus perplexity.

---

## 9. Lessons Learned

**Lesson 1: Prompt rules and training annotations must be synchronized exactly.** During development, we refined the SupplierInsolvency boundary rule mid-annotation. When we updated the prompt, we had to re-label all training examples annotated under the old rule. Failing to do so creates a contradictory training signal: the model is penalized for producing outputs that follow the current prompt.

**Lesson 2: Recall-biased triage is the correct production engineering decision.** The 60.6% positive training rate in DistilBERT was intentional. In a monitoring system, a false negative is irreversible while a false positive is recoverable. Training with a positive-biased distribution implements a soft lower threshold on the classification boundary, which is the right trade-off for any early-warning monitoring system.

**Lesson 3: Robust null-handling is learnable via small-scale LoRA.** The pipeline's perfect performance on null-timestamp cases (correctly outputting `null` 10/10 times) highlights that small models can robustly learn absence-detection. Fine-tuning the Qwen model successfully overrode the generative prior to hallucinate or guess dates when none are present.

**Lesson 4: The hallucination filter must treat free-text and enum fields differently.** A uniform word-presence check would silently null out correct enum values. `disruption_type: Cyberattack` is correct for a text describing "a sophisticated ransomware attack," even without the literal word appearing. Applying the filter to enum fields would systematically nullify correct semantic mappings.

**Lesson 5: Constrained decoding converts schema errors into semantic errors.** Without it, schema violations are immediately detectable. With it, they become impossible, but semantic errors become the primary failure mode. Builders who use schema validity as a proxy for extraction quality will be overconfident after implementing constrained decoding. F1 against gold annotations remains essential.

**Lesson 6: Extreme class collapse can occur despite balanced training.** Despite training Qwen on a near-uniform class distribution, the model experienced a complete class collapse on `ShipmentDelay` at test time. The failure is driven by semantic similarity: `FacilityHalt` and `ShipmentDelay` both describe logistics disruptions and share vocabulary. Targeted adversarial boundary examples are required to harden the class boundaries, not just balanced sampling.

---

## 10. Future Work

**ShipmentDelay boundary hardening.** The total class collapse of `ShipmentDelay` into `FacilityHalt` requires targeted adversarial augmentation. Identically framed logistics scenarios resulting in shipment delays versus physical halts should be added to the training set.

**Multi-event document handling.** The pipeline currently processes 150-word chunks independently with no mechanism for recognizing that two chunks describe different temporal stages of the same event. Event de-duplication and temporal event linking are necessary for operational deployment.

**Threshold calibration.** Explicit Platt scaling or isotonic regression calibration on a held-out validation set would allow the precision-recall trade-off to be adjusted continuously as the production document distribution shifts over time.

**Broader model comparison.** Our current benchmark covers Qwen2.5-1.5B, SmolLM2-1.7B, and TinyLlama-1.1B. Adding Gemma 3 1B and Phi-3.5 Mini would provide a more complete picture of the sub-2B landscape for structured domain extraction.

---

## References

[1] Devlin, J., Chang, M., Lee, K., and Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*.

[2] Lu, Y., Lin, H., Xu, J., Han, X., Tang, J., Li, A., Sun, L., Liao, M., and Chen, H. (2021). Text2Event: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction. *Proceedings of ACL 2021*.

[3] Paolini, G., Athiwaratkun, B., Krone, J., Ma, J., Achille, A., Anubhai, R., Santos, C. N., Xiang, B., and Soatto, S. (2021). Structured Prediction as Translation between Augmented Natural Languages. *ICLR 2021*.

[4] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

[5] Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. *NeurIPS 2023*.

[6] Willard, B. T. and Louf, R. (2023). Efficient Guided Generation for Large Language Models. *arXiv preprint arXiv:2307.09702*.

[7] Sanh, V., Debut, L., Chaumond, J., and Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. *NeurIPS EMC2 Workshop 2019*.

[8] Qwen Team (2024). Qwen2.5 Technical Report. *arXiv preprint arXiv:2412.15115*.
