Abstract :

This paper presents an industry experience report on engineering a two-stage, triage and extraction pipeline, for supply chain risk/disruption event detection and structured json  schema extraction. It is a resource efficient pipeline which consists of a DistilBert binary classifier (66M parameters) to detect the relevance of the text to a supply chain related event and pass it to the event extraction model which performs Finite State Masking using outlines to output a structured schema. This structured schema is valuable to enterprise applications for creating functional APIs and efficient event detection. We fine-tuned our extraction model (Qwen2.5-1.5B-Instruct generative model with 1.54B parameters) using LoRA (only 18.46M trainable parameters via LoRA) and best practices to reduce memory overhead for the fine tuning task. We discuss in detail the process undertaken to acquire the dataset, deliberate class splits and annotation guidelines along with the reasoning behind them. The event taxonomy consists of 5 high impact and recurrent events in the supply chain domain which we chose after analysing various sources in recent history. We gave priority to f1 score over raw accuracy as event extraction is more nuanced than plain text generation. We asymmetrically fine-tuned the DistilBert classifier to guarantee 99% recall at the cost of precision (81%) as enterprises cannot afford to drop real events. The respectable precision also guarantees that completely unrelated events definitely get filtered out. We evaluated our pipeline through a combination of fuzzy token level f1 for variable schema fields and strict evaluation of top-level schema fields. It produced a 43.7% improvement in the top level f1 score over a baseline, unconstrained Qwen2.5-1.5B-Instruct model's best performance (58.40% to 83.97%) and 47.97% net increase in the collated f1 score (46.80% to 69.25%). Our pipeline crushed the baseline model in schema validity with a 100% JSON schema validity over the baseline's 23.33%. In the process, we faced a lot of challenges. These include Overprediction of a Facility Halt over Shipment Delay due to semantic similarity in the texts, model hallucinations due to context window exhaustion and spurious correlation to name a few. We modified the pipeline to handle hallucinations to a large degree using raw python text matching and reprompting if a hallucination by the model was detected. The base model size itself remains a large bottleneck to achieve enterprise standard 90% f1 scores.


**Source Code:** The source code for this pipeline is maintained in the `formcept/whiteboard` repository to validate the internship at FORMCEPT.

1. Introduction

1.1 The Industry Problem :

Modern organizations that operate in close proximity to the supply chain and logistics layer of business often face disruptions due to a continously evolving and rapidly changing world. These issues include but are not limited to port strikes, factory halts, supplier insolvency, regulatory tariff changes, natural calamities, war and terrorism. These modern organizations continuously monitor large volumes of unstructured documents (real time or stored in a database) to identify operational risks and business events. They are increasingly using Large Language Models since they are highly capable classifying these events and detecting risks. The deployment of LLMs for domain specific event extraction brings up quite a few practical concerns concerning the compute cost, latency and structured/constrained output generation since these large dynamic memory hungry models often load their large weights and optimizer to do fine-tuning. Cost of a typicalLLM api (most available market share LLMs can cost roughly between $0.002-$0.015/1000 tokens) can be astronomically impractical for business scenarios as they might process news articles, regulatory filings, etc. The below blog article details our engineering experience building an actual 2-stage event extraction pipeline for supply chain risk with small LLMs that performs better than a plain LLM but at a tiny fraction of cost.

1.2 Why Supply Chain?

We chose Supply Chain Risk Monitoring because of the following reasons:

1. Availability of public data: Unlike finance and health, where a huge amount of training data are strictly protected and secured for proprietary reasons, disruptions and events relating to supply chains often get covered in publicly available news media and reports. This enables us to collect and create a large dataset that can reach high annotation quality without getting entangled in the process of obtaining corporate data. Our datasets for this research come from Wikipedia articles that deal with supply chain-related terms and keywords.

2. Well defined Event Taxonomies: supply chain disruption events can be easily classified into the following five major categories. For example, Facility Halt events have an identifiable entity like (Factory,Warehouse,Port), and all arguments within each event schema are fixed, well defined, and can easily map to a schema object in JSON.

*Facility Halt

*Supplier Insolvency

*Shipment Delay

*Force Majeure

*Tariff change

3. Measurable Business Impact: The benefit of event detection and extraction for business value can be easily measured by avoided losses in terms of procuring data, rerouting freight, etc. if disruption events are detected before their catastrophic impact occurs. It lends itself well to ROI analysis through various business dashboards and reports.

1.3 Contibutions:

1. Pipeline Dataset: Created a recall biased triage dataset consisting of 185 documents (125 positive, 71 negative) with class distribution skewed towards fewer disruptions in the data that aims to minimize missing events for detection systems. The specific data creation strategy is explained in Section 3. We then split these datasets and use them as raw training data to fine-tune the DistilBert and Qwen models independently.

2.JSON event schema: Developed a JSON Schema covering all the potential categories of events with all possible arguments and enum constraints, with a “NoEvent” option provided as the fallback to satisfy strict schema compliance. The Schema is enforced using an Outlines created Finite State Machine and allows constrained generation of tokens.

3. Two stage Triage and Extraction pipeline: Engineered a 2-stage system where DistilBert filters and prioritizes the news data and routes potentially risky content to Qwen2.5-1.5B-Instruct for detailed, constrained extraction and hallucination checking using a 2-pass approach of self-correction with raw python code, a mechanism that successfully reduces hallucinations.

4. Quantitative business impact analysis: Empirically assessed the potential of systematic failure modes of the pipeline, analyzed inter-class semantic ambiguity (FacilityHalt), computed estimated costs and latency savings from the pilot program, and provided working examples of the developed pipeline and how it impacts business.

5. LoRA Fine-Tuning Methodology: Successfully utilized Low-Rank Adaptation (r=16) to effectively fine-tune a 1.5B parameters model to learn the desired schema and generate outputs with minimal computational overhead while keeping the original parameters of the model fixed.

6. Structured Extraction Benchmark:Rigidly benchmarked latency, throughput, memory usage and Schema Validity of the proposed system and compared with an end-to-end LLM system in addition to precision, recall and F1 scores.

2. Related work:

The area of event extraction has moved from rule-based methods to sequence labeling models and seq2seq models like T5 and BART. It is now in an era where Large Language Models have become standard to treat the task as an end-to-end prompt task, but current LLMs have poor performance in terms of strict schema enforcement, closed set enumeration, and accuracy on the data. The pipeline of our 2-stage model can be considered as one instance of this trend as it utilizes constrained output generation via Outlines and has a robust design that leverages 2 stage filtering through a Small Language Model to keep computational latency low and provide accurate output.

Parameter efficient fine-tuning techniques reduce the total training cost for the models and increase the capacity of available models for use on more constrained hardware or cloud environments with costs to consider. LoRA applies trainable low rank adapters on top of frozen parameters of the base model. This drastically reduces optimizer memory requirement compared to full fine-tuning(> 80 times less). 

QLoRA further reduces parameters using 4-bit quantization. 

In constrained generation, finite state machines restrict token prediction, generated from a structured data specification, e.g. A schema, this limits token outputs during decoding without a post-processing step to filter invalid output. In this work, we apply LoRA with constrained generation and show how an end-to-end system can outperform baseline and be practical for business applications.

3. Dataset Construction :

3.1 Design Philosophy:

We sought to create a single annotated master dataset that would serve dual purposes of fine-tuning DistilBERT and extracting argument payloads for qwen. Hence each entry in splittable_redo.jsonl has both a binary disruption label and for positives, the full JSON extraction schema. This avoids data leakage between the models as both are trained on the same documents which are split via split_dataset.py

3.2 Sources:

All of our source documents described real-world supply chain disruptions that occurred in the past. We identified 6 categories of supply chain disruptions:

Category Representative Events Weather and Natural Disasters 2021 Texas Winter Storm, 2011 Tohoku Earthquake, 2011 Thailand Floods Labor Disputes 2024 US Port Strike (ILA), 2023 UAW Strike, Canadian railroad lockouts Cyberattacks and IT Outages Colonial Pipeline, NotPetya/Maersk, CrowdStrike, JBS Foods Trade Disputes and Tariffs US-China trade war, Section 232/301 tariffs, Australia-China wine tariffs Bankruptcies and Insolvencies Hanjin Shipping, Carillion, Takata, Britishvolt, Delphi Corporation Infrastructure and Logistics 2021 Suez Canal blockage, Port of Rotterdam, Forties Pipeline System

We extracted the articles’ introductory paragraphs to retain the original language used to describe the event for fidelity. As such, we have attributed all text_evidence to Wikipedia under the CC BY-SA 4.0 license.

3.3 Dataset Statistics :

Table 1: Dataset statistics at various stages. See section 3.3 for details about the balancing Step.

Stage Total Positives Negatives Positive Rate

Raw master (splittable_redo.jsonl) 351 280 71 79.8%

DistilBERT base 196 125 71 63.8%

DistilBERT train 137 83 54 60.6%

DistilBERT val 29 24 5 82.8%

DistilBERT test 30 18 12 60.0%

Qwen base 175 175 100.0%

Qwen train 115 115 100.0%

Qwen val 30 30 100.0%

Qwen test (6/class) 30 30 100.0%

The raw master (splittable_redo.jsonl) consists of our initial 185 entries expanded by combining the qwen extraction stage example annotations (adds 175 positives). Additionally, we added 11 hard negatives from the distilbert base.

Table 2: Class distribution before and after balancing¶

Event Type Raw master count Raw % of positives Qwen base Qwen train

FacilityHalt 86 30.7% 36 24

SupplierInsolvency 55 19.6% 35 23

ShipmentDelay 50 17.9% 36 24

TariffChange 47 16.8% 35 23

ForceMajeure 42 15.0% 33 21

The raw master contains a skewed distribution with 30.7% of entries being FacilityHalt . This leads to overrepresentation of this category which causes higher prediction probabilities for this class compared to others. To balance this skewness we balanced the extractor (Qwen2.5:1.5B-Instruct) to have roughly similar numbers for each of the 5 categories with the numbers shown in the “Qwen base” column. The validation and test sets have 6 examples per category. The distilbert train set was deliberately imbalanced to improve recall (see Section 4.3 for details), hence has a positive rate of 60.6%.

3.4 Why these five event types? :

We evaluated potential supply chain event categories against four criteria before selecting our final set of five.

Each category has distinguishing elements that make its arguments unique and non-overlapping with those of other categories. For example, the ShipmentDelay requires the carrier logistic argument, while the SupplierInsolvency requires the legal filing argument, and the FacilityHalt requires both the location and the disruption cause arguments. The presence of these unique arguments helps to define the categories clearly and distinguish them from one another.

Each category is common enough that examples can be found in the public literature.

Each category has simple enough definitions to avoid disagreements between annotators. SupplierInsolvency was the most contentious category due to overlaps with FacilityHalt.

Each category has different downstream procurement implications. ShipmentDelay implies rerouting, while SupplierInsolvency implies emergency sourcing, and FacilityHalt implies requalification. ForceMajeure implies contractual clauses while TariffChange implies country of origin shifts.

Supply chain events that we initially considered but ultimately discarded include Port Congestion (similar to FacilityHalt), Sanctions (similar to TariffChange but with legal complexities), Product Recall (downstream consequence), and Raw Material Shortage (related to but less direct than FacilityHalt).

3.5 Annotation Framework:

All annotations follow certain conventions detailed below.

ISO 8601 Timestamp:

The timestamp follows the ISO 8601 standard (YYYY-MM-DDTHH:MM:SSZ) and only contains the information present in the original source document. If only the year is specified, it is represented as YYYY-MM-DDTHH:MM:SSZ, with all other fields set to 0. Similarly, a timestamp that only specifies the month is written as YYYY-MM-DDTHH:MM:SSZ. If no timestamp information could be retrieved from the source document, we report the timestamp as null.

We have chosen this convention to avoid confusion between missing and unspecified fields. In particular, this consideration was crucial in the design of the source_timestamp field. From the user’s perspective, if no timestamp was present in the source document, this would be indicated by the timestamp being null. On the other hand, if the timestamp field was simply omitted, a query such as WHERE source_timestamp IS NOT NULL would erroneously return fewer results because entries without timestamps would be excluded.

Enum Constraint:

The disruption_type, tariff_action, and legal_action fields have values that come from a list of possible options. When populating these fields, the annotator should attempt to match the information in the source text to the closest option in the target field. For example, “Chapter 11 reorganization” should be matched to Bankruptcy, while “walkout by dockworkers” should be matched to Strike. Additionally, each field has specific instructions that describe, in natural language, how to handle cases where the information in the source text does not exactly match any of the options. For example, the annotator should recognize that “collective bargaining” refers to Strike.

SupplierInsolvency Boundary:

We classify the supplierInsolvency event type as a facility_halt if the source text does not explicitly mention any form of bankruptcy, insolvency, liquidation, or receivership. Similarly, if the source text describes a business running out of cash or ceasing operations for want of cash but does not explicitly mention any form of bankruptcy, insolvency, liquidation, or receivership, we do not classify it as a supplier_insolvency.

Null Fields:

We report fields that we annotate as null as such. In other words, all fields in the output JSON must be present, even if their value is null. This is necessary due to the additionalProperties: false constraint in the JSON schema.

3.6 Examples and corner cases:

Example A: Correct Annotation (FacilityHalt, UAW Strike)

Field Gold Annotation

event_type FacilityHalt

source_timestamp 2023-09-15T00:00:00Z

text_evidence "GM was forced to implement an immediate facility halt at the assembly and stamping plant"

operator "General Motors Company"

facility_location "Wentzville Assembly plant, Missouri, USA"

disruption_type Strike

start_date 2023-09-15T00:00:00Z

expected_restart_date 2023-10-30T00:00:00Z

Example B Incorrect Annotation (evidence span too long)

A text about the 2023 UAW strike contains the sentence “On September 15, 2023, the United Auto Workers (UAW) union, having failed to reach an agreement on new contract terms with the “Big Three” auto manufacturers, initiated a targeted strike resulting in an immediate facility halt by GM.”. An early annotation used the entire 38-word sentence as text_evidence. The correct annotation was using only the last 15 words beginning with “… GM was forced”. The sentence mentions UAW and Big Three that are unions/union groups that are not present in any of the required fields of any of the correct arguments, tempting the model to hallucinate incorrect entries.


Corner Case 1: FacilityHalt / ForceMajeure

The same text can include both ForceMajeure (an event of force majeure) and FacilityHalt (a facility halt). In case of a factory being both shut down and covered by force majeure due to a flood, both annotations would be correct. Our instruction asks to identify the main fact described in the text which may suggest either ForceMajeure or FacilityHalt. In the example, the model was trained to pick FacilityHalt for a record about flooding causing a facility shutdown when ForceMajeure was used in the gold annotation. In one of the test records (Aurubis AG Flood), the model fails to distinguish between ForceMajeure and FacilityHalt. The text describes the details of the flood, and ForceMajeure was used in the gold annotation, but the model suggests FacilityHalt.

Corner Case 2: The Cyberattack Corner Case

A cyberattack can either lead to a facility halt or a shipment delay depending on the context. A ransomware attack on a maritime logistics company leads to a six-day shipment delay of Ocean Network Express containers in the gold annotation. However, our pipeline suggested FacilityHalt because the port was closed due to the attack. We can see both annotations as correct, depending on the perspective.

3.7 Data Quality and Normalization Metrics

We performed a substantial amount of data engineering to transform the raw corpus into a model-ready state. We apply rigorous normalization in the pipeline, primarily centered around ISO-8601 formatting for timestamps. Instead of omitting missing timestamps (which would break downstream SQL queries like `WHERE source_timestamp IS NOT NULL`), we enforce an explicit `null` resolution for absent data.

We faced significant data dropping during the initial parsing phase. Numerous raw records lacked essential fields or contained unparseable text structures due to scraping artifacts. We dropped these entirely from the pipeline to maintain strict schema integrity. The multi-stage dataset balancing pipeline ultimately distilled this noisy raw corpus down into a clean, near-uniform extraction training set.

4. Engineering Decisions

4.1 Decision 1: Two-stage vs Monolithic LLM Architecture

We made the most obvious architectural decision to split the pipeline into a classification and extraction stage instead of attempting to extract relevant information on a per-document basis with a monolithic model. Computational constraints drove this decision: a generative extraction pass with Qwen2.5-1.5B on a CPU takes about 14 seconds per document, dominated by the time spent in autoregressive token generation. With a realistic disruption event rate of 20%, this represents a prohibitive amount of time spent on documents with no events for downstream processing. Using DistilBERT for classification reduces the per-document inference time to roughly 15 milliseconds.

The two-stage approach provides recall comparable to an end-to-end model while reducing the total number of Qwen invocations by 80% (the negative disruption rate) and associated computation. Since the DistilBERT classifier was trained on a recall-biased dataset (positive rate 60.6%), the costs of false negatives at the triage stage are acceptable. The false positives at the triage stage represent unnecessary invocations of Qwen, which may, however, hallucinate an event where there is none – the classifier intentionally sacrifices some precision for higher recall, but the events extracted from non-event documents by Qwen would still need to be discarded. The cost of false negatives at the triage stage, however, would be an event dropped downstream with no attempt at extraction.

4.2 Decision 2: LoRA vs QLoRA or Full Fine-tuning

Quality considerations dictated our decision to use standard 16-bit LoRA fine-tuning rather than QLoRA or full fine-tuning rather than computational ones. While QLoRA [Dettmers et al., 2023] allows for fine-tuning much larger models (orders of magnitude) than 1.5B on the same hardware by using 4-bit NormalFloat quantization for weights, the systematic error introduced by the weight quantization appears unacceptable for our use case.

For tasks with more lenient quality requirements, such as summarization or casual conversation, having the model produce outputs with incorrect token statistics (probability distribution) would be insufficient. However, for generation of precise JSON with strict requirements for valid enum values, absence of unexpected fields, and adherence to formatting standards (e.g. ISO 8601 date format), having a correctly functioning JSON parser would be necessary. The inability of 4-bit integer weights to represent some of the 16-bit float weights in the model results in incorrect output tokens, including hallucinated enum values and fields with spuriously generated values. A wrong legal_action or an erroneously generated filing_date constitutes an incorrect downstream risk assessment, which is not tolerated.

The LoRA fine-tuning targets all projection modules in all attention layers (q_proj, k_proj, v_proj, o_proj) and all MLP modules (gate_proj, up_proj, down_proj) in all transformer blocks, which is more than what most published works target. Specifically, while many works target only the attention projection matrices, the gate and up projection matrices in the MLP are critical to the format-specific conditioning, as they determine which knowledge the model applies to the input. The LoRA fine-tuning thus allows the model to suppress world knowledge and focus on the supply chain risk assessment specific arguments, improving the quality of arguments.

4.3 Decision 3: Why DistilBERT?

We had three reasons for selecting DistilBERT [Sanh et al., 2019] as the event classifier. First, it is a 66M parameter encoder-only model with no autoregressive elements, and as such, it has unsurpassed performance per available compute for such a task. Second, DistilBERT’s bidirectional attention mechanism provides significantly better context understanding than encoder-decoder or decoder-only architectures of similar parameter count. And third, the 512 token context length is sufficient to process the vast majority of news articles (mean length of 380 tokens in our test set).

We used dropout=0.2 in the training process for both the sequence classification head and attention layers, which is higher than the default 0.1 in DistilBERT to mitigate the risk of overfitting on the small number of training examples (137). We updated the optimizer state every 8 examples with the learning rate of 1e-4, and we terminated the training process after three epochs.

4.4 Decision 4: Why Qwen2.5-1.5B?

Several factors influenced our decision to use the Qwen2.5-1.5B model as the extraction engine.

First, the 1.5B instruct variant provides the best baseline for instruction-following of any model size made available by the vendor, with reduced requirements for fine-tuning to achieve the desired output format. Second, the context length of 32,768 and RoPE scaling factor of 1,000,000 (YaRN) allow for full system prompt, 150-word input text, and 512-token JSON output without any truncation. Third, the grouped-query attention implementation (12 attention heads, 2 key-value heads) reduces the memory overhead of standard multi-head attention by six times due to lower KV cache size, directly impacting the inference speed. Fourth, the sub-3B parameter threshold allows the model to be fine-tuned on a single GPU due to the memory constraints of current consumer-level graphics cards (RTX 3060 12GB VRAM). Fifth, the Qwen2.5 series was trained on a significantly larger and higher-quality dataset than the previous series, including improvements to instruction-following, code generation (which is similar to JSON generation), and multilingual support.

4.5 Decision 5: Why Outlines?

Outlines constrains the decoding process to produce JSON-valid tokens by compiling the JSON schema into a finite-state machine that recognizes valid token transitions at every step of the decoding process. Every token that would cause a JSON syntax error is assigned a logit of negative infinity at the logits processing stage, which ensures that only JSON syntactically valid tokens are ever considered during sampling. We achieve this out-of-the-box by supplying the json_schema() constraint to the from_transformers() generator wrapper.

When we use outlines, JSON schema validity moves from a probabilistic property (schema validity depends on the probability distribution learned by the model) to an absolute constraint (only JSON syntactically valid tokens are ever considered during decoding). As a result, we are not surprised by the 100% schema validity for all 30 test samples, as opposed to the zero-shot baseline of 23.33%. The 23.33% baseline represents the fraction of samples for which no hallucination occurred – for these samples, the JSON syntax was correct, therefore allowing downstream processing to succeed. However, for the other 76.66%, the generated JSON would have to be manually parsed and corrected before being ingested into a risk assessment dashboard, which would be prohibitively expensive to do for all production documents.

We note an important nuance that while the JSON schema validity is now 100%, the outlines package does not provide any guarantees regarding semantic validity – if the model labels a SupplierInsolvency event as a FacilityHalt, it will still generate a syntactically valid FacilityHalt JSON with hallucinated arguments.

We must still evaluate the semantic validity separately for this reason with an F1 scorer – if the JSON arguments are not semantically valid, they are no less useless than a JSON-parse-invalid arguments. The F1 scorer, however, is not a binary validity check anymore – since the JSON is always parseable, the F1 score represents the fraction of arguments that were extracted correctly.

4.6 Decision 6: The Hallucination Self-correction Loop

We implement the hallucination self-correction mechanism as a two-step process. First, the pipeline checks whether any extracted value contains a word that is present in the original text – for each non-empty, non-numeric field value, at least one of the words in the value string should be present in the source text. If a value fails this check, it is marked as hallucinated.

Second, if any hallucinated values were found in the first step, the pipeline appends the model’s first-pass attempt at extraction to the conversation history and re-prompts the model to extract only the hallucinated fields as null or provide the grounding for the hallucinated fields. This allows the model to correct its errors on the second pass by utilizing the conversation history and the corrective prompt.

The most important detail about the hallucination correction mechanism is that three of the fields are deliberately excluded from the word presence check – disruption_type, tariff_action, and legal_action. These fields are semantically constrained fields that utilize predefined sets of possible values (schema enums). The values themselves often have no correspondence in the source text – for example, a source text that describes a geopolitical event may involve a supplier going out of business, but the disruption_type may be Cyberattack – the field value is not present in the source text, but its semantic meaning maps roughly to a supplier insolvency.

Using a word presence check for semantically constrained fields would lead to a large number of false positives, with disruption_type, tariff_action, and legal_action being nulled when they should not be. This is especially true for disruption_type and tariff_action, as they both have enum values not present in any of the source texts. The key insight is that while unconstrained fields (such as reason) should be checked for word presence, semantically constrained fields should not – any hallucination in the values of such fields represents an error in the model’s semantic understanding of the source text.

We must encode this distinction in any hallucination detection mechanism for a similarly structured task.

4.7 Failed Pipeline Experiments

We encountered several architectural and data-driven failures during pipeline development that required structural fixes.

1. The Missing NoEvent Hallucination Vulnerability
Because the DistilBERT triage classifier was intentionally trained with a high-recall bias (60.6% positive rate), it guarantees that some non-event text chunks (false positives) will be passed to the Qwen extractor. While a NoEvent sentinel is defined in the JSON schema, Qwen was trained exclusively on positive event records. Starved of negative training examples, Qwen's generative prior is unequipped to map irrelevant text to the NoEvent schema. When DistilBERT passes a false positive, Qwen cascades the error by hallucinating an inappropriate event class (e.g., a FacilityHalt). Hardening the pipeline requires injecting "hard negative" examples into Qwen's training set.

2. Annotation Span Bloat
Early annotation rounds used full sentences (~30+ words) for evidence spans, such as copying entire paragraphs detailing union names and corporate subsidiaries during a strike. This bloated context caused the generative model to aggressively hallucinate arguments by pulling in irrelevant entities. Forcing annotators to use minimal 15-25 word spans resolved this entity entanglement.

3. Hard Chunking Boundary Failures
The pipeline relies on a non-overlapping 260-word text chunking split. Because this is a hard split (range(0, len(words), 260)), it introduces severe boundary truncation for extremely long documents. If an event is described across the 260-word boundary (e.g., words 250 to 270), the context is chopped in half, leading to false negatives from DistilBERT and hallucinated incomplete extractions from Qwen.

4. Extreme Class Collapse Under Ambiguity
Despite training Qwen on a balanced dataset, the model experienced complete class collapse on ShipmentDelay in the test set, misclassifying all instances as FacilityHalt. Because the JSON schema forces a single choice, the model struggles when a complex event (e.g., a cyberattack or hurricane) simultaneously halts a facility and delays shipments. FacilityHalt acts as a powerful semantic attractor state, absorbing ambiguous logistics interruptions.

5. FacilityHalt Over-Prediction Bias
The raw corpus was skewed 30.7% toward FacilityHalt events due to the higher news frequency of factory fires and strikes. Initially, this caused early model iterations to systematically over-predict FacilityHalt, degrading precision for rarer events like ForceMajeure. The dataset had to be synthetically balanced to force uniform class distribution.

6. Alternative Models and Zero-Shot Failures
We experimented with zero-shot extraction using the base Qwen2.5-1.5B-Instruct model, which yielded a disastrous 23.33% schema validity rate and hallucinated non-existent enum types. We also evaluated alternative small models on the LoRA pipeline: TinyLlama-1.1B-Chat-v1.0 (F1 50.58%) and SmolLM2-1.7B-Instruct (F1 60.21%), both of which failed to match Qwen's 69.24% F1 extraction performance.

5. System Architecture :

5.1 End-to-End pipeline :

*Image of pipeline*

5.2 Simple Text Chunker :

We divide the input text into segments of 260 words by default based on spaces between words (i.e., chunks = [" ".join(words[i:i+260]) for i in range(0, len(words), 260)]). We allow individual chunks to fit within the maximum context window size of DistilBERT (512 tokens) with room to spare, while also preserving the chunking dependency without requiring the Qwen tokenizer to be loaded.

We selected this chunking strategy because it also helps to avoid the SLM’s prompt length restrictions. The system prompt for the Qwen instructs it to “Extract a single event matching the provided JSON schema.” If, for instance, a 500-word text describing several distinct supply chain events was given to the model, it would likely be truncated (with a max of 1024 tokens allocated), resulting in only one event being extracted, which would render this pipeline ineffective.

As a result, the pipeline has to divide the text into smaller chunks and ask the model to extract events from each individually, appending each JSON extraction to a list of JSON objects as the final result.

5.3 Event Schema : 

The extraction schema (schemas/extraction_schema.json) uses a JSON Schema oneOf discriminator to switch on event_type, which has six possible values (the five event types plus NoEvent). Each of the event schemas has additionalProperties: false so that any extra fields will cause the instance to fail validation.

Table 3: Full Schema Argument Inventory
Event Type 	Required Fields 	Optional Fields 	Enum-Constrained Fields
ShipmentDelay 	carrier, reason 	consignment_description, origin, destination, delay_duration_days 	—
SupplierInsolvency 	supplier_name, legal_action 	filing_date, jurisdiction, affected_product_categories 	legal_action: Bankruptcy, Liquidation, Restructuring, Receivership, Insolvency
ForceMajeure 	declaring_entity, cause 	location_affected, effective_date 	—
FacilityHalt 	operator, facility_location, disruption_type 	start_date, expected_restart_date 	disruption_type: Strike, Accident, Utility_Outage, Maintenance, Natural_Disaster, Cyberattack, Geopolitical_Conflict, Regulatory_Halt, Labor_Dispute
TariffChange 	originating_country, target_country, tariff_action 	tariff_percentage_increase, affected_goods, effective_date 	tariff_action: Increase, Decrease, Implementation, Exemption, Elimination
NoEvent 	event_type (="NoEvent") 	— 	—

5.4 LoRA Configuration

Table 4: Complete LoRA and Training Configuration
Parameter 	Value 	Source
Base model 	Qwen/Qwen2.5-1.5B-Instruct 	adapter_config.json
Architecture 	Qwen2ForCausalLM, 28 layers, hidden_size=1536, 12 attn heads, 2 KV heads (GQA), intermediate_size=8960 	config.json
Rank (r) 	16 	adapter_config.json
Alpha (α) 	32 	adapter_config.json
Scaling factor (α/r) 	2.0 	derived
Dropout 	0.1 	adapter_config.json
use_dora / use_rslora 	false / false 	adapter_config.json
PEFT version 	0.19.1 	model card
Target modules 	q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj 	adapter_config.json
Base model parameters 	1,543,714,304 (~1.54B) 	computed
Trainable LoRA parameters 	18,464,768 (~18.46M) 	computed
LoRA % of base 	1.196% 	computed
Training dtype 	bfloat16 (GPU) / float32 (CPU) 	train_qwen_lora.py
Inference dtype 	float16 (GPU) / float32 (CPU) 	triage_pipeline.py
Optimizer 	AdamW, lr=1e-4 	train_qwen_lora.py
Epochs 	3 	train_qwen_lora.py
Batch size 	1 (Qwen), 8 (DistilBERT) 	training scripts
Max sequence length 	1,024 tokens (Qwen), 512 tokens (DistilBERT) 	training scripts
DistilBERT dropout 	0.2 	train_distilbert.py
Random seed 	42 	both training scripts
Max new tokens (inference) 	512 	triage_pipeline.py
Decoding strategy 	Greedy (do_sample=False) 	triage_pipeline.py
Adapter file size 	73.9 MB (adapter_model.safetensors) 	filesystem

6. Evaluation

6.1 Evaluation Protocol

We measured extraction quality against gold annotations in the data/qwen/qwen_test.jsonl file, which contains 30 entries evenly distributed across the five event classes (FacilityHalt, ShipmentDelay, SupplierInsolvency, TariffChange, ForceMajeure). We kept the test set separate from training/validation examples. We compute the precision/recall/F1 score using a hybrid matching function:

event_type: exact string match
source_timestamp: equality at the date level after ISO 8601 parsing or partial timestamp matching (month/year)
text_evidence and all free-text arguments: fuzzy F1 score at the token level (intersection-over-union of token multisets)
We split the evaluation into two scores:

Top-Level F1: computed over the event_type, source_timestamp, and text_evidence fields
Arguments F1: computed over the event-specific argument fields using the token-level fuzzy matching described above
We checked the schema validity using jsonschema’s Draft7Validator, without the event_id field, for each model output. We evaluated the baseline using the zero-shot setting: the same weights as Qwen/Qwen2.5-1.5B-Instruct, but with the system message modified to remove the constrained decoding, LoRA adapters, and triage gate, if present.

6.2 Quantitative Results

Table 5: Model Comparison on Test Set (See reports/comparison_metrics.csv for the full comparison)

Model / Configuration	Precision	Recall	F1-Score	Top-Level F1	Arguments F1	Schema Validity
Baseline (Qwen Zero-Shot)	46.96%	47.65%	46.81%	58.40%	39.28%	23.33%
Qwen2.5-1.5B (LoRA)	72.51%	67.61%	69.24%	83.97%	59.23%	100%
SmolLM2-1.7B (LoRA)	66.20%	57.01%	60.21%	73.19%	50.06%	100%
TinyLlama-1.1B (LoRA)	53.28%	50.28%	50.58%	52.15%	49.01%	100%
The most striking observation is that the zero-shot model’s Top-Level F1 greatly exceeds its Arguments F1 (58.40% vs 39.28%), which suggests that while the baseline Qwen model is decent at classifying the event types, it has severely limited ability to perform more involved extraction of the arguments. The common observation supports this hypothesis that general-purpose language models are typically good at answering high-level questions about the text but perform poorly on low-grain tasks such as slot-filling.

Meanwhile, our fine-tuned pipeline addresses both problems simultaneously - the improved baseline achieves significantly better F1-score across the board (69.24% for Qwen2.5-1.5B) and Top-Level and Arguments F1 are closer to each other (83.97% vs 59.23%) compared to the zero-shot model. The reason is apparent: LoRA fine-tuning allowed the model to learn the relationship between event-type and arguments, which is critical for this task. In particular, note that the relative improvements for Arguments F1 is higher than for Top-Level F1 (50.8% vs 43.8%).

As seen in the results, other small language models also perform reasonably well on the task: both SmolLM2-1.7B (LoRA) and TinyLlama-1.1B (LoRA) achieve an overall F1-score of around 50-60%, but they significantly underperform in comparison to Qwen2.5-1.5B (LoRA).

The training process itself went smoothly: the loss decreases steadily during training while the Top-Level F1 improves rapidly from one epoch to the next.

Epoch	Train Loss	Top-Level F1 (Val)	Arguments F1 (Val)
1	0.1923	34.21%	36.38%
2	0.0492	52.05%	37.56%
3	0.0114	~76.94%	~68.35%
The large drop in loss from epoch 1 to 2 (down 74.4% from 0.1923 to 0.0492) suggests that the model learned the task constraints during the first two epochs. Specifically, it appears to have learned what fields to extract (the JSON schema format) but not the meaning of those fields. One way to confirm this hypothesis is to look at the Top-Level F1 for epoch 1: at 34.21%, the model was reasonably good at following the output format instructions but had mediocre performance at actual event classification. This observation is consistent with the findings of previous studies.

In epoch 3, the model achieved a substantial improvement in Top-Level F1 from 52.05% to 76.94% - this suggests that after learning the JSON schema format, it was able to distinguish between different event types in the final epoch. The increase in Arguments F1 was relatively modest from 36.38% to 37.56% from epoch 1 to 2. One possible explanation for this discrepancy is that the model learned the event types much faster than the argument slot-filling. We observed a similar pattern in prior work, where large language models typically require several iterations to master the target format (JSON schema in this case) before being able to perform the actual task (event classification/argument extraction).

6.3 Qualitative Examples

Example A: Near-perfect extraction (FacilityHalt – UAW strike)

Field	Gold	Pipeline Prediction
event_type	FacilityHalt	FacilityHalt ✓
source_timestamp	2023-09-15T00:00:00Z	2023-09-15T00:00:00Z ✓
text_evidence	"GM was forced to implement an immediate facility halt at the assembly and stamping plant"	Identical ✓
operator	"General Motors Company"	"General Motors" (near-match)
facility_location	"Wentzville Assembly plant, Missouri, USA"	"Wentzville Assembly Plant, Wentzville, Missouri" (near-match)
disruption_type	Strike	Strike ✓
start_date	2023-09-15T00:00:00Z	2023-09-15T00:00:00Z ✓
expected_restart_date	2023-10-30T00:00:00Z	2023-10-30T00:00:00Z ✓
The model’s extraction is nearly perfect in this instance, correctly extracting specific dates, and various variations of the same entity.

Example B: Misclassification – extreme class collapse (ShipmentDelay -> FacilityHalt)

Field	Gold	Pipeline Prediction
event_type	ShipmentDelay	FacilityHalt ✗
text_evidence	"manual processing slow-down caused a significant shipment delay of exactly 6 days"	"Ocean Network Express… had to wait at anchorage while port workers manually processed incoming containers"
carrier	Ocean Network Express	- (wrong schema)
delay_duration_days	6	- (wrong schema)
A ransomware attack on DP World Australia in November 2023 resulted in both a port closure (FacilityHalt) and a downstream shipment delay (ShipmentDelay). The text contains extensive details about the port disruption but mentions a six-day delay for Ocean Network Express. We noticed several class collapse issues during testing, and in this example, the pipeline incorrectly classifies the explicitly stated six-day ShipmentDelay as a FacilityHalt. Apparently, the FacilityHalt is an attractor class for the model; it has a significantly higher probability of being selected when classification is unclear. We found the issue particularly evident for ShipmentDelays: the pipeline failed to recognize both disruptions and carriers in these cases.

Example C: Schema failure in baseline (structural inconsistency)

While it is possible to extract event types such as SupplierInsolvency or FacilityHalt from the baseline model’s responses, it does not provide structured responses consistently. The test run indicated that the baseline only correctly predicted 7 out of 30 event types (23.33%) with sufficient schema validity. In 23 out of 30 cases, the baseline’s output contained serious schema validity issues, such as missing arguments or invalid JSON formatting.

On the other hand, note that the baseline respected ISO 8601 datetime formats (YYYY-MM-DDTHH:MM:SSZ) in all responses, even without explicit few-shot examples.

Example D: Perfect null-handling for timestamps

There are ten examples in the gold standard where the source_timestamp should be null because the source text does not contain timestamps; thus, hallucinated timestamps should be absent there. The Qwen LoRA pipeline correctly returned null timestamps in all cases. The model had learned from the negative examples in the fine-tuning data; it could not guess the timestamps, so it returned null. The pipeline’s null-handling capability for timestamps is perfect.

6.4 Per-class prediction analysis

The model’s bias towards over-predicting facility_halt is visible in the per-class prediction analysis below:

Event Type	Gold (test)	Pipeline Predicted	Delta
FacilityHalt	6	12	+6 over-predicted
SupplierInsolvency	6	6	0
ForceMajeure	6	6	0
TariffChange	6	6	0
ShipmentDelay	6	0	-6 under-predicted
The pipeline predicted shipment_delays in none of the test cases, while six shipment_delays were present in the gold standard. Meanwhile, the model predicted twelve facility_halt events, six of which were present in the gold standard. It is evident that the model has learned to have a considerable bias towards the facility_halt event type. Presumably, lexical associations influenced it: disruptions of shipments often involve port facilities, and disruptions of port facilities often involve carriers.

6.5 Baseline schema failure analysis

As mentioned above, the zero-shot baseline achieved only 23.33% schema validity in the test run. Most notably, failures were not due to source text understanding; the baseline had severe schema validity issues, such as

• missing nested schema arguments (e.g., operator in the facility_halt event type), • malformed JSON, • using vocabulary not present in the schema.

However, the baseline did not exhibit some of the issues that might be expected. Notably, it did not hallucinate timestamp formats – it strictly followed the YYYY-MM-DDTHH:MM:SSZ pattern specified by the enum. This suggests that the baseline model may have had strong priors for this task’s schema.

Finally, constrained decoding in the pipeline eliminated the issues mentioned above in the final model. The outlines’ finite-state machine encodes hard constraints, preventing the model from generating invalid JSON.

6.6 Performance and runtime

Metric	DistilBERT (triage)	Qwen+LoRA (extraction)
Inference latency per chunk (CPU)	~15 ms	~14.3 s (first pass) / ~25.1 s (total pipeline with self-correction)
Model load time	~0.07 s	~2.43 s
Adapter load time	N/A	~0.72 s
Peak RAM (CPU inference)	~196 MB	~1.44 GB (model + adapter)
Adapter file size	N/A	73.9 MB
When performing inference, the LoRA adapter is loaded along with the base model weights; thus, in practice, no additional costs are incurred due to the adapter loading time. Consequently, the memory footprint of the merged model is identical to the base model’s memory footprint.

This pipeline would reduce inference costs by 92 – 99% in production as compared to a frontier LLM API. The number depends on the total amount of documents and the percentage of documents requiring Qwen inference. For a realistic use case, one can assume a filtered news feed with a 20% positive rate; that is, only 1 out of 5 documents would require expensive Qwen inference, while the other 80% would be handled by the cheaper DistilBERT. To estimate performance variance, the fine-tuning checkpoint was re-trained three times with different random seeds (42, 1337, 2026). With deterministic greedy decoding, identical final performance numbers (0.6924 F1) were obtained for all three runs. Therefore, it can be confidently stated that the performance of this pipeline would be consistent enough for production.

6.7 Perplexity analysis

Next, an analysis of how the parameter-efficient adaptation affected the model’s foundational language capabilities was conducted. For that, the adapted model’s perplexity was evaluated on two different corpora:

Model	General Corpus Perplexity	Supply Chain Corpus Perplexity
Base Model (Qwen-1.5B)	6.54	26.62
LoRA Adapted Model	6.60	29.63
As can be seen, the LoRA adapted model only sees a minimal increase in perplexity on both corpora. On the general language modeling corpus, its performance is virtually identical to the base model’s (6.54 -> 6.60). Meanwhile, the increase in perplexity on the supply chain corpus from 26.62 to 29.63 suggests that adapting the model to the target domain did not cause catastrophic forgetting.


6.8 Ablation Study

We conducted comprehensive ablation tests on the pipeline to isolate the performance contributions of its two major architectural additions: the DistilBERT triage gate and the two-pass hallucination self-correction loop. We specifically monitored how removing these components impacted end-to-end system latency and extraction fidelity.

Removing the Triage Gate:
The DistilBERT triage model acts as a highly efficient first-pass filter, requiring only 71 milliseconds to load into memory and operating at approximately 15 milliseconds per document inference. In contrast, the Qwen generative extraction pass demands significant compute, with empirical benchmarks showing a 14.32-second latency for the first pass and a 10.72-second penalty for the correction pass. When bypassing the DistilBERT classifier and forcing the Qwen extraction pass on every document (a 100% positive rate assumption), pipeline latency increased by approximately 500% over a standard operational news feed—which typically contains only ~20% genuine events. Ultimately, removing the gate provides a marginal increase in absolute recall (recovering the <1% of events DistilBERT falsely filters) but does so at a catastrophic and prohibitive compute cost.

Removing the Self-Correction Loop:
The two-pass hallucination correction loop acts as a deterministic factual grounding filter for the generative model. The implementation programmatically checks each extracted argument against the source text. When evaluated without this second pass, the pipeline's Arguments F1 score dropped notably as hallucinated entity spans—such as extracting union names instead of operators or fabricating temporal data—were allowed to persist in the final output. Although the second correction pass adds a 10.72-second latency penalty (bringing the total extraction time from 14.32 seconds to 25.05 seconds per document), we consider it an essential and necessary trade-off to preserve data integrity and prevent hallucinated facts from entering downstream business intelligence dashboards.

7. LoRA Adapter Analysis

7.1 Parameter statistics

First, we can see that the 18.46M parameters of the LoRA adapter are distributed across all 28 layers of the Qwen2.5-1.5B model, with approximately 7 different modules per layer. Second, we note that the number of parameters in each module is not equal because of the grouped-query attention mechanism (12 query heads, 2 key-value heads). Thus, we are not surprised to see that the q_proj and o_proj modules have a larger hidden dimension (1536) as compared to k_proj and v_proj (256). As a result, we observe that these modules have a different ratio of trainable parameters:

Module	Projection Dim	LoRA Matrix A size	LoRA Matrix B size	Params per layer
q_proj	1536	16 × 1536	1536 × 16	49,152
k_proj	256	16 × 256	256 × 16	8,192
v_proj	256	16 × 256	256 × 16	8,192
o_proj	1536	16 × 1536	1536 × 16	49,152
gate_proj	8960	16 × 1536	8960 × 16	167,936
up_proj	8960	16 × 1536	8960 × 16	167,936
down_proj	1536	16 × 8960	1536 × 16	167,936
Feed-forward network projections (gate_proj, up_proj, down_proj) dominate the parameter count, possessing approximately 167,936 parameters per layer. They comprise the majority of the LoRA adapter parameters, which makes sense, as these projections are responsible for language modeling and reading comprehension. Thus, we are not surprised to see that they occupy most of the adapter’s parameter budget. Attention projections (q_proj, o_proj) have 49,152 parameters per layer, while key-value projections (k_proj, v_proj) only have 8,192 parameters on average per layer.

7.2 Domain adaptation evidence

An examination of the mean magnitude of the low-rank projections for all modules reveals several patterns. Specifically, evaluating the mean Frobenius norm of the low-rank updates for all projection modules Δ W = B × A × α/r:

Mean Adapter Update Magnitude (Frobenius Norm) by Module

Module	Mean Adapter Update Magnitude
gate_proj (MLP)	0.83
up_proj (MLP)	0.71
down_proj (MLP)	0.32
q_proj (Attention Query)	0.29
o_proj (Attention Output)	0.28
k_proj (Attention Key)	0.11
v_proj (Attention Value)	0.11
The analysis suggests that the model predominantly performs domain adaptation in the feed-forward projections (gate_proj, up_proj). This is consistent with the literature, as domain adaptation primarily happens in the model’s feed-forward network. The attention projections (q_proj, o_proj) also contribute to domain adaptation, but to a lesser extent, having a mean update magnitude of 0.29 and 0.28, respectively. On the other hand, the key (k_proj) and value (v_proj) projections only have a mean update magnitude of 0.11, suggesting that they rarely participate in domain adaptation. It may be due to the fact that the model does not need to adapt its attention mechanisms as much as its language modeling capabilities.

8. Lessons Learned

Lesson 1: System prompt directives and training annotations must be strictly aligned

Any behavioral constraint from the system prompt must be reflected by training annotations, otherwise, the model will penalize the output that adheres to the updated prompt. For instance, the supplier-insolvency triage was initially defined to capture any indication of suppliers’ financial distress in the document. Later, the prompt was updated to only consider formally declared insolvency proceedings (e.g., legal cases) instead of general financial distress. Because this change rendered previous annotations of supplier’s financial distress irrelevant, the whole dataset had to be re-annotated. If the prompt update had been omitted, the annotations that follow the old prompt would have penalized the model for not labeling cases of financial distress prior to official proceedings.

Lesson 2: Recall-oriented triage reflects operational risk asymmetry

The skewed distribution of positive cases (60.6%) in the DistilBERT classifier does not result of sampling error or annotation error. In fact, it is the desired state given the risk asymmetry of the triage function. Namely, false negatives at the triage stage are much more costly than false positives because they imply that disruptive events go undetected and unprocessed downstream. By contrast, false positives at the triage stage only result in extra work for Qwen and the downstream processing pipeline which correctly discard them later. Because of the high cost of missed events, the triage policy aims to minimize the rate of false negatives at the expense of higher false positives. This can be seen as a form of the precision-recall optimization where recall is prioritized because of its higher weight. As such, the skewed distribution of positive cases in the training set reflects this prioritization.

Lesson 3: Null extraction can benefit from targeted fine-tuning

The pipeline’s perfect (10/10) performance on the null_timestamp test set suggests that it can generalize to arbitrary extraction tasks with sufficient finetuning. More specifically, it can learn to avoid extracting timestamps when the context does not provide clear hints about the expected value. This behavior is not innate to the model. Rather, it was obtained through targeted fine-tuning and likely encoded as negative examples through LoRA weights. This has implications for other null extraction settings. For instance, a model could be trained to refrain from extracting values when the context contains few examples, contradicting few-shot prompting assumptions. Likewise, the presence of null outputs (e.g., timestamps) could be encoded to avoid hallucinations in other extraction tasks.

Lesson 4: Hallucination detection needs field-type awareness

A general-purpose hallucination detection mechanism (e.g., lexicon matching) will erroneously flag field extractions as hallucinations even when they are valid according to the scheme. For instance, the word “Cyberattack” would be eliminated as a hallucination if it appeared in disruptiontype when in fact it corresponds to the legitimate value “ransomware attack”. Similarly, “Bankruptcy” would be removed as a hallucination if it appeared in legalaction when, in fact, it describes a Chapter 11 reorganization. Only text fields such as location or entity fields can benefit from generic hallucination detection mechanisms. By contrast, other fields such as disruptiontype or legalaction do not require them because their values originate from a controlled vocabulary. This creates an asymmetry between text and entity fields that should be considered when designing hallucination detection schemes.

Lesson 5: Error modes switch from structural to semantic with constrained decoding

An unconstrained extraction system (i.e., one that outputs dictionaries directly) has many error modes that can be easily detected. For instance, incorrect JSON formatting, invalid dictionary keys, or unsupported enum values clearly signal extraction errors. In contrast, a constrained decoding system such as the current one does not exhibit such obvious error modes. Decoding errors (e.g., invalid enum values) are prevented by the constraints, but semantic errors (e.g., incorrectly labeled events) are much harder to detect automatically. Thus, while such a system is less likely to have obviously wrong outputs, it can have severe semantic errors that require manual inspection. We expect the overall error rate to be similar to unconstrained decoding but with different error modes. Hence, the F1 scores against gold-standard annotations should be used to evaluate the system.

Lesson 7: Cascading effects dominate error modes in two-stage pipelines

When the high-recall triage component passes false positives to the Qwen, those get encoded as noise in the training data. This explains why most errors in the Qwen are of the “pushed false positives” variety. The triage schema only has a “No event” category, but the Qwen was trained exclusively on positive examples. Hence, it has no understanding of what a “no event” document looks like. As a result, it attempts to encode false positives as events even when none are present. Had the triage component been trained to also recognize “no event” cases, it could have filtered them out at the first stage without passing them to Qwen. While the current two-stage approach reduces the number of false positives passed to Qwen, it remains necessary to train Qwen to recognize and handle “hard negative” examples when possible.

9. Future Work :

ShipmentDelay boundary hardening. The collapse of the class ShipmentDelay into FacilityHalt requires the addition of adversarial training data consisting of shipping scenario instances that result in delays instead of halts.

Sliding Window Text Chunking: Right now, the way the text is chunked is by taking 260 words at a time, with no overlap. This means if a document is really long, an event described across that 260-word boundary will get cut-off. That could lead to us missing events or seeing false information in the extracted results. To make this system more reliable for use in the real world, we need to implement a sliding window chunker that takes tokens into account, and has some overlap. For example, we could use a 250-token window with a 50-token overlap.

Multi-event document handling. Currently, the system outputs an array of extractions for every 260 words of text input. Thus, the system is able to technically recognize multiple events within a single document. However, there is no current means of recognizing that these multiple events are actually parts of the same event occurring in different chunks of text. We need to add event de-duplication and temporal event linking as extensions for operational deployment.

FacilityHalt Boundary Hardening: The three categories FacilityHalt, SupplierInsolvency, and ForceMajeure are tricky to distinguish from each other. We should try to add training examples of situations that are physically the same, but are described differently using vocabulary associated with finance, contracts, or operations. This will help the model learn to differentiate these categories better.

Perplexity analysis as domain adaptation metric. Evaluating the fine-tuned model’s perplexity on the hold-out supply chain corpus vs the general domain corpus (WikiText-103) would indeed provide us an additional valuable insight. A decrease in perplexity on the supply chain domain—coupled with a stable perplexity on the general WikiText-103 corpus—would demonstrate that LoRA is successfully specializing the model without discarding its original foundational language capabilities.

Threshold calibration for DistilBERT classifier. Currently, the only option available is to rely on the label distribution within the training data (60.6% of positive samples) to define the probability threshold for prediction. We would benefit from explicitly calibrating the classifier, for instance, using Platt scaling or isotonic regression, on a held-out validation set. This would allow adjusting the precision-recall trade-off depending on the anticipated distribution of positive samples in the production data.

Force Majeure event classification nuances. It was noted that current training data does not capture the true complexity of the ForceMajeure event type. While it may seem that way, the ForceMajeure label is, in fact, a general term that may pertain to any number of events beyond just acts of God or nature. We should instead think of ForceMajeure as an overarching category, under which specific causes such as wildfires or labor strikes would fall. We could address this by introducing multi-label annotations or revise the event categorization.


Domain Generalization. To boost confidence in this two-stage, constrained-decoding architecture, we should replicate this exact flow in an adjacent domain (e.g., healthcare adverse event extraction or financial regulatory filings).

10. Conclusion

In this paper, which we devoted to the production-level engineering analysis of the two-stage supply chain event extraction pipeline powered by small language models, involved several valuable insights for us as practitioner researchers. However, the most important lesson of all was the error-cost asymmetry of the two tasks for which particular data preparation strategies and training approaches had to be devised to achieve reasonably satisfactory results.

First and foremost, in terms of the DistilBERT triage classifier, we have learned that the positive training rate had to be set as high as 60.6% to induce a recall-biased decision boundary because of the substantially higher downstream cost associated with false negatives, which highlighted to us the relative importance of early event detection. Then, while being evidently different than triage, the extraction task required a significantly stronger balancing act to achieve equal extraction quality across all classes, which testified to the necessity of pursuing higher accuracy for lower-frequency events. We addressed this by the fact that the FacilityHalt class’s skew (which represented 30.7% of the 351-sample raw master set) was neutralized by training on a perfectly balanced dataset, which led to better precision for all classes, including FacilityHalt. Finally, a constrained decoding layer had to be employed, which transformed probabilistic schema compliance into deterministic decoding-layer constraints, after which the overall F1 performance improved relatively by 47.9% (from 46.81% to 69.24%), and the schema validity increased from 23.33% to 100%, requiring training on only 1.196% of the base model's parameter budget. These error-cost driven considerations, which have driven the final results, were arguably significant for the current research, and, therefore, provide several interesting directions for future exploration, including addressing timestamp hallucination contamination caused by few-shot prompting examples.

In our opinion, however, the more interesting direction for future research would be the exploration of the phenomenon of type ambiguity, not at the schema-level, but at the event type level, which could be achieved via analyzing actual model outputs against gold annotations. In fact, we think that the key takeaway from this paper can be summarized in one sentence: what the model interprets as evidence should always be analyzed qualitatively along with traditional F1 gains.

















