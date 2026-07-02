# Supply Chain Event Extraction: Annotation Guidelines

This document provides the standard operating procedures and guidelines for human annotators labeling texts for the supply chain disruption event extraction dataset. It is strictly enforced during prompt engineering and model evaluation.

## 1. Core Principles
* **Explicit Extraction Only:** You must extract only facts directly stated in the text. Do not infer, guess, or extrapolate based on world knowledge.
* **Null Handling:** If a required field is not explicitly mentioned in the text, it must be mapped to `null`.
* **Minimal Evidence Span:** The `text_evidence` field must contain the *smallest possible contiguous substring* from the source text that directly supports the event. Do not extract entire sentences if a 3-10 word phrase suffices, but ensure it captures the core event logic (typically 15-25 words based on empirical F1 optimization).

## 2. Event Types and Separation Rules
* **FacilityHalt:** Used for physical suspensions of operations (e.g., maintenance, strikes, natural disasters, regulatory shutoffs). *Do not use this for bankruptcies.*
* **ShipmentDelay:** Used for transport and transit delays involving a carrier, origin, and destination.
* **ForceMajeure:** Strictly for legal declarations of force majeure by corporate entities or governments.
* **SupplierInsolvency:** Strictly for financial distress, bankruptcy, or legal restructuring filings.
* **TariffChange:** Used for changes to customs duties or tariffs on imported/exported goods.

*Concept Mixing Rule:* If an event implies multiple types (e.g., a strike causing a shipment delay), annotate based on the primary explicit subject of the text. If the event is too ambiguous, then annotate based on any one of the events that occur.

## 3. Date Standardization (ISO 8601)
All extracted dates must be standardized into the ISO 8601 format: `YYYY-MM-DDT00:00:00Z`.
* **Full Date Provided:** `2025-07-14` $\rightarrow$ `2025-07-14T00:00:00Z`
* **Month and Year Only:** Default to the first day of the month. E.g., `July 2025` $\rightarrow$ `2025-07-01T00:00:00Z`
* **Year Only:** Default to the first day of the year. E.g., `2025` $\rightarrow$ `2025-01-01T00:00:00Z`
* **No Date Provided:** Map the field to `null`. Do not attempt to guess based on the publication date unless it is explicitly referenced (e.g., "today").

## 4. Enum Constraints
Certain arguments are restricted to a closed list of enumerations. Annotators must map the text's description to the closest matching enum. For example, `disruption_type` in `FacilityHalt` must be one of:
`["Strike", "Accident", "Utility_Outage", "Maintenance", "Natural_Disaster", "Cyberattack", "Geopolitical_Conflict", "Regulatory_Halt", "Labor_Dispute"]`.
