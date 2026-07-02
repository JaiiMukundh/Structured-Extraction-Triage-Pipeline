# Dataset Collation & Refining Strategy

Building a high-quality dataset is the most critical component of a structured extraction pipeline. Because this architecture relies on a **Two-Stage Triage** system, the dataset required two parallel forms of annotation, strict class balancing, and rigorous split-alignment to prevent data leakage.

## 1. Dual-Objective Annotation
The data pipeline was designed to train two distinct models simultaneously. I curated texts to serve both stages:
* **The Triage Stage (DistilBERT):** Required binary labels (`1` for an active disruption, `0` for noise).
* **The Extraction Stage (Qwen-LoRA):** Required complex, schema-compliant JSON payloads for the positive events.

By labeling the positive events with their JSON schema payloads and leaving the negative events with `null` payloads, we created a master unified dataset (e.g., `splittable_redo.jsonl`) that serves both models.

## 2. Hard Negatives & Adversarial Noise
To ensure the 66M DistilBERT model acted as an effective gatekeeper, the dataset was enriched with **Hard Negatives**. These are texts that contain trigger words (e.g., "halt," "delay," "factory") but do not describe an actual supply chain disruption. 

**Examples included in the dataset:**
* Routine corporate PR: *"General Motors announced the routine appointment of a new VP of Global Logistics... routes remain fully active."*
* Expansions: *"Amazon celebrated the grand opening of a fully automated fulfillment center..."*

By injecting these hard negatives, the triage model learned to rely on contextual semantic meaning rather than simply acting as a keyword-matching regex, drastically reducing false positives downstream.

## 3. Strict Class Balancing
Generative LLMs are highly susceptible to class imbalance; if 70% of the training data consists of `FacilityHalt` events, the model will bias heavily toward hallucinating `FacilityHalt` when uncertain. 

To prevent this, the extraction training set was nearly balanced. The `qwen_train.jsonl` dataset contains exactly **115 records, nearly balanced across the 5 event classes** (24 `FacilityHalt`, 24 `ShipmentDelay`, 23 `SupplierInsolvency`, 23 `TariffChange`, and 21 `ForceMajeure`).

## 4. Pipeline Split Alignment (Zero Data Leakage)
A common pitfall in multi-stage pipelines is evaluating Stage 2 (Extraction) on data that Stage 1 (Triage) saw during training, causing data leakage and artificially inflated accuracy scores.

To solve this, the `split_dataset.py` script was written to process both datasets simultaneously. It uses a text-matching hash function (stripping out whitespace and normalizing smart quotes) to align the binary triage dataset (`distilbert_base.jsonl`) with the generative dataset (`qwen_base.jsonl`). 
Once aligned, it performs a strict **70% / 15% / 15%** Train/Val/Test split across the unified list of 196 total records, resulting in:
* **Train split:** 137 records (which filters down to 115 active event records for Qwen)
* **Validation split:** 29 records (which filters down to 20 active event records for Qwen, or 19 in file due to newline handling)
* **Test split:** 30 records (which filters down to 28 active event records for Qwen)

This split guarantees that:
1. DistilBERT and Qwen evaluate their accuracy on the exact same holdout test set.
2. Absolutely no text from the test set ever leaks into the training phase of either model.

## 5. Dataset Sources

The raw data sources curated to construct the dataset were derived directly from historical real-world disruption events across industries. Below is the full list of sources modeled:

1. Weather and Natural Disasters

    2021 Texas Winter Storm (Winter Storm Uri):

        https://en.wikipedia.org/wiki/2021_Texas_power_crisis

    2011 Tohoku Earthquake & Tsunami:

        https://en.wikipedia.org/wiki/2011_T%C5%8Dhoku_earthquake_and_tsunami

        https://en.wikipedia.org/wiki/Toyota_Motor_Kyushu

    2011 Thailand Floods:

        https://en.wikipedia.org/wiki/2011_Thailand_floods

    2021 European Floods (Germany):

        https://en.wikipedia.org/wiki/2021_European_floods

    Mississippi River Floods:

        https://en.wikipedia.org/wiki/Great_Flood_of_1993

        https://en.wikipedia.org/wiki/Cargill (used in connection with 2011 floods)

    Eyjafjallajökull Volcanic Eruption (2010):

        https://en.wikipedia.org/wiki/Air_travel_disruption_after_the_2010_Eyjafjallaj%C3%B6kull_eruption

    2018 European / Rhine River Droughts:

        https://en.wikipedia.org/wiki/2018_European_heatwave

    Typhoon Lekima (2019):

        https://en.wikipedia.org/wiki/Typhoon_Lekima

    Hurricane Sandy (2012):

        https://en.wikipedia.org/wiki/Hurricane_Sandy

    2021 British Columbia Atmospheric River Floods:

        https://en.wikipedia.org/wiki/November_2021_Pacific_Northwest_floods

2. Strikes and Labor Disputes

    US Port and Railroad Strikes:

        https://en.wikipedia.org/wiki/2024_United_States_port_strike

        https://en.wikipedia.org/wiki/International_Longshore_and_Warehouse_Union

        https://en.wikipedia.org/wiki/Pacific_Maritime_Association

        https://en.wikipedia.org/wiki/Port_of_Oakland

    Canadian Railroad Strikes & Lockouts:

        https://en.wikipedia.org/wiki/Canadian_Pacific_Railway

        https://en.wikipedia.org/wiki/Canadian_National_Railway

    United Auto Workers (UAW) Strikes:

        https://en.wikipedia.org/wiki/2023_United_Auto_Workers_strike

        https://en.wikipedia.org/wiki/2019_General_Motors_strike

    Refinery Strikes (US & France):

        https://en.wikipedia.org/wiki/2015_United_Steel_Workers_Oil_Refinery_strike

        https://en.wikipedia.org/wiki/TotalEnergies (CGT refinery strikes in France)

    European Transportation Strikes:

        https://en.wikipedia.org/wiki/2010_Spanish_air_traffic_controllers_strike

        https://en.wikipedia.org/wiki/2022%E2%80%932023_United_Kingdom_railway_strikes

        https://en.wikipedia.org/wiki/2011_United_Kingdom_public_sector_strikes

        https://en.wikipedia.org/wiki/Central_Organisation_of_Finnish_Trade_Unions

    Aerospace Machinists Strikes:

        https://en.wikipedia.org/wiki/Boeing_Everett_Factory

3. Cyberattacks and IT Outages

    Colonial Pipeline Ransomware Attack (2021):

        https://en.wikipedia.org/wiki/Colonial_Pipeline_cyberattack

    NotPetya Cyberattack (2017):

        https://en.wikipedia.org/wiki/Maersk

    LockerGoga / Norsk Hydro Ransomware (2019):

        https://en.wikipedia.org/wiki/Norsk_Hydro

    REvil / JBS Foods Ransomware (2021):

        https://en.wikipedia.org/wiki/JBS_S.A.

    Expeditors International Ransomware (2022):

        https://en.wikipedia.org/wiki/Expeditors_International

    CrowdStrike Global IT Outage (2024):

        https://en.wikipedia.org/wiki/CrowdStrike

    WannaCry / TSMC Operational Disruption (2018):

        https://en.wikipedia.org/wiki/TSMC

    Other Port/Corporate Cyber Incidents:

        https://en.wikipedia.org/wiki/DP_World (DP World Australia cyber disruption)

        https://en.wikipedia.org/wiki/Toyota (Kojima Industries supplier cyberattack)

        https://en.wikipedia.org/wiki/Honda (Snake/Ekans ransomware)

4. Trade Disputes and Tariff Changes

    US Section 232/301 Tariffs & Exclusions:

        https://en.wikipedia.org/wiki/Trump_tariffs

        https://en.wikipedia.org/wiki/United_States_tariffs

    US-China Trade War (General):

        https://en.wikipedia.org/wiki/China%E2%80%93United_States_trade_war

    Australia-China Wine Tariffs:

        https://en.wikipedia.org/wiki/Australia%E2%80%93China_relations

5. Bankruptcies, Insolvencies, and Liquidations

    Automotive & Battery Suppliers:

        https://en.wikipedia.org/wiki/Takata_Corporation (airbag recall insolvency)

        https://en.wikipedia.org/wiki/A123_Systems (lithium-ion battery Chapter 11)

        https://en.wikipedia.org/wiki/Delphi_Corporation (automotive parts Chapter 11)

        https://en.wikipedia.org/wiki/Britishvolt (UK battery startup administration)

        https://en.wikipedia.org/wiki/Fisker_Automotive (EV startup collapse)

        https://en.wikipedia.org/wiki/Collins_%26_Aikman (interiors parts Chapter 11)

    Renewable Energy Suppliers:

        https://en.wikipedia.org/wiki/Solyndra

        https://en.wikipedia.org/wiki/Q-Cells

        https://en.wikipedia.org/wiki/SunEdison

        https://en.wikipedia.org/wiki/Abengoa

        https://en.wikipedia.org/wiki/Suntech_Power

        https://en.wikipedia.org/wiki/ON_Semiconductor (acquired GT Advanced Technologies)

    Logistics & Transportation Providers:

        https://en.wikipedia.org/wiki/Hanjin_Shipping (ocean cargo carrier bankruptcy)

        https://en.wikipedia.org/wiki/Air_Berlin (airline insolvency)

    Industrial Conglomerates, Construction, and Tech:

        https://en.wikipedia.org/wiki/Carillion (UK government contractor liquidation)

        https://en.wikipedia.org/wiki/Nortel (Canadian telecom giant insolvency)

        https://en.wikipedia.org/wiki/Westinghouse_Electric_Company (nuclear reactor supplier Chapter 11)

        https://en.wikipedia.org/wiki/Peabody_Energy (coal mining Chapter 11)

        https://en.wikipedia.org/wiki/OneWeb (satellite communications Chapter 11)

        https://en.wikipedia.org/wiki/Katerra (modular construction startup collapse)

    Buyers and Retailers:

        https://en.wikipedia.org/wiki/General_Motors_Chapter_11_reorganization (Chrysler/GM supplier impacts)

        https://en.wikipedia.org/wiki/Toys_%22R%22_Us (retail liquidation supplier impacts)

        https://en.wikipedia.org/wiki/Hostess_Brands (consumer food Chapter 11)

6. Infrastructure, Logistics Choke Points, and General Operations

    Suez Canal:

        https://en.wikipedia.org/wiki/2021_Suez_Canal_obstruction

    Ports & Canals:

        https://en.wikipedia.org/wiki/Port_of_Los_Angeles

        https://en.wikipedia.org/wiki/Port_of_Vancouver

    Pipeline Systems:

        https://en.wikipedia.org/wiki/Forties_pipeline_system

    Corporate Profiles:

        https://en.wikipedia.org/wiki/BASF

        https://en.wikipedia.org/wiki/CF_Industries

        https://en.wikipedia.org/wiki/LyondellBasell

        https://en.wikipedia.org/wiki/Dow_Chemical_Company

        https://en.wikipedia.org/wiki/Shell_plc

        https://en.wikipedia.org/wiki/PetroChina

        https://en.wikipedia.org/wiki/Asahi_Kasei

        https://en.wikipedia.org/wiki/ASML_Holding

        https://en.wikipedia.org/wiki/Chapter_11,_Title_11,_United_States_Code (referenced for legal background on US bankruptcies)

These sources cover the breadth of real-world disruptions modeled across the dataset batches.
