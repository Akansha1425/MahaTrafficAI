# Dataset Limitations & Methodological Constraints — MahaTraffic AI

## 1. Executive Summary

This document provides a comprehensive, rigorous disclosure of the data scope, spatial-temporal resolution, assumptions, biases, and methodological limitations inherent in the datasets used throughout the MahaTraffic AI project.

A key objective of this audit is to provide full academic transparency regarding what the system can and cannot infer, ensuring that stakeholders do not treat retrospective descriptive statistics as predictive clairvoyance or real-time traffic sensing.

---

## 2. Inventory of Datasets

The project relies on two primary empirical datasets processed and stored in Parquet format:

| Dataset Identifier | Storage Format | Raw Dimensions | Processed Dimensions | Temporal Coverage | Spatial Granularity | Description |
|---|---|---|---|---|---|---|
| **Accident Statistics Dataset** | `maharashtra_accidents_clean.parquet` | 2,460 rows | 2,460 rows | 2019–2023 (60 months) | 34 Maharashtra Districts | Monthly aggregated accident records categorized by district, road type, and primary cause. |
| **Public Social Perception Dataset** | `social_posts_clean.parquet` | 1,500 posts | 1,500 posts | 2022–2023 | 5 Cities (Mumbai, Pune, Nagpur, Nashik, Thane) | Curated citizen complaints, sentiment scores, and traffic hazard discussions from social platforms. |
| **Risk Feature Matrix** | `risk_features.parquet` | 2,460 rows | 2,460 rows | 2019–2023 | 34 Districts | Feature engineered dataset with cyclically encoded temporal terms, severity ratios, and risk scores. |

---

## 3. Detailed Dataset Disclosures & Limitations

### 3.1 Spatial Resolution (District Aggregation vs. Micro-Location Black Spots)
- **Granularity:** The accident dataset is aggregated at the **district level** (e.g., Pune, Nagpur, Nashik).
- **Consequence:** There are no geographic coordinates (latitude/longitude), road link IDs, or intersection identifiers in the primary dataset.
- **Specific Finding:** Pune district contains **120 historical records** across the 5-year period (representing 60 months $\times$ 2 road type categories: National Highway vs State/Other roads).
- **Limitation:** The system can identify district-level high-risk severity patterns, but **cannot pinpoint exact street intersections or micro-segment black spots** from tabular accident counts alone. Black spot definitions provided in responses are derived from MoRTH statutory knowledge documents, not GPS clustering on local coordinates.

### 3.2 Temporal Resolution & Historical Scope
- **Time Window:** 2019 to 2023 (monthly aggregates).
- **Absence of Real-Time Feeds:** The platform does not ingest live GPS probes, CCTV video feeds, or real-time traffic police dispatch data.
- **Limitation:** All risk scores, rankings, and statistical analyses are strictly **retrospective historical summaries**. They do not represent live traffic conditions or immediate emergency alerts.

### 3.3 Target Derivation & ML Feature Leakage Characteristics
- **Target Variable Definition:** In `risk_features.parquet`, `risk_category` is a discrete tier (`MEDIUM` vs `HIGH`) obtained by binning `record_risk_score`.
- **Derivation Formula:**
  $$\text{record\_risk\_score} = 0.40 \times \text{Severity} + 0.30 \times \text{Frequency} + 0.15 \times \text{RoadFactor} + 0.15 \times \text{TimeFactor}$$
- **Direct Correlation:** Because `Severity` is computed directly from recorded `deaths`, `fatal_accidents`, and `injuries`, these features exhibit high Pearson correlations with the target:
  - `deaths` $\leftrightarrow$ `record_risk_score`: $r = 0.733$
  - `fatal_accidents` $\leftrightarrow$ `record_risk_score`: $r = 0.702$
- **Methodological Disclosure:** The Random Forest Classifier achieves $>98\%$ accuracy because it is learning the decision boundaries of this historical severity rule. The model is **classifying historical severity tiers**, NOT forecasting future accidents months in advance.

### 3.4 Non-Causal Nature of Social Media Data
- **Sample Bias:** The social dataset represents active digital platform users in major metropolitan centers (Mumbai, Pune, Nagpur). Rural districts and offline demographics are substantially underrepresented.
- **Correlation vs. Causation:** Citizen social media complaints (e.g. expressing frustration over potholes, signal timings, or congestion) represent **public perception and civic dissatisfaction**, NOT verified physical causes of fatal vehicular crashes.
- **Guardrail Enforcement:** The multi-agent reviewer strictly enforces that social trends cannot be cited as causal explanations for accident volumes.

### 3.5 Class Distribution & Missing Tiers
- **Tier Representation:** In the 2,460 district records, 1,250 records are categorized as `HIGH` risk and 1,210 as `MEDIUM` risk.
- **Absence of 'LOW' Category:** Zero records fall into the `LOW` category at the district aggregate level because every Maharashtra district experienced baseline fatal accidents throughout 2019–2023. Low risk only manifests at micro localized road segment levels not captured in the district data.

---

## 4. Methodological Safeguards Implemented

To ensure users and downstream systems do not misuse the data:
1. **Mandatory Analytical Disclaimer:** Every system response automatically includes an analytical notice explicitly disclosing that outputs are historical calculations from 2019–2023.
2. **Missing Location Disclosures:** Unknown or non-Maharashtra locations (e.g. "Atlantis", "Bangalore") return explicit negative disclosures (`district_found: false`) rather than falling back or hallucinating synthetic statistics.
3. **Exact Sample Size & Wilson Confidence Reporting:** All empirical rates are presented with explicit sample sizes ($N$) and Wilson score 95% confidence intervals to avoid misleading claims of 100% real-world perfection.
