# RAG Intelligence Layer Evaluation & Retrieval Audit — MahaTraffic AI

## 1. Executive Summary

The MahaTraffic AI Retrieval-Augmented Generation (RAG) system provides domain-specific knowledge retrieval over authoritative Indian road safety documents, including:
- **MoRTH Road Safety Guidelines (2022)**
- **IRC:SP:88-2019 Road Safety Audit Manual**
- **Maharashtra Motor Vehicles Rules & Safety Summary**

This document reports the empirical evaluation of the RAG retrieval pipeline conducted via [`rag/evaluation/test_retrieval.py`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/rag/evaluation/test_retrieval.py).

Evaluation Date: October 2026  
Retriever Index: TF-IDF vectorizer + cosine similarity over semantic chunks (`rag/index/rag_index.pkl`)  
Corpus Size: 3 primary documents, 9 processed chunks, 1,499 unique vocabulary terms  

---

## 2. Empirical Retrieval Performance

The retrieval system was evaluated against 5 standard road safety benchmark queries covering causes of accidents, recommended safety measures, Maharashtra-specific concerns, accident severity factors, and engineering interventions.

| Query ID | Benchmark Query | Retrieved Chunks ($k=5$) | Max Similarity Score | Relevance Tier | Retrieval Latency | Primary Retrieved Document Source |
|---|---|---|---|---|---|---|
| **Q1** | "What are common causes of road accidents?" | 5 | 0.0903 | MEDIUM | 1.2 ms | `morth_road_safety_guidelines_2022` |
| **Q2** | "What road-safety measures are recommended?" | 5 | 0.0950 | MEDIUM | 0.7 ms | `irc_sp_88_road_safety_audit_manual` |
| **Q3** | "What are major road-safety concerns in Maharashtra?" | 5 | 0.1389 | MEDIUM | 0.6 ms | `irc_sp_88_road_safety_audit_manual` |
| **Q4** | "What factors are associated with accident severity?" | 5 | 0.1014 | MEDIUM | 0.6 ms | `morth_road_safety_guidelines_2022` |
| **Q5** | "What measures are recommended for improving road safety?" | 5 | 0.0950 | MEDIUM | 0.6 ms | `irc_sp_88_road_safety_audit_manual` |

### Summary Statistics:
- **Total Queries Evaluated:** 5
- **Average Retrieval Latency:** **0.74 ms** (sub-millisecond local execution)
- **Document Coverage:** 100% of queries successfully matched authoritative chunks from at least 2 distinct source manuals.
- **Top Matching Sources:** `irc_sp_88_road_safety_audit_manual` (60% top rank), `morth_road_safety_guidelines_2022` (40% top rank).

---

## 3. Groundedness Evaluation Rubric & Manual Verification

To evaluate factual groundedness without artificial inflation, an empirical 10-query question-answer rubric was evaluated comparing model claims against the source document chunks:

| QA Test ID | Question | Model Assertion / Response | Supporting Source Chunk | Groundedness Determination | Notes |
|---|---|---|---|---|---|
| **G-01** | Definition of Black Spot | Road segment $\le 500\text{ m}$ with $\ge 5$ fatal accidents or $\ge 10$ fatalities in 3 years | `morth_road_safety_guidelines_2022` chunk 2 | **GROUNDED** | Exact alignment with MoRTH standard criteria |
| **G-02** | Expressway Speed Limit | 120 km/h for LMVs, 80 km/h for HCVs | `morth_road_safety_guidelines_2022` chunk 1 | **GROUNDED** | Verbatim rule match |
| **G-03** | RSA Stage Requirements | 5 audit stages (Planning, Detailed Design, Construction, Pre-Opening, Operation) | `irc_sp_88_road_safety_audit_manual` chunk 1 | **GROUNDED** | Matches IRC:SP:88 specifications |
| **G-04** | Drunk Driving Penalty | ₹10,000 fine and/or up to 6 months imprisonment under MVA Sec 185 | `maharashtra_motor_vehicles_rules` chunk 2 | **GROUNDED** | Matches statutory provisions |
| **G-05** | Helmet Violation Fine | ₹1,000 fine and 3-month license disqualification under Sec 194D | `maharashtra_motor_vehicles_rules` chunk 2 | **GROUNDED** | Matches statutory provisions |
| **G-06** | Monsoon Hazard Factors | Hydroplaning risk, reduced skid resistance, reduced braking friction | `irc_sp_88_road_safety_audit_manual` chunk 3 | **GROUNDED** | Documented engineering factor |
| **G-07** | High Fatality Corridors | Mumbai-Pune Expressway, NH-60 (Nashik-Pune), Nagpur-Amravati corridor | `morth_road_safety_guidelines_2022` chunk 3 | **GROUNDED** | Referenced in Maharashtra appendix |
| **G-08** | Pedestrian Facility Minimums | Footpath width $\ge 1.8\text{ m}$ in commercial zones, pedestrian refuge islands | `irc_sp_88_road_safety_audit_manual` chunk 2 | **GROUNDED** | Standard IRC guidelines |
| **G-09** | Pune Accident Volume | 24,300 total accidents from 120 historical district records | Parquet dataset tool payload | **GROUNDED** | Clean match with empirical accident database |
| **G-10** | Social Media Causal Link | Citizen complaints reflect public sentiment, NOT direct physical accident causes | System guardrail non-causal policy | **GROUNDED** | Explicitly framed as non-causal signal |

**Empirical Groundedness Rate on Verified Rubric:** $10/10$ ($100.0\%$, 95% Wilson CI: $[72.25\%, 100.0\%]$).

---

## 4. Technical Architecture & Indexing Details

- **Chunking Strategy:** Semantic section splitting using markdown headers with recursive chunking ($\sim 500$ characters target size with 50-character overlap).
- **Vector Representation:** Scikit-learn TF-IDF Vectorizer with English stop-word filtering and sublinear TF scaling.
- **Similarity Metric:** Dot product cosine similarity across normalized term frequency vectors.
- **Fallback Mechanism:** If a query contains out-of-vocabulary terms or yields similarity scores $<0.01$, the retriever falls back to top category guideline summaries, explicitly notifying the user of low similarity confidence.

---

## 5. Limitations & Future Directions

1. **Vocabulary Mismatch in TF-IDF:** Because the retriever uses lexical TF-IDF rather than dense neural embeddings (e.g. BERT/Sentence-Transformers), synonyms not present in the document text (e.g. "car smash" instead of "vehicle collision") produce lower similarity scores.
2. **Corpus Scale:** The current knowledge base comprises 3 curated policy and guideline documents (9 chunks). While these cover Maharashtra and Indian statutory baselines, comprehensive coverage of local municipal bylaws (e.g. Pune Municipal Corporation traffic resolutions) would require corpus expansion.
3. **Absence of Real-Time Advisory Updates:** Changes to traffic fines or road diversion notices enacted after 2023 are not captured in the offline document store.
