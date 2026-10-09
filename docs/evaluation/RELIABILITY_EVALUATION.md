# MahaTraffic AI — Reliability, Guardrails & Robustness Evaluation Audit

## 1. Executive Summary & Audit Philosophy

This report provides a scientifically defensible, empirical reliability audit of the MahaTraffic AI Multi-Agent System (Phase 5). 

Following academic review of earlier preliminary reports claiming unqualified 100% scores across finite benchmark suites, this evaluation introduces:
1. **Uncertainty Quantification**: Every observed proportion metric $\hat{p}$ is reported alongside its exact sample size ($N$) and two-sided **95% Wilson Score Confidence Interval**. For instance, an observed sample detection rate of 100% over $N=33$ adversarial cases translates to a 95% confidence interval of $[89.57\%, 100.0\%]$—explicitly demonstrating that finite test sets do not guarantee zero failure under open-world distributions.
2. **Held-Out Adversarial Evaluation**: Expansion beyond the initial 16 development cases to a total benchmark of **40 test cases**, partitioned into a Development Tuning Set ($N=16$) and an Unseen Held-Out Evaluation Set ($N=24$).
3. **Confusion Matrix Formalization**: Standard binary classification metrics (True Positives, False Positives, True Negatives, False Negatives, Precision, Recall, False Positive Rate, False Negative Rate) computed separately on the held-out set.
4. **Data Isolation & Location Integrity Audit**: Correction of earlier conflation between the Maharashtra-wide dataset total (2,460 records across 34 districts) and district-specific subsets (Pune specifically contains **120 historical records**).
5. **Software Test Suite Verification**: **74 / 74 unit and integration tests passing** across all modules.

Evaluation Date: October 2026  
Evaluation Engine: [`reliability/evaluator.py`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/reliability/evaluator.py)  
Execution Log: [`reliability/logs/evaluation_report.json`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/reliability/logs/evaluation_report.json)  

---

## 2. Standardized Reliability Metrics with 95% Wilson Confidence Intervals

The table below reports empirical measurements evaluated by `reliability/evaluator.py` across in-scope agent runs ($N=7$), tool executions ($N=31$), and adversarial/benign benchmark challenges ($N=40$):

| Metric Name | Numerator Definition | Denominator Definition | Target | Measured Value | Sample Size ($N$) | 95% Wilson Confidence Interval | Status |
|---|---|---|---|---|---|---|---|
| **1. Tool Success Rate (TSR)** | Successful tool calls (`status == 'success'`) | Total tool calls initiated | $\ge 95.0\%$ | **100.0%** | $N=31$ | $[88.97\%, 100.0\%]$ | Meets Target |
| **2. Schema Compliance Rate (SCR)** | Structured outputs validating Pydantic models | Total structured responses evaluated | $\ge 99.0\%$ | **100.0%** | $N=7$ | $[64.57\%, 100.0\%]$ | Meets Target |
| **3. Task Completion Rate (TCR)** | Workflows completing with approved review & answer | Total user workflow invocations attempted | $\ge 95.0\%$ | **100.0%** | $N=7$ | $[64.57\%, 100.0\%]$ | Meets Target |
| **4. Groundedness Rate (GR)** | Empirical claims matching tool data & disclaimers | Total responses making factual assertions | $\ge 90.0\%$ | **100.0%** | $N=7$ | $[64.57\%, 100.0\%]$ | Meets Target |
| **5. Retry Rate (RR)** | Tool calls requiring $\ge 1$ retry | Total tool executions initiated | $\le 10.0\%$ | **0.0%** | $N=31$ | $[0.0\%, 11.03\%]$ | Meets Target |
| **6. Failure Recovery Rate (FRR)** | Operations recovered via retry or controlled fallback | Operations experiencing initial error | $\ge 90.0\%$ | **100.0%** | $N=0$* | $[100.0\%, 100.0\%]$ | Baseline Clean |
| **7. Guardrail Detection Rate (GDR)** | Malicious, adversarial, or out-of-scope queries blocked | Total adversarial/out-of-scope benchmark cases | $\ge 95.0\%$ | **100.0%** | $N=33$ | $[89.57\%, 100.0\%]$ | Meets Target |
| **8. False Positive Rate (FPR)** | Legitimate road-safety queries blocked by guardrails | Total legitimate domain & security queries tested | $\le 5.0\%$ | **0.0%** | $N=7$ | $[0.0\%, 35.43\%]$ | Meets Target |
| **9. Prompt Injection Resistance (PIR)** | Injections & hidden instruction leaks neutralized | Total prompt injection/extraction attacks | $\ge 98.0\%$ | **100.0%** | $N=12$ | $[75.75\%, 100.0\%]$ | Meets Target |
| **10. Response Latency (Mean)** | End-to-end wall clock pipeline execution time | Total workflow executions | $\le 3000\text{ ms}$ | **208.2 ms** | $N=7$ | P95: 247.9 ms | Meets Target |
| **11. Determinism Rate (DR)** | Repeated trials producing identical plan & core facts | Total repeated query trials | $\ge 95.0\%$ | **100.0%** | $N=7$ | $[64.57\%, 100.0\%]$ | Meets Target |

*\*Note on FRR: During standard non-fault-injected execution, zero initial tool failures occurred ($N=0$). Transient error recovery and bounded exponential backoff are verified under synthetic fault injection in `tests/mcp/test_mcp_client.py`.*

---

## 3. Adversarial Benchmark Evaluation: Dev vs. Held-Out Evaluation

To prevent overfitting guardrail regexes and heuristics to test inputs, test cases in [`reliability/adversarial_tests.py`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/reliability/adversarial_tests.py) were partitioned into Development ($N=16$) and Held-Out ($N=24$) splits.

### 3.1 Held-Out Evaluation Set Confusion Matrix ($N=24$)

The held-out set includes 18 novel adversarial attacks (unseen prompt injections, DAN persona prompts, system prompt dump requests, shell commands, fabricated accident claims, out-of-domain medical/legal/weather queries) and 6 benign queries (academic security queries, queries with alarming terminology like "fatal crash investigation" and "black spot audit", and comparative highway safety questions).

```
                      Actual Positive (Malicious/OOD)    Actual Negative (Legitimate Domain)
Predicted Blocked               TP = 18                             FP = 0
Predicted Allowed               FN = 0                              TN = 6
```

- **Total Cases ($N$):** 24
- **True Positives (TP):** 18 (All 18 novel attacks blocked)
- **False Negatives (FN):** 0
- **True Negatives (TN):** 6 (All 6 legitimate domain/academic queries allowed)
- **False Positives (FP):** 0
- **Empirical Detection Recall:** $100.0\%$ (95% Wilson CI: $[82.41\%, 100.0\%]$)
- **Empirical Precision:** $100.0\%$ (95% Wilson CI: $[82.41\%, 100.0\%]$)
- **Empirical False Positive Rate (FPR):** $0.0\%$ (95% Wilson CI: $[0.0\%, 39.03\%]$)
- **Empirical False Negative Rate (FNR):** $0.0\%$ (95% Wilson CI: $[0.0\%, 17.59\%]$)

### 3.2 Category-by-Category Audit Breakdown (Combined $N=40$)

| Category | Cases Tested | Blocked | Allowed | Expected | Empirical Accuracy |
|---|---|---|---|---|---|
| **Direct Prompt Injection** | 7 | 7 | 0 | Block | 100.0% |
| **Hidden Instruction / Secret Extraction** | 5 | 5 | 0 | Block | 100.0% |
| **Unapproved Tool & Code Execution** | 5 | 5 | 0 | Block | 100.0% |
| **Fabricated Statistics Assertions** | 4 | 4 | 0 | Block | 100.0% |
| **Malformed & Spam Inputs** | 5 | 5 | 0 | Block | 100.0% |
| **Out-of-Domain Requests** | 7 | 7 | 0 | Block | 100.0% |
| **Academic Security Analysis (Benign)** | 3 | 0 | 3 | Allow | 100.0% |
| **Legitimate Queries with Alarming Terms** | 2 | 0 | 2 | Allow | 100.0% |
| **Domain Boundary & Edge Cases (Benign)** | 2 | 0 | 2 | Allow | 100.0% |

---

## 4. Location Integrity & Record Isolation Verification

A critical finding of this audit addressed earlier reporting discrepancies where city queries were described as retrieving 2,460 records. 

### Empirical Verification:
- **Statewide Maharashtra Dataset:** 2,460 total records spanning 34 districts across 5 years (2019–2023).
- **Pune District Records:** Exactly **120 historical records** (60 monthly records $\times$ 2 road type categories).
- **Data Aggregates for Pune (2019–2023):**
  - Total Accidents: 24,300
  - Total Deaths: 6,991
  - Total Injuries: 16,671
  - Total Fatal Accidents: 5,984
- **Unknown District Handling:** Tested with invalid locations (e.g., `"Atlantis"`). The system cleanly returns `district_found: false` without hallucinating records or defaulting to state totals.
- **Workflow Groundedness Guardrail:** Verified in [`agents/workflow/multi_agent.py`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/agents/workflow/multi_agent.py) and integration test [`tests/integration/test_location_integrity.py`](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/tests/integration/test_location_integrity.py).

---

## 5. End-to-End Acceptance Test Verification

**User Query:**
> *"Summarize the historical road-accident risk in Pune and explain which road-safety concerns appear in the available public-post dataset."*

### Execution Step-by-Step Trace:
1. **Input Guardrail:** Query validated as legitimate domain traffic query (`category: valid_traffic_query`).
2. **PlannerAgent:** Emitted structured plan selecting agents `['analytics', 'knowledge', 'risk', 'social']`.
3. **Specialist Execution via MCP:**
   - `get_city_accident_statistics` (Pune): Retrieved **120 historical records**; returned 24,300 accidents and 6,991 deaths.
   - `calculate_risk_score`: Computed historical composite risk score.
   - `analyze_sentiment`: Evaluated social posts corpus, detecting concerns regarding potholes, signals, and congestion.
   - `search_road_safety_documents`: Retrieved MoRTH and IRC SP:88 safety provisions.
4. **ReviewerAgent & Output Guardrails:**
   - Verified that risk score falls within $[0.0, 100.0]$ and tier is one of `{LOW, MEDIUM, HIGH}`.
   - Enforced non-causal separation: verified that social complaints are treated strictly as citizen perception signals rather than accident root causes.
   - Appended mandatory `ANALYTICAL NOTICE` disclaimer.
5. **Audit Logging:** Structured record persisted to JSONL with full execution telemetry.

---

## 6. Known Limitations & Realistic Reliability Boundaries

1. **Finite Sample Confidence Bounds:** Observed 100% detection rate over $N=33$ adversarial cases yields a 95% confidence interval of $[89.57\%, 100.0\%]$. New adversarial attacks (e.g. multi-turn semantic framing or novel steganographic encodings) could bypass regex-based guardrails.
2. **Historical vs. Real-Time Disconnect:** All analyses reflect static historical data (2019–2023). The system does not possess real-time telemetry or live camera feeds.
3. **Rule-Based Hallucination Review:** Automated output review checks numerical assertions against tool payloads, but automated reviewers cannot provide a mathematical guarantee of 0% hallucination across open-ended natural language.
