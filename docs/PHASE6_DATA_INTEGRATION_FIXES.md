# MahaTraffic AI — Phase 6: Data Integrity, UI Accuracy & Reliability Fixes Report

This document records the root-cause analysis, implementation details, verification evidence, and methodological disclosures for the five Phase 6 data integrity and UI accuracy fixes.

---

## 1. Summary of Issues & Resolution Status

| # | Issue Description | Status | Root Cause | Files Modified |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **Backend Port Indicator & Base URL** | **Fixed** | Frontend hardcoded port 8000 and assumed "Online" status before performing a health check. | `frontend/index.html` |
| **2** | **Social Feed `undefined` Badge** | **Fixed** | Frontend read `p.sentiment` instead of `p.sentiment_label` returned by `/api/v1/social/posts/by-location`. | `frontend/index.html`, `tests/backend/test_api_endpoints.py` |
| **3** | **Scientifically Honest Reliability Metrics** | **Fixed** | Static "Certified Evaluation" language and aggregation of disparate evaluation types without explicit sample sizes, confidence intervals, or target leakage notes. | `frontend/index.html` |
| **4** | **District Risk Scores & Ranking Clarity** | **Fixed** | Lack of explanation regarding multi-factor composite weighting (`0.40×Severity + 0.25×Frequency + 0.20×Road + 0.15×Time`), causing confusion over why high-volume districts like Mumbai rank above high-fatality districts like Pune. | `frontend/index.html`, `tests/backend/test_api_endpoints.py` |
| **5** | **ML Prediction Probability Labeling** | **Fixed** | Displaying `Confidence: 99.4%` without clarifying that this is an internal Random Forest class probability (`predict_proba`) rather than an empirical guarantee of future accuracy. | `frontend/index.html`, `tests/backend/test_api_endpoints.py` |

---

## 2. Detailed Root Cause Analysis & Fix Implementations

### Issue 1: Backend Port Indicator & API Base Configuration
- **Root Cause:** The frontend script previously defined a static URL (`http://127.0.0.1:8000`) and rendered static "Online" text in HTML before verifying backend availability.
- **Fix Implementation:**
  - Configured a single source of truth for the API base: `const API_BASE = (window.MAHA_API_BASE || 'http://127.0.0.1:8081').replace(/\/+$/, '');`.
  - Parsed host and port dynamically with the `URL` constructor (`updateBackendConfigDisplay()`) and rendered the configured endpoint in the sidebar footer (`Host: 127.0.0.1:8081`).
  - Replaced hardcoded "Online" HTML status with a dynamic "Checking..." state on initial load.
  - Bound the green "Online" badge exclusively to a verified HTTP 200 response from `${API_BASE}/health` returning `{"status": "ok"}`.
  - Added a manual "Retry" trigger.

---

### Issue 2: Social Feed `undefined` Badge
- **Root Cause:** The backend endpoint `GET /api/v1/social/posts/by-location` returns a list of dictionaries with the field name `sentiment_label` (`"NEGATIVE"`, `"NEUTRAL"`, `"POSITIVE"`). The frontend JavaScript template accessed `p.sentiment`, evaluating to `undefined` and rendering `<span class="risk-tag">undefined</span>`.
- **Fix Implementation:**
  - Updated the template to read `p.sentiment_label || p.sentiment || null`.
  - Added null/empty safety checks so that badges are only rendered when a valid label exists.
  - Escaped all post fields using `escapeHtml()` to protect against XSS.
  - Added automated integration tests in `tests/backend/test_api_endpoints.py` verifying that `/api/v1/social/posts/by-location` returns non-empty strings for `sentiment_label`.

---

### Issue 3: Scientifically Honest Reliability & Audit Metrics
- **Root Cause:** The header previously claimed "Certified Evaluation" and displayed an undifferentiated mix of unit test counts and model accuracy percentages without confidence intervals or sample size disclaimers.
- **Fix Implementation:**
  - Replaced "Certified Evaluation" with "Measured Evaluation Results & Audit Summary".
  - Separated the evaluation center into distinct, scientifically categorized cards:
    1. **Software Unit & Integration Tests:** 74/74 (100%) deterministic tests passed.
    2. **ML Classification Evaluation:** 97.15% Test Accuracy (Wilson 95% CI: `[95.28%, 98.31%]`, `n = 492` test samples, `1,968` train samples).
    3. **Adversarial Benchmark:** 85.0% Block Rate (34/40 blocked, Wilson 95% CI: `[70.91%, 92.94%]`, `n = 40`).
    4. **RAG Groundedness:** 100.0% Groundedness on finite benchmark set (Wilson 95% CI: `[61.00%, 100.00%]`, `n = 6`).
    5. **Location Integrity:** 100% record isolation verified (`Pune: 120/120 records` vs `2,460 total dataset records`).
  - **Critical Methodological Limitation Disclosed:** Explicitly noted that `risk_category` is rule-derived from `record_risk_score`, where `deaths`, `fatal_accidents`, and `severity_index` are inputs. The model learns historical rule-derived classifications rather than predicting independent future accident occurrences.

---

### Issue 4: District Risk Scores & Ranking Clarity
- **Root Cause:** Mumbai (45.07 score, 28,372 accidents, 3,560 deaths) ranks above Pune (43.31 score, 24,300 accidents, 6,991 deaths) in composite risk score. Users viewing raw fatality counts were confused without understanding the 4-factor formula.
- **Formula Breakdown:**
  $$\text{Composite Score} = 0.40 \times \text{Severity} + 0.25 \times \text{Frequency} + 0.20 \times \text{RoadFactor} + 0.15 \times \text{TimeFactor}$$
  - **Frequency Component (25%):** Normalized against the maximum district total in Maharashtra ($28,372$ accidents in Mumbai).
    - Mumbai frequency score: $\frac{28,372}{28,372} \times 100 = 100.0 \implies 25.0\text{ points}$.
    - Pune frequency score: $\frac{24,300}{28,372} \times 100 = 85.65 \implies 21.41\text{ points}$ (Mumbai $+3.59\text{ pts}$).
  - **Severity Component (40%):** $\frac{\text{Deaths} \times 1.0 + \text{Injuries} \times 0.3}{\text{Accidents} \times 2.5} \times 100$.
    - Pune severity score: $19.51 \implies 7.80\text{ points}$.
    - Mumbai severity score: $14.36 \implies 5.74\text{ points}$ (Pune $+2.06\text{ pts}$).
  - **Net Result:** Mumbai ($45.07$) ranks ahead of Pune ($43.31$) because its volume lead outweighs the per-accident severity difference.
- **Fix Implementation:**
  - Labeled table column: `Composite Score (0–100 Scale)`.
  - Added an explanatory callout box detailing the 4 components and clarifying that composite score is a balanced multi-factor aggregate across 2019–2023, not a pure death tally.

---

### Issue 5: ML Prediction Probability Labeling
- **Root Cause:** Displaying `Confidence: 99.4%` could lead users to believe the model provides a 99.4% guarantee of future outcome accuracy.
- **Fix Implementation:**
  - Updated the output label to: `Model-estimated class probability: XX.X%`.
  - Added the mandatory explanatory note:
    > *"This represents the Random Forest tree ensemble's internal class probability (`predict_proba`) for the predicted risk tier. It is not a guarantee of correctness and is not calibrated to real-world probabilities. The model classifies historical risk tiers and is not a validated predictor of future accidents."*
  - Populated probability breakdown bars for all classes (`HIGH`, `MEDIUM`) dynamically from the backend response.

---

## 3. Verification & Test Evidence

### Pytest Execution
Command:
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.11.7, pytest-8.4.2, pluggy-1.6.0
collected 77 items

tests/agents/test_adversarial.py::test_adversarial_benchmark_suite PASSED [  1%]
tests/agents/test_agents.py::test_agent_workflow_accident_query PASSED   [  2%]
...
tests/backend/test_api_endpoints.py::test_dashboard_overview PASSED      [ 12%]
tests/backend/test_api_endpoints.py::test_social_posts_by_location PASSED [ 18%]
tests/backend/test_api_endpoints.py::test_risk_score_calculation PASSED  [ 19%]
tests/backend/test_api_endpoints.py::test_risk_ml_predict PASSED         [ 20%]
...
tests/test_config.py::test_settings_defaults PASSED                      [100%]
======================== 77 passed, 1 warning in 6.78s ========================
```

### System Verification (`run_all.py`)
Command:
```powershell
.venv\Scripts\python.exe run_all.py
```
Output:
```
[1] DATA CHECKS            [OK]
[2] ML MODEL ARTIFACTS     [OK]
[3] RAG INDEX              [OK]
[4] SOCIAL ANALYTICS       [OK]
[5] BACKEND IMPORT CHECK   [OK]
[6] AGENT PIPELINE CHECK   [OK]
[7] RELIABILITY REPORT     [OK]
ALL CHECKS PASSED — System ready.
```

---

## 4. Remaining Limitations & Boundaries

1. **Synthetic / Curated Social & RAG Benchmarks:** The RAG groundedness benchmark ($n=6$) and adversarial test suite ($n=40$) are finite evaluation datasets. Performance on these benchmarks establishes compliance with specified test criteria, but does not guarantee generalized behavior on arbitrary open-domain prompts.
2. **Feature Leakage in Historical Tier Modeling:** Because historical accident metrics (`deaths`, `fatal_accidents`) are mathematically related to the formula that derived `risk_category`, the Random Forest classifier functions as an automated surrogate classifier for the formula, not an independent causal forecaster of future collisions.
