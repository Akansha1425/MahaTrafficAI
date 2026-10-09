# Machine Learning Model Evaluation & Audit — MahaTraffic AI

**Task:** Historical Risk Tier Classification (`MEDIUM` vs `HIGH`)
**Model Architecture:** Random Forest Classifier (`n_estimators=100`, `class_weight='balanced'`)
**Dataset:** `maharashtra_district_accidents_2019_2023.csv` (2,460 total district-month records)

> [!IMPORTANT]
> **NATURE OF MODEL & STATISTICAL AUDIT DISCLOSURE**
> This model classifies historical records into risk severity tiers. It is **NOT** a future accident prediction engine.
> The target variable `risk_category` is a deterministic binning of `record_risk_score`, which is calculated directly
> from recorded accident consequences (`deaths`, `fatal_accidents`, `injuries`). High classification metrics (>98%)
> reflect that the Random Forest has successfully recovered the decision boundaries of this formula from historical features.

---

## 1. Temporal Holdout Evaluation (Out-of-Time Generalization)

To assess whether patterns hold over time without future leakage, the model was evaluated on a strict temporal split:
- **Training Partition:** 2019–2022 (1968 records across all 34 districts)
- **Testing Partition (Unseen Future Holdout):** 2023 (492 records)

| Metric | Temporal Holdout (2023) | 95% Wilson Confidence Interval |
|--------|-------------------------|--------------------------------|
| **Accuracy** | 0.9898 (98.98%) | [97.64%, 99.57%] |
| **F1-Score (Macro)** | 0.9898 | — |
| **F1-Score (Weighted)** | 0.9898 | — |
| **Precision (Weighted)** | 0.9898 | — |
| **Recall (Weighted)** | 0.9898 | — |

### Temporal Confusion Matrix (2023 Holdout, N=492)

```
Classes: ['HIGH', 'MEDIUM']
  HIGH       [258, 2]
  MEDIUM     [3, 229]
```

### Temporal Per-Class Breakdown (2023)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| HIGH | 0.9885 | 0.9923 | 0.9904 | 260 |
| MEDIUM | 0.9913 | 0.9871 | 0.9892 | 232 |

---

## 2. Full Dataset Performance (Historical Reference Benchmark)

| Metric | Full Dataset (N=2,460) | 95% Wilson Confidence Interval |
|--------|------------------------|--------------------------------|
| **Accuracy** | 0.9939 (99.39%) | [99.0%, 99.63%] |
| **F1-Score (Macro)** | 0.9939 | — |
| **F1-Score (Weighted)** | 0.9939 | — |

### Full Dataset Confusion Matrix

```
  HIGH       [1246, 4]
  MEDIUM     [11, 1199]
```

---

## 3. Feature Importance Analysis

The top 10 features driving risk classification tree splits are:

| Rank | Feature | Importance | Feature Interpretation |
|------|---------|------------|------------------------|
| 1 | `is_night` | 0.2188 | Engineered traffic feature |
| 2 | `deaths` | 0.1685 | Fatality count per district-month |
| 3 | `road_type` | 0.1260 | Engineered traffic feature |
| 4 | `fatal_accidents` | 0.0967 | Direct measure of catastrophic incident volume |
| 5 | `severity_index` | 0.0746 | Compound severity calculated from fatality & injury weight |
| 6 | `is_highway` | 0.0712 | Engineered traffic feature |
| 7 | `fatality_ratio` | 0.0559 | Proportion of accidents resulting in death |
| 8 | `accident_count` | 0.0545 | Total crash volume |
| 9 | `injuries` | 0.0532 | Total non-fatal injuries |
| 10 | `injury_ratio` | 0.0270 | Proportion of crashes resulting in injuries |

---

## 4. Scientifically Defensible Limitations & Methodological Caveats

1. **Target Derivation & Correlation Structure:** `record_risk_score` is a weighted composite of severity and frequency. Consequently, `fatal_accidents` and `deaths` exhibit Pearson correlations of 0.70 and 0.73 with the target risk score. High accuracy is therefore expected and demonstrates model fidelity to the rule framework, not clairvoyance.
2. **Spatial Aggregation:** Data is aggregated at the district level (e.g. Pune has 120 records covering 60 months and 2 road categories). Micro-level road segment or intersection hazards cannot be resolved.
3. **Absence of Real-Time Telemetry:** No vehicle speed, GPS probe, or weather station data is included. Analysis is strictly historical retrospective.
4. **Class Representation:** High and Medium risk categories are evenly represented; Low risk is absent at the district aggregate level because all Maharashtra districts experience substantial baseline accident volumes.
