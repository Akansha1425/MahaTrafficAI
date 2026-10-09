# MahaTraffic AI — Phase 6: Frontend API Integration Specification

This document maps all FastAPI backend endpoints (`/api/v1/*`) to the interactive frontend dashboard views, detailing schemas, data types, parameter contracts, and error handling.

---

## 1. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Frontend Dashboard (Single Page App)                  │
│   [Executive Overview] [Historical Analytics] [Risk & ML Playground]   │
│   [Social Sentiment]   [AI Agent Console]     [Reliability Center]     │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │ HTTP (REST / JSON)
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend (:8000)                            │
│  ├── /health                       -> Health Check                     │
│  ├── /api/v1/dashboard/overview    -> Aggregated Multi-Service Data     │
│  ├── /api/v1/analytics/*           -> Historical Big Data Analytics     │
│  ├── /api/v1/risk/*                -> Risk Scoring & ML Random Forest  │
│  ├── /api/v1/social/*              -> NLP Sentiment & Public Concern    │
│  └── /api/v1/agent/query           -> Multi-Agent Orchestration + RAG   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. API Endpoint & Frontend Integration Map

| HTTP Method | Endpoint Path | Query / Body Parameters | Response Schema Summary | Target Frontend Component / View |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | None | `{"status": "ok", "project": "...", "version": "...", "environment": "..."}` | Sidebar Status Dot & Connection Banner |
| `GET` | `/api/v1/dashboard/overview` | None | `{summary, yearly_trends, top_risk_districts, ml_model, social_analytics}` | Overview Dashboard KPIs, Charts, Mini-tables |
| `GET` | `/api/v1/analytics/yearly` | None | `[{"year": int, "total_accidents": int, "total_deaths": int, "total_injuries": int, "fatal_accidents": int, "fatality_rate": float}]` | Historical Analytics: Multi-year trend chart |
| `GET` | `/api/v1/analytics/districts` | `district` *(optional string)* | `[{"district": str, "total_accidents": int, "total_deaths": int, "total_injuries": int, "avg_fatality_ratio": float, "risk_score": float, "risk_category": str}]` | Historical Analytics: 34 Districts Ranking Table |
| `GET` | `/api/v1/analytics/model-metrics` | None | `{"accuracy": float, "f1_macro": float, "f1_weighted": float, "cv_accuracy_mean": float, "cv_accuracy_std": float, "train_samples": int, "test_samples": int}` | ML Model View & Reliability Cards |
| `GET` | `/api/v1/analytics/feature-importances` | None | `{"feature_name": importance_weight}` | ML Model Feature Importance Horizontal Bar Chart |
| `POST` | `/api/v1/risk/score` | JSON: `accident_count`, `fatal_accidents`, `deaths`, `injuries`, `road_type`, `time_period` | `{"risk_score": float, "risk_category": "HIGH"|"MEDIUM"|"LOW", "components": {...}, "formula": str}` | Interactive Formula Risk Calculator Gauge |
| `POST` | `/api/v1/risk/predict` | JSON: `district`, `road_type`, `primary_cause`, `year`, `month`, `accident_count`, `deaths`, `injuries`, `is_monsoon`, `is_night`, `is_highway`, etc. | `{"predicted_category": str, "confidence": float, "class_probabilities": {...}, "model": str}` | ML Risk Inference Simulator & Probability Bars |
| `GET` | `/api/v1/risk/districts` | `district` *(optional string)* | `[{"district": str, "risk_score": float, "risk_category": str, ...}]` | Risk Explorer & District Comparison View |
| `GET` | `/api/v1/social/sentiment` | None | `{"POSITIVE": int, "NEGATIVE": int, "NEUTRAL": int}` | Social Sentiment Donut Chart & Ratio KPIs |
| `GET` | `/api/v1/social/concern-ranking` | `top_n` *(default: 10)* | `[{"location": str, "avg_concern": float, "post_count": int, "peak_concern": float}]` | Public Concern Ranking Chart & Location Cards |
| `GET` | `/api/v1/social/topics` | None | `{"road_condition": int, "infrastructure": int, "traffic_management": int, "accident_report": int, "speeding": int}` | Public Concern Topic Cloud & Frequency Breakdown |
| `GET` | `/api/v1/social/trend` | None | `[{"year": int, "avg_sentiment": float}]` | Multi-Year Sentiment Trend Line Chart |
| `GET` | `/api/v1/social/posts/concerning` | `top_n` *(default: 10)* | `[{"post_id": str, "text": str, "location": str, "concern_score": float, "sentiment": str, "timestamp": str}]` | High-Priority Social Feed & Urgent Safety Alerts |
| `GET` | `/api/v1/social/posts/by-location` | `location` *(required string)*, `top_n` *(default: 20)* | `[{"post_id": str, "text": str, "location": str, "concern_score": float, "sentiment": str, "timestamp": str}]` | Location-Filtered Social Feed Inspector |
| `GET` | `/api/v1/social/summary` | None | `{"total_posts": int, "unique_locations": int, "avg_concern_score": float, "negative_posts_pct": float}` | Social Intelligence Summary Cards |
| `POST` | `/api/v1/agent/query` | JSON: `{"query": str, "district"?: str, "road_type"?: str, ...}` | `{"query": str, "plan": [...], "agents_used": [...], "final_response": str, "latency_ms": float, "planner_reasoning": str, "knowledge_sources": [...]}` | AI Agent Chat Console, Execution Trace & Citations |

---

## 3. Data Flow & Guardrails Integration

1. **Input Guardrails & Validation:**
   - Any query sent to `/api/v1/agent/query` is validated against domain keywords and adversarial patterns.
   - If invalid or adversarial, the endpoint responds with HTTP `422 Unprocessable Entity` and a descriptive message. The frontend catches this and displays a safety badge with the rejection reason.

2. **Reliability & Wilson 95% Confidence Intervals:**
   - Classification accuracy: **97.15%** (Wilson 95% CI: **95.28% – 98.31%**).
   - In-scope query completion: **100%** (Wilson 95% CI: **61.0% – 100.0%**).
   - Adversarial guardrail block rate: **85.0%** (Wilson 95% CI: **70.9% – 92.9%**).

3. **Historical Data Notice:**
   - All frontend views prominently display: `"HISTORICAL DATA (2019-2023) — ANALYTICAL INTELLIGENCE ONLY"`.
