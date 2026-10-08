# System Architecture — MahaTraffic AI

## High-Level System Architecture

The **MahaTraffic AI** system is structured as an end-to-end historical road-risk intelligence platform utilizing Big Data pipelines, Social Media analytics, Model Context Protocol (MCP), and Guardrailed Multi-Agent systems.

```
USER
  ↓
React Frontend (Vite + TypeScript + Tailwind CSS)
  ↓
FastAPI Backend (/api/v1)
  ↓
Input Guardrail (Validation, Domain Check & Injection Detection)
  ↓
Planner Agent (Task Decomposition & Execution Planning)
  ↓
LangGraph Workflow / Orchestrator
  ↓
Specialized Agents:
  ├── Analytics Agent (Spark & Historical Statistics)
  ├── Risk Agent (Random Forest Prediction & Severity Scoring)
  ├── Social Media Agent (Sentiment, Complaint Signals & Trends)
  └── Knowledge/RAG Agent (FAISS & MoRTH Road Safety Retrieval)
  ↓
MCP Client
  ↓
MCP Server (Strict Tool Allowlist & Sandboxed Execution)
  ↓
Controlled MCP Tools:
  ├── get_city_accident_statistics
  ├── get_monthly_accident_statistics
  ├── get_yearly_accident_statistics
  ├── calculate_risk_score
  ├── predict_risk
  ├── analyze_sentiment
  ├── get_social_trends
  └── search_road_safety_documents
  ↓
Reviewer Agent (Evidence Verification & Fact Checking)
  ↓
Output Guardrail (Schema Enforcement, Numerical Consistency & Grounding)
  ↓
Structured Final Response
  ↓
React Dashboard / Chat Interface
```

## Modular Components
1. **Big Data Processing Layer (PySpark / Parquet / Hadoop ecosystem):** Handles historical accident records, spatiotemporal aggregations, and stream algorithm simulations.
2. **Machine Learning Layer (scikit-learn):** Provides predictive risk scoring models based on historical road conditions, district geography, and accident severity.
3. **Social Media Analytics Layer (NLP & NetworkX):** Analyzes citizen feedback and civic complaint signals as public perception data (explicitly decoupled from causality).
4. **Agentic & MCP Layer (LangGraph & MCP):** Coordinates autonomous reasoning with verified, sandboxed tool executions.
5. **Guardrails & Reliability Engine:** Continuous validation of inputs, outputs, and tool calls with automated tracking of system metrics.
