# MahaTraffic AI

**A Reliable Multi-Agent Road-Risk Intelligence System using Big Data, Social Media Analytics, MCP and Guardrails**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![Status: Phase 1 Initialized](https://img.shields.io/badge/Status-Phase%201%20Initialized-orange.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 1. Project Objective & Vision

**MahaTraffic AI** is an advanced academic and portfolio system designed to analyze **historical road-traffic accident patterns across Maharashtra**, synthesizing official accident statistics and road-safety guidelines with controlled public/social-media complaint signals.

The system combines:
- **Big Data Analytics:** Scalable spatiotemporal aggregations (PySpark, Parquet, Hadoop ecosystem).
- **Social Media Analytics:** Citizen perception, complaint topic clustering, and sentiment signals (NLP, NetworkX).
- **Machine Learning:** Historical road-risk index computation and Random Forest classification.
- **Retrieval-Augmented Generation (RAG):** Grounded road-safety manual retrieval via FAISS.
- **Agentic AI & Model Context Protocol (MCP):** Coordinated multi-agent reasoning (LangGraph) over secure, allowlisted MCP tools.
- **AI Guardrails & Reliability:** Three-tier validation (Input, Tool, Output) and continuous benchmarking of agent correctness and determinism.
- **Full-Stack Dashboard:** Interactive React/Vite dashboard for exploratory analytics and agent chat.

> ⚠️ **CRITICAL DISCLAIMER: HISTORICAL INTELLIGENCE ONLY**
> 
> MahaTraffic AI is strictly a **historical road-risk intelligence system**, **NOT** a live traffic monitoring or navigation platform.
> - **No live traffic congestion detection:** We do not ingest real-time GPS feeds or vehicle telemetry.
> - **No real-time accident prediction:** Predictions reflect historical risk distributions given temporal and road condition factors.
> - **No causal claims:** Social-media sentiment is analyzed solely as an independent **public-perception / civic-complaint signal** (e.g., citizen grievances about potholes or lighting), **never** as direct causal evidence of traffic accidents.
> - **No official government risk ratings:** Risk scores are academic and statistical estimations.

---

## 2. Problem Statement

Road traffic accidents in Maharashtra represent a critical public safety challenge characterized by complex spatiotemporal patterns, seasonal variations (e.g., monsoon disruptions), infrastructure conditions, and district-level disparities. While historical accident reports exist across public portals, they are often fragmented, difficult to query interactively, and detached from public perception and authoritative safety guidelines. 

MahaTraffic AI solves this by unifying:
1. Historical accident data pipelines.
2. Controlled public perception signals.
3. Autonomous AI agents capable of answering complex risk queries with verifiable citations and guardrails against hallucination.

---

## 3. High-Level Architecture

```
USER
  ↓
React Frontend (Vite + TypeScript + Tailwind CSS)
  ↓
FastAPI Backend (/api/v1)
  ↓
Input Guardrail (Validation, Domain Check & Injection Detection)
  ↓
Planner Agent (Task Decomposition & Structured Planning)
  ↓
LangGraph Workflow / Orchestrator
  ↓
Specialized Agents:
  ├── Analytics Agent (Spark & Historical Statistics)
  ├── Risk Agent (Random Forest Prediction & Severity Scoring)
  ├── Social Media Agent (Public Sentiment, Complaint Signals & Trends)
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

---

## 4. Planned Technology Stack

| Layer | Technologies |
|---|---|
| **Backend & API** | Python 3.11, FastAPI, Pydantic v2, Uvicorn, httpx |
| **Big Data Engineering** | Apache Spark, PySpark, Spark SQL, Parquet, Hadoop/HDFS, MapReduce |
| **Data Storage** | MongoDB (application metadata, logs, aggregates), Parquet (columnar data lake) |
| **Machine Learning** | scikit-learn, Random Forest, Pandas, NumPy, joblib |
| **Social Analytics** | NLP preprocessing, sentiment analysis, topic modeling, NetworkX, Louvain |
| **Agentic AI & MCP** | LangGraph, LangChain, Model Context Protocol (MCP), OpenRouter LLMs |
| **RAG & Knowledge Base** | FAISS, sentence-transformers (`all-MiniLM-L6-v2`), MoRTH guidelines |
| **Security & Guardrails** | Input Guardrail, MCP Tool Allowlisting, Output Schema Enforcement |
| **Reliability & Testing** | Pytest, pytest-asyncio, adversarial testing suites, execution trace logging |
| **Frontend Dashboard** | React 18, Vite, TypeScript, Tailwind CSS, shadcn/ui, Recharts |

---

## 5. Repository Structure

```
mahatraffic-ai/
├── README.md                           # Master project documentation
├── .gitignore                          # Comprehensive Git ignore rules
├── .env.example                        # Environment configuration template
├── requirements.txt                    # Pinned Python package dependencies
├── pyproject.toml                      # Modern Python build & packaging metadata
├── docker-compose.yml                  # Multi-service container specification
│
├── data/                               # Data storage (gitignored raw/processed data)
│   ├── raw/                            # Unmodified source files
│   │   ├── accidents/                  # Historical accident records
│   │   ├── social_media/               # Public posts and civic complaint feeds
│   │   └── road_safety/                # Official manuals and guidelines
│   ├── processed/                      # Transformed columnar datasets
│   │   ├── parquet/                    # Partitioned Parquet tables
│   │   ├── spark/                      # Spark warehouse tables
│   │   └── features/                   # ML feature matrices
│   └── sample/                         # Representative test subsets
│
├── bigdata/                            # Big Data processing modules
│   ├── hdfs/                           # HDFS configuration & ingest scripts
│   ├── mapreduce/                      # MapReduce batch jobs
│   ├── hive/                           # Hive DDL and queries
│   ├── spark/                          # PySpark pipelines
│   │   ├── ingestion/                  # Raw file ingest to Spark
│   │   ├── cleaning/                   # Data cleansing & type casting
│   │   ├── feature_engineering/        # Feature extraction
│   │   └── analytics/                  # Spatiotemporal aggregations
│   └── streaming_algorithms/           # Stream algorithm simulations
│       ├── bloom_filter/               # Duplicate event detection
│       ├── dgim/                       # Bit-stream window counting
│       └── flajolet_martin/            # Distinct location estimation
│
├── social_analytics/                   # Public perception & social signal modules
│   ├── preprocessing/                  # Text cleaning & normalization
│   ├── sentiment/                      # Sentiment classification
│   ├── topics/                         # Topic modeling & hashtag extraction
│   ├── trends/                         # Temporal complaint trend analysis
│   ├── engagement/                     # Interaction & reach metrics
│   └── community_detection/            # NetworkX & Louvain community clustering
│
├── ml/                                 # Machine learning pipelines
│   ├── training/                       # Model training routines
│   ├── models/                         # Serialized model artifacts (.joblib)
│   ├── evaluation/                     # Confusion matrices & ROC-AUC curves
│   └── prediction/                     # Inference engine
│
├── backend/                            # FastAPI backend service
│   ├── app/
│   │   ├── main.py                     # Application entry point & /health
│   │   ├── config.py                   # Pydantic BaseSettings configuration
│   │   ├── routes/                     # REST API route handlers
│   │   ├── services/                   # Business logic and domain service interfaces
│   │   ├── models/                     # Request and response Pydantic schemas
│   │   └── utils/                      # Logging & utility functions
│   └── tests/                          # Backend-specific unit tests
│
├── mcp_server/                         # Model Context Protocol (MCP) server
│   ├── server.py                       # MCP Server implementation
│   ├── tools/                          # Allowlisted MCP tool modules
│   └── schemas/                        # Pydantic tool input/output contracts
│
├── agents/                             # LangGraph multi-agent architecture
│   ├── planner/                        # Query decomposition & intent extraction
│   ├── analytics_agent/                # Historical statistical reasoning
│   ├── risk_agent/                     # Severity analysis & ML risk scoring
│   ├── social_agent/                   # Perception & civic complaint reasoning
│   ├── knowledge_agent/                # RAG retrieval over road safety manuals
│   ├── reviewer/                       # Verification, citation checks & synthesis
│   ├── guardrails/                     # Input, tool, and output guardrail filters
│   └── workflow/                       # LangGraph state machine & graph runner
│
├── rag/                                # Retrieval-Augmented Generation layer
│   ├── documents/                      # Official road safety documents & manuals
│   ├── preprocessing/                  # Chunking and text splitting
│   ├── embeddings/                     # Embedding generation pipeline
│   ├── index/                          # FAISS vector index files
│   └── retrieval.py                    # Semantic retriever interface
│
├── reliability/                        # System reliability & benchmarking harness
│   ├── metrics.py                      # Trace schemas & reliability KPI models
│   ├── evaluator.py                    # Batch benchmark evaluation runner
│   ├── adversarial_tests.py            # Prompt injection & jailbreak test cases
│   └── logs/                           # Execution traces & run logs
│
├── frontend/                           # React + Vite dashboard (scaffold)
│   └── README.md                       # Frontend specifications & roadmap
│
├── notebooks/                          # Academic Jupyter experiment notebooks
│   ├── 01_data_exploration.ipynb       # EDA on historical accident records
│   ├── 02_spark_processing.ipynb       # PySpark ingestion & Parquet transformations
│   ├── 03_social_analysis.ipynb        # Sentiment & complaint topic analysis
│   ├── 04_risk_model.ipynb             # Feature engineering & Random Forest training
│   └── 05_evaluation.ipynb             # Agent reliability & guardrail benchmarks
│
├── tests/                              # Comprehensive pytest suite
│   ├── data/                           # Data integrity tests
│   ├── backend/                        # API route & service tests
│   ├── agents/                         # Agent workflow tests
│   ├── mcp/                            # MCP tool execution tests
│   └── integration/                    # End-to-end integration tests
│
└── docs/                               # Project documentation & viva material
    ├── architecture/                   # Detailed system architectural blueprints
    ├── datasets/                       # Dataset acquisition and schema plans
    ├── api/                            # RESTful API specifications
    ├── agent_design/                   # Multi-agent interactions and guardrails
    ├── evaluation/                     # Reliability metrics and evaluation methods
    └── viva/                           # Academic viva preparation & examiner guide
```

---

## 6. Development Roadmap

| Phase | Description | Status |
|---|---|:---:|
| **PHASE 1** | Project structure, environment configuration, FastAPI `/health` entry point | **Completed** |
| **PHASE 2** | Dataset discovery, sourcing and data schema design | *Next Step* |
| **PHASE 3** | Data cleaning, validation, and preprocessing pipelines | *Pending* |
| **PHASE 4** | Hadoop / HDFS / MapReduce / Hive batch experiments | *Pending* |
| **PHASE 5** | PySpark analytics, aggregations, and Parquet data pipeline | *Pending* |
| **PHASE 6** | Social-media sentiment, topic modeling, and network analysis | *Pending* |
| **PHASE 7** | Risk feature engineering and Random Forest model training | *Pending* |
| **PHASE 8** | MongoDB integration for execution logs and analytical caches | *Pending* |
| **PHASE 9** | RAG knowledge base creation with FAISS & MoRTH manuals | *Pending* |
| **PHASE 10** | Model Context Protocol (MCP) server & allowlisted tool implementations | *Pending* |
| **PHASE 11** | LangGraph Planner + Specialized Agents implementation | *Pending* |
| **PHASE 12** | Reviewer Agent + Three-tier Guardrails integration | *Pending* |
| **PHASE 13** | Reliability benchmarking, adversarial tests, and metric logging | *Pending* |
| **PHASE 14** | FastAPI complete integration connecting pipelines to API routes | *Pending* |
| **PHASE 15** | React + Vite frontend dashboard implementation | *Pending* |
| **PHASE 16** | End-to-end integration and smoke testing | *Pending* |
| **PHASE 17** | Performance optimization, load testing, and documentation finalize | *Pending* |
| **PHASE 18** | Demonstration prep, research report, and technical viva defense | *Pending* |

---

## 7. Setup & Getting Started

### Prerequisites
- **Python:** 3.11+ (recommended: 3.11.x)
- **Git**
- **Node.js:** v18+ (for Phase 15 frontend)
- **MongoDB:** (Local or Docker for Phase 8+)

### Local Environment Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd MahaTrafficAI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell:
   .\.venv\Scripts\Activate.ps1
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   copy .env.example .env
   # Edit .env with your local settings (OpenRouter API key, MongoDB URI, etc.)
   ```

5. **Run the test suite:**
   ```bash
   pytest -v
   ```

6. **Start the FastAPI development server:**
   ```bash
   python -m uvicorn backend.app.main:app --reload --port 8000
   ```

7. **Verify the Health Check endpoint:**
   - URL: `http://localhost:8000/health`
   - Interactive OpenAPI documentation: `http://localhost:8000/docs`

---

## 8. Academic & Viva Alignment

This project is tailored for academic evaluation, advanced coursework, and technical vivas. It satisfies:
- **Big Data Engineering Practicals:** PySpark DataFrame operations, Parquet storage, stream algorithms (Bloom filter, DGIM, Flajolet-Martin).
- **Social Media Analytics:** Sentiment polarity, topic extraction, Louvain community detection.
- **Agentic AI & MCP Standards:** Distributed task planning, secure tool allowlisting, stateful LangGraph workflows.
- **Reliability & Trustworthy AI:** Quantitative measurement of tool success rate, schema compliance, prompt injection resistance, and groundedness.

---

## 9. Current Phase Status

> **Current Phase:** **Phase 1 — Project Initialization**  
> All base folders, module namespaces, configuration loaders, Pydantic schemas, initial documentation blueprints, and live FastAPI health endpoints have been initialized and verified.
> 
> **Next Scheduled Step:** **Phase 2 — Dataset Discovery and Data Schema Design** (identifying open government accident data sources for Maharashtra).
