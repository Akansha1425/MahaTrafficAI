# Technical Viva & Evaluation Guide — MahaTraffic AI

## Project Overview for Examiners
**MahaTraffic AI** is an advanced academic and portfolio system demonstrating the intersection of **Big Data Engineering**, **Machine Learning**, **Social Media Analytics**, and **Agentic AI / MCP Architecture**.

## Key Viva Talking Points & Clarifications

### 1. Core Distinction: Historical Intelligence vs Live Monitoring
- **Question:** *Is this a live traffic monitoring or navigation system?*
- **Answer:** **No.** MahaTraffic AI is explicitly designed as a **historical road-risk intelligence system**. It mines historical multi-year records to uncover systemic risk patterns, seasonal trends, and road condition factors. It does not monitor live vehicle feeds or claim real-time accident prevention.

### 2. Social Media Signal Treatment
- **Question:** *Does social media sentiment cause accidents?*
- **Answer:** **No.** Public social media posts and civic complaints are treated strictly as an **independent public perception signal**. They highlight citizen grievances (potholes, bad streetlights, waterlogging) but are never claimed as a causal determinant of historical accident figures.

### 3. Role of Model Context Protocol (MCP)
- **Question:** *Why use MCP instead of direct LLM tool calling?*
- **Answer:** MCP introduces a standardized, security-isolated protocol between AI agents and external tools. It enforces strict tool allowlisting, schema boundaries, timeouts, and logging, preventing arbitrary code or destructive operations.

### 4. Big Data & Academic Syllabus Alignment
- **PySpark & Parquet:** Columnar spatiotemporal aggregations across large accident datasets.
- **Hadoop Ecosystem:** MapReduce & Hive concepts applied to large-scale data queries.
- **Streaming Algorithms:** Simulated Bloom Filters, DGIM (for bit-stream frequency), and Flajolet-Martin (distinct count estimation).
- **Social Network Analysis:** Louvain community detection and sentiment distribution.

### 5. Multi-Agent System & Guardrails
- Implements **LangGraph** orchestrating Planner, Analytics, Risk, Social, RAG, and Reviewer agents.
- Three-tier guardrails (Input Guardrail, Tool Guardrail, Output Guardrail) ensure robustness and zero hallucinated claims.
