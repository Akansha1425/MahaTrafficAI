# API Design and Specification Plan — MahaTraffic AI

## Overview
The MahaTraffic AI backend exposes RESTful endpoints under `/api/v1` for dashboard visualization, historical analytics, risk intelligence, and interactive multi-agent chat.

## Endpoint Specifications

### 1. System & Health
- `GET /health`: Operational status and service information.

### 2. Executive Dashboard
- `GET /api/v1/dashboard/summary`: High-level summary of total historical accidents, analyzed locations, high-risk clusters, and active data years.

### 3. Historical Big Data Analytics
- `GET /api/v1/analytics/yearly`: Aggregated annual trends of accident frequencies and fatalities across Maharashtra.
- `GET /api/v1/analytics/monthly`: Monthly distribution and seasonal fluctuations (e.g. monsoon accident surges).
- `GET /api/v1/analytics/cities`: Comparative accident statistics across districts (Pune, Mumbai, Nagpur, Nashik, etc.).
- `GET /api/v1/analytics/road-types`: Breakdowns by road classification (National Highways, State Highways, Expressways, Rural Roads).

### 4. Road Risk Assessment
- `GET /api/v1/risk/{location}`: Precalculated risk metrics, historical fatality ratios, and severity indices for a given district.
- `POST /api/v1/risk/predict`: ML-based risk tier prediction (Low, Medium, High, Critical) based on road characteristics and temporal features.

### 5. Social Media Perception Signals
- `GET /api/v1/social/{location}`: Public sentiment scores, civic complaint topics (potholes, waterlogging, signage), and trend counts.

### 6. AI Agent Chat & Reasoning
- `POST /api/v1/agent/chat`: Natural language query processing via LangGraph multi-agent pipeline with verified evidence citations and guardrail enforcement.

> **Status:** Route definitions and Pydantic schemas scaffolded in Phase 1. Endpoints will be connected to active pipeline services in Phase 14.
