# Multi-Agent System Architecture — MahaTraffic AI

## Agent Orchestration Workflow

The agentic layer utilizes **LangGraph** to coordinate autonomous, specialized agents with explicit boundaries, guardrails, and verified Model Context Protocol (MCP) tool execution.

```
User Query
    │
    ▼
[Input Guardrail] ──────── (Fails validation / prompt injection) ──► Refusal / Safe Error
    │ (Pass)
    ▼
[Planner Agent]
    │ Decomposes query into structured task plan with intent and location
    ▼
[Task Router / LangGraph]
    ├──► [Analytics Agent]       ──► Calls MCP: get_city_accident_statistics, get_yearly_accident_statistics
    ├──► [Risk Agent]            ──► Calls MCP: calculate_risk_score, predict_risk
    ├──► [Social Agent]          ──► Calls MCP: analyze_sentiment, get_social_trends
    └──► [Knowledge/RAG Agent]   ──► Calls MCP: search_road_safety_documents
    │
    ▼
[Reviewer Agent]
    │ Synthesizes answers, validates citations against evidence, checks numerical consistency
    ▼
[Output Guardrail] ─────── (Schema mismatch / unsupported claim) ──► Repair / Fallback
    │ (Pass)
    ▼
Final Structured Response (Answer + Evidence + Recommendations + Metrics)
```

## Agent Roles and Responsibilities

1. **Planner Agent:**
   - Parses user questions (e.g., *"Why is Pune historically high risk during monsoons?"*).
   - Generates structured Pydantic plan specifying sub-tasks and required data domains.

2. **Analytics Agent:**
   - Solicits historical statistics, accident counts, and trends from PySpark / Parquet data stores via MCP.

3. **Risk Agent:**
   - Computes weighted severity scores and requests Random Forest risk level predictions via MCP.

4. **Social Agent:**
   - Retrieves public sentiment and civic complaints. Strictly contextualizes findings as perception signals rather than accident causes.

5. **Knowledge / RAG Agent:**
   - Retrieves authoritative road safety recommendations and Indian Roads Congress (IRC) standards from FAISS.

6. **Reviewer Agent:**
   - Cross-checks all assertions in the final response against evidence payloads returned by the MCP tools. Flags hallucinated statistics.

## Guardrail Safeguards
- **Input Guardrail:** Rejects prompt injections, jailbreak instructions, and unrelated out-of-domain questions.
- **Tool Guardrail:** Sandboxes execution to only allowlisted MCP tools; prevents shell execution, file system tampering, or destructive database calls.
- **Output Guardrail:** Validates structured response schema, verifies that numerical claims match tool evidence, and ensures clear disclaimers are present.
