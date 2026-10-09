# Model Context Protocol (MCP) Design & Implementation

## 1. Protocol Overview & Architectural Distinction

In MahaTraffic AI, the Model Context Protocol (MCP) layer decouples agent reasoning from underlying data and analytical services. 

### Implementation Architecture
- **In-Process & Local Protocol Interface**: Implemented via `mcp_server.client.MCPClient` and `mcp_server.tools.ALLOWED_TOOLS_REGISTRY`.
- **Standalone Server Interface**: Also mounted as a dedicated FastAPI tool server on port `8001` with OpenAPI schema validation and endpoints for external AI agent integration.
- **Protocol Distinction**: The system implements structured JSON-RPC / REST tool contracts using Pydantic schemas. It does not introduce unverified or unpinned external MCP packages, maintaining full Windows / VS Code runtime compatibility without external server daemons.

---

## 2. Canonical Approved Tools Registry

The tool server enforces a strict allowlist of 7 core tools:

| Tool Identifier | Module / Service | Description | Input Parameters | Output Payload |
|---|---|---|---|---|
| `get_city_accident_statistics` | `mcp_server.tools.accident_tools` | Historical district accident statistics (2019–2023) | `city_name` (str), `year` (optional int) | Total accidents, deaths, injuries, top causes |
| `get_monthly_accident_statistics` | `mcp_server.tools.accident_tools` | Seasonality and monthly accident patterns | `year` (optional int), `district` (optional str) | Monthly breakdown, monsoon vs non-monsoon stats |
| `calculate_risk_score` | `mcp_server.tools.risk_tools` | Formula-based normalized risk index [0–100] | Accident count, deaths, injuries, road type, time | `risk_score` (float), `risk_level` (str), components |
| `predict_risk` | `mcp_server.tools.risk_tools` | Random Forest ML classification tier | District, road type, primary cause, accident metrics | `predicted_category`, `confidence`, top factors |
| `analyze_sentiment` | `mcp_server.tools.social_tools` | Citizen grievance and sentiment distribution | `location` (optional str), `texts` (optional list) | Sentiment counts (pos/neu/neg), sample posts, non-causal disclaimer |
| `get_social_trends` | `mcp_server.tools.social_tools` | Recurring road complaint topics & rankings | `location` (optional str), `limit` (int) | Top topics, high-concern locations, monthly volume |
| `search_road_safety_documents` | `mcp_server.tools.rag_tools` | Semantic retrieval from MoRTH & IRC guidelines | `query` (str), `top_k` (int) | Retrieved chunks, similarity scores, source citations |

---

## 3. Tool Guardrail & Parameter Sanitization

Before any tool is executed by `MCPClient`:
1. **Allowlist Verification**: Any tool name not in `ALLOWED_TOOLS_REGISTRY` is immediately rejected with a `ToolSecurityError`.
2. **Argument Sanitization**: Parameters are inspected for dangerous injection vectors (`os.system`, `subprocess`, `rm -rf`, `eval(`, `__import__`).
3. **Execution Timeout**: Configurable timeout (default 10s) prevents runaway operations.

---

## 4. Bounded Retries & Controlled Fallback

- **Bounded Retry**: If a tool encounters a transient error (e.g. disk I/O contention or serialization failure), `MCPClient` executes at most **two retries** with exponential backoff (initial + 2 retries = 3 attempts total).
- **Controlled Fallback**: If all attempts fail:
  - The client does *not* crash or raise unhandled exceptions.
  - The client *never* replaces missing data with invented or fabricated numbers.
  - Returns `ToolResponse(status="fallback", data={"is_fallback": True, "message": "Service temporarily unavailable", ...})`.
  - The fallback status is transparently logged in audit traces.
