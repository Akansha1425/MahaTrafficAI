# Reliability and Evaluation Plan — MahaTraffic AI

## Evaluation Methodology

The MahaTraffic AI system incorporates continuous reliability tracking and rigorous evaluation across data pipelines, machine learning models, and multi-agent workflows.

## Primary System Evaluation Metrics

1. **Tool Success Rate (TSR):** Ratio of MCP tool executions that complete successfully without errors or timeouts.
2. **Task Completion Rate (TCR):** Proportion of user queries where all planned agent sub-tasks are fully resolved.
3. **Schema Compliance Rate:** Verification that all agent outputs adhere 100% to strict Pydantic models.
4. **Groundedness & Evidence Accuracy:** Percentage of factual claims in the synthesis directly supported by MCP tool output.
5. **Prompt Injection Resistance:** Percentage of adversarial or malicious inputs successfully intercepted and neutralized by the Input Guardrail.
6. **Guardrail False Positive Rate:** Frequency with which benign domain-relevant queries are mistakenly blocked.
7. **Failure Recovery Rate:** Success rate of agent retries and fallbacks when an MCP tool or upstream LLM call encounters transient failure.
8. **End-to-End Latency:** Mean and 95th-percentile response times for agent workflow execution.
9. **Determinism & Consistency:** Variance in generated analytical conclusions across multiple runs on identical inputs.

## Logging and Observability
Every agent execution generates a persistent `ExecutionTraceRecord` containing:
- `run_id` (UUID)
- `timestamp`
- `user_question`
- `planner_version`
- `tools_called` & `tool_success`
- `retry_count`
- `guardrail_status`
- `latency_ms`
- `final_status`

Traces are stored in MongoDB and analyzed through `reliability/evaluator.py`.
