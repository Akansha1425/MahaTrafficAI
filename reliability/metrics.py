"""Reliability Metrics Definitions and Schemas.

Defines tracking schemas for multi-agent reliability, tool success rate,
schema compliance, guardrail detection, and system repeatability.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ExecutionTraceRecord(BaseModel):
    """Structured record schema for each agent run evaluation."""

    run_id: str = Field(..., description="Unique run identifier (UUID)")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_question: str
    planner_version: str = "v1"
    agent: str = "orchestrator"
    tools_called: List[str] = Field(default_factory=list)
    tool_success: bool = True
    retry_count: int = 0
    review_status: str = "pending"
    risk_score: Optional[float] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    final_status: str = "success"


class ReliabilityMetricsReport(BaseModel):
    """Aggregated benchmark metrics across evaluation test suite."""

    total_runs: int = 0
    tool_success_rate: float = 0.0
    task_completion_rate: float = 0.0
    schema_compliance_rate: float = 0.0
    groundedness_score: float = 0.0
    retry_rate: float = 0.0
    failure_recovery_rate: float = 0.0
    guardrail_detection_rate: float = 0.0
    false_positive_rate: float = 0.0
    prompt_injection_resistance_rate: float = 0.0
    mean_latency_ms: float = 0.0
    determinism_score: float = 0.0
