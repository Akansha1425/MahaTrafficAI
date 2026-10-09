"""Agent query route — multi-agent orchestration endpoint."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent", tags=["Multi-Agent System"])


class AgentQueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="Road safety query for the multi-agent system",
        example="Which districts in Maharashtra have the highest road accident risk?",
    )
    district: str | None = Field(None, description="Optional district context")
    road_type: str | None = Field(None, description="Optional road type context")
    accident_count: int | None = Field(None, description="Optional accident count")
    deaths: int | None = Field(None, description="Optional death count")
    injuries: int | None = Field(None, description="Optional injury count")


@router.post("/query", summary="Submit a query to the multi-agent system")
async def agent_query(req: AgentQueryRequest) -> Dict[str, Any]:
    """Run the full MahaTraffic AI multi-agent pipeline.

    Agents invoked:
    - PlannerAgent: Routes query to relevant specialists
    - AnalyticsAgent: Historical accident data analysis
    - RiskAgent: Risk score computation and ML inference
    - SocialAgent: Social media sentiment analysis
    - KnowledgeAgent: RAG-based road safety knowledge retrieval
    - ReviewerAgent: Assembles and validates final response
    """
    # Apply input guardrails
    from agents.guardrails.guardrails import apply_input_guardrails
    guard_result = apply_input_guardrails(req.query)

    if not guard_result["is_valid"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "Input guardrail rejection",
                "reason": guard_result["reason"],
                "query": req.query,
            },
        )

    try:
        from agents.workflow.multi_agent import run_agent_workflow

        extra = {}
        if req.district:
            extra["district"] = req.district
        if req.road_type:
            extra["road_type"] = req.road_type
        if req.accident_count is not None:
            extra["accident_count"] = req.accident_count
        if req.deaths is not None:
            extra["deaths"] = req.deaths
        if req.injuries is not None:
            extra["injuries"] = req.injuries

        context = run_agent_workflow(
            query=guard_result["sanitized_query"],
            extra_context=extra if extra else None,
        )

        return {
            "query": req.query,
            "plan": context.get("plan", []),
            "agents_used": context.get("agents_used", []),
            "final_response": context.get("final_response", ""),
            "latency_ms": context.get("latency_ms"),
            "planner_reasoning": context.get("planner_reasoning", ""),
            "analytics_data": context.get("analytics_data"),
            "risk_data": context.get("risk_data"),
            "social_data": context.get("social_data"),
            "knowledge_sources": context.get("knowledge_sources", []),
        }

    except Exception as e:
        logger.error("Agent workflow error: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow failed: {str(e)}",
        )
