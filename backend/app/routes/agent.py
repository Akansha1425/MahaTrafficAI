"""Agent orchestrator routes for natural language inquiries."""

from fastapi import APIRouter, HTTPException, status
from backend.app.models.requests import AgentChatRequest
from backend.app.models.responses import AgentChatResponse

router = APIRouter(prefix="/agent", tags=["Multi-Agent Orchestrator"])


@router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Process traffic intelligence query via multi-agent pipeline",
)
async def query_agent(request: AgentChatRequest):
    """Execute LangGraph multi-agent pipeline with Guardrails and MCP tools.

    Pipeline flow: Input Guardrail -> Planner -> Specialized Agents (Analytics, Risk,
    Social, RAG) -> Reviewer -> Output Guardrail.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Multi-agent orchestrator pipeline scheduled for Phase 14 integration.",
    )
