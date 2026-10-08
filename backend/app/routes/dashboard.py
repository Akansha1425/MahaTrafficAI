"""Dashboard route definitions for executive summaries and key aggregates."""

from fastapi import APIRouter, HTTPException, status
from backend.app.models.responses import DashboardSummaryResponse

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get(
    "/summary",
    response_model=DashboardSummaryResponse,
    summary="Get high-level historical accident and risk summary",
)
async def get_dashboard_summary():
    """Retrieve high-level dashboard aggregate metrics.

    Note: This is an architectural scaffold. Real aggregations will be plugged
    in during Phase 14 after Spark and MongoDB data pipelines are finalized.
    """
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Dashboard summary pipeline scheduled for Phase 14 integration.",
    )
