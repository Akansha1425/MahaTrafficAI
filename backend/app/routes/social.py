"""Social media analytics routes."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/social", tags=["Social Media Analytics"])


def _get_service():
    from backend.app.services.social_service import get_social_service
    return get_social_service()


@router.get("/sentiment", summary="Overall sentiment distribution")
async def get_sentiment_distribution() -> Dict[str, int]:
    """POSITIVE/NEGATIVE/NEUTRAL post count distribution."""
    try:
        return _get_service().get_sentiment_distribution()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/concern-ranking", summary="Location public concern ranking")
async def get_concern_ranking(
    top_n: int = Query(10, ge=1, le=50, description="Number of locations to return")
) -> List[Dict[str, Any]]:
    """Ranked list of Maharashtra locations by average public road-safety concern."""
    try:
        return _get_service().get_location_concern_ranking(top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/topics", summary="Topic frequency distribution")
async def get_topic_frequencies() -> Dict[str, int]:
    """Road-safety topic labels and their frequencies from public posts."""
    try:
        return _get_service().get_topic_frequencies()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/trend", summary="Yearly sentiment trend")
async def get_sentiment_trend() -> List[Dict[str, Any]]:
    """Average sentiment score per year (2021-2023)."""
    try:
        return _get_service().get_yearly_sentiment_trend()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/posts/concerning", summary="Most concerning posts")
async def get_concerning_posts(
    top_n: int = Query(10, ge=1, le=50)
) -> List[Dict[str, Any]]:
    """Top posts ranked by concern score."""
    try:
        return _get_service().get_most_concerning_posts(top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/posts/by-location", summary="Posts filtered by location")
async def get_posts_by_location(
    location: str = Query(..., description="Maharashtra city/district name"),
    top_n: int = Query(20, ge=1, le=100),
) -> List[Dict[str, Any]]:
    """Retrieve social posts filtered by location."""
    try:
        return _get_service().get_location_posts(location=location, top_n=top_n)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/summary", summary="Social analytics summary statistics")
async def get_social_summary() -> Dict[str, Any]:
    """High-level summary of the social media dataset."""
    try:
        return _get_service().get_summary_stats()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
