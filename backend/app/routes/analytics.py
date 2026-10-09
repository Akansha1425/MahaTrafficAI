"""Analytics routes — historical accident pattern endpoints."""

from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException, Query, status
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analytics", tags=["Historical Analytics"])


def _get_service():
    from backend.app.services.risk_service import get_risk_service
    return get_risk_service()


@router.get("/yearly", summary="Get yearly accident trend statistics")
async def get_yearly_analytics() -> List[Dict[str, Any]]:
    """Historical accident patterns aggregated by year (2019-2023)."""
    try:
        return _get_service().get_yearly_trend()
    except Exception as e:
        logger.error("Yearly analytics error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/districts", summary="Get district-wise risk summary")
async def get_district_analytics(
    district: str = Query(None, description="Filter by district name (optional)")
) -> List[Dict[str, Any]]:
    """Historical accident risk summary per Maharashtra district."""
    try:
        return _get_service().get_district_risk_summary(district=district)
    except Exception as e:
        logger.error("District analytics error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/model-metrics", summary="Get ML model evaluation metrics")
async def get_model_metrics() -> Dict[str, Any]:
    """Random Forest Classifier evaluation metrics from training."""
    try:
        return _get_service().get_model_metrics()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/feature-importances", summary="Get ML feature importances")
async def get_feature_importances() -> Dict[str, float]:
    """Feature importance rankings from the trained Random Forest model."""
    try:
        return _get_service().get_feature_importances()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
