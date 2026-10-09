"""Risk assessment routes — formula-based and ML inference endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/risk", tags=["Risk Assessment"])


class RiskScoreRequest(BaseModel):
    accident_count: int = Field(50, ge=0, description="Total accident count")
    fatal_accidents: int = Field(5, ge=0, description="Number of fatal accidents")
    deaths: int = Field(5, ge=0, description="Total deaths")
    injuries: int = Field(30, ge=0, description="Total injuries")
    road_type: str = Field("Urban / City Road", description="Road classification")
    time_period: str = Field("day", description="Time period (day/night/peak hour)")
    district: Optional[str] = Field(None, description="Maharashtra district name")


class MLPredictRequest(BaseModel):
    district: str = Field("Pune", description="Maharashtra district")
    road_type: str = Field("Urban / City Road", description="Road type")
    primary_cause: str = Field("Over-speeding", description="Primary accident cause")
    year: int = Field(2023, ge=2019, le=2030)
    month: int = Field(6, ge=1, le=12)
    accident_count: int = Field(50, ge=0)
    fatal_accidents: int = Field(5, ge=0)
    deaths: int = Field(5, ge=0)
    injuries: int = Field(30, ge=0)
    is_monsoon: int = Field(0, ge=0, le=1)
    is_night: int = Field(0, ge=0, le=1)
    is_highway: int = Field(0, ge=0, le=1)
    fatality_ratio: float = Field(0.10, ge=0.0, le=1.0)
    injury_ratio: float = Field(0.60, ge=0.0, le=1.0)
    severity_index: float = Field(0.50, ge=0.0)


def _get_service():
    from backend.app.services.risk_service import get_risk_service
    return get_risk_service()


@router.post("/score", summary="Calculate composite risk score (formula-based)")
async def calculate_risk_score(req: RiskScoreRequest) -> Dict[str, Any]:
    """Compute composite road risk score using the MahaTraffic formula."""
    try:
        return _get_service().calculate_risk_score(
            accident_count=req.accident_count,
            fatal_accidents=req.fatal_accidents,
            deaths=req.deaths,
            injuries=req.injuries,
            road_type=req.road_type,
            time_period=req.time_period,
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/predict", summary="Predict risk category using ML model")
async def predict_risk_category(req: MLPredictRequest) -> Dict[str, Any]:
    """Classify risk category using the trained Random Forest model."""
    try:
        return _get_service().predict_risk_category(req.model_dump())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get("/districts", summary="Get district risk rankings")
async def get_district_risk(
    district: str = Query(None, description="Optional: filter by district name")
) -> List[Dict[str, Any]]:
    """Historical risk ranking for all Maharashtra districts."""
    try:
        return _get_service().get_district_risk_summary(district=district)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
