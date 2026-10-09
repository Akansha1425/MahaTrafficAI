"""MCP Tool implementations for risk index calculations and ML predictions."""

from __future__ import annotations
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


def calculate_risk_score(
    accident_count: int = 50,
    fatal_accidents: int = 5,
    deaths: int = 5,
    injuries: int = 30,
    road_type: str = "National Highway",
    time_period: str = "night",
    accident_severity: Optional[float] = None,
    accident_frequency: Optional[float] = None,
    road_risk_factor: Optional[float] = None,
    time_risk_factor: Optional[float] = None,
) -> Dict[str, Any]:
    """Calculate normalized historical road risk score (0-100) and risk level.

    Returns:
        Structured dictionary with risk_score, risk_level, components, and disclaimers.
    """
    if accident_severity is not None and accident_frequency is not None:
        from backend.app.services.intelligence_service import calculate_risk
        res = calculate_risk(
            accident_severity=accident_severity,
            accident_frequency=accident_frequency,
            road_risk_factor=road_risk_factor if road_risk_factor is not None else 50.0,
            time_risk_factor=time_risk_factor if time_risk_factor is not None else 50.0,
        )
        return {
            "status": "success",
            "risk_score": float(res["risk_score"]),
            "risk_level": str(res["risk_level"]),
            "risk_category": str(res["risk_level"]),
            "components": res.get("components", {}),
            "disclaimer": res.get("disclaimer", "Historical analytical score. Not an official government standard."),
        }

    from backend.app.services.risk_service import get_risk_service
    svc = get_risk_service()
    res = svc.calculate_risk_score(
        accident_count=accident_count,
        fatal_accidents=fatal_accidents,
        deaths=deaths,
        injuries=injuries,
        road_type=road_type,
        time_period=time_period,
    )
    score = float(res["risk_score"])
    category = str(res["risk_category"])
    return {
        "status": "success",
        "risk_score": score,
        "risk_level": category,
        "risk_category": category,
        "components": res.get("components", {}),
        "disclaimer": "Historical formula calculation (2019-2023 baseline). Not for emergency navigation.",
    }


def predict_risk(
    district: str = "Pune",
    road_type: str = "Urban / City Road",
    primary_cause: str = "Over-speeding",
    year: int = 2023,
    month: int = 6,
    accident_count: int = 50,
    fatal_accidents: int = 5,
    deaths: int = 5,
    injuries: int = 30,
    time_period: str = "Day (06:00-18:00)",
) -> Dict[str, Any]:
    """Predict risk classification tier using trained Random Forest model.

    Returns:
        Predicted risk level, confidence score, and top feature influences.
    """
    from backend.app.services.intelligence_service import predict_risk as _predict

    res = _predict(
        year=year,
        month=month,
        district=district,
        road_type=road_type,
        primary_cause=primary_cause,
        accident_count=accident_count,
        fatal_accidents=fatal_accidents,
        deaths=deaths,
        injuries=injuries,
        time_period=time_period,
    )

    if "error" in res and res.get("risk_level") is None:
        # Fallback to formula risk classification if ML model missing
        formula_res = calculate_risk_score(
            accident_count=accident_count,
            fatal_accidents=fatal_accidents,
            deaths=deaths,
            injuries=injuries,
            road_type=road_type,
        )
        return {
            "status": "fallback",
            "risk_score": formula_res["risk_score"],
            "risk_level": formula_res["risk_level"],
            "risk_category": formula_res["risk_category"],
            "predicted_category": formula_res["risk_category"],
            "confidence": 0.70,
            "is_fallback": True,
            "disclaimer": "Fallback formula risk score utilized (ML model unavailable).",
        }

    level = res.get("risk_level") or "MEDIUM"
    return {
        "status": "success",
        "risk_score": res.get("risk_score", 50.0),
        "risk_level": level,
        "risk_category": level,
        "predicted_category": level,
        "confidence": res.get("confidence", 0.85),
        "top_factors": res.get("top_factors", []),
        "disclaimer": "Classification based on observed historical patterns. Does not predict real-time accidents.",
    }
