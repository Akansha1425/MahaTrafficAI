"""Dashboard route — aggregated overview endpoint for the frontend."""

from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/overview", summary="Get dashboard overview data")
async def get_dashboard_overview() -> Dict[str, Any]:
    """Aggregated overview combining accident analytics, ML metrics, and social sentiment."""
    try:
        from backend.app.services.risk_service import get_risk_service
        from backend.app.services.social_service import get_social_service

        risk_svc = get_risk_service()
        social_svc = get_social_service()

        yearly = risk_svc.get_yearly_trend()
        districts = risk_svc.get_district_risk_summary()
        model_metrics = risk_svc.get_model_metrics()
        feature_importances = risk_svc.get_feature_importances()

        social_summary = social_svc.get_summary_stats()
        sentiment_dist = social_svc.get_sentiment_distribution()
        concern_ranking = social_svc.get_location_concern_ranking(top_n=5)
        topics = social_svc.get_topic_frequencies()
        sentiment_trend = social_svc.get_yearly_sentiment_trend()

        # Summary statistics
        total_accidents = sum(y.get("total_accidents", 0) for y in yearly)
        total_deaths = sum(y.get("total_deaths", 0) for y in yearly)
        total_injuries = sum(y.get("total_injuries", 0) for y in yearly)

        return {
            "summary": {
                "total_accidents_2019_2023": int(total_accidents),
                "total_deaths_2019_2023": int(total_deaths),
                "total_injuries_2019_2023": int(total_injuries),
                "districts_analyzed": len(districts),
                "ml_accuracy": model_metrics.get("accuracy"),
                "ml_f1_macro": model_metrics.get("f1_macro"),
                "social_posts_analyzed": social_summary.get("total_posts", 0),
                "negative_sentiment_pct": social_summary.get("negative_posts_pct", 0),
            },
            "yearly_trends": yearly,
            "top_risk_districts": districts[:10],
            "ml_model": {
                "metrics": model_metrics,
                "top_features": dict(list(feature_importances.items())[:8]),
            },
            "social_analytics": {
                "summary": social_summary,
                "sentiment_distribution": sentiment_dist,
                "concern_ranking": concern_ranking,
                "topic_frequencies": topics,
                "sentiment_trend": sentiment_trend,
            },
        }

    except Exception as e:
        logger.error("Dashboard overview error: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
