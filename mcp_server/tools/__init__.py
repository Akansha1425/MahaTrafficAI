"""MCP tools package exporting allowed tool registry."""

from mcp_server.tools.accident_tools import (
    get_city_accident_statistics,
    get_monthly_accident_statistics,
    get_yearly_accident_statistics,
)
from mcp_server.tools.risk_tools import (
    calculate_risk_score,
    predict_risk,
)
from mcp_server.tools.social_tools import (
    analyze_sentiment,
    get_social_trends,
)
from mcp_server.tools.rag_tools import (
    search_road_safety_documents,
)

ALLOWED_TOOLS_REGISTRY = {
    "get_city_accident_statistics": get_city_accident_statistics,
    "get_monthly_accident_statistics": get_monthly_accident_statistics,
    "get_yearly_accident_statistics": get_yearly_accident_statistics,
    "calculate_risk_score": calculate_risk_score,
    "predict_risk": predict_risk,
    "analyze_sentiment": analyze_sentiment,
    "get_social_trends": get_social_trends,
    "search_road_safety_documents": search_road_safety_documents,
}

__all__ = [
    "ALLOWED_TOOLS_REGISTRY",
    "get_city_accident_statistics",
    "get_monthly_accident_statistics",
    "get_yearly_accident_statistics",
    "calculate_risk_score",
    "predict_risk",
    "analyze_sentiment",
    "get_social_trends",
    "search_road_safety_documents",
]
