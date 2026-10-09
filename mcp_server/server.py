"""MCP (Model Context Protocol) Server — MahaTraffic AI.

Implements the MCP server as a standalone FastAPI service that exposes
structured tool calls for external AI agents to consume.

Tools exposed:
  - get_accident_statistics  : District/yearly accident data
  - calculate_risk_score     : Formula-based risk computation
  - predict_risk_category    : ML model inference
  - get_social_sentiment     : Social media analytics
  - search_road_safety_docs  : RAG knowledge retrieval
  - get_model_metrics        : ML evaluation metrics

Runs on port 8001 (separate from main API on 8000).
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mcp.server")

app = FastAPI(
    title="MahaTraffic AI — MCP Tool Server",
    description="Model Context Protocol tool server for road-safety agent access.",
    version="1.0.0",
    docs_url="/mcp/docs",
    openapi_url="/mcp/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Tool: Accident Statistics ─────────────────────────────────────────────────

class AccidentStatsRequest(BaseModel):
    district: Optional[str] = Field(None, description="Maharashtra district name (optional)")
    year: Optional[int] = Field(None, description="Filter by year (2019-2023)")


@app.post("/mcp/tools/get_accident_statistics", tags=["MCP Tools"])
async def tool_get_accident_statistics(req: AccidentStatsRequest) -> Dict[str, Any]:
    """MCP Tool: Retrieve historical accident statistics.

    Returns district-level or year-filtered accident data from 2019-2023.
    """
    try:
        from backend.app.services.risk_service import get_risk_service
        svc = get_risk_service()

        districts = svc.get_district_risk_summary(district=req.district)
        yearly = svc.get_yearly_trend()

        if req.year:
            yearly = [y for y in yearly if int(y.get("year", 0)) == req.year]

        return {
            "tool": "get_accident_statistics",
            "status": "success",
            "district_filter": req.district,
            "year_filter": req.year,
            "districts": districts[:10] if not req.district else districts,
            "yearly_trend": yearly,
            "data_period": "2019-2023",
            "source": "MahaTraffic AI — Historical Accident Dataset",
            "disclaimer": "ANALYTICAL DATA ONLY: Based on historical records. Not for emergency use.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Tool: Risk Score ──────────────────────────────────────────────────────────

class RiskScoreRequest(BaseModel):
    accident_count: int = Field(50, ge=0)
    fatal_accidents: int = Field(5, ge=0)
    deaths: int = Field(5, ge=0)
    injuries: int = Field(30, ge=0)
    road_type: str = Field("National Highway", description="Road classification")
    time_period: str = Field("night", description="Time period: day/night/peak hour")


@app.post("/mcp/tools/calculate_risk_score", tags=["MCP Tools"])
async def tool_calculate_risk_score(req: RiskScoreRequest) -> Dict[str, Any]:
    """MCP Tool: Calculate composite risk score using the MahaTraffic formula."""
    try:
        from backend.app.services.risk_service import get_risk_service
        result = get_risk_service().calculate_risk_score(
            accident_count=req.accident_count,
            fatal_accidents=req.fatal_accidents,
            deaths=req.deaths,
            injuries=req.injuries,
            road_type=req.road_type,
            time_period=req.time_period,
        )
        return {
            "tool": "calculate_risk_score",
            "status": "success",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Tool: ML Predict ─────────────────────────────────────────────────────────

class MLPredictRequest(BaseModel):
    district: str = Field("Pune")
    road_type: str = Field("Urban / City Road")
    primary_cause: str = Field("Over-speeding")
    year: int = Field(2023)
    month: int = Field(6)
    accident_count: int = Field(50)
    fatal_accidents: int = Field(5)
    deaths: int = Field(5)
    injuries: int = Field(30)
    is_monsoon: int = Field(0)
    is_night: int = Field(0)
    is_highway: int = Field(0)
    fatality_ratio: float = Field(0.10)
    injury_ratio: float = Field(0.60)
    severity_index: float = Field(0.50)


@app.post("/mcp/tools/predict_risk_category", tags=["MCP Tools"])
async def tool_predict_risk_category(req: MLPredictRequest) -> Dict[str, Any]:
    """MCP Tool: Predict risk category using the trained Random Forest model."""
    try:
        from backend.app.services.risk_service import get_risk_service
        result = get_risk_service().predict_risk_category(req.model_dump())
        return {
            "tool": "predict_risk_category",
            "status": "success",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Tool: Social Sentiment ────────────────────────────────────────────────────

class SocialSentimentRequest(BaseModel):
    location: Optional[str] = Field(None, description="Maharashtra city/district")
    top_n: int = Field(5, ge=1, le=20)


@app.post("/mcp/tools/get_social_sentiment", tags=["MCP Tools"])
async def tool_get_social_sentiment(req: SocialSentimentRequest) -> Dict[str, Any]:
    """MCP Tool: Get social media sentiment analysis for Maharashtra road safety."""
    try:
        from backend.app.services.social_service import get_social_service
        svc = get_social_service()

        result = {
            "tool": "get_social_sentiment",
            "status": "success",
            "sentiment_distribution": svc.get_sentiment_distribution(),
            "summary": svc.get_summary_stats(),
            "topic_frequencies": svc.get_topic_frequencies(),
            "concern_ranking": svc.get_location_concern_ranking(top_n=req.top_n),
            "yearly_trend": svc.get_yearly_sentiment_trend(),
        }

        if req.location:
            result["location_posts"] = svc.get_location_posts(req.location, top_n=5)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Tool: RAG Knowledge Search ───────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Road safety question to search in knowledge base")
    top_k: int = Field(5, ge=1, le=10)


@app.post("/mcp/tools/search_road_safety_docs", tags=["MCP Tools"])
async def tool_search_road_safety_docs(req: RAGQueryRequest) -> Dict[str, Any]:
    """MCP Tool: Search the road safety knowledge base (RAG retrieval)."""
    try:
        from backend.app.services.rag_service import get_rag_service
        svc = get_rag_service()
        result = svc.query(req.query, top_k=req.top_k)
        return {
            "tool": "search_road_safety_docs",
            "status": "success",
            **result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mcp/tools/search_road_safety_documents", tags=["MCP Tools"])
async def tool_search_road_safety_documents(req: RAGQueryRequest) -> Dict[str, Any]:
    """MCP Tool: Search road safety documents (canonical route)."""
    from mcp_server.tools.rag_tools import search_road_safety_documents
    return search_road_safety_documents(query=req.query, top_k=req.top_k)


# ─── Canonical Endpoints for All Phase 5 Tools ────────────────────────────────

class CityStatsRequest(BaseModel):
    city_name: str = Field(..., description="Target district/city")
    year: Optional[int] = Field(None, description="Calendar year")

@app.post("/mcp/tools/get_city_accident_statistics", tags=["MCP Tools"])
async def tool_get_city_accident_statistics(req: CityStatsRequest) -> Dict[str, Any]:
    """MCP Tool: Retrieve city/district historical accident stats."""
    from mcp_server.tools.accident_tools import get_city_accident_statistics
    return get_city_accident_statistics(city_name=req.city_name, year=req.year)


class MonthlyStatsRequest(BaseModel):
    year: Optional[int] = Field(None)
    district: Optional[str] = Field(None)

@app.post("/mcp/tools/get_monthly_accident_statistics", tags=["MCP Tools"])
async def tool_get_monthly_accident_statistics(req: MonthlyStatsRequest) -> Dict[str, Any]:
    """MCP Tool: Retrieve monthly accident seasonality patterns."""
    from mcp_server.tools.accident_tools import get_monthly_accident_statistics
    return get_monthly_accident_statistics(year=req.year, district=req.district)


@app.post("/mcp/tools/predict_risk", tags=["MCP Tools"])
async def tool_predict_risk(req: MLPredictRequest) -> Dict[str, Any]:
    """MCP Tool: Predict risk category using ML model (canonical route)."""
    from mcp_server.tools.risk_tools import predict_risk
    return predict_risk(
        district=req.district,
        road_type=req.road_type,
        primary_cause=req.primary_cause,
        year=req.year,
        month=req.month,
        accident_count=req.accident_count,
        fatal_accidents=req.fatal_accidents,
        deaths=req.deaths,
        injuries=req.injuries,
    )


class SentimentAnalysisRequest(BaseModel):
    location: Optional[str] = Field(None)
    texts: Optional[list[str]] = Field(None)

@app.post("/mcp/tools/analyze_sentiment", tags=["MCP Tools"])
async def tool_analyze_sentiment(req: SentimentAnalysisRequest) -> Dict[str, Any]:
    """MCP Tool: Analyze public sentiment (canonical route)."""
    from mcp_server.tools.social_tools import analyze_sentiment
    return analyze_sentiment(location=req.location, texts=req.texts)


class SocialTrendsRequest(BaseModel):
    location: Optional[str] = Field(None)
    limit: int = Field(10, ge=1, le=50)

@app.post("/mcp/tools/get_social_trends", tags=["MCP Tools"])
async def tool_get_social_trends(req: SocialTrendsRequest) -> Dict[str, Any]:
    """MCP Tool: Retrieve social trends and complaints (canonical route)."""
    from mcp_server.tools.social_tools import get_social_trends
    return get_social_trends(location=req.location, limit=req.limit)


# ─── Tool: Model Metrics ──────────────────────────────────────────────────────

@app.get("/mcp/tools/get_model_metrics", tags=["MCP Tools"])
async def tool_get_model_metrics() -> Dict[str, Any]:
    """MCP Tool: Get the ML model evaluation metrics."""
    try:
        from backend.app.services.risk_service import get_risk_service
        svc = get_risk_service()
        return {
            "tool": "get_model_metrics",
            "status": "success",
            "metrics": svc.get_model_metrics(),
            "feature_importances": dict(list(svc.get_feature_importances().items())[:10]),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Tool Registry ────────────────────────────────────────────────────────────

@app.get("/mcp/tools", tags=["MCP"])
async def list_tools() -> Dict[str, Any]:
    """List all available MCP tools."""
    return {
        "server": "MahaTraffic AI MCP Tool Server",
        "version": "1.0.0",
        "tools": [
            {
                "name": "get_accident_statistics",
                "endpoint": "POST /mcp/tools/get_accident_statistics",
                "description": "Historical district-wise accident statistics (2019-2023)",
            },
            {
                "name": "calculate_risk_score",
                "endpoint": "POST /mcp/tools/calculate_risk_score",
                "description": "Formula-based risk score computation",
            },
            {
                "name": "predict_risk_category",
                "endpoint": "POST /mcp/tools/predict_risk_category",
                "description": "ML Random Forest risk category prediction",
            },
            {
                "name": "get_social_sentiment",
                "endpoint": "POST /mcp/tools/get_social_sentiment",
                "description": "Social media sentiment analysis results",
            },
            {
                "name": "search_road_safety_docs",
                "endpoint": "POST /mcp/tools/search_road_safety_docs",
                "description": "RAG knowledge retrieval from road safety documents",
            },
            {
                "name": "get_model_metrics",
                "endpoint": "GET /mcp/tools/get_model_metrics",
                "description": "ML model evaluation metrics and feature importances",
            },
        ],
    }


@app.get("/mcp/health", tags=["MCP"])
async def mcp_health():
    """MCP server health check."""
    return {"status": "ok", "server": "MahaTraffic AI MCP Server", "port": 8001}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mcp_server.server:app", host="127.0.0.1", port=8001, reload=False)
