"""Pydantic response schemas for API endpoints."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""

    status: str = Field(..., description="Operational status of the service", examples=["ok"])
    project: str = Field(..., description="Project name", examples=["MahaTraffic AI"])
    version: str = Field(default="0.1.0", description="Semantic service version", examples=["0.1.0"])
    environment: str = Field(default="development", description="Current execution environment")


class DashboardSummaryResponse(BaseModel):
    """High-level metrics summary schema for executive dashboard."""

    total_historical_accidents: Optional[int] = None
    analyzed_locations_count: Optional[int] = None
    high_risk_clusters_count: Optional[int] = None
    latest_data_year: Optional[int] = None
    status: str = "scaffold"


class RiskPredictionResponse(BaseModel):
    """Historical risk estimation response schema."""

    location: str
    risk_score: Optional[float] = None
    risk_category: Optional[str] = None
    contributing_factors: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None


class AgentChatResponse(BaseModel):
    """Structured response schema from multi-agent pipeline."""

    run_id: str
    question: str
    intent: Optional[str] = None
    answer: str
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    guardrail_status: str = "passed"
