"""Pydantic request schemas for API endpoints."""

from typing import Dict, Optional, Any
from pydantic import BaseModel, Field


class AgentChatRequest(BaseModel):
    """Request payload for multi-agent investigation query."""

    question: str = Field(..., description="Natural language traffic or risk query")
    location: Optional[str] = Field(None, description="Optional target location/city filter")
    session_id: Optional[str] = Field(None, description="Optional session tracker for conversation")


class RiskPredictionRequest(BaseModel):
    """Request payload for historical risk score computation/prediction."""

    location: str = Field(..., description="District or urban center name in Maharashtra")
    year: Optional[int] = Field(None, description="Target historical year")
    month: Optional[int] = Field(None, ge=1, le=12, description="Target month (1-12)")
    road_type: Optional[str] = Field(None, description="Type of road (e.g. National Highway, State Highway)")
    weather_condition: Optional[str] = Field(None, description="Atmospheric or surface condition")
