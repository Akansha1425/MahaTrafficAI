"""Pydantic schemas for MCP tool definitions, inputs, and outputs."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MCPToolDefinition(BaseModel):
    """Metadata schema defining an allowed MCP tool."""

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Semantic purpose of the tool")
    input_schema: Dict[str, Any] = Field(..., description="JSON Schema of parameters")
    output_schema: Dict[str, Any] = Field(..., description="JSON Schema of returned data")
    timeout_seconds: int = Field(default=30, description="Maximum execution timeout")


# Input / Output schemas for future tools
class CityStatisticsInput(BaseModel):
    city_name: str = Field(..., description="Target Maharashtra district/city name")
    year: Optional[int] = Field(None, description="Optional filter for specific year")


class CityStatisticsOutput(BaseModel):
    city_name: str
    year: Optional[int] = None
    total_accidents: Optional[int] = None
    fatalities: Optional[int] = None
    status: str = "scaffold"


class RiskCalculationInput(BaseModel):
    location: str
    fatal_accidents: int
    serious_accidents: int
    minor_accidents: int
    total_accidents: int


class RiskCalculationOutput(BaseModel):
    location: str
    risk_score: float
    risk_tier: str


class SocialTrendsInput(BaseModel):
    location: str
    limit: int = 10


class SocialTrendsOutput(BaseModel):
    location: str
    top_complaint_topics: List[str] = Field(default_factory=list)
    sentiment_distribution: Dict[str, float] = Field(default_factory=dict)


class DocumentSearchInput(BaseModel):
    query: str
    top_k: int = 3


class DocumentSearchOutput(BaseModel):
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
