"""Unit tests for the Model Context Protocol (MCP) tool endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from mcp_server.server import app


@pytest.mark.asyncio
async def test_mcp_accident_statistics():
    """Verify MCP tool for retrieving accident statistics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/tools/get_accident_statistics", json={"district": "Pune"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "get_accident_statistics"
        assert data["status"] == "success"
        assert "districts" in data


@pytest.mark.asyncio
async def test_mcp_calculate_risk_score():
    """Verify MCP formula-based risk calculation."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "accident_count": 50,
            "fatal_accidents": 5,
            "deaths": 5,
            "injuries": 20,
            "road_type": "National Highway",
            "time_period": "night"
        }
        resp = await client.post("/mcp/tools/calculate_risk_score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert "risk_score" in data
        assert "risk_category" in data


@pytest.mark.asyncio
async def test_mcp_social_sentiment():
    """Verify MCP tool for social media sentiment retrieval."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/mcp/tools/get_social_sentiment", json={"location": "Pune"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["tool"] == "get_social_sentiment"
        assert data["status"] == "success"
