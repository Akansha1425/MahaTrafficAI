"""Integration tests for MahaTraffic AI FastAPI endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_dashboard_overview():
    """Verify GET /api/v1/dashboard/overview returns aggregated statistics."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/dashboard/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "yearly_trends" in data
        assert "top_risk_districts" in data


@pytest.mark.asyncio
async def test_analytics_yearly():
    """Verify GET /api/v1/analytics/yearly returns yearly historical stats."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/analytics/yearly")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 5


@pytest.mark.asyncio
async def test_risk_districts():
    """Verify GET /api/v1/analytics/districts returns district rankings."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/analytics/districts")
        assert resp.status_code == 200
        districts = resp.json()
        assert isinstance(districts, list)
        assert len(districts) > 0


@pytest.mark.asyncio
async def test_social_sentiment():
    """Verify GET /api/v1/social/sentiment returns sentiment analysis summary."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/social/sentiment")
        assert resp.status_code == 200
        data = resp.json()
        assert "NEGATIVE" in data or "POSITIVE" in data or "NEUTRAL" in data


@pytest.mark.asyncio
async def test_agent_valid_query():
    """Verify POST /api/v1/agent/query accepts traffic questions."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/agent/query", json={"query": "Which district has the highest accident risk in Maharashtra?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "final_response" in data
        assert len(data["final_response"]) > 0


@pytest.mark.asyncio
async def test_agent_offtopic_guardrail_rejection():
    """Verify POST /api/v1/agent/query blocks off-topic queries with HTTP 422."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/agent/query", json={"query": "Can you recommend a recipe for chocolate cake?"})
        assert resp.status_code == 422
