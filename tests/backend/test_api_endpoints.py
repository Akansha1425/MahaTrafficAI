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
async def test_social_posts_by_location():
    """Verify GET /api/v1/social/posts/by-location returns valid post schema with sentiment_label."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/social/posts/by-location?location=Pune&top_n=5")
        assert resp.status_code == 200
        posts = resp.json()
        assert isinstance(posts, list)
        if len(posts) > 0:
            first = posts[0]
            assert "post_id" in first
            assert "text" in first
            assert "location" in first
            assert "concern_score" in first
            assert "sentiment_label" in first
            assert first["sentiment_label"] in ["NEGATIVE", "NEUTRAL", "POSITIVE"]


@pytest.mark.asyncio
async def test_risk_score_calculation():
    """Verify POST /api/v1/risk/score computes formula-based composite score."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "accident_count": 100,
            "fatal_accidents": 10,
            "deaths": 10,
            "injuries": 60,
            "road_type": "Urban / City Road",
            "time_period": "night"
        }
        resp = await client.post("/api/v1/risk/score", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "risk_score" in data
        assert "risk_category" in data
        assert "components" in data
        assert 0 <= data["risk_score"] <= 100
        assert data["risk_category"] in ["HIGH", "MEDIUM", "LOW"]


@pytest.mark.asyncio
async def test_risk_ml_predict():
    """Verify POST /api/v1/risk/predict returns model-estimated class probability."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "district": "Pune",
            "road_type": "Urban / City Road",
            "primary_cause": "Over-speeding",
            "year": 2023,
            "month": 6,
            "accident_count": 50,
            "fatal_accidents": 5,
            "deaths": 5,
            "injuries": 30,
            "is_monsoon": 0,
            "is_night": 1,
            "is_highway": 0,
            "fatality_ratio": 0.1,
            "injury_ratio": 0.6,
            "severity_index": 0.5
        }
        resp = await client.post("/api/v1/risk/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_category" in data
        assert "confidence" in data
        assert "class_probabilities" in data
        assert 0.0 <= data["confidence"] <= 1.0


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
