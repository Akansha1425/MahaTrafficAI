"""Test suite for FastAPI health check and root endpoints."""

import pytest
from httpx import AsyncClient, ASGITransport
from backend.app.main import app


@pytest.mark.asyncio
async def test_health_check_endpoint():
    """Verify GET /health returns 200 OK and expected project identity."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["project"] == "MahaTraffic AI"
        assert "version" in data
        assert "environment" in data


@pytest.mark.asyncio
async def test_openapi_docs_endpoint():
    """Verify OpenAPI documentation endpoint is reachable."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "MahaTraffic AI"
