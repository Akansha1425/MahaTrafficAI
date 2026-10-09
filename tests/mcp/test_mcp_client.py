"""Unit tests for Phase 5 MCP Client, Tool Allowlist, Bounded Retries, and Fallback."""

import pytest
from mcp_server.client import MCPClient, ToolSecurityError
from agents.schemas import ToolRequest, ToolResponse


def test_mcp_client_allowlist_enforcement():
    """Verify MCP client rejects unapproved tools."""
    client = MCPClient()
    resp = client.execute_tool("arbitrary_bash_executor", {"cmd": "ls"})
    assert resp.status == "error"
    assert "not in the approved mcp tool allowlist" in resp.error.lower()


def test_mcp_client_blocked_dangerous_parameters():
    """Verify MCP client blocks dangerous injection parameter patterns."""
    client = MCPClient()
    resp = client.execute_tool("get_city_accident_statistics", {"city_name": "Pune; rm -rf /"})
    assert resp.status == "error"
    assert "dangerous parameter" in resp.error.lower()


def test_mcp_client_executes_canonical_tools():
    """Verify all 7 canonical MCP tools execute successfully."""
    client = MCPClient()

    # 1. City accident statistics
    r1 = client.execute_tool("get_city_accident_statistics", {"city_name": "Pune"})
    assert r1.status == "success"
    assert r1.data["city_name"] == "Pune"
    assert r1.data["district_found"] is True

    # 2. Monthly accident statistics
    r2 = client.execute_tool("get_monthly_accident_statistics", {"district": "Pune"})
    assert r2.status == "success"
    assert "monthly_breakdown" in r2.data

    # 3. Calculate risk score
    r3 = client.execute_tool("calculate_risk_score", {
        "accident_count": 60, "fatal_accidents": 6, "deaths": 6, "injuries": 35
    })
    assert r3.status == "success"
    assert 0 <= r3.data["risk_score"] <= 100
    assert r3.data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    # 4. Predict risk
    r4 = client.execute_tool("predict_risk", {"district": "Pune"})
    assert r4.status in ["success", "fallback"]
    assert "risk_level" in r4.data

    # 5. Analyze sentiment
    r5 = client.execute_tool("analyze_sentiment", {"location": "Pune"})
    assert r5.status == "success"
    assert "sentiment_distribution" in r5.data

    # 6. Get social trends
    r6 = client.execute_tool("get_social_trends", {"location": "Pune"})
    assert r6.status == "success"
    assert "top_complaint_topics" in r6.data

    # 7. Search road safety documents
    r7 = client.execute_tool("search_road_safety_documents", {"query": "MoRTH speed limits", "top_k": 2})
    assert r7.status == "success"
    assert r7.data["retrieved_chunks"] > 0


def test_mcp_client_bounded_retries_and_fallback():
    """Verify MCP client does not crash on simulated failure and applies bounded retry + fallback."""
    # Temporarily register a failing test tool
    failing_calls = 0

    def mock_flaky_tool(params):
        nonlocal failing_calls
        failing_calls += 1
        raise ConnectionResetError("Simulated socket drop")

    client = MCPClient(max_retries=2)
    client.allowlist["mock_failing_tool"] = mock_flaky_tool

    try:
        resp = client.execute_tool("mock_failing_tool", {})
        assert resp.status == "fallback"
        assert resp.retry_count == 2  # exactly 2 retries
        assert failing_calls == 3     # initial + 2 retries = 3 attempts total
        assert "fallback notice" in resp.data["disclaimer"].lower()
    finally:
        client.allowlist.pop("mock_failing_tool", None)
