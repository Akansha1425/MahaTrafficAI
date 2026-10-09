"""Unit tests for MahaTraffic AI multi-agent workflow, schemas, reviewer, and guardrails."""

import pytest
from agents.workflow.multi_agent import run_agent_workflow
from agents.guardrails.guardrails import validate_input, review_response_content
from agents.schemas import (
    PlannerPlan,
    ToolRequest,
    ToolResponse,
    ReviewResult,
    validate_risk_score,
    validate_risk_level,
)


def test_agent_workflow_accident_query():
    """Verify agent workflow processes an accident risk query end-to-end."""
    result = run_agent_workflow("Which districts in Maharashtra have high accident risk?")
    assert "final_response" in result
    assert len(result["final_response"]) > 50
    assert "agents_used" in result
    assert len(result["agents_used"]) > 0


def test_agent_workflow_sentiment_query():
    """Verify agent workflow handles social media sentiment inquiries."""
    result = run_agent_workflow("What are public complaints about potholes in Pune?")
    assert "final_response" in result
    assert len(result["final_response"]) > 50


def test_guardrails_valid_traffic_queries():
    """Verify on-topic traffic safety queries pass guardrails."""
    valid_queries = [
        "What is the fatality rate on National Highways in Maharashtra?",
        "Show accident statistics for Mumbai and Pune",
        "MoRTH black spot guidelines and mitigation measures"
    ]
    for q in valid_queries:
        res = validate_input(q)
        assert res["is_valid"], f"Expected '{q}' to be valid, but got: {res.get('reason')}"


def test_guardrails_reject_off_topic():
    """Verify off-topic queries are blocked by guardrails."""
    off_topic = [
        "How do I invest in cryptocurrency?",
        "Write a poem about sunflowers",
        "Give me recipes for Italian pizza"
    ]
    for q in off_topic:
        res = validate_input(q)
        assert not res["is_valid"], f"Expected '{q}' to be blocked"


def test_guardrail_allows_legitimate_security_analysis():
    """Verify guardrails do NOT block legitimate academic security analysis queries."""
    query = "Analyze how prompt injection attacks like 'ignore previous instructions' could impact traffic safety agents."
    res = validate_input(query)
    assert res["is_valid"] is True, f"Legitimate security analysis was incorrectly blocked: {res.get('reason')}"


def test_schema_validation_risk_bounds():
    """Verify schema validators reject invalid risk levels and out-of-bounds scores."""
    # Valid scores pass
    assert validate_risk_score(45.5) == 45.5
    assert validate_risk_level("HIGH") == "HIGH"

    # Out of bounds score fails
    with pytest.raises(ValueError):
        validate_risk_score(150.0)

    with pytest.raises(ValueError):
        validate_risk_score(-5.0)

    # Invalid risk level fails
    with pytest.raises(ValueError):
        validate_risk_level("EXTREME_CRITICAL")


def test_reviewer_detects_unsupported_causal_claims():
    """Verify reviewer flags assertions that social sentiment causes accidents."""
    hallucinated_text = "Twitter complaints caused the fatal crash on the highway yesterday."
    res = review_response_content(hallucinated_text)
    assert res.is_approved is False
    assert res.non_causal_social_verified is False
    assert any("causal claim" in w for w in res.warnings)


def test_end_to_end_acceptance_pune_query():
    """Section 14 Acceptance Test: Full multi-agent execution on Pune risk and public complaints."""
    query = "Summarize the historical road-accident risk in Pune and explain which road-safety concerns appear in the available public-post dataset."
    result = run_agent_workflow(query)

    assert result["status"] == "success"
    assert "Pune" in result["final_response"]
    assert "HISTORICAL ACCIDENT ANALYTICS" in result["final_response"]
    assert "PUBLIC CITIZEN PERCEPTION" in result["final_response"]
    assert "OFFICIAL ROAD SAFETY KNOWLEDGE" in result["final_response"] or "RISK ASSESSMENT" in result["final_response"]

    # Verify review result metadata
    review = result["review_result"]
    assert review["is_approved"] is True
    assert review["non_causal_social_verified"] is True
    assert "ANALYTICAL NOTICE" in result["final_response"]

    # Verify tools were invoked
    tools_called = [t["tool_name"] for t in result["tools_executed"]]
    assert "get_city_accident_statistics" in tools_called
    assert "analyze_sentiment" in tools_called
