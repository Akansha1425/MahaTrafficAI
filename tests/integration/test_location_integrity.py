"""Integration tests for location integrity, record isolation, and dataset bounds."""

import pytest
from mcp_server.tools.accident_tools import get_city_accident_statistics
from backend.app.services.intelligence_service import get_accident_statistics
from agents.workflow.multi_agent import run_agent_workflow


def test_pune_record_count_isolation():
    """Verify Pune statistics return exactly 120 district records, not 2,460 total dataset records."""
    stats = get_city_accident_statistics("Pune")
    assert stats["district_found"] is True
    assert stats["records_count"] == 120
    assert stats["total_accidents"] == 24300
    assert stats["total_deaths"] == 6991
    assert stats["total_injuries"] == 16671
    assert stats["total_fatal_accidents"] == 5984


def test_dataset_wide_aggregation_vs_district():
    """Verify statewide aggregation totals 2,460 records across 34 districts."""
    state_stats = get_accident_statistics(district=None)
    pune_stats = get_accident_statistics(district="Pune")

    assert state_stats["records_count"] == 2460
    assert state_stats["district_count"] == 34
    assert pune_stats["records_count"] == 120
    assert pune_stats["district_count"] == 1
    assert state_stats["records_count"] > pune_stats["records_count"]


def test_unknown_location_handling():
    """Verify nonexistent districts are gracefully identified without hallucinating records."""
    unknown = get_city_accident_statistics("Atlantis")
    assert unknown["district_found"] is False
    assert "No historical records found" in unknown["message"]


def test_agent_workflow_pune_does_not_conflate_totals():
    """Verify multi-agent workflow text clearly reports Pune's 120 historical records."""
    result = run_agent_workflow("Provide historical accident statistics for Pune district.")
    assert result["status"] == "success"
    response = result["final_response"]
    assert "Pune" in response
    assert "120 historical records" in response
    # Verify no false claim of 2,460 records for Pune
    assert "2,460 records for pune" not in response.lower()
    assert "2460 records for pune" not in response.lower()
