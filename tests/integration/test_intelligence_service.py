"""Phase 4 Tests — Unified Intelligence Service Integration."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestIntelligenceService:
    """Tests for backend/app/services/intelligence_service.py"""

    def test_get_accident_statistics(self):
        from backend.app.services.intelligence_service import get_accident_statistics
        stats = get_accident_statistics()
        assert "total_accidents" in stats
        assert stats["total_accidents"] > 0
        assert "district_count" in stats

    def test_get_accident_statistics_with_filter(self):
        from backend.app.services.intelligence_service import get_accident_statistics
        stats = get_accident_statistics(district="Pune")
        assert "total_accidents" in stats
        assert stats["total_accidents"] > 0

    def test_calculate_risk(self):
        from backend.app.services.intelligence_service import calculate_risk
        res = calculate_risk(
            accident_severity=80.0,
            accident_frequency=75.0,
            road_risk_factor=60.0,
            time_risk_factor=50.0,
        )
        assert "risk_score" in res
        assert "risk_level" in res
        assert res["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_analyze_sentiment_corpus(self):
        from backend.app.services.intelligence_service import analyze_sentiment
        res = analyze_sentiment()
        assert res is not None
        assert "sentiment_distribution" in res or "overall_sentiment" in res or "type" in res

    def test_get_social_trends(self):
        from backend.app.services.intelligence_service import get_social_trends
        res = get_social_trends()
        assert res is not None
        assert "total_posts" in res or "monthly_trends" in res

    def test_search_road_safety_documents(self):
        from backend.app.services.intelligence_service import search_road_safety_documents
        res = search_road_safety_documents("accident black spot identification guidelines", top_k=3)
        assert "query" in res
        assert "retrieved_chunks" in res
        assert res["retrieved_chunks"] > 0
        assert len(res["sources"]) > 0
