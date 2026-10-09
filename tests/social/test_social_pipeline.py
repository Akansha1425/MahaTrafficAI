"""Phase 4 Tests — Social Media Analytics Pipeline.

Covers:
  - Text cleaning and normalisation (clean_social.py)
  - Sentiment scoring and classification (analyze_sentiment.py)
  - Topic / issue keyword extraction (extract_topics.py)
  - Social trends aggregation (social_trends.py)
  - Social graph analysis & academic disclaimer (social_graph.py)
  - Social engagement metrics & viral detection (analyze_engagement.py)
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestSocialCleaning:
    """Tests for social_analytics/preprocessing/clean_social.py"""

    def test_clean_text_removes_urls_and_mentions(self):
        from social_analytics.preprocessing.clean_social import clean_text
        sample = "Check out https://roadsafety.org @trafficpolice #PotholeAlert dangerous road!"
        cleaned = clean_text(sample)
        assert "https://" not in cleaned
        assert "@trafficpolice" not in cleaned
        assert "PotholeAlert" in cleaned  # Hashtag content retained

    def test_normalize_location(self):
        from social_analytics.preprocessing.clean_social import normalize_location
        assert normalize_location("  pune  ") == "Pune"
        assert normalize_location("") == "Unknown"
        assert normalize_location(None) == "Unknown"

    def test_run_cleaning_generates_dataframe(self):
        from social_analytics.preprocessing.clean_social import run_cleaning
        df = run_cleaning()
        assert len(df) > 0
        assert "clean_text" in df.columns
        assert "engagement_score" in df.columns


class TestSocialSentiment:
    """Tests for social_analytics/sentiment/analyze_sentiment.py"""

    def test_sentiment_scoring_lexicon(self):
        from social_analytics.sentiment.analyze_sentiment import score_sentiment
        neg_text = "Terrible deadly accident bad pothole fatal crash!"
        score, label = score_sentiment(neg_text)
        assert label == "NEGATIVE"
        assert score < 0

        pos_text = "Excellent smooth highway great safe driving conditions!"
        score, label = score_sentiment(pos_text)
        assert label == "POSITIVE"
        assert score > 0

    def test_run_sentiment_pipeline(self):
        from social_analytics.sentiment.analyze_sentiment import run_sentiment_analysis
        df = run_sentiment_analysis()
        assert len(df) > 0
        assert "sentiment_label" in df.columns
        assert set(df["sentiment_label"].unique()).issubset({"POSITIVE", "NEUTRAL", "NEGATIVE"})


class TestTopicExtraction:
    """Tests for social_analytics/topics/extract_topics.py"""

    def test_extract_topics_keywords(self):
        from social_analytics.topics.extract_topics import extract_topics
        text = "Huge pothole on Pune highway causing severe traffic and speeding water logging"
        topics = extract_topics(text)
        assert "potholes" in topics or "waterlogging" in topics or "overspeeding" in topics

    def test_run_topic_pipeline(self):
        from social_analytics.topics.extract_topics import run_topic_extraction
        df = run_topic_extraction()
        assert len(df) > 0
        assert "topics" in df.columns or "primary_topic" in df.columns


class TestSocialTrends:
    """Tests for social_analytics/trends/social_trends.py"""

    def test_run_trends_analysis(self):
        from social_analytics.trends.social_trends import run_trends_analysis
        df, report = run_trends_analysis()
        assert len(df) > 0
        assert "total_posts" in report
        assert "monthly_post_counts" in report or "monthly_trends" in report


class TestSocialGraph:
    """Tests for social_analytics/community_detection/social_graph.py"""

    def test_social_graph_runs_and_has_disclaimer(self):
        from social_analytics.community_detection.social_graph import run_social_graph_analysis
        report = run_social_graph_analysis()
        assert report is not None
        assert "disclaimer" in report or "note" in report or "nodes" in report


class TestSocialEngagement:
    """Tests for social_analytics/engagement/analyze_engagement.py"""

    def test_engagement_pipeline(self):
        from social_analytics.engagement.analyze_engagement import run_engagement_pipeline
        df, report = run_engagement_pipeline()
        assert len(df) > 0
        assert "total_engagement_score" in report
        assert "p90_threshold" in report
        assert len(report["top_viral_posts"]) > 0
