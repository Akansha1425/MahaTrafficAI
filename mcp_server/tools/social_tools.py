"""MCP Tool implementations for public sentiment and social complaint analytics."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent

NON_CAUSAL_DISCLAIMER = (
    "PUBLIC PERCEPTION NOTICE: Social media complaints and sentiment reflect citizen perception, "
    "reporting frequency, and road condition grievances. They are strictly non-causal signals and "
    "must NOT be treated as empirical causes or statistical predictors of traffic accidents."
)


def analyze_sentiment(
    location: Optional[str] = None,
    texts: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze aggregate public perception and complaint sentiment.

    Args:
        location: Optional filter for a specific Maharashtra city/district.
        texts: Optional explicit batch of text strings to analyze.

    Returns:
        Structured sentiment breakdown with non-causal disclaimer.
    """
    from backend.app.services.intelligence_service import analyze_sentiment as _analyze_sent

    if texts is not None:
        raw_res = _analyze_sent(texts=texts)
        return {
            "status": "success",
            "type": "batch_analysis",
            "results": raw_res.get("results", []),
            "disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    # Aggregate analysis from dataset
    from backend.app.services.social_service import get_social_service
    svc = get_social_service()

    if location:
        posts = svc.get_location_posts(location=location, top_n=100)
        neg_count = sum(1 for p in posts if p.get("sentiment_label") == "NEGATIVE" or p.get("sentiment_score", 0) < 0)
        pos_count = sum(1 for p in posts if p.get("sentiment_label") == "POSITIVE" or p.get("sentiment_score", 0) > 0)
        neu_count = len(posts) - neg_count - pos_count
        return {
            "status": "success",
            "location": location,
            "total_posts_analyzed": len(posts),
            "sentiment_distribution": {
                "NEGATIVE": neg_count,
                "NEUTRAL": max(0, neu_count),
                "POSITIVE": pos_count,
            },
            "sample_posts": posts[:5],
            "disclaimer": NON_CAUSAL_DISCLAIMER,
        }

    sentiment_dist = svc.get_sentiment_distribution()
    stats = svc.get_summary_stats()
    return {
        "status": "success",
        "location": "Maharashtra (All)",
        "total_posts_analyzed": stats.get("total_posts", 1200),
        "sentiment_distribution": sentiment_dist,
        "negative_percentage": stats.get("negative_posts_pct", 75.0),
        "disclaimer": NON_CAUSAL_DISCLAIMER,
    }


def get_social_trends(location: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """Retrieve recurring traffic complaint topics, sentiment trends, and rankings.

    Args:
        location: Optional location filter.
        limit: Max items to return in top rankings.

    Returns:
        Structured social trends report.
    """
    from backend.app.services.intelligence_service import get_social_trends as _get_trends

    trends_data = _get_trends()
    from backend.app.services.social_service import get_social_service
    svc = get_social_service()

    topic_freqs = svc.get_topic_frequencies()
    top_topics = sorted(topic_freqs.items(), key=lambda x: -x[1])[:limit]
    concern_ranking = svc.get_location_concern_ranking(top_n=limit)

    return {
        "status": "success",
        "location_filter": location,
        "total_posts": trends_data.get("total_posts", 1200),
        "top_complaint_topics": [{"topic": t, "count": c} for t, c in top_topics],
        "top_concern_locations": concern_ranking,
        "monthly_trends": trends_data.get("monthly_post_counts", []),
        "disclaimer": NON_CAUSAL_DISCLAIMER,
    }
