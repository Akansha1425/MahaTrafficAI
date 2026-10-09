"""Social Media Analytics Service — MahaTraffic AI.

Provides programmatic access to sentiment analysis results,
concern rankings, topic distributions, and temporal trends.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import pandas as pd

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
SOCIAL_FEATURES_PATH = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "social_features.parquet"
SENTIMENT_REPORT_PATH = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "sentiment_report.json"

_service_instance: Optional["SocialService"] = None


class SocialService:
    """Service providing social media analytics data for routes and agents."""

    def __init__(self):
        self._df: Optional[pd.DataFrame] = None
        self._report: Optional[dict] = None
        self._load()

    def _load(self) -> None:
        """Load social features parquet and sentiment report."""
        if SOCIAL_FEATURES_PATH.exists():
            self._df = pd.read_parquet(SOCIAL_FEATURES_PATH, engine="pyarrow")
            logger.info("Social features loaded: %d records.", len(self._df))
        else:
            logger.warning("Social features parquet not found: %s", SOCIAL_FEATURES_PATH)

        if SENTIMENT_REPORT_PATH.exists():
            with open(SENTIMENT_REPORT_PATH, "r", encoding="utf-8") as f:
                self._report = json.load(f)
            logger.info("Sentiment report loaded.")
        else:
            logger.warning("Sentiment report not found: %s", SENTIMENT_REPORT_PATH)

    def get_sentiment_distribution(self) -> Dict[str, int]:
        """Return overall sentiment distribution counts."""
        if self._report:
            return self._report.get("sentiment_distribution", {})
        if self._df is not None and "sentiment_label" in self._df.columns:
            return self._df["sentiment_label"].value_counts().to_dict()
        return {}

    def get_location_concern_ranking(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return top-N locations by average public concern score."""
        if self._report:
            return self._report.get("location_concern_ranking", [])[:top_n]
        if self._df is None:
            return []
        agg = (
            self._df.groupby("location")["concern_score"]
            .agg(["mean", "count", "max"])
            .rename(columns={"mean": "avg_concern", "count": "post_count", "max": "peak_concern"})
            .round(2)
            .sort_values("avg_concern", ascending=False)
            .head(top_n)
            .reset_index()
        )
        return agg.to_dict(orient="records")

    def get_topic_frequencies(self) -> Dict[str, int]:
        """Return topic label frequencies."""
        if self._report:
            return self._report.get("topic_frequency", {})
        if self._df is None or "topics" not in self._df.columns:
            return {}
        all_topics = []
        for t in self._df["topics"]:
            all_topics.extend([x.strip() for x in str(t).split(",")])
        return pd.Series(all_topics).value_counts().to_dict()

    def get_yearly_sentiment_trend(self) -> List[Dict[str, Any]]:
        """Return average sentiment score per year."""
        if self._report:
            return self._report.get("yearly_sentiment_trend", [])
        if self._df is None or "year" not in self._df.columns:
            return []
        return (
            self._df.groupby("year")["sentiment_score"]
            .mean()
            .round(4)
            .reset_index()
            .rename(columns={"sentiment_score": "avg_sentiment"})
            .to_dict(orient="records")
        )

    def get_most_concerning_posts(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Return the most concerning posts by concern score."""
        if self._report:
            return self._report.get("most_concerning_posts", [])[:top_n]
        if self._df is None:
            return []
        cols = [c for c in ["post_id", "text", "location", "concern_score", "sentiment_label"] if c in self._df.columns]
        return self._df.nlargest(top_n, "concern_score")[cols].to_dict(orient="records")

    def get_location_posts(self, location: str, top_n: int = 20) -> List[Dict[str, Any]]:
        """Get posts filtered by location."""
        if self._df is None:
            return []
        mask = self._df["location"].str.lower() == location.lower()
        subset = self._df[mask].nlargest(top_n, "concern_score")
        cols = [c for c in ["post_id", "text", "location", "concern_score", "sentiment_label", "topics"] if c in subset.columns]
        return subset[cols].to_dict(orient="records")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Return high-level summary statistics."""
        if self._df is None:
            return {"error": "Social data not available."}
        return {
            "total_posts": int(len(self._df)),
            "unique_locations": int(self._df["location"].nunique()) if "location" in self._df.columns else 0,
            "avg_concern_score": round(float(self._df["concern_score"].mean()), 2) if "concern_score" in self._df.columns else 0,
            "negative_posts_pct": round(
                (self._df["sentiment_label"] == "NEGATIVE").sum() / len(self._df) * 100, 1
            ) if "sentiment_label" in self._df.columns else 0,
        }


def get_social_service() -> SocialService:
    """Get singleton SocialService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SocialService()
    return _service_instance
