"""Social Trends Analysis — Phase 4, MahaTraffic AI.

Generates trend analytics from social posts:
  - Monthly post counts
  - Monthly negative sentiment
  - Top complaint topics
  - Location-wise complaint frequency
  - Engagement trends

Input:  data/processed/social_sentiment.parquet
        (fallback: social_clean.parquet)
Output: data/processed/social_trends.parquet
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.trends")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SENTIMENT_PARQUET = BASE_DIR / "data" / "processed" / "social_sentiment.parquet"
TOPICS_PARQUET = BASE_DIR / "data" / "processed" / "social_topics.parquet"
CLEAN_PARQUET = BASE_DIR / "data" / "processed" / "social_clean.parquet"
FALLBACK_PARQUET = (
    BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
)
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_trends.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "social_trends_report.json"


def _load_data() -> pd.DataFrame:
    """Load best available social data with sentiment if possible."""
    if SENTIMENT_PARQUET.exists():
        logger.info("Loading sentiment Parquet.")
        return pd.read_parquet(SENTIMENT_PARQUET, engine="pyarrow")
    if CLEAN_PARQUET.exists():
        logger.info("Loading clean Parquet.")
        return pd.read_parquet(CLEAN_PARQUET, engine="pyarrow")
    if FALLBACK_PARQUET.exists():
        logger.info("Loading fallback Parquet.")
        return pd.read_parquet(FALLBACK_PARQUET, engine="pyarrow")
    raise FileNotFoundError("No social Parquet found. Run sentiment pipeline first.")


def _ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure year/month exist."""
    if "year" not in df.columns or "month" not in df.columns:
        ts_col = next((c for c in ["timestamp", "post_date", "date"] if c in df.columns), None)
        if ts_col:
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df["year"] = ts.dt.year.fillna(0).astype(int)
            df["month"] = ts.dt.month.fillna(0).astype(int)
        else:
            df["year"] = 0
            df["month"] = 0
    return df


def _ensure_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add dummy sentiment if not present."""
    if "sentiment_label" not in df.columns:
        df["sentiment_label"] = "NEUTRAL"
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = 0.0
    return df


def _monthly_post_counts(df: pd.DataFrame) -> list[dict]:
    """Monthly post count trend."""
    df_valid = df[(df["year"] > 0) & (df["month"] > 0)]
    if df_valid.empty:
        return []
    trend = (
        df_valid.groupby(["year", "month"])
        .size()
        .reset_index(name="post_count")
        .sort_values(["year", "month"])
    )
    trend["period"] = trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
    return trend.to_dict(orient="records")


def _monthly_negative_sentiment(df: pd.DataFrame) -> list[dict]:
    """Monthly count and percentage of NEGATIVE posts."""
    df_valid = df[(df["year"] > 0) & (df["month"] > 0)]
    if df_valid.empty:
        return []
    df_valid = df_valid.copy()
    df_valid["is_negative"] = (df_valid["sentiment_label"] == "NEGATIVE").astype(int)
    trend = (
        df_valid.groupby(["year", "month"])
        .agg(total=("is_negative", "count"), negative=("is_negative", "sum"))
        .reset_index()
    )
    trend["negative_pct"] = (trend["negative"] / trend["total"] * 100).round(2)
    trend["period"] = trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
    return trend.sort_values(["year", "month"]).to_dict(orient="records")


def _top_complaint_topics(df: pd.DataFrame) -> list[dict]:
    """Top topics from topics Parquet (or category_tag if available)."""
    # Try topics Parquet
    if TOPICS_PARQUET.exists():
        df_topics = pd.read_parquet(TOPICS_PARQUET, engine="pyarrow")
        if "topics" in df_topics.columns:
            df_exp = df_topics.explode("topics")
            freq = df_exp["topics"].value_counts().head(15).reset_index()
            freq.columns = ["topic", "count"]
            return freq.to_dict(orient="records")

    # Fallback to category_tag
    if "category_tag" in df.columns:
        freq = df["category_tag"].value_counts().head(15).reset_index()
        freq.columns = ["topic", "count"]
        return freq.to_dict(orient="records")

    return []


def _location_complaint_frequency(df: pd.DataFrame) -> list[dict]:
    """Post count per location, sorted descending."""
    if "location" not in df.columns:
        return []
    freq = (
        df.groupby("location")
        .agg(
            post_count=("location", "count"),
            negative_count=("sentiment_label", lambda x: (x == "NEGATIVE").sum()),
            avg_sentiment=("sentiment_score", "mean") if "sentiment_score" in df.columns else ("location", "count"),
        )
        .round(4)
        .sort_values("post_count", ascending=False)
        .head(20)
        .reset_index()
    )
    return freq.to_dict(orient="records")


def _engagement_trends(df: pd.DataFrame) -> list[dict]:
    """Monthly engagement trend if engagement_score is available."""
    if "engagement_score" not in df.columns:
        return []
    df_valid = df[(df["year"] > 0) & (df["month"] > 0)]
    if df_valid.empty:
        return []
    trend = (
        df_valid.groupby(["year", "month"])["engagement_score"]
        .agg(["mean", "sum"])
        .rename(columns={"mean": "avg_engagement", "sum": "total_engagement"})
        .round(2)
        .reset_index()
    )
    trend["period"] = trend["year"].astype(str) + "-" + trend["month"].astype(str).str.zfill(2)
    return trend.sort_values(["year", "month"]).to_dict(orient="records")


def build_trends() -> pd.DataFrame:
    """Build and save social trends Parquet."""
    df = _load_data()
    df = _ensure_time_columns(df)
    df = _ensure_sentiment(df)
    logger.info("Building trends from %d posts.", len(df))

    # Build monthly summary
    df_valid = df[(df["year"] > 0) & (df["month"] > 0)].copy()
    df_valid["is_negative"] = (df_valid["sentiment_label"] == "NEGATIVE").astype(int)

    monthly_agg = (
        df_valid.groupby(["year", "month"])
        .agg(
            post_count=("is_negative", "count"),
            negative_count=("is_negative", "sum"),
            avg_sentiment=("sentiment_score", "mean") if "sentiment_score" in df_valid.columns else ("is_negative", "count"),
        )
        .round(4)
        .reset_index()
    )
    monthly_agg["negative_pct"] = (monthly_agg["negative_count"] / monthly_agg["post_count"] * 100).round(2)
    monthly_agg["period"] = monthly_agg["year"].astype(str) + "-" + monthly_agg["month"].astype(str).str.zfill(2)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    monthly_agg.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Trends saved: %s", OUTPUT_PARQUET)
    return df


def compute_trends_report(df: pd.DataFrame) -> dict:
    df = _ensure_time_columns(df)
    df = _ensure_sentiment(df)

    return {
        "total_posts": int(len(df)),
        "monthly_post_counts": _monthly_post_counts(df),
        "monthly_negative_sentiment": _monthly_negative_sentiment(df),
        "top_complaint_topics": _top_complaint_topics(df),
        "location_complaint_frequency": _location_complaint_frequency(df),
        "engagement_trends": _engagement_trends(df),
    }


def run_trends_analysis() -> tuple[pd.DataFrame, dict]:
    df = build_trends()
    report = compute_trends_report(df)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return df, report


def main():
    df, report = run_trends_analysis()
    logger.info("Trends report saved: %s", REPORT_PATH)

    print("\n" + "=" * 65)
    print("SOCIAL TRENDS SUMMARY")
    print("=" * 65)
    print(f"  Total posts:            {report['total_posts']}")
    print(f"\n  Top Complaint Topics:")
    for t in report["top_complaint_topics"][:8]:
        print(f"    {t['topic']:<30} {t['count']}")
    print(f"\n  Top 10 Locations by Post Volume:")
    for loc in report["location_complaint_frequency"][:10]:
        print(f"    {loc['location']:<20} posts={loc['post_count']}  negative={loc.get('negative_count', 'N/A')}")
    print("=" * 65 + "\n")

    return report


if __name__ == "__main__":
    main()
