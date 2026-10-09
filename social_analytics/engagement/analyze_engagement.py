"""Social Engagement Analytics — Phase 4, MahaTraffic AI.

Calculates engagement patterns and public traction from social media:
  - Aggregate engagement statistics (mean, median, max likes, shares, comments)
  - Top viral/high-impact safety complaints
  - Engagement distribution across sentiment categories
  - Location-based engagement summaries
  - Engagement velocity and reach estimates

Inputs:  data/processed/social_sentiment.parquet
         (or data/processed/social_clean.parquet)
Outputs: data/processed/social_engagement.parquet
         data/processed/social_engagement_report.json
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.engagement")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SENTIMENT_PARQUET = BASE_DIR / "data" / "processed" / "social_sentiment.parquet"
CLEAN_PARQUET = BASE_DIR / "data" / "processed" / "social_clean.parquet"
FALLBACK_PARQUET = (
    BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
)
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_engagement.parquet"
REPORT_JSON = BASE_DIR / "data" / "processed" / "social_engagement_report.json"


def load_social_data() -> pd.DataFrame:
    """Load the most enriched available social media dataset."""
    for p in [SENTIMENT_PARQUET, CLEAN_PARQUET, FALLBACK_PARQUET]:
        if p.exists():
            logger.info("Loading dataset from: %s", p)
            return pd.read_parquet(p, engine="pyarrow")
    raise FileNotFoundError("No social dataset found. Please run preprocessing first.")


def compute_engagement_metrics(df: pd.DataFrame) -> dict:
    """Compute comprehensive engagement statistics across the corpus."""
    likes = df["likes"] if "likes" in df.columns else pd.Series([0] * len(df))
    shares = df["shares"] if "shares" in df.columns else pd.Series([0] * len(df))
    comments = df["comments"] if "comments" in df.columns else pd.Series([0] * len(df))

    if "engagement_score" not in df.columns:
        df["engagement_score"] = (likes + shares * 2.0 + comments * 1.5).round(2)

    total_engagement = float(df["engagement_score"].sum())
    mean_engagement = float(df["engagement_score"].mean())
    median_engagement = float(df["engagement_score"].median())
    max_engagement = float(df["engagement_score"].max())

    # High engagement threshold: 90th percentile
    p90 = float(np.percentile(df["engagement_score"], 90))
    high_engagement_posts = df[df["engagement_score"] >= p90]

    # By sentiment if present
    sentiment_engagement = {}
    if "sentiment_label" in df.columns:
        sentiment_agg = df.groupby("sentiment_label")["engagement_score"].agg(["count", "mean", "sum"])
        for label, row in sentiment_agg.iterrows():
            sentiment_engagement[label] = {
                "count": int(row["count"]),
                "mean_engagement": round(float(row["mean"]), 2),
                "total_engagement": round(float(row["sum"]), 2),
            }

    # By location (top 10 by engagement)
    location_engagement = []
    if "location" in df.columns:
        loc_agg = (
            df.groupby("location")
            .agg(
                post_count=("engagement_score", "count"),
                avg_engagement=("engagement_score", "mean"),
                total_engagement=("engagement_score", "sum"),
            )
            .sort_values(by="total_engagement", ascending=False)
            .head(10)
        )
        for loc, row in loc_agg.iterrows():
            location_engagement.append({
                "location": str(loc),
                "post_count": int(row["post_count"]),
                "avg_engagement": round(float(row["avg_engagement"]), 2),
                "total_engagement": round(float(row["total_engagement"]), 2),
            })

    # Top 5 most engaged posts
    top_posts = []
    text_col = "clean_text" if "clean_text" in df.columns else "text"
    if text_col in df.columns:
        top_rows = df.sort_values(by="engagement_score", ascending=False).head(5)
        for _, r in top_rows.iterrows():
            top_posts.append({
                "post_id": str(r.get("post_id", "")),
                "text": str(r.get(text_col, ""))[:120],
                "location": str(r.get("location", "Unknown")),
                "engagement_score": float(r["engagement_score"]),
                "likes": int(r.get("likes", 0)),
                "shares": int(r.get("shares", 0)),
            })

    report = {
        "total_posts": len(df),
        "total_engagement_score": round(total_engagement, 2),
        "mean_engagement_score": round(mean_engagement, 2),
        "median_engagement_score": round(median_engagement, 2),
        "max_engagement_score": round(max_engagement, 2),
        "p90_threshold": round(p90, 2),
        "high_engagement_count": int(len(high_engagement_posts)),
        "engagement_by_sentiment": sentiment_engagement,
        "top_locations_by_engagement": location_engagement,
        "top_viral_posts": top_posts,
    }
    return report


def run_engagement_pipeline() -> tuple[pd.DataFrame, dict]:
    """Execute engagement analytics and persist artifacts."""
    df = load_social_data()
    report = compute_engagement_metrics(df)

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Saved engagement Parquet: %s", OUTPUT_PARQUET)

    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved engagement report: %s", REPORT_JSON)

    print("\n" + "=" * 65)
    print("SOCIAL ENGAGEMENT ANALYTICS SUMMARY")
    print("=" * 65)
    print(f"  Total posts:          {report['total_posts']}")
    print(f"  Mean engagement:      {report['mean_engagement_score']}")
    print(f"  P90 threshold:        {report['p90_threshold']}")
    print(f"  High-engagement posts:{report['high_engagement_count']}")
    print(f"  Output Parquet:       {OUTPUT_PARQUET}")
    print(f"  Output JSON:          {REPORT_JSON}")
    print("=" * 65 + "\n")

    return df, report


if __name__ == "__main__":
    run_engagement_pipeline()
