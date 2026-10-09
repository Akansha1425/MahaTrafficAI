"""Social Media Sentiment Analysis Pipeline — MahaTraffic AI.

Performs:
  1. Rule-based + lexicon sentiment scoring (no external API required)
  2. Topic classification for road-safety categories
  3. Location-level public concern aggregation
  4. Temporal trend analysis

Input:  data/processed/parquet/social_media/maharashtra_social_clean.parquet
Output: data/processed/parquet/social_media/social_features.parquet
        data/processed/parquet/social_media/sentiment_report.json
"""

from pathlib import Path
import json
import logging
import pandas as pd
import numpy as np
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.sentiment")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
OUTPUT_FEATURES = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "social_features.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "sentiment_report.json"

# ─── Lexicons ────────────────────────────────────────────────────────────────
NEGATIVE_TERMS = [
    "accident", "crash", "fatal", "death", "died", "killed", "injury", "injured",
    "pothole", "dangerous", "hazard", "reckless", "drunk", "speeding", "worst",
    "terrible", "awful", "horrible", "bad", "poor", "broken", "damaged", "unsafe",
    "risk", "tragedy", "tragic", "victim", "casualty", "block", "jammed", "congestion",
    "hit", "collision", "overturned", "fell", "fled", "negligence", "unattended",
]

POSITIVE_TERMS = [
    "safe", "improved", "better", "fixed", "repaired", "smooth", "clear",
    "well-maintained", "good", "excellent", "helpful", "efficient", "signal",
    "divider", "footpath", "zebra", "awareness", "campaign", "patrol", "rescue",
]

INTENSIFIERS = ["very", "extremely", "highly", "severely", "absolutely", "completely"]

TOPIC_KEYWORDS = {
    "road_condition": [
        "pothole", "road", "highway", "surface", "divider", "median", "lane",
        "footpath", "pavement", "broken road", "bad road",
    ],
    "accident_report": [
        "accident", "crash", "collision", "hit", "fell", "overturned", "fatal",
        "death", "injury", "victim",
    ],
    "traffic_management": [
        "signal", "traffic", "jam", "congestion", "block", "police", "patrol",
        "challan", "fine", "violation",
    ],
    "drunk_driving": [
        "drunk", "alcohol", "liquor", "intoxicated", "drunken",
    ],
    "speeding": [
        "speed", "speeding", "fast", "overspeed", "rash", "reckless",
    ],
    "infrastructure": [
        "bridge", "flyover", "underpass", "intersection", "junction",
        "expressway", "bypass",
    ],
}


# ─── Scoring Functions ────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase tokenize, remove punctuation."""
    return re.findall(r'\b[a-z]+\b', text.lower())


def score_sentiment(text: str) -> tuple[float, str]:
    """Compute sentiment score [-1, +1] and label."""
    tokens = _tokenize(text)
    neg_count = sum(1 for t in tokens if t in NEGATIVE_TERMS)
    pos_count = sum(1 for t in tokens if t in POSITIVE_TERMS)
    intensifier_boost = sum(0.2 for t in tokens if t in INTENSIFIERS)

    raw = (pos_count - neg_count) + intensifier_boost
    # Normalize to [-1, 1]
    total = max(pos_count + neg_count, 1)
    score = float(np.clip(raw / total, -1.0, 1.0))

    if score >= 0.15:
        label = "POSITIVE"
    elif score <= -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return round(score, 4), label


def classify_topics(text: str) -> list[str]:
    """Assign one or more road-safety topic labels to a post."""
    text_lower = text.lower()
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            matched.append(topic)
    return matched if matched else ["other"]


def concern_score(row: pd.Series) -> float:
    """Composite public concern score for a post (0-100).
    Higher means stronger negative public perception.
    """
    engagement_norm = min(row.get("engagement_score", 0) / 100.0, 1.0)
    sentiment_weight = max(-row["sentiment_score"], 0)  # negative sentiment → higher concern
    retweet_weight = min(row.get("shares", 0) / 20.0, 1.0)
    raw = (0.5 * sentiment_weight + 0.3 * engagement_norm + 0.2 * retweet_weight) * 100
    return round(float(raw), 2)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_sentiment_pipeline() -> pd.DataFrame:
    """Execute full sentiment analysis pipeline."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Social Parquet not found: {INPUT_PATH}")

    logger.info("Loading social data from: %s", INPUT_PATH)
    df = pd.read_parquet(INPUT_PATH, engine="pyarrow")
    logger.info("Loaded %d posts.", len(df))

    # Sentiment scoring
    logger.info("Applying sentiment scoring...")
    sentiment_results = df["text"].apply(score_sentiment)
    df["sentiment_score"] = sentiment_results.apply(lambda x: x[0])
    df["sentiment_label"] = sentiment_results.apply(lambda x: x[1])

    # Topic classification
    logger.info("Classifying topics...")
    df["topics"] = df["text"].apply(lambda t: ", ".join(classify_topics(t)))

    # Concern score
    logger.info("Computing concern scores...")
    df["concern_score"] = df.apply(concern_score, axis=1)

    # Extract year/month for temporal analysis
    for date_col in ["post_date", "timestamp", "date"]:
        if date_col in df.columns:
            df["post_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["year"] = df["post_date"].dt.year
            df["month"] = df["post_date"].dt.month
            break
    # Ensure year/month columns exist
    if "year" not in df.columns:
        df["year"] = 2022
    if "month" not in df.columns:
        df["month"] = 1

    return df


def compute_report(df: pd.DataFrame) -> dict:
    """Compute aggregated analytics for reporting."""
    # Overall sentiment distribution
    sentiment_dist = df["sentiment_label"].value_counts().to_dict()

    # Location-level concern
    location_concern = (
        df.groupby("location")["concern_score"]
        .agg(["mean", "count", "max"])
        .rename(columns={"mean": "avg_concern", "count": "post_count", "max": "peak_concern"})
        .round(2)
        .sort_values("avg_concern", ascending=False)
        .head(10)
        .reset_index()
        .to_dict(orient="records")
    )

    # Yearly trend
    yearly_sentiment = (
        df.groupby("year")["sentiment_score"]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={"sentiment_score": "avg_sentiment"})
        .to_dict(orient="records")
    )

    # Topic frequency
    all_topics = []
    for t in df["topics"]:
        all_topics.extend([x.strip() for x in t.split(",")])
    topic_series = pd.Series(all_topics)
    topic_freq = topic_series.value_counts().to_dict()

    # Most concerning posts (top 5)
    top_concerning = (
        df.nlargest(5, "concern_score")[["post_id", "text", "location", "concern_score", "sentiment_label"]]
        .to_dict(orient="records")
    )

    return {
        "total_posts": int(len(df)),
        "sentiment_distribution": {k: int(v) for k, v in sentiment_dist.items()},
        "location_concern_ranking": location_concern,
        "yearly_sentiment_trend": yearly_sentiment,
        "topic_frequency": {k: int(v) for k, v in topic_freq.items()},
        "most_concerning_posts": top_concerning,
    }


def main():
    df = run_sentiment_pipeline()

    # Save enriched features
    df.to_parquet(OUTPUT_FEATURES, engine="pyarrow", compression="snappy", index=False)
    logger.info("Social features saved: %s", OUTPUT_FEATURES)

    # Compute report
    report = compute_report(df)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Sentiment report saved: %s", REPORT_PATH)

    # Print summary
    print("\n" + "=" * 60)
    print("SOCIAL MEDIA SENTIMENT ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Total Posts Analyzed:  {report['total_posts']}")
    print(f"  Sentiment Distribution: {report['sentiment_distribution']}")
    print("\n  Top 5 Most Concerning Locations:")
    for i, loc in enumerate(report["location_concern_ranking"][:5], 1):
        print(f"    {i}. {loc['location']}: avg concern={loc['avg_concern']}, posts={loc['post_count']}")
    print("\n  Topic Frequencies:")
    for topic, freq in sorted(report["topic_frequency"].items(), key=lambda x: -x[1]):
        print(f"    {topic:<25} {freq}")
    print("=" * 60 + "\n")

    return df, report


if __name__ == "__main__":
    main()
