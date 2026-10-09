"""Sentiment Analysis Pipeline — Phase 4, MahaTraffic AI.

Performs lightweight, locally runnable sentiment analysis on cleaned
social posts. Uses rule-based lexicon scoring (no external API or GPU).

Labels: POSITIVE | NEUTRAL | NEGATIVE

Input:  data/processed/social_clean.parquet
        (fallback: data/processed/parquet/social_media/maharashtra_social_clean.parquet)

Output: data/processed/social_sentiment.parquet

Analytics computed:
  - Overall sentiment distribution
  - Sentiment by location
  - Sentiment by month
  - Negative sentiment percentage

DISCLAIMER: Social sentiment is treated as a PUBLIC PERCEPTION /
            COMPLAINT SIGNAL. It is NOT treated as a cause of accidents.
            No causal relationship is inferred.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import re
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.sentiment")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_clean.parquet"
FALLBACK_PARQUET = (
    BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
)
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_sentiment.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "social_sentiment_report.json"

# ─── Lexicons ────────────────────────────────────────────────────────────────

NEGATIVE_TERMS = frozenset([
    "accident", "crash", "fatal", "death", "died", "killed", "injury", "injured",
    "pothole", "dangerous", "hazard", "reckless", "drunk", "speeding", "worst",
    "terrible", "awful", "horrible", "bad", "poor", "broken", "damaged", "unsafe",
    "risk", "tragedy", "tragic", "victim", "casualty", "blocked", "jammed",
    "congestion", "hit", "collision", "overturned", "fled", "negligence", "unattended",
    "waterlogging", "flooded", "skid", "skidding", "potholed", "cracked", "dark",
    "unlit", "unguarded", "no signal", "demand", "complaint", "concern", "serious",
    "emergency", "urgent", "shameful", "disappointing", "frustrated", "helpless",
])

POSITIVE_TERMS = frozenset([
    "safe", "improved", "better", "fixed", "repaired", "smooth", "clear",
    "well-maintained", "good", "excellent", "helpful", "efficient", "functioning",
    "protected", "safe road", "awareness", "campaign", "patrol", "rescue",
    "appreciation", "thanks", "congratulations", "well done", "commendable",
    "speed breaker", "zebra crossing", "footpath", "divider",
])

INTENSIFIERS = frozenset([
    "very", "extremely", "highly", "severely", "absolutely", "completely",
    "totally", "deeply", "seriously", "badly",
])


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-z]+\b', text.lower())


def score_sentiment(text: str) -> tuple[float, str]:
    """Compute sentiment score in [-1, +1] and label."""
    tokens = _tokenize(text)
    neg = sum(1 for t in tokens if t in NEGATIVE_TERMS)
    pos = sum(1 for t in tokens if t in POSITIVE_TERMS)
    boost = sum(0.25 for t in tokens if t in INTENSIFIERS)

    raw = (pos - neg) + boost
    total = max(pos + neg, 1)
    score = float(np.clip(raw / total, -1.0, 1.0))

    if score >= 0.15:
        label = "POSITIVE"
    elif score <= -0.15:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"

    return round(score, 4), label


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    if INPUT_PARQUET.exists():
        return pd.read_parquet(INPUT_PARQUET, engine="pyarrow")
    if FALLBACK_PARQUET.exists():
        return pd.read_parquet(FALLBACK_PARQUET, engine="pyarrow")
    raise FileNotFoundError(
        "No social clean Parquet found. Run clean_social.py first."
    )


def run_sentiment_analysis() -> pd.DataFrame:
    """Execute full sentiment analysis pipeline."""
    df = _load_data()
    logger.info("Loaded %d posts for sentiment analysis.", len(df))

    # Determine text column
    text_col = "clean_text" if "clean_text" in df.columns else "text"

    # Sentiment scoring
    logger.info("Scoring sentiment...")
    results = df[text_col].fillna("").apply(score_sentiment)
    df["sentiment_score"] = results.apply(lambda x: x[0])
    df["sentiment_label"] = results.apply(lambda x: x[1])

    # Ensure year/month exist
    if "year" not in df.columns:
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], errors="coerce")
            df["year"] = ts.dt.year.fillna(0).astype(int)
            df["month"] = ts.dt.month.fillna(0).astype(int)
        else:
            df["year"] = 0
            df["month"] = 0

    # Reference text for output
    df["text_ref"] = df[text_col].str[:120]

    # ── Save ──
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Sentiment data saved: %s", OUTPUT_PARQUET)
    return df


def compute_sentiment_analytics(df: pd.DataFrame) -> dict:
    """Compute aggregated sentiment analytics."""
    total = len(df)

    # Overall distribution
    dist = df["sentiment_label"].value_counts().to_dict()
    dist_pct = {k: round(v / total * 100, 2) for k, v in dist.items()}
    neg_pct = dist_pct.get("NEGATIVE", 0.0)

    # Sentiment by location
    if "location" in df.columns:
        loc_sentiment = (
            df.groupby("location")["sentiment_score"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_sentiment", "count": "post_count"})
            .round(4)
            .sort_values("avg_sentiment")
            .head(10)
            .reset_index()
            .to_dict(orient="records")
        )
    else:
        loc_sentiment = []

    # Sentiment by month
    if "month" in df.columns:
        monthly = (
            df[df["month"] > 0]
            .groupby("month")["sentiment_score"]
            .mean()
            .round(4)
            .reset_index()
            .rename(columns={"sentiment_score": "avg_sentiment"})
            .to_dict(orient="records")
        )
    else:
        monthly = []

    return {
        "total_posts": int(total),
        "sentiment_distribution": {k: int(v) for k, v in dist.items()},
        "sentiment_distribution_pct": dist_pct,
        "negative_sentiment_pct": neg_pct,
        "sentiment_by_location": loc_sentiment,
        "sentiment_by_month": monthly,
        "disclaimer": (
            "Social sentiment reflects public perception / complaint signals. "
            "No causal relationship between sentiment and accident occurrence is implied."
        ),
    }


def main():
    df = run_sentiment_analysis()
    analytics = compute_sentiment_analytics(df)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(analytics, f, indent=2, ensure_ascii=False)
    logger.info("Sentiment report saved: %s", REPORT_PATH)

    print("\n" + "=" * 65)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("(Public perception / complaint signal — not causal)")
    print("=" * 65)
    print(f"  Total posts:            {analytics['total_posts']}")
    print(f"  Negative sentiment:     {analytics['negative_sentiment_pct']:.1f}%")
    print("\n  Distribution:")
    for label, cnt in analytics["sentiment_distribution"].items():
        pct = analytics["sentiment_distribution_pct"].get(label, 0)
        bar = "█" * int(pct / 5)
        print(f"    {label:<10} {cnt:>5} ({pct:.1f}%)  {bar}")
    print("\n  Most negative locations (complaint signals):")
    for row in analytics["sentiment_by_location"][:5]:
        print(f"    {row['location']:<20} avg={row['avg_sentiment']:.4f}  posts={row['post_count']}")
    print("=" * 65 + "\n")

    return df, analytics


if __name__ == "__main__":
    main()
