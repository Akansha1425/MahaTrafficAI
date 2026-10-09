"""Social media preprocessing pipeline for batch perception signals.

Prepares cleaned social posts for downstream sentiment analysis, topic modeling,
and community detection (Phase 6). Runs locally using Pandas and PyArrow.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.prep")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "social_features.parquet"


def prepare_social_analytics() -> pd.DataFrame:
    """Prepare structured features from cleaned social media posts."""
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Cleaned social Parquet missing at: {INPUT_PARQUET}")

    logger.info("Reading cleaned social Parquet from: %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")

    # 1. Hashtag extraction and token splitting
    df["hashtag_list"] = df["hashtags"].fillna("").apply(
        lambda x: [tag.strip() for tag in str(x).split(";") if tag.strip()]
    )
    df["hashtag_count"] = df["hashtag_list"].apply(len)

    # 2. Civic complaint topic classification signals
    topic_keywords = {
        "pothole_signal": ["pothole", "crater", "trench", "gravel"],
        "monsoon_flood_signal": ["waterlogging", "rain", "flood", "skid"],
        "speeding_signal": ["speeding", "rash", "racing", "speed"],
        "infrastructure_signal": ["streetlight", "dark", "signal", "zebra", "barrier"],
    }

    for signal_col, keywords in topic_keywords.items():
        pattern = "|".join(keywords)
        df[signal_col] = df["clean_text"].str.contains(pattern, case=False, regex=True).astype(int)

    # 3. Normalized engagement index (0 - 100)
    raw_eng = df["engagement_score"]
    eng_max = raw_eng.max() if raw_eng.max() > 0 else 1
    df["engagement_normalized"] = (raw_eng / eng_max * 100).round(2)

    # 4. Save enhanced Parquet
    logger.info("Saving enriched social features Parquet to: %s", OUTPUT_PARQUET)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)

    print("\n" + "=" * 60)
    print("SOCIAL MEDIA PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total processed posts:      {len(df)}")
    print(f"Unique municipal locations: {df['location'].nunique()} -> {df['location'].unique().tolist()}")
    print(f"Total hashtags parsed:      {df['hashtag_count'].sum()}")
    print(f"Output Parquet path:        {OUTPUT_PARQUET}")
    print("=" * 60 + "\n")

    return df


if __name__ == "__main__":
    prepare_social_analytics()
