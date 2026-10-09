"""Social Media Cleaning Pipeline — Phase 4, MahaTraffic AI.

Cleans and normalises the raw social media posts dataset:
  data/raw/social_media/maharashtra_road_safety_public_posts.csv

Operations:
  - Text cleaning (URL removal, normalisation, whitespace)
  - Duplicate removal
  - Missing value handling
  - Timestamp parsing
  - Engagement normalisation

Saves:
  data/processed/social_clean.parquet

NOTE: Social posts are treated as public perception / complaint signals.
      They are NOT treated as causes of accidents.
"""

from __future__ import annotations
from pathlib import Path
import logging
import re
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.clean")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "social_media" / "maharashtra_road_safety_public_posts.csv"
PARQUET_INPUT = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_clean.parquet"


# ─── Text Cleaning ────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HASHTAG_RE = re.compile(r"#\w+")
_MENTION_RE = re.compile(r"@\w+")
_PUNCT_RE = re.compile(r"[^\w\s.,!?'-]")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Clean a single post text string."""
    if not isinstance(text, str):
        return ""
    t = _URL_RE.sub(" ", text)
    t = _MENTION_RE.sub(" ", t)
    # Keep hashtag text but remove the '#' symbol
    t = _HASHTAG_RE.sub(lambda m: m.group(0)[1:], t)
    t = _PUNCT_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t


def normalize_location(loc: str) -> str:
    """Standardise location strings."""
    if not isinstance(loc, str) or loc.strip() == "":
        return "Unknown"
    return loc.strip().title()


def parse_hashtags(raw: str) -> list[str]:
    """Parse semicolon-separated hashtag strings into a list."""
    if not isinstance(raw, str):
        return []
    return [t.strip().lstrip("#") for t in raw.split(";") if t.strip()]


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_cleaning() -> pd.DataFrame:
    """Execute the full cleaning pipeline."""
    # Prefer existing clean Parquet; fall back to raw CSV
    if PARQUET_INPUT.exists():
        logger.info("Loading cleaned Parquet: %s", PARQUET_INPUT)
        df = pd.read_parquet(PARQUET_INPUT, engine="pyarrow")
    elif RAW_CSV.exists():
        logger.info("Loading raw CSV: %s", RAW_CSV)
        df = pd.read_csv(RAW_CSV)
    else:
        raise FileNotFoundError(
            f"No source found. Expected:\n  {PARQUET_INPUT}\n  {RAW_CSV}"
        )

    logger.info("Loaded %d posts.", len(df))
    before = len(df)

    # ── Duplicate removal ──
    df = df.drop_duplicates(subset=["post_id"] if "post_id" in df.columns else None)
    logger.info("After dedup: %d (removed %d)", len(df), before - len(df))

    # ── Missing value handling ──
    text_col = "text" if "text" in df.columns else "clean_text"
    if text_col not in df.columns:
        raise ValueError("No text column found in dataset.")

    df = df[df[text_col].notna() & (df[text_col].str.strip() != "")].copy()

    # ── Text cleaning ──
    df["clean_text"] = df[text_col].apply(clean_text)

    # ── Location normalisation ──
    if "location" in df.columns:
        df["location"] = df["location"].apply(normalize_location)

    # ── Timestamp parsing ──
    timestamp_col = next((c for c in ["timestamp", "date", "post_date"] if c in df.columns), None)
    if timestamp_col:
        df["post_date"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df["year"] = df["post_date"].dt.year.fillna(0).astype(int)
        df["month"] = df["post_date"].dt.month.fillna(0).astype(int)
    else:
        df["post_date"] = pd.NaT
        df["year"] = 0
        df["month"] = 0

    # ── Hashtag parsing ──
    if "hashtags" in df.columns:
        df["hashtag_list"] = df["hashtags"].apply(parse_hashtags)
        df["hashtag_count"] = df["hashtag_list"].apply(len)
    else:
        df["hashtag_list"] = [[] for _ in range(len(df))]
        df["hashtag_count"] = 0

    # ── Engagement score ──
    likes_col = "likes" if "likes" in df.columns else None
    shares_col = "shares" if "shares" in df.columns else None
    comments_col = "comments" if "comments" in df.columns else None

    if not ("engagement_score" in df.columns):
        likes = df[likes_col].fillna(0) if likes_col else 0
        shares = df[shares_col].fillna(0) if shares_col else 0
        comments = df[comments_col].fillna(0) if comments_col else 0
        df["engagement_score"] = (likes + shares * 2 + comments * 1.5).round(2)

    # ── Save ──
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Clean social data saved: %s", OUTPUT_PARQUET)

    print("\n" + "=" * 65)
    print("SOCIAL DATA CLEANING SUMMARY")
    print("=" * 65)
    print(f"  Input records:          {before}")
    print(f"  Clean records:          {len(df)}")
    print(f"  Unique locations:       {df['location'].nunique() if 'location' in df.columns else 'N/A'}")
    print(f"  Date range:             {df['year'].min()} – {df['year'].max()}")
    print(f"  Output Parquet:         {OUTPUT_PARQUET}")
    print("=" * 65 + "\n")

    return df


if __name__ == "__main__":
    run_cleaning()
