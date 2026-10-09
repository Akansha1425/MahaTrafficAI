"""Social media data cleaning and Parquet pipeline module.

Executes batch data cleaning: text whitespace normalization, timestamp parsing,
duplicate removal, engagement score calculation, and Parquet persistence.
Preserves original text without destructive redaction.
"""

from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleaning.social")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "social_media" / "maharashtra_road_safety_public_posts.csv"
OUTPUT_PARQUET_DIR = BASE_DIR / "data" / "processed" / "parquet" / "social_media"
OUTPUT_CSV_FILE = BASE_DIR / "data" / "processed" / "social_media_clean.csv"


def clean_social_pipeline() -> pd.DataFrame:
    """Run full cleaning and Parquet transformation on social complaints dataset."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw social dataset not found at: {RAW_FILE}")

    logger.info("Loading raw social dataset from %s", RAW_FILE)
    df_raw = pd.read_csv(RAW_FILE)
    input_records = len(df_raw)

    df = df_raw.copy()

    # 1. Duplicate detection on post_id or identical text
    dup_ids = int(df["post_id"].duplicated().sum()) if "post_id" in df.columns else 0
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # 2. Text sanitization (trim whitespace, remove empty or null text while preserving original content)
    df["original_text"] = df["text"].astype(str)
    df["clean_text"] = df["original_text"].str.strip().str.replace(r"\s+", " ", regex=True)
    initial_len = len(df)
    df = df[df["clean_text"] != ""].copy()
    empty_removed = initial_len - len(df)

    # 3. Timestamp normalization to ISO datetime
    df["parsed_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["post_year"] = df["parsed_timestamp"].dt.year
    df["post_month"] = df["parsed_timestamp"].dt.month

    # 4. Location normalization
    location_map = {
        "Poona": "Pune",
        "Bombay": "Mumbai",
        "Aurangabad": "Chhatrapati Sambhajinagar",
    }
    df["location"] = df["location"].astype(str).str.strip().replace(location_map)

    # 5. Engagement Metrics computation
    for col in ["likes", "shares", "comments"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Weighted engagement formula (shares and comments indicate higher civic concern)
    df["engagement_score"] = df["likes"] * 1 + df["shares"] * 2 + df["comments"] * 3
    df["text_length"] = df["clean_text"].str.len()

    # 6. Save to Parquet
    OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_PARQUET_DIR / "maharashtra_social_clean.parquet"
    
    logger.info("Writing clean social Parquet to %s", parquet_path)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    # Save CSV reference
    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_FILE, index=False)

    removed_records = dup_ids + empty_removed
    final_records = len(df)
    transformed_records = final_records

    # Print reporting metrics
    print("\n" + "=" * 60)
    print("SOCIAL DATA CLEANING & PARQUET CONVERSION REPORT")
    print("=" * 60)
    print(f"input records:       {input_records}")
    print(f"removed records:     {removed_records}")
    print(f"transformed records: {transformed_records}")
    print(f"final records:       {final_records}")
    print(f"output parquet:      {parquet_path}")
    print(f"parquet size (bytes):{parquet_path.stat().st_size:,}")
    print("=" * 60 + "\n")

    return df


if __name__ == "__main__":
    clean_social_pipeline()
