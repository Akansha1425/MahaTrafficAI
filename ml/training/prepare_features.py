"""Machine Learning feature engineering pipeline.

Transforms processed accident records into a standardized ML feature matrix
saved as columnar Parquet for downstream model training (Random Forest in Phase 7).
Runs locally using Pandas, NumPy, and PyArrow.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.features")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "features"
OUTPUT_PARQUET = OUTPUT_DIR / "risk_features.parquet"


def build_risk_features() -> pd.DataFrame:
    """Engineer tabular features and ground-truth risk category targets."""
    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(f"Cleaned accident Parquet not found at: {INPUT_PARQUET}")

    logger.info("Loading cleaned accident Parquet from: %s", INPUT_PARQUET)
    df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")
    logger.info("Raw input records: %d", len(df))

    # 1. Feature Engineering
    # Temporal cyclical features for month (sine and cosine transforms)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

    # Boolean indicator features
    df["is_monsoon"] = (df["season"] == "Monsoon").astype(int)
    df["is_night"] = df["time_period"].str.contains("Night", case=False, na=False).astype(int)
    df["is_highway"] = df["road_type"].str.contains("Highway|Expressway", case=False, na=False).astype(int)

    # Risk Score Target per record (Normalized 0 - 100)
    # Severity Component (0-100)
    sev_raw = (df["deaths"] * 1.0 + df["injuries"] * 0.3) / df["accident_count"].replace(0, 1)
    sev_norm = (sev_raw / sev_raw.max() * 100).clip(0, 100)

    # Frequency Component (0-100)
    freq_norm = (df["accident_count"] / df["accident_count"].max() * 100).clip(0, 100)

    # Road Factor (Highways get 85, urban get 45)
    road_factor = np.where(df["road_type"].str.contains("National Highway", na=False), 90.0,
                  np.where(df["road_type"].str.contains("State Highway", na=False), 75.0,
                  np.where(df["road_type"].str.contains("Expressway", na=False), 65.0, 40.0)))

    # Time Factor (Night gets 80, Day gets 40)
    time_factor = np.where(df["is_night"] == 1, 80.0, 40.0)

    # Composite analytical risk score per historical batch slice
    score = (0.40 * sev_norm + 0.25 * freq_norm + 0.20 * road_factor + 0.15 * time_factor).round(2)
    df["record_risk_score"] = score

    # Target class labels
    conditions = [
        (df["record_risk_score"] <= 33.0),
        (df["record_risk_score"] > 33.0) & (df["record_risk_score"] <= 66.0),
        (df["record_risk_score"] > 66.0),
    ]
    choices = ["LOW", "MEDIUM", "HIGH"]
    df["risk_category"] = np.select(conditions, choices, default="MEDIUM")

    # Select core feature subset
    feature_cols = [
        "year",
        "month",
        "month_sin",
        "month_cos",
        "is_monsoon",
        "is_night",
        "is_highway",
        "district",
        "road_type",
        "primary_cause",
        "accident_count",
        "fatal_accidents",
        "deaths",
        "injuries",
        "fatality_ratio",
        "injury_ratio",
        "severity_index",
        "record_risk_score",
        "risk_category",
    ]
    features_df = df[feature_cols].copy()

    # Save to Parquet
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Persisting risk features matrix to %s", OUTPUT_PARQUET)
    features_df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Total feature records:   {len(features_df)}")
    print(f"Total feature columns:   {len(features_df.columns)}")
    print(f"Output Parquet:          {OUTPUT_PARQUET}")
    print("\nClass distribution ('risk_category'):")
    print(features_df["risk_category"].value_counts().to_string())
    print("=" * 60 + "\n")

    return features_df


if __name__ == "__main__":
    build_risk_features()
