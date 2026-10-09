"""Accident Feature Engineering — Phase 4, MahaTraffic AI.

Generates a comprehensive ML feature matrix from the primary accident dataset:
  data/raw/accidents/maharashtra_district_accidents_2019_2023.csv

Features engineered:
  - accident_frequency, accident_severity, fatal_accident_count
  - injury_accident_count (fatal vs non-fatal split)
  - month, year, time_period, road_type, primary_cause, district
  - Cyclical month encoding (sin/cos)
  - Boolean flags: is_night, is_highway, is_monsoon
  - Aggregated district-level statistics

Saves:
  data/processed/features/accident_features.parquet

NOTE: All features are derived from observed historical records.
      This is an analytical pipeline, not a real-time prediction system.
"""

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.feature_engineering")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_CSV = BASE_DIR / "data" / "raw" / "accidents" / "maharashtra_district_accidents_2019_2023.csv"
CLEAN_PARQUET = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "features"
OUTPUT_PARQUET = OUTPUT_DIR / "accident_features.parquet"


# ─── Loaders ──────────────────────────────────────────────────────────────────

def _load_source() -> pd.DataFrame:
    """Load from Parquet if available, otherwise from raw CSV."""
    if CLEAN_PARQUET.exists():
        logger.info("Loading from cleaned Parquet: %s", CLEAN_PARQUET)
        return pd.read_parquet(CLEAN_PARQUET, engine="pyarrow")
    logger.info("Parquet not found; falling back to raw CSV: %s", RAW_CSV)
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"Raw accident CSV not found: {RAW_CSV}")
    return pd.read_csv(RAW_CSV)


# ─── Cleaning helpers ─────────────────────────────────────────────────────────

def _clean_base(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values, types, and duplicates on raw accident data."""
    # Drop complete duplicates
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicates removed: %d", before - len(df))

    # Numeric coercion
    for col in ["accident_count", "fatal_accidents", "deaths", "injuries"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Guard against zero accident_count
    df = df[df["accident_count"] > 0].copy()

    # Year / month validation
    df = df[(df["year"] >= 2015) & (df["year"] <= 2030)].copy()
    df = df[(df["month"] >= 1) & (df["month"] <= 12)].copy()

    # Categorical normalization
    str_cols = ["district", "road_type", "primary_cause", "time_period", "jurisdiction"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").str.strip()

    return df


# ─── Feature engineering ──────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full feature matrix from cleaned accident data."""
    df = _clean_base(df)
    logger.info("Records after cleaning: %d", len(df))

    # ── Temporal features ──
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)

    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    if "season" not in df.columns:
        df["season"] = df["month"].map(season_map).fillna("Unknown")

    # ── Boolean flags ──
    df["is_monsoon"] = (df["season"] == "Monsoon").astype(int)
    df["is_night"] = df["time_period"].str.contains("Night", case=False, na=False).astype(int)
    df["is_highway"] = df["road_type"].str.contains(
        "National Highway|State Highway|Expressway", case=False, na=False
    ).astype(int)

    # ── Severity components ──
    # accident_severity: deaths-weighted severity per accident
    df["accident_severity"] = (
        (df["deaths"] * 1.0 + df["injuries"] * 0.3) / df["accident_count"].replace(0, 1)
    ).round(4)

    df["fatal_accident_count"] = df["fatal_accidents"].astype(int)
    df["injury_accident_count"] = (df["accident_count"] - df["fatal_accidents"]).clip(lower=0).astype(int)
    df["minor_accident_count"] = df["injury_accident_count"]  # alias for spec alignment

    # Fatality / injury ratios
    if "fatality_ratio" not in df.columns:
        df["fatality_ratio"] = (df["deaths"] / df["accident_count"].replace(0, 1)).round(4)
    if "injury_ratio" not in df.columns:
        df["injury_ratio"] = (df["injuries"] / df["accident_count"].replace(0, 1)).round(4)
    if "severity_index" not in df.columns:
        df["severity_index"] = (df["accident_severity"]).round(4)

    # ── District-level aggregated frequency ──
    # accident_frequency: how often this district appears in the dataset (normalized rank)
    district_counts = df.groupby("district")["accident_count"].transform("sum")
    max_district = district_counts.max() if district_counts.max() > 0 else 1
    df["accident_frequency"] = (district_counts / max_district * 100).round(2)

    # ── Road risk factor (numeric encoding of road_type) ──
    road_risk_map = {
        "National Highway": 90,
        "State Highway": 75,
        "Expressway": 65,
        "Urban Road": 40,
        "Unknown": 50,
    }
    df["road_risk_factor"] = df["road_type"].map(road_risk_map).fillna(50).astype(float)

    # ── Time risk factor ──
    df["time_risk_factor"] = np.where(df["is_night"] == 1, 80.0, 40.0)

    # ── Final feature selection ──
    feature_cols = [
        "year", "month", "month_sin", "month_cos",
        "season", "is_monsoon", "is_night", "is_highway",
        "district", "jurisdiction", "road_type", "primary_cause", "time_period",
        "accident_count", "accident_frequency",
        "fatal_accident_count", "injury_accident_count", "minor_accident_count",
        "deaths", "injuries",
        "accident_severity", "fatality_ratio", "injury_ratio", "severity_index",
        "road_risk_factor", "time_risk_factor",
    ]
    # Only include columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    features_df = df[feature_cols].reset_index(drop=True)

    return features_df


# ─── Save ──────────────────────────────────────────────────────────────────────

def save_features(features_df: pd.DataFrame) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Accident features saved: %s", OUTPUT_PARQUET)
    return OUTPUT_PARQUET


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_feature_engineering() -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df_raw = _load_source()
    features_df = engineer_features(df_raw)
    save_features(features_df)

    print("\n" + "=" * 65)
    print("ACCIDENT FEATURE ENGINEERING SUMMARY")
    print("=" * 65)
    print(f"  Source records:         {len(df_raw)}")
    print(f"  Feature records:        {len(features_df)}")
    print(f"  Feature columns:        {len(features_df.columns)}")
    print(f"  Output Parquet:         {OUTPUT_PARQUET}")
    print(f"\n  Columns: {list(features_df.columns)}")
    print(f"\n  Districts (unique):     {features_df['district'].nunique()}")
    print(f"  Road types (unique):    {features_df['road_type'].nunique()}")
    print(f"  Cause types (unique):   {features_df['primary_cause'].nunique()}")
    print(f"  Year range:             {features_df['year'].min()}–{features_df['year'].max()}")
    print("=" * 65 + "\n")

    return features_df


if __name__ == "__main__":
    run_feature_engineering()
