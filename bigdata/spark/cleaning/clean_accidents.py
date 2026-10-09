"""Accident data cleaning and Parquet pipeline module.

Executes batch data transformations: column normalization, type casting,
district name standardization, anomaly resolution, and Parquet columnar storage.
Runs directly in Python using Pandas and PyArrow.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cleaning.accidents")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
RAW_FILE = BASE_DIR / "data" / "raw" / "accidents" / "maharashtra_district_accidents_2019_2023.csv"
OUTPUT_PARQUET_DIR = BASE_DIR / "data" / "processed" / "parquet" / "accidents"
OUTPUT_CSV_FILE = BASE_DIR / "data" / "processed" / "accidents_clean.csv"


def clean_accidents_pipeline() -> pd.DataFrame:
    """Run full cleaning and Parquet transformation on raw accident records."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw accidents file not found at: {RAW_FILE}")

    logger.info("Loading raw accident dataset from %s", RAW_FILE)
    df_raw = pd.read_csv(RAW_FILE)
    input_records = len(df_raw)

    # 1. Column normalization (clean whitespace, lowercase, standardized names)
    df = df_raw.copy()
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # 2. Duplicate detection and removal
    duplicates_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    # 3. Numeric conversion and range enforcement
    numeric_fields = ["year", "month", "accident_count", "fatal_accidents", "deaths", "injuries"]
    for field in numeric_fields:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce").fillna(0).astype(int)

    # Range validation: drop non-sensical records if any
    initial_len = len(df)
    df = df[(df["accident_count"] >= 0) & (df["deaths"] >= 0) & (df["injuries"] >= 0)]
    df = df[(df["year"] >= 2000) & (df["year"] <= 2030)]
    df = df[(df["month"] >= 1) & (df["month"] <= 12)]
    removed_records = duplicates_count + (initial_len - len(df))

    # 4. Location & Category Normalization
    # Standardize updated administrative names in Maharashtra
    district_mapping = {
        "Osmanabad": "Dharashiv",
        "Aurangabad": "Chhatrapati Sambhajinagar",
        "Bombay": "Mumbai",
        "Poona": "Pune",
    }
    df["district"] = df["district"].astype(str).str.strip().replace(district_mapping)
    if "police_unit" in df.columns:
        df["police_unit"] = df["police_unit"].astype(str).str.strip()

    # 5. Derived Big Data Analytical Features
    # Safe division helpers
    total_acc = df["accident_count"].replace(0, 1)
    df["fatality_ratio"] = (df["deaths"] / total_acc).round(4)
    df["injury_ratio"] = (df["injuries"] / total_acc).round(4)
    df["severity_index"] = ((df["deaths"] * 1.0 + df["injuries"] * 0.3) / total_acc).round(4)

    # Seasonal categorization
    def assign_season(month: int) -> str:
        if month in [6, 7, 8, 9]:
            return "Monsoon"
        elif month in [10, 11]:
            return "Post-Monsoon"
        elif month in [12, 1, 2]:
            return "Winter"
        else:
            return "Summer"

    df["season"] = df["month"].apply(assign_season)

    # 6. Parquet Columnar Persistence
    OUTPUT_PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = OUTPUT_PARQUET_DIR / "maharashtra_accidents_clean.parquet"
    
    logger.info("Persisting columnar Parquet file to %s", parquet_path)
    df.to_parquet(parquet_path, engine="pyarrow", compression="snappy", index=False)

    # Also persist reference CSV for inspection
    OUTPUT_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_FILE, index=False)

    final_records = len(df)
    transformed_records = final_records

    # Print required reporting metrics
    print("\n" + "=" * 60)
    print("ACCIDENT DATA CLEANING & PARQUET CONVERSION REPORT")
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
    clean_accidents_pipeline()
