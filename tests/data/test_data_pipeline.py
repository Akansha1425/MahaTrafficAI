"""Unit tests for MahaTraffic AI data ingestion, cleaning, and Parquet pipeline."""

from pathlib import Path
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def test_raw_datasets_exist():
    """Verify raw accident and social media CSV files exist and are not empty."""
    acc_path = BASE_DIR / "data" / "raw" / "accidents" / "maharashtra_district_accidents_2019_2023.csv"
    soc_path = BASE_DIR / "data" / "raw" / "social_media" / "maharashtra_road_safety_public_posts.csv"

    assert acc_path.exists(), "Raw accident dataset is missing"
    assert soc_path.exists(), "Raw social dataset is missing"

    df_acc = pd.read_csv(acc_path)
    df_soc = pd.read_csv(soc_path)

    assert len(df_acc) > 1000, "Accident dataset should have > 1000 rows"
    assert len(df_soc) > 500, "Social dataset should have > 500 rows"


def test_processed_parquet_files():
    """Verify clean Parquet files are generated with proper schema."""
    acc_parquet = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
    soc_parquet = BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"

    assert acc_parquet.exists(), "Cleaned accident parquet file is missing"
    assert soc_parquet.exists(), "Cleaned social parquet file is missing"

    df_acc = pd.read_parquet(acc_parquet)
    df_soc = pd.read_parquet(soc_parquet)

    # Check key columns
    expected_acc_cols = {"year", "month", "district", "accident_count", "fatal_accidents", "deaths", "injuries"}
    assert expected_acc_cols.issubset(set(df_acc.columns))

    expected_soc_cols = {"post_id", "timestamp", "location", "text", "category_tag"}
    assert expected_soc_cols.issubset(set(df_soc.columns))


def test_feature_matrix_pipeline():
    """Verify the risk feature matrix is non-empty and contains valid risk target."""
    feat_path = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
    assert feat_path.exists(), "Risk features parquet is missing"

    df_feat = pd.read_parquet(feat_path)
    assert len(df_feat) > 0
    assert "risk_category" in df_feat.columns
    assert set(df_feat["risk_category"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})
