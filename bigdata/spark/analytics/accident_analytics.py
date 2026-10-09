"""Local Big Data batch analytics engine.

Executes columnar aggregations, spatiotemporal grouping, and multi-factor
analytical risk ranking over processed Parquet files without cluster dependencies.
"""

from pathlib import Path
from typing import Any, Dict
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analytics.accidents")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PARQUET_FILE = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"


class AccidentAnalytics:
    """Batch analytical engine querying historical Parquet accident datasets."""

    def __init__(self, parquet_path: Path = PARQUET_FILE):
        if not parquet_path.exists():
            raise FileNotFoundError(f"Cleaned Parquet file missing at: {parquet_path}")
        self.parquet_path = parquet_path
        # Demonstrate efficient columnar reading via PyArrow
        logger.info("Reading processed accident Parquet from: %s", parquet_path)
        self.df = pd.read_parquet(parquet_path, engine="pyarrow")

    def get_yearly_accidents(self) -> pd.DataFrame:
        """Aggregate total accidents, fatalities, and injuries by year."""
        yearly = (
            self.df.groupby("year")
            .agg(
                total_accidents=("accident_count", "sum"),
                fatal_accidents=("fatal_accidents", "sum"),
                total_deaths=("deaths", "sum"),
                total_injuries=("injuries", "sum"),
            )
            .reset_index()
            .sort_values("year")
        )
        yearly["fatality_rate_%"] = (yearly["total_deaths"] / yearly["total_accidents"] * 100).round(2)
        return yearly

    def get_monthly_seasonality(self) -> pd.DataFrame:
        """Aggregate monthly accident distributions across multi-year period."""
        monthly = (
            self.df.groupby("month")
            .agg(
                mean_accidents=("accident_count", "mean"),
                total_accidents=("accident_count", "sum"),
                total_deaths=("deaths", "sum"),
            )
            .reset_index()
            .sort_values("month")
        )
        monthly["mean_accidents"] = monthly["mean_accidents"].round(1)
        return monthly

    def get_location_analysis(self) -> pd.DataFrame:
        """Aggregate accident totals and rankings across Maharashtra districts."""
        locations = (
            self.df.groupby("district")
            .agg(
                total_accidents=("accident_count", "sum"),
                fatal_accidents=("fatal_accidents", "sum"),
                deaths=("deaths", "sum"),
                injuries=("injuries", "sum"),
            )
            .reset_index()
            .sort_values("total_accidents", ascending=False)
        )
        return locations

    def get_road_type_analysis(self) -> pd.DataFrame:
        """Aggregate accident frequency and mortality grouped by road classification."""
        road_types = (
            self.df.groupby("road_type")
            .agg(
                accident_count=("accident_count", "sum"),
                deaths=("deaths", "sum"),
                injuries=("injuries", "sum"),
            )
            .reset_index()
            .sort_values("accident_count", ascending=False)
        )
        road_types["fatality_rate_%"] = (road_types["deaths"] / road_types["accident_count"] * 100).round(2)
        return road_types

    def calculate_project_risk_ranking(self) -> pd.DataFrame:
        """Calculate Project Analytical Risk Ranking across Maharashtra districts.

        Analytical Formula (Project-Defined, NOT an official government score):
        Risk Score = 0.40 * Severity + 0.25 * Frequency + 0.20 * Road Factor + 0.15 * Time Factor
        Normalized to 0 - 100 scale:
        - 0 to 33: LOW
        - 34 to 66: MEDIUM
        - 67 to 100: HIGH
        """
        loc_df = self.get_location_analysis()

        # Component 1: Accident Severity (Deaths & Fatalities)
        raw_severity = loc_df["deaths"] * 1.0 + loc_df["injuries"] * 0.3
        sev_min, sev_max = raw_severity.min(), raw_severity.max()
        norm_severity = (raw_severity - sev_min) / (sev_max - sev_min + 1e-6) * 100

        # Component 2: Accident Frequency
        raw_freq = loc_df["total_accidents"]
        freq_min, freq_max = raw_freq.min(), raw_freq.max()
        norm_freq = (raw_freq - freq_min) / (freq_max - freq_min + 1e-6) * 100

        # Component 3: Road Risk Factor (Ratio of Highway/Expressway accidents)
        highway_acc = (
            self.df[self.df["road_type"].str.contains("Highway|Expressway", case=False, na=False)]
            .groupby("district")["accident_count"]
            .sum()
        )
        road_risk_ratio = loc_df["district"].map(highway_acc).fillna(0) / loc_df["total_accidents"]
        norm_road = road_risk_ratio * 100

        # Component 4: Time Risk Factor (Night-time accident vulnerability)
        night_acc = (
            self.df[self.df["time_period"].str.contains("Night", case=False, na=False)]
            .groupby("district")["accident_count"]
            .sum()
        )
        time_ratio = loc_df["district"].map(night_acc).fillna(0) / loc_df["total_accidents"]
        norm_time = time_ratio * 100

        # Weighted Composite Score
        composite_score = (
            0.40 * norm_severity
            + 0.25 * norm_freq
            + 0.20 * norm_road
            + 0.15 * norm_time
        ).round(2)

        loc_df["project_risk_score"] = composite_score

        # Category mapping
        def assign_category(score: float) -> str:
            if score <= 33.0:
                return "LOW"
            elif score <= 66.0:
                return "MEDIUM"
            else:
                return "HIGH"

        loc_df["project_risk_level"] = loc_df["project_risk_score"].apply(assign_category)
        ranking = loc_df.sort_values("project_risk_score", ascending=False).reset_index(drop=True)
        ranking.index = ranking.index + 1
        ranking.index.name = "analytical_rank"
        return ranking


def run_analytics_suite() -> None:
    """Execute complete analytics suite and present verified outputs."""
    analytics = AccidentAnalytics()

    print("\n" + "=" * 70)
    print("MAHATRAFFIC AI — HISTORICAL ACCIDENT BATCH ANALYTICS")
    print("=" * 70)

    print("\n--- 1. YEARLY ACCIDENT TRENDS ---")
    yearly = analytics.get_yearly_accidents()
    print(yearly.to_string(index=False))

    print("\n--- 2. MONTHLY SEASONALITY (ALL YEARS) ---")
    monthly = analytics.get_monthly_seasonality()
    print(monthly.to_string(index=False))

    print("\n--- 3. ROAD TYPE DISTRIBUTION ---")
    road_types = analytics.get_road_type_analysis()
    print(road_types.to_string(index=False))

    print("\n--- 4. TOP 10 PROJECT ANALYTICAL RISK LOCATIONS ---")
    print("NOTICE: Project-level analytical ranking for research; NOT an official government score.")
    ranking = analytics.calculate_project_risk_ranking()
    display_cols = ["district", "total_accidents", "deaths", "injuries", "project_risk_score", "project_risk_level"]
    print(ranking[display_cols].head(10).to_string())

    print("\n" + "=" * 70)
    print("Analytics Completed Successfully.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_analytics_suite()
