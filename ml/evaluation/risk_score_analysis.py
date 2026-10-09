"""Risk Score Analysis — Phase 4, MahaTraffic AI.

Generates analytical reports from computed risk scores:
  - Risk distribution
  - Top historical high-risk locations
  - Yearly risk trends
  - Monthly risk trends
  - Risk by road type

Input:  data/processed/features/risk_scores.parquet
Output: printed report + summary dict

NOTE: All results are derived from historical observed accident records.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.risk_analysis")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RISK_PARQUET = BASE_DIR / "data" / "processed" / "features" / "risk_scores.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "features"
REPORT_PATH = OUTPUT_DIR / "risk_analysis_report.json"


def load_risk_scores() -> pd.DataFrame:
    if not RISK_PARQUET.exists():
        raise FileNotFoundError(
            f"risk_scores.parquet not found at {RISK_PARQUET}. "
            "Run ml/prediction/risk_score.py first."
        )
    return pd.read_parquet(RISK_PARQUET, engine="pyarrow")


def risk_distribution(df: pd.DataFrame) -> dict:
    """Count and percentage of each risk level."""
    dist = df["risk_level"].value_counts().to_dict()
    total = len(df)
    return {
        level: {"count": int(cnt), "pct": round(cnt / total * 100, 2)}
        for level, cnt in dist.items()
    }


def top_high_risk_locations(df: pd.DataFrame, top_n: int = 10) -> list[dict]:
    """Districts with highest mean risk score (historical)."""
    agg = (
        df.groupby("district")
        .agg(
            mean_risk=("risk_score", "mean"),
            max_risk=("risk_score", "max"),
            total_accidents=("accident_count", "sum"),
            total_deaths=("deaths", "sum"),
            record_count=("risk_score", "count"),
        )
        .round(2)
        .sort_values("mean_risk", ascending=False)
        .head(top_n)
        .reset_index()
    )
    return agg.to_dict(orient="records")


def yearly_risk_trend(df: pd.DataFrame) -> list[dict]:
    """Mean risk score per year."""
    trend = (
        df.groupby("year")
        .agg(
            mean_risk=("risk_score", "mean"),
            high_risk_count=("risk_level", lambda x: (x == "HIGH").sum()),
            total_accidents=("accident_count", "sum"),
        )
        .round(2)
        .reset_index()
    )
    return trend.to_dict(orient="records")


def monthly_risk_trend(df: pd.DataFrame) -> list[dict]:
    """Mean risk score per month across all years."""
    trend = (
        df.groupby("month")
        .agg(mean_risk=("risk_score", "mean"), total_accidents=("accident_count", "sum"))
        .round(2)
        .reset_index()
    )
    month_names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    trend["month_name"] = trend["month"].map(month_names)
    return trend.to_dict(orient="records")


def risk_by_road_type(df: pd.DataFrame) -> list[dict]:
    """Mean risk score grouped by road type."""
    if "road_type" not in df.columns:
        return []
    agg = (
        df.groupby("road_type")
        .agg(mean_risk=("risk_score", "mean"), record_count=("risk_score", "count"))
        .round(2)
        .sort_values("mean_risk", ascending=False)
        .reset_index()
    )
    return agg.to_dict(orient="records")


def risk_by_cause(df: pd.DataFrame) -> list[dict]:
    """Mean risk score grouped by primary cause."""
    if "primary_cause" not in df.columns:
        return []
    agg = (
        df.groupby("primary_cause")
        .agg(mean_risk=("risk_score", "mean"), record_count=("risk_score", "count"))
        .round(2)
        .sort_values("mean_risk", ascending=False)
        .reset_index()
    )
    return agg.to_dict(orient="records")


def run_analysis() -> dict:
    """Execute full risk score analysis pipeline."""
    df = load_risk_scores()
    logger.info("Loaded %d risk score records.", len(df))

    report = {
        "total_records": len(df),
        "score_range": {"min": float(df["risk_score"].min()), "max": float(df["risk_score"].max())},
        "mean_score": round(float(df["risk_score"].mean()), 2),
        "risk_distribution": risk_distribution(df),
        "top_high_risk_locations": top_high_risk_locations(df),
        "yearly_risk_trend": yearly_risk_trend(df),
        "monthly_risk_trend": monthly_risk_trend(df),
        "risk_by_road_type": risk_by_road_type(df),
        "risk_by_cause": risk_by_cause(df),
    }

    # Save report
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Risk analysis report saved: %s", REPORT_PATH)

    return report


def print_report(report: dict) -> None:
    print("\n" + "=" * 70)
    print("MAHATRAFFIC AI — RISK SCORE ANALYSIS REPORT")
    print("(Analytical scores derived from historical accident patterns)")
    print("=" * 70)
    print(f"  Total records:   {report['total_records']}")
    print(f"  Score range:     {report['score_range']['min']:.2f} – {report['score_range']['max']:.2f}")
    print(f"  Mean score:      {report['mean_score']:.2f}")

    print("\n  Risk Level Distribution:")
    for level, data in report["risk_distribution"].items():
        bar = "█" * int(data["pct"] / 5)
        print(f"    {level:<10} {data['count']:>5} records  ({data['pct']:.1f}%)  {bar}")

    print("\n  Top 10 High-Risk Districts (historical mean risk score):")
    for i, loc in enumerate(report["top_high_risk_locations"], 1):
        print(f"    {i:>2}. {loc['district']:<22} mean={loc['mean_risk']:.2f}  "
              f"deaths={loc['total_deaths']}")

    print("\n  Yearly Risk Trend:")
    for row in report["yearly_risk_trend"]:
        print(f"    {row['year']}:  mean_risk={row['mean_risk']:.2f}  "
              f"high_risk_records={row['high_risk_count']}")

    print("\n  Monthly Risk Trend (all years averaged):")
    for row in report["monthly_risk_trend"]:
        bar = "▓" * int(row["mean_risk"] / 5)
        print(f"    {row.get('month_name', row['month']):<5}  {row['mean_risk']:.2f}  {bar}")

    print("\n  Risk by Road Type:")
    for row in report["risk_by_road_type"]:
        print(f"    {row['road_type']:<25} mean={row['mean_risk']:.2f}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    report = run_analysis()
    print_report(report)
