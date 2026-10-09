"""MCP Tool implementations for historical accident statistics."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent.parent


def get_city_accident_statistics(city_name: str, year: Optional[int] = None) -> Dict[str, Any]:
    """Retrieve historical accident statistics for a specified city/district.

    Args:
        city_name: Maharashtra district or city name.
        year: Specific calendar year (2019-2023) filter.

    Returns:
        Structured historical accident summary or clear disclosure if not found.
    """
    from backend.app.services.intelligence_service import get_accident_statistics

    stats = get_accident_statistics(district=city_name, year=year)
    if "error" in stats:
        return {
            "city_name": city_name,
            "district_found": False,
            "year": year,
            "message": f"No historical records found for district '{city_name}' in the 2019-2023 dataset.",
            "data_period": "2019-2023",
            "source": "maharashtra_district_accidents_2019_2023",
        }

    return {
        "city_name": city_name,
        "district_found": True,
        "year": year,
        "records_count": stats.get("records_count", 0),
        "total_accidents": stats.get("total_accidents", 0),
        "total_deaths": stats.get("total_deaths", 0),
        "total_injuries": stats.get("total_injuries", 0),
        "total_fatal_accidents": stats.get("total_fatal_accidents", 0),
        "top_causes": stats.get("top_causes", {}),
        "accidents_by_year": stats.get("accidents_by_year", {}),
        "data_period": stats.get("year_range", "2019-2023"),
        "source": stats.get("data_source", "maharashtra_district_accidents_2019_2023"),
        "disclaimer": "Historical records only. Not an emergency or real-time assessment.",
    }


def get_monthly_accident_statistics(year: Optional[int] = None, district: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve monthly accident seasonality patterns across Maharashtra.

    Args:
        year: Optional calendar year (2019-2023).
        district: Optional district filter.

    Returns:
        Dictionary of monthly totals and monsoon/non-monsoon comparisons.
    """
    parquet = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
    csv = BASE_DIR / "data" / "processed" / "accidents_clean.csv"

    if parquet.exists():
        df = pd.read_parquet(parquet, engine="pyarrow")
    elif csv.exists():
        df = pd.read_csv(csv)
    else:
        return {"error": "Accident dataset not available."}

    if district:
        df = df[df["district"].str.lower() == district.lower()]
    if year:
        df = df[df["year"] == year]

    if df.empty:
        return {
            "district": district,
            "year": year,
            "records_found": 0,
            "message": "No data found matching the specified district/year criteria."
        }

    monthly_summary = (
        df.groupby("month")
        .agg(
            total_accidents=("accident_count", "sum"),
            total_deaths=("deaths", "sum"),
            total_injuries=("injuries", "sum"),
        )
        .reset_index()
    )
    monthly_data = [
        {
            "month": int(r["month"]),
            "accidents": int(r["total_accidents"]),
            "deaths": int(r["total_deaths"]),
            "injuries": int(r["total_injuries"]),
        }
        for _, r in monthly_summary.iterrows()
    ]

    # Monsoon (June-September) breakdown
    monsoon_months = [6, 7, 8, 9]
    monsoon_df = df[df["month"].isin(monsoon_months)]
    non_monsoon_df = df[~df["month"].isin(monsoon_months)]

    return {
        "year_filter": year,
        "district_filter": district,
        "records_count": len(df),
        "monthly_breakdown": monthly_data,
        "monsoon_accidents": int(monsoon_df["accident_count"].sum()),
        "monsoon_deaths": int(monsoon_df["deaths"].sum()),
        "non_monsoon_accidents": int(non_monsoon_df["accident_count"].sum()),
        "non_monsoon_deaths": int(non_monsoon_df["deaths"].sum()),
        "source": "maharashtra_district_accidents_2019_2023",
    }


def get_yearly_accident_statistics(district: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve multi-year accident trends (2019-2023)."""
    parquet = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
    if not parquet.exists():
        return {"error": "Clean accident dataset not found."}

    df = pd.read_parquet(parquet, engine="pyarrow")
    if district:
        df = df[df["district"].str.lower() == district.lower()]

    yearly = (
        df.groupby("year")
        .agg(
            total_accidents=("accident_count", "sum"),
            total_deaths=("deaths", "sum"),
            total_injuries=("injuries", "sum"),
        )
        .reset_index()
    )
    return {
        "district": district,
        "yearly_trend": [
            {
                "year": int(r["year"]),
                "accidents": int(r["total_accidents"]),
                "deaths": int(r["total_deaths"]),
                "injuries": int(r["total_injuries"]),
            }
            for _, r in yearly.iterrows()
        ],
        "source": "maharashtra_district_accidents_2019_2023",
    }
