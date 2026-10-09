"""Project-Defined Historical Risk Score — Phase 4, MahaTraffic AI.

Implements the analytical risk scoring formula defined for this project:

    Risk Score = 0.40 × Accident Severity
               + 0.25 × Accident Frequency
               + 0.20 × Road Risk Factor
               + 0.15 × Time Risk Factor

    Normalized to 0–100.
    Risk levels: 0–33 LOW | 34–66 MEDIUM | 67–100 HIGH

IMPORTANT DISCLAIMER:
    This is an ANALYTICAL SCORE defined for this academic project.
    It is NOT an official government risk standard.
    It classifies HISTORICAL OBSERVED patterns — it does NOT predict
    exact future accident occurrence.

Saves:
    data/processed/features/risk_scores.parquet
"""

from __future__ import annotations
from pathlib import Path
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.risk_score")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PARQUET = BASE_DIR / "data" / "processed" / "features" / "accident_features.parquet"
FALLBACK_PARQUET = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "features"
OUTPUT_PARQUET = OUTPUT_DIR / "risk_scores.parquet"

# ─── Risk Score Weights (project-defined) ─────────────────────────────────────
W_SEVERITY  = 0.40
W_FREQUENCY = 0.25
W_ROAD      = 0.20
W_TIME      = 0.15

THRESHOLDS = {"LOW": (0, 33), "MEDIUM": (34, 66), "HIGH": (67, 100)}


# ─── Core Functions ───────────────────────────────────────────────────────────

def _normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 100]."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(2)


def calculate_risk_score(
    severity: float | pd.Series,
    frequency: float | pd.Series,
    road_risk_factor: float | pd.Series,
    time_risk_factor: float | pd.Series,
    pre_normalized: bool = False,
) -> pd.Series:
    """Calculate the project-defined historical risk score (0–100).

    Args:
        severity: Accident severity component (raw or pre-normalized 0–100).
        frequency: Accident frequency component (raw or pre-normalized 0–100).
        road_risk_factor: Road risk (0–100 scale, e.g. 90 for NH, 40 for Urban).
        time_risk_factor: Time risk (0–100 scale, e.g. 80 for Night, 40 for Day).
        pre_normalized: If True, inputs are already on 0–100 scale.

    Returns:
        pd.Series of risk scores in [0, 100].
    """
    sev = pd.Series(severity) if not isinstance(severity, pd.Series) else severity
    frq = pd.Series(frequency) if not isinstance(frequency, pd.Series) else frequency
    road = pd.Series(road_risk_factor) if not isinstance(road_risk_factor, pd.Series) else road_risk_factor
    time_ = pd.Series(time_risk_factor) if not isinstance(time_risk_factor, pd.Series) else time_risk_factor

    if not pre_normalized:
        sev = _normalize_0_100(sev)
        frq = _normalize_0_100(frq)
        # road and time are already 0-100 by design

    score = (W_SEVERITY * sev + W_FREQUENCY * frq + W_ROAD * road + W_TIME * time_).clip(0, 100).round(2)
    return score


def classify_risk_level(score: float | pd.Series) -> str | pd.Series:
    """Classify a risk score into LOW / MEDIUM / HIGH.

    Args:
        score: Single value or Series of scores in [0, 100].

    Returns:
        Label or Series of labels.
    """
    if isinstance(score, pd.Series):
        conditions = [
            score <= 33,
            (score > 33) & (score <= 66),
            score > 66,
        ]
        return pd.Series(
            np.select(conditions, ["LOW", "MEDIUM", "HIGH"], default="MEDIUM"),
            index=score.index,
        )
    # scalar
    if score <= 33:
        return "LOW"
    elif score <= 66:
        return "MEDIUM"
    return "HIGH"


# ─── Pipeline ──────────────────────────────────────────────────────────────────

def _load_features() -> pd.DataFrame:
    """Load the accident features Parquet."""
    if FEATURES_PARQUET.exists():
        logger.info("Loading accident_features.parquet")
        return pd.read_parquet(FEATURES_PARQUET, engine="pyarrow")
    if FALLBACK_PARQUET.exists():
        logger.info("Falling back to risk_features.parquet")
        return pd.read_parquet(FALLBACK_PARQUET, engine="pyarrow")
    raise FileNotFoundError(
        "Neither accident_features.parquet nor risk_features.parquet found. "
        "Run feature_engineering.py first."
    )


def build_risk_scores() -> pd.DataFrame:
    """Compute risk scores for all records and save to Parquet."""
    df = _load_features()
    logger.info("Loaded %d feature records.", len(df))

    # Build component columns (normalized 0-100)
    sev_norm = _normalize_0_100(df["accident_severity"] if "accident_severity" in df.columns else df["severity_index"])
    frq_norm = _normalize_0_100(df["accident_frequency"] if "accident_frequency" in df.columns else df["accident_count"].astype(float))

    road_factor = df["road_risk_factor"].values if "road_risk_factor" in df.columns else np.where(
        df["road_type"].str.contains("National Highway", na=False), 90.0,
        np.where(df["road_type"].str.contains("State Highway", na=False), 75.0,
        np.where(df["road_type"].str.contains("Expressway", na=False), 65.0, 40.0))
    )
    time_factor = df["time_risk_factor"].values if "time_risk_factor" in df.columns else np.where(
        df["time_period"].str.contains("Night", na=False), 80.0, 40.0
    )

    df["sev_norm"] = sev_norm.values
    df["frq_norm"] = frq_norm.values
    df["road_factor"] = road_factor
    df["time_factor"] = time_factor

    # Composite score
    df["risk_score"] = calculate_risk_score(
        df["sev_norm"], df["frq_norm"], df["road_factor"], df["time_factor"],
        pre_normalized=True,
    ).values

    # Risk level
    df["risk_level"] = classify_risk_level(df["risk_score"]).values

    # ── Build output DataFrame ──
    out_cols = [
        "year", "month", "district", "road_type", "primary_cause", "time_period",
        "accident_count", "fatal_accident_count", "deaths", "injuries",
        "accident_severity", "accident_frequency",
        "sev_norm", "frq_norm", "road_factor", "time_factor",
        "risk_score", "risk_level",
    ]
    # Only include columns that exist
    out_cols = [c for c in out_cols if c in df.columns]
    out_df = df[out_cols].reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Risk scores saved: %s", OUTPUT_PARQUET)

    return out_df


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = build_risk_scores()

    print("\n" + "=" * 65)
    print("RISK SCORE SUMMARY (Project-Defined Analytical Score)")
    print("=" * 65)
    print(f"  Total records scored:   {len(df)}")
    print(f"  Score range:            {df['risk_score'].min():.2f} – {df['risk_score'].max():.2f}")
    print(f"  Mean score:             {df['risk_score'].mean():.2f}")
    print("\n  Risk Level Distribution:")
    for level, count in df["risk_level"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {level:<10} {count:>5}  ({pct:.1f}%)")
    print("\n  Top 5 High-Risk Districts (mean score):")
    top = df.groupby("district")["risk_score"].mean().nlargest(5).reset_index()
    for _, row in top.iterrows():
        print(f"    {row['district']:<20} {row['risk_score']:.2f}")
    print("=" * 65 + "\n")
