"""Risk Model Prediction API — Phase 4, MahaTraffic AI.

Loads the trained joblib risk model and returns structured predictions:

    {
        "risk_score": float,
        "risk_level": "LOW" | "MEDIUM" | "HIGH",
        "confidence": float,
        "top_factors": [{"feature": str, "importance": float}, ...]
    }

NOTE: These predictions represent HISTORICAL RISK CLASSIFICATION
      based on observed historical patterns. They do NOT predict
      exact future accident occurrence.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger("ml.predict_risk")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
ENCODERS_PATH = MODEL_DIR / "risk_model_encoders.joblib"
IMPORTANCES_PATH = MODEL_DIR / "risk_model_feature_importances.json"

NUMERIC_FEATURES = [
    "year", "month", "month_sin", "month_cos",
    "is_monsoon", "is_night", "is_highway",
    "accident_count", "fatal_accidents", "deaths", "injuries",
    "fatality_ratio", "injury_ratio", "severity_index",
]
CATEGORICAL_FEATURES = ["district", "road_type", "primary_cause"]


# ─── Singleton loader ─────────────────────────────────────────────────────────

_model = None
_encoders = None
_importances = None


def _load_artifacts() -> tuple:
    global _model, _encoders, _importances
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run train_risk_model.py first."
            )
        _model = joblib.load(MODEL_PATH)
        _encoders = joblib.load(ENCODERS_PATH)
        if IMPORTANCES_PATH.exists():
            with open(IMPORTANCES_PATH, encoding="utf-8") as f:
                _importances = json.load(f)
        else:
            _importances = {}
        logger.info("Risk model loaded from: %s", MODEL_PATH)
    return _model, _encoders, _importances


# ─── Prediction ───────────────────────────────────────────────────────────────

def predict_risk(
    year: int,
    month: int,
    district: str,
    road_type: str,
    primary_cause: str,
    accident_count: int = 100,
    fatal_accidents: int = 10,
    deaths: int = 12,
    injuries: int = 40,
    time_period: str = "Day (06:00-18:00)",
    season: str = "Unknown",
) -> dict:
    """Return a risk classification for the given input parameters.

    Returns:
        {
            "risk_score": float (0–100, project-defined analytical score),
            "risk_level": "LOW" | "MEDIUM" | "HIGH",
            "confidence": float (0–1, model class probability),
            "top_factors": [{"feature": str, "importance": float}, ...]
        }
    """
    clf, encoders, importances = _load_artifacts()

    # Derived features
    month_sin = round(float(np.sin(2 * np.pi * month / 12)), 4)
    month_cos = round(float(np.cos(2 * np.pi * month / 12)), 4)
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Summer", 4: "Summer", 5: "Summer",
        6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
        10: "Post-Monsoon", 11: "Post-Monsoon",
    }
    effective_season = season_map.get(month, "Unknown")
    is_monsoon = int(effective_season == "Monsoon")
    is_night = int("Night" in time_period)
    is_highway = int(any(t in road_type for t in ["National Highway", "State Highway", "Expressway"]))
    fatality_ratio = round(deaths / max(accident_count, 1), 4)
    injury_ratio = round(injuries / max(accident_count, 1), 4)
    severity_index = round((deaths * 1.0 + injuries * 0.3) / max(accident_count, 1), 4)

    row = {
        "year": year, "month": month, "month_sin": month_sin, "month_cos": month_cos,
        "is_monsoon": is_monsoon, "is_night": is_night, "is_highway": is_highway,
        "accident_count": accident_count, "fatal_accidents": fatal_accidents,
        "deaths": deaths, "injuries": injuries,
        "fatality_ratio": fatality_ratio, "injury_ratio": injury_ratio,
        "severity_index": severity_index,
        "district": district, "road_type": road_type, "primary_cause": primary_cause,
    }

    df_input = pd.DataFrame([row])

    # Encode categoricals
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        known = set(le.classes_)
        val = df_input[col].iloc[0]
        if val not in known:
            val = le.classes_[0]
        df_input[col] = le.transform([val])

    X = df_input[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    pred_encoded = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    pred_label = encoders["target"].inverse_transform([pred_encoded])[0]
    confidence = round(float(proba.max()), 4)

    # Risk score (project formula)
    road_factor = (
        90.0 if "National Highway" in road_type else
        75.0 if "State Highway" in road_type else
        65.0 if "Expressway" in road_type else 40.0
    )
    time_factor = 80.0 if is_night else 40.0
    # Simplified normalization (single-record)
    sev_norm = min(severity_index * 20, 100)
    frq_norm = min(accident_count / 10, 100)
    risk_score = round(
        0.40 * sev_norm + 0.25 * frq_norm + 0.20 * road_factor + 0.15 * time_factor,
        2,
    )
    risk_score = min(max(risk_score, 0), 100)

    top_factors = [
        {"feature": k, "importance": round(v, 4)}
        for k, v in list(importances.items())[:5]
    ]

    return {
        "risk_score": risk_score,
        "risk_level": pred_label,
        "confidence": confidence,
        "top_factors": top_factors,
        "disclaimer": (
            "Historical risk classification based on observed historical patterns. "
            "Does not predict exact future accident occurrence."
        ),
    }


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    test_cases = [
        {
            "year": 2023, "month": 7, "district": "Pune",
            "road_type": "National Highway", "primary_cause": "Over-speeding",
            "accident_count": 300, "fatal_accidents": 80, "deaths": 90, "injuries": 200,
            "time_period": "Night (18:00-06:00)",
        },
        {
            "year": 2023, "month": 3, "district": "Mumbai",
            "road_type": "Urban Road", "primary_cause": "Signal Jumping",
            "accident_count": 50, "fatal_accidents": 3, "deaths": 3, "injuries": 20,
            "time_period": "Day (06:00-18:00)",
        },
    ]

    print("\n" + "=" * 70)
    print("RISK PREDICTION DEMO")
    print("=" * 70)
    for i, tc in enumerate(test_cases, 1):
        result = predict_risk(**tc)
        print(f"\n  Test Case {i}: {tc['district']} | {tc['road_type']} | Month={tc['month']}")
        print(f"    Risk Score:  {result['risk_score']}")
        print(f"    Risk Level:  {result['risk_level']}")
        print(f"    Confidence:  {result['confidence']:.2%}")
        print(f"    Top Factors: {[f['feature'] for f in result['top_factors']]}")
    print("=" * 70 + "\n")
