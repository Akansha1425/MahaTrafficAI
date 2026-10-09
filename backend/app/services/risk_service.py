"""Risk Assessment Service — MahaTraffic AI.

Implements the risk scoring formula and ML model inference.
Formula: Risk Score = (0.40 × Severity) + (0.25 × Frequency) + (0.20 × Road Factor) + (0.15 × Time Factor)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import logging
import pickle
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MODEL_PATH = BASE_DIR / "ml" / "models" / "risk_classifier.pkl"
ENCODERS_PATH = BASE_DIR / "ml" / "models" / "label_encoders.pkl"
METRICS_PATH = BASE_DIR / "ml" / "models" / "metrics.json"
IMPORTANCES_PATH = BASE_DIR / "ml" / "models" / "feature_importances.json"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
ACCIDENTS_PATH = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"

_ROAD_FACTOR_MAP = {
    "national highway": 90.0,
    "state highway": 75.0,
    "expressway": 65.0,
    "urban / city road": 40.0,
    "urban / expressway": 65.0,
}

_TIME_FACTOR_MAP = {
    "night": 80.0,
    "late night": 85.0,
    "day": 40.0,
    "peak hour": 55.0,
}


class RiskService:
    """Service facade for statistical risk scoring and Random Forest ML model inference."""

    _instance: Optional["RiskService"] = None
    _model = None
    _encoders: Optional[dict] = None
    _metrics: Optional[dict] = None
    _importances: Optional[dict] = None
    _accidents_df: Optional[pd.DataFrame] = None

    def __init__(self):
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load ML model, encoders, and processed data on first use."""
        try:
            if MODEL_PATH.exists():
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("ML model loaded from: %s", MODEL_PATH)

            if ENCODERS_PATH.exists():
                with open(ENCODERS_PATH, "rb") as f:
                    self._encoders = pickle.load(f)
                logger.info("Label encoders loaded.")

            if METRICS_PATH.exists():
                with open(METRICS_PATH, "r") as f:
                    self._metrics = json.load(f)

            if IMPORTANCES_PATH.exists():
                with open(IMPORTANCES_PATH, "r") as f:
                    self._importances = json.load(f)

            if ACCIDENTS_PATH.exists():
                self._accidents_df = pd.read_parquet(ACCIDENTS_PATH, engine="pyarrow")
                logger.info("Accidents data loaded: %d records.", len(self._accidents_df))

        except Exception as e:
            logger.warning("Non-critical artifact load error: %s", e)

    # ── Formula-based risk score ────────────────────────────────────────────

    def calculate_risk_score(
        self,
        accident_count: int,
        fatal_accidents: int,
        deaths: int,
        injuries: int,
        road_type: str = "urban / city road",
        time_period: str = "day",
        max_accidents_ref: int = 500,
    ) -> Dict[str, Any]:
        """Calculate composite risk score using the established formula.

        Risk Score = (0.40 × Severity) + (0.25 × Frequency) + (0.20 × Road Factor) + (0.15 × Time Factor)
        """
        # Severity component (0-100)
        sev_raw = (deaths * 1.0 + injuries * 0.3) / max(accident_count, 1)
        max_sev = 2.5  # calibration constant
        severity_score = min((sev_raw / max_sev) * 100, 100)

        # Frequency component (0-100)
        frequency_score = min((accident_count / max_accidents_ref) * 100, 100)

        # Road factor
        road_key = road_type.lower().strip()
        road_factor = _ROAD_FACTOR_MAP.get(road_key, 50.0)

        # Time factor
        time_key = time_period.lower().strip()
        time_factor = _TIME_FACTOR_MAP.get(time_key, 50.0)

        # Composite score
        score = (
            0.40 * severity_score +
            0.25 * frequency_score +
            0.20 * road_factor +
            0.15 * time_factor
        )
        score = round(float(score), 2)

        if score >= 67:
            category = "HIGH"
        elif score >= 34:
            category = "MEDIUM"
        else:
            category = "LOW"

        return {
            "risk_score": score,
            "risk_category": category,
            "components": {
                "severity_score": round(severity_score, 2),
                "frequency_score": round(frequency_score, 2),
                "road_factor": road_factor,
                "time_factor": time_factor,
            },
            "formula": "0.40×Severity + 0.25×Frequency + 0.20×RoadFactor + 0.15×TimeFactor",
            "inputs": {
                "accident_count": accident_count,
                "fatal_accidents": fatal_accidents,
                "deaths": deaths,
                "injuries": injuries,
                "road_type": road_type,
                "time_period": time_period,
            },
        }

    # ── ML Model inference ───────────────────────────────────────────────────

    def predict_risk_category(self, feature_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Perform inference using the trained Random Forest model."""
        if self._model is None or self._encoders is None:
            return {
                "error": "ML model not available. Run ml/training/train_model.py first.",
                "fallback": "Use calculate_risk_score() for formula-based scoring.",
            }

        try:
            # Encode categoricals
            enc_district = self._encoders["district"].transform(
                [str(feature_payload.get("district", "Pune"))]
            )[0]
            enc_road = self._encoders["road_type"].transform(
                [str(feature_payload.get("road_type", "Urban / City Road"))]
            )[0]
            enc_cause = self._encoders["primary_cause"].transform(
                [str(feature_payload.get("primary_cause", "Over-speeding"))]
            )[0]

            row = pd.DataFrame([{
                "year": int(feature_payload.get("year", 2023)),
                "month": int(feature_payload.get("month", 6)),
                "month_sin": round(np.sin(2 * np.pi * int(feature_payload.get("month", 6)) / 12), 4),
                "month_cos": round(np.cos(2 * np.pi * int(feature_payload.get("month", 6)) / 12), 4),
                "is_monsoon": int(feature_payload.get("is_monsoon", 0)),
                "is_night": int(feature_payload.get("is_night", 0)),
                "is_highway": int(feature_payload.get("is_highway", 0)),
                "accident_count": int(feature_payload.get("accident_count", 50)),
                "fatal_accidents": int(feature_payload.get("fatal_accidents", 5)),
                "deaths": int(feature_payload.get("deaths", 5)),
                "injuries": int(feature_payload.get("injuries", 30)),
                "fatality_ratio": float(feature_payload.get("fatality_ratio", 0.1)),
                "injury_ratio": float(feature_payload.get("injury_ratio", 0.6)),
                "severity_index": float(feature_payload.get("severity_index", 0.5)),
                "district": enc_district,
                "road_type": enc_road,
                "primary_cause": enc_cause,
            }])

            pred_encoded = self._model.predict(row)[0]
            pred_proba = self._model.predict_proba(row)[0]
            pred_label = self._encoders["target"].inverse_transform([pred_encoded])[0]
            class_names = self._encoders["target"].classes_.tolist()

            return {
                "predicted_category": pred_label,
                "confidence": round(float(max(pred_proba)), 4),
                "class_probabilities": {
                    cls: round(float(prob), 4)
                    for cls, prob in zip(class_names, pred_proba)
                },
                "model": "RandomForestClassifier(n_estimators=200)",
                "disclaimer": "ANALYTICAL TOOL: Based on historical patterns only.",
            }

        except Exception as e:
            logger.error("ML inference error: %s", e)
            return {"error": str(e)}

    # ── District analytics ────────────────────────────────────────────────────

    def get_district_risk_summary(self, district: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get risk summary per district aggregated from historical data."""
        if self._accidents_df is None:
            return []

        df = self._accidents_df.copy()
        if district:
            df = df[df["district"].str.lower() == district.lower()]

        agg = (
            df.groupby("district")
            .agg(
                total_accidents=("accident_count", "sum"),
                total_deaths=("deaths", "sum"),
                total_injuries=("injuries", "sum"),
                fatality_ratio=("fatality_ratio", "mean"),
            )
            .reset_index()
        )

        results = []
        for _, row in agg.iterrows():
            score_result = self.calculate_risk_score(
                accident_count=int(row["total_accidents"]),
                fatal_accidents=int(row["total_deaths"] * 0.9),
                deaths=int(row["total_deaths"]),
                injuries=int(row["total_injuries"]),
                max_accidents_ref=int(agg["total_accidents"].max()),
            )
            results.append({
                "district": row["district"],
                "total_accidents": int(row["total_accidents"]),
                "total_deaths": int(row["total_deaths"]),
                "total_injuries": int(row["total_injuries"]),
                "avg_fatality_ratio": round(float(row["fatality_ratio"]), 4),
                "risk_score": score_result["risk_score"],
                "risk_category": score_result["risk_category"],
            })

        results.sort(key=lambda x: x["risk_score"], reverse=True)
        return results

    def get_model_metrics(self) -> Dict[str, Any]:
        """Return ML model evaluation metrics."""
        if self._metrics is None:
            return {"error": "Metrics not available. Run ml/training/train_model.py first."}
        return {
            "accuracy": self._metrics.get("accuracy"),
            "f1_macro": self._metrics.get("f1_macro"),
            "f1_weighted": self._metrics.get("f1_weighted"),
            "cv_accuracy_mean": self._metrics.get("cv_accuracy_mean"),
            "cv_accuracy_std": self._metrics.get("cv_accuracy_std"),
            "train_samples": self._metrics.get("train_samples"),
            "test_samples": self._metrics.get("test_samples"),
        }

    def get_feature_importances(self) -> Dict[str, float]:
        """Return top feature importances from the trained model."""
        if self._importances is None:
            return {}
        return self._importances

    def get_yearly_trend(self) -> List[Dict[str, Any]]:
        """Return year-by-year accident trend."""
        if self._accidents_df is None:
            return []
        df = self._accidents_df
        agg = (
            df.groupby("year")
            .agg(
                total_accidents=("accident_count", "sum"),
                total_deaths=("deaths", "sum"),
                total_injuries=("injuries", "sum"),
                fatal_accidents=("fatal_accidents", "sum"),
            )
            .reset_index()
        )
        agg["fatality_rate"] = (agg["fatal_accidents"] / agg["total_accidents"] * 100).round(2)
        return agg.to_dict(orient="records")


# ── Singleton ──────────────────────────────────────────────────────────────────

_service: Optional[RiskService] = None


def get_risk_service() -> RiskService:
    """Get singleton RiskService instance."""
    global _service
    if _service is None:
        _service = RiskService()
    return _service
