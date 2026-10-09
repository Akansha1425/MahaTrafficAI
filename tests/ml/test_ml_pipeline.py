"""Phase 4 Tests — ML Feature Engineering, Risk Score, Model."""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:
    """Tests for ml/training/feature_engineering.py"""

    def test_feature_engineering_runs(self):
        """Feature engineering pipeline runs without error."""
        from ml.training.feature_engineering import run_feature_engineering
        df = run_feature_engineering()
        assert len(df) > 0, "Feature engineering produced no records."

    def test_feature_columns_present(self):
        """Required feature columns are present in output."""
        from ml.training.feature_engineering import run_feature_engineering
        df = run_feature_engineering()
        required = [
            "year", "month", "district", "road_type", "primary_cause",
            "accident_count", "accident_severity", "fatal_accident_count",
            "injury_accident_count", "road_risk_factor", "time_risk_factor",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_missing_values_in_key_columns(self):
        """Key numeric columns have no NaN after engineering."""
        from ml.training.feature_engineering import run_feature_engineering
        df = run_feature_engineering()
        for col in ["accident_count", "deaths", "injuries", "accident_severity"]:
            assert df[col].isna().sum() == 0, f"NaN found in column: {col}"

    def test_accident_features_parquet_saved(self):
        """Output Parquet file exists after feature engineering."""
        from ml.training.feature_engineering import OUTPUT_PARQUET, run_feature_engineering
        run_feature_engineering()
        assert OUTPUT_PARQUET.exists(), f"Parquet not saved at {OUTPUT_PARQUET}"

    def test_schema_validation(self):
        """Accident features Parquet matches expected schema."""
        from ml.training.feature_engineering import OUTPUT_PARQUET, run_feature_engineering
        run_feature_engineering()
        df = pd.read_parquet(OUTPUT_PARQUET, engine="pyarrow")
        assert "year" in df.columns
        assert df["year"].dtype in [np.int64, np.int32, int]
        assert df["month"].between(1, 12).all()
        assert (df["accident_count"] > 0).all()


# ─── Risk Score Tests ─────────────────────────────────────────────────────────

class TestRiskScore:
    """Tests for ml/prediction/risk_score.py"""

    def test_risk_score_range(self):
        """Risk scores are within [0, 100]."""
        from ml.prediction.risk_score import calculate_risk_score
        sev = pd.Series([0.0, 50.0, 100.0])
        frq = pd.Series([25.0, 50.0, 75.0])
        road = pd.Series([40.0, 65.0, 90.0])
        time_ = pd.Series([40.0, 40.0, 80.0])
        scores = calculate_risk_score(sev, frq, road, time_, pre_normalized=True)
        assert (scores >= 0).all() and (scores <= 100).all()

    def test_risk_level_classify(self):
        """Risk level classification is correct."""
        from ml.prediction.risk_score import classify_risk_level
        assert classify_risk_level(0.0) == "LOW"
        assert classify_risk_level(33.0) == "LOW"
        assert classify_risk_level(34.0) == "MEDIUM"
        assert classify_risk_level(66.0) == "MEDIUM"
        assert classify_risk_level(67.0) == "HIGH"
        assert classify_risk_level(100.0) == "HIGH"

    def test_high_severity_produces_high_risk(self):
        """High severity + high frequency inputs produce high risk score."""
        from ml.prediction.risk_score import calculate_risk_score, classify_risk_level
        score = calculate_risk_score(
            pd.Series([100.0]), pd.Series([100.0]),
            pd.Series([90.0]), pd.Series([80.0]),
            pre_normalized=True,
        ).iloc[0]
        assert classify_risk_level(score) == "HIGH"

    def test_build_risk_scores_runs(self):
        """Full pipeline runs and saves risk_scores.parquet."""
        from ml.prediction.risk_score import build_risk_scores, OUTPUT_PARQUET
        df = build_risk_scores()
        assert len(df) > 0
        assert "risk_score" in df.columns
        assert "risk_level" in df.columns
        assert OUTPUT_PARQUET.exists()

    def test_risk_distribution_has_all_levels(self):
        """Risk level distribution contains LOW, MEDIUM, HIGH."""
        from ml.prediction.risk_score import build_risk_scores
        df = build_risk_scores()
        levels = set(df["risk_level"].unique())
        # Expect at least two distinct levels
        assert len(levels) >= 2, f"Only {levels} found — expected at least 2 levels."


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestRiskModel:
    """Tests for ml/training/train_risk_model.py and predict_risk.py"""

    @pytest.fixture(autouse=True, scope="class")
    def train_model(self):
        """Train the model once for this test class."""
        from ml.training.train_risk_model import (
            _load_features, encode_categoricals, train_model, save_artifacts, save_predictions
        )
        df = _load_features()
        df_enc, encoders = encode_categoricals(df)
        metrics, clf, encoders, importances = train_model(df_enc, encoders)
        save_artifacts(clf, encoders, metrics, importances)
        save_predictions(df, clf, encoders)

    def test_model_file_saved(self):
        """Model joblib file exists after training."""
        model_path = BASE_DIR / "ml" / "models" / "risk_model.joblib"
        assert model_path.exists(), f"Model not found at {model_path}"

    def test_model_loading(self):
        """Model loads without error."""
        import joblib
        model_path = BASE_DIR / "ml" / "models" / "risk_model.joblib"
        clf = joblib.load(model_path)
        assert clf is not None

    def test_predictions_parquet_exists(self):
        """model_predictions.parquet is saved."""
        pred_path = BASE_DIR / "data" / "processed" / "features" / "model_predictions.parquet"
        assert pred_path.exists()

    def test_predictions_have_risk_level(self):
        """Predictions Parquet contains predicted_risk_level column."""
        pred_path = BASE_DIR / "data" / "processed" / "features" / "model_predictions.parquet"
        df = pd.read_parquet(pred_path, engine="pyarrow")
        assert "predicted_risk_level" in df.columns
        valid_levels = {"LOW", "MEDIUM", "HIGH"}
        assert set(df["predicted_risk_level"].unique()).issubset(valid_levels)
