"""Random Forest Training Pipeline — Phase 4, MahaTraffic AI.

Trains a Random Forest Classifier for historical risk classification.

Target:   risk_level (LOW / MEDIUM / HIGH)
Features: Selected from actual available accident data.

Model saved to: ml/models/risk_model.joblib

NOTE: This model performs HISTORICAL RISK CLASSIFICATION based on
      observed historical patterns. It does NOT predict exact future
      accident occurrence.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.train_risk_model")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
RISK_SCORES_PATH = BASE_DIR / "data" / "processed" / "features" / "risk_scores.parquet"
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
ENCODERS_PATH = MODEL_DIR / "risk_model_encoders.joblib"
METRICS_PATH = MODEL_DIR / "risk_model_metrics.json"
IMPORTANCES_PATH = MODEL_DIR / "risk_model_feature_importances.json"
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "features" / "model_predictions.parquet"

NUMERIC_FEATURES = [
    "year", "month", "month_sin", "month_cos",
    "is_monsoon", "is_night", "is_highway",
    "accident_count", "fatal_accidents", "deaths", "injuries",
    "fatality_ratio", "injury_ratio", "severity_index",
]

CATEGORICAL_FEATURES = ["district", "road_type", "primary_cause"]
TARGET_COL = "risk_category"


def _load_features() -> pd.DataFrame:
    """Load feature matrix, preferring the existing risk_features.parquet."""
    if FEATURES_PATH.exists():
        logger.info("Loading: %s", FEATURES_PATH)
        return pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    if RISK_SCORES_PATH.exists():
        logger.info("risk_features.parquet not found, using risk_scores.parquet")
        df = pd.read_parquet(RISK_SCORES_PATH, engine="pyarrow")
        # Build missing columns from what we have
        if "risk_category" not in df.columns and "risk_level" in df.columns:
            df["risk_category"] = df["risk_level"]
        # Build boolean flags if missing
        if "is_night" not in df.columns:
            df["is_night"] = df["time_period"].str.contains("Night", na=False).astype(int)
        if "is_highway" not in df.columns:
            df["is_highway"] = df["road_type"].str.contains(
                "National Highway|State Highway|Expressway", na=False, case=False
            ).astype(int)
        if "is_monsoon" not in df.columns:
            df["is_monsoon"] = 0
        if "month_sin" not in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12).round(4)
        if "month_cos" not in df.columns:
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12).round(4)
        if "severity_index" not in df.columns:
            df["severity_index"] = df.get("accident_severity", 0)
        if "fatal_accidents" not in df.columns:
            df["fatal_accidents"] = df.get("fatal_accident_count", 0)
        if "fatality_ratio" not in df.columns:
            df["fatality_ratio"] = (df["deaths"] / df["accident_count"].replace(0, 1)).round(4)
        if "injury_ratio" not in df.columns:
            df["injury_ratio"] = (df["injuries"] / df["accident_count"].replace(0, 1)).round(4)
        return df
    raise FileNotFoundError(
        "No feature Parquet found. Run ml/training/feature_engineering.py "
        "and ml/prediction/risk_score.py first."
    )


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns (encoded_df, encoders_dict)."""
    encoders: dict = {}
    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = "Unknown"
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
        logger.info("Encoded '%s': %d unique classes.", col, len(le.classes_))
    return df_enc, encoders


def train_model(df: pd.DataFrame, encoders: dict) -> tuple[dict, RandomForestClassifier, dict, dict]:
    """Train Random Forest and compute evaluation metrics."""
    # Ensure all numeric features exist (fill missing with 0)
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    encoders["target"] = target_le

    logger.info("Class distribution: %s",
                dict(zip(*np.unique(y_encoded, return_counts=True))))

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    logger.info("Train: %d, Test: %d", len(X_train), len(X_test))

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    logger.info("Training Random Forest (200 trees)...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_pred_labels = target_le.inverse_transform(y_pred)
    y_test_labels = target_le.inverse_transform(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X, y_encoded, cv=cv, scoring="f1_macro")

    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()
    class_names = target_le.classes_.tolist()

    feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importances = dict(sorted(
        zip(feat_names, clf.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    ))

    metrics = {
        "model": "RandomForestClassifier",
        "task": "Historical risk classification (LOW/MEDIUM/HIGH)",
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_f1_macro_mean": round(float(cv_f1.mean()), 4),
        "cv_f1_macro_std": round(float(cv_f1.std()), 4),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": int(len(feat_names)),
        "class_names": class_names,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    return metrics, clf, encoders, importances


def save_artifacts(clf, encoders, metrics, importances) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    logger.info("Model saved (joblib): %s", MODEL_PATH)

    joblib.dump(encoders, ENCODERS_PATH)
    logger.info("Encoders saved: %s", ENCODERS_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", METRICS_PATH)

    with open(IMPORTANCES_PATH, "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)
    logger.info("Feature importances saved: %s", IMPORTANCES_PATH)


def save_predictions(df: pd.DataFrame, clf, encoders: dict) -> None:
    """Save test-set predictions to Parquet for downstream evaluation."""
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "Unknown"

    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = encoders[col]
        known = set(le.classes_)
        df_enc[col] = df_enc[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df_enc[col] = le.transform(df_enc[col])

    X_all = df_enc[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    preds_encoded = clf.predict(X_all)
    proba = clf.predict_proba(X_all)
    preds_labels = encoders["target"].inverse_transform(preds_encoded)

    out = df[["year", "month", "district", "road_type", "primary_cause"]].copy()
    out["predicted_risk_level"] = preds_labels
    out["prediction_confidence"] = proba.max(axis=1).round(4)
    out["true_risk_level"] = df[TARGET_COL].values if TARGET_COL in df.columns else "UNKNOWN"

    out.to_parquet(PREDICTIONS_PATH, engine="pyarrow", compression="snappy", index=False)
    logger.info("Predictions saved: %s", PREDICTIONS_PATH)


def print_training_report(metrics: dict) -> None:
    print("\n" + "=" * 70)
    print("MAHATRAFFIC AI — ML TRAINING REPORT")
    print("(Historical Risk Classification — Random Forest)")
    print("=" * 70)
    print(f"  Accuracy (Test):        {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score (Macro):       {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted):    {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Weighted):   {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):      {metrics['recall_weighted']:.4f}")
    print(f"  CV Accuracy:            {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"  CV F1 Macro:            {metrics['cv_f1_macro_mean']:.4f} ± {metrics['cv_f1_macro_std']:.4f}")
    print(f"  Train / Test samples:   {metrics['train_samples']} / {metrics['test_samples']}")
    print(f"  Feature count:          {metrics['n_features']}")
    print(f"  Classes:                {metrics['class_names']}")
    print("\n  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    df = _load_features()
    df_enc, encoders = encode_categoricals(df)
    metrics, clf, encoders, importances = train_model(df_enc, encoders)
    save_artifacts(clf, encoders, metrics, importances)
    save_predictions(df, clf, encoders)
    print_training_report(metrics)
