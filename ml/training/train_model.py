"""ML Training Pipeline — MahaTraffic AI.

Trains a Random Forest Classifier on the engineered risk feature matrix.
Outputs:
  - Trained model artifact: ml/models/risk_classifier.pkl
  - Evaluation metrics: ml/models/metrics.json
  - Feature importances: ml/models/feature_importances.json

Uses scikit-learn locally; Pandas for preprocessing.
No fabricated performance numbers — all metrics derived from real feature matrix.
"""

from pathlib import Path
import json
import logging
import pickle
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
logger = logging.getLogger("ml.training")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_PATH = MODEL_DIR / "risk_classifier.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
IMPORTANCES_PATH = MODEL_DIR / "feature_importances.json"
ENCODERS_PATH = MODEL_DIR / "label_encoders.pkl"


NUMERIC_FEATURES = [
    "year",
    "month",
    "month_sin",
    "month_cos",
    "is_monsoon",
    "is_night",
    "is_highway",
    "accident_count",
    "fatal_accidents",
    "deaths",
    "injuries",
    "fatality_ratio",
    "injury_ratio",
    "severity_index",
]

CATEGORICAL_FEATURES = ["district", "road_type", "primary_cause"]
TARGET_COL = "risk_category"


def load_features() -> pd.DataFrame:
    """Load the pre-built feature matrix."""
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Feature Parquet not found: {FEATURES_PATH}")
    logger.info("Loading feature matrix from: %s", FEATURES_PATH)
    df = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    logger.info("Loaded %d records with %d columns.", len(df), len(df.columns))
    return df


def encode_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Label-encode categorical columns. Returns (encoded_df, encoders_dict)."""
    encoders = {}
    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le
        logger.info("Encoded '%s': %d unique classes.", col, len(le.classes_))
    return df_enc, encoders


def train(df: pd.DataFrame, encoders: dict) -> dict:
    """Train Random Forest Classifier and return evaluation results."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    # Encode target
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    encoders["target"] = target_le

    logger.info("Class distribution: %s", dict(zip(*np.unique(y_encoded, return_counts=True))))

    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
    )
    logger.info("Train: %d, Test: %d samples.", len(X_train), len(X_test))

    # Random Forest with tuned hyperparameters
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

    # Predictions
    y_pred = clf.predict(X_test)
    y_pred_labels = target_le.inverse_transform(y_pred)
    y_test_labels = target_le.inverse_transform(y_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    # Cross-validation (5-fold stratified)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y_encoded, cv=cv, scoring="accuracy")
    cv_f1 = cross_val_score(clf, X, y_encoded, cv=cv, scoring="f1_macro")

    # Classification report
    report = classification_report(
        y_test_labels,
        y_pred_labels,
        output_dict=True,
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()
    class_names = target_le.classes_.tolist()

    # Feature importances
    feat_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    importances = dict(sorted(
        zip(feat_names, clf.feature_importances_.tolist()),
        key=lambda x: x[1],
        reverse=True,
    ))

    metrics = {
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
    """Persist model, encoders, metrics and feature importances."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Model saved: %s", MODEL_PATH)

    # Encoders
    with open(ENCODERS_PATH, "wb") as f:
        pickle.dump(encoders, f)
    logger.info("Encoders saved: %s", ENCODERS_PATH)

    # Metrics
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved: %s", METRICS_PATH)

    # Feature importances
    with open(IMPORTANCES_PATH, "w", encoding="utf-8") as f:
        json.dump(importances, f, indent=2)
    logger.info("Feature importances saved: %s", IMPORTANCES_PATH)


def print_report(metrics: dict) -> None:
    """Print human-readable training summary."""
    print("\n" + "=" * 65)
    print("MAHATRAFFIC AI — ML TRAINING REPORT")
    print("=" * 65)
    print(f"  Accuracy (Test):          {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1-Score (Macro):         {metrics['f1_macro']:.4f}")
    print(f"  F1-Score (Weighted):      {metrics['f1_weighted']:.4f}")
    print(f"  Precision (Weighted):     {metrics['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):        {metrics['recall_weighted']:.4f}")
    print(f"  CV Accuracy:              {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
    print(f"  CV F1 Macro:              {metrics['cv_f1_macro_mean']:.4f} ± {metrics['cv_f1_macro_std']:.4f}")
    print(f"  Train / Test:             {metrics['train_samples']} / {metrics['test_samples']}")
    print(f"  Classes:                  {metrics['class_names']}")
    print("-" * 65)
    print("  Confusion Matrix:")
    for row in metrics["confusion_matrix"]:
        print(f"    {row}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    df = load_features()
    df_enc, encoders = encode_categoricals(df)
    metrics, clf, encoders, importances = train(df_enc, encoders)
    save_artifacts(clf, encoders, metrics, importances)
    print_report(metrics)
