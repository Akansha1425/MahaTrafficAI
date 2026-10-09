"""ML Model Evaluation — Phase 4, MahaTraffic AI.

Evaluates the trained Random Forest on the full feature matrix:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion matrix
  - Classification report
  - Feature importances

Saves:
  data/processed/features/model_predictions.parquet
  docs/evaluation/ML_EVALUATION.md

NOTE: Model performs HISTORICAL RISK CLASSIFICATION based on
      observed historical patterns, NOT prediction of future accidents.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml.evaluate_model")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = BASE_DIR / "ml" / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
ENCODERS_PATH = MODEL_DIR / "risk_model_encoders.joblib"
IMPORTANCES_PATH = MODEL_DIR / "risk_model_feature_importances.json"
METRICS_PATH = MODEL_DIR / "risk_model_metrics.json"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "features" / "risk_features.parquet"
PREDICTIONS_PATH = BASE_DIR / "data" / "processed" / "features" / "model_predictions.parquet"
EVAL_DOC = BASE_DIR / "docs" / "evaluation" / "ML_EVALUATION.md"

NUMERIC_FEATURES = [
    "year", "month", "month_sin", "month_cos",
    "is_monsoon", "is_night", "is_highway",
    "accident_count", "fatal_accidents", "deaths", "injuries",
    "fatality_ratio", "injury_ratio", "severity_index",
]
CATEGORICAL_FEATURES = ["district", "road_type", "primary_cause"]
TARGET_COL = "risk_category"


def load_all() -> tuple:
    for path in [MODEL_PATH, ENCODERS_PATH, FEATURES_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    clf = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    df = pd.read_parquet(FEATURES_PATH, engine="pyarrow")
    importances = {}
    if IMPORTANCES_PATH.exists():
        with open(IMPORTANCES_PATH, encoding="utf-8") as f:
            importances = json.load(f)
    return clf, encoders, df, importances


def evaluate(clf, encoders, df: pd.DataFrame) -> dict:
    """Run full evaluation on the complete dataset."""
    df_enc = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = "Unknown"
        le = encoders[col]
        known = set(le.classes_)
        df_enc[col] = df_enc[col].astype(str).apply(
            lambda x: x if x in known else le.classes_[0]
        )
        df_enc[col] = le.transform(df_enc[col])

    for col in NUMERIC_FEATURES:
        if col not in df_enc.columns:
            df_enc[col] = 0

    X = df_enc[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_true_labels = df[TARGET_COL].values

    target_le = encoders["target"]
    y_true = target_le.transform(y_true_labels)
    y_pred = clf.predict(X)
    proba = clf.predict_proba(X)

    y_pred_labels = target_le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true_labels, y_pred_labels, output_dict=True)

    return {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "precision_weighted": round(float(precision), 4),
        "recall_weighted": round(float(recall), 4),
        "confusion_matrix": cm,
        "class_names": target_le.classes_.tolist(),
        "classification_report": report,
        "y_pred_labels": y_pred_labels.tolist(),
        "y_true_labels": y_true_labels.tolist(),
        "confidence": proba.max(axis=1).round(4).tolist(),
    }


def save_predictions(df: pd.DataFrame, eval_results: dict) -> None:
    out = df[["year", "month", "district", "road_type", "primary_cause"]].copy()
    out["true_risk_level"] = eval_results["y_true_labels"]
    out["predicted_risk_level"] = eval_results["y_pred_labels"]
    out["prediction_confidence"] = eval_results["confidence"]
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PREDICTIONS_PATH, engine="pyarrow", compression="snappy", index=False)
    logger.info("Predictions saved: %s", PREDICTIONS_PATH)


def write_evaluation_doc(metrics: dict, eval_results: dict, importances: dict) -> None:
    """Write ML_EVALUATION.md."""
    EVAL_DOC.parent.mkdir(parents=True, exist_ok=True)
    report = eval_results["classification_report"]
    cm = eval_results["confusion_matrix"]
    classes = eval_results["class_names"]

    lines = [
        "# ML Model Evaluation — MahaTraffic AI",
        "",
        "**Task:** Historical Risk Classification (LOW / MEDIUM / HIGH)",
        "**Model:** Random Forest Classifier (scikit-learn)",
        "**Dataset:** `maharashtra_district_accidents_2019_2023.csv`",
        "",
        "> **DISCLAIMER:** This model performs historical risk classification",
        "> based on observed historical patterns. It does NOT predict exact",
        "> future accident occurrence.",
        "",
        "---",
        "",
        "## Performance Metrics (Full Dataset)",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {eval_results['accuracy']:.4f} ({eval_results['accuracy']*100:.2f}%) |",
        f"| F1-Score (Macro) | {eval_results['f1_macro']:.4f} |",
        f"| F1-Score (Weighted) | {eval_results['f1_weighted']:.4f} |",
        f"| Precision (Weighted) | {eval_results['precision_weighted']:.4f} |",
        f"| Recall (Weighted) | {eval_results['recall_weighted']:.4f} |",
        "",
    ]

    if metrics:
        lines += [
            "## Training Metrics (from training run)",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| CV Accuracy Mean | {metrics.get('cv_accuracy_mean', 'N/A')} |",
            f"| CV Accuracy Std | {metrics.get('cv_accuracy_std', 'N/A')} |",
            f"| CV F1 Macro Mean | {metrics.get('cv_f1_macro_mean', 'N/A')} |",
            f"| Train Samples | {metrics.get('train_samples', 'N/A')} |",
            f"| Test Samples | {metrics.get('test_samples', 'N/A')} |",
            "",
        ]

    lines += [
        "## Confusion Matrix",
        "",
        f"Classes: {classes}",
        "",
        "```",
    ]
    for i, row in enumerate(cm):
        lines.append(f"  {classes[i]:<10} {row}")
    lines += [
        "```",
        "",
        "## Per-Class Classification Report",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|-------|-----------|--------|----------|---------|",
    ]
    for cls in classes:
        if cls in report:
            r = report[cls]
            lines.append(
                f"| {cls} | {r['precision']:.4f} | {r['recall']:.4f} | "
                f"{r['f1-score']:.4f} | {int(r['support'])} |"
            )
    lines += [
        "",
        "## Top Feature Importances",
        "",
        "| Rank | Feature | Importance |",
        "|------|---------|------------|",
    ]
    for i, (feat, imp) in enumerate(list(importances.items())[:10], 1):
        lines.append(f"| {i} | {feat} | {imp:.4f} |")

    lines += [
        "",
        "## Limitations",
        "",
        "- Classification is based on historical district-level aggregated data.",
        "- Individual road-segment level data is not available.",
        "- Weather and road condition fields are not in the primary dataset.",
        "- Class imbalance (MEDIUM dominates) is mitigated by `class_weight='balanced'`.",
        "- The risk score formula is project-defined, not an official government standard.",
        "",
    ]

    with open(EVAL_DOC, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("ML evaluation doc written: %s", EVAL_DOC)


def main() -> dict:
    clf, encoders, df, importances = load_all()
    logger.info("Evaluating model on %d records...", len(df))

    eval_results = evaluate(clf, encoders, df)
    save_predictions(df, eval_results)

    metrics = {}
    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = json.load(f)

    write_evaluation_doc(metrics, eval_results, importances)

    print("\n" + "=" * 70)
    print("ML MODEL EVALUATION — MAHATRAFFIC AI")
    print("=" * 70)
    print(f"  Accuracy:              {eval_results['accuracy']:.4f}")
    print(f"  F1 Macro:              {eval_results['f1_macro']:.4f}")
    print(f"  F1 Weighted:           {eval_results['f1_weighted']:.4f}")
    print(f"  Precision (Weighted):  {eval_results['precision_weighted']:.4f}")
    print(f"  Recall (Weighted):     {eval_results['recall_weighted']:.4f}")
    print("\n  Confusion Matrix:")
    for i, row in enumerate(eval_results["confusion_matrix"]):
        print(f"    {eval_results['class_names'][i]:<10} {row}")
    print("=" * 70 + "\n")

    return eval_results


if __name__ == "__main__":
    main()
