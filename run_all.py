#!/usr/bin/env python3
"""MahaTraffic AI — One-Shot Run Script.

Verifies the pipeline end-to-end:
  1. Data files exist
  2. ML model artifacts exist
  3. RAG index built
  4. Backend imports cleanly
  5. Agent workflow executes

Usage:
  python run_all.py          # Verify only
  python run_all.py --serve  # Verify + start FastAPI server
"""

import sys
import argparse
import subprocess
from pathlib import Path

BASE = Path(__file__).resolve().parent

def check(condition, msg):
    if condition:
        print(f"  [OK] {msg}")
    else:
        print(f"  [MISSING] {msg}")
    return condition

def main():
    parser = argparse.ArgumentParser(description="MahaTraffic AI startup script")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI backend after checks")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild ML model and RAG index")
    args = parser.parse_args()

    sys.path.insert(0, str(BASE))

    print("\n" + "=" * 65)
    print("MAHATRAFFIC AI — SYSTEM VERIFICATION")
    print("=" * 65)

    all_ok = True

    print("\n[1] DATA CHECKS")
    all_ok &= check((BASE/"data"/"raw"/"accidents"/"maharashtra_district_accidents_2019_2023.csv").exists(),
                    "Raw accident dataset")
    all_ok &= check((BASE/"data"/"raw"/"social_media"/"maharashtra_road_safety_public_posts.csv").exists(),
                    "Raw social media dataset")
    all_ok &= check((BASE/"data"/"processed"/"parquet"/"accidents"/"maharashtra_accidents_clean.parquet").exists(),
                    "Cleaned accident Parquet")
    all_ok &= check((BASE/"data"/"processed"/"parquet"/"social_media"/"maharashtra_social_clean.parquet").exists(),
                    "Cleaned social Parquet")
    all_ok &= check((BASE/"data"/"processed"/"features"/"risk_features.parquet").exists(),
                    "ML feature matrix Parquet")

    print("\n[2] ML MODEL ARTIFACTS")
    if args.rebuild:
        print("  Rebuilding ML model...")
        subprocess.run([sys.executable, "ml/training/train_model.py"], cwd=BASE, check=True)
    all_ok &= check((BASE/"ml"/"models"/"risk_classifier.pkl").exists(), "Trained Random Forest model")
    all_ok &= check((BASE/"ml"/"models"/"label_encoders.pkl").exists(), "Label encoders")
    all_ok &= check((BASE/"ml"/"models"/"metrics.json").exists(), "Training metrics")
    all_ok &= check((BASE/"ml"/"models"/"feature_importances.json").exists(), "Feature importances")

    print("\n[3] RAG INDEX")
    if args.rebuild or not (BASE/"rag"/"index"/"rag_index.pkl").exists():
        print("  Building RAG index...")
        from rag.retrieval import RAGRetriever
        r = RAGRetriever()
        r.initialize(force_rebuild=True)
    all_ok &= check((BASE/"rag"/"index"/"rag_index.pkl").exists(), "RAG TF-IDF index")
    all_ok &= check((BASE/"rag"/"documents"/"morth_road_safety_guidelines_2022.md").exists(), "MoRTH guidelines doc")

    print("\n[4] SOCIAL ANALYTICS")
    all_ok &= check((BASE/"data"/"processed"/"parquet"/"social_media"/"social_features.parquet").exists(),
                    "Social sentiment features")
    all_ok &= check((BASE/"data"/"processed"/"parquet"/"social_media"/"sentiment_report.json").exists(),
                    "Sentiment report JSON")

    print("\n[5] BACKEND IMPORT CHECK")
    try:
        from backend.app.main import app
        route_count = len([r for r in app.routes if hasattr(r, 'path')])
        check(route_count >= 20, f"FastAPI app — {route_count} routes registered")
    except Exception as e:
        check(False, f"FastAPI import failed: {e}")
        all_ok = False

    print("\n[6] AGENT PIPELINE CHECK")
    try:
        from agents.workflow.multi_agent import run_agent_workflow
        ctx = run_agent_workflow("Which districts in Maharashtra have the highest accident risk?")
        resp = ctx.get("final_response","")
        check(len(resp) > 100, f"Agent workflow — response {len(resp)} chars, latency {ctx.get('latency_ms')}ms")
    except Exception as e:
        check(False, f"Agent workflow failed: {e}")
        all_ok = False

    print("\n[7] RELIABILITY REPORT")
    check((BASE/"reliability"/"logs"/"evaluation_report.json").exists(), "Evaluation report")

    print("\n" + "=" * 65)
    if all_ok:
        print("ALL CHECKS PASSED — System ready.")
    else:
        print("SOME CHECKS FAILED — See above for details.")
    print("=" * 65)

    if args.serve:
        print("\nStarting FastAPI server on http://localhost:8000 ...")
        print("API Docs: http://localhost:8000/docs")
        print("Frontend: open frontend/index.html in browser\n")
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.app.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
        ], cwd=BASE)


if __name__ == "__main__":
    main()
