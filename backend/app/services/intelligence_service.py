"""Unified Intelligence Service — Phase 4, MahaTraffic AI.

Exposes unified functions for all intelligence layer operations:
  - get_accident_statistics()
  - calculate_risk()
  - predict_risk()
  - analyze_sentiment()
  - get_social_trends()
  - search_road_safety_documents()

These functions are designed to be consumed by MCP tools and API routes.
All results are derived from historical data only.

DISCLAIMER: Historical analytics only. No real-time prediction.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("services.intelligence")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

# ─── Accident Statistics ───────────────────────────────────────────────────────

def get_accident_statistics(
    district: str | None = None,
    year: int | None = None,
    road_type: str | None = None,
) -> dict[str, Any]:
    """Return aggregated historical accident statistics.

    Args:
        district: Filter by district name (optional).
        year: Filter by year (optional).
        road_type: Filter by road type (optional).

    Returns:
        Dict with total_accidents, total_deaths, total_injuries,
        top_causes, year_range, district_count.
    """
    import pandas as pd

    parquet = BASE_DIR / "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"
    csv = BASE_DIR / "data" / "processed" / "accidents_clean.csv"

    try:
        if parquet.exists():
            df = pd.read_parquet(parquet, engine="pyarrow")
        elif csv.exists():
            df = pd.read_csv(csv)
        else:
            return {"error": "No accident data available."}

        if district:
            df = df[df["district"].str.lower() == district.lower()]
        if year:
            df = df[df["year"] == year]
        if road_type:
            df = df[df["road_type"].str.lower() == road_type.lower()]

        if df.empty:
            return {"error": f"No data found for the given filters."}

        top_causes = df.groupby("primary_cause")["accident_count"].sum().nlargest(5).to_dict()
        by_year = df.groupby("year")["accident_count"].sum().to_dict()

        return {
            "total_accidents": int(df["accident_count"].sum()),
            "total_deaths": int(df["deaths"].sum()),
            "total_injuries": int(df["injuries"].sum()),
            "total_fatal_accidents": int(df["fatal_accidents"].sum()),
            "district_count": int(df["district"].nunique()),
            "year_range": f"{int(df['year'].min())}–{int(df['year'].max())}",
            "top_causes": {k: int(v) for k, v in top_causes.items()},
            "accidents_by_year": {int(k): int(v) for k, v in by_year.items()},
            "filters_applied": {
                "district": district,
                "year": year,
                "road_type": road_type,
            },
            "data_source": "maharashtra_district_accidents_2019_2023",
        }
    except Exception as e:
        logger.exception("Error in get_accident_statistics")
        return {"error": str(e)}


# ─── Risk Score Calculation ────────────────────────────────────────────────────

def calculate_risk(
    accident_severity: float,
    accident_frequency: float,
    road_risk_factor: float,
    time_risk_factor: float,
) -> dict[str, Any]:
    """Calculate project-defined historical risk score.

    Risk Score = 0.40 × Severity + 0.25 × Frequency + 0.20 × Road + 0.15 × Time
    Normalized to 0–100. Level: LOW (0–33), MEDIUM (34–66), HIGH (67–100).

    NOTE: Project-defined analytical score. NOT an official government standard.
    """
    try:
        from ml.prediction.risk_score import calculate_risk_score, classify_risk_level
        import pandas as pd

        score = float(calculate_risk_score(
            pd.Series([accident_severity]),
            pd.Series([accident_frequency]),
            pd.Series([road_risk_factor]),
            pd.Series([time_risk_factor]),
            pre_normalized=True,
        ).iloc[0])

        level = classify_risk_level(score)
        return {
            "risk_score": round(score, 2),
            "risk_level": level,
            "components": {
                "severity_weight": 0.40,
                "frequency_weight": 0.25,
                "road_weight": 0.20,
                "time_weight": 0.15,
            },
            "disclaimer": "Project-defined analytical score. Not an official government risk standard.",
        }
    except Exception as e:
        logger.exception("Error in calculate_risk")
        return {"error": str(e)}


# ─── ML Risk Prediction ────────────────────────────────────────────────────────

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
) -> dict[str, Any]:
    """Predict risk level using the trained Random Forest model.

    Returns risk_score, risk_level, confidence, top_factors.

    NOTE: Historical risk classification based on observed historical patterns.
          Does NOT predict exact future accident occurrence.
    """
    try:
        from ml.prediction.predict_risk import predict_risk as _predict
        return _predict(
            year=year, month=month, district=district,
            road_type=road_type, primary_cause=primary_cause,
            accident_count=accident_count, fatal_accidents=fatal_accidents,
            deaths=deaths, injuries=injuries, time_period=time_period,
        )
    except FileNotFoundError:
        return {
            "error": "Model not trained yet. Run ml/training/train_risk_model.py first.",
            "risk_score": None,
            "risk_level": None,
            "confidence": None,
        }
    except Exception as e:
        logger.exception("Error in predict_risk")
        return {"error": str(e)}


# ─── Sentiment Analysis ────────────────────────────────────────────────────────

def analyze_sentiment(texts: list[str] | None = None) -> dict[str, Any]:
    """Analyse sentiment of social posts.

    If texts is None, returns aggregated analytics from stored results.
    If texts is provided, analyses the given texts.

    NOTE: Sentiment is a PUBLIC PERCEPTION / COMPLAINT SIGNAL.
          No causal relationship with accidents is implied.
    """
    try:
        from social_analytics.sentiment.analyze_sentiment import score_sentiment
        import pandas as pd

        if texts is not None:
            results = [{"text": t[:100], **dict(zip(["sentiment_score", "sentiment_label"], score_sentiment(t)))}
                       for t in texts]
            return {
                "type": "real_time_batch",
                "results": results,
                "disclaimer": "Public perception / complaint signal. Not causal.",
            }

        # Return stored analytics
        report_path = BASE_DIR / "data" / "processed" / "social_sentiment_report.json"
        if report_path.exists():
            import json
            with open(report_path, encoding="utf-8") as f:
                return json.load(f)

        # Build on the fly
        from social_analytics.sentiment.analyze_sentiment import run_sentiment_analysis, compute_sentiment_analytics
        df = run_sentiment_analysis()
        return compute_sentiment_analytics(df)

    except Exception as e:
        logger.exception("Error in analyze_sentiment")
        return {"error": str(e)}


# ─── Social Trends ─────────────────────────────────────────────────────────────

def get_social_trends() -> dict[str, Any]:
    """Return social trend analytics (monthly counts, topics, locations)."""
    try:
        report_path = BASE_DIR / "data" / "processed" / "social_trends_report.json"
        if report_path.exists():
            import json
            with open(report_path, encoding="utf-8") as f:
                return json.load(f)

        from social_analytics.trends.social_trends import compute_trends_report
        import pandas as pd
        parquet = BASE_DIR / "data" / "processed" / "social_sentiment.parquet"
        if parquet.exists():
            df = pd.read_parquet(parquet, engine="pyarrow")
        else:
            from social_analytics.preprocessing.clean_social import run_cleaning
            df = run_cleaning()
        return compute_trends_report(df)

    except Exception as e:
        logger.exception("Error in get_social_trends")
        return {"error": str(e)}


# ─── RAG Document Search ───────────────────────────────────────────────────────

def search_road_safety_documents(query: str, top_k: int = 5) -> dict[str, Any]:
    """Search official road-safety documents using RAG retrieval.

    Args:
        query: Natural language question.
        top_k: Number of top chunks to retrieve (default 5).

    Returns:
        Dict with query, retrieved_chunks, context, sources.
        Sources contain doc name and similarity score.
    """
    try:
        from rag.retrieval import RAGRetriever
        retriever = RAGRetriever()
        retriever.initialize()
        result = retriever.query(query, top_k=top_k)

        # Add metadata from registry
        from rag.preprocessing.process_documents import DOCUMENT_REGISTRY
        enriched_sources = []
        for src in result.get("sources", []):
            doc_id = src["doc"]
            meta = DOCUMENT_REGISTRY.get(doc_id, {})
            enriched_sources.append({
                "document_id": doc_id,
                "similarity_score": src["score"],
                "title": meta.get("title", doc_id),
                "source": meta.get("source", "Unknown"),
                "year": meta.get("year", "Unknown"),
                "document_type": meta.get("document_type", "Unknown"),
            })

        return {
            "query": query,
            "retrieved_chunks": result.get("retrieved_chunks", 0),
            "context": result.get("context", ""),
            "sources": enriched_sources,
        }
    except Exception as e:
        logger.exception("Error in search_road_safety_documents")
        return {"error": str(e), "query": query, "retrieved_chunks": 0}
