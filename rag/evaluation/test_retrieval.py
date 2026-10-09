"""RAG Retrieval Evaluation — Phase 4, MahaTraffic AI.

Tests the RAG system with road-safety queries and evaluates:
  - Retrieval relevance (similarity scores)
  - Source availability (document coverage)
  - Retrieval latency

Test queries cover:
  1. Common causes of road accidents
  2. Recommended road-safety measures
  3. Major road-safety concerns in Maharashtra
  4. Factors associated with accident severity
  5. Measures for improving road safety

Results do NOT fabricate answers — only retrieved document chunks
from the actual corpus are reported.
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
import logging
import time

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.evaluation")

REPORT_PATH = BASE_DIR / "data" / "processed" / "rag_retrieval_report.json"

TEST_QUERIES = [
    "What are common causes of road accidents?",
    "What road-safety measures are recommended?",
    "What are major road-safety concerns in Maharashtra?",
    "What factors are associated with accident severity?",
    "What measures are recommended for improving road safety?",
]


def run_retrieval_evaluation(top_k: int = 5) -> dict:
    """Run evaluation queries against the RAG system."""
    from rag.retrieval import RAGRetriever

    retriever = RAGRetriever()
    logger.info("Initialising RAG retriever...")
    retriever.initialize(force_rebuild=True)

    results = []
    total_latency = 0.0

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL EVALUATION — MAHATRAFFIC AI")
    print("=" * 70)

    for i, query in enumerate(TEST_QUERIES, 1):
        t0 = time.perf_counter()
        result = retriever.query(query, top_k=top_k)
        latency = round((time.perf_counter() - t0) * 1000, 2)  # ms
        total_latency += latency

        sources = result.get("sources", [])
        retrieved_count = result.get("retrieved_chunks", 0)

        # Relevance assessment
        if sources:
            max_score = max(s["score"] for s in sources)
            avg_score = round(sum(s["score"] for s in sources) / len(sources), 4)
        else:
            max_score = 0.0
            avg_score = 0.0

        relevance_label = (
            "HIGH" if max_score >= 0.20 else
            "MEDIUM" if max_score >= 0.05 else
            "LOW"
        )

        query_result = {
            "query_id": i,
            "query": query,
            "retrieved_chunks": retrieved_count,
            "max_similarity": max_score,
            "avg_similarity": avg_score,
            "relevance_assessment": relevance_label,
            "latency_ms": latency,
            "sources": sources,
            "context_preview": result.get("context", "")[:300] + "...",
        }
        results.append(query_result)

        print(f"\nQ{i}: {query}")
        print(f"   Chunks retrieved: {retrieved_count}")
        print(f"   Max similarity:   {max_score:.4f}  ({relevance_label})")
        print(f"   Latency:          {latency:.1f} ms")
        print(f"   Sources:          {[s['doc'] for s in sources]}")
        if result.get("context"):
            print(f"   Context preview:  {result['context'][:200]}...")

    avg_latency = round(total_latency / len(TEST_QUERIES), 2)
    high_rel = sum(1 for r in results if r["relevance_assessment"] == "HIGH")
    medium_rel = sum(1 for r in results if r["relevance_assessment"] == "MEDIUM")

    summary = {
        "total_queries": len(TEST_QUERIES),
        "avg_latency_ms": avg_latency,
        "high_relevance_queries": high_rel,
        "medium_relevance_queries": medium_rel,
        "low_relevance_queries": len(TEST_QUERIES) - high_rel - medium_rel,
        "query_results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"  Total queries:          {summary['total_queries']}")
    print(f"  Avg retrieval latency:  {avg_latency:.1f} ms")
    print(f"  HIGH relevance:         {high_rel}/{len(TEST_QUERIES)}")
    print(f"  MEDIUM relevance:       {medium_rel}/{len(TEST_QUERIES)}")
    print(f"  Report saved:           {REPORT_PATH}")
    print("=" * 70 + "\n")

    summary["num_queries"] = summary["total_queries"]
    summary["avg_chunks_retrieved"] = round(sum(r["retrieved_chunks"] for r in results) / len(results), 2)
    return summary


run_evaluation = run_retrieval_evaluation

if __name__ == "__main__":
    run_retrieval_evaluation(top_k=5)
