"""MCP Tool implementations for RAG official road safety document retrieval."""

from __future__ import annotations
from typing import Any, Dict


def search_road_safety_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Retrieve grounded passages from official road safety guidelines and MoRTH manuals.

    Args:
        query: Road safety natural language inquiry.
        top_k: Maximum number of relevant chunks to retrieve.

    Returns:
        Structured dictionary with retrieved passages, sources, and similarity scores.
    """
    from backend.app.services.intelligence_service import search_road_safety_documents as _search_docs

    res = _search_docs(query=query, top_k=top_k)
    return {
        "status": "success",
        "query": query,
        "retrieved_chunks": res.get("retrieved_chunks", 0),
        "context": res.get("context", ""),
        "sources": res.get("sources", []),
        "disclaimer": "Passages retrieved from indexed official guidelines (MoRTH, MMVR, IRC).",
    }
