"""RAG Service — MahaTraffic AI.

Wraps the RAG retrieval pipeline for use in FastAPI routes and agents.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import logging
import sys

logger = logging.getLogger(__name__)

# Add project root to path so rag module can be imported
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

_retriever = None


def _get_retriever():
    """Lazy-load and cache the RAG retriever."""
    global _retriever
    if _retriever is None:
        try:
            from rag.retrieval import RAGRetriever
            _retriever = RAGRetriever()
            # Try loading from disk; fall back to rebuild on any error
            try:
                _retriever.initialize(force_rebuild=False)
            except Exception:
                logger.warning("RAG index load failed (pickle namespace issue), rebuilding...")
                _retriever.initialize(force_rebuild=True)
        except Exception as e:
            logger.error("Failed to initialize RAG retriever: %s", e)
    return _retriever


class RAGService:
    """Service wrapper for the RAG retrieval pipeline."""

    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG system and return retrieved context."""
        retriever = _get_retriever()
        if retriever is None:
            return {
                "error": "RAG system not available.",
                "context": "",
                "sources": [],
                "retrieved_chunks": 0,
            }
        return retriever.query(question, top_k=top_k)

    def answer(self, question: str, top_k: int = 5) -> str:
        """Generate a knowledge-grounded answer."""
        retriever = _get_retriever()
        if retriever is None:
            return "RAG system unavailable. Please check that documents are indexed."
        return retriever.answer_with_context(question, top_k=top_k)

    def get_document_list(self) -> list[str]:
        """Return the list of indexed document names."""
        retriever = _get_retriever()
        if retriever is None or retriever._index is None:
            return []
        doc_ids = list({chunk["doc_id"] for chunk in retriever._index.chunks})
        return sorted(doc_ids)


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get singleton RAGService instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
