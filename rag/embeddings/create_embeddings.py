"""RAG Embeddings Builder — Phase 4, MahaTraffic AI.

Creates TF-IDF vector embeddings for the RAG document corpus.
Uses scikit-learn TF-IDF (no GPU, no external API required).

Input:  rag/index/processed_chunks.json  (or rebuild from documents/)
Output: rag/index/tfidf_index.pkl

Note: FAISS is listed in requirements.txt but may not be installed.
This module falls back to sklearn cosine similarity which provides
equivalent functionality for the corpus size in this project.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import pickle

logger = logging.getLogger("rag.embeddings")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CHUNKS_JSON = BASE_DIR / "rag" / "index" / "processed_chunks.json"
INDEX_DIR = BASE_DIR / "rag" / "index"
TFIDF_INDEX_PATH = INDEX_DIR / "tfidf_index.pkl"


def build_tfidf_index(chunks: list[dict] | None = None) -> object:
    """Build and persist a TF-IDF index from document chunks.

    Args:
        chunks: Pre-loaded chunk records. If None, loads from processed_chunks.json.

    Returns:
        The built RAGIndex object (from rag/retrieval.py).
    """
    from rag.retrieval import RAGIndex

    if chunks is None:
        if not CHUNKS_JSON.exists():
            logger.warning("processed_chunks.json not found. Building from documents...")
            from rag.preprocessing.process_documents import process_documents
            chunks = process_documents()
        else:
            with open(CHUNKS_JSON, encoding="utf-8") as f:
                chunks = json.load(f)

    logger.info("Building TF-IDF index over %d chunks...", len(chunks))
    index = RAGIndex()
    index.build(chunks)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(TFIDF_INDEX_PATH, "wb") as f:
        pickle.dump(index, f)
    logger.info("TF-IDF index saved: %s", TFIDF_INDEX_PATH)

    print(f"  Embeddings index built: {len(chunks)} chunks, saved to {TFIDF_INDEX_PATH}")
    return index


def load_index() -> object:
    """Load the persisted TF-IDF index."""
    if not TFIDF_INDEX_PATH.exists():
        logger.info("Index not found. Building...")
        return build_tfidf_index()
    with open(TFIDF_INDEX_PATH, "rb") as f:
        index = pickle.load(f)
    logger.info("TF-IDF index loaded: %d chunks.", len(index.chunks))
    return index


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    idx = build_tfidf_index()
    print(f"Index ready. Vocabulary size: {len(idx.vectorizer.vocabulary_)}")
