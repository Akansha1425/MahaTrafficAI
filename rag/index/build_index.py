"""RAG Index Builder — Phase 4, MahaTraffic AI.

Builds the retrieval index from processed document chunks.
Coordinates preprocessing → embedding → index persistence.

Output: rag/index/rag_index.pkl
"""

from __future__ import annotations
from pathlib import Path
import logging

logger = logging.getLogger("rag.build_index")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INDEX_PATH = BASE_DIR / "rag" / "index" / "rag_index.pkl"


def build_index(force_rebuild: bool = False) -> object:
    """Build or load the RAG retrieval index.

    Args:
        force_rebuild: If True, rebuild even if index exists on disk.

    Returns:
        RAGIndex instance.
    """
    from rag.retrieval import RAGIndex, load_documents, INDEX_PATH as RETRIEVAL_INDEX_PATH

    if not force_rebuild and RETRIEVAL_INDEX_PATH.exists():
        logger.info("Loading existing RAG index from: %s", RETRIEVAL_INDEX_PATH)
        return RAGIndex.load(RETRIEVAL_INDEX_PATH)

    logger.info("Building RAG index from documents...")
    records = load_documents()
    if not records:
        raise RuntimeError("No document chunks found. Check rag/documents/ directory.")

    index = RAGIndex()
    index.build(records)
    index.save(RETRIEVAL_INDEX_PATH)

    print(f"\n  RAG index built: {len(records)} chunks, vocabulary={len(index.vectorizer.vocabulary_)}")
    print(f"  Saved to: {RETRIEVAL_INDEX_PATH}\n")
    return index


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("\nBuilding RAG index...")
    idx = build_index(force_rebuild=True)
    print("RAG index build complete.")
