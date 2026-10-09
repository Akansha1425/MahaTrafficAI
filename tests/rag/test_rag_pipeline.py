"""Phase 4 Tests — RAG Intelligence Layer (Document Processing, Embeddings, Indexing, Retrieval)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestRAGPreprocessing:
    """Tests for rag/preprocessing/process_documents.py"""

    def test_process_documents_pipeline(self):
        from rag.preprocessing.process_documents import process_all_documents, CHUNKS_PARQUET
        chunks = process_all_documents()
        assert len(chunks) > 0
        assert CHUNKS_PARQUET.exists()
        sample = chunks[0]
        assert "document_id" in sample
        assert "text" in sample
        assert "chunk_id" in sample


class TestRAGIndexingAndRetrieval:
    """Tests for rag/retrieval.py and rag/index/build_index.py"""

    def test_build_and_initialize_index(self):
        from rag.retrieval import RAGRetriever
        retriever = RAGRetriever()
        retriever.initialize()
        assert retriever._index is not None
        assert len(retriever._index.chunks) > 0

    def test_retriever_query(self):
        from rag.retrieval import RAGRetriever
        retriever = RAGRetriever()
        retriever.initialize()
        res = retriever.query("potholes road safety black spots", top_k=3)
        assert "context" in res
        assert "sources" in res
        assert len(res["sources"]) > 0


class TestRAGEvaluation:
    """Tests for rag/evaluation/test_retrieval.py"""

    def test_run_retrieval_evaluation(self):
        from rag.evaluation.test_retrieval import run_evaluation
        summary = run_evaluation()
        assert "num_queries" in summary
        assert "avg_chunks_retrieved" in summary
        assert summary["num_queries"] > 0
