"""RAG (Retrieval Augmented Generation) Pipeline — MahaTraffic AI.

Implements a complete local RAG system using:
  - TF-IDF vectorization for document chunking and retrieval (no GPU/API required)
  - FAISS-like cosine similarity via sklearn for fast semantic search
  - Knowledge base from official road safety documents

Provides:
  - Document chunking and indexing
  - Query-based retrieval
  - Context assembly for agent responses
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import pickle
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("rag.retrieval")

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "documents"
INDEX_DIR = BASE_DIR / "index"
INDEX_PATH = INDEX_DIR / "rag_index.pkl"

# ─── Chunking Config ──────────────────────────────────────────────────────────
CHUNK_SIZE = 250      # approximate words per chunk
CHUNK_OVERLAP = 50    # overlapping words between chunks


# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:  # skip tiny chunks
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR) -> list[dict]:
    """Load all markdown documents, chunk them, and attach source metadata."""
    records = []
    md_files = list(docs_dir.glob("*.md"))
    logger.info("Found %d markdown documents in %s", len(md_files), docs_dir)

    for md_file in md_files:
        if md_file.name.startswith("."):
            continue
        text = md_file.read_text(encoding="utf-8")
        # Remove markdown formatting for cleaner embeddings
        clean_text = re.sub(r"[#*`|>-]+", " ", text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        chunks = chunk_text(clean_text)
        for i, chunk in enumerate(chunks):
            records.append({
                "doc_id": md_file.stem,
                "chunk_id": f"{md_file.stem}__chunk_{i}",
                "source_file": md_file.name,
                "text": chunk,
            })

    logger.info("Total chunks produced: %d", len(records))
    return records


# ─── Indexing ─────────────────────────────────────────────────────────────────

class RAGIndex:
    """TF-IDF based document index for retrieval."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
            sublinear_tf=True,
            stop_words="english",
        )
        self.chunks: list[dict] = []
        self.matrix = None
        self._built = False

    def build(self, records: list[dict]) -> None:
        """Build the TF-IDF index from document records."""
        self.chunks = records
        texts = [r["text"] for r in records]
        self.matrix = self.vectorizer.fit_transform(texts)
        self._built = True
        logger.info("RAG index built with %d chunks, vocabulary size %d.",
                    len(records), len(self.vectorizer.vocabulary_))

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the top-k most relevant chunks for a query."""
        if not self._built:
            raise RuntimeError("Index has not been built. Call build() first.")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0.01:  # minimum relevance threshold
                results.append({
                    **self.chunks[idx],
                    "similarity_score": round(float(scores[idx]), 4),
                })
        return results

    def save(self, path: Path = INDEX_PATH) -> None:
        """Persist index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("RAG index saved: %s", path)

    @classmethod
    def load(cls, path: Path = INDEX_PATH) -> "RAGIndex":
        """Load persisted index."""
        with open(path, "rb") as f:
            idx = pickle.load(f)
        logger.info("RAG index loaded: %d chunks.", len(idx.chunks))
        return idx


# ─── Context Assembly ──────────────────────────────────────────────────────────

def assemble_context(results: list[dict], max_tokens: int = 1200) -> str:
    """Combine retrieved chunks into a single context block."""
    context_parts = []
    total_words = 0
    for r in results:
        words = r["text"].split()
        if total_words + len(words) > max_tokens:
            break
        context_parts.append(
            f"[Source: {r['doc_id']} | Relevance: {r['similarity_score']}]\n{r['text']}"
        )
        total_words += len(words)
    return "\n\n---\n\n".join(context_parts)


# ─── High-level API ───────────────────────────────────────────────────────────

class RAGRetriever:
    """High-level RAG interface used by agents and MCP tools."""

    def __init__(self):
        self._index: RAGIndex | None = None

    def initialize(self, force_rebuild: bool = False) -> None:
        """Initialize RAG index (load from disk or rebuild)."""
        if not force_rebuild and INDEX_PATH.exists():
            logger.info("Loading existing RAG index from disk...")
            self._index = RAGIndex.load(INDEX_PATH)
        else:
            logger.info("Building RAG index from scratch...")
            records = load_documents()
            self._index = RAGIndex()
            self._index.build(records)
            self._index.save(INDEX_PATH)

    def query(self, question: str, top_k: int = 5) -> dict:
        """Query the RAG system. Returns context and source metadata."""
        if self._index is None:
            self.initialize()

        results = self._index.retrieve(question, top_k=top_k)
        context = assemble_context(results)

        return {
            "query": question,
            "retrieved_chunks": len(results),
            "context": context,
            "sources": [{"doc": r["doc_id"], "score": r["similarity_score"]} for r in results],
        }

    def answer_with_context(self, question: str, top_k: int = 5) -> str:
        """Generate a knowledge-grounded answer using retrieved context."""
        result = self.query(question, top_k=top_k)

        if not result["sources"]:
            return (
                "No relevant road safety documentation found for this query. "
                "Please consult official MoRTH or Maharashtra RTO publications."
            )

        # Template-based answer grounded in retrieved context
        answer = (
            f"Based on official road safety documentation:\n\n"
            f"{result['context']}\n\n"
            f"[Retrieved from: {', '.join(s['doc'] for s in result['sources'])}]"
        )
        return answer


# ─── Singleton ────────────────────────────────────────────────────────────────

_retriever: RAGRetriever | None = None


def get_retriever() -> RAGRetriever:
    """Get or create the global RAG retriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()
        _retriever.initialize()
    return _retriever


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("\n" + "=" * 60)
    print("MAHATRAFFIC AI — RAG PIPELINE TEST")
    print("=" * 60)

    retriever = RAGRetriever()
    retriever.initialize(force_rebuild=True)

    test_queries = [
        "What defines a road accident black spot in Maharashtra?",
        "What are the speed limits on national highways in India?",
        "How does monsoon season affect road accidents?",
        "What is the penalty for drunk driving in Maharashtra?",
        "What are the recommended guardrail standards for highways?",
    ]

    for q in test_queries:
        print(f"\nQ: {q}")
        res = retriever.query(q, top_k=3)
        print(f"   Retrieved {res['retrieved_chunks']} chunks")
        print(f"   Sources: {res['sources']}")
        print(f"   Context preview: {res['context'][:200]}...")

    print("\n" + "=" * 60)
    print("RAG Pipeline Test Complete.")
    print("=" * 60 + "\n")
