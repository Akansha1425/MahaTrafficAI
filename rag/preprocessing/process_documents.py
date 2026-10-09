"""RAG Document Preprocessing Pipeline — Phase 4, MahaTraffic AI.

Processes markdown road-safety documents from rag/documents/ into
chunked, metadata-annotated records for vector indexing.

Documents:
  - MoRTH Road Safety Guidelines 2022
  - Maharashtra Motor Vehicles Rules Safety Summary
  - IRC SP 88 Road Safety Audit Manual

Metadata schema:
    {
        "document_id": "...",
        "source": "...",
        "title": "...",
        "year": "...",
        "location": "...",
        "document_type": "...",
        "chunk_id": "..."
    }

Output:
    data/processed/rag_chunks.parquet
    rag/index/processed_chunks.json
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag.preprocessing")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DOCS_DIR = BASE_DIR / "rag" / "documents"
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "rag_chunks.parquet"
CHUNKS_PARQUET = OUTPUT_PARQUET
OUTPUT_JSON = BASE_DIR / "rag" / "index" / "processed_chunks.json"

CHUNK_SIZE_WORDS = 200
CHUNK_OVERLAP_WORDS = 40

# ─── Document metadata registry ───────────────────────────────────────────────

DOCUMENT_REGISTRY: dict[str, dict] = {
    "morth_road_safety_guidelines_2022": {
        "source": "Ministry of Road Transport & Highways (MoRTH), Government of India",
        "title": "Road Safety Guidelines 2022 — MoRTH",
        "year": "2022",
        "location": "India",
        "document_type": "Government Guidelines",
    },
    "maharashtra_motor_vehicles_rules_safety_summary": {
        "source": "Maharashtra State Government — Motor Vehicles Department",
        "title": "Maharashtra Motor Vehicles Rules — Safety Provisions Summary",
        "year": "2022",
        "location": "Maharashtra",
        "document_type": "State Regulations Summary",
    },
    "irc_sp_88_road_safety_audit_manual": {
        "source": "Indian Roads Congress (IRC) — SP 88",
        "title": "IRC SP 88 Road Safety Audit Manual",
        "year": "2010",
        "location": "India",
        "document_type": "Technical Manual",
    },
}


# ─── Text Processing ──────────────────────────────────────────────────────────

def _clean_markdown(text: str) -> str:
    """Strip markdown formatting for clean text extraction."""
    # Remove headers, bold, code, tables
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"`[^`]+`", "", text)
    text = re.sub(r"\|[^\n]*\|", " ", text)
    text = re.sub(r"[-*>]+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS, overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split text into overlapping word-window chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ─── Processing ───────────────────────────────────────────────────────────────

def process_documents() -> list[dict]:
    """Process all documents in rag/documents/ into chunked records."""
    md_files = [f for f in DOCS_DIR.glob("*.md") if not f.name.startswith(".")]
    logger.info("Found %d markdown documents in %s", len(md_files), DOCS_DIR)

    all_chunks = []
    for md_file in sorted(md_files):
        doc_id = md_file.stem
        meta = DOCUMENT_REGISTRY.get(doc_id, {
            "source": "Unknown",
            "title": doc_id.replace("_", " ").title(),
            "year": "Unknown",
            "location": "Unknown",
            "document_type": "Document",
        })

        logger.info("Processing: %s", md_file.name)
        raw_text = md_file.read_text(encoding="utf-8")
        clean_text = _clean_markdown(raw_text)
        chunks = _chunk_text(clean_text)

        for i, chunk in enumerate(chunks):
            chunk_record = {
                "document_id": doc_id,
                "chunk_id": f"{doc_id}__chunk_{i:03d}",
                "source": meta["source"],
                "title": meta["title"],
                "year": meta["year"],
                "location": meta["location"],
                "document_type": meta["document_type"],
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
                "word_count": len(chunk.split()),
            }
            all_chunks.append(chunk_record)

        logger.info("  %s: %d chunks produced.", doc_id, len(chunks))

    logger.info("Total chunks: %d", len(all_chunks))
    return all_chunks


def save_chunks(chunks: list[dict]) -> None:
    """Save processed chunks to Parquet and JSON."""
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(chunks)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Chunks saved to Parquet: %s", OUTPUT_PARQUET)

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info("Chunks saved to JSON: %s", OUTPUT_JSON)


def main() -> list[dict]:
    chunks = process_documents()
    save_chunks(chunks)

    print("\n" + "=" * 65)
    print("RAG DOCUMENT PREPROCESSING SUMMARY")
    print("=" * 65)
    print(f"  Documents processed:    {len(DOCUMENT_REGISTRY)}")
    print(f"  Total chunks created:   {len(chunks)}")
    print(f"  Chunk size (words):     {CHUNK_SIZE_WORDS} (overlap={CHUNK_OVERLAP_WORDS})")
    print(f"  Output Parquet:         {OUTPUT_PARQUET}")
    print(f"  Output JSON:            {OUTPUT_JSON}")
    print("\n  Documents:")
    for doc_id, meta in DOCUMENT_REGISTRY.items():
        doc_chunks = [c for c in chunks if c["document_id"] == doc_id]
        print(f"    {meta['title'][:50]:<50} ({len(doc_chunks)} chunks)")
    print("=" * 65 + "\n")
    return chunks


process_all_documents = main

if __name__ == "__main__":
    main()
