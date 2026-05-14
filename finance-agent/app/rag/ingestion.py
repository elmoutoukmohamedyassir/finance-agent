"""
rag/ingestion.py — PDF ingestion pipeline: PDF → text → chunks → embeddings → ChromaDB.

IMPROVEMENTS over the original ingest.py:
  1. Uses improved chunker (sentence-aware, with metadata)
  2. Singleton embedding model (loaded once, reused) — original reloaded on every call
  3. Deduplication: tracks which files are already ingested via stored metadata
  4. Batch embedding for efficiency (instead of one-by-one)
  5. Proper error handling per file (one bad PDF doesn't break the whole run)
  6. Configurable paths from settings (not hardcoded)
  7. Ingestion report returned instead of just print()

HOW TO USE:
  # From command line:
  python -m app.rag.ingestion

  # Or via API endpoint:
  POST /rag/ingest
"""

import os
import logging
from pathlib import Path

from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.rag.chunker import create_chunks, clean_pdf_text

logger = logging.getLogger(__name__)
settings = get_settings()

COLLECTION_NAME = "saas_finance_docs"

# Embedding model is loaded once at module level (expensive operation)
# All functions share this single instance
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy singleton for the embedding model — loads only when first needed."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _embedding_model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded.")
    return _embedding_model


def get_chroma_collection() -> chromadb.Collection:
    """Returns the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity is better for text
    )
    return collection


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts all text from a PDF file.
    Returns empty string if extraction fails (logged as warning).
    """
    try:
        reader = PdfReader(pdf_path)
        pages_text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_text.append(page_text)
        return "\n".join(pages_text)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def get_already_ingested_sources(collection: chromadb.Collection) -> set[str]:
    """
    Returns set of filenames already in ChromaDB.
    Prevents re-ingesting the same PDF on restart.
    """
    try:
        results = collection.get(include=["metadatas"])
        sources = {m["source"] for m in results["metadatas"] if "source" in m}
        return sources
    except Exception:
        return set()


def ingest_pdf(pdf_path: str, collection: chromadb.Collection, force: bool = False) -> dict:
    """
    Ingests a single PDF into ChromaDB.

    Args:
        pdf_path: Path to the PDF file.
        collection: ChromaDB collection to add to.
        force: If True, re-ingests even if already present.

    Returns:
        Dict with ingestion result info.
    """
    filename = Path(pdf_path).name
    already_ingested = get_already_ingested_sources(collection)

    if filename in already_ingested and not force:
        logger.info(f"Skipping '{filename}' — already ingested.")
        return {"file": filename, "status": "skipped", "chunks": 0}

    # Extract and clean text
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        logger.warning(f"No text extracted from '{filename}'")
        return {"file": filename, "status": "empty", "chunks": 0}

    cleaned_text = clean_pdf_text(raw_text)

    # Create chunks with metadata
    chunks = create_chunks(text=cleaned_text, source_filename=filename)
    if not chunks:
        logger.warning(f"No chunks created from '{filename}'")
        return {"file": filename, "status": "no_chunks", "chunks": 0}

    # Batch embed all chunks
    model = get_embedding_model()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks from '{filename}'...")
    embeddings = model.encode(texts, show_progress_bar=False).tolist()

    # Build unique IDs for ChromaDB
    # Format: filename_chunkN (safe for ChromaDB)
    safe_name = filename.replace(" ", "_").replace(".", "_")
    ids = [f"{safe_name}_chunk{i}" for i in range(len(chunks))]

    # If re-ingesting, delete old entries first
    if filename in already_ingested and force:
        try:
            existing = collection.get(where={"source": filename})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
                logger.info(f"Deleted {len(existing['ids'])} old chunks for '{filename}'")
        except Exception as e:
            logger.warning(f"Could not delete old chunks: {e}")

    # Add to ChromaDB
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    logger.info(f"✓ Ingested '{filename}': {len(chunks)} chunks")
    return {"file": filename, "status": "ingested", "chunks": len(chunks)}


def ingest_all_documents(force: bool = False) -> list[dict]:
    """
    Ingests all PDFs found in the configured docs directory.

    Args:
        force: If True, re-ingests files even if already present.

    Returns:
        List of per-file ingestion results.
    """
    docs_dir = Path(settings.docs_dir)
    if not docs_dir.exists():
        logger.warning(f"Docs directory not found: {docs_dir}")
        return []

    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {docs_dir}")
        return []

    collection = get_chroma_collection()
    results = []

    for pdf_path in pdf_files:
        result = ingest_pdf(str(pdf_path), collection, force=force)
        results.append(result)

    total_ingested = sum(r["chunks"] for r in results if r["status"] == "ingested")
    logger.info(f"Ingestion complete. Total new chunks: {total_ingested}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = ingest_all_documents()
    for r in results:
        print(f"  {r['file']}: {r['status']} ({r['chunks']} chunks)")
