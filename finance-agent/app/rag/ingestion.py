"""
rag/ingestion.py — PDF → chunks → embeddings → ChromaDB.

Uses the embedder abstraction so you can swap between
nomic-embed-text (Ollama) and sentence-transformers in .env.
"""

import logging
from pathlib import Path

from pypdf import PdfReader
import chromadb

from app.core.config import get_settings
from app.rag.chunker import create_chunks, clean_pdf_text
from app.rag.embedder import get_embedder

logger = logging.getLogger(__name__)
settings = get_settings()
COLLECTION_NAME = "saas_finance_docs"


def get_chroma_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(pdf_path: str) -> str:
    try:
        reader = PdfReader(pdf_path)
        pages = []
        for page in reader.pages:
            t = page.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        logger.warning(f"PDF read failed for {pdf_path}: {e}")
        return ""


def get_ingested_sources(collection: chromadb.Collection) -> set[str]:
    try:
        result = collection.get(include=["metadatas"])
        return {m["source"] for m in result["metadatas"] if "source" in m}
    except Exception:
        return set()


def ingest_pdf(pdf_path: str, collection: chromadb.Collection, force: bool = False) -> dict:
    filename = Path(pdf_path).name
    ingested = get_ingested_sources(collection)

    if filename in ingested and not force:
        logger.info(f"Skipping (already ingested): {filename}")
        return {"file": filename, "status": "skipped", "chunks": 0}

    raw = extract_text_from_pdf(pdf_path)
    if not raw.strip():
        return {"file": filename, "status": "empty", "chunks": 0}

    text = clean_pdf_text(raw)
    chunks = create_chunks(text=text, source_filename=filename)
    if not chunks:
        return {"file": filename, "status": "no_chunks", "chunks": 0}

    embedder = get_embedder()
    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    logger.info(f"Embedding {len(texts)} chunks from '{filename}'...")
    embeddings = embedder.embed(texts)

    safe = filename.replace(" ", "_").replace(".", "_").replace("(", "").replace(")", "")
    ids = [f"{safe}_chunk{i}" for i in range(len(chunks))]

    # Delete old entries if force re-ingesting
    if filename in ingested and force:
        try:
            old = collection.get(where={"source": filename})
            if old["ids"]:
                collection.delete(ids=old["ids"])
        except Exception as e:
            logger.warning(f"Could not delete old entries: {e}")

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    logger.info(f"✓ Ingested '{filename}': {len(chunks)} chunks")
    return {"file": filename, "status": "ingested", "chunks": len(chunks)}


def ingest_all_documents(force: bool = False) -> list[dict]:
    docs_dir = Path(settings.docs_dir)
    if not docs_dir.exists():
        logger.warning(f"docs/ directory not found: {docs_dir}")
        return []

    pdfs = list(docs_dir.glob("*.pdf"))
    if not pdfs:
        logger.info(f"No PDFs found in {docs_dir}. Drop PDFs there and re-run.")
        return []

    collection = get_chroma_collection()
    results = []
    for pdf in pdfs:
        result = ingest_pdf(str(pdf), collection, force=force)
        results.append(result)

    total = sum(r["chunks"] for r in results if r["status"] == "ingested")
    logger.info(f"Ingestion done. New chunks: {total}")
    return results
