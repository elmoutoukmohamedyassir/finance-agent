"""
api/routers/rag.py — RAG document management endpoints.

Endpoints:
  POST /rag/ingest  — Trigger PDF ingestion from the docs/ folder
  GET  /rag/search  — Search the knowledge base directly
  GET  /rag/status  — Check how many documents are indexed
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from app.rag.ingestion import ingest_all_documents, get_chroma_collection
from app.rag.retriever import retrieve_raw
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG / Knowledge Base"])


@router.post("/ingest")
def ingest_documents(force: bool = Query(default=False, description="Re-ingest already-indexed files")):
    """
    Ingest all PDF files from the `docs/` directory into ChromaDB.

    - Skips already-indexed files by default (set `force=true` to re-index)
    - Safe to call multiple times (idempotent by default)
    - Drop new PDFs into the `docs/` folder and call this endpoint
    """
    try:
        results = ingest_all_documents(force=force)
        ingested = [r for r in results if r["status"] == "ingested"]
        skipped = [r for r in results if r["status"] == "skipped"]
        return {
            "message": f"Ingestion complete. {len(ingested)} file(s) ingested, {len(skipped)} skipped.",
            "details": results,
        }
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/search")
def search_knowledge_base(
    query: str = Query(..., min_length=3, description="Search query"),
    top_k: int = Query(default=4, ge=1, le=10),
):
    """
    Search the finance knowledge base directly.
    Returns relevant chunks with similarity scores.
    Useful for debugging RAG quality.
    """
    try:
        results = retrieve_raw(query=query, top_k=top_k)
        return {
            "query": query,
            "results_found": len(results),
            "results": results,
        }
    except Exception as e:
        logger.error(f"RAG search error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Search failed")


@router.get("/status")
def rag_status():
    """
    Returns the current state of the knowledge base.
    """
    try:
        collection = get_chroma_collection()
        count = collection.count()

        # Get list of source documents
        if count > 0:
            results = collection.get(include=["metadatas"])
            sources = list({m["source"] for m in results["metadatas"] if "source" in m})
        else:
            sources = []

        return {
            "status": "ready" if count > 0 else "empty",
            "total_chunks": count,
            "indexed_documents": sources,
            "message": (
                f"{count} chunks from {len(sources)} document(s) indexed."
                if count > 0
                else "No documents indexed yet. POST /rag/ingest to add documents."
            ),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
