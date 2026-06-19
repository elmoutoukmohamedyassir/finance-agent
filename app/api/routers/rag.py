from fastapi import APIRouter, HTTPException, Query
from app.rag.ingestion import ingest_all_documents, get_chroma_collection
from app.rag.retriever import retrieve_raw
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag", tags=["RAG / Knowledge Base"])


@router.post("/ingest")
def ingest(force: bool = Query(default=False, description="Re-ingest already indexed files")):
    """
    Ingest all PDFs from the docs/ folder into ChromaDB.
    Safe to call multiple times — skips already-indexed files by default.
    Add new PDFs to docs/ and call this endpoint to make them searchable.
    """
    try:
        results = ingest_all_documents(force=force)
        ingested = [r for r in results if r["status"] == "ingested"]
        skipped = [r for r in results if r["status"] == "skipped"]
        return {
            "message": f"{len(ingested)} file(s) ingested, {len(skipped)} skipped.",
            "details": results,
        }
    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
def search(
    query: str = Query(..., min_length=3),
    top_k: int = Query(default=5, ge=1, le=10),
):
    """
    Search the knowledge base and see similarity scores.
    Use this to debug retrieval quality — are relevant chunks being found?
    """
    try:
        results = retrieve_raw(query=query, top_k=top_k)
        return {"query": query, "results_found": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def status():
    """Check how many documents and chunks are indexed."""
    try:
        collection = get_chroma_collection()
        count = collection.count()
        sources = []
        if count > 0:
            res = collection.get(include=["metadatas"])
            sources = list({m["source"] for m in res["metadatas"] if "source" in m})
        return {
            "status": "ready" if count > 0 else "empty",
            "total_chunks": count,
            "indexed_documents": sources,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
