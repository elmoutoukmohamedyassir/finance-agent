"""
rag/retriever.py — Semantic search over ChromaDB.

Filters results by relevance score — chunks below the threshold
are discarded so we never inject irrelevant context into the LLM prompt.
"""

import logging
from typing import Optional

import chromadb

from app.core.config import get_settings
from app.rag.embedder import get_embedder
from app.rag.ingestion import COLLECTION_NAME

logger = logging.getLogger(__name__)
settings = get_settings()

RELEVANCE_THRESHOLD = 0.32  # cosine similarity floor — below this = noise

_chroma_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def retrieve_context(query: str, top_k: Optional[int] = None, confidence_threshold: Optional[float] = None) -> tuple[str, float, bool]:
    """
    Embed the query → search ChromaDB → filter by relevance → format for LLM.
    Returns (context_text, confidence_score, should_use_fallback).
    
    Args:
        query: User question
        top_k: Number of results to retrieve
        confidence_threshold: Min average similarity to use RAG (default from config)
    
    Returns:
        (context: str, confidence: float, use_fallback: bool)
    """
    threshold = confidence_threshold or settings.rag_confidence_threshold
    results = retrieve_raw(query, top_k)
    
    if not results:
        return "", 0.0, True
    
    # Calculate average confidence
    avg_confidence = sum(r["similarity"] for r in results) / len(results)
    should_fallback = avg_confidence < threshold
    
    context = _format_context(results)
    logger.info(f"RAG confidence: {avg_confidence:.2f} (threshold: {threshold}, fallback: {should_fallback})")
    
    return context, avg_confidence, should_fallback


def retrieve_raw(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Returns scored results. Used by /rag/search endpoint for debugging."""
    k = top_k or settings.rag_top_k
    try:
        client = _get_client()
        try:
            collection = client.get_collection(name=COLLECTION_NAME)
        except Exception:
            logger.warning("RAG collection not found. Run ingest.py first.")
            return []

        count = collection.count()
        if count == 0:
            logger.warning("RAG collection is empty.")
            return []

        embedder = get_embedder()
        query_vec = embedder.embed_one(query)

        results = collection.query(
            query_embeddings=[query_vec],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # cosine distance from chromadb: 0=identical → similarity = 1 - dist/2
        raw = [
            {
                "text": doc,
                "source": meta.get("source", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "similarity": round(1 - (d / 2), 4),
            }
            for doc, meta, d in zip(docs, metas, dists)
        ]

        filtered = [r for r in raw if r["similarity"] >= RELEVANCE_THRESHOLD]
        logger.info(
            f"RAG: '{query[:50]}' → {len(raw)} results → {len(filtered)} above threshold"
        )
        return filtered

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


def _format_context(results: list[dict]) -> str:
    parts = [f"[Source: {r['source']} | Relevance: {r['similarity']}]\n{r['text']}" for r in results]
    return "\n\n---\n\n".join(parts)
