"""
rag/retriever.py — Semantic retrieval from ChromaDB.

IMPROVEMENTS over the original retriever.py:
  1. Relevance threshold filtering — low-similarity chunks are excluded
     (original returned anything, even irrelevant results)
  2. Source metadata included in formatted context (attribution)
  3. Singleton client and model (original re-instantiated on every call)
  4. Graceful fallback when collection is empty or query fails
  5. Returns structured result with scores for debugging
  6. Context formatted cleanly for LLM injection

WHY relevance thresholding matters:
  Without it, the LLM gets chunks that have nothing to do with the query,
  which can confuse it and increase hallucination risk.
  With it, we only inject context that's actually useful.
"""

import logging
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.rag.ingestion import get_embedding_model, COLLECTION_NAME

logger = logging.getLogger(__name__)
settings = get_settings()

# Relevance threshold: cosine similarity must be above this to be included.
# Range 0-1. 0.3 is a practical floor — below this the chunk is likely noise.
RELEVANCE_THRESHOLD = 0.30

# Cached ChromaDB client
_chroma_client: Optional[chromadb.ClientAPI] = None


def get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _chroma_client


def retrieve_context(query: str, top_k: Optional[int] = None) -> str:
    """
    Main retrieval function. Embeds the query, searches ChromaDB,
    filters by relevance, and returns formatted context string.

    Args:
        query: The user's question or a finance-related search query.
        top_k: Number of chunks to retrieve. Defaults to settings.rag_top_k.

    Returns:
        Formatted context string ready to inject into a prompt.
        Returns empty string if nothing relevant is found.
    """
    top_k = top_k or settings.rag_top_k

    try:
        raw_results = _query_chroma(query, top_k)
        if not raw_results:
            return ""

        relevant = _filter_by_relevance(raw_results)
        if not relevant:
            logger.info(f"RAG: no chunks above threshold for query: '{query[:60]}'")
            return ""

        context = _format_context(relevant)
        logger.info(f"RAG: retrieved {len(relevant)} relevant chunks for query: '{query[:60]}'")
        return context

    except Exception as e:
        logger.error(f"RAG retrieval failed: {e}")
        return ""  # Fail silently — agent can still answer without RAG context


def retrieve_raw(query: str, top_k: Optional[int] = None) -> list[dict]:
    """
    Returns raw retrieval results with scores. Useful for debugging
    and for the /rag/search API endpoint.
    """
    top_k = top_k or settings.rag_top_k
    raw = _query_chroma(query, top_k)
    return _filter_by_relevance(raw)


def _query_chroma(query: str, top_k: int) -> list[dict]:
    """
    Embed the query and search ChromaDB. Returns list of raw result dicts.
    """
    client = get_chroma_client()

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        logger.warning("RAG collection not found — have you ingested documents?")
        return []

    # Check collection isn't empty
    count = collection.count()
    if count == 0:
        logger.warning("RAG collection is empty — ingest documents first.")
        return []

    # Embed the query using the same model used at ingestion time
    model = get_embedding_model()
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, count),  # can't request more than we have
        include=["documents", "metadatas", "distances"],
    )

    # Unpack ChromaDB's nested list structure
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # ChromaDB cosine distance: 0 = identical, 2 = opposite
    # Convert to similarity: similarity = 1 - (distance / 2)
    return [
        {
            "text": doc,
            "source": meta.get("source", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
            "similarity": round(1 - (dist / 2), 4),
        }
        for doc, meta, dist in zip(docs, metadatas, distances)
    ]


def _filter_by_relevance(results: list[dict]) -> list[dict]:
    """Filter out chunks below the relevance threshold."""
    filtered = [r for r in results if r["similarity"] >= RELEVANCE_THRESHOLD]
    logger.debug(
        f"Relevance filter: {len(results)} → {len(filtered)} chunks "
        f"(threshold={RELEVANCE_THRESHOLD})"
    )
    return filtered


def _format_context(results: list[dict]) -> str:
    """
    Formats retrieved chunks into a clean string for LLM injection.

    Format:
      [Source: filename.pdf]
      <chunk text>

      [Source: filename.pdf]
      <chunk text>
    """
    parts = []
    for r in results:
        parts.append(f"[Source: {r['source']}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)
