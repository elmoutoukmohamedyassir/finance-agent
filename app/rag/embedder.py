"""
rag/embedder.py — Embedding model abstraction.

WHY nomic-embed-text over all-MiniLM-L6-v2:

  all-MiniLM-L6-v2:
    - 384 dimensions
    - Trained on general text (Wikipedia, news)
    - Good for general semantic similarity
    - Fast, tiny model

  nomic-embed-text:
    - 768 dimensions (2× richer representation)
    - Trained specifically on longer documents with better domain coverage
    - Understands financial/business language better
    - Supports 8192 token context (vs 512 for MiniLM)
    - Consistently outperforms MiniLM on retrieval benchmarks

  For finance PDFs (long, domain-specific), nomic-embed-text gives
  significantly more relevant retrieval — fewer hallucinations from bad context.

HOW TO USE nomic-embed-text:
  1. Install Ollama: https://ollama.com
  2. Run: ollama pull nomic-embed-text
  3. Make sure Ollama is running: ollama serve
  4. Set in .env: EMBEDDING_BACKEND=ollama, EMBEDDING_MODEL=nomic-embed-text

FALLBACK:
  If Ollama is not available, the system falls back to sentence-transformers
  (all-MiniLM-L6-v2). Set EMBEDDING_BACKEND=sentence-transformers in .env.

IMPORTANT: You must use the same embedding model for BOTH ingestion AND retrieval.
If you switch models, delete data/chroma/ and re-run ingest.py.
"""

import logging
from typing import Protocol
import requests

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class Embedder(Protocol):
    """Interface that all embedders must implement."""
    def embed(self, texts: list[str]) -> list[list[float]]: ...
    def embed_one(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """
    Embedding via Ollama running locally.
    Supports nomic-embed-text, mxbai-embed-large, and others.

    FIX: Uses /api/embed (batch endpoint) instead of the removed /api/embeddings.
    Ollama v0.1.26+ deprecated /api/embeddings in favour of /api/embed.
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._test_connection()

    def _test_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)
            if not available:
                # Hard stop — don't let it silently fail later during embed()
                raise RuntimeError(
                    f"\n\n  ✗ Model '{self.model}' is NOT available in Ollama.\n"
                    f"  Available models: {models}\n\n"
                    f"  Fix: run   ollama pull {self.model}\n"
                    f"  Then re-run ingest.py.\n\n"
                    f"  Or switch to sentence-transformers (no Ollama needed):\n"
                    f"  Set in .env:  EMBEDDING_BACKEND=sentence-transformers\n"
                    f"               EMBEDDING_MODEL=all-MiniLM-L6-v2\n"
                )
            logger.info(f"Ollama connection OK. Using model: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"\n\n  ✗ Cannot connect to Ollama at {self.base_url}.\n"
                f"  Make sure Ollama is running:  ollama serve\n\n"
                f"  Or switch to sentence-transformers (no Ollama needed):\n"
                f"  Set in .env:  EMBEDDING_BACKEND=sentence-transformers\n"
                f"               EMBEDDING_MODEL=all-MiniLM-L6-v2\n"
            )

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed texts in small batches to avoid Ollama timeouts on large documents.

        Sending 2000+ chunks in one request times out after 120s.
        Processing in batches of 32 keeps each request well under the timeout.

        NOTE: /api/embeddings (old, single-text) was removed in Ollama v0.1.26+.
              /api/embed (new, accepts a list) is the correct endpoint.
        """
        all_embeddings = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total + batch_size - 1) // batch_size
            logger.info(f"  Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

            try:
                resp = requests.post(
                    f"{self.base_url}/api/embed",
                    json={"model": self.model, "input": batch},
                    timeout=120,
                )
                if not resp.ok:
                    # Log the actual Ollama error body before raising
                    logger.error(f"Ollama error body: {resp.text}")
                resp.raise_for_status()
                all_embeddings.extend(resp.json()["embeddings"])
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 400:
                    # A chunk in this batch is bad (empty, too long, invalid chars).
                    # Fall back to embedding one-by-one so we can isolate and skip it.
                    logger.warning(
                        f"Batch {batch_num} got 400 — falling back to one-by-one to find bad chunk..."
                    )
                    for j, text in enumerate(batch):
                        if not text or not text.strip():
                            logger.warning(f"  Skipping empty chunk at batch {batch_num} index {j}")
                            all_embeddings.append([0.0] * 768)  # nomic-embed-text dim
                            continue
                        try:
                            single_resp = requests.post(
                                f"{self.base_url}/api/embed",
                                json={"model": self.model, "input": [text]},
                                timeout=30,
                            )
                            single_resp.raise_for_status()
                            all_embeddings.extend(single_resp.json()["embeddings"])
                        except Exception as single_e:
                            logger.warning(
                                f"  Skipping bad chunk at batch {batch_num} index {j}: {single_e}\n"
                                f"  Chunk preview: {repr(text[:200])}"
                            )
                            all_embeddings.append([0.0] * 768)  # placeholder, won't be retrieved
                else:
                    logger.error(f"Ollama embedding failed on batch {batch_num}: {e}")
                    raise RuntimeError(f"Embedding failed at batch {batch_num}/{total_batches}: {e}")
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama embedding failed on batch {batch_num}: {e}")
                raise RuntimeError(f"Embedding failed at batch {batch_num}/{total_batches}: {e}")

        return all_embeddings


class SentenceTransformerEmbedder:
    """
    Fallback embedder using sentence-transformers.
    Works without Ollama. Weaker for domain-specific finance text.
    Install: pip install sentence-transformers
    """

    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Run: pip install sentence-transformers"
            )
        logger.info(f"Loading sentence-transformers model: {model}")
        self._model = SentenceTransformer(model)
        logger.info("Sentence-transformers model loaded.")

    def embed_one(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()


# ── Singleton factory ─────────────────────────────────────────────────────────

_embedder_instance: Embedder | None = None


def get_embedder() -> Embedder:
    """
    Returns the configured embedder (singleton — loaded once).

    Reads EMBEDDING_BACKEND from .env:
      "ollama"                → OllamaEmbedder (nomic-embed-text, recommended)
      "sentence-transformers" → SentenceTransformerEmbedder (fallback, no Ollama needed)
    """
    global _embedder_instance
    if _embedder_instance is not None:
        return _embedder_instance

    backend = settings.embedding_backend.lower()
    model = settings.embedding_model

    if backend == "ollama":
        logger.info(f"Using Ollama embedder: {model}")
        _embedder_instance = OllamaEmbedder(model=model, base_url=settings.ollama_base_url)
    elif backend == "sentence-transformers":
        logger.info(f"Using sentence-transformers embedder: {model}")
        _embedder_instance = SentenceTransformerEmbedder(model=model)
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND='{backend}'. "
            "Use 'ollama' or 'sentence-transformers'."
        )

    return _embedder_instance