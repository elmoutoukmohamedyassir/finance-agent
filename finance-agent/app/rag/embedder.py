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
    """

    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._test_connection()

    def _test_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code != 200:
                raise ConnectionError(f"Ollama returned {resp.status_code}")
            models = [m["name"] for m in resp.json().get("models", [])]
            # Check if our model is available (name might have :latest suffix)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)
            if not available:
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}\n"
                    f"Available: {models}"
                )
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running: ollama serve"
            )

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Ollama doesn't support true batching so we loop."""
        embeddings = []
        for text in texts:
            try:
                resp = requests.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                    timeout=30,
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            except requests.exceptions.RequestException as e:
                logger.error(f"Ollama embedding failed: {e}")
                raise RuntimeError(f"Embedding failed: {e}")
        return embeddings


class SentenceTransformerEmbedder:
    """
    Fallback embedder using sentence-transformers.
    Works without Ollama. Weaker for domain-specific finance text.
    """

    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer
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
      "sentence-transformers" → SentenceTransformerEmbedder (fallback)
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
