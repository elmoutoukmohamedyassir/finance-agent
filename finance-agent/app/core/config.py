"""
config.py — All configuration in one place, loaded from your .env file.

Why this exists:
  Instead of scattered os.getenv() calls throughout the code, everything
  is defined here with types and defaults. If a required value is missing
  the app crashes on startup with a clear error — not silently mid-request.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):

    # ── Groq API ──────────────────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama3-70b-8192"

    # ── App ───────────────────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ── RAG ───────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    docs_dir: str = "./docs"
    # nomic-embed-text produces much richer embeddings than all-MiniLM-L6-v2.
    # It understands domain language better → more relevant retrieval.
    # Requires Ollama running locally: `ollama pull nomic-embed-text`
    # To use sentence-transformers instead, set: all-MiniLM-L6-v2
    embedding_model: str = "nomic-embed-text"
    embedding_backend: str = "ollama"   # "ollama" or "sentence-transformers"
    ollama_base_url: str = "http://localhost:11434"
    rag_top_k: int = 5

    # ── Session ───────────────────────────────────────────────────────────
    session_ttl_minutes: int = 60
    max_sessions: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
