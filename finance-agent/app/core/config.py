"""
config.py — Central configuration using Pydantic BaseSettings.

WHY: Having all settings in one place means:
  - No scattered os.getenv() calls throughout the codebase
  - Type validation on startup (wrong type → crash early, not silently)
  - Easy to see every configurable value at a glance
  - Works with .env files automatically via python-dotenv
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Groq ──────────────────────────────────────────────
    groq_api_key: str
    groq_model: str = "llama3-70b-8192"

    # ── App ───────────────────────────────────────────────
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    debug: bool = True

    # ── RAG ───────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    docs_dir: str = "./docs"
    embedding_model: str = "all-MiniLM-L6-v2"
    rag_top_k: int = 4

    # ── Session ───────────────────────────────────────────
    session_ttl_minutes: int = 60
    max_sessions: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    
    lru_cache means Settings() is only instantiated once — 
    subsequent calls return the same object. This avoids 
    re-reading the .env file on every request.
    """
    return Settings()
