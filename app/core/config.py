"""
core/config.py — All configuration from .env
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # App
    app_env:  str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug:    bool = Field(default=True)

    # Groq
    groq_api_key: str = Field(default="")
    groq_model:   str = Field(default="llama-3.3-70b-versatile")

    # Gemini (Fallback LLM)
    gemini_api_key: Optional[str] = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-pro")

    # RAG / Embeddings
    chroma_persist_dir: str   = Field(default="./data/chroma")
    docs_dir:           str   = Field(default="./docs")
    embedding_model:    str   = Field(default="nomic-embed-text")
    embedding_backend:  str   = Field(default="ollama")
    ollama_base_url:    str   = Field(default="http://localhost:11434")
    rag_top_k:          int   = Field(default=5)
    rag_confidence_threshold: float = Field(default=0.5)

    # Session
    session_ttl_minutes: int = Field(default=60)
    max_sessions:        int = Field(default=500)

    # Database (Postgresql for sessions & analytics)
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/finance_agent")

    # CORS
    allowed_origins: str = Field(default="*")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
