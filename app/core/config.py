"""
core/config.py — All configuration from .env
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

# Always look for .env next to this file's package root (finance-agent/),
# regardless of which directory uvicorn / pytest is launched from.
_HERE = Path(__file__).resolve().parent          # app/core/
_ENV_FILE = _HERE.parent.parent / ".env"         # finance-agent/.env


class Settings(BaseSettings):
    # App
    app_env:  str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    debug:    bool = Field(default=True)

    # Groq
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="llama-3.3-70b-versatile")

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

    # Auth (JWT)
    secret_key: str = Field(default="")
    algorithm:  str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=60 * 24)  # 24h
    refresh_token_expire_days: int = Field(default=30)

    # Database (Postgresql for sessions & analytics)
    database_url: str = Field(default="postgresql://postgres:postgres@localhost:5432/finance_agent")

    # CORS
    allowed_origins: str = Field(default="*")

    # ML — Risk classifier
    risk_model_path: str = Field(default="data/ml/risk_model.joblib")

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()