from pydantic import model_validator
from pydantic_settings import BaseSettings
from functools import lru_cache

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
    embedding_model: str = "nomic-embed-text"
    embedding_backend: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    rag_top_k: int = 5

    # ── Session ───────────────────────────────────────────────────────────
    session_ttl_minutes: int = 60
    max_sessions: int = 500

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",    # ← silently ignores APP_ENV and anything else not declared
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()