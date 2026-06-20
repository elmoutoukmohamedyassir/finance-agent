"""
main.py — FastAPI application entry point.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import chat, metrics, scenario, rag
from app.core.config import get_settings
from app.services.session_service import get_session_count

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 50)
    logger.info("AI SaaS Finance Agent starting up")
    logger.info(f"  LLM Model  : {settings.groq_model}")
    logger.info(f"  Embedder   : {settings.embedding_backend} / {settings.embedding_model}")
    logger.info(f"  Docs dir   : {settings.docs_dir}")
    logger.info(f"  ChromaDB   : {settings.chroma_persist_dir}")
    logger.info("=" * 50)

    # Auto-ingest PDFs in docs/ on every startup (skips already-indexed)
    try:
        from app.rag.ingestion import ingest_all_documents
        results = ingest_all_documents()
        new_files = [r for r in results if r["status"] == "ingested"]
        if new_files:
            logger.info(f"  ✓ Ingested {len(new_files)} new document(s)")
        else:
            logger.info("  ✓ No new documents to ingest")
    except Exception as e:
        logger.warning(f"  ⚠ Auto-ingestion failed: {e}")
        logger.warning("    The agent will work without RAG context.")
        logger.warning("    Fix the embedding setup and run: POST /api/v1/rag/ingest")

    yield
    logger.info("Finance Agent shutting down.")


app = FastAPI(
    title="AI SaaS Finance Agent",
    description="AI-powered SaaS finance advisor. Analyzes metrics, projects scenarios, and answers finance questions grounded in your documents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
)

PREFIX = "/api/v1"
app.include_router(chat.router, prefix=PREFIX)
app.include_router(metrics.router, prefix=PREFIX)
app.include_router(scenario.router, prefix=PREFIX)
app.include_router(rag.router, prefix=PREFIX)


@app.get("/", tags=["Health"])
def root():
    return {
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs",
        "active_sessions": get_session_count(),
    }


@app.get("/api/v1/health", tags=["Health"])
def health():
    from app.rag.ingestion import get_chroma_collection
    try:
        col = get_chroma_collection()
        chunks = col.count()
        rag_status = "ready" if chunks > 0 else "empty — add PDFs to docs/ and POST /api/v1/rag/ingest"
    except Exception:
        chunks = 0
        rag_status = "error"

    return {
        "status": "healthy",
        "llm": {"provider": "groq", "model": settings.groq_model},
        "embedder": {"backend": settings.embedding_backend, "model": settings.embedding_model},
        "rag": {"status": rag_status, "chunks": chunks},
        "sessions": {"active": get_session_count()},
    }