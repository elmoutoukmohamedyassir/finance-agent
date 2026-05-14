"""
main.py — FastAPI application entry point.

IMPROVEMENTS over the original:
  1. Routers registered cleanly (not everything in one file)
  2. CORS configured for frontend development
  3. Startup event auto-ingests documents from docs/
  4. Health check endpoint with session count + RAG status
  5. Consistent API versioning under /api/v1
  6. Proper logging setup
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import chat, metrics, scenario, rag
from app.core.config import get_settings
from app.services.session_service import get_session_count

# ── Logging Setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
settings = get_settings()


# ── Startup / Shutdown ─────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on startup and shutdown.
    
    On startup: auto-ingest any PDFs in the docs/ folder.
    This means dropping a PDF into docs/ and restarting the server
    is all you need to add new knowledge to the RAG system.
    """
    logger.info("🚀 Finance Agent starting up...")
    logger.info(f"   Model: {settings.groq_model}")
    logger.info(f"   Docs dir: {settings.docs_dir}")
    logger.info(f"   ChromaDB: {settings.chroma_persist_dir}")

    # Auto-ingest documents on startup (skips already-indexed files)
    try:
        from app.rag.ingestion import ingest_all_documents
        results = ingest_all_documents()
        ingested_count = sum(1 for r in results if r["status"] == "ingested")
        if ingested_count:
            logger.info(f"   ✓ Ingested {ingested_count} new document(s)")
        else:
            logger.info("   ✓ No new documents to ingest")
    except Exception as e:
        logger.warning(f"   ⚠ Document ingestion failed on startup: {e}")

    yield  # App is running

    logger.info("Finance Agent shutting down.")


# ── App Creation ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI SaaS Finance Agent",
    description=(
        "An AI-powered finance advisor specialized in SaaS business analysis. "
        "Helps founders evaluate financial health, calculate metrics, and "
        "simulate business scenarios."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI at /docs
    redoc_url="/redoc",    # ReDoc at /redoc
)

# ── CORS ───────────────────────────────────────────────────────────────────────
# Allow any origin in development. Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
API_PREFIX = "/api/v1"

app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(metrics.router, prefix=API_PREFIX)
app.include_router(scenario.router, prefix=API_PREFIX)
app.include_router(rag.router, prefix=API_PREFIX)


# ── Health Check ───────────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def health_check():
    """Health check — returns server status and basic stats."""
    return {
        "status": "running",
        "version": "2.0.0",
        "model": settings.groq_model,
        "active_sessions": get_session_count(),
    }


@app.get("/api/v1/health", tags=["Health"])
def detailed_health():
    """Detailed health check including RAG system status."""
    from app.rag.ingestion import get_chroma_collection
    try:
        collection = get_chroma_collection()
        rag_chunks = collection.count()
        rag_status = "ready" if rag_chunks > 0 else "empty"
    except Exception:
        rag_chunks = 0
        rag_status = "error"

    return {
        "status": "healthy",
        "components": {
            "llm": {"provider": "groq", "model": settings.groq_model, "status": "configured"},
            "rag": {"status": rag_status, "chunks_indexed": rag_chunks},
            "sessions": {"active": get_session_count(), "max": settings.max_sessions},
        }
    }
