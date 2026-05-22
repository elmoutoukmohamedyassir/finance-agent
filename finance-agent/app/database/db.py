"""
database/db.py — SQLAlchemy engine and session management.
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from pathlib import Path
import logging

from app.core.config import get_settings
from app.database.models import Base

logger = logging.getLogger(__name__)
settings = get_settings()

# Create engine with SQLite (or other DB if DATABASE_URL configured)
db_url = getattr(settings, "db_path", "./data/finance_agent.db")
if not db_url.startswith("sqlite"):
    # Assume it's a file path, convert to sqlite URL
    db_url = f"sqlite:///{db_url}"

engine = create_engine(
    db_url,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized")


def get_db() -> Generator[Session, None, None]:
    """Get a database session (dependency injection)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()
