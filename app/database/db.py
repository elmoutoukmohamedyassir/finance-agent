"""
database/db.py — SQLAlchemy engine and session management.
"""
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
import logging

from app.core.config import get_settings
from app.database.models import Base

logger = logging.getLogger(__name__)
settings = get_settings()

# PostgreSQL connection URL — expects DATABASE_URL in settings, e.g.:
#   postgresql://user:password@localhost:5432/finance_agent
# Or split credentials via individual settings fields (see below).
db_url = settings.database_url  # e.g. "postgresql://user:pass@host:5432/dbname"

engine = create_engine(
    db_url,
    # PostgreSQL connection pool settings
    pool_size=5,           # Number of persistent connections kept in the pool
    max_overflow=10,       # Extra connections allowed above pool_size under load
    pool_pre_ping=True,    # Verify connection health before each checkout (handles dropped connections)
    pool_recycle=1800,     # Recycle connections after 30 minutes (avoids stale connections)
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