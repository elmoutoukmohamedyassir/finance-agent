"""
services/session_service.py — In-memory session storage.

WHY in-memory instead of a database:
  - Zero setup — no Redis, no Postgres needed for a student project
  - Sufficient for demos and PFE purposes
  - Sessions expire automatically via TTL cleanup
  - Easy to swap for Redis later (same interface, just change the backend)

LIMITATION: Sessions are lost on server restart. This is acceptable for
development and demo purposes. For production, swap to Redis.

HOW IT WORKS:
  - Sessions stored in a dict: {session_id: ConversationSession}
  - get_or_create() returns existing session or makes a new one
  - cleanup_expired() is called on each get to remove old sessions
  - Thread safety: Python's GIL makes dict ops safe for single-process FastAPI
"""

import uuid
import logging
from datetime import datetime, timedelta

from app.schemas.session import ConversationSession, BusinessState
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# The in-memory store — just a plain dict
_sessions: dict[str, ConversationSession] = {}


def get_or_create_session(session_id: str | None) -> ConversationSession:
    """
    Returns an existing session if session_id is provided and valid,
    otherwise creates a new session with a fresh UUID.

    Args:
        session_id: The client-provided session ID, or None for a new session.

    Returns:
        A ConversationSession object (new or existing).
    """
    # Clean up old sessions first (lightweight housekeeping)
    _cleanup_expired_sessions()

    if session_id and session_id in _sessions:
        session = _sessions[session_id]
        logger.debug(f"Loaded existing session: {session_id}")
        return session

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(session_id=new_id)
    _sessions[new_id] = session
    logger.info(f"Created new session: {new_id}")
    return session


def save_session(session: ConversationSession) -> None:
    """
    Persist updates to a session back to the store.
    Simple assignment since we're working with in-memory dict.
    """
    session.updated_at = datetime.utcnow()
    _sessions[session.session_id] = session


def update_business_state(session: ConversationSession, extracted_info: dict) -> None:
    """
    Merges newly extracted business info into the session's business_state.
    
    WHY: Instead of replacing the whole state, we only update fields that
    were just provided. This way earlier information is never lost.
    
    Args:
        session: The current session.
        extracted_info: Dict of field_name → value extracted from the latest message.
    """
    state = session.business_state

    field_map = {
        "business_name": (str, "business_name"),
        "business_model": (str, "business_model"),
        "target_audience": (str, "target_audience"),
        "funding_stage": (str, "funding_stage"),
        "mrr": (float, "mrr"),
        "arr": (float, "arr"),
        "customer_count": (int, "customer_count"),
        "arpu": (float, "arpu"),
        "pricing_plan": (str, "pricing_plan"),
        "churn_rate": (float, "churn_rate"),
        "new_customers_per_month": (int, "new_customers_per_month"),
        "growth_rate": (float, "growth_rate"),
        "monthly_costs": (float, "monthly_costs"),
        "marketing_budget": (float, "marketing_budget"),
        "cac": (float, "cac"),
        "gross_margin": (float, "gross_margin"),
    }

    for field, value in extracted_info.items():
        if field in field_map and value is not None:
            type_cast, attr_name = field_map[field]
            try:
                setattr(state, attr_name, type_cast(value))
                logger.debug(f"Updated business_state.{attr_name} = {value}")
            except (ValueError, TypeError):
                logger.warning(f"Could not cast {field}={value} to {type_cast}")

    session.business_state = state


def _cleanup_expired_sessions() -> None:
    """
    Remove sessions older than SESSION_TTL_MINUTES.
    Also enforces MAX_SESSIONS cap by removing oldest sessions.
    
    Called on every session access — lightweight since Python dict iteration is fast.
    """
    ttl = timedelta(minutes=settings.session_ttl_minutes)
    now = datetime.utcnow()
    expired = [
        sid for sid, session in _sessions.items()
        if now - session.updated_at > ttl
    ]
    for sid in expired:
        del _sessions[sid]
        logger.debug(f"Expired session: {sid}")

    # Enforce max cap: remove oldest if over limit
    if len(_sessions) > settings.max_sessions:
        sorted_sessions = sorted(_sessions.items(), key=lambda x: x[1].updated_at)
        to_remove = sorted_sessions[:len(_sessions) - settings.max_sessions]
        for sid, _ in to_remove:
            del _sessions[sid]
            logger.warning(f"Evicted session due to cap: {sid}")


def get_session_count() -> int:
    """Returns current number of active sessions. Useful for health checks."""
    return len(_sessions)
