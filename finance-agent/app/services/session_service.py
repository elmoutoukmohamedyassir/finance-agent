"""
services/session_service.py — In-memory session storage.

Sessions live here on the server. The client only needs to send session_id.
State is NOT managed by the client (unlike the original design).
"""

import uuid
import logging
from datetime import datetime, timedelta

from app.schemas.session import ConversationSession, BusinessState
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_sessions: dict[str, ConversationSession] = {}


def get_or_create_session(session_id: str | None) -> ConversationSession:
    """Returns existing session or creates a new one."""
    _cleanup_expired()

    if session_id and session_id in _sessions:
        return _sessions[session_id]

    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(session_id=new_id)
    _sessions[new_id] = session
    logger.info(f"New session: {new_id}")
    return session


def save_session(session: ConversationSession) -> None:
    session.updated_at = datetime.utcnow()
    _sessions[session.session_id] = session


def update_business_state(session: ConversationSession, extracted: dict) -> None:
    """
    Merges extracted data into the session's BusinessState.
    Only updates fields that were just found — never overwrites with None.
    """
    state = session.business_state
    type_map = {
        "business_name": str, "target_audience": str, "business_model": str,
        "funding_stage": str, "pricing_plan": str,
        "mrr": float, "arr": float, "arpu": float,
        "monthly_costs": float, "marketing_budget": float, "cac": float,
        "churn_rate": float, "growth_rate": float, "gross_margin": float,
        "customer_count": int, "new_customers_per_month": int,
    }
    for field, value in extracted.items():
        if field in type_map and value is not None:
            try:
                setattr(state, field, type_map[field](value))
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not set {field}={value}: {e}")
    session.business_state = state


def get_session_count() -> int:
    return len(_sessions)


def _cleanup_expired() -> None:
    ttl = timedelta(minutes=settings.session_ttl_minutes)
    now = datetime.utcnow()
    expired = [sid for sid, s in _sessions.items() if now - s.updated_at > ttl]
    for sid in expired:
        del _sessions[sid]
    if len(_sessions) > settings.max_sessions:
        oldest = sorted(_sessions.items(), key=lambda x: x[1].updated_at)
        for sid, _ in oldest[:len(_sessions) - settings.max_sessions]:
            del _sessions[sid]
