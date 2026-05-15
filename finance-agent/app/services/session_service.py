"""
services/session_service.py — In-memory session storage.

Sessions live here on the server. The client only needs to send session_id.
State is NOT managed by the client.

CHANGE FROM ORIGINAL:
  update_business_state() previously had a hardcoded type_map with SaaS fields
  (mrr, arr, arpu, churn_rate, cac...). Replaced with dynamic field mapping
  driven by BusinessState's own schema — no hardcoding, always in sync.
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Any

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
    Uses BusinessState's Pydantic schema as the source of truth for field types.
    Only updates fields that were just found — never overwrites existing with None.

    Replaces the old hardcoded SaaS type_map with dynamic reflection.
    """
    state = session.business_state

    # Get field types directly from the Pydantic schema
    schema_fields = BusinessState.model_fields

    for field_name, value in extracted.items():
        if value is None:
            continue
        if field_name not in schema_fields:
            logger.debug(f"Ignoring unknown field '{field_name}' from extraction")
            continue

        # Only update if field is currently None (don't overwrite confirmed data)
        current = getattr(state, field_name, None)
        if current is not None:
            continue

        # Coerce to the correct type based on Pydantic annotation
        try:
            coerced = _coerce(value, schema_fields[field_name].annotation)
            setattr(state, field_name, coerced)
            logger.debug(f"Set {field_name} = {coerced}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not set {field_name}={value!r}: {e}")

    session.business_state = state


def load_hypothesis_output(session: ConversationSession, hypothesis_json: dict) -> None:
    """
    Load a full HypothesisOutput JSON directly into the session's BusinessState.
    Used when the Finance Agent is called programmatically by the Hypothesis Agent.

    The hypothesis_ingestor translates H-variables → BusinessState fields.
    This is the clean agent-to-agent communication path.
    """
    from app.schemas.hypothesis_output import HypothesisOutput
    from app.tools.hypothesis_ingestor import ingest_hypothesis

    try:
        hypothesis = HypothesisOutput.model_validate(hypothesis_json)
        financial_data, derived, proj_inputs = ingest_hypothesis(hypothesis)

        # Store the full FinancialData fields into BusinessState
        state = session.business_state
        state.entity_type = "corporate"
        state.entity_name = financial_data.entity_name
        state.sector = financial_data.sector
        state.total_revenue = financial_data.total_revenue
        state.cost_of_goods_sold = financial_data.cost_of_goods_sold
        state.operating_expenses = financial_data.operating_expenses
        state.salaries_and_benefits = financial_data.salaries_and_benefits
        state.depreciation_amortization = financial_data.depreciation_amortization
        state.interest_expense = financial_data.interest_expense
        state.total_assets = financial_data.total_assets
        state.total_equity = financial_data.total_equity
        state.total_debt = financial_data.total_debt
        state.cash_inflow = financial_data.cash_inflow
        state.cash_outflow = financial_data.cash_outflow
        state.own_capital_invested = financial_data.own_capital_invested
        state.external_funding = financial_data.external_funding

        # Store derived and projection inputs in session for plan generator
        session.derived_variables = derived
        session.projection_inputs = proj_inputs
        session.business_state = state

        logger.info(f"HypothesisOutput loaded into session {session.session_id}")
    except Exception as e:
        logger.error(f"Failed to load HypothesisOutput: {e}")
        raise


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
        for sid, _ in oldest[: len(_sessions) - settings.max_sessions]:
            del _sessions[sid]


def _coerce(value: Any, annotation) -> Any:
    """
    Coerce a value to the annotated type.
    Handles Optional[X] by unwrapping to X.
    """
    import typing
    origin = getattr(annotation, "__origin__", None)

    # Handle Optional[X] = Union[X, None]
    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args:
            annotation = args[0]

    if annotation is int:
        return int(float(value))
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    if annotation is bool:
        return bool(value)
    return value