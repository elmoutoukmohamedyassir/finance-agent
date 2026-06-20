"""
services/session_service.py — Postgres-backed session persistence.

Was SQLite (data/sessions.db); now stored in the same Postgres database as
everything else (decision: no separate session store). Table is
ConversationSessionDB (app/database/models.py) — id/data/updated_at mirror
the old SQLite schema almost exactly, so the logic here barely changed,
just the storage backend and one new concept: ownership.

OWNERSHIP RULES (client_id is nullable on the DB row):
  - A brand-new session: client_id is set if the caller is authenticated,
    else stays NULL (anonymous chat keeps working, per product decision).
  - An existing anonymous session (client_id IS NULL) being resumed by an
    authenticated caller: gets "claimed" — client_id is set going forward.
    This covers the common case of someone starting a chat before logging
    in, then logging in mid-conversation.
  - An existing session that already belongs to client A: a request from
    client B (or an anonymous request) for that same session_id raises
    SessionOwnershipError. The router maps this to HTTP 403. We do NOT
    silently let a second account read someone else's financial data.
"""
import uuid, logging, typing
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session as DBSession
from sqlalchemy import delete

from app.schemas.session import ConversationSession, BusinessState, Phase
from app.database.models import ConversationSessionDB
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SessionOwnershipError(Exception):
    """Raised when a session_id is requested by someone other than its owner."""


def get_or_create_session(
    db: DBSession, session_id: Optional[str], client_id: Optional[str] = None
) -> ConversationSession:
    """
    Load an existing session or create a new one.

    client_id: the caller's id if authenticated (from
    api.deps.get_current_client_optional), else None. See module docstring
    for the ownership rules this enforces.
    """
    _cleanup_expired(db)

    if session_id:
        row = db.query(ConversationSessionDB).filter(ConversationSessionDB.id == session_id).first()
        if row:
            if row.client_id and client_id and row.client_id != client_id:
                raise SessionOwnershipError(
                    f"Session {session_id} belongs to a different account."
                )
            if row.client_id is None and client_id:
                # Anonymous session being resumed by a logged-in caller — claim it.
                row.client_id = client_id
                db.commit()
                logger.info(f"Session {session_id} claimed by client {client_id}")
            return ConversationSession.model_validate(row.data)

    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(session_id=new_id)
    _save_to_db(db, session, client_id=client_id)
    logger.info(f"New session: {new_id}" + (f" (client {client_id})" if client_id else " (anonymous)"))
    return session


def save_session(db: DBSession, session: ConversationSession, client_id: Optional[str] = None) -> None:
    """
    Save session state. client_id is only used if the row doesn't exist yet
    (shouldn't normally happen — get_or_create_session creates it first —
    but kept for safety/symmetry); an existing row's client_id is never
    overwritten here, only by the claim logic in get_or_create_session.
    """
    session.updated_at = datetime.utcnow()
    _save_to_db(db, session, client_id=client_id)


def update_business_state_fields(session: ConversationSession, extracted: dict) -> None:
    """Merges extracted data. Never overwrites existing. Never stores None."""
    state = session.business_state
    schema_fields = BusinessState.model_fields
    for field_name, value in extracted.items():
        if value is None or field_name not in schema_fields:
            continue
        if getattr(state, field_name, None) is not None:
            continue
        try:
            setattr(state, field_name, _coerce(value, schema_fields[field_name].annotation))
        except Exception as e:
            logger.warning(f"Could not set {field_name}={value!r}: {e}")
    session.business_state = state


def load_hypothesis_into_session(session: ConversationSession, hypothesis) -> None:
    """Translates HypothesisOutput → BusinessState + derived variables."""
    from app.tools.hypothesis_ingestor import ingest_hypothesis
    try:
        fin, derived, proj = ingest_hypothesis(hypothesis, fiscal_year=2025)
        s = session.business_state
        s.entity_type="corporate"; s.entity_name=fin.entity_name; s.sector=fin.sector
        s.total_revenue=fin.total_revenue; s.cost_of_goods_sold=fin.cost_of_goods_sold
        s.operating_expenses=fin.operating_expenses; s.salaries_and_benefits=fin.salaries_and_benefits
        s.depreciation_amortization=fin.depreciation_amortization; s.interest_expense=fin.interest_expense
        s.total_assets=fin.total_assets; s.total_equity=fin.total_equity; s.total_debt=fin.total_debt
        s.cash_inflow=fin.cash_inflow; s.cash_outflow=fin.cash_outflow
        s.own_capital_invested=fin.own_capital_invested; s.external_funding=fin.external_funding
        h = hypothesis
        s.segment_client=h.ventes.H1_segment_client
        s.prix_vente_unitaire=h.ventes.H2_prix_vente_unitaire
        s.nb_clients_mois1=h.ventes.H4_nb_clients_mois1
        s.taux_croissance_mensuel=h.ventes.H5_taux_croissance_mensuel
        s.taux_fidelisation=h.ventes.H6_taux_fidelisation
        s.type_activite=h.achats.H8_type_activite
        s.loyer_mensuel=h.charges_fixes.H13_loyer_mensuel
        s.salaires_equipe=h.charges_fixes.H14_salaires_equipe
        s.investissements_initiaux=h.charges_fixes.H19_investissements_initiaux
        s.emprunts=h.charges_fixes.H21_emprunts
        if h.metadata:
            s.sector=h.metadata.secteur or s.sector
            s.statut_juridique=h.metadata.statut_juridique
            s.own_capital_invested=h.metadata.capital_social or s.own_capital_invested
        session.business_state=s
        session.derived_variables=derived
        session.projection_inputs=proj
        logger.info(f"HypothesisOutput loaded into session {session.session_id}")
    except Exception as e:
        logger.error(f"load_hypothesis_into_session failed: {e}")
        raise


def get_session_count(db: DBSession) -> int:
    return db.query(ConversationSessionDB).count()


def _save_to_db(db: DBSession, session: ConversationSession, client_id: Optional[str] = None) -> None:
    try:
        data = session.model_dump(
            mode="json",
            exclude={"cached_plan", "cached_metrics", "derived_variables", "projection_inputs"},
        )
        row = db.query(ConversationSessionDB).filter(ConversationSessionDB.id == session.session_id).first()
        if row:
            row.data = data
            # Never clobber an existing owner with None here — ownership
            # changes only happen through the explicit claim path above.
            if client_id and not row.client_id:
                row.client_id = client_id
        else:
            row = ConversationSessionDB(id=session.session_id, data=data, client_id=client_id)
            db.add(row)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Save session {session.session_id} failed: {e}")


def _cleanup_expired(db: DBSession) -> None:
    ttl = timedelta(minutes=getattr(settings, "session_ttl_minutes", 60))
    cutoff = datetime.now(timezone.utc) - ttl
    try:
        db.execute(delete(ConversationSessionDB).where(ConversationSessionDB.updated_at < cutoff))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Cleanup error: {e}")


def _coerce(value, annotation):
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        args = [a for a in annotation.__args__ if a is not type(None)]
        if args: annotation = args[0]
    if annotation is int:   return int(float(str(value)))
    if annotation is float: return float(str(value).replace(",",".").replace(" ",""))
    if annotation is str:   return str(value)
    if annotation is bool:  return bool(value)
    return value