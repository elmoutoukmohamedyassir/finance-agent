"""
services/session_service.py — SQLite-backed session persistence.

WHY SQLite over in-memory dict:
  - Sessions survive server restarts and hot-reloads
  - No Redis/Postgres needed — single file
  - Works on Windows and Linux identically

Schema: sessions(id TEXT PK, data TEXT, updated_at TEXT)
data = ConversationSession JSON (excludes non-serializable runtime objects)
"""
import uuid, json, sqlite3, logging, typing
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from app.schemas.session import ConversationSession, BusinessState, Phase
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()
DB_PATH = Path(getattr(settings, "db_path", "./data/sessions.db"))


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY, data TEXT NOT NULL, updated_at TEXT NOT NULL)""")
    conn.commit()
    return conn


def get_or_create_session(session_id: Optional[str]) -> ConversationSession:
    _cleanup_expired()
    if session_id:
        s = _load(session_id)
        if s:
            return s
    new_id = session_id or str(uuid.uuid4())
    session = ConversationSession(session_id=new_id)
    _save_to_db(session)
    logger.info(f"New session: {new_id}")
    return session


def save_session(session: ConversationSession) -> None:
    session.updated_at = datetime.utcnow()
    _save_to_db(session)


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


def get_session_count() -> int:
    with _get_conn() as c:
        return c.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]


def _load(sid: str) -> Optional[ConversationSession]:
    try:
        with _get_conn() as c:
            row = c.execute("SELECT data FROM sessions WHERE id=?", (sid,)).fetchone()
        if row:
            return ConversationSession.model_validate(json.loads(row[0]))
    except Exception as e:
        logger.warning(f"Load session {sid} failed: {e}")
    return None


def _save_to_db(session: ConversationSession) -> None:
    try:
        data = session.model_dump_json(
            exclude={"cached_plan","cached_metrics","derived_variables","projection_inputs"}
        )
        with _get_conn() as c:
            c.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?)",
                      (session.session_id, data, session.updated_at.isoformat()))
            c.commit()
    except Exception as e:
        logger.error(f"Save session {session.session_id} failed: {e}")


def _cleanup_expired() -> None:
    ttl = timedelta(minutes=getattr(settings, "session_ttl_minutes", 60))
    cutoff = (datetime.utcnow() - ttl).isoformat()
    try:
        with _get_conn() as c:
            c.execute("DELETE FROM sessions WHERE updated_at<?", (cutoff,))
            c.commit()
    except Exception as e:
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