"""
database/models.py — SQLAlchemy ORM models for persistent analytics.

Models:
  - Client: User/organization information
  - KPISnapshot: Calculated KPI values with explanations
  - BusinessPlan: Generated financial plans (P&L, cash flow, etc.)
"""
from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


def _uuid() -> str:
    """Generate a new UUID string (used as default for primary keys)."""
    return str(uuid.uuid4())


def _now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class Client(Base):
    """Represents a client/entrepreneur using the finance agent."""
    __tablename__ = "clients"

    # Native PostgreSQL UUID type — more efficient than VARCHAR(36)
    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    phone = Column(String(20), nullable=True)
    sector = Column(String(100), nullable=True)

    # Auth — nullable: see class docstring
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, default=True)

    # timezone=True stores timestamps as TIMESTAMPTZ in PostgreSQL (always UTC-aware)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    # Relationships
    kpis = relationship("KPISnapshot", back_populates="client", cascade="all, delete-orphan")
    plans = relationship("BusinessPlan", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, name={self.name}, email={self.email})>"


class KPISnapshot(Base):
    """Stores calculated KPI values with explanations and metadata."""
    __tablename__ = "kpi_snapshots"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    client_id = Column(UUID(as_uuid=False), ForeignKey("clients.id"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=False), nullable=False, index=True)

    kpi_name = Column(String(100), nullable=False, index=True)  # e.g., "chiffre_affaires", "seuil_rentabilite"
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)       # e.g., "MAD", "units", "%"
    explanation = Column(Text, nullable=True)      # Short 1-2 sentence explanation

    # Metadata for tracking
    calculation_type = Column(String(50), nullable=True)  # "automatic" or "user_requested"
    calculated_at = Column(DateTime(timezone=True), default=_now, index=True)
    created_at = Column(DateTime(timezone=True), default=_now)

    # Relationships
    client = relationship("Client", back_populates="kpis")

    def __repr__(self):
        return f"<KPISnapshot(kpi_name={self.kpi_name}, value={self.value}, unit={self.unit})>"


class BusinessPlan(Base):
    """Stores generated business financial plans."""
    __tablename__ = "business_plans"

    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    client_id = Column(UUID(as_uuid=False), ForeignKey("clients.id"), nullable=False, index=True)
    session_id = Column(UUID(as_uuid=False), nullable=False, index=True)

    # Plan components — JSONB is PostgreSQL-native: binary storage, indexable, faster queries
    executive_summary = Column(Text, nullable=True)
    financial_highlights = Column(JSONB, nullable=True)   # {"year1_revenue": "...", "breakeven_month": "...", etc.}
    annee1_data = Column(JSONB, nullable=True)             # Year 1 P&L, cash flow, balance sheet
    annee2_data = Column(JSONB, nullable=True)             # Year 2 P&L, cash flow, balance sheet
    plan_financement = Column(JSONB, nullable=True)        # Financing structure
    key_risks = Column(JSONB, nullable=True)               # List of identified risks
    action_plan_6months = Column(JSONB, nullable=True)     # ["Month 1: ...", "Month 2: ...", ...]

    # Metadata
    narrative = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now)

    # Relationships
    client = relationship("Client", back_populates="plans")

    def __repr__(self):
        return f"<BusinessPlan(id={self.id}, client_id={self.client_id}, created_at={self.created_at})>"

class ConversationSessionDB(Base):
    """
    Persisted chat session state — replaces the old SQLite (data/sessions.db)
    store. Mirrors the old schema's shape (id, JSON blob, updated_at) so
    services/session_service.py logic barely changes, plus:
 
      - client_id: nullable FK. Chat works anonymously (client_id stays
        NULL); api/deps.get_current_client_optional attaches the owner on
        any request carrying a valid JWT. Once set, the session is "claimed"
        and session_service must refuse to let a different client (or an
        anonymous request) keep using that session_id.
      - JSONB instead of TEXT for `data`: queryable/indexable in Postgres,
        and avoids a manual json.loads/dumps round trip in the service layer.
 
    Table name is `conversation_sessions` to avoid clashing with the Pydantic
    `ConversationSession` schema in app/schemas/session.py — that schema is
    still what's stored *inside* `data`; this table is just its container.
    """
    __tablename__ = "conversation_sessions"
 
    id = Column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    client_id = Column(UUID(as_uuid=False), ForeignKey("clients.id"), nullable=True, index=True)
 
    data = Column(JSONB, nullable=False)  # full ConversationSession.model_dump_json() payload
 
    created_at = Column(DateTime(timezone=True), default=_now, index=True)
    updated_at = Column(DateTime(timezone=True), default=_now, onupdate=_now, index=True)
 
    # Relationships
    client = relationship("Client", back_populates="conversation_sessions")
 
    def __repr__(self):
        return f"<ConversationSessionDB(id={self.id}, client_id={self.client_id}, updated_at={self.updated_at})>"