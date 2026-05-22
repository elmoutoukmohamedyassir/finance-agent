"""
database/models.py — SQLAlchemy ORM models for persistent analytics.

Models:
  - Client: User/organization information
  - KPISnapshot: Calculated KPI values with explanations
  - BusinessPlan: Generated financial plans (P&L, cash flow, etc.)
"""
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Client(Base):
    """Represents a client/entrepreneur using the finance agent."""
    __tablename__ = "clients"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, unique=True, index=True)
    phone = Column(String(20), nullable=True)
    sector = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    kpis = relationship("KPISnapshot", back_populates="client", cascade="all, delete-orphan")
    plans = relationship("BusinessPlan", back_populates="client", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Client(id={self.id}, name={self.name}, email={self.email})>"


class KPISnapshot(Base):
    """Stores calculated KPI values with explanations and metadata."""
    __tablename__ = "kpi_snapshots"

    id = Column(String(36), primary_key=True)
    client_id = Column(String(36), ForeignKey("clients.id"), nullable=False, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    
    kpi_name = Column(String(100), nullable=False, index=True)  # e.g., "chiffre_affaires", "seuil_rentabilite"
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)  # e.g., "MAD", "units", "%"
    explanation = Column(Text, nullable=True)  # Short 1-2 sentence explanation
    
    # Metadata for tracking
    calculation_type = Column(String(50), nullable=True)  # "automatic" or "user_requested"
    calculated_at = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    client = relationship("Client", back_populates="kpis")

    def __repr__(self):
        return f"<KPISnapshot(kpi_name={self.kpi_name}, value={self.value}, unit={self.unit})>"


class BusinessPlan(Base):
    """Stores generated business financial plans."""
    __tablename__ = "business_plans"

    id = Column(String(36), primary_key=True)
    client_id = Column(String(36), ForeignKey("clients.id"), nullable=False, index=True)
    session_id = Column(String(36), nullable=False, index=True)
    
    # Plan components (stored as JSON for flexibility)
    executive_summary = Column(Text, nullable=True)  # Narrative description
    financial_highlights = Column(JSON, nullable=True)  # {"year1_revenue": "...", "breakeven_month": "...", etc.}
    annee1_data = Column(JSON, nullable=True)  # Year 1 P&L, cash flow, balance sheet
    annee2_data = Column(JSON, nullable=True)  # Year 2 P&L, cash flow, balance sheet
    plan_financement = Column(JSON, nullable=True)  # Financing structure
    key_risks = Column(JSON, nullable=True)  # List of identified risks
    action_plan_6months = Column(JSON, nullable=True)  # ["Month 1: ...", "Month 2: ...", ...]
    
    # Metadata
    narrative = Column(Text, nullable=True)  # Full narrative interpretation
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    client = relationship("Client", back_populates="plans")

    def __repr__(self):
        return f"<BusinessPlan(id={self.id}, client_id={self.client_id}, created_at={self.created_at})>"
