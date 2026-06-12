"""
services/kpi_service.py — Store and retrieve KPI snapshots with explanations.
"""
import uuid
import logging
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session

from app.database.models import KPISnapshot

logger = logging.getLogger(__name__)


def create_kpi_snapshot(
    db: Session,
    client_id: str,
    session_id: str,
    kpi_name: str,
    value: float,
    unit: Optional[str] = None,
    explanation: Optional[str] = None,
    calculation_type: str = "automatic"
) -> KPISnapshot:
    """Create and store a KPI snapshot."""
    kpi = KPISnapshot(
        id=str(uuid.uuid4()),
        client_id=client_id,
        session_id=session_id,
        kpi_name=kpi_name,
        value=value,
        unit=unit,
        explanation=explanation,
        calculation_type=calculation_type,
        calculated_at=datetime.utcnow()
    )
    db.add(kpi)
    db.commit()
    db.refresh(kpi)
    logger.info(f"Created KPI: {kpi_name}={value} {unit}")
    return kpi


def get_kpis_by_client(db: Session, client_id: str) -> List[KPISnapshot]:
    """Get all KPIs for a client."""
    return db.query(KPISnapshot).filter(
        KPISnapshot.client_id == client_id
    ).order_by(KPISnapshot.calculated_at.desc()).all()


def get_kpis_by_session(db: Session, session_id: str) -> List[KPISnapshot]:
    """Get all KPIs for a specific session."""
    return db.query(KPISnapshot).filter(
        KPISnapshot.session_id == session_id
    ).order_by(KPISnapshot.calculated_at.desc()).all()


def get_latest_kpi(db: Session, session_id: str, kpi_name: str) -> Optional[KPISnapshot]:
    """Get the latest value of a specific KPI in a session."""
    return db.query(KPISnapshot).filter(
        KPISnapshot.session_id == session_id,
        KPISnapshot.kpi_name == kpi_name
    ).order_by(KPISnapshot.calculated_at.desc()).first()


def bulk_create_kpis(
    db: Session,
    client_id: str,
    session_id: str,
    kpis: List[dict]
) -> List[KPISnapshot]:
    """Create multiple KPI snapshots at once.
    
    Args:
        kpis: List of dicts with keys: kpi_name, value, unit, explanation, calculation_type
    """
    snapshots = []
    for kpi_data in kpis:
        snapshot = create_kpi_snapshot(
            db=db,
            client_id=client_id,
            session_id=session_id,
            kpi_name=kpi_data.get("kpi_name"),
            value=kpi_data.get("value"),
            unit=kpi_data.get("unit"),
            explanation=kpi_data.get("explanation"),
            calculation_type=kpi_data.get("calculation_type", "automatic")
        )
        snapshots.append(snapshot)
    return snapshots
