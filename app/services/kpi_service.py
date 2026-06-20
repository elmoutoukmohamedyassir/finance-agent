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


# kpi_name -> (unit, short explanation). Used by save_kpis_from_computed()
# below to turn the engine's raw numbers into self-describing snapshot rows.
_KPI_DEFINITIONS = {
    "seuil_rentabilite_clients": ("clients/mois", "Nombre de clients mensuels nécessaires pour couvrir les charges fixes."),
    "seuil_rentabilite_ca":      ("MAD", "Chiffre d'affaires mensuel minimum pour atteindre l'équilibre."),
    "bfr":                       ("MAD", "Besoin en fonds de roulement — trésorerie à immobiliser pour financer le cycle d'exploitation."),
    "taux_marge_brute":          ("%", "Part du chiffre d'affaires conservée après coûts variables."),
    "marge_brute_unitaire":      ("MAD", "Marge brute générée par unité vendue."),
    "charges_fixes_mensuelles_totales": ("MAD", "Total des charges fixes mensuelles (loyer, salaires, etc.)."),
    "mois_point_mort":           ("mois", "Mois où la trésorerie cumulée devient positive."),
    "roi_annee1":                ("%", "Retour sur investissement sur la première année."),
    "roi_annee2":                ("%", "Retour sur investissement sur la deuxième année."),
    "dscr_annee1":               ("x", "Capacité de l'entreprise à couvrir le service de la dette (Année 1)."),
}


def save_kpis_from_computed(
    db: Session,
    client_id: str,
    session_id: str,
    computed,  # tools.plan_pipeline.ComputedPlan
) -> List[KPISnapshot]:
    """
    Bridge between the deterministic engine (DerivedVariables +
    BusinessPlan24M, see tools/hypothesis_ingestor.py and
    tools/plan_generator.py) and bulk_create_kpis() above.

    Pulls a curated set of named KPIs off computed.derived and
    computed.plan, skips any that the engine couldn't compute (None —
    e.g. dscr_annee1 with no debt), and stores the rest as KPISnapshot
    rows. calculation_type is always "automatic" since these come
    straight out of the engine, never user-entered.
    """
    derived, plan = computed.derived, computed.plan
    raw_values = {
        "seuil_rentabilite_clients": getattr(derived, "seuil_rentabilite_clients", None),
        "seuil_rentabilite_ca":      getattr(derived, "seuil_rentabilite_ca", None),
        "bfr":                       getattr(derived, "bfr", None),
        "taux_marge_brute":          getattr(derived, "taux_marge_brute", None),
        "marge_brute_unitaire":      getattr(derived, "marge_brute_unitaire", None),
        "charges_fixes_mensuelles_totales": getattr(derived, "charges_fixes_mensuelles_totales", None),
        "mois_point_mort":           plan.mois_point_mort,
        "roi_annee1":                plan.roi_annee1,
        "roi_annee2":                plan.roi_annee2,
        "dscr_annee1":               plan.dscr_annee1,
    }

    kpis = [
        {
            "kpi_name": name,
            "value": float(value),
            "unit": _KPI_DEFINITIONS[name][0],
            "explanation": _KPI_DEFINITIONS[name][1],
            "calculation_type": "automatic",
        }
        for name, value in raw_values.items()
        if value is not None
    ]
    return bulk_create_kpis(db, client_id, session_id, kpis)