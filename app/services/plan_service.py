"""
services/plan_service.py — Store and retrieve generated business plans.
"""
import uuid
import logging
import dataclasses
from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.database.models import BusinessPlan

logger = logging.getLogger(__name__)


def create_business_plan(
    db: Session,
    client_id: str,
    session_id: str,
    executive_summary: Optional[str] = None,
    financial_highlights: Optional[Dict[str, Any]] = None,
    annee1_data: Optional[Dict[str, Any]] = None,
    annee2_data: Optional[Dict[str, Any]] = None,
    plan_financement: Optional[Dict[str, Any]] = None,
    key_risks: Optional[list] = None,
    action_plan_6months: Optional[list] = None,
    narrative: Optional[str] = None
) -> BusinessPlan:
    """Create and store a business plan."""
    plan = BusinessPlan(
        id=str(uuid.uuid4()),
        client_id=client_id,
        session_id=session_id,
        executive_summary=executive_summary,
        financial_highlights=financial_highlights,
        annee1_data=annee1_data,
        annee2_data=annee2_data,
        plan_financement=plan_financement,
        key_risks=key_risks,
        action_plan_6months=action_plan_6months,
        narrative=narrative,
        created_at=datetime.utcnow()
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    logger.info(f"Created business plan: {plan.id} for client {client_id}")
    return plan


def get_business_plan_by_id(db: Session, plan_id: str) -> Optional[BusinessPlan]:
    """Retrieve business plan by ID."""
    return db.query(BusinessPlan).filter(BusinessPlan.id == plan_id).first()


def get_latest_plan_by_session(db: Session, session_id: str) -> Optional[BusinessPlan]:
    """Get the most recent plan for a session."""
    return db.query(BusinessPlan).filter(
        BusinessPlan.session_id == session_id
    ).order_by(BusinessPlan.created_at.desc()).first()


def get_plans_by_client(db: Session, client_id: str) -> list:
    """Get all plans for a client."""
    return db.query(BusinessPlan).filter(
        BusinessPlan.client_id == client_id
    ).order_by(BusinessPlan.created_at.desc()).all()


def update_business_plan(
    db: Session,
    plan_id: str,
    **kwargs
) -> Optional[BusinessPlan]:
    """Update business plan fields."""
    plan = db.query(BusinessPlan).filter(BusinessPlan.id == plan_id).first()
    if not plan:
        return None
    
    for key, value in kwargs.items():
        if hasattr(plan, key) and value is not None:
            setattr(plan, key, value)
    
    plan.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(plan)
    logger.info(f"Updated business plan: {plan_id}")
    return plan


def save_plan_from_computed(
    db: Session,
    client_id: str,
    session_id: str,
    computed,  # tools.plan_pipeline.ComputedPlan
    narrative: Optional[str] = None,
) -> BusinessPlan:
    """
    Bridge between the deterministic engine (tools/plan_pipeline.compute_plan,
    tools/plan_generator.generate_24m_plan) and create_business_plan() above.

    computed.plan is a BusinessPlan24M dataclass: annee1/annee2 are already
    plain dicts, but plan_financement/bilan_annee1/bilan_annee2 are nested
    dataclasses — dataclasses.asdict() converts the whole tree to plain
    dicts in one call, which is what the JSONB columns need.

    narrative: the LLM's executive-summary text for this plan (phase3
    agent's "Génère une synthèse exécutive..." response), if available.
    key_risks / action_plan_6months are left None for now — the LLM
    narrative currently returns them as embedded prose, not a separate
    structured list, so there's nothing reliable to split out yet.
    """
    plan = computed.plan
    financial_highlights = {
        "year1_revenue": plan.annee1.get("ca_total"),
        "year1_net_result": plan.annee1.get("resultat_net"),
        "year2_revenue": plan.annee2.get("ca_total"),
        "year2_net_result": plan.annee2.get("resultat_net"),
        "breakeven_clients_per_month": plan.seuil_rentabilite_clients,
        "breakeven_month": plan.mois_point_mort,
        "roi_year1_pct": plan.roi_annee1,
        "roi_year2_pct": plan.roi_annee2,
        "dscr_year1": plan.dscr_annee1,
    }

    return create_business_plan(
        db=db,
        client_id=client_id,
        session_id=session_id,
        executive_summary=narrative,
        financial_highlights=financial_highlights,
        annee1_data=plan.annee1,
        annee2_data=plan.annee2,
        plan_financement=dataclasses.asdict(plan.plan_financement),
        narrative=narrative,
    )