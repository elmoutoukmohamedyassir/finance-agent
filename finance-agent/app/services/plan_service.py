"""
services/plan_service.py — Store and retrieve generated business plans.
"""
import uuid
import logging
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
