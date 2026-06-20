"""
schemas/client.py — Response models for an authenticated client's own data
(business plans, KPI history). Separate from schemas/auth.py because these
aren't auth concerns — they're what auth unlocks access to.
"""
from datetime import datetime
from typing import Optional, Any, Dict
from pydantic import BaseModel


class BusinessPlanSummary(BaseModel):
    """
    Lightweight summary for listing a client's plans — deliberately omits
    the heavy JSONB blobs (annee1_data, annee2_data, plan_financement) that
    a list view doesn't need. Fetch BusinessPlan by id directly if the
    full plan is needed later.
    """
    id: str
    session_id: str
    executive_summary: Optional[str] = None
    financial_highlights: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class KPISnapshotOut(BaseModel):
    id: str
    session_id: str
    kpi_name: str
    value: float
    unit: Optional[str] = None
    explanation: Optional[str] = None
    calculation_type: Optional[str] = None
    calculated_at: datetime

    model_config = {"from_attributes": True}