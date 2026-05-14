"""
api/routers/scenario.py — Financial scenario simulation endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from app.tools.scenario_engine import build_standard_scenarios
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scenario", tags=["Scenarios"])


class ScenarioRequest(BaseModel):
    starting_customers: int = Field(..., gt=0)
    monthly_price: float = Field(..., gt=0)
    monthly_costs: float = Field(..., ge=0)
    starting_cash: float = Field(default=50000, ge=0)
    months: int = Field(default=12, ge=1, le=60)

    class Config:
        json_schema_extra = {
            "example": {
                "starting_customers": 50,
                "monthly_price": 99,
                "monthly_costs": 8000,
                "starting_cash": 50000,
                "months": 12
            }
        }


@router.post("/analyze")
def analyze_scenarios(request: ScenarioRequest):
    """
    Run pessimistic, realistic, and optimistic 12-month projections.

    Returns monthly snapshots for each scenario including:
    - Customer count
    - MRR
    - Monthly costs
    - Profit/loss
    - Cumulative cash position

    **All calculations are pure Python math — no LLM involved.**
    """
    try:
        scenarios = build_standard_scenarios(
            starting_customers=request.starting_customers,
            monthly_price=request.monthly_price,
            monthly_costs=request.monthly_costs,
            starting_cash=request.starting_cash,
            months=request.months,
        )
        return {"scenarios": scenarios}
    except Exception as e:
        logger.error(f"Scenario error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Scenario calculation failed")
