"""
api/routers/scenario.py — Enterprise financial scenario projection endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from app.tools.scenario_engine import build_standard_scenarios
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scenario", tags=["Scenarios"])


class ScenarioRequest(BaseModel):
    entity_type: str = Field(default="corporate", description="'corporate' or 'government'")
    starting_revenue: float = Field(..., gt=0, description="Revenue/recettes base year (MAD millions)")
    starting_costs: float = Field(..., ge=0, description="Total operating costs base year (MAD millions)")
    starting_cash: float = Field(default=10.0, ge=0, description="Opening cash position (MAD millions)")
    starting_debt: float = Field(default=0.0, ge=0, description="Total debt at start (MAD millions)")
    debt_service_annual: float = Field(default=0.0, ge=0, description="Annual debt service (MAD millions)")
    capex_annual: float = Field(default=0.0, ge=0, description="Annual CAPEX (MAD millions)")
    years: int = Field(default=3, ge=1, le=10, description="Projection horizon in years")

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_type": "corporate",
                "starting_revenue": 87.5,
                "starting_costs": 62.0,
                "starting_cash": 8.0,
                "starting_debt": 35.0,
                "debt_service_annual": 5.0,
                "capex_annual": 4.0,
                "years": 3,
            }
        }
    }


@router.post("/analyze")
def analyze(request: ScenarioRequest):
    """
    Run Pessimiste / Réaliste / Optimiste projections over N years.

    Each scenario shows yearly revenue, costs, EBITDA, cash position,
    remaining debt, and DSCR. Pure Python — no LLM involved.

    Growth rates are calibrated for:
    - Corporate: Moroccan large enterprise norms (3% / 8% / 15%)
    - Government: Budget evolution patterns from Bulletin mensuel (2% / 5% / 9%)
    """
    try:
        scenarios = build_standard_scenarios(
            starting_revenue=request.starting_revenue,
            starting_costs=request.starting_costs,
            starting_cash=request.starting_cash,
            starting_debt=request.starting_debt,
            debt_service_annual=request.debt_service_annual,
            capex_annual=request.capex_annual,
            years=request.years,
            entity_type=request.entity_type,
        )
        return {"entity_type": request.entity_type, "scenarios": scenarios}
    except Exception as e:
        logger.error(f"Scenario error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Calcul de scénario échoué")