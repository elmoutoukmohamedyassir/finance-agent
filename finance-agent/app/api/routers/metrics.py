"""
api/routers/metrics.py — Direct SaaS metric calculation endpoint.

Use this when you want to calculate metrics directly without going
through the conversational flow. Useful for:
  - Dashboards that calculate metrics on the fly
  - Testing the calculator with known values
  - Integrations that already have structured data
"""

from fastapi import APIRouter, HTTPException
from app.schemas.metrics import MetricsInput, MetricsOutput
from app.tools.metrics_calculator import calculate_metrics
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.post("/calculate", response_model=MetricsOutput)
def calculate(inputs: MetricsInput) -> MetricsOutput:
    """
    Calculate SaaS financial metrics from raw inputs.

    All fields are optional — the calculator will compute whatever
    it can from the data provided. Missing inputs result in `null`
    values in the response rather than errors.

    **Anti-hallucination guarantee**: This endpoint uses pure Python math.
    No LLM is involved. Results are deterministic and accurate.
    """
    try:
        return calculate_metrics(inputs)
    except Exception as e:
        logger.error(f"Metrics calculation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Calculation failed")


@router.get("/benchmark")
def get_benchmarks():
    """
    Returns industry benchmark values for key SaaS metrics.
    Useful as a reference for interpreting your own metrics.
    """
    return {
        "ltv_cac_ratio": {
            "healthy": ">= 3x",
            "warning": "1x - 3x",
            "critical": "< 1x",
            "description": "How much revenue you earn per $ spent acquiring a customer"
        },
        "monthly_churn_rate": {
            "healthy": "< 2%",
            "acceptable": "2% - 5%",
            "warning": "5% - 10%",
            "critical": "> 10%",
            "description": "Percentage of customers who cancel each month"
        },
        "cac_payback_months": {
            "healthy": "< 12 months",
            "warning": "12 - 18 months",
            "critical": "> 18 months",
            "description": "How many months until you recover your acquisition cost"
        },
        "gross_margin": {
            "healthy": "> 70%",
            "acceptable": "50% - 70%",
            "warning": "< 50%",
            "description": "Typical SaaS gross margins are 70-80%+"
        },
        "mrr_growth_rate": {
            "strong": "> 15%/month",
            "healthy": "5% - 15%/month",
            "slow": "< 5%/month",
            "description": "Monthly MRR growth rate"
        }
    }
