from fastapi import APIRouter, HTTPException
from app.schemas.metrics import MetricsInput, MetricsOutput
from app.tools.metrics_calculator import calculate_metrics
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/metrics", tags=["Metrics"])


@router.post("/calculate", response_model=MetricsOutput)
def calculate(inputs: MetricsInput) -> MetricsOutput:
    """
    Calculate SaaS financial metrics directly without going through the chat flow.
    All fields are optional — calculates whatever is possible from what you provide.
    No LLM involved — pure deterministic math.
    """
    try:
        return calculate_metrics(inputs)
    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Calculation failed")


@router.get("/benchmarks")
def get_benchmarks():
    """Industry benchmark values for key SaaS metrics."""
    return {
        "ltv_cac_ratio": {"healthy": "≥ 3x", "warning": "1–3x", "critical": "< 1x"},
        "monthly_churn": {"healthy": "< 2%", "acceptable": "2–5%", "warning": "5–10%", "critical": "> 10%"},
        "cac_payback_months": {"healthy": "< 12", "warning": "12–18", "critical": "> 18"},
        "gross_margin": {"healthy": "> 70%", "acceptable": "50–70%", "warning": "< 50%"},
        "mrr_growth_monthly": {"strong": "> 15%", "healthy": "5–15%", "slow": "< 5%"},
    }
