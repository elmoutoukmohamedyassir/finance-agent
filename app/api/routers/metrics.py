"""
api/routers/metrics.py — Enterprise metrics REST endpoints.
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
    Calculate enterprise financial KPIs directly without going through chat.

    Supports both corporate and government entities.
    All monetary values in MAD millions.
    Returns all computable metrics — fields not provided are skipped.
    No LLM involved — pure deterministic math.
    """
    try:
        return calculate_metrics(inputs)
    except Exception as e:
        logger.error(f"Metrics error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Calcul échoué")


@router.get("/benchmarks")
def get_benchmarks():
    """
    Reference benchmarks for enterprise financial ratios.
    Covers both corporate and government/public sector norms.
    """
    return {
        "corporate": {
            "ebitda_margin": {
                "excellent": "> 25%",
                "bon": "15–25%",
                "correct": "8–15%",
                "faible": "< 8%",
                "critique": "< 0%",
            },
            "net_profit_margin": {
                "très bon": "> 15%",
                "bon": "8–15%",
                "correct": "3–8%",
                "faible": "< 3%",
                "perte": "< 0%",
            },
            "debt_to_equity": {
                "très faible endettement": "< 0.3x",
                "sain": "0.3–1.0x",
                "modéré": "1.0–2.0x",
                "élevé": "2.0–3.0x",
                "très risqué": "> 3.0x",
            },
            "debt_service_coverage_dscr": {
                "solide": "> 2.5x",
                "acceptable": "1.5–2.5x",
                "risqué": "1.0–1.5x",
                "critique": "< 1.0x",
            },
            "current_ratio": {
                "confortable": "> 2.0x",
                "correct": "1.5–2.0x",
                "tendu": "1.0–1.5x",
                "illiquidité": "< 1.0x",
            },
            "return_on_equity": {
                "excellent": "> 20%",
                "bon": "10–20%",
                "correct": "5–10%",
                "faible": "< 5%",
            },
        },
        "government": {
            "budget_execution_rate": {
                "excellent": "> 95%",
                "bon": "85–95%",
                "acceptable": "70–85%",
                "insuffisant": "50–70%",
                "très faible": "< 50%",
            },
            "salary_ratio_of_expenditure": {
                "faible": "< 30%",
                "normal": "30–45%",
                "élevé": "45–60%",
                "critique": "> 60%",
            },
            "capex_ratio_of_expenditure": {
                "bon": "> 30%",
                "acceptable": "20–30%",
                "faible": "< 20%",
            },
            "debt_service_coverage_dscr": {
                "solide": "> 2.5x",
                "acceptable": "1.5–2.5x",
                "risqué": "< 1.5x",
            },
            "debt_to_revenue": {
                "faible": "< 0.5x",
                "modéré": "0.5–1.0x",
                "élevé": "1.0–2.0x",
                "très élevé": "> 2.0x",
            },
        },
    }