"""
tools/metrics_calculator.py — Pure Python SaaS financial math.

NO LLM INVOLVED HERE. Every number is deterministic.
The LLM only receives and interprets these results — it never recalculates them.
This is the single most important anti-hallucination measure in the whole system.

Formulas:
  ARR = MRR × 12
  ARPU = MRR / customer_count
  Retention Rate = (1 - churn_rate/100) × 100
  Customer Lifetime = 1 / (churn_rate/100) months
  LTV = ARPU × Lifetime × (gross_margin/100)
  LTV:CAC = LTV / CAC
  CAC Payback = CAC / ARPU  (months)
  Monthly Profit = MRR - monthly_costs
"""

import logging
from typing import Optional

from app.schemas.metrics import MetricsInput, MetricsOutput
from app.schemas.session import BusinessState

logger = logging.getLogger(__name__)

# SaaS industry benchmarks used for health assessment
LTV_CAC_HEALTHY = 3.0
LTV_CAC_CRITICAL = 1.0
CHURN_HEALTHY = 2.0
CHURN_WARNING = 5.0
CHURN_CRITICAL = 10.0
PAYBACK_HEALTHY = 12
PAYBACK_WARNING = 18
DEFAULT_GROSS_MARGIN = 75.0  # typical SaaS gross margin


def calculate_metrics(inputs: MetricsInput) -> MetricsOutput:
    """
    Calculate all computable SaaS metrics from provided inputs.
    Returns None for metrics that cannot be computed.
    """
    # ── Normalize base values ─────────────────────────────────────────────
    mrr = inputs.mrr
    arr = inputs.arr
    customers = inputs.customer_count
    arpu = inputs.arpu
    churn_pct = inputs.churn_rate
    cac = inputs.cac
    costs = inputs.monthly_costs
    mkt_budget = inputs.marketing_budget
    new_custs = inputs.new_customers_per_month
    gm_pct = inputs.gross_margin or DEFAULT_GROSS_MARGIN

    # Derive MRR ↔ ARR
    if mrr and not arr:
        arr = round(mrr * 12, 2)
    elif arr and not mrr:
        mrr = round(arr / 12, 2)

    # Derive any of MRR / ARPU / customer_count from the other two
    if mrr and customers and customers > 0 and not arpu:
        arpu = round(mrr / customers, 2)
    elif mrr and arpu and arpu > 0 and not customers:
        customers = int(mrr / arpu)
    elif arpu and customers and not mrr:
        mrr = round(arpu * customers, 2)
        arr = round(mrr * 12, 2)

    # Derive CAC from marketing spend
    if not cac and mkt_budget and new_custs and new_custs > 0:
        cac = round(mkt_budget / new_custs, 2)

    # ── Churn & retention ─────────────────────────────────────────────────
    retention_rate = None
    lifetime_months = None
    if churn_pct is not None:
        churn_decimal = churn_pct / 100
        retention_rate = round((1 - churn_decimal) * 100, 2)
        if churn_decimal > 0:
            lifetime_months = round(1 / churn_decimal, 1)

    # ── LTV ───────────────────────────────────────────────────────────────
    ltv = None
    if arpu and lifetime_months:
        ltv = round(arpu * lifetime_months * (gm_pct / 100), 2)

    # ── Unit economics ────────────────────────────────────────────────────
    ltv_cac_ratio = None
    payback = None
    if ltv and cac and cac > 0:
        ltv_cac_ratio = round(ltv / cac, 2)
    if cac and arpu and arpu > 0:
        payback = round(cac / arpu, 1)

    # ── Profitability ─────────────────────────────────────────────────────
    monthly_profit = None
    profit_margin = None
    burn = None
    if mrr is not None and costs is not None:
        monthly_profit = round(mrr - costs, 2)
        profit_margin = round((monthly_profit / mrr * 100), 1) if mrr > 0 else None
        if monthly_profit < 0:
            burn = round(abs(monthly_profit), 2)

    # ── Health assessment ─────────────────────────────────────────────────
    warnings, health_score = _assess_health(ltv_cac_ratio, churn_pct, payback, profit_margin)

    return MetricsOutput(
        mrr=round(mrr, 2) if mrr else None,
        arr=round(arr, 2) if arr else None,
        arpu=round(arpu, 2) if arpu else None,
        churn_rate=churn_pct,
        retention_rate=retention_rate,
        customer_lifetime_months=lifetime_months,
        ltv=ltv,
        cac=round(cac, 2) if cac else None,
        ltv_cac_ratio=ltv_cac_ratio,
        cac_payback_months=payback,
        monthly_profit=monthly_profit,
        profit_margin=profit_margin,
        burn_rate=burn,
        health_score=health_score,
        warnings=warnings,
    )


def calculate_from_business_state(state: BusinessState) -> MetricsOutput:
    """Convenience: feed a BusinessState directly into the calculator."""
    return calculate_metrics(MetricsInput(
        mrr=state.mrr,
        arr=state.arr,
        customer_count=state.customer_count,
        arpu=state.arpu,
        churn_rate=state.churn_rate,
        cac=state.cac,
        monthly_costs=state.monthly_costs,
        marketing_budget=state.marketing_budget,
        gross_margin=state.gross_margin,
    ))


def _assess_health(
    ltv_cac: Optional[float],
    churn_pct: Optional[float],
    payback: Optional[float],
    profit_margin: Optional[float],
) -> tuple[list[str], str]:
    """
    Benchmarks metrics against SaaS industry standards.
    Returns (warnings_list, health_score_string).
    """
    warnings = []
    flags = []

    if ltv_cac is not None:
        if ltv_cac < LTV_CAC_CRITICAL:
            warnings.append(
                f"LTV:CAC of {ltv_cac}x is critical — you lose money on every customer. "
                "Rethink pricing or drastically reduce acquisition costs."
            )
            flags.append("critical")
        elif ltv_cac < LTV_CAC_HEALTHY:
            warnings.append(
                f"LTV:CAC of {ltv_cac}x is below the 3x benchmark. "
                "Healthy SaaS businesses target 3x or higher."
            )
            flags.append("warning")
        else:
            flags.append("good")

    if churn_pct is not None:
        if churn_pct > CHURN_CRITICAL:
            warnings.append(
                f"Monthly churn of {churn_pct}% is very high. "
                "At this rate customers are leaving faster than you can acquire them. Retention is your #1 priority."
            )
            flags.append("critical")
        elif churn_pct > CHURN_WARNING:
            warnings.append(
                f"Monthly churn of {churn_pct}% exceeds the 5% warning threshold. "
                "Investigate why customers leave and implement a retention program."
            )
            flags.append("warning")
        else:
            flags.append("good")

    if payback is not None:
        if payback > PAYBACK_WARNING:
            warnings.append(
                f"CAC payback of {payback} months is long — you need significant capital to scale. "
                "Aim for under 12 months."
            )
            flags.append("warning")
        elif payback <= PAYBACK_HEALTHY:
            flags.append("good")

    if profit_margin is not None:
        if profit_margin < 0:
            warnings.append(
                f"Unprofitable at {profit_margin}% margin. "
                "Make sure your runway covers the time needed to reach break-even."
            )
            flags.append("critical")
        elif profit_margin < 10:
            warnings.append(f"Thin profit margin of {profit_margin}%. Consider cost reduction or price increases.")
            flags.append("warning")
        else:
            flags.append("good")

    if "critical" in flags:
        score = "Critical ⚠️"
    elif "warning" in flags:
        score = "Needs Attention ⚡"
    elif flags:
        score = "Healthy ✅"
    else:
        score = "Insufficient data"

    return warnings, score
