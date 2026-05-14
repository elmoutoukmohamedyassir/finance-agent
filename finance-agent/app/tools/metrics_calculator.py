"""
tools/metrics_calculator.py — Pure Python SaaS financial metric calculations.

WHY NO LLM HERE: This is the core anti-hallucination measure.
All math is deterministic Python. The LLM only interprets results,
never recalculates them.

Formulas used:
  ARR = MRR × 12
  ARPU = MRR / customer_count
  Retention = 1 - churn_rate
  Customer Lifetime = 1 / churn_rate (in months, for monthly churn)
  LTV = ARPU × Customer Lifetime × gross_margin_factor
  LTV:CAC ratio = LTV / CAC
  CAC Payback = CAC / ARPU
  Profit = MRR - monthly_costs
  Profit Margin = Profit / MRR × 100
"""

import logging
from typing import Optional

from app.schemas.metrics import MetricsInput, MetricsOutput
from app.schemas.session import BusinessState

logger = logging.getLogger(__name__)


def calculate_metrics(inputs: MetricsInput) -> MetricsOutput:
    """
    Calculate all possible SaaS metrics from provided inputs.
    
    Returns MetricsOutput with None for metrics that cannot be calculated.
    Never raises — missing data results in None, not an error.
    """
    # Normalize: if both MRR and ARR provided, MRR takes precedence
    mrr = inputs.mrr
    arr = inputs.arr
    customer_count = inputs.customer_count
    arpu = inputs.arpu
    churn_rate_pct = inputs.churn_rate       # as percentage (e.g. 3.5 for 3.5%)
    cac = inputs.cac
    monthly_costs = inputs.monthly_costs
    marketing_budget = inputs.marketing_budget
    new_customers = inputs.new_customers_per_month
    gross_margin_pct = inputs.gross_margin or 80.0  # default 80% for SaaS

    # ── Derive missing base values ─────────────────────────────────────────
    # MRR ↔ ARR
    if mrr and not arr:
        arr = mrr * 12
    elif arr and not mrr:
        mrr = arr / 12

    # MRR ↔ ARPU ↔ customer_count (any two → third)
    if mrr and customer_count and customer_count > 0 and not arpu:
        arpu = mrr / customer_count
    elif mrr and arpu and arpu > 0 and not customer_count:
        customer_count = int(mrr / arpu)
    elif arpu and customer_count and not mrr:
        mrr = arpu * customer_count
        arr = mrr * 12

    # CAC from marketing budget + new customers
    if not cac and marketing_budget and new_customers and new_customers > 0:
        cac = marketing_budget / new_customers

    # ── Retention and churn ────────────────────────────────────────────────
    churn_rate = None
    retention_rate = None
    customer_lifetime_months = None

    if churn_rate_pct is not None:
        churn_rate = churn_rate_pct
        churn_decimal = churn_rate_pct / 100.0
        retention_rate = round((1 - churn_decimal) * 100, 2)

        if churn_decimal > 0:
            customer_lifetime_months = round(1 / churn_decimal, 1)
        else:
            customer_lifetime_months = None  # infinite lifetime, don't display

    # ── LTV ───────────────────────────────────────────────────────────────
    ltv = None
    if arpu and customer_lifetime_months:
        gross_margin_factor = gross_margin_pct / 100.0
        ltv = round(arpu * customer_lifetime_months * gross_margin_factor, 2)

    # ── LTV:CAC Ratio ──────────────────────────────────────────────────────
    ltv_cac_ratio = None
    if ltv and cac and cac > 0:
        ltv_cac_ratio = round(ltv / cac, 2)

    # ── CAC Payback Period ─────────────────────────────────────────────────
    cac_payback_months = None
    if cac and arpu and arpu > 0:
        cac_payback_months = round(cac / arpu, 1)

    # ── Profitability ──────────────────────────────────────────────────────
    monthly_profit = None
    profit_margin = None
    burn_rate = None

    if mrr and monthly_costs is not None:
        monthly_profit = round(mrr - monthly_costs, 2)
        if mrr > 0:
            profit_margin = round((monthly_profit / mrr) * 100, 1)
        if monthly_profit < 0:
            burn_rate = abs(monthly_profit)

    # ── Health Assessment ──────────────────────────────────────────────────
    warnings, health_score = _assess_health(
        ltv_cac_ratio=ltv_cac_ratio,
        churn_rate=churn_rate,
        cac_payback_months=cac_payback_months,
        profit_margin=profit_margin,
    )

    return MetricsOutput(
        mrr=round(mrr, 2) if mrr else None,
        arr=round(arr, 2) if arr else None,
        arpu=round(arpu, 2) if arpu else None,
        churn_rate=churn_rate,
        retention_rate=retention_rate,
        customer_lifetime_months=customer_lifetime_months,
        ltv=ltv,
        cac=round(cac, 2) if cac else None,
        ltv_cac_ratio=ltv_cac_ratio,
        cac_payback_months=cac_payback_months,
        monthly_profit=monthly_profit,
        profit_margin=profit_margin,
        burn_rate=burn_rate,
        health_score=health_score,
        warnings=warnings,
    )


def calculate_from_business_state(state: BusinessState) -> MetricsOutput:
    """
    Convenience function: calculate metrics directly from a session's BusinessState.
    Converts BusinessState → MetricsInput → MetricsOutput.
    """
    inputs = MetricsInput(
        mrr=state.mrr,
        arr=state.arr,
        customer_count=state.customer_count,
        arpu=state.arpu,
        churn_rate=state.churn_rate,
        cac=state.cac,
        monthly_costs=state.monthly_costs,
        marketing_budget=state.marketing_budget,
        gross_margin=state.gross_margin,
    )
    return calculate_metrics(inputs)


def _assess_health(
    ltv_cac_ratio: Optional[float],
    churn_rate: Optional[float],
    cac_payback_months: Optional[float],
    profit_margin: Optional[float],
) -> tuple[list[str], str]:
    """
    Produces warnings and an overall health score based on industry benchmarks.

    SaaS benchmarks used:
      LTV:CAC  ≥ 3x  → healthy (SaaS standard)
      Churn    ≤ 5%  → acceptable monthly churn for SMB SaaS
      Payback  ≤ 12 months → acceptable for SMB
      Profit   > 0   → profitable
    """
    warnings = []
    score_flags = []  # "good", "warning", "critical"

    # LTV:CAC
    if ltv_cac_ratio is not None:
        if ltv_cac_ratio < 1:
            warnings.append(
                f"⚠️ LTV:CAC ratio is {ltv_cac_ratio}x — you're spending more to acquire "
                f"customers than you earn from them. Business model needs rethinking."
            )
            score_flags.append("critical")
        elif ltv_cac_ratio < 3:
            warnings.append(
                f"⚠️ LTV:CAC ratio is {ltv_cac_ratio}x — below the healthy 3x benchmark. "
                f"Consider reducing CAC or improving retention."
            )
            score_flags.append("warning")
        else:
            score_flags.append("good")

    # Churn
    if churn_rate is not None:
        if churn_rate > 10:
            warnings.append(
                f"🔴 Monthly churn of {churn_rate}% is very high. "
                f"Focus urgently on retention — this will kill growth."
            )
            score_flags.append("critical")
        elif churn_rate > 5:
            warnings.append(
                f"⚠️ Monthly churn of {churn_rate}% is above the 5% healthy benchmark. "
                f"Investigate why customers are leaving."
            )
            score_flags.append("warning")
        else:
            score_flags.append("good")

    # Payback period
    if cac_payback_months is not None:
        if cac_payback_months > 18:
            warnings.append(
                f"⚠️ CAC payback of {cac_payback_months} months is long. "
                f"You'll need significant capital to sustain growth."
            )
            score_flags.append("warning")
        elif cac_payback_months <= 12:
            score_flags.append("good")

    # Profitability
    if profit_margin is not None:
        if profit_margin < 0:
            warnings.append(
                f"🔴 Currently unprofitable at {profit_margin}% margin. "
                f"Ensure your runway covers the path to profitability."
            )
            score_flags.append("critical")
        elif profit_margin < 10:
            warnings.append(
                f"⚠️ Profit margin of {profit_margin}% is thin. "
                f"Look for ways to reduce costs or increase pricing."
            )
            score_flags.append("warning")
        else:
            score_flags.append("good")

    # Overall health
    if "critical" in score_flags:
        health_score = "Critical"
    elif "warning" in score_flags:
        health_score = "Warning"
    elif score_flags:
        health_score = "Healthy"
    else:
        health_score = "Insufficient data"

    return warnings, health_score
