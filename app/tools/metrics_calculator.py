"""
tools/metrics_calculator.py — Enterprise financial math. Pure Python, no LLM.

Supports both Corporate and Government/Public sector entities.
All monetary values in MAD millions.
The LLM only interprets these results — it never recalculates them.
"""

import logging
from app.schemas.metrics import MetricsInput, MetricsOutput
from app.schemas.financial_variables import FinancialData
from app.tools.kpi_engine import KPIEngine

logger = logging.getLogger(__name__)

# ── Corporate benchmarks ──────────────────────────────────────────────────────
EBITDA_MARGIN_HEALTHY = 15.0
EBITDA_MARGIN_WARNING = 8.0
NET_MARGIN_HEALTHY = 8.0
NET_MARGIN_WARNING = 3.0
DSCR_HEALTHY = 2.5
DSCR_WARNING = 1.5
CURRENT_RATIO_HEALTHY = 1.5
CURRENT_RATIO_WARNING = 1.0
DEBT_EQUITY_WARNING = 2.0
DEBT_EQUITY_CRITICAL = 3.0

# ── Government benchmarks ─────────────────────────────────────────────────────
BUDGET_EXECUTION_HEALTHY = 85.0
BUDGET_EXECUTION_WARNING = 70.0
FISCAL_PRESSURE_WARNING = 55.0   # % of receipts from taxes
SALARY_RATIO_GOV_WARNING = 45.0  # % of total expenditure
CAPEX_RATIO_MIN = 20.0           # CAPEX should be at least 20% of total spending


def calculate_metrics(inputs: MetricsInput) -> MetricsOutput:
    """
    Calculate all enterprise KPIs from provided inputs.
    Returns None for metrics that cannot be computed due to missing data.
    """
    # Build FinancialData from inputs
    data = FinancialData(
        entity_type=inputs.entity_type,
        entity_name=inputs.entity_name,
        sector=inputs.sector,
        fiscal_year=inputs.fiscal_year,
        total_revenue=inputs.total_revenue or 0.0,
        operating_revenue=inputs.operating_revenue or 0.0,
        revenue_year2=inputs.revenue_year2 or 0.0,
        tax_revenue=inputs.tax_revenue or 0.0,
        non_tax_revenue=inputs.non_tax_revenue or 0.0,
        grants_and_transfers=inputs.grants_and_transfers or 0.0,
        cost_of_goods_sold=inputs.cost_of_goods_sold or 0.0,
        operating_expenses=inputs.operating_expenses or 0.0,
        salaries_and_benefits=inputs.salaries_and_benefits or 0.0,
        depreciation_amortization=inputs.depreciation_amortization or 0.0,
        interest_expense=inputs.interest_expense or 0.0,
        tax_expense=inputs.tax_expense or 0.0,
        total_expenditure=inputs.total_expenditure or 0.0,
        capital_expenditure=inputs.capital_expenditure or 0.0,
        recurrent_expenditure=inputs.recurrent_expenditure or 0.0,
        debt_service=inputs.debt_service or 0.0,
        subsidies_paid=inputs.subsidies_paid or 0.0,
        total_assets=inputs.total_assets or 0.0,
        current_assets=inputs.current_assets or 0.0,
        current_liabilities=inputs.current_liabilities or 0.0,
        total_equity=inputs.total_equity or 0.0,
        total_debt=inputs.total_debt or 0.0,
        cash_and_equivalents=inputs.cash_and_equivalents or 0.0,
        cash_inflow=inputs.cash_inflow or 0.0,
        cash_outflow=inputs.cash_outflow or 0.0,
        operating_cash_flow=inputs.operating_cash_flow or 0.0,
        investing_cash_flow=inputs.investing_cash_flow or 0.0,
        own_capital_invested=inputs.own_capital_invested or 0.0,
        external_funding=inputs.external_funding or 0.0,
        investment_budget=inputs.investment_budget or 0.0,
        investment_executed=inputs.investment_executed or 0.0,
        months=inputs.months,
    )

    engine = KPIEngine(data)
    kpis = engine.get_all_kpis()
    statuses = engine.get_all_statuses()

    warnings, health_score = _assess_health(engine, inputs.entity_type)

    return MetricsOutput(
        entity_type=inputs.entity_type,
        total_revenue_mad_m=_r(kpis.get("total_revenue_mad_m")),
        revenue_growth_rate_pct=_r(kpis.get("revenue_growth_rate_pct")),
        gross_profit_mad_m=_r(kpis.get("gross_profit_mad_m")),
        gross_profit_margin_pct=_r(kpis.get("gross_profit_margin_pct")),
        ebitda_mad_m=_r(kpis.get("ebitda_mad_m")),
        ebitda_margin_pct=_r(kpis.get("ebitda_margin_pct")),
        operating_profit_mad_m=_r(kpis.get("operating_profit_mad_m")),
        operating_margin_pct=_r(kpis.get("operating_margin_pct")),
        net_profit_mad_m=_r(kpis.get("net_profit_mad_m")),
        net_profit_margin_pct=_r(kpis.get("net_profit_margin_pct")),
        debt_to_equity=_r(kpis.get("debt_to_equity")),
        debt_to_revenue=_r(kpis.get("debt_to_revenue")),
        debt_service_coverage=_r(kpis.get("debt_service_coverage")),
        equity_ratio_pct=_r(kpis.get("equity_ratio_pct")),
        current_ratio=_r(kpis.get("current_ratio")),
        cash_flow_net_mad_m=_r(kpis.get("cash_flow_net_mad_m")),
        free_cash_flow_mad_m=_r(kpis.get("free_cash_flow_mad_m")),
        cash_coverage_months=_r(kpis.get("cash_coverage_months")),
        total_investment_mad_m=_r(kpis.get("total_investment_mad_m")),
        roi_pct=_r(kpis.get("roi_pct")),
        return_on_assets_pct=_r(kpis.get("return_on_assets_pct")),
        return_on_equity_pct=_r(kpis.get("return_on_equity_pct")),
        salary_ratio_pct=_r(kpis.get("salary_ratio_pct")),
        primary_balance_mad_m=_r(kpis.get("primary_balance_mad_m")),
        overall_balance_mad_m=_r(kpis.get("overall_balance_mad_m")),
        fiscal_pressure_pct=_r(kpis.get("fiscal_pressure_pct")),
        capex_ratio_pct=_r(kpis.get("capex_ratio_pct")),
        subsidies_ratio_pct=_r(kpis.get("subsidies_ratio_pct")),
        budget_execution_rate_pct=_r(kpis.get("budget_execution_rate_pct")),
        health_score=health_score,
        warnings=warnings,
        statuses=statuses,
    )


def _r(v, decimals: int = 2):
    """Round if not None."""
    return round(v, decimals) if v is not None else None


def _assess_health(engine: KPIEngine, entity_type: str) -> tuple[list[str], str]:
    """
    Benchmark KPIs against enterprise standards.
    Returns (warnings_list, health_score_string).
    """
    warnings = []
    flags = []
    is_gov = entity_type.lower() == "government"

    # ── EBITDA margin ────────────────────────────────────────────────────
    ebitda_m = engine.ebitda_margin()
    if ebitda_m is not None and not is_gov:
        if ebitda_m < 0:
            warnings.append(
                f"EBITDA négatif ({ebitda_m:.1f}%) — l'entreprise brûle plus qu'elle ne génère "
                "avant intérêts et amortissements. Revue urgente des charges d'exploitation."
            )
            flags.append("critical")
        elif ebitda_m < EBITDA_MARGIN_WARNING:
            warnings.append(
                f"Marge EBITDA faible ({ebitda_m:.1f}% < seuil {EBITDA_MARGIN_WARNING}%). "
                "Optimisation des coûts opérationnels recommandée."
            )
            flags.append("warning")
        else:
            flags.append("good")

    # ── Net margin ───────────────────────────────────────────────────────
    net_m = engine.net_profit_margin()
    if net_m is not None:
        if net_m < 0:
            warnings.append(
                f"Résultat net négatif ({net_m:.1f}%) — situation déficitaire. "
                "Analyser la structure de financement et réduire les charges financières."
            )
            flags.append("critical")
        elif net_m < NET_MARGIN_WARNING:
            warnings.append(
                f"Marge nette insuffisante ({net_m:.1f}% < {NET_MARGIN_WARNING}%). "
                "Risque de fragilité en cas de choc externe."
            )
            flags.append("warning")
        else:
            flags.append("good")

    # ── Debt service coverage ─────────────────────────────────────────────
    dscr = engine.debt_service_coverage()
    if dscr is not None:
        if dscr < DSCR_WARNING:
            warnings.append(
                f"Couverture du service de la dette critique (DSCR={dscr:.2f}x < {DSCR_WARNING}x). "
                "L'entité risque de ne pas honorer ses échéances de dette."
            )
            flags.append("critical")
        elif dscr < DSCR_HEALTHY:
            warnings.append(
                f"Couverture du service de la dette tendue (DSCR={dscr:.2f}x). "
                f"Objectif recommandé : {DSCR_HEALTHY}x minimum."
            )
            flags.append("warning")
        else:
            flags.append("good")

    # ── Liquidity ────────────────────────────────────────────────────────
    cr = engine.current_ratio()
    if cr is not None:
        if cr < CURRENT_RATIO_WARNING:
            warnings.append(
                f"Ratio de liquidité critique ({cr:.2f}x < 1.0x). "
                "L'actif circulant ne couvre pas le passif à court terme — risque d'insolvabilité immédiate."
            )
            flags.append("critical")
        elif cr < CURRENT_RATIO_HEALTHY:
            warnings.append(
                f"Liquidité tendue (ratio={cr:.2f}x). Objectif recommandé ≥ {CURRENT_RATIO_HEALTHY}x."
            )
            flags.append("warning")
        else:
            flags.append("good")

    # ── Leverage ─────────────────────────────────────────────────────────
    dte = engine.debt_to_equity()
    if dte is not None and not is_gov:
        if dte > DEBT_EQUITY_CRITICAL:
            warnings.append(
                f"Endettement très élevé (D/E={dte:.2f}x > {DEBT_EQUITY_CRITICAL}x). "
                "Structure financière fragilisée — renforcement des capitaux propres nécessaire."
            )
            flags.append("critical")
        elif dte > DEBT_EQUITY_WARNING:
            warnings.append(
                f"Ratio dette/fonds propres élevé ({dte:.2f}x). "
                "Surveiller l'évolution de l'endettement et les conditions de refinancement."
            )
            flags.append("warning")
        else:
            flags.append("good")

    # ── Government-specific ───────────────────────────────────────────────
    if is_gov:
        exec_rate = engine.budget_execution_rate()
        if exec_rate is not None:
            if exec_rate < BUDGET_EXECUTION_WARNING:
                warnings.append(
                    f"Taux d'exécution budgétaire très faible ({exec_rate:.1f}% < {BUDGET_EXECUTION_WARNING}%). "
                    "Risque de sous-investissement et d'impact sur les services publics."
                )
                flags.append("warning")
            elif exec_rate >= BUDGET_EXECUTION_HEALTHY:
                flags.append("good")

        sal_ratio = engine.salary_ratio()
        if sal_ratio is not None and sal_ratio > SALARY_RATIO_GOV_WARNING:
            warnings.append(
                f"Masse salariale élevée ({sal_ratio:.1f}% des recettes > {SALARY_RATIO_GOV_WARNING}%). "
                "Pression sur les marges de manœuvre budgétaires pour l'investissement."
            )
            flags.append("warning")

        overall = engine.overall_balance()
        if overall < 0:
            warnings.append(
                f"Solde budgétaire global déficitaire ({overall:.2f} MMAD). "
                "Nécessite un financement par emprunt ou tirage sur réserves."
            )
            flags.append("warning")

    # ── Score ─────────────────────────────────────────────────────────────
    if "critical" in flags:
        score = "Critique ⚠️"
    elif "warning" in flags:
        score = "Vigilance requise ⚡"
    elif flags:
        score = "Sain ✅"
    else:
        score = "Données insuffisantes"

    return warnings, score


def calculate_from_business_state(state) -> MetricsOutput:
    """Convenience: feed a BusinessState directly into the calculator."""
    return calculate_metrics(MetricsInput(
        entity_type=getattr(state, "entity_type", "corporate"),
        entity_name=getattr(state, "entity_name", None),
        sector=getattr(state, "sector", None),
        total_revenue=getattr(state, "total_revenue", None),
        revenue_year2=getattr(state, "revenue_year2", None),
        tax_revenue=getattr(state, "tax_revenue", None),
        non_tax_revenue=getattr(state, "non_tax_revenue", None),
        grants_and_transfers=getattr(state, "grants_and_transfers", None),
        cost_of_goods_sold=getattr(state, "cost_of_goods_sold", None),
        operating_expenses=getattr(state, "operating_expenses", None),
        salaries_and_benefits=getattr(state, "salaries_and_benefits", None),
        depreciation_amortization=getattr(state, "depreciation_amortization", None),
        interest_expense=getattr(state, "interest_expense", None),
        tax_expense=getattr(state, "tax_expense", None),
        total_expenditure=getattr(state, "total_expenditure", None),
        capital_expenditure=getattr(state, "capital_expenditure", None),
        recurrent_expenditure=getattr(state, "recurrent_expenditure", None),
        debt_service=getattr(state, "debt_service", None),
        subsidies_paid=getattr(state, "subsidies_paid", None),
        total_assets=getattr(state, "total_assets", None),
        current_assets=getattr(state, "current_assets", None),
        current_liabilities=getattr(state, "current_liabilities", None),
        total_equity=getattr(state, "total_equity", None),
        total_debt=getattr(state, "total_debt", None),
        cash_and_equivalents=getattr(state, "cash_and_equivalents", None),
        cash_inflow=getattr(state, "cash_inflow", None),
        cash_outflow=getattr(state, "cash_outflow", None),
        operating_cash_flow=getattr(state, "operating_cash_flow", None),
        own_capital_invested=getattr(state, "own_capital_invested", None),
        external_funding=getattr(state, "external_funding", None),
        investment_budget=getattr(state, "investment_budget", None),
        investment_executed=getattr(state, "investment_executed", None),
        months=getattr(state, "months", 12),
    ))