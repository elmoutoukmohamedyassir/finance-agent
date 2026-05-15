"""
kpi_engine.py — Enterprise KPI calculations for Corporate and Government entities.

All monetary values in MAD millions.
Statuses use French labels to match target market (Morocco).
No SaaS metrics (no LTV, CAC, churn, MRR, ARPU).
"""

from __future__ import annotations
from app.schemas.financial_variables import FinancialData


class KPIEngine:
    def __init__(self, data: FinancialData):
        self.data = data
        self._is_gov = data.entity_type.lower() == "government"

    # ── Revenue & Growth ──────────────────────────────────────────────────

    def total_receipts(self) -> float:
        """Total revenue/receipts for the period."""
        return self.data.total_revenue or (
            self.data.tax_revenue
            + self.data.non_tax_revenue
            + self.data.grants_and_transfers
        )

    def revenue_growth_rate(self) -> float | None:
        """YoY revenue growth rate (%)."""
        if not self.data.revenue_year2 or self.data.revenue_year2 == 0:
            return None
        return ((self.data.total_revenue - self.data.revenue_year2)
                / self.data.revenue_year2) * 100

    # ── Profitability ─────────────────────────────────────────────────────

    def gross_profit(self) -> float:
        return self.data.total_revenue - self.data.cost_of_goods_sold

    def gross_profit_margin(self) -> float | None:
        if self.data.total_revenue == 0:
            return None
        return (self.gross_profit() / self.data.total_revenue) * 100

    def ebitda(self) -> float:
        """EBITDA = Revenue - COGS - OpEx + D&A."""
        return (self.data.total_revenue
                - self.data.cost_of_goods_sold
                - self.data.operating_expenses
                + self.data.depreciation_amortization)

    def ebitda_margin(self) -> float | None:
        if self.data.total_revenue == 0:
            return None
        return (self.ebitda() / self.data.total_revenue) * 100

    def operating_profit(self) -> float:
        """EBIT = EBITDA - D&A."""
        return self.ebitda() - self.data.depreciation_amortization

    def operating_margin(self) -> float | None:
        if self.data.total_revenue == 0:
            return None
        return (self.operating_profit() / self.data.total_revenue) * 100

    def net_profit(self) -> float:
        """Net profit after interest and tax."""
        return (self.operating_profit()
                - self.data.interest_expense
                - self.data.tax_expense)

    def net_profit_margin(self) -> float | None:
        if self.data.total_revenue == 0:
            return None
        return (self.net_profit() / self.data.total_revenue) * 100

    # ── Government-specific: budget execution ────────────────────────────

    def budget_execution_rate(self) -> float | None:
        """Investment execution rate: réalisé / budgété (%)."""
        if not self.data.investment_budget or self.data.investment_budget == 0:
            return None
        return (self.data.investment_executed / self.data.investment_budget) * 100

    def primary_balance(self) -> float:
        """Solde budgétaire primaire (hors service de la dette)."""
        receipts = self.total_receipts()
        spending = self.data.total_expenditure or (
            self.data.recurrent_expenditure + self.data.capital_expenditure
        )
        return receipts - spending

    def overall_balance(self) -> float:
        """Solde budgétaire global (incluant service de la dette)."""
        return self.primary_balance() - self.data.debt_service

    def fiscal_pressure(self) -> float | None:
        """Pression fiscale: recettes fiscales / PIB proxy (recettes totales)."""
        total = self.total_receipts()
        if total == 0:
            return None
        return (self.data.tax_revenue / total) * 100

    def capex_ratio(self) -> float | None:
        """Part des dépenses d'investissement dans les dépenses totales (%)."""
        total_exp = self.data.total_expenditure
        if not total_exp or total_exp == 0:
            return None
        return (self.data.capital_expenditure / total_exp) * 100

    def subsidies_ratio(self) -> float | None:
        """Part des subventions dans les dépenses totales (%)."""
        total_exp = self.data.total_expenditure
        if not total_exp or total_exp == 0:
            return None
        return (self.data.subsidies_paid / total_exp) * 100

    # ── Solvency & Leverage ───────────────────────────────────────────────

    def debt_to_equity(self) -> float | None:
        if not self.data.total_equity or self.data.total_equity == 0:
            return None
        return self.data.total_debt / self.data.total_equity

    def debt_to_revenue(self) -> float | None:
        """Dette totale / recettes totales — pertinent pour les entités publiques."""
        revenue = self.total_receipts()
        if revenue == 0:
            return None
        return self.data.total_debt / revenue

    def debt_service_coverage(self) -> float | None:
        """DSCR = EBITDA / service de la dette."""
        if not self.data.debt_service or self.data.debt_service == 0:
            return None
        return self.ebitda() / self.data.debt_service

    def equity_ratio(self) -> float | None:
        """Ratio d'autonomie financière = capitaux propres / total actif."""
        if not self.data.total_assets or self.data.total_assets == 0:
            return None
        return (self.data.total_equity / self.data.total_assets) * 100

    # ── Liquidity ────────────────────────────────────────────────────────

    def current_ratio(self) -> float | None:
        """Ratio de liquidité générale = actif circulant / passif circulant."""
        if not self.data.current_liabilities or self.data.current_liabilities == 0:
            return None
        return self.data.current_assets / self.data.current_liabilities

    def cash_flow_net(self) -> float:
        """Flux de trésorerie net = encaissements - décaissements."""
        return self.data.cash_inflow - self.data.cash_outflow

    def free_cash_flow(self) -> float:
        """FCF = cash opérationnel - CAPEX."""
        return self.data.operating_cash_flow - self.data.capital_expenditure

    def cash_coverage_months(self) -> float | None:
        """Combien de mois les réserves couvrent les dépenses courantes."""
        monthly_out = self.data.cash_outflow / self.data.months if self.data.months > 0 else 0
        if monthly_out == 0:
            return None
        return self.data.cash_and_equivalents / monthly_out

    # ── Investment & Financing ────────────────────────────────────────────

    def total_investment(self) -> float:
        return self.data.own_capital_invested + self.data.external_funding

    def roi(self) -> float | None:
        """ROI = bénéfice net / investissement total."""
        inv = self.total_investment()
        if inv == 0:
            return None
        return (self.net_profit() / inv) * 100

    def return_on_assets(self) -> float | None:
        """ROA = résultat net / total actif."""
        if not self.data.total_assets or self.data.total_assets == 0:
            return None
        return (self.net_profit() / self.data.total_assets) * 100

    def return_on_equity(self) -> float | None:
        """ROE = résultat net / capitaux propres."""
        if not self.data.total_equity or self.data.total_equity == 0:
            return None
        return (self.net_profit() / self.data.total_equity) * 100

    # ── Salaries ─────────────────────────────────────────────────────────

    def salary_ratio(self) -> float | None:
        """Masse salariale / recettes totales (%)."""
        revenue = self.total_receipts()
        if revenue == 0:
            return None
        return (self.data.salaries_and_benefits / revenue) * 100

    # ─────────────────────────────────────────────────────────────────────
    # STATUS INTERPRETATIONS (labels in French)
    # ─────────────────────────────────────────────────────────────────────

    def _get_status(self, value, thresholds):
        if value is None:
            return "N/D"
        for limit, label in thresholds:
            if value < limit:
                return label
        return thresholds[-1][1]

    def gross_profit_margin_status(self):
        return self._get_status(self.gross_profit_margin(), [
            (10, "faible"), (25, "correct"), (40, "bon"), (float('inf'), "très bon")
        ])

    def ebitda_margin_status(self):
        return self._get_status(self.ebitda_margin(), [
            (0, "déficitaire"), (8, "faible"), (15, "correct"),
            (25, "bon"), (float('inf'), "excellent")
        ])

    def net_profit_margin_status(self):
        return self._get_status(self.net_profit_margin(), [
            (0, "perte"), (3, "faible"), (8, "correct"),
            (15, "bon"), (float('inf'), "très bon")
        ])

    def debt_to_equity_status(self):
        return self._get_status(self.debt_to_equity(), [
            (0.3, "très faible endettement"), (1.0, "sain"),
            (2.0, "modéré"), (3.0, "élevé"), (float('inf'), "très risqué")
        ])

    def debt_service_coverage_status(self):
        return self._get_status(self.debt_service_coverage(), [
            (1.0, "critique"), (1.5, "risqué"),
            (2.5, "acceptable"), (float('inf'), "solide")
        ])

    def current_ratio_status(self):
        return self._get_status(self.current_ratio(), [
            (1.0, "illiquidité"), (1.5, "tendu"),
            (2.0, "correct"), (float('inf'), "confortable")
        ])

    def roi_status(self):
        return self._get_status(self.roi(), [
            (0, "perte"), (5, "faible"), (10, "correct"),
            (20, "bon"), (float('inf'), "excellent")
        ])

    def cash_flow_status(self):
        return self._get_status(self.cash_flow_net(), [
            (0, "négatif"), (0.5, "faible"), (5, "stable"), (float('inf'), "très bon")
        ])

    def budget_execution_status(self):
        return self._get_status(self.budget_execution_rate(), [
            (50, "très faible"), (70, "insuffisant"),
            (85, "acceptable"), (95, "bon"), (float('inf'), "excellent")
        ])

    def salary_ratio_status(self):
        if self._is_gov:
            return self._get_status(self.salary_ratio(), [
                (30, "faible"), (45, "normal"), (60, "élevé"), (float('inf'), "critique")
            ])
        return self._get_status(self.salary_ratio(), [
            (20, "faible"), (35, "normal"), (50, "élevé"), (float('inf'), "critique")
        ])

    # ─────────────────────────────────────────────────────────────────────
    # AGGREGATE OUTPUTS
    # ─────────────────────────────────────────────────────────────────────

    def get_all_kpis(self) -> dict:
        base = {
            "total_revenue_mad_m": self.total_receipts(),
            "revenue_growth_rate_pct": self.revenue_growth_rate(),
            "gross_profit_mad_m": self.gross_profit(),
            "gross_profit_margin_pct": self.gross_profit_margin(),
            "ebitda_mad_m": self.ebitda(),
            "ebitda_margin_pct": self.ebitda_margin(),
            "operating_profit_mad_m": self.operating_profit(),
            "operating_margin_pct": self.operating_margin(),
            "net_profit_mad_m": self.net_profit(),
            "net_profit_margin_pct": self.net_profit_margin(),
            "debt_to_equity": self.debt_to_equity(),
            "debt_to_revenue": self.debt_to_revenue(),
            "debt_service_coverage": self.debt_service_coverage(),
            "equity_ratio_pct": self.equity_ratio(),
            "current_ratio": self.current_ratio(),
            "cash_flow_net_mad_m": self.cash_flow_net(),
            "free_cash_flow_mad_m": self.free_cash_flow(),
            "cash_coverage_months": self.cash_coverage_months(),
            "total_investment_mad_m": self.total_investment(),
            "roi_pct": self.roi(),
            "return_on_assets_pct": self.return_on_assets(),
            "return_on_equity_pct": self.return_on_equity(),
            "salary_ratio_pct": self.salary_ratio(),
        }
        if self._is_gov:
            base.update({
                "primary_balance_mad_m": self.primary_balance(),
                "overall_balance_mad_m": self.overall_balance(),
                "fiscal_pressure_pct": self.fiscal_pressure(),
                "capex_ratio_pct": self.capex_ratio(),
                "subsidies_ratio_pct": self.subsidies_ratio(),
                "budget_execution_rate_pct": self.budget_execution_rate(),
            })
        return {k: v for k, v in base.items() if v is not None}

    def get_all_statuses(self) -> dict:
        base = {
            "gross_profit_margin": self.gross_profit_margin_status(),
            "ebitda_margin": self.ebitda_margin_status(),
            "net_profit_margin": self.net_profit_margin_status(),
            "debt_to_equity": self.debt_to_equity_status(),
            "debt_service_coverage": self.debt_service_coverage_status(),
            "current_ratio": self.current_ratio_status(),
            "roi": self.roi_status(),
            "cash_flow": self.cash_flow_status(),
            "salary_ratio": self.salary_ratio_status(),
        }
        if self._is_gov:
            base["budget_execution"] = self.budget_execution_status()
        return base