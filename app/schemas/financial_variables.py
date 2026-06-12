"""
financial_variables.py — Enterprise financial data model.

Covers both Corporate and Government/Public sector entities.
All monetary values in MAD (Moroccan Dirham), millions scale.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinancialData:
    """
    Core financial inputs for enterprise analysis.
    All monetary fields in MAD millions unless noted.
    entity_type: "corporate" | "government"
    """

    entity_type: str = "corporate"         # "corporate" or "government"
    entity_name: Optional[str] = None
    sector: Optional[str] = None           # e.g. "manufacturing", "public health", "retail"
    fiscal_year: Optional[int] = None

    # ── Revenue / Receipts ────────────────────────────────────────────────
    total_revenue: float = 0.0             # Corporate: net revenue / Gov: total receipts
    operating_revenue: float = 0.0        # Revenue from core operations
    non_operating_revenue: float = 0.0    # Dividends, asset sales, grants
    revenue_year2: float = 0.0            # Prior year (for growth calc)
    revenue_year3: float = 0.0            # Two years prior

    # ── Government-specific receipts ──────────────────────────────────────
    tax_revenue: float = 0.0              # IR, IS, TVA, droits de douane
    non_tax_revenue: float = 0.0          # Redevances, amendes, recettes domaniales
    grants_and_transfers: float = 0.0     # Subventions reçues

    # ── Costs / Expenditures ──────────────────────────────────────────────
    cost_of_goods_sold: float = 0.0       # COGS / coût des biens vendus
    operating_expenses: float = 0.0       # Charges d'exploitation hors COGS
    salaries_and_benefits: float = 0.0    # Masse salariale
    depreciation_amortization: float = 0.0  # DAP
    interest_expense: float = 0.0         # Charges financières
    tax_expense: float = 0.0              # IS / impôt sur les sociétés
    total_expenditure: float = 0.0        # Gov: total dépenses budgétaires

    # ── Government-specific expenditures ──────────────────────────────────
    capital_expenditure: float = 0.0      # CAPEX / investissement public
    recurrent_expenditure: float = 0.0    # Dépenses de fonctionnement
    debt_service: float = 0.0             # Principal + intérêts remboursés
    subsidies_paid: float = 0.0           # Subventions versées

    # ── Balance Sheet ─────────────────────────────────────────────────────
    total_assets: float = 0.0
    current_assets: float = 0.0
    non_current_assets: float = 0.0
    total_liabilities: float = 0.0
    current_liabilities: float = 0.0
    non_current_liabilities: float = 0.0
    total_equity: float = 0.0             # Capitaux propres / situation nette

    # ── Debt & Financing ──────────────────────────────────────────────────
    short_term_debt: float = 0.0
    long_term_debt: float = 0.0
    total_debt: float = 0.0               # short + long term
    new_borrowings: float = 0.0
    loan_repayments: float = 0.0

    # ── Cash & Liquidity ──────────────────────────────────────────────────
    cash_and_equivalents: float = 0.0
    cash_inflow: float = 0.0              # Total encaissements période
    cash_outflow: float = 0.0             # Total décaissements période
    operating_cash_flow: float = 0.0      # Cash from operations
    investing_cash_flow: float = 0.0      # Cash from investing activities
    financing_cash_flow: float = 0.0      # Cash from financing activities

    # ── Investment & Capital ──────────────────────────────────────────────
    own_capital_invested: float = 0.0     # Fonds propres investis
    external_funding: float = 0.0         # Emprunts + financements extérieurs
    investment_budget: float = 0.0        # Budget d'investissement approuvé
    investment_executed: float = 0.0      # Investissement réellement exécuté

    # ── Period ────────────────────────────────────────────────────────────
    months: int = 12                      # Reporting period length