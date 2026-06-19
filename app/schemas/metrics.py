"""
schemas/metrics.py — Enterprise financial metrics schemas.
All monetary values in MAD millions.
"""

from typing import Optional
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    entity_type: str = Field(default="corporate", description="'corporate' or 'government'")
    entity_name: Optional[str] = None
    sector: Optional[str] = None
    fiscal_year: Optional[int] = None

    # Revenue
    total_revenue: Optional[float] = Field(default=None, ge=0, description="MAD millions")
    operating_revenue: Optional[float] = Field(default=None, ge=0)
    revenue_year2: Optional[float] = Field(default=None, ge=0, description="Prior year revenue")

    # Government receipts
    tax_revenue: Optional[float] = Field(default=None, ge=0)
    non_tax_revenue: Optional[float] = Field(default=None, ge=0)
    grants_and_transfers: Optional[float] = Field(default=None, ge=0)

    # Costs
    cost_of_goods_sold: Optional[float] = Field(default=None, ge=0)
    operating_expenses: Optional[float] = Field(default=None, ge=0)
    salaries_and_benefits: Optional[float] = Field(default=None, ge=0)
    depreciation_amortization: Optional[float] = Field(default=None, ge=0)
    interest_expense: Optional[float] = Field(default=None, ge=0)
    tax_expense: Optional[float] = Field(default=None, ge=0)
    total_expenditure: Optional[float] = Field(default=None, ge=0)

    # Government expenditures
    capital_expenditure: Optional[float] = Field(default=None, ge=0)
    recurrent_expenditure: Optional[float] = Field(default=None, ge=0)
    debt_service: Optional[float] = Field(default=None, ge=0)
    subsidies_paid: Optional[float] = Field(default=None, ge=0)

    # Balance sheet
    total_assets: Optional[float] = Field(default=None, ge=0)
    current_assets: Optional[float] = Field(default=None, ge=0)
    current_liabilities: Optional[float] = Field(default=None, ge=0)
    total_equity: Optional[float] = Field(default=None)
    total_debt: Optional[float] = Field(default=None, ge=0)

    # Cash
    cash_and_equivalents: Optional[float] = Field(default=None, ge=0)
    cash_inflow: Optional[float] = Field(default=None, ge=0)
    cash_outflow: Optional[float] = Field(default=None, ge=0)
    operating_cash_flow: Optional[float] = Field(default=None)
    investing_cash_flow: Optional[float] = Field(default=None)

    # Investment
    own_capital_invested: Optional[float] = Field(default=None, ge=0)
    external_funding: Optional[float] = Field(default=None, ge=0)
    investment_budget: Optional[float] = Field(default=None, ge=0)
    investment_executed: Optional[float] = Field(default=None, ge=0)

    months: int = Field(default=12, ge=1, le=12)

    model_config = {
        "json_schema_extra": {
            "example": {
                "entity_type": "corporate",
                "entity_name": "Groupe OCP",
                "sector": "chimie / phosphates",
                "total_revenue": 87.5,
                "cost_of_goods_sold": 42.0,
                "operating_expenses": 12.0,
                "salaries_and_benefits": 8.5,
                "depreciation_amortization": 3.2,
                "interest_expense": 1.8,
                "tax_expense": 4.5,
                "total_assets": 220.0,
                "current_assets": 45.0,
                "current_liabilities": 28.0,
                "total_equity": 110.0,
                "total_debt": 65.0,
                "cash_inflow": 95.0,
                "cash_outflow": 88.0,
                "own_capital_invested": 30.0,
                "external_funding": 20.0,
            }
        }
    }


class MetricsOutput(BaseModel):
    entity_type: str = "corporate"

    # Revenue
    total_revenue_mad_m: Optional[float] = None
    revenue_growth_rate_pct: Optional[float] = None

    # Profitability
    gross_profit_mad_m: Optional[float] = None
    gross_profit_margin_pct: Optional[float] = None
    ebitda_mad_m: Optional[float] = None
    ebitda_margin_pct: Optional[float] = None
    operating_profit_mad_m: Optional[float] = None
    operating_margin_pct: Optional[float] = None
    net_profit_mad_m: Optional[float] = None
    net_profit_margin_pct: Optional[float] = None

    # Solvency
    debt_to_equity: Optional[float] = None
    debt_to_revenue: Optional[float] = None
    debt_service_coverage: Optional[float] = None
    equity_ratio_pct: Optional[float] = None

    # Liquidity
    current_ratio: Optional[float] = None
    cash_flow_net_mad_m: Optional[float] = None
    free_cash_flow_mad_m: Optional[float] = None
    cash_coverage_months: Optional[float] = None

    # Investment
    total_investment_mad_m: Optional[float] = None
    roi_pct: Optional[float] = None
    return_on_assets_pct: Optional[float] = None
    return_on_equity_pct: Optional[float] = None

    # Other
    salary_ratio_pct: Optional[float] = None

    # Government only
    primary_balance_mad_m: Optional[float] = None
    overall_balance_mad_m: Optional[float] = None
    fiscal_pressure_pct: Optional[float] = None
    capex_ratio_pct: Optional[float] = None
    subsidies_ratio_pct: Optional[float] = None
    budget_execution_rate_pct: Optional[float] = None

    # Health
    health_score: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    statuses: dict = Field(default_factory=dict)