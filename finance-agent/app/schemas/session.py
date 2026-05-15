"""
schemas/session.py — Session and business state for enterprise financial agent.

Replaces SaaS-specific fields (MRR, ARR, churn, ARPU) with enterprise fields
covering both Corporate and Government entities. Values in MAD millions.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class BusinessState(BaseModel):
    """
    Structured enterprise financial data collected through conversation.
    Every field starts as None and gets filled progressively.
    entity_type drives which KPIs are computed and which questions are asked.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None     # "corporate" or "government"
    sector: Optional[str] = None
    fiscal_year: Optional[int] = None

    # ── Revenue / Receipts (MAD millions) ────────────────────────────────
    total_revenue: Optional[float] = None
    revenue_year2: Optional[float] = None   # Prior year for growth calc

    # Government-specific
    tax_revenue: Optional[float] = None
    non_tax_revenue: Optional[float] = None
    grants_and_transfers: Optional[float] = None

    # ── Costs / Expenditures (MAD millions) ──────────────────────────────
    cost_of_goods_sold: Optional[float] = None
    operating_expenses: Optional[float] = None
    salaries_and_benefits: Optional[float] = None
    depreciation_amortization: Optional[float] = None
    interest_expense: Optional[float] = None
    tax_expense: Optional[float] = None
    total_expenditure: Optional[float] = None

    # Government-specific
    capital_expenditure: Optional[float] = None
    recurrent_expenditure: Optional[float] = None
    debt_service: Optional[float] = None
    subsidies_paid: Optional[float] = None

    # ── Balance Sheet (MAD millions) ─────────────────────────────────────
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    total_debt: Optional[float] = None

    # ── Cash (MAD millions) ──────────────────────────────────────────────
    cash_and_equivalents: Optional[float] = None
    cash_inflow: Optional[float] = None
    cash_outflow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    investing_cash_flow: Optional[float] = None

    # ── Investment (MAD millions) ────────────────────────────────────────
    own_capital_invested: Optional[float] = None
    external_funding: Optional[float] = None
    investment_budget: Optional[float] = None
    investment_executed: Optional[float] = None

    months: int = 12

    # ── Readiness checks ─────────────────────────────────────────────────

    def has_revenue_info(self) -> bool:
        if self.entity_type == "government":
            return bool(
                self.total_revenue
                or (self.tax_revenue and self.non_tax_revenue)
            )
        return bool(self.total_revenue)

    def has_cost_info(self) -> bool:
        if self.entity_type == "government":
            return bool(
                self.total_expenditure
                or (self.recurrent_expenditure and self.capital_expenditure)
            )
        return bool(
            self.operating_expenses is not None
            or self.cost_of_goods_sold is not None
        )

    def is_ready_for_analysis(self) -> bool:
        """Minimum viable: entity type + revenue + costs."""
        return bool(
            self.entity_type
            and self.has_revenue_info()
            and self.has_cost_info()
        )

    def filled_fields(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ConversationSession(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    conversation_history: list[dict] = Field(default_factory=list)
    business_state: BusinessState = Field(default_factory=BusinessState)
    questions_asked: list[str] = Field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()