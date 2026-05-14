"""
schemas/session.py — Data models for conversation sessions.

BusinessState holds structured financial data collected progressively.
ConversationSession ties history + state together per user.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class BusinessState(BaseModel):
    """
    Structured SaaS business data collected through conversation.
    Every field starts as None and gets filled progressively.
    """
    business_name: Optional[str] = None
    business_model: Optional[str] = None
    target_audience: Optional[str] = None
    funding_stage: Optional[str] = None

    mrr: Optional[float] = None
    arr: Optional[float] = None
    customer_count: Optional[int] = None
    arpu: Optional[float] = None
    pricing_plan: Optional[str] = None

    churn_rate: Optional[float] = None
    new_customers_per_month: Optional[int] = None
    growth_rate: Optional[float] = None

    monthly_costs: Optional[float] = None
    marketing_budget: Optional[float] = None
    cac: Optional[float] = None
    gross_margin: Optional[float] = None

    def has_revenue_info(self) -> bool:
        """True if we can determine MRR somehow."""
        return bool(
            self.mrr
            or (self.customer_count and self.arpu)
            or (self.arr)
        )

    def has_cost_info(self) -> bool:
        """True if we have operating costs."""
        return self.monthly_costs is not None

    def is_ready_for_analysis(self) -> bool:
        """
        Minimum viable data for a meaningful financial analysis.
        We need at least revenue info + cost info.
        """
        return self.has_revenue_info() and self.has_cost_info()

    def filled_fields(self) -> dict:
        """Returns only fields that have been filled."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ConversationSession(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Raw conversation turns for LLM context
    conversation_history: list[dict] = Field(default_factory=list)

    # Structured business data extracted from conversation
    business_state: BusinessState = Field(default_factory=BusinessState)

    # Tracks questions asked to avoid repetition
    questions_asked: list[str] = Field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()
