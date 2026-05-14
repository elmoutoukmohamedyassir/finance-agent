"""
schemas/session.py — Data models for conversation sessions.

A session holds:
  1. conversation_history: The raw message turns (role + content)
  2. business_state: Structured data about the user's SaaS business
                     collected progressively through conversation

WHY separate business_state from conversation_history:
  - history = raw text turns for LLM context
  - business_state = structured, validated data for calculation
  - This lets us pass structured data to the metrics calculator
    without re-parsing the whole conversation every time
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class BusinessState(BaseModel):
    """
    Structured representation of the user's SaaS business information.
    
    Fields are populated progressively as the user shares information.
    None = not yet provided.
    
    WHY we track this separately: The question_agent uses this to determine
    what to ask next. The metrics_calculator uses this for calculations.
    """
    # Identity
    business_name: Optional[str] = None
    business_model: Optional[str] = None       # B2B | B2C | B2B2C
    target_audience: Optional[str] = None
    funding_stage: Optional[str] = None

    # Revenue inputs
    mrr: Optional[float] = None
    arr: Optional[float] = None
    customer_count: Optional[int] = None
    arpu: Optional[float] = None               # $/month per customer
    pricing_plan: Optional[str] = None

    # Growth & churn
    churn_rate: Optional[float] = None         # monthly %
    new_customers_per_month: Optional[int] = None
    growth_rate: Optional[float] = None        # monthly %

    # Costs
    monthly_costs: Optional[float] = None
    marketing_budget: Optional[float] = None
    cac: Optional[float] = None
    gross_margin: Optional[float] = None       # %

    def fields_collected(self) -> list[str]:
        """Returns list of field names that have been filled."""
        return [k for k, v in self.model_dump().items() if v is not None]

    def missing_critical_fields(self) -> list[str]:
        """
        Returns the critical fields still missing for analysis.
        We need at minimum revenue info + cost info.
        """
        critical = []
        has_revenue = self.mrr or (self.customer_count and self.arpu)
        has_costs = self.monthly_costs is not None

        if not has_revenue:
            if not self.mrr:
                critical.append("mrr")
            if not self.customer_count:
                critical.append("customer_count")
        if not has_costs:
            critical.append("monthly_costs")

        return critical

    def is_ready_for_analysis(self) -> bool:
        """True when we have enough to run meaningful financial analysis."""
        return len(self.missing_critical_fields()) == 0


class ConversationSession(BaseModel):
    """
    Full session object stored in memory.
    """
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    conversation_history: list[dict] = Field(
        default_factory=list,
        description="List of {'role': 'user'|'assistant', 'content': '...'}"
    )
    business_state: BusinessState = Field(default_factory=BusinessState)
    questions_asked: list[str] = Field(
        default_factory=list,
        description="Tracks which questions have been asked to avoid repeating"
    )

    def add_message(self, role: str, content: str):
        """Append a message to history and update timestamp."""
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()
