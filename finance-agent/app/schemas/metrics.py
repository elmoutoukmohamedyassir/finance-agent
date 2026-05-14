"""
schemas/metrics.py — Models for the direct metrics calculation endpoint.

This endpoint is for when the user wants to provide all numbers upfront
and get metrics back immediately (bypasses the conversational flow).
"""

from typing import Optional
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    """
    Financial inputs for SaaS metric calculation.
    All fields are optional — we calculate what we can from what we have.
    """
    mrr: Optional[float] = Field(default=None, ge=0, description="Monthly Recurring Revenue ($)")
    arr: Optional[float] = Field(default=None, ge=0, description="Annual Recurring Revenue ($)")
    customer_count: Optional[int] = Field(default=None, ge=0, description="Total paying customers")
    arpu: Optional[float] = Field(default=None, ge=0, description="Avg Revenue Per User ($/month)")
    churn_rate: Optional[float] = Field(default=None, ge=0, le=100, description="Monthly churn rate (%)")
    cac: Optional[float] = Field(default=None, ge=0, description="Customer Acquisition Cost ($)")
    monthly_costs: Optional[float] = Field(default=None, ge=0, description="Total monthly costs ($)")
    marketing_budget: Optional[float] = Field(default=None, ge=0, description="Monthly marketing spend ($)")
    new_customers_per_month: Optional[int] = Field(default=None, ge=0, description="New customers acquired per month")
    gross_margin: Optional[float] = Field(default=None, ge=0, le=100, description="Gross margin (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "mrr": 15000,
                "customer_count": 120,
                "churn_rate": 3.5,
                "cac": 200,
                "monthly_costs": 8000,
                "marketing_budget": 2000,
                "new_customers_per_month": 20
            }
        }


class MetricsOutput(BaseModel):
    """
    Calculated SaaS metrics returned to the client.
    None means the metric could not be calculated (missing inputs).
    """
    # Core revenue metrics
    mrr: Optional[float] = Field(default=None, description="Monthly Recurring Revenue ($)")
    arr: Optional[float] = Field(default=None, description="Annual Recurring Revenue ($)")
    arpu: Optional[float] = Field(default=None, description="Average Revenue Per User ($/month)")

    # Growth & retention
    churn_rate: Optional[float] = Field(default=None, description="Monthly churn rate (%)")
    retention_rate: Optional[float] = Field(default=None, description="Monthly retention rate (%)")
    customer_lifetime_months: Optional[float] = Field(default=None, description="Expected customer lifetime (months)")

    # Unit economics
    ltv: Optional[float] = Field(default=None, description="Customer Lifetime Value ($)")
    cac: Optional[float] = Field(default=None, description="Customer Acquisition Cost ($)")
    ltv_cac_ratio: Optional[float] = Field(default=None, description="LTV:CAC ratio (healthy = 3x or above)")
    cac_payback_months: Optional[float] = Field(default=None, description="Months to recoup CAC")

    # Profitability
    monthly_profit: Optional[float] = Field(default=None, description="Monthly profit/loss ($)")
    profit_margin: Optional[float] = Field(default=None, description="Profit margin (%)")
    burn_rate: Optional[float] = Field(default=None, description="Monthly burn rate if unprofitable ($)")

    # Health assessment
    health_score: Optional[str] = Field(default=None, description="Overall health: Healthy | Warning | Critical")
    warnings: list[str] = Field(default_factory=list, description="Metrics that need attention")

    class Config:
        json_schema_extra = {
            "example": {
                "mrr": 15000,
                "arr": 180000,
                "arpu": 125.0,
                "churn_rate": 3.5,
                "retention_rate": 96.5,
                "customer_lifetime_months": 28.6,
                "ltv": 3571.0,
                "cac": 200.0,
                "ltv_cac_ratio": 17.9,
                "cac_payback_months": 1.6,
                "monthly_profit": 7000,
                "profit_margin": 46.7,
                "burn_rate": None,
                "health_score": "Healthy",
                "warnings": []
            }
        }
