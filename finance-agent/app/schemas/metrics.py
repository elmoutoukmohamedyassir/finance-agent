from typing import Optional
from pydantic import BaseModel, Field


class MetricsInput(BaseModel):
    mrr: Optional[float] = Field(default=None, ge=0)
    arr: Optional[float] = Field(default=None, ge=0)
    customer_count: Optional[int] = Field(default=None, ge=0)
    arpu: Optional[float] = Field(default=None, ge=0)
    churn_rate: Optional[float] = Field(default=None, ge=0, le=100, description="Monthly churn as percent e.g. 5 = 5%")
    cac: Optional[float] = Field(default=None, ge=0)
    monthly_costs: Optional[float] = Field(default=None, ge=0)
    marketing_budget: Optional[float] = Field(default=None, ge=0)
    new_customers_per_month: Optional[int] = Field(default=None, ge=0)
    gross_margin: Optional[float] = Field(default=None, ge=0, le=100, description="Gross margin percent e.g. 75 = 75%")

    class Config:
        json_schema_extra = {
            "example": {
                "mrr": 12000,
                "customer_count": 80,
                "churn_rate": 4.0,
                "cac": 250,
                "monthly_costs": 7000,
                "marketing_budget": 1500,
                "new_customers_per_month": 6
            }
        }


class MetricsOutput(BaseModel):
    mrr: Optional[float] = None
    arr: Optional[float] = None
    arpu: Optional[float] = None
    churn_rate: Optional[float] = None
    retention_rate: Optional[float] = None
    customer_lifetime_months: Optional[float] = None
    ltv: Optional[float] = None
    cac: Optional[float] = None
    ltv_cac_ratio: Optional[float] = None
    cac_payback_months: Optional[float] = None
    monthly_profit: Optional[float] = None
    profit_margin: Optional[float] = None
    burn_rate: Optional[float] = None
    health_score: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
