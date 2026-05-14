from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Reuse this across messages to keep conversation context. Omit for a new session."
    )
    message: str = Field(..., min_length=1, max_length=3000)

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc-123",
                "message": "I'm building a B2B SaaS for HR teams. We have 40 customers at $150/month."
            }
        }


class ChatResponse(BaseModel):
    session_id: str
    message: str
    agent_mode: str = Field(
        description="gathering_info | analyzing | answering | off_topic"
    )
    metrics_calculated: Optional[dict] = None
    business_state: Optional[dict] = None
