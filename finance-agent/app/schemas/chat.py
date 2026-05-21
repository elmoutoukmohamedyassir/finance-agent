"""schemas/chat.py"""
from typing import Optional
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=5000)
    hypothesis_payload: Optional[dict] = Field(
        default=None,
        description="Full HypothesisOutput JSON from Hypothesis Agent — bypasses collection phase"
    )

class ChatResponse(BaseModel):
    session_id: str = ""
    message: str
    agent_mode: str = "collecting"
    current_phase: str = "welcome"
    metrics_calculated: Optional[dict] = None
    business_state: Optional[dict] = None
    plan_output: Optional[dict] = None