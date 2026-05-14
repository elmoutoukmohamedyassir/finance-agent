"""
schemas/chat.py — Request and response models for the chat endpoint.

WHY Pydantic schemas:
  - FastAPI auto-validates incoming requests against these models
  - If a required field is missing or wrong type → 422 error with clear message
  - Auto-generates OpenAPI docs (Swagger UI) for free
  - Single source of truth for what the API accepts and returns
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    What the client sends to /chat.
    
    session_id: Identifies the conversation. Client should generate a UUID
                and reuse it across messages in the same conversation.
                If omitted, a new session is created.
    message: The user's current message.
    """
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation session ID. Generate a UUID client-side and reuse it."
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's message"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "message": "I'm building a SaaS project management tool targeting small agencies."
            }
        }


class ChatResponse(BaseModel):
    """What the server returns from /chat."""
    session_id: str = Field(description="Use this in subsequent requests")
    message: str = Field(description="The agent's response")
    agent_mode: str = Field(
        description="What mode the agent was in: 'gathering_info' | 'analyzing' | 'answering' | 'off_topic'"
    )
    metrics_calculated: Optional[dict] = Field(
        default=None,
        description="Any metrics that were calculated this turn (for UI display)"
    )
    business_state: Optional[dict] = Field(
        default=None,
        description="Current collected business info (so UI can show progress)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "abc123",
                "message": "Great! What's your expected monthly pricing per customer?",
                "agent_mode": "gathering_info",
                "metrics_calculated": None,
                "business_state": {
                    "business_name": "ProjectFlow",
                    "target_audience": "small agencies"
                }
            }
        }
