"""
api/routers/chat.py — The main conversational chat endpoint.

This is the primary interface users interact with.
It delegates all logic to the finance agent and session service.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.finance_agent import handle_chat
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Main conversational endpoint.

    Send messages here to interact with the Finance Agent.
    The agent will:
    - Ask follow-up questions to gather business info
    - Calculate SaaS metrics when enough data is collected
    - Provide financial analysis and recommendations

    **session_id**: Generate a UUID client-side and reuse it across messages
    to maintain conversation continuity. If omitted, a new session starts.
    """
    try:
        return handle_chat(
            session_id=request.session_id,
            user_message=request.message,
        )
    except RuntimeError as e:
        # RuntimeError from groq_client means API issue
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
