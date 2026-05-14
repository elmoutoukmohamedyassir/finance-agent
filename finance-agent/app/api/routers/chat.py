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
    
    - Send your first message to start a new session (omit session_id)
    - Copy the returned session_id and reuse it for all follow-up messages
    - The agent collects your business data progressively, then analyzes
    """
    try:
        return handle_chat(
            session_id=request.session_id,
            user_message=request.message,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error")
