"""api/routers/chat.py"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.schemas.chat import ChatRequest, ChatResponse
from app.agents.phase_router import PhaseRouter
from app.agents.base_agent import AgentMessage
from app.database.db import get_db, init_db
from app.services.client_service import get_or_create_client
from app.services.session_service import get_or_create_session, save_session
import logging
import uuid

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

# Initialize database on startup
init_db()

# Phase router orchestrator
phase_router = PhaseRouter()


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    try:
        # Get or create session (no db parameter - uses SQLite directly)
        session = get_or_create_session(request.session_id)

        # Get or create client if email provided
        if request.client_email:
            get_or_create_client(db, request.client_email)

        # Save the user's message to conversation history before routing
        session.add_message("user", request.message)

        # Prepare agent message — pass full conversation history and session state
        agent_message = AgentMessage(
            sender_agent_id="user",
            session_id=session.session_id,
            user_message=request.message,
            intent="chat",
            context={
                "business_state": session.business_state.model_dump() if session.business_state else {},
                "in_collection": session.phase == "collection",
                "phase": session.phase or "ideation",
                # Pass persisted per-session Q&A tracking data to agents
                "conversation_history": session.conversation_history[:-1],  # exclude the message we just added
                "asked_questions": session.questions_asked,
                "pending_question": session.pending_question,
            },
        )

        # Route to appropriate phase agent
        agent_response = phase_router.route_message(agent_message, agent_message.context)

        # Save the assistant's reply to conversation history
        if agent_response.message:
            session.add_message("assistant", agent_response.message)

        # Update session state
        if agent_response.business_state:
            session.business_state = agent_response.business_state
        if agent_response.structured_output:
            phase = agent_response.structured_output.get("phase", session.phase)
            session.phase = phase
            # Persist Q&A tracking state back to the session
            if "asked_questions" in agent_response.structured_output:
                session.questions_asked = agent_response.structured_output["asked_questions"]
            if "pending_question" in agent_response.structured_output:
                session.pending_question = agent_response.structured_output["pending_question"]

        # Save session to SQLite
        save_session(session)

        # Build response
        return ChatResponse(
            session_id=session.session_id,
            message=agent_response.message,
            agent_mode=agent_response.agent_mode,
            metrics_calculated=agent_response.metrics_calculated,
            business_state=agent_response.business_state,
            kpi_suggestions=None,
            kpi_details=None,
            action_items=None,
            metadata={
                "llm_used": "groq",
                "phase": agent_response.structured_output.get("phase", "unknown") if agent_response.structured_output else "unknown",
            },
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return ChatResponse(
            session_id=request.session_id or "error",
            message=f"Erreur: {str(e)}",
            agent_mode="error",
            metadata={"error": str(e)},
        )