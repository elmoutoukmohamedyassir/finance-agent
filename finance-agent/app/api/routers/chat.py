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
    """
    Multi-Phase Finance Chat - From Idea to Business Plan
    
    This is your financial advisor for entrepreneurs in Morocco.
    We work in phases:
    
    **Phase 1 (Ideation)**: Chat naturally about your business idea
    **Phase 2 (Data)**: Answer specific questions about your business model & finances
    **Phase 3 (Analysis)**: Get detailed financial analysis, KPIs, tax guidance, business plan
    **Phase 4 (Scaling)**: Ongoing support after launch
    
    Just start by describing your idea!
    
    ## Examples:
    
    **Phase 1 - Ideation:**
    ```json
    {
      "message": "I want to start a consulting business in Morocco"
    }
    ```
    Response: Natural conversation about your idea
    
    **Phase 2 - Data Collection (automatic):**
    After discussing, agent asks specific questions about business model, costs, etc.
    
    **Phase 3 - Analysis (automatic):**
    After data collected, agent provides:
    - Break-even calculations
    - Monthly cash flow projections
    - Tax obligations in Morocco (CNSS, TVA, IS)
    - 3 financial scenarios (optimistic, pessimistic, realistic)
    - Recommended business structure
    - 24-month business plan
    
    **With Email Tracking:**
    ```json
    {
      "message": "Help me build my financial plan",
      "client_email": "entrepreneur@example.com"
    }
    ```
    Data is saved for follow-up.
    
    **Multi-turn:**
    ```json
    {
      "session_id": "session-from-first-message",
      "message": "What if my costs were 20% higher?"
    }
    ```
    Agent remembers context and adjusts analysis.
    """
    try:
        # Get or create session (no db parameter - uses SQLite directly)
        session = get_or_create_session(request.session_id)
        
        # Get or create client if email provided
        if request.client_email:
            get_or_create_client(db, request.client_email)
        
        # Prepare agent message
        agent_message = AgentMessage(
            sender_agent_id="user",
            session_id=session.session_id,
            user_message=request.message,
            intent="chat",
            context={
                "business_state": session.business_state.model_dump() if session.business_state else {},
                "in_collection": session.phase == "collection",
                "phase": session.phase or "ideation",
            },
        )
        
        # Route to appropriate phase agent
        agent_response = phase_router.route_message(agent_message, agent_message.context)
        
        # Update session state
        if agent_response.business_state:
            session.business_state = agent_response.business_state
        if agent_response.structured_output:
            phase = agent_response.structured_output.get("phase", session.phase)
            session.phase = phase
        
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

