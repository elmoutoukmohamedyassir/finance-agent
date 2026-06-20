"""api/routers/chat.py"""
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response
from sqlalchemy.orm import Session
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.session import BusinessState
from app.agents.phase_router import PhaseRouter
from app.agents.base_agent import AgentMessage
from app.database.db import get_db, init_db
from app.services.client_service import get_or_create_client
from app.services.session_service import get_or_create_session, save_session
from app.tools.plan_pipeline import compute_plan, has_minimum_data
from app.tools.plan_pdf import build_plan_pdf
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
                # router_phase is the ONLY phase signal the live phase-agent
                # pipeline trusts. It's a free-form string persisted on the
                # session (see schemas/session.py) — never the strict Phase enum.
                "router_phase": session.router_phase,
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

        # Merge any business_state updates the agent produced. Agents may put
        # it on the top-level `business_state` field (Phase 3) or inside
        # `structured_output["business_state"]` (Phase 2) — check both, and
        # MERGE rather than overwrite so we never lose previously collected
        # fields. We rebuild a validated BusinessState (never assign a raw
        # dict directly — that breaks every later `.model_dump()` call).
        updated_bs = None
        if agent_response.business_state:
            updated_bs = agent_response.business_state
        elif agent_response.structured_output and agent_response.structured_output.get("business_state"):
            updated_bs = agent_response.structured_output["business_state"]

        if updated_bs:
            merged = {
                **session.business_state.model_dump(),
                **{k: v for k, v in updated_bs.items() if v is not None and k in BusinessState.model_fields},
            }
            session.business_state = BusinessState(**merged)

        # Update session router state
        if agent_response.structured_output:
            if "next_phase" in agent_response.structured_output:
                session.router_phase = agent_response.structured_output["next_phase"]
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
            business_state=session.business_state.model_dump(),
            kpi_suggestions=None,
            kpi_details=None,
            action_items=None,
            metadata={
                "llm_used": "groq",
                "phase": session.router_phase,
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


@router.get("/{session_id}/plan/pdf")
async def download_plan_pdf(session_id: str):
    """
    Generate and download the full business plan as a PDF for this session.
    Recomputes from the session's stored business_state using the same
    tools.plan_pipeline logic Phase 3 uses in chat — so the PDF always
    matches whatever numbers the user already saw, never a separate
    calculation. Returns 400 (not 500) if there isn't enough data yet,
    since that's an expected state, not a server error.
    """
    session = get_or_create_session(session_id)
    business_state = session.business_state.model_dump() if session.business_state else {}

    if not has_minimum_data(business_state):
        raise HTTPException(
            status_code=400,
            detail=(
                "Pas encore assez de données pour générer le plan financier complet. "
                "Terminez la collecte des informations (prix de vente, clients, charges fixes) "
                "avant de télécharger le PDF."
            ),
        )

    computed = compute_plan(business_state)
    if not computed:
        raise HTTPException(
            status_code=400,
            detail="Le calcul du plan financier a échoué. Vérifiez les données saisies.",
        )

    pdf_bytes = build_plan_pdf(business_state, computed)

    entity_name = business_state.get("entity_name") or "business_plan"
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", entity_name).strip("_") or "business_plan"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="business_plan_{safe_name}.pdf"'},
    )