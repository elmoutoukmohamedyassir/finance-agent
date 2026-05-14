"""
agents/finance_agent.py — Main orchestrator.

SCOPE CONTROL: finance_guard.py is REMOVED.
Scope is enforced through FINANCE_AGENT_SYSTEM_PROMPT which contains explicit
instructions on how to handle off-topic questions. This is simpler, cheaper
(no extra API call), and more context-aware (the LLM understands the full
conversation, a keyword filter does not).

FLOW:
  1. Load/create session
  2. Extract any business data from message (LLM extraction)
  3. If not enough data → ask next question
  4. If enough data → calculate metrics (pure Python) → retrieve RAG →
     send pre-calculated data to LLM for interpretation
"""

import logging

from app.core.groq_client import groq_client
from app.core.prompts import (
    FINANCE_AGENT_SYSTEM_PROMPT,
    build_analysis_system_prompt,
    build_analysis_user_message,
)
from app.tools.metrics_calculator import calculate_from_business_state
from app.tools.scenario_engine import build_standard_scenarios, format_scenarios_for_prompt
from app.agents.question_agent import extract_business_info, get_next_question, should_analyze
from app.rag.retriever import retrieve_context
from app.schemas.chat import ChatResponse
from app.schemas.session import ConversationSession
from app.services.session_service import (
    get_or_create_session,
    save_session,
    update_business_state,
)

logger = logging.getLogger(__name__)


def handle_chat(session_id: str | None, user_message: str) -> ChatResponse:
    """Entry point: takes a message, returns a structured response."""
    session = get_or_create_session(session_id)

    # Extract any structured data from this message
    extracted = extract_business_info(user_message)
    if extracted:
        update_business_state(session, extracted)

    session.add_message("user", user_message)

    state = session.business_state

    if not should_analyze(state):
        response = _ask_next_question(session)
    else:
        response = _run_analysis(session)

    session.add_message("assistant", response.message)
    save_session(session)

    response.session_id = session.session_id
    response.business_state = state.filled_fields()
    return response


def _ask_next_question(session: ConversationSession) -> ChatResponse:
    """Asks the next missing question. On first turn, adds a welcome message."""
    question = get_next_question(
        state=session.business_state,
        asked_questions=session.questions_asked,
    )

    if not question:
        # Fallback: we somehow have no more questions but can't analyze yet
        return _general_answer(session)

    session.questions_asked.append(question)

    is_first_turn = len(session.conversation_history) == 1
    if is_first_turn:
        message = (
            "Hi! I'm your AI SaaS Finance Advisor. I help founders evaluate "
            "their business financially before and during launch.\n\n"
            f"Let's get started — {question}"
        )
    else:
        message = question

    return ChatResponse(
        session_id=session.session_id,
        message=message,
        agent_mode="gathering_info",
    )


def _run_analysis(session: ConversationSession) -> ChatResponse:
    """
    Full analysis:
    1. Pure Python metric calculation (no LLM → no hallucinations)
    2. Pure Python scenario projection
    3. RAG retrieval (relevant finance document chunks)
    4. LLM interprets pre-calculated data (grounded, not free-form)
    """
    state = session.business_state

    # Step 1: Calculate metrics — pure Python
    metrics = calculate_from_business_state(state)
    metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

    # Step 2: Scenarios — pure Python
    scenarios_str = ""
    if state.customer_count and state.monthly_costs is not None:
        price = state.arpu or (state.mrr / state.customer_count if state.mrr and state.customer_count else 0)
        if price > 0:
            scenarios = build_standard_scenarios(
                starting_customers=state.customer_count,
                monthly_price=price,
                monthly_costs=state.monthly_costs,
                starting_cash=50000,
                months=12,
            )
            scenarios_str = format_scenarios_for_prompt(scenarios)

    # Step 3: RAG retrieval — finance documents
    rag_query = _build_rag_query(state, metrics_dict)
    rag_context = retrieve_context(rag_query)

    # Step 4: Build prompt and call LLM
    system_prompt = build_analysis_system_prompt(rag_context=rag_context)
    user_msg = build_analysis_user_message(
        state_dict=state.filled_fields(),
        metrics_dict=metrics_dict,
        scenarios_str=scenarios_str,
    )

    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=user_msg,
        conversation_history=session.conversation_history[-6:],
        temperature=0.2,
        max_tokens=1800,
    )

    return ChatResponse(
        session_id=session.session_id,
        message=response_text,
        agent_mode="analyzing",
        metrics_calculated=metrics_dict,
    )


def _general_answer(session: ConversationSession) -> ChatResponse:
    """
    General finance Q&A — used for follow-up questions after analysis,
    or exploratory finance questions not requiring full business data.
    
    The system prompt handles off-topic rejection through prompt engineering.
    """
    last_message = session.conversation_history[-1]["content"]
    rag_context = retrieve_context(last_message)
    system_prompt = build_analysis_system_prompt(rag_context=rag_context)

    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=last_message,
        conversation_history=session.conversation_history[-10:],
        temperature=0.3,
        max_tokens=1000,
    )

    return ChatResponse(
        session_id=session.session_id,
        message=response_text,
        agent_mode="answering",
    )


def _build_rag_query(state, metrics_dict: dict) -> str:
    """Build a targeted semantic search query based on the business situation."""
    parts = ["SaaS financial analysis startup metrics"]
    if state.churn_rate and state.churn_rate > 5:
        parts.append("customer churn retention strategies")
    if metrics_dict.get("ltv_cac_ratio") and metrics_dict["ltv_cac_ratio"] < 3:
        parts.append("LTV CAC ratio improvement acquisition cost")
    if metrics_dict.get("burn_rate"):
        parts.append("burn rate runway cash management")
    if state.business_model:
        parts.append(f"{state.business_model} SaaS pricing model")
    return " ".join(parts)
