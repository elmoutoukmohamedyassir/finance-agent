"""
agents/finance_agent.py — Main orchestrator for enterprise financial analysis.

Supports both Corporate and Government/Public sector entities.
All monetary values in MAD millions.

FLOW:
  1. Load/create session
  2. Extract any financial data from message (LLM extraction)
  3. If not enough data → ask next question
  4. If enough data → calculate KPIs (pure Python) → retrieve RAG →
     send pre-calculated data to LLM for interpretation only
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
        return _general_answer(session)

    session.questions_asked.append(question)

    is_first_turn = len(session.conversation_history) == 1
    if is_first_turn:
        message = (
            "Bonjour ! Je suis FinanceGPT, votre conseiller IA en analyse financière "
            "d'entreprises et de finances publiques au Maroc.\n\n"
            "Je peux analyser la santé financière d'entreprises (PME, grands groupes) "
            "ou d'entités publiques (ministères, collectivités, établissements publics) "
            "et vous fournir des indicateurs clés, des alertes et des recommandations actionnables.\n\n"
            f"Commençons — {question}"
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
    1. Pure Python KPI calculation (no LLM → no hallucinations)
    2. Pure Python scenario projection (3 years, 3 scenarios)
    3. RAG retrieval (relevant chunks from Bulletin mensuel, CGI, etc.)
    4. LLM interprets pre-calculated data only (grounded, not free-form)
    """
    state = session.business_state

    # Step 1: Calculate KPIs — pure Python
    metrics = calculate_from_business_state(state)
    metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

    # Step 2: Scenarios — pure Python
    scenarios_str = ""
    if state.total_revenue and (state.operating_expenses or state.total_expenditure):
        costs = (
            (state.cost_of_goods_sold or 0)
            + (state.operating_expenses or 0)
            + (state.salaries_and_benefits or 0)
        ) or state.total_expenditure or 0

        if costs > 0:
            scenarios = build_standard_scenarios(
                starting_revenue=state.total_revenue,
                starting_costs=costs,
                starting_cash=state.cash_and_equivalents or 0,
                starting_debt=state.total_debt or 0,
                debt_service_annual=state.debt_service or 0,
                capex_annual=state.capital_expenditure or 0,
                years=3,
                entity_type=state.entity_type or "corporate",
            )
            scenarios_str = format_scenarios_for_prompt(scenarios, years=3)

    # Step 3: RAG retrieval
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
    General finance Q&A — for follow-up questions after analysis,
    or exploratory questions that don't need full business data.
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
    """Build a targeted semantic search query based on the entity situation."""
    parts = ["analyse financière entreprise finances publiques Maroc"]

    if state.entity_type == "government":
        parts.append("budget état finances publiques recettes fiscales dépenses")
        if metrics_dict.get("budget_execution_rate_pct") and metrics_dict["budget_execution_rate_pct"] < 70:
            parts.append("taux exécution budgétaire faible investissement public")
        if metrics_dict.get("overall_balance_mad_m") and metrics_dict["overall_balance_mad_m"] < 0:
            parts.append("déficit budgétaire financement dette publique")
    else:
        if metrics_dict.get("ebitda_margin_pct") and metrics_dict["ebitda_margin_pct"] < 8:
            parts.append("amélioration marge EBITDA rentabilité opérationnelle")
        if metrics_dict.get("debt_service_coverage") and metrics_dict["debt_service_coverage"] < 1.5:
            parts.append("couverture service dette restructuration financière")
        if metrics_dict.get("current_ratio") and metrics_dict["current_ratio"] < 1:
            parts.append("liquidité trésorerie court terme financement")

    if state.sector:
        parts.append(state.sector)

    return " ".join(parts)