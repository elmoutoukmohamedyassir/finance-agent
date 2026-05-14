"""
agents/finance_agent.py — Main orchestrator for the finance agent workflow.

IMPROVEMENTS over the original:
  1. Fully conversational: progressively collects info, then analyzes
  2. Grounded analysis: LLM interprets pre-calculated metrics, never recalculates
  3. RAG context injected when relevant (not always — reduces noise)
  4. Session-based state: each conversation persists across messages
  5. Clear agent modes: gathering_info | analyzing | answering | off_topic

WORKFLOW:
  Message received
    → finance_guard: is it finance-related?
      → No: return polite rejection
      → Yes:
          → extract any business info from the message
          → update session state
          → do we have enough info to analyze?
            → No: ask next question (question_agent)
            → Yes: run full analysis
                → calculate metrics (pure math)
                → build scenarios (pure math)
                → retrieve RAG context (if relevant)
                → LLM interprets results (grounded prompt)
                → return structured response
"""

import logging

from app.core.groq_client import groq_client
from app.core.prompts import (
    build_metrics_prompt,
    build_rag_system_prompt,
    build_scenario_prompt,
    FINANCE_AGENT_SYSTEM_PROMPT,
)
from app.tools.finance_guard import is_finance_related, get_rejection_message
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
    """
    Main entry point for the chat endpoint.

    Takes a user message + optional session ID, returns a structured ChatResponse.
    All state management is handled here.
    """
    # Load or create conversation session
    session = get_or_create_session(session_id)

    # Step 1: Finance scope check
    # Only check on first message — subsequent messages are assumed in-context
    is_first_message = len(session.conversation_history) == 0
    if is_first_message and not is_finance_related(user_message):
        # Add to history so we have context
        session.add_message("user", user_message)
        session.add_message("assistant", get_rejection_message())
        save_session(session)
        return ChatResponse(
            session_id=session.session_id,
            message=get_rejection_message(),
            agent_mode="off_topic",
        )

    # Step 2: Extract any business data from the user's message
    extracted_info = extract_business_info(user_message)
    if extracted_info:
        logger.info(f"Extracted from message: {extracted_info}")
        update_business_state(session, extracted_info)

    # Step 3: Add user message to conversation history
    session.add_message("user", user_message)

    # Step 4: Decide what to do next
    state = session.business_state

    if not should_analyze(state):
        # Not enough info yet — ask the next question
        response = _ask_next_question(session)
    else:
        # We have enough data — run full analysis
        response = _run_full_analysis(session)

    # Step 5: Add response to history and persist
    session.add_message("assistant", response.message)
    save_session(session)

    # Attach current business state to response (for UI progress display)
    response.session_id = session.session_id
    response.business_state = {
        k: v for k, v in state.model_dump().items() if v is not None
    }

    return response


def _ask_next_question(session: ConversationSession) -> ChatResponse:
    """
    Determines and returns the next question to collect missing business info.
    """
    question = get_next_question(
        state=session.business_state,
        asked_questions=session.questions_asked,
    )

    if question:
        session.questions_asked.append(question)

        # For the very first message, prepend a welcome if we haven't yet
        is_first_response = len(session.conversation_history) == 1  # just the user msg
        if is_first_response:
            message = (
                "Welcome! I'm your AI SaaS Finance Advisor. I'll help you analyze "
                "the financial health and viability of your business.\n\n"
                f"Let's start: {question}"
            )
        else:
            message = question

        return ChatResponse(
            session_id=session.session_id,
            message=message,
            agent_mode="gathering_info",
        )
    else:
        # Somehow we got here without enough info and without a question
        # Fall back to general finance chat
        return _general_finance_answer(session)


def _run_full_analysis(session: ConversationSession) -> ChatResponse:
    """
    Runs the complete financial analysis workflow:
    1. Calculate metrics (pure math, no LLM)
    2. Build scenarios (pure math, no LLM)
    3. Retrieve RAG context (semantic search)
    4. LLM interprets pre-calculated results (grounded)
    """
    state = session.business_state

    # Step A: Calculate metrics — pure Python, no hallucinations
    metrics = calculate_from_business_state(state)
    metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

    # Step B: Build financial scenarios — pure Python
    scenarios = None
    if state.mrr and state.customer_count and state.monthly_costs is not None:
        scenarios_list = build_standard_scenarios(
            starting_customers=state.customer_count,
            monthly_price=state.arpu or (state.mrr / state.customer_count),
            monthly_costs=state.monthly_costs,
            starting_cash=50000,  # Default assumption if not provided
            months=12,
        )
        scenarios = format_scenarios_for_prompt(scenarios_list)

    # Step C: Retrieve RAG context for the query
    rag_query = _build_rag_query(state, metrics_dict)
    rag_context = retrieve_context(rag_query)

    # Step D: Build grounded prompt and call LLM
    business_context_dict = {k: v for k, v in state.model_dump().items() if v is not None}

    if rag_context:
        system_prompt = build_rag_system_prompt(rag_context)
    else:
        system_prompt = build_metrics_prompt(metrics_dict, business_context_dict)

    # Build the user message for the LLM with all calculated data pre-loaded
    analysis_request = _build_analysis_request(state, metrics_dict, scenarios)

    llm_response = groq_client.chat(
        system_prompt=system_prompt,
        user_message=analysis_request,
        conversation_history=session.conversation_history[-6:],  # recent context
        temperature=0.3,
        max_tokens=1500,
    )

    return ChatResponse(
        session_id=session.session_id,
        message=llm_response,
        agent_mode="analyzing",
        metrics_calculated=metrics_dict,
    )


def _general_finance_answer(session: ConversationSession) -> ChatResponse:
    """
    Falls back to a general finance Q&A mode when not running full analysis.
    Used for follow-up questions after analysis, or exploratory conversations.
    """
    rag_context = retrieve_context(session.conversation_history[-1]["content"])

    if rag_context:
        system_prompt = build_rag_system_prompt(rag_context)
    else:
        system_prompt = FINANCE_AGENT_SYSTEM_PROMPT

    response = groq_client.chat(
        system_prompt=system_prompt,
        user_message=session.conversation_history[-1]["content"],
        conversation_history=session.conversation_history[-8:],
        temperature=0.4,
        max_tokens=1000,
    )

    return ChatResponse(
        session_id=session.session_id,
        message=response,
        agent_mode="answering",
    )


def _build_rag_query(state, metrics_dict: dict) -> str:
    """
    Builds a targeted RAG search query from business context.
    More specific queries return more relevant chunks.
    """
    parts = ["SaaS finance metrics analysis"]

    if state.churn_rate and state.churn_rate > 5:
        parts.append("customer retention churn reduction strategies")
    if metrics_dict.get("ltv_cac_ratio") and metrics_dict["ltv_cac_ratio"] < 3:
        parts.append("LTV CAC ratio improvement customer acquisition cost")
    if metrics_dict.get("burn_rate"):
        parts.append("runway burn rate cash management startup")
    if state.business_model:
        parts.append(f"{state.business_model} SaaS business model")

    return " ".join(parts)


def _build_analysis_request(state, metrics_dict: dict, scenarios: str | None) -> str:
    """
    Builds the analysis request message to send to the LLM.
    This is NOT the system prompt — it's the "user turn" content
    that describes what analysis is needed.
    """
    lines = [
        f"Please analyze the financial health of this SaaS business: {state.business_name or 'unnamed'}",
        "",
        "CALCULATED METRICS (do not recalculate):",
    ]
    for k, v in metrics_dict.items():
        if k not in ("health_score", "warnings"):
            lines.append(f"  {k}: {v}")

    if metrics_dict.get("warnings"):
        lines.append("\nFLAGGED CONCERNS:")
        for w in metrics_dict["warnings"]:
            lines.append(f"  {w}")

    if scenarios:
        lines.append(f"\n12-MONTH SCENARIO PROJECTIONS:\n{scenarios}")

    lines.append(
        "\nPlease provide:\n"
        "1. A clear financial health summary\n"
        "2. The 2-3 most important things to fix or focus on\n"
        "3. One concrete next step recommendation"
    )

    return "\n".join(lines)
