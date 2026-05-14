"""
agents/question_agent.py — Collects missing business info conversationally.

Two responsibilities:
  1. Extract structured data from natural language (using LLM)
  2. Decide what to ask next (using priority list)
"""

import json
import logging
from typing import Optional

from app.core.groq_client import groq_client
from app.core.prompts import EXTRACTION_SYSTEM_PROMPT
from app.schemas.session import BusinessState

logger = logging.getLogger(__name__)

# Priority order: most impactful for financial analysis first
FIELD_PRIORITY = [
    "business_name",
    "target_audience",
    "customer_count",
    "arpu",
    "mrr",
    "monthly_costs",
    "churn_rate",
    "marketing_budget",
    "business_model",
    "funding_stage",
]

FIELD_QUESTIONS = {
    "business_name": (
        "What's your SaaS product called, and what does it do in one sentence?"
    ),
    "target_audience": (
        "Who are your target customers? "
        "(e.g. 'small marketing agencies', 'solo developers', 'e-commerce stores')"
    ),
    "customer_count": (
        "How many paying customers do you have right now "
        "(or expect at launch)?"
    ),
    "arpu": (
        "What's your monthly price per customer? "
        "If you have multiple plans, what's the most common one?"
    ),
    "mrr": (
        "What's your current Monthly Recurring Revenue — "
        "that's simply your number of customers × monthly price."
    ),
    "monthly_costs": (
        "What are your total monthly operating costs? "
        "Include hosting, salaries, tools, and any subscriptions."
    ),
    "churn_rate": (
        "What percentage of customers cancel per month? "
        "If pre-launch, estimate — industry average for SMB SaaS is 3–5%."
    ),
    "marketing_budget": (
        "How much do you spend on marketing and customer acquisition each month?"
    ),
    "business_model": (
        "Is your SaaS B2B (selling to businesses), B2C (consumers), or B2B2C?"
    ),
    "funding_stage": (
        "What's your funding situation? "
        "(bootstrapped, pre-seed, seed, Series A, etc.)"
    ),
}


def extract_business_info(message: str) -> dict:
    """
    Uses the LLM to extract structured financial data from natural language.
    
    E.g. "we charge $99/month and have 45 clients" → {arpu: 99, customer_count: 45}
    Returns {} if nothing extractable.
    """
    try:
        response = groq_client.chat(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_message=message,
            temperature=0.0,
            max_tokens=300,
        )
        clean = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        result = {k: v for k, v in parsed.items() if v is not None}
        if result:
            logger.info(f"Extracted: {result}")
        return result
    except Exception as e:
        logger.warning(f"Extraction parse failed: {e}")
        return {}


def get_next_question(state: BusinessState, asked_questions: list[str]) -> Optional[str]:
    """
    Returns the next question to ask, or None if enough info collected.
    Never repeats a question already asked this session.
    """
    if state.is_ready_for_analysis():
        return None

    for field in FIELD_PRIORITY:
        value = getattr(state, field, None)
        if value is None:
            question = FIELD_QUESTIONS.get(field)
            if question and question not in asked_questions:
                return question
            # Already asked this question, skip to next field

    # If all priority fields asked, check if we can proceed anyway
    if state.has_revenue_info() and state.has_cost_info():
        return None

    # Last resort fallback
    fallback = "Could you share your approximate monthly revenue and monthly costs?"
    return fallback if fallback not in asked_questions else None


def should_analyze(state: BusinessState) -> bool:
    return state.is_ready_for_analysis()
