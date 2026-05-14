"""
agents/question_agent.py — Intelligently asks follow-up questions to gather
missing business information from the founder.

WHY a separate agent for this:
  Single Responsibility Principle. This agent has one job:
  given what we know, decide what to ask next.

  It does NOT analyze or advise — that's finance_agent.py's job.
  Clean separation makes both easier to debug and improve.

HOW IT WORKS:
  1. Inspect the current BusinessState to find missing fields
  2. Use a priority order (most impactful fields first)
  3. Extract any numbers/facts the user mentioned in their latest message
  4. Update the state with newly discovered info
  5. Return the next question to ask, or signal "ready for analysis"

EXTRACTION APPROACH:
  We use the LLM to extract structured data from natural language.
  E.g. "I charge $50/month to about 30 customers" → mrr=1500, customer_count=30, arpu=50
  
  This is far more robust than regex and handles varied phrasing naturally.
"""

import json
import logging
import re
from typing import Optional

from app.core.groq_client import groq_client
from app.schemas.session import BusinessState, ConversationSession
from app.core.prompts import QUESTION_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Priority order for collecting fields — most important for analysis first
FIELD_PRIORITY = [
    "business_name",
    "target_audience",
    "business_model",
    "customer_count",
    "arpu",          # if no MRR, get ARPU + count
    "mrr",
    "monthly_costs",
    "churn_rate",
    "marketing_budget",
    "cac",
    "funding_stage",
    "growth_rate",
]

FIELD_QUESTIONS = {
    "business_name": (
        "To get started, what's your SaaS product called or what does it do? "
        "Give me a quick one-sentence description."
    ),
    "target_audience": (
        "Who are your target customers? "
        "(e.g., 'small marketing agencies', 'solo developers', 'e-commerce stores')"
    ),
    "business_model": (
        "Is your SaaS targeting businesses (B2B), consumers (B2C), "
        "or both (B2B2C)?"
    ),
    "customer_count": (
        "How many paying customers do you have right now, "
        "or expect to have in your first month of launch?"
    ),
    "arpu": (
        "What's your monthly price per customer? "
        "(If you have multiple tiers, what's the average or most common price?)"
    ),
    "mrr": (
        "What's your current Monthly Recurring Revenue (MRR)? "
        "This is simply: number of customers × monthly price."
    ),
    "monthly_costs": (
        "What are your total monthly operating costs? "
        "Include hosting, salaries, tools, and any other regular expenses."
    ),
    "churn_rate": (
        "What percentage of your customers cancel each month? "
        "If you're pre-launch, what do you estimate? "
        "(Industry average for SMB SaaS is 3–5%)"
    ),
    "marketing_budget": (
        "How much do you spend on marketing and customer acquisition each month?"
    ),
    "cac": (
        "Do you know your Customer Acquisition Cost (CAC)? "
        "That's how much you spend on average to acquire one paying customer."
    ),
    "funding_stage": (
        "What's your current funding situation? "
        "(Bootstrapped, pre-seed, seed, Series A, etc.)"
    ),
    "growth_rate": (
        "What monthly growth rate are you targeting or currently seeing? "
        "(e.g., '10% month-over-month')"
    ),
}

EXTRACTION_SYSTEM_PROMPT = """You are a data extraction assistant. 
Extract financial and business information from the user's message.

Return ONLY a valid JSON object with these possible fields (include only fields you can extract with confidence):
{
  "business_name": "string or null",
  "target_audience": "string or null",
  "business_model": "B2B or B2C or B2B2C or null",
  "customer_count": number or null,
  "arpu": number or null,
  "mrr": number or null,
  "monthly_costs": number or null,
  "churn_rate": number or null,
  "marketing_budget": number or null,
  "cac": number or null,
  "funding_stage": "string or null",
  "growth_rate": number or null,
  "new_customers_per_month": number or null
}

Rules:
- Extract ONLY what is clearly stated. Do not guess or infer.
- All monetary values should be in dollars (monthly amounts).
- churn_rate and growth_rate should be percentages (e.g. 5 for 5%).
- If the user says "50 euros" convert to dollar equivalent only if obvious, otherwise use as-is.
- If nothing relevant is in the message, return an empty JSON object: {}
- Return ONLY the JSON. No explanation, no markdown.
"""


def extract_business_info(message: str) -> dict:
    """
    Uses the LLM to extract structured business data from a natural language message.
    
    Returns a dict of field_name → value for any fields found.
    Returns empty dict if nothing extractable found.
    """
    try:
        response = groq_client.chat(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_message=message,
            temperature=0.0,
            max_tokens=300,
        )
        # Strip any accidental markdown fences
        clean = response.strip().strip("```json").strip("```").strip()
        extracted = json.loads(clean)
        # Filter out nulls
        return {k: v for k, v in extracted.items() if v is not None}
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"Extraction failed: {e}. Raw response: {response[:100] if 'response' in dir() else 'N/A'}")
        return {}


def get_next_question(state: BusinessState, asked_questions: list[str]) -> Optional[str]:
    """
    Returns the next question to ask, or None if we have enough info.

    Args:
        state: Current collected business information.
        asked_questions: Questions already asked this session (to avoid repeating).

    Returns:
        Question string, or None if ready for analysis.
    """
    if state.is_ready_for_analysis():
        return None

    # Determine which fields are missing, in priority order
    for field in FIELD_PRIORITY:
        value = getattr(state, field, None)
        if value is None:
            question = FIELD_QUESTIONS.get(field)
            if question and question not in asked_questions:
                return question

            # Skip if already asked (avoid repetition)
            # Try next priority field instead
            continue

    # If we've asked everything in priority list, check if state is workable
    if state.mrr or (state.customer_count and state.arpu):
        return None  # Enough to analyze

    # Fallback: ask for MRR directly
    fallback = "Could you tell me your approximate Monthly Recurring Revenue (MRR)?"
    if fallback not in asked_questions:
        return fallback

    return None


def should_analyze(state: BusinessState) -> bool:
    """
    Determines if we have enough information to run financial analysis.
    Mirrors BusinessState.is_ready_for_analysis() but can add extra conditions.
    """
    return state.is_ready_for_analysis()
