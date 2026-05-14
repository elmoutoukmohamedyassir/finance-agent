"""
tools/finance_guard.py — Finance topic validator.

WHY: The agent is specialized in finance/SaaS. If users ask off-topic
questions (recipes, code, general knowledge), we reject them politely.

This runs BEFORE any expensive LLM calls, acting as a cheap filter.

APPROACH: We use two strategies:
  1. Keyword matching (fast, zero API cost) for obvious cases
  2. LLM classification (slow, costs API call) for ambiguous cases

The keyword approach catches 80%+ of cases without any API call.
"""

import re
import logging
from app.core.groq_client import groq_client
from app.core.prompts import FINANCE_GUARD_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Keywords that strongly indicate finance/business intent
FINANCE_KEYWORDS = {
    # Metrics
    "mrr", "arr", "ltv", "cac", "churn", "arpu", "runway", "burn rate",
    "revenue", "profit", "loss", "margin", "roi", "ebitda",
    # Business terms
    "saas", "startup", "business", "customer", "pricing", "subscription",
    "investor", "funding", "valuation", "growth", "retention", "acquisition",
    # Financial actions
    "calculate", "analyze", "project", "forecast", "simulate", "evaluate",
    "financial", "finance", "money", "cost", "budget", "expense", "income",
    # SaaS-specific
    "monthly recurring", "annual recurring", "lifetime value", "payback period",
    "unit economics", "net revenue retention", "gross margin",
}

# Keywords that clearly indicate off-topic content
OFF_TOPIC_KEYWORDS = {
    "recipe", "cook", "food", "movie", "song", "music", "sport", "game",
    "weather", "translate", "poem", "story", "joke", "travel", "health",
    "medical", "relationship", "love", "politics", "religion",
}

POLITE_REJECTION = (
    "I'm specialized in SaaS finance and business analysis. "
    "That question seems outside my area of expertise. "
    "I'd be happy to help with topics like:\n"
    "• SaaS financial metrics (MRR, ARR, LTV, CAC, churn)\n"
    "• Business viability analysis\n"
    "• Financial projections and scenarios\n"
    "• Pricing strategy\n"
    "• Unit economics\n\n"
    "Do you have a SaaS business or idea you'd like to evaluate financially?"
)


def is_finance_related(message: str) -> bool:
    """
    Determines if a message is finance/business related.

    Strategy:
    1. Check for obvious off-topic keywords → reject immediately (no API cost)
    2. Check for obvious finance keywords → accept immediately (no API cost)
    3. For ambiguous cases → use LLM classifier

    Returns True if the message is finance/business related.
    """
    message_lower = message.lower()

    # Step 1: Check obvious off-topic
    for keyword in OFF_TOPIC_KEYWORDS:
        if keyword in message_lower:
            # But make sure it's not "business recipe" or "financial health"
            if not any(fk in message_lower for fk in FINANCE_KEYWORDS):
                logger.info(f"Finance guard: rejected (off-topic keyword: '{keyword}')")
                return False

    # Step 2: Check obvious finance keywords
    for keyword in FINANCE_KEYWORDS:
        if keyword in message_lower:
            logger.info(f"Finance guard: accepted (finance keyword: '{keyword}')")
            return True

    # Step 3: Short greetings / contextual messages → assume finance context
    # (e.g., "yes", "ok", "tell me more" are continuation messages)
    if len(message.split()) <= 5:
        logger.info("Finance guard: accepted (short message, assumed continuation)")
        return True

    # Step 4: LLM classification for ambiguous cases
    return _llm_classify(message)


def _llm_classify(message: str) -> bool:
    """
    Uses the LLM to classify ambiguous messages.
    Returns True if finance-related, False otherwise.
    """
    try:
        response = groq_client.chat(
            system_prompt=FINANCE_GUARD_SYSTEM_PROMPT,
            user_message=message,
            temperature=0.0,  # 0 = most deterministic for classification
            max_tokens=5,
        )
        result = response.strip().lower()
        is_finance = result == "yes"
        logger.info(f"Finance guard LLM classification: '{result}' → {is_finance}")
        return is_finance
    except Exception as e:
        # If classification fails, default to accepting the message
        # Better to allow a borderline message than to reject a valid finance question
        logger.warning(f"Finance guard LLM failed: {e}. Defaulting to accept.")
        return True


def get_rejection_message() -> str:
    """Returns the polite rejection message."""
    return POLITE_REJECTION
