"""
groq_client.py — Thin wrapper around the Groq SDK.

All LLM calls go through here. If we ever switch provider,
only this file changes.
"""

import logging
from functools import lru_cache
from typing import Optional

from groq import Groq, APIError, APIConnectionError, RateLimitError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GroqClient:

    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. "
                "Copy .env.example → .env and add your key from https://console.groq.com"
            )
        self._client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        """
        Send a request to Groq and return the response text.

        temperature=0.2 is intentionally low for finance — we want precise,
        consistent answers, not creative ones.
        """
        messages = self._build_messages(system_prompt, user_message, conversation_history)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content

        except RateLimitError:
            raise RuntimeError("The AI service is busy. Please try again in a moment.")
        except APIConnectionError:
            raise RuntimeError("Cannot connect to AI service. Check your internet connection.")
        except APIError as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"AI service error: {str(e)}")

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]],
    ) -> list[dict]:
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            messages.extend(conversation_history[-10:])  # last 10 turns max
        messages.append({"role": "user", "content": user_message})
        return messages


@lru_cache(maxsize=1)
def _get_groq_client() -> "GroqClient":
    """Lazy singleton — created on first use, not at import time."""
    return GroqClient()


class _LazyGroqProxy:
    """Proxy that defers GroqClient instantiation until first call.
    Keeps the old ``groq_client.chat(...)`` call-sites working unchanged."""

    def __getattr__(self, name):
        return getattr(_get_groq_client(), name)


groq_client = _LazyGroqProxy()