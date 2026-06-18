"""
core/gemini_client.py — Thin wrapper around Google Gemini API.

Used as fallback when RAG confidence is low or when natural language
reasoning is needed. Implements same interface as groq_client.
"""

import logging
from functools import lru_cache
from typing import Optional

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class GeminiClient:
    """Google Gemini fallback LLM client."""

    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            logger.warning("GEMINI_API_KEY not set. Gemini fallback unavailable.")
            self._client = None
            self.model = settings.gemini_model
            return

        genai.configure(api_key=settings.gemini_api_key)
        self._client = genai.GenerativeModel(model_name=settings.gemini_model)
        self.model = settings.gemini_model
        logger.info(f"Gemini client initialized with model: {self.model}")

    def is_available(self) -> bool:
        """Check if Gemini client is configured."""
        return self._client is not None

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        temperature: float = 0.2,
        max_tokens: int = 1500,
    ) -> str:
        if not self._client:
            raise RuntimeError("Gemini client not configured. Set GEMINI_API_KEY.")

        messages = self._build_messages(system_prompt, user_message, conversation_history)

        try:
            response = self._client.generate_content(
                contents=messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                safety_settings=[
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            )

            if not response.text:
                logger.warning("Gemini returned empty response")
                return ""

            return response.text

        except GoogleAPIError as e:
            logger.error(f"Gemini API error: {e}")
            raise RuntimeError(f"Gemini service error: {str(e)}")
        except Exception as e:
            logger.error(f"Gemini unexpected error: {e}")
            raise RuntimeError(f"Gemini service error: {str(e)}")

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]],
    ) -> list:
        messages = []

        if system_prompt:
            messages.append({"role": "user", "content": system_prompt})
            messages.append({
                "role": "model",
                "content": "I understand. I am an expert financial advisor for entrepreneurs in Morocco. I will provide concise, accurate, and culturally appropriate financial guidance."
            })

        if conversation_history:
            for msg in conversation_history[-10:]:
                role = "user" if msg.get("role") == "user" else "model"
                messages.append({"role": role, "content": msg.get("content", "")})

        messages.append({"role": "user", "content": user_message})
        return messages


@lru_cache(maxsize=1)
def _get_gemini_client() -> "GeminiClient":
    """Lazy singleton — created on first use, not at import time."""
    return GeminiClient()


class _LazyGeminiProxy:
    """Proxy that defers GeminiClient instantiation until first call."""

    def __getattr__(self, name):
        return getattr(_get_gemini_client(), name)


gemini_client = _LazyGeminiProxy()