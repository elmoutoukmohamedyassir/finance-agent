"""
groq_client.py — Clean, reusable wrapper around the Groq API.

WHY: Instead of scattering raw Groq API calls everywhere, we centralize:
  - Authentication
  - Error handling
  - Retry logic
  - Response parsing
  - Logging

This means if Groq changes their API, or we want to swap to another
provider (OpenAI, Anthropic, etc.), we only change this one file.

IMPORTANT: The LLM is ONLY used for reasoning and explanation.
All financial calculations happen in metrics_calculator.py with pure Python math.
This is the key anti-hallucination strategy.
"""

import logging
from typing import Optional
from groq import Groq, APIError, APIConnectionError, RateLimitError

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GroqClient:
    """
    Thin wrapper around the Groq SDK.
    
    Usage:
        client = GroqClient()
        response = client.chat(
            system_prompt="You are a finance advisor.",
            user_message="What is LTV?",
            conversation_history=[...]  # optional
        )
    """

    def __init__(self):
        self._client = Groq(api_key=settings.groq_api_key)
        self.model = settings.groq_model

    def chat(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> str:
        """
        Send a chat request to Groq and return the response text.

        Args:
            system_prompt: The system-level instructions for the LLM.
            user_message: The current user message.
            conversation_history: List of {"role": ..., "content": ...} dicts
                                   for multi-turn memory. Optional.
            temperature: Lower = more deterministic/factual. Keep at 0.3
                         for finance to reduce hallucinations.
            max_tokens: Max response length.

        Returns:
            The assistant's response as a plain string.

        Raises:
            RuntimeError: If the API call fails after handling known errors.
        """
        messages = self._build_messages(system_prompt, user_message, conversation_history)

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            logger.debug(f"Groq response: {content[:100]}...")
            return content

        except RateLimitError:
            logger.warning("Groq rate limit hit — consider adding retry logic")
            raise RuntimeError(
                "The AI service is currently busy. Please try again in a moment."
            )
        except APIConnectionError:
            logger.error("Could not connect to Groq API")
            raise RuntimeError(
                "Could not connect to the AI service. Check your internet connection."
            )
        except APIError as e:
            logger.error(f"Groq API error: {e}")
            raise RuntimeError(f"AI service error: {str(e)}")

    def _build_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: Optional[list[dict]],
    ) -> list[dict]:
        """
        Assemble the messages array Groq expects.
        
        Structure:
          [system] → [history turn 1] → [history turn 2] → ... → [current user msg]
        
        We cap history at last 10 messages to stay within context limits.
        """
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            # Only keep recent history to avoid context overflow
            recent_history = conversation_history[-10:]
            messages.extend(recent_history)

        messages.append({"role": "user", "content": user_message})
        return messages


# Module-level singleton — import this instead of instantiating everywhere
groq_client = GroqClient()
