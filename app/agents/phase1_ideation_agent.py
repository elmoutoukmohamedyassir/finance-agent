"""
Phase 1: Ideation Agent - ChatGPT-style conversation about business ideas.
"""

from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from app.core.groq_client import groq_client
import logging

logger = logging.getLogger(__name__)


PHASE1_SYSTEM_PROMPT = """Tu es un conseiller financier IA spécialisé dans l'accompagnement d'entrepreneurs au Maroc.

L'utilisateur a une idée de business et veut discuter de sa viabilité financière.

Sois:
- Naturel et conversationnel (comme ChatGPT)
- Bienveillant et encourageant
- Pragmatique sur la faisabilité financière
- Focus sur la finance (pas de conseils juridiques/marketing sauf si demandé)

NE POSE PAS de questions structurées. Laisse la conversation se dérouler naturellement.
Si l'utilisateur mentionne des chiffres (revenue, coûts, etc.), note-les mentalement.
Après 2-3 échanges, propose de passer à une analyse financière détaillée si intéressé.

Contexte Maroc:
- TVA: 20% standard, 7% et 10% secteurs spécifiques
- CNSS: Cotisation obligatoire ~8% salaires
- IS (Impôt Société): 30% + additionnels (prélevée à la source)
- Auto-entrepreneur: Forfait simplifié possible jusqu'à 500k MAD CA
- SARL minimum: 1000 MAD capital

Réponds TOUJOURS en Français sauf si l'utilisateur demande l'Anglais.
Sois bref (2-4 lignes max par réponse, sauf si question complexe demande plus).
"""


class Phase1IdeationAgent(BaseAgent):
    agent_id = "phase1_ideation"
    agent_version = "1.0.0"
    description = "Chat agent for business ideation and feasibility discussion"

    def process(self, message: AgentMessage) -> AgentResponse:
        try:
            if not message.user_message:
                return AgentResponse(
                    agent_id=self.agent_id,
                    session_id=message.session_id,
                    intent=message.intent,
                    success=False,
                    error="No user message provided",
                )

            # Call Groq with conversation history so the AI remembers context
            conversation_history = message.context.get("conversation_history", [])
            llm_response = groq_client.chat(
                system_prompt=PHASE1_SYSTEM_PROMPT,
                user_message=message.user_message,
                conversation_history=conversation_history,
                temperature=0.7,
                max_tokens=400,
            )

            should_advance = self._check_readiness_for_phase2(message.user_message, conversation_history)

            return AgentResponse(
                agent_id=self.agent_id,
                session_id=message.session_id,
                intent=message.intent,
                message=llm_response,
                agent_mode="ideation",
                success=True,
                structured_output={
                    "ready_for_phase2": should_advance,
                    # "next_phase" is what PhaseRouter/chat.py persist as
                    # session.router_phase. It is a free-form string, never
                    # written into the strict `Phase` enum field.
                    "next_phase": "phase2" if should_advance else "phase1",
                },
            )

        except Exception as e:
            logger.error(f"Phase1 agent error: {e}")
            return self._make_error_response(message, str(e))

    def can_handle(self, intent: str) -> bool:
        return intent in ("ideation", "idea_validation", "chat")

    def _check_readiness_for_phase2(self, user_message: str, conversation_history: list) -> bool:
        lower = user_message.lower()
        keywords = [
            "analyser", "analyze", "calculer", "calculate", "chiffre",
            "combien", "how much", "coût", "cost", "revenue", "chiffre d'affaires",
            "équipe", "team", "investissement", "investment", "plan financier",
            "projection", "prévision", "détail", "detailed", "business plan",
            "étapes", "steps", "quoi faire", "what to do", "suivant", "next",
        ]
        if any(kw in lower for kw in keywords):
            return True

        # Safety net: the system prompt promises the LLM will suggest moving
        # on "après 2-3 échanges" — but that suggestion was only ever text,
        # it never actually drove routing. A real conversation that never
        # happens to use one of the keywords above would stay in phase1
        # forever. Count user turns so it advances regardless after a few
        # exchanges, matching what the prompt already tells the user to expect.
        user_turns = sum(1 for m in conversation_history if m.get("role") == "user")
        return user_turns >= 3