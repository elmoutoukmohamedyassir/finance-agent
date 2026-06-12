"""
Phase 2: Data Collection Agent - Structured Q&A to gather financial data.

When user is ready, this agent asks specific questions to gather:
- Business basics (name, sector, structure)
- Financial data (revenue model, costs, team)
- Investment and capital information

This feeds into Phase 3 analysis.
"""

from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from app.agents.question_agent import (
    get_next_question, extract_and_validate, is_finance_question,
    FIELD_QUESTIONS_FR, FIELD_CONTEXT_FR, PRE_CREATION_MINIMUM
)
from app.core.groq_client import groq_client
import logging

logger = logging.getLogger(__name__)


class Phase2DataCollectionAgent(BaseAgent):
    agent_id = "phase2_collection"
    agent_version = "1.0.0"
    description = "Structured data collection for financial analysis"

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Guide structured data collection.
        Maintains business_state in session and asks next question.
        """
        try:
            if not message.session_id:
                return self._make_error_response(message, "No session_id provided")

            # Get or initialize business state from context
            business_state = message.context.get("business_state", {})
            asked_questions = message.context.get("asked_questions", [])
            pending_question = message.context.get("pending_question")

            # User is answering a question — extract data
            if message.user_message and pending_question:
                extracted, validation_error = extract_and_validate(
                    user_message=message.user_message,
                    pending_field=pending_question,
                    conversation_history=[],
                )

                if validation_error:
                    # Reprompt for this field
                    return AgentResponse(
                        agent_id=self.agent_id,
                        session_id=message.session_id,
                        intent=message.intent,
                        message=f"Je n'ai pas bien compris. {validation_error}\n\nPouvez-vous réessayer ?",
                        agent_mode="collecting",
                        structured_output={
                            "business_state": business_state,
                            "asked_questions": asked_questions,
                            "pending_question": pending_question,
                        },
                    )

                # Data extracted successfully — update state
                if extracted:
                    business_state.update(extracted)

            # Check if ready for Phase 3
            if self._is_minimum_data_collected(business_state):
                return AgentResponse(
                    agent_id=self.agent_id,
                    session_id=message.session_id,
                    intent=message.intent,
                    message=(
                        "Parfait ! J'ai suffisamment d'informations pour faire une analyse financière détaillée. "
                        "Passons à la phase d'analyse..."
                    ),
                    agent_mode="ready_for_analysis",
                    structured_output={
                        "business_state": business_state,
                        "ready_for_phase3": True,
                        "phase": "collection",
                    },
                )

            # Ask next question
            next_field, next_question = get_next_question(
                state=business_state,
                asked_questions=asked_questions,
                phase="pre_creation",
            )

            if not next_field:
                # No more questions — ready for analysis
                return AgentResponse(
                    agent_id=self.agent_id,
                    session_id=message.session_id,
                    intent=message.intent,
                    message="Je vais maintenant préparer votre analyse financière personnalisée...",
                    agent_mode="ready_for_analysis",
                    structured_output={
                        "business_state": business_state,
                        "ready_for_phase3": True,
                        "phase": "collection",
                    },
                )

            # Ask the next question with context
            asked_questions.append(next_field)
            context_text = FIELD_CONTEXT_FR.get(next_field, "")
            question_full = f"{next_question}\n\n_({context_text})_"

            return AgentResponse(
                agent_id=self.agent_id,
                session_id=message.session_id,
                intent=message.intent,
                message=question_full,
                agent_mode="collecting",
                structured_output={
                    "business_state": business_state,
                    "asked_questions": asked_questions,
                    "pending_question": next_field,
                    "phase": "collection",
                },
            )

        except Exception as e:
            logger.error(f"Phase2 agent error: {e}")
            return self._make_error_response(message, str(e))

    def can_handle(self, intent: str) -> bool:
        """Handle data collection and questions."""
        return intent in ("collect_data", "answer_question", "continue_collection")

    def _is_minimum_data_collected(self, business_state: dict) -> bool:
        """Check if minimum required fields are filled."""
        minimum_fields = {
            "entity_type", "segment_client", "prix_vente_unitaire",
            "nb_clients_mois1", "taux_croissance_mensuel",
            "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
        }
        filled = set(business_state.keys())
        return minimum_fields.issubset(filled)
