"""
Phase 2: Data Collection Agent - Structured Q&A to gather financial data.
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
        try:
            if not message.session_id:
                return self._make_error_response(message, "No session_id provided")

            # Load Q&A tracking from session context (persisted via SQLite)
            business_state = message.context.get("business_state", {})
            asked_questions = list(message.context.get("asked_questions", []))
            pending_question = message.context.get("pending_question")

            # User is answering a question — extract data
            if message.user_message and pending_question:
                extracted, validation_error = extract_and_validate(
                    user_message=message.user_message,
                    pending_field=pending_question,
                    conversation_history=[],
                )

                if validation_error:
                    return AgentResponse(
                        agent_id=self.agent_id,
                        session_id=message.session_id,
                        intent=message.intent,
                        message=f"Je n'ai pas bien compris. {validation_error}\n\nPouvez-vous réessayer ?",
                        agent_mode="collecting",
                        business_state=business_state,
                        structured_output={
                            "business_state": business_state,
                            "asked_questions": asked_questions,
                            "pending_question": pending_question,
                            "next_phase": "phase2",
                        },
                    )

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
                    business_state=business_state,
                    structured_output={
                        "business_state": business_state,
                        "ready_for_phase3": True,
                        "next_phase": "phase3",
                    },
                )

            # Ask next question
            next_field, next_question = get_next_question(
                state=business_state,
                asked_questions=asked_questions,
                phase="pre_creation",
            )

            if not next_field:
                return AgentResponse(
                    agent_id=self.agent_id,
                    session_id=message.session_id,
                    intent=message.intent,
                    message="Je vais maintenant préparer votre analyse financière personnalisée...",
                    agent_mode="ready_for_analysis",
                    business_state=business_state,
                    structured_output={
                        "business_state": business_state,
                        "ready_for_phase3": True,
                        "next_phase": "phase3",
                    },
                )

            asked_questions.append(next_field)
            context_text = FIELD_CONTEXT_FR.get(next_field, "")
            question_full = f"{next_question}\n\n_({context_text})_"

            return AgentResponse(
                agent_id=self.agent_id,
                session_id=message.session_id,
                intent=message.intent,
                message=question_full,
                agent_mode="collecting",
                business_state=business_state,
                structured_output={
                    "business_state": business_state,
                    "asked_questions": asked_questions,
                    "pending_question": next_field,
                    "next_phase": "phase2",
                },
            )

        except Exception as e:
            logger.error(f"Phase2 agent error: {e}")
            return self._make_error_response(message, str(e))

    def can_handle(self, intent: str) -> bool:
        return intent in ("collect_data", "answer_question", "continue_collection")

    def _is_minimum_data_collected(self, business_state: dict) -> bool:
        minimum_fields = {
            "entity_type", "segment_client", "prix_vente_unitaire",
            "nb_clients_mois1", "taux_croissance_mensuel",
            "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
        }
        # business_state usually comes from BusinessState.model_dump(), which
        # always includes every field name (most set to None). Checking
        # .keys() alone would make this always-True from the very first turn.
        filled = {k for k, v in business_state.items() if v is not None and v != ""}
        return minimum_fields.issubset(filled)