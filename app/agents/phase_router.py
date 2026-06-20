"""
Phase Router - Orchestrates multi-phase system.
"""

from app.agents.phase1_ideation_agent import Phase1IdeationAgent
from app.agents.phase2_collection_agent import Phase2DataCollectionAgent
from app.agents.phase3_analysis_agent import Phase3AnalysisAgent
from app.agents.base_agent import AgentMessage, AgentResponse
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PhaseRouter:
    """Routes messages to the appropriate phase agent."""

    def __init__(self):
        self.phase1 = Phase1IdeationAgent()
        self.phase2 = Phase2DataCollectionAgent()
        self.phase3 = Phase3AnalysisAgent()

    def get_current_phase(self, session_context: dict) -> str:
        """
        Determine current phase based on session context.
        Returns: "phase1" | "phase2" | "phase3" | "phase4"

        Priority:
          1. business_state.is_created -> phase4 (post-creation), always wins.
          2. Enough fields already collected -> phase3, even if we haven't
             been told to leave phase2/phase1 yet (lets a user who blurts out
             every number at once skip straight to analysis).
          3. Otherwise, stick to whatever `router_phase` was persisted on the
             session last turn (set by the previous agent's `next_phase`).
          4. Default: phase1.
        """
        business_state = session_context.get("business_state", {})

        if business_state.get("is_created"):
            return "phase4"

        minimum_fields = {
            "entity_type", "segment_client", "prix_vente_unitaire",
            "nb_clients_mois1", "taux_croissance_mensuel",
            "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
        }
        filled_fields = {k for k, v in business_state.items() if v is not None and v != ""}
        if minimum_fields.issubset(filled_fields):
            return "phase3"

        router_phase = session_context.get("router_phase", "phase1")
        if router_phase in ("phase2", "phase3", "phase4"):
            # Not enough fields for phase3 (checked above), but we've already
            # left phase1 — keep collecting rather than restarting ideation.
            return "phase2"

        return "phase1"

    def route_message(self, message: AgentMessage, session_context: dict) -> AgentResponse:
        """
        Route message to appropriate phase agent.
        """
        current_phase = self.get_current_phase(session_context)

        try:
            if current_phase == "phase1":
                response = self.phase1.process(message)
                if response.structured_output and response.structured_output.get("next_phase") == "phase2":
                    response.message += (
                        "\n\nProchaine étape : je vais vous poser quelques questions chiffrées "
                        "pour préparer une vraie analyse financière."
                    )
                return response

            elif current_phase == "phase2":
                return self.phase2.process(message)

            elif current_phase == "phase3":
                return self.phase3.process(message)

            elif current_phase == "phase4":
                return AgentResponse(
                    agent_id="phase4_ongoing",
                    session_id=message.session_id,
                    intent=message.intent,
                    message="Vous êtes en phase post-création. Comment puis-je vous aider ?",
                    agent_mode="ongoing",
                    structured_output={"next_phase": "phase4"},
                )

        except Exception as e:
            logger.error(f"Router error: {e}")
            return AgentResponse(
                agent_id="phase_router",
                session_id=message.session_id,
                intent=message.intent,
                success=False,
                error=str(e),
                message="Une erreur est survenue. Veuillez réessayer.",
            )

        return AgentResponse(
            agent_id="phase_router",
            session_id=message.session_id,
            intent=message.intent,
            message="Phase not recognized",
            success=False,
        )