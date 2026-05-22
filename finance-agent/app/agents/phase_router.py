"""
Phase Router - Orchestrates multi-phase system.

Determines which phase the user is in and routes to the correct agent.
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
        """
        business_state = session_context.get("business_state", {})
        
        # Phase 4: Post-creation (if business is created)
        if business_state.get("is_created"):
            return "phase4"
        
        # Phase 3: Ready for analysis (has minimum data)
        minimum_fields = {
            "entity_type", "segment_client", "prix_vente_unitaire",
            "nb_clients_mois1", "taux_croissance_mensuel",
            "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
        }
        if minimum_fields.issubset(set(business_state.keys())):
            return "phase3"
        
        # Phase 2: In data collection
        if session_context.get("in_collection") or business_state:
            return "phase2"
        
        # Phase 1: Ideation (default)
        return "phase1"

    def route_message(self, message: AgentMessage, session_context: dict) -> AgentResponse:
        """
        Route message to appropriate phase agent.
        """
        current_phase = self.get_current_phase(session_context)
        
        try:
            if current_phase == "phase1":
                # Check if user wants to move to Phase 2
                response = self.phase1.process(message)
                if response.structured_output and response.structured_output.get("ready_for_phase2"):
                    # Transition to Phase 2
                    message.context["in_collection"] = True
                    session_context["phase"] = "phase2"
                    response.message += "\n\nProchaine étape: Je vais poser des questions spécifiques pour affiner l'analyse."
                return response

            elif current_phase == "phase2":
                return self.phase2.process(message)

            elif current_phase == "phase3":
                return self.phase3.process(message)

            elif current_phase == "phase4":
                # Phase 4: Ongoing support (future)
                return AgentResponse(
                    agent_id="phase4_ongoing",
                    session_id=message.session_id,
                    intent=message.intent,
                    message="Vous êtes en phase post-création. Comment puis-je vous aider ?",
                    agent_mode="ongoing",
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
