"""
Phase 3: Financial Analysis Agent - KPI calculations, scenarios, business plan generation.

Uses collected data to:
- Calculate financial KPIs (break-even, profitability, cash flow, etc.)
- Provide Morocco tax guidance
- Create financial projections
- Generate business plan
- Answer finance-related questions
"""

from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from app.core.groq_client import groq_client
import logging

logger = logging.getLogger(__name__)


PHASE3_ANALYSIS_SYSTEM_PROMPT = """Tu es un analyste financier spécialisé dans les projections pour entrepreneurs au Maroc.

Tu reçois des données financières (CA, coûts, investissement, etc.) et tu dois:

1. CALCULER les KPIs importants:
   - Seuil de rentabilité (break-even en unités/CA)
   - Marge brute unitaire et globale
   - Ratio de profitabilité
   - Flux de trésorerie mensuel
   - Besoin en fonds de roulement (BFR)
   - ROI et payback period
   - Taux d'endettement

2. EXPLIQUER chaque KPI en termes simples (2-3 lignes max)

3. FOURNIR des scénarios:
   - Cas optimiste (croissance +50%)
   - Cas pessimiste (croissance -50%)
   - Cas réaliste (projections baseline)

4. GUIDANCE MAROC:
   - TVA applicable (20% standard, 7-10% pour certains secteurs)
   - Cotisations CNSS (~8% sur salaires)
   - Déclaration IS (30% + additionnels)
   - Obligations comptables et fiscales
   - Statut juridique recommandé (SARL vs auto-entrepreneur vs SA)

5. RÉPONDRE à des questions de finance (impôts, cash flow, etc.)

Format de réponse:
- Sois pédagogue mais concis
- Utilise des tableaux pour présenter les KPIs
- Structure: Analyse → Recommandations → Prochaines étapes
- Propose un business plan de 24 mois si demandé

Langue: Français par défaut sauf si demandé anglais
"""


class Phase3AnalysisAgent(BaseAgent):
    agent_id = "phase3_analysis"
    agent_version = "1.0.0"
    description = "Financial analysis, KPI calculation, business plan generation"

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Perform financial analysis on collected data.
        Can also answer follow-up finance questions while in this phase.
        """
        try:
            business_state = message.structured_payload or message.context.get("business_state", {})

            if not business_state:
                return self._make_error_response(message, "No business data provided")

            # Format data for analysis
            data_summary = self._format_business_data(business_state)

            # Build analysis prompt
            if message.user_message:
                # User asking a follow-up question
                user_request = message.user_message
            else:
                # Initial analysis request
                user_request = f"""Analyse ma situation financière basée sur ces données:

{data_summary}

Fournir:
1. Calcul du seuil de rentabilité et de la marge
2. Analyse du cash flow sur 12 mois
3. Recommandations de structure juridique et obligations fiscales
4. 3 scénarios (optimiste, pessimiste, réaliste)
5. Prochaines étapes pour lancer l'entreprise
"""

            # Call LLM for analysis
            analysis_response = groq_client.chat(
                system_prompt=PHASE3_ANALYSIS_SYSTEM_PROMPT,
                user_message=user_request,
                temperature=0.5,  # More factual, structured
                max_tokens=1200,
            )

            return AgentResponse(
                agent_id=self.agent_id,
                session_id=message.session_id,
                intent=message.intent,
                message=analysis_response,
                agent_mode="analyzing",
                business_state=business_state,
                structured_output={
                    "phase": "analysis",
                    "analysis_completed": True,
                },
            )

        except Exception as e:
            logger.error(f"Phase3 agent error: {e}")
            return self._make_error_response(message, str(e))

    def can_handle(self, intent: str) -> bool:
        """Handle financial analysis, scenarios, business planning."""
        return intent in ("analyze", "business_plan", "financial_guidance", "scenario", "kpi_calculation")

    def _format_business_data(self, business_state: dict) -> str:
        """Format business data into readable summary."""
        lines = []
        
        if business_state.get("entity_name"):
            lines.append(f"Entreprise: {business_state['entity_name']}")
        if business_state.get("sector"):
            lines.append(f"Secteur: {business_state['sector']}")
        if business_state.get("statut_juridique"):
            lines.append(f"Statut: {business_state['statut_juridique']}")
        if business_state.get("prix_vente_unitaire"):
            lines.append(f"Prix de vente: {business_state['prix_vente_unitaire']} MAD")
        if business_state.get("nb_clients_mois1"):
            lines.append(f"Clients M1: {business_state['nb_clients_mois1']}")
        if business_state.get("taux_croissance_mensuel"):
            lines.append(f"Croissance/mois: {business_state['taux_croissance_mensuel']}%")
        if business_state.get("taux_fidelisation"):
            lines.append(f"Fidélisation: {business_state['taux_fidelisation']}%")
        if business_state.get("cout_fabrication_unitaire"):
            lines.append(f"Coût unitaire: {business_state['cout_fabrication_unitaire']} MAD")
        if business_state.get("loyer_mensuel"):
            lines.append(f"Loyer: {business_state['loyer_mensuel']} MAD/mois")
        if business_state.get("salaires_equipe"):
            lines.append(f"Salaires: {business_state['salaires_equipe']} MAD/mois")
        if business_state.get("investissements_initiaux"):
            lines.append(f"Investissement initial: {business_state['investissements_initiaux']} MAD")
        if business_state.get("emprunts"):
            lines.append(f"Emprunt bancaire: {business_state['emprunts']} MAD")
        if business_state.get("own_capital_invested"):
            lines.append(f"Capital propre: {business_state['own_capital_invested']} MAD")

        return "\n".join(lines)
