"""
Phase 3: Financial Analysis Agent - KPI calculations, scenarios, business plan generation.

REWRITTEN: the previous version asked the LLM to "CALCULER les KPIs" in free
text — meaning every break-even point, margin and cash-flow figure shown to
users was an LLM guess, not a real calculation.

This version runs the project's own deterministic engine
(hypothesis_ingestor.ingest_hypothesis + plan_generator.generate_24m_plan,
the same tools used by the Bloc-2/Bloc-3 pipeline) and only asks the LLM to
*interpret* the resulting numbers — per the project's own prompt rule:
"Utiliser UNIQUEMENT les MÉTRIQUES CALCULÉES — ne jamais recalculer."

Uses collected data to:
- Calculate real financial KPIs (break-even, BFR, margins, 24-month plan)
- Ground Morocco tax guidance with RAG context when available
- Explain the numbers, propose scenarios, and answer follow-up questions
"""
import logging

from app.agents.base_agent import BaseAgent, AgentMessage, AgentResponse
from app.core.groq_client import groq_client
from app.core.prompts import build_analysis_system_prompt
from app.schemas.hypothesis_output import (
    HypothesisOutput, BlocVentes, BlocAchats, BlocChargesFixes,
    BlocEncaissements, HypothesisMetadata,
)
from app.tools.hypothesis_ingestor import ingest_hypothesis, format_derived_for_prompt
from app.tools.plan_generator import generate_24m_plan, format_plan_for_prompt

logger = logging.getLogger(__name__)


def _nature_clients_to_literal(value) -> str:
    """
    BlocEncaissements.H22_nature_clients is a payment-terms literal
    ("comptant" | "credit" | "mixte"), but Phase 2 collects it as a
    B2C/B2B/Mixte choice (see question_agent.py: "B2C = comptant, B2B =
    délai, Mixte") — translate rather than passing the segment through
    directly, which would fail pydantic validation.
    """
    mapping = {
        "b2c": "comptant", "b2b": "credit", "mixte": "mixte",
        "comptant": "comptant", "credit": "credit",
    }
    return mapping.get(str(value or "").strip().lower(), "comptant")


def _business_state_to_hypothesis(bs: dict) -> HypothesisOutput:
    """
    Map the flat business_state dict (the H1-H22 startup-style fields collected
    by Phase 2 — prix_vente_unitaire, nb_clients_mois1, loyer_mensuel, etc.)
    onto the structured HypothesisOutput the calculation engine expects.
    """
    g = bs.get
    return HypothesisOutput(
        ventes=BlocVentes(
            H1_segment_client=g("segment_client") or "B2C",
            H2_prix_vente_unitaire=g("prix_vente_unitaire"),
            H4_nb_clients_mois1=g("nb_clients_mois1"),
            H5_taux_croissance_mensuel=g("taux_croissance_mensuel", 5.0),
            H6_taux_fidelisation=g("taux_fidelisation", 85.0),
        ),
        achats=BlocAchats(
            H8_type_activite=g("type_activite") or "service",
            H9_cout_fabrication_unitaire=g("cout_fabrication_unitaire", 0),
            H11_cout_infra_numerique=g("cout_infra_numerique", 0),
        ),
        charges_fixes=BlocChargesFixes(
            H13_loyer_mensuel=g("loyer_mensuel"),
            H14_salaires_equipe=g("salaires_equipe"),
            H15_charges_utilites=g("charges_utilites"),
            H17_budget_marketing=g("budget_marketing"),
            H19_investissements_initiaux=g("investissements_initiaux"),
            H21_emprunts=g("emprunts"),
        ),
        encaissements=BlocEncaissements(
            H22_nature_clients=_nature_clients_to_literal(g("nature_clients_encaissements")),
            delai_jours=int(g("delai_jours") or 0),
        ),
        metadata=HypothesisMetadata(
            description_projet=g("entity_name"),
            secteur=g("sector"),
            statut_juridique=g("statut_juridique") or "Auto-entrepreneur",
            capital_social=g("own_capital_invested"),
        ),
    )


class Phase3AnalysisAgent(BaseAgent):
    agent_id = "phase3_analysis"
    agent_version = "2.0.0"
    description = "Financial analysis backed by the real KPI/break-even engine — the LLM only interprets numbers, never invents them."

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Perform financial analysis on collected data using the real
        deterministic engine, then ask the LLM to interpret/explain it.
        Can also answer follow-up finance questions while staying grounded
        in the same calculated numbers.
        """
        try:
            business_state = message.structured_payload or message.context.get("business_state", {})
            if not business_state or not any(v is not None for v in business_state.values()):
                return self._make_error_response(message, "No business data provided")

            conversation_history = message.context.get("conversation_history", [])

            # ── 1. Run the REAL deterministic engine — no LLM math ─────────
            derived_summary = ""
            plan_summary = ""
            try:
                hypothesis = _business_state_to_hypothesis(business_state)
                financial_data, derived, proj_inputs = ingest_hypothesis(hypothesis, fiscal_year=2025)
                derived_summary = format_derived_for_prompt(derived)
                plan = generate_24m_plan(
                    proj=proj_inputs,
                    derived=derived,
                    capital_propre=business_state.get("own_capital_invested") or 0,
                    emprunt=business_state.get("emprunts") or 0,
                )
                plan_summary = format_plan_for_prompt(plan)
            except Exception as e:
                # Missing/invalid required fields (e.g. prix_vente_unitaire=0)
                # — be transparent about it below rather than crashing or
                # letting the LLM fabricate numbers.
                logger.warning(f"Phase3: calculation engine could not run: {e}")

            # ── 2. Optional RAG grounding for Morocco tax/legal context ────
            rag_context = ""
            try:
                from app.rag.retriever import retrieve_context
                sector = business_state.get("sector") or ""
                rag_context, confidence, use_fallback = retrieve_context(
                    f"seuil de rentabilité BFR fiscalité création entreprise {sector} Maroc"
                )
                if use_fallback:
                    rag_context = ""
            except Exception as e:
                logger.info(f"Phase3: RAG unavailable, continuing without it: {e}")

            system_prompt = build_analysis_system_prompt(phase="pre_creation", rag_context=rag_context)
            data_summary = self._format_business_data(business_state)

            if message.user_message and derived_summary:
                # Follow-up question while already in phase3 — stay grounded
                # in the SAME calculated numbers, don't recompute per-turn.
                user_request = (
                    f"Question de l'entrepreneur : {message.user_message}\n\n"
                    f"RAPPEL — VARIABLES CALCULÉES (ne pas recalculer) :\n{derived_summary}\n\n"
                    f"RAPPEL — PLAN 24 MOIS (ne pas recalculer) :\n{plan_summary}"
                )
            elif derived_summary:
                user_request = (
                    f"DONNÉES DU PROJET :\n{data_summary}\n\n"
                    f"VARIABLES DÉRIVÉES (calculées — ne pas recalculer) :\n{derived_summary}\n\n"
                    f"PLAN FINANCIER 24 MOIS (calculé — ne pas recalculer) :\n{plan_summary}\n\n"
                    "Fournis :\n"
                    "1. Synthèse de la viabilité financière (2-3 phrases avec les chiffres réels ci-dessus)\n"
                    "2. Explication du seuil de rentabilité et du BFR en termes simples\n"
                    "3. 3 scénarios qualitatifs (optimiste/réaliste/pessimiste) en t'appuyant sur le taux "
                    "de croissance fourni — sans recalculer le seuil de rentabilité, seulement le commenter\n"
                    "4. Statut juridique recommandé + obligations fiscales clés (TVA, IS/IR, CNSS)\n"
                    "5. Une action prioritaire immédiate"
                )
            else:
                # Engine couldn't run — say so plainly instead of fabricating numbers.
                user_request = (
                    f"DONNÉES DISPONIBLES :\n{data_summary}\n\n"
                    "Les données collectées sont insuffisantes pour calculer un seuil de rentabilité "
                    "fiable (prix de vente, nombre de clients ou charges fixes manquants ou nuls). "
                    "Indique précisément quelles informations manquent et pourquoi elles sont "
                    "nécessaires. NE PAS inventer de chiffres."
                )

            analysis_response = groq_client.chat(
                system_prompt=system_prompt,
                user_message=user_request,
                conversation_history=conversation_history[-6:],
                temperature=0.3,
                max_tokens=1400,
            )

            return AgentResponse(
                agent_id=self.agent_id,
                session_id=message.session_id,
                intent=message.intent,
                message=analysis_response,
                agent_mode="analyzing",
                business_state=business_state,
                metrics_calculated=(
                    {"derived_summary": derived_summary, "plan_summary": plan_summary}
                    if derived_summary else None
                ),
                structured_output={
                    "next_phase": "phase3",
                    "analysis_completed": bool(derived_summary),
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