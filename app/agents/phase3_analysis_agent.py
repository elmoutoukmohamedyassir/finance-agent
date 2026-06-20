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
from app.tools.plan_pipeline import compute_plan, format_full_plan_tables

logger = logging.getLogger(__name__)


def _wants_full_plan(user_message: str, conversation_history: list) -> bool:
    """
    Detect whether the user is asking for the complete business plan
    (compte de résultat, plan de trésorerie, plan de financement, bilan),
    either explicitly, or by saying "oui" right after we offered it.
    """
    lower = (user_message or "").lower().strip()
    explicit = [
        "business plan complet", "plan complet", "compte de résultat",
        "plan de trésorerie", "plan de financement", "bilan prévisionnel",
        "génère le plan", "generate the plan", "full business plan",
    ]
    if any(kw in lower for kw in explicit):
        return True

    affirmative = ("oui", "yes", "ok", "d'accord", "daccord", "vas-y", "go", "allez", "bien sûr", "bien sur")
    if any(lower == a or lower.startswith(a + " ") or lower.startswith(a + ",") for a in affirmative):
        last_assistant = next(
            (m.get("content", "") for m in reversed(conversation_history) if m.get("role") == "assistant"),
            "",
        )
        return "business plan complet" in last_assistant.lower()

    return False


PLAN_OFFER_SUFFIX = (
    "\n\n---\n📄 **Souhaitez-vous que je génère votre Business Plan complet ?**\n"
    "*(Compte de résultat · Plan de trésorerie · Plan de financement · Bilan simplifié, sur 2 ans)*\n\n"
    "Répondez **oui** pour le générer, ou posez-moi une question sur les résultats."
)


class Phase3AnalysisAgent(BaseAgent):
    agent_id = "phase3_analysis"
    agent_version = "2.1.0"
    description = "Financial analysis backed by the real KPI/break-even engine — the LLM only interprets numbers, never invents them."

    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Perform financial analysis on collected data using the real
        deterministic engine, then ask the LLM to interpret/explain it.
        Can also answer follow-up finance questions, or generate the full
        business plan document, while staying grounded in the same
        calculated numbers.
        """
        try:
            business_state = message.structured_payload or message.context.get("business_state", {})
            if not business_state or not any(v is not None for v in business_state.values()):
                return self._make_error_response(message, "No business data provided")

            conversation_history = message.context.get("conversation_history", [])

            # ── 1. Run the REAL deterministic engine — no LLM math ─────────
            computed = compute_plan(business_state, fiscal_year=2025)
            derived_summary = computed.derived_summary if computed else ""
            plan_summary = computed.plan_summary if computed else ""
            plan = computed.plan if computed else None
            if not computed:
                # Missing/invalid required fields (e.g. prix_vente_unitaire=0)
                # — be transparent about it below rather than crashing or
                # letting the LLM fabricate numbers.
                logger.warning("Phase3: calculation engine could not run (insufficient/invalid business_state)")

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

            data_summary = self._format_business_data(business_state)

            # Track whether we've already given the initial analysis in an
            # earlier turn. conversation_history is never empty by the time
            # we reach phase3 (phase1+phase2 already happened), so we can't
            # use "history exists" as the signal — we need an explicit marker
            # that survives across turns. Reuse `asked_questions`, which
            # chat.py already persists back onto the session for us.
            asked_questions = list(message.context.get("asked_questions", []))
            is_followup = "phase3_analysis_done" in asked_questions

            # ── 3. Full Business Plan path ──────────────────────────────────
            if plan and _wants_full_plan(message.user_message, conversation_history):
                full_tables = format_full_plan_tables(plan)
                system_prompt = build_analysis_system_prompt(phase="business_plan", rag_context=rag_context)
                narrative = groq_client.chat(
                    system_prompt=system_prompt,
                    user_message=(
                        "Génère une synthèse exécutive du Business Plan ci-dessous : résumé exécutif, "
                        "analyse des 3 principaux risques avec mitigation, recommandations stratégiques, "
                        "et un plan d'action concret pour les 6 premiers mois.\n\n"
                        f"DONNÉES (calculées — ne pas recalculer) :\n{full_tables}"
                    ),
                    conversation_history=conversation_history[-4:],
                    temperature=0.3,
                    max_tokens=2200,
                )
                full_message = (
                    f"{narrative}\n\n{full_tables}\n\n"
                    "---\n✅ **Business Plan généré.** Vous pouvez me poser des questions sur "
                    "n'importe quelle ligne, ou demander un scénario optimiste/pessimiste."
                )
                return AgentResponse(
                    agent_id=self.agent_id,
                    session_id=message.session_id,
                    intent=message.intent,
                    message=full_message,
                    agent_mode="business_plan",
                    business_state=business_state,
                    metrics_calculated={
                        "derived_summary": derived_summary,
                        "plan_summary": plan_summary,
                        "full_plan_tables": full_tables,
                    },
                    structured_output={
                        "next_phase": "phase3",
                        "analysis_completed": True,
                        "full_plan_generated": True,
                        "asked_questions": asked_questions,
                    },
                )

            # ── 4. Standard interpretive analysis / follow-up Q&A ───────────
            system_prompt = build_analysis_system_prompt(phase="pre_creation", rag_context=rag_context)

            if is_followup and derived_summary:
                user_request = (
                    f"Question de l'entrepreneur : {message.user_message}\n\n"
                    f"RAPPEL — VARIABLES CALCULÉES (ne pas recalculer) :\n{derived_summary}\n\n"
                    f"RAPPEL — PLAN 24 MOIS (ne pas recalculer) :\n{plan_summary}"
                )
                offer_suffix = ""
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
                # Only offer the full plan right after the FIRST analysis pass,
                # not on every follow-up turn. Mark it done so the NEXT turn
                # is correctly treated as a follow-up rather than redoing this.
                offer_suffix = PLAN_OFFER_SUFFIX
                if "phase3_analysis_done" not in asked_questions:
                    asked_questions.append("phase3_analysis_done")
            else:
                # Engine couldn't run — say so plainly instead of fabricating numbers.
                user_request = (
                    f"DONNÉES DISPONIBLES :\n{data_summary}\n\n"
                    "Les données collectées sont insuffisantes pour calculer un seuil de rentabilité "
                    "fiable (prix de vente, nombre de clients ou charges fixes manquants ou nuls). "
                    "Indique précisément quelles informations manquent et pourquoi elles sont "
                    "nécessaires. NE PAS inventer de chiffres."
                )
                offer_suffix = ""

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
                message=analysis_response + offer_suffix,
                agent_mode="analyzing",
                business_state=business_state,
                metrics_calculated=(
                    {"derived_summary": derived_summary, "plan_summary": plan_summary}
                    if derived_summary else None
                ),
                structured_output={
                    "next_phase": "phase3",
                    "analysis_completed": bool(derived_summary),
                    "asked_questions": asked_questions,
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