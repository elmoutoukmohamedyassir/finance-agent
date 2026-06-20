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


def _format_full_plan_tables(plan) -> str:
    """
    The complete plan financier: plan de financement, compte de résultat (2 ans),
    plan de trésorerie, bilan simplifié (2 ans), KPIs. This is the full
    "Business Plan" — format_plan_for_prompt() (used for grounding the LLM
    on every turn) is a condensed few-line version of the same data; this is
    the full version actually shown to the user once they ask for it.
    """
    a1, a2 = plan.annee1, plan.annee2
    fin = plan.plan_financement
    b1, b2 = plan.bilan_annee1, plan.bilan_annee2

    lines = [
        "━━━ PLAN DE FINANCEMENT INITIAL ━━━",
        f"Besoins    : {fin.total_besoins:>12,.0f} MAD",
        f"Ressources : {fin.total_ressources:>12,.0f} MAD",
        f"Solde      : {fin.solde:>12,.0f} MAD  "
        + ("✅ équilibré" if fin.solde >= 0 else "❌ DÉFICIT — financement complémentaire nécessaire"),
        "",
        "━━━ COMPTE DE RÉSULTAT PRÉVISIONNEL ━━━",
        f"{'':<28}{'Année 1':>14}{'Année 2':>14}",
        f"{'Chiffre d’affaires':<28}{a1['ca_total']:>14,.0f}{a2['ca_total']:>14,.0f}",
        f"{'Marge brute':<28}{a1['marge_brute']:>14,.0f}{a2['marge_brute']:>14,.0f}",
        f"{'EBITDA':<28}{a1['ebitda']:>14,.0f}{a2['ebitda']:>14,.0f}",
        f"{'Résultat net':<28}{a1['resultat_net']:>14,.0f}{a2['resultat_net']:>14,.0f}",
        f"{'Marge nette':<28}{str(a1.get('marge_nette_pct','N/A'))+'%':>14}{str(a2.get('marge_nette_pct','N/A'))+'%':>14}",
        "",
        "━━━ PLAN DE TRÉSORERIE ━━━",
        f"Trésorerie fin année 1 : {a1['tresorerie_fin']:>12,.0f} MAD",
        f"Trésorerie fin année 2 : {a2['tresorerie_fin']:>12,.0f} MAD",
        "",
        "━━━ BILAN SIMPLIFIÉ ━━━",
        f"{'':<28}{'Année 1':>14}{'Année 2':>14}",
        f"{'Immobilisations nettes':<28}{b1.immobilisations_nettes:>14,.0f}{b2.immobilisations_nettes:>14,.0f}",
        f"{'Stocks':<28}{b1.stocks:>14,.0f}{b2.stocks:>14,.0f}",
        f"{'Créances clients':<28}{b1.creances_clients:>14,.0f}{b2.creances_clients:>14,.0f}",
        f"{'Trésorerie':<28}{b1.tresorerie:>14,.0f}{b2.tresorerie:>14,.0f}",
        f"{'TOTAL ACTIF':<28}{b1.total_actif:>14,.0f}{b2.total_actif:>14,.0f}",
        f"{'Capital social':<28}{b1.capital_social:>14,.0f}{b2.capital_social:>14,.0f}",
        f"{'Réserves / résultats':<28}{b1.reserves_resultats:>14,.0f}{b2.reserves_resultats:>14,.0f}",
        f"{'Dettes bancaires':<28}{b1.dettes_bancaires:>14,.0f}{b2.dettes_bancaires:>14,.0f}",
        f"{'TOTAL PASSIF':<28}{b1.total_passif:>14,.0f}{b2.total_passif:>14,.0f}",
        "",
        "━━━ KPIs CLÉS ━━━",
        f"Seuil de rentabilité : {plan.seuil_rentabilite_clients:.0f} clients/mois" if plan.seuil_rentabilite_clients else "Seuil de rentabilité : non calculable",
        f"Point mort           : mois {plan.mois_point_mort}" if plan.mois_point_mort else "Point mort           : non atteint sur 24 mois",
        f"ROI année 1          : {plan.roi_annee1:.1f}%" if plan.roi_annee1 else "",
        f"ROI année 2          : {plan.roi_annee2:.1f}%" if plan.roi_annee2 else "",
        f"DSCR année 1         : {plan.dscr_annee1:.2f}x" if plan.dscr_annee1 else "",
    ]
    return "\n".join(l for l in lines if l)


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
            derived_summary = ""
            plan_summary = ""
            plan = None
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
                full_tables = _format_full_plan_tables(plan)
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