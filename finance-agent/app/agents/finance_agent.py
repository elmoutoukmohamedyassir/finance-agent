"""
agents/finance_agent.py — Complete enterprise finance agent.

PHASES:
  pre_creation  → Collect H1–H22, validate, calculate Bloc 2, 24-month plan,
                  offer business plan generation.
  creation      → Legal setup, fiscal obligations, CNSS/AMO, plan financement.
  post_creation → Ongoing KPIs, alerts, scenarios, RAG advisory.
  qa            → Any finance question at any phase.

KEY FIXES vs old agent:
  1. No more silent acceptance of "t", "g" or garbage — strict typed validation
  2. Agent asks ONE question at a time with a WHY explanation
  3. HypothesisOutput JSON ingested directly (bypass conversational collection)
  4. Phase transitions: pre_creation → creation → post_creation
  5. Business plan offered after pre-creation analysis
  6. Sessions persisted to SQLite
"""

import logging
from typing import Optional

from app.core.groq_client import groq_client
from app.core.prompts import (
    build_analysis_system_prompt,
    build_analysis_user_message,
)
from app.tools.metrics_calculator import calculate_from_business_state
from app.tools.scenario_engine import build_standard_scenarios, format_scenarios_for_prompt
from app.tools.hypothesis_ingestor import ingest_hypothesis, format_derived_for_prompt
from app.tools.plan_generator import generate_24m_plan, format_plan_for_prompt
from app.tools.fiscal_constants import get_fiscal_constants
from app.agents.question_agent import (
    extract_and_validate,
    get_next_question,
    is_finance_question,
    FIELD_QUESTIONS_FR,
    FIELD_TYPES,
    FIELD_CONTEXT_FR,
)
from app.rag.retriever import retrieve_context
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.session import ConversationSession, Phase
from app.schemas.hypothesis_output import HypothesisOutput
from app.services.session_service import (
    get_or_create_session,
    save_session,
    update_business_state_fields,
    load_hypothesis_into_session,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def handle_chat(
    session_id: Optional[str],
    user_message: str,
    hypothesis_payload: Optional[dict] = None,
) -> ChatResponse:
    session = get_or_create_session(session_id)

    # Direct Hypothesis Agent ingestion path
    if hypothesis_payload:
        return _handle_hypothesis_payload(session, hypothesis_payload, user_message)

    session.add_message("user", user_message)
    phase = session.phase

    # Finance Q&A always available (but not during active collection)
    if is_finance_question(user_message) and phase not in (Phase.WELCOME, Phase.COLLECTING):
        response = _answer_finance_question(session, user_message)
    elif phase in (Phase.WELCOME, Phase.COLLECTING):
        response = _handle_collection(session, user_message)
    elif phase == Phase.AWAITING_PLAN_CONFIRM:
        response = _handle_plan_confirmation(session, user_message)
    elif phase == Phase.PRE_CREATION:
        response = _handle_pre_creation(session, user_message)
    elif phase == Phase.CREATION:
        response = _handle_creation(session, user_message)
    elif phase == Phase.POST_CREATION:
        response = _handle_post_creation(session, user_message)
    else:
        response = _handle_collection(session, user_message)

    session.add_message("assistant", response.message)
    save_session(session)
    response.session_id = session.session_id
    response.business_state = session.business_state.filled_fields()
    response.current_phase = session.phase.value
    return response


# ─────────────────────────────────────────────────────────────────────────────
# HYPOTHESIS AGENT DIRECT INGESTION
# ─────────────────────────────────────────────────────────────────────────────

def _handle_hypothesis_payload(session, payload, user_message):
    try:
        hypothesis = HypothesisOutput.model_validate(payload)
        load_hypothesis_into_session(session, hypothesis)
        session.phase = Phase.PRE_CREATION

        state = session.business_state
        derived = session.derived_variables
        proj_inputs = session.projection_inputs

        metrics = calculate_from_business_state(state)
        metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

        plan = generate_24m_plan(
            proj=proj_inputs,
            derived=derived,
            capital_propre=state.own_capital_invested or 0,
            emprunt=state.total_debt or 0,
        )
        plan_summary = format_plan_for_prompt(plan)
        derived_summary = format_derived_for_prompt(derived)

        rag_context = retrieve_context(
            f"business plan pre-creation {state.sector or ''} Maroc"
        )
        system_prompt = build_analysis_system_prompt(phase="pre_creation", rag_context=rag_context)
        user_msg = (
            f"Hypothèses Agent ingérées.\n\n"
            f"BLOC 2 DÉRIVÉ:\n{derived_summary}\n\n"
            f"{build_analysis_user_message(state.filled_fields(), metrics_dict, '')}\n\n"
            f"PLAN 24 MOIS:\n{plan_summary}"
        )

        response_text = groq_client.chat(
            system_prompt=system_prompt,
            user_message=user_msg,
            temperature=0.2,
            max_tokens=2000,
        )
        response_text += (
            "\n\n---\n📄 **Souhaitez-vous que je génère votre Business Plan complet ?**\n"
            "Répondez **oui** pour lancer la génération."
        )

        session.phase = Phase.AWAITING_PLAN_CONFIRM
        session.cached_plan = plan
        session.cached_metrics = metrics_dict
        session.add_message("user", user_message)
        session.add_message("assistant", response_text)
        save_session(session)

        return ChatResponse(
            session_id=session.session_id,
            message=response_text,
            agent_mode="pre_creation_analysis",
            metrics_calculated=metrics_dict,
            current_phase="pre_creation",
        )
    except Exception as e:
        logger.error(f"Hypothesis ingestion failed: {e}", exc_info=True)
        msg = (
            f"⚠️ Impossible de lire le JSON de l'Agent Hypothèses : {e}\n\n"
            "Vérifiez le schéma HypothesisOutput ou décrivez votre projet manuellement."
        )
        session.add_message("user", user_message)
        session.add_message("assistant", msg)
        save_session(session)
        return ChatResponse(session_id=session.session_id, message=msg, agent_mode="error")


# ─────────────────────────────────────────────────────────────────────────────
# COLLECTION PHASE — H1 to H22 conversationally
# ─────────────────────────────────────────────────────────────────────────────

def _handle_collection(session, user_message):
    state = session.business_state

    # First turn: welcome
    if session.phase == Phase.WELCOME:
        session.phase = Phase.COLLECTING
        first_q = _get_first_question(state)
        return ChatResponse(
            session_id=session.session_id,
            message=(
                "Bonjour ! Je suis votre conseiller financier IA pour la création "
                "et la gestion d'entreprise au Maroc.\n\n"
                "Je vous accompagne en **3 phases** :\n"
                "1️⃣ **Pré-création** — Validation financière, seuil de rentabilité, plan prévisionnel\n"
                "2️⃣ **Création** — Structure juridique, obligations fiscales (IS/TVA/CNSS)\n"
                "3️⃣ **Post-création** — Suivi KPIs, alertes, recommandations continues\n\n"
                f"Commençons. {first_q}"
            ),
            agent_mode="collecting",
        )

    # Extract + strict validate from message
    if user_message.strip():
        extracted, validation_error = extract_and_validate(
            user_message=user_message,
            pending_field=session.pending_question,
            conversation_history=session.conversation_history[-4:],
        )
        if validation_error:
            return ChatResponse(
                session_id=session.session_id,
                message=_build_reprompt(session.pending_question, validation_error, user_message),
                agent_mode="collecting",
            )
        if extracted:
            update_business_state_fields(session, extracted)

    if _is_ready_for_pre_creation(session.business_state):
        return _transition_to_pre_creation(session)

    next_field, next_question = get_next_question(
        state=session.business_state,
        asked_questions=session.questions_asked,
        phase="pre_creation",
    )
    if not next_field:
        return _transition_to_pre_creation(session)

    session.questions_asked.append(next_field)
    session.pending_question = next_field
    question_text = _ask_with_context(next_field, next_question, state)

    return ChatResponse(
        session_id=session.session_id,
        message=question_text,
        agent_mode="collecting",
    )


def _get_first_question(state):
    _, q = get_next_question(state, asked_questions=[], phase="pre_creation")
    return q or "Décrivez votre projet en quelques mots."


def _build_reprompt(field, error, user_answer):
    """
    Clear rejection when answer fails validation.
    Fixes the 't' / 'g' / garbage acceptance bug.
    """
    label = FIELD_QUESTIONS_FR.get(field, {}).get("label", "cette information") if field else "cette information"
    ftype = FIELD_TYPES.get(field, "texte") if field else "texte"
    hints = {
        "numerique": "J'ai besoin d'un **nombre en MAD** (ex: 5000, 12500).",
        "pourcentage": "J'ai besoin d'un **pourcentage** entre 0 et 100 (ex: 8, 15).",
        "choix": "Choisissez parmi les options indiquées entre parenthèses.",
        "booleen_details": "Répondez par oui ou non, et précisez si nécessaire.",
        "numerique_details": "J'ai besoin d'un nombre ou d'un détail chiffré.",
    }
    return (
        f"⚠️ Je n'ai pas pu enregistrer **\"{user_answer[:60]}\"** pour **{label}**.\n\n"
        f"Raison : {error}\n\n"
        f"{hints.get(ftype, 'Pouvez-vous préciser ?')}"
    )


def _ask_with_context(field, raw_question, state):
    """Adds a one-sentence WHY before each question. Falls back to raw_question on error."""
    context_hint = FIELD_CONTEXT_FR.get(field, "")
    prompt = (
        f"Pose cette question à l'entrepreneur en une phrase d'explication + la question.\n\n"
        f"Question : {raw_question}\n"
        f"Contexte : {context_hint}\n\n"
        f"RÈGLES STRICTES : max 3 lignes · 1 seule question · stop après la question · "
        f"NE PAS inventer d'autres questions"
    )
    try:
        return groq_client.chat(
            system_prompt=(
                "Tu es un assistant Business Plan. Tu poses UNE question à la fois. "
                "Court, clair, en français."
            ),
            user_message=prompt,
            temperature=0.1,
            max_tokens=120,
        )
    except Exception:
        return raw_question


# ─────────────────────────────────────────────────────────────────────────────
# PRE-CREATION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _transition_to_pre_creation(session):
    session.phase = Phase.PRE_CREATION
    state = session.business_state

    hypothesis = _state_to_hypothesis(state)
    try:
        financial_data, derived, proj_inputs = ingest_hypothesis(hypothesis, fiscal_year=2025)
        session.derived_variables = derived
        session.projection_inputs = proj_inputs
    except Exception as e:
        logger.warning(f"Ingestor fallback: {e}")
        derived = None
        proj_inputs = None

    metrics = calculate_from_business_state(state)
    metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

    plan = None
    plan_summary = ""
    if proj_inputs and derived:
        plan = generate_24m_plan(
            proj=proj_inputs,
            derived=derived,
            capital_propre=state.own_capital_invested or 0,
            emprunt=state.total_debt or 0,
        )
        plan_summary = format_plan_for_prompt(plan)

    derived_summary = format_derived_for_prompt(derived) if derived else ""
    rag_context = retrieve_context(
        f"pre-creation business plan {state.sector or ''} seuil rentabilité BFR Maroc"
    )

    system_prompt = build_analysis_system_prompt(phase="pre_creation", rag_context=rag_context)
    user_msg = (
        f"PHASE PRÉ-CRÉATION\n\n"
        + (f"VARIABLES DÉRIVÉES:\n{derived_summary}\n\n" if derived_summary else "")
        + build_analysis_user_message(state.filled_fields(), metrics_dict, "")
        + (f"\n\nPLAN 24 MOIS:\n{plan_summary}" if plan_summary else "")
    )

    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=user_msg,
        conversation_history=session.conversation_history[-6:],
        temperature=0.2,
        max_tokens=2000,
    )
    response_text += (
        "\n\n---\n📄 **Souhaitez-vous que je génère votre Business Plan complet ?**\n"
        "*(Compte de résultat · Plan de trésorerie · Plan de financement · Bilan)*\n\n"
        "Répondez **oui** pour lancer, ou posez-moi une question sur les résultats."
    )

    session.phase = Phase.AWAITING_PLAN_CONFIRM
    session.cached_plan = plan
    session.cached_metrics = metrics_dict

    return ChatResponse(
        session_id=session.session_id,
        message=response_text,
        agent_mode="pre_creation_analysis",
        metrics_calculated=metrics_dict,
    )


def _handle_pre_creation(session, user_message):
    lower = user_message.lower()
    if any(w in lower for w in ["oui", "yes", "plan", "génère", "genere"]):
        session.phase = Phase.AWAITING_PLAN_CONFIRM
        return _handle_plan_confirmation(session, "oui")
    if any(w in lower for w in ["non", "pas maintenant", "plus tard", "continuer", "création", "suite"]):
        session.phase = Phase.CREATION
        return _handle_creation_transition(session)
    return _answer_finance_question(session, user_message)


# ─────────────────────────────────────────────────────────────────────────────
# BUSINESS PLAN GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _handle_plan_confirmation(session, user_message):
    lower = user_message.lower()
    if not any(w in lower for w in ["oui", "yes", "ok", "allez", "go", "génère", "plan", "bien sûr"]):
        session.phase = Phase.PRE_CREATION
        return _answer_finance_question(session, user_message)

    plan = session.cached_plan
    metrics_dict = session.cached_metrics or {}

    if not plan and session.projection_inputs and session.derived_variables:
        plan = generate_24m_plan(
            proj=session.projection_inputs,
            derived=session.derived_variables,
            capital_propre=session.business_state.own_capital_invested or 0,
            emprunt=session.business_state.total_debt or 0,
        )

    if not plan:
        session.phase = Phase.CREATION
        return _handle_creation_transition(session)

    a1, a2 = plan.annee1, plan.annee2
    fin = plan.plan_financement

    tables = (
        f"\n\n━━━ PLAN DE FINANCEMENT INITIAL ━━━\n"
        f"Besoins   : {fin.total_besoins:>12,.0f} MAD\n"
        f"Ressources: {fin.total_ressources:>12,.0f} MAD\n"
        f"Solde     : {fin.solde:>12,.0f} MAD  {'✅ équilibré' if fin.solde >= 0 else '❌ DÉFICIT'}\n\n"
        f"━━━ COMPTE DE RÉSULTAT PRÉVISIONNEL ━━━\n"
        f"{'':35} {'An 1':>12} {'An 2':>12}\n"
        f"{'─'*59}\n"
        f"{'Chiffre d affaires':<35} {a1['ca_total']:>12,.0f} {a2['ca_total']:>12,.0f}\n"
        f"{'Marge brute':<35} {a1['marge_brute']:>12,.0f} {a2['marge_brute']:>12,.0f}\n"
        f"{'EBITDA':<35} {a1['ebitda']:>12,.0f} {a2['ebitda']:>12,.0f}\n"
        f"{'Résultat net':<35} {a1['resultat_net']:>12,.0f} {a2['resultat_net']:>12,.0f}\n"
        f"{'Marge nette':<35} {str(a1.get('marge_nette_pct','N/A'))+'%':>12} {str(a2.get('marge_nette_pct','N/A'))+'%':>12}\n\n"
        f"━━━ PLAN DE TRÉSORERIE ━━━\n"
        f"Trésorerie fin an 1 : {a1['tresorerie_fin']:>12,.0f} MAD\n"
        f"Trésorerie fin an 2 : {a2['tresorerie_fin']:>12,.0f} MAD\n\n"
        f"━━━ KPIs CLÉS ━━━\n"
        + (f"Seuil rentabilité : {plan.seuil_rentabilite_clients:.0f} clients/mois\n" if plan.seuil_rentabilite_clients else "")
        + f"Point mort        : mois {plan.mois_point_mort or 'Non atteint sur 24 mois'}\n"
        + (f"ROI an 1          : {plan.roi_annee1:.1f}%\n" if plan.roi_annee1 else "")
        + (f"DSCR an 1         : {plan.dscr_annee1:.2f}x\n" if plan.dscr_annee1 else "")
    )

    rag_context = retrieve_context("business plan executive summary Maroc PME recommandations")
    system_prompt = build_analysis_system_prompt(phase="business_plan", rag_context=rag_context)
    narrative = groq_client.chat(
        system_prompt=system_prompt,
        user_message=(
            f"Génère une synthèse executive du Business Plan, une analyse des risques, "
            f"des recommandations stratégiques et un plan d'action pour les 6 premiers mois.\n\n"
            f"DONNÉES:\n{tables}"
        ),
        temperature=0.3,
        max_tokens=2500,
    )

    full_response = (
        f"{narrative}\n{tables}\n\n"
        "---\n✅ **Business Plan généré.**\n\n"
        "**Étape suivante : Phase Création** 🏗️\n"
        "Tapez **continuer** pour passer à la création de l'entreprise "
        "(statut juridique, immatriculation, obligations fiscales)."
    )

    session.phase = Phase.CREATION

    return ChatResponse(
        session_id=session.session_id,
        message=full_response,
        agent_mode="business_plan",
        metrics_calculated=metrics_dict,
        plan_output={
            "annee1": a1, "annee2": a2,
            "plan_financement": {"total_besoins": fin.total_besoins, "total_ressources": fin.total_ressources, "solde": fin.solde},
            "kpis": {"seuil_rentabilite_clients": plan.seuil_rentabilite_clients, "mois_point_mort": plan.mois_point_mort, "roi_annee1": plan.roi_annee1},
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# CREATION PHASE
# ─────────────────────────────────────────────────────────────────────────────

def _handle_creation_transition(session):
    fc = get_fiscal_constants(2025)
    state = session.business_state
    rag_context = retrieve_context(
        f"creation entreprise Maroc SARL SA immatriculation OMPIC CRI obligations fiscales {state.sector or ''}"
    )
    system_prompt = build_analysis_system_prompt(phase="creation", rag_context=rag_context)
    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=(
            f"PHASE CRÉATION — Guide l'entrepreneur.\n"
            f"Capital prévu: {state.own_capital_invested or 0:,.0f} MAD · "
            f"Emprunt: {state.total_debt or 0:,.0f} MAD · Secteur: {state.sector or 'N/D'}\n\n"
            f"IS 2025 (progressif): 10%/20%/28.5%/35% · "
            f"CNSS patronal: {fc.cnss.patronal_part*100:.1f}% · "
            f"SMIG: {fc.smig_mensuel:,.0f} MAD/mois\n\n"
            "Donne les étapes concrètes, délais et coûts estimés."
        ),
        temperature=0.2,
        max_tokens=1500,
    )
    return ChatResponse(session_id=session.session_id, message=response_text, agent_mode="creation_guidance")


def _handle_creation(session, user_message):
    lower = user_message.lower()
    if any(w in lower for w in ["continuer", "suite", "post", "suivant", "après", "lancé", "créée"]):
        session.phase = Phase.POST_CREATION
        return _handle_post_creation_welcome(session)
    return _answer_finance_question(session, user_message, phase_hint="creation")


# ─────────────────────────────────────────────────────────────────────────────
# POST-CREATION PHASE
# ─────────────────────────────────────────────────────────────────────────────

def _handle_post_creation_welcome(session):
    return ChatResponse(
        session_id=session.session_id,
        message=(
            "🎉 **Phase Post-Création activée.**\n\n"
            "Je suis votre conseiller financier continu. Je peux :\n"
            "• 📊 Analyser vos **KPIs** (EBITDA, marges, trésorerie, ratios)\n"
            "• ⚠️ Détecter des **alertes** financières\n"
            "• 📈 Projeter des **scénarios** sur 3 ans\n"
            "• 📄 Mettre à jour votre **Business Plan**\n"
            "• 💡 Répondre à toutes vos questions financières\n\n"
            "Partagez vos chiffres actuels ou posez votre question."
        ),
        agent_mode="post_creation",
    )


def _handle_post_creation(session, user_message):
    extracted, _ = extract_and_validate(user_message, pending_field=None, conversation_history=session.conversation_history[-4:])
    if extracted:
        update_business_state_fields(session, extracted)
    return _run_post_creation_analysis(session, user_message)


def _run_post_creation_analysis(session, user_message=""):
    state = session.business_state
    metrics = calculate_from_business_state(state)
    metrics_dict = {k: v for k, v in metrics.model_dump().items() if v is not None}

    scenarios_str = ""
    if state.total_revenue and (state.operating_expenses or state.cost_of_goods_sold):
        costs = (state.cost_of_goods_sold or 0) + (state.operating_expenses or 0) + (state.salaries_and_benefits or 0)
        if costs > 0:
            scenarios = build_standard_scenarios(
                starting_revenue=state.total_revenue,
                starting_costs=costs,
                starting_cash=state.cash_and_equivalents or 0,
                starting_debt=state.total_debt or 0,
                debt_service_annual=state.debt_service or 0,
                capex_annual=state.capital_expenditure or 0,
                years=3,
                entity_type=state.entity_type or "corporate",
            )
            scenarios_str = format_scenarios_for_prompt(scenarios, years=3)

    rag_context = retrieve_context(_build_rag_query(state, metrics_dict))
    system_prompt = build_analysis_system_prompt(phase="post_creation", rag_context=rag_context)
    user_msg = build_analysis_user_message(state.filled_fields(), metrics_dict, scenarios_str)
    if user_message:
        user_msg = f"Question : {user_message}\n\n{user_msg}"

    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=user_msg,
        conversation_history=session.conversation_history[-6:],
        temperature=0.2,
        max_tokens=1800,
    )
    return ChatResponse(session_id=session.session_id, message=response_text, agent_mode="post_creation_analysis", metrics_calculated=metrics_dict)


# ─────────────────────────────────────────────────────────────────────────────
# GENERAL FINANCE Q&A
# ─────────────────────────────────────────────────────────────────────────────

def _answer_finance_question(session, user_message, phase_hint=""):
    rag_context = retrieve_context(user_message)
    phase = phase_hint or session.phase.value
    system_prompt = build_analysis_system_prompt(phase=phase, rag_context=rag_context)
    response_text = groq_client.chat(
        system_prompt=system_prompt,
        user_message=user_message,
        conversation_history=session.conversation_history[-8:],
        temperature=0.25,
        max_tokens=1000,
    )
    return ChatResponse(session_id=session.session_id, message=response_text, agent_mode="qa")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _is_ready_for_pre_creation(state):
    h = state.filled_fields()
    has_revenue = bool(state.total_revenue or (h.get("prix_vente_unitaire") and h.get("nb_clients_mois1")))
    has_costs = any(v is not None for v in [state.operating_expenses, state.cost_of_goods_sold, state.salaries_and_benefits])
    return bool(state.entity_type and has_revenue and has_costs)


def _state_to_hypothesis(state):
    from app.schemas.hypothesis_output import (
        HypothesisOutput, BlocVentes, BlocAchats, BlocChargesFixes,
        BlocEncaissements, HypothesisMetadata
    )
    h = state.filled_fields()
    return HypothesisOutput(
        ventes=BlocVentes(
            H1_segment_client=h.get("segment_client", "B2C"),
            H2_prix_vente_unitaire=h.get("prix_vente_unitaire") or state.total_revenue,
            H4_nb_clients_mois1=h.get("nb_clients_mois1"),
            H5_taux_croissance_mensuel=h.get("taux_croissance_mensuel", 5.0),
            H6_taux_fidelisation=h.get("taux_fidelisation", 85.0),
        ),
        achats=BlocAchats(
            H8_type_activite=h.get("type_activite", "service"),
            H9_cout_fabrication_unitaire=h.get("cout_fabrication_unitaire", 0),
            H11_cout_infra_numerique=h.get("cout_infra_numerique", 0),
        ),
        charges_fixes=BlocChargesFixes(
            H13_loyer_mensuel=h.get("loyer_mensuel"),
            H14_salaires_equipe=h.get("salaires_equipe") or state.salaries_and_benefits,
            H15_charges_utilites=h.get("charges_utilites"),
            H17_budget_marketing=h.get("budget_marketing"),
            H19_investissements_initiaux=h.get("investissements_initiaux"),
            H21_emprunts=state.total_debt,
        ),
        encaissements=BlocEncaissements(
            H22_nature_clients=h.get("segment_client", "B2C"),
            delai_jours=h.get("delai_jours", 0),
        ),
        metadata=HypothesisMetadata(
            description_projet=state.entity_name,
            secteur=state.sector,
            statut_juridique=h.get("statut_juridique", "SARL"),
            capital_social=state.own_capital_invested,
        ),
    )


def _build_rag_query(state, metrics_dict):
    parts = ["analyse financière entreprise Maroc"]
    if getattr(state, "entity_type", None) == "government":
        parts.append("budget état finances publiques")
    else:
        if metrics_dict.get("ebitda_margin_pct", 100) < 8:
            parts.append("amélioration marge EBITDA rentabilité")
        if metrics_dict.get("debt_service_coverage", 10) < 1.5:
            parts.append("couverture dette restructuration financière")
    if getattr(state, "sector", None):
        parts.append(state.sector)
    return " ".join(parts)