"""
agents/question_agent.py — Strict data collection for H1–H22.

KEY FIXES vs old version:
  1. extract_and_validate() uses typed validation — rejects "t", "g", "beaucoup"
     for numeric fields. Agent NEVER silently stores garbage.
  2. FIELD_TYPES drives validation — not guesswork.
  3. is_finance_question() detects Q&A intent before trying to extract data.
  4. Phase-aware priority: pre_creation asks H1–H22, post_creation asks enterprise.
"""

import json
import re
import logging
from typing import Optional, Tuple

from app.core.groq_client import groq_client

logger = logging.getLogger(__name__)


# ── Field types ───────────────────────────────────────────────────────────────
FIELD_TYPES = {
    "entity_type": "choix", "entity_name": "texte", "sector": "texte",
    "statut_juridique": "choix", "segment_client": "choix",
    "prix_vente_unitaire": "numerique", "abonnement_mensuel": "numerique",
    "nb_clients_mois1": "numerique", "taux_croissance_mensuel": "pourcentage",
    "taux_fidelisation": "pourcentage", "saisonnalite": "booleen_details",
    "type_activite": "choix", "cout_fabrication_unitaire": "numerique",
    "quantite_min_commande": "numerique", "cout_infra_numerique": "numerique",
    "delai_fournisseur_jours": "numerique", "loyer_mensuel": "numerique",
    "salaires_equipe": "numerique", "charges_utilites": "numerique",
    "licences_logicielles": "numerique", "budget_marketing": "numerique",
    "honoraires_conseil": "numerique", "investissements_initiaux": "numerique",
    "certifications": "numerique", "emprunts": "numerique_details",
    "nature_clients_encaissements": "choix", "delai_jours": "numerique",
    "own_capital_invested": "numerique", "total_revenue": "numerique",
    "operating_expenses": "numerique", "salaries_and_benefits": "numerique",
    "cost_of_goods_sold": "numerique", "total_debt": "numerique",
    "total_equity": "numerique", "total_assets": "numerique",
}

# Question definitions (label + question text + optional choices)
FIELD_QUESTIONS_FR = {
    "entity_type":            {"label": "type d'entité",                   "question": "S'agit-il d'une entreprise privée/publique (corporate) ou d'une entité publique (gouvernement) ?", "choices": ["corporate", "government"]},
    "entity_name":            {"label": "nom du projet",                   "question": "Quel est le nom de votre entreprise ou projet ?"},
    "sector":                 {"label": "secteur d'activité",              "question": "Dans quel secteur opérez-vous ? (ex: restauration, tech, BTP, conseil...)"},
    "statut_juridique":       {"label": "statut juridique",                "question": "Quel statut juridique envisagez-vous ? (SARL, SA, Auto-entrepreneur, SAS)", "choices": ["sarl", "sa", "auto-entrepreneur", "sas"]},
    "segment_client":         {"label": "segment client",                  "question": "Quel est votre segment client principal ? (B2C, B2B ou Mixte)", "choices": ["b2c", "b2b", "mixte"]},
    "prix_vente_unitaire":    {"label": "prix de vente unitaire",          "question": "Quel est votre prix de vente unitaire en MAD ?"},
    "abonnement_mensuel":     {"label": "abonnement mensuel",              "question": "Proposez-vous un abonnement mensuel ? Si oui, quel montant en MAD ? (0 si non)"},
    "nb_clients_mois1":       {"label": "clients au 1er mois",             "question": "Combien de clients prévoyez-vous le premier mois ?"},
    "taux_croissance_mensuel":{"label": "taux de croissance mensuel",      "question": "Quel est votre taux de croissance mensuel estimé en % ? (ex: 8 pour 8%)"},
    "taux_fidelisation":      {"label": "taux de fidélisation",            "question": "Quel est votre taux de fidélisation client estimé en % ? (ex: 85 pour 85%)"},
    "saisonnalite":           {"label": "saisonnalité",                    "question": "Votre activité est-elle saisonnière ? Si oui, quels mois sont forts/faibles ?"},
    "type_activite":          {"label": "type d'activité",                 "question": "Quel est le type de votre activité ? (service, produit ou hybride)", "choices": ["service", "produit", "hybride"]},
    "cout_fabrication_unitaire": {"label": "coût de fabrication unitaire", "question": "Quel est le coût de fabrication ou de revient unitaire en MAD ? (0 si service pur)"},
    "quantite_min_commande":  {"label": "quantité minimum commande",       "question": "Quelle est la quantité minimum de commande auprès de vos fournisseurs ?"},
    "cout_infra_numerique":   {"label": "coût infrastructure numérique",   "question": "Quel est le coût de votre infrastructure numérique par mois en MAD ? (hébergement, SaaS... 0 si aucun)"},
    "delai_fournisseur_jours":{"label": "délai fournisseur",               "question": "Quel est le délai moyen de livraison de vos fournisseurs en jours ?"},
    "loyer_mensuel":          {"label": "loyer mensuel",                   "question": "Quel est le montant de votre loyer mensuel en MAD ? (0 si travail à domicile)"},
    "salaires_equipe":        {"label": "salaires nets mensuels",          "question": "Quel est le total des salaires nets mensuels de votre équipe en MAD ?"},
    "charges_utilites":       {"label": "charges utilités mensuelles",     "question": "Quel est le montant mensuel de vos charges d'utilités (eau, électricité, internet) en MAD ?"},
    "licences_logicielles":   {"label": "licences logicielles mensuelles", "question": "Quel est le coût mensuel de vos licences logicielles en MAD ? (0 si aucun)"},
    "budget_marketing":       {"label": "budget marketing mensuel",        "question": "Quel est votre budget marketing mensuel en MAD ?"},
    "honoraires_conseil":     {"label": "honoraires conseil annuels",      "question": "Quel est le montant annuel de vos honoraires de conseil en MAD ? (comptable, juriste...)"},
    "investissements_initiaux":{"label": "investissements initiaux",       "question": "Quel est le montant total de vos investissements initiaux en MAD ? (matériel, aménagement...)"},
    "certifications":         {"label": "coût certifications",             "question": "Quel est le coût de vos certifications et normes en MAD ? (0 si aucun)"},
    "emprunts":               {"label": "montant emprunt",                 "question": "Avez-vous un emprunt bancaire ? Si oui, quel est le montant total en MAD ? (0 si non)"},
    "nature_clients_encaissements": {"label": "nature clients encaissements", "question": "Quelle est la nature de vos clients pour les encaissements ? (B2C = comptant, B2B = délai, Mixte)", "choices": ["b2c", "b2b", "mixte"]},
    "delai_jours":            {"label": "délai d'encaissement",            "question": "Quel est votre délai moyen d'encaissement en jours ? (0 pour comptant, 30 ou 60 pour B2B)"},
    "own_capital_invested":   {"label": "capital propre investi",          "question": "Quel est le montant du capital propre que vous apportez en MAD ?"},
}

# WHY context shown before each question
FIELD_CONTEXT_FR = {
    "entity_type":               "Détermine les KPIs et obligations fiscales applicables.",
    "segment_client":            "Le segment client détermine le modèle de revenus et les délais d'encaissement.",
    "prix_vente_unitaire":       "Le prix de vente est la base de tous les calculs de chiffre d'affaires.",
    "nb_clients_mois1":          "Le nombre de clients au démarrage fixe le CA initial et le seuil de rentabilité.",
    "taux_croissance_mensuel":   "Le taux de croissance projette votre CA sur 24 mois.",
    "taux_fidelisation":         "La fidélisation calcule votre churn et la valeur client long terme.",
    "type_activite":             "Service ou produit détermine les coûts variables et la marge brute.",
    "cout_fabrication_unitaire": "Le coût de revient détermine votre marge brute unitaire.",
    "loyer_mensuel":             "Le loyer est une charge fixe clé dans le calcul du seuil de rentabilité.",
    "salaires_equipe":           "La masse salariale est souvent la charge fixe principale — elle impacte directement le BFR.",
    "investissements_initiaux":  "Les investissements déterminent le plan de financement et les amortissements.",
    "emprunts":                  "L'emprunt détermine vos mensualités, votre DSCR et votre plan de financement.",
    "own_capital_invested":      "Le capital propre détermine l'équilibre de votre plan de financement.",
    "nature_clients_encaissements": "Le mode d'encaissement détermine votre Besoin en Fonds de Roulement (BFR).",
}

# Collection priority order (H1 → H22)
PRE_CREATION_PRIORITY = [
    "entity_name", "entity_type", "sector", "statut_juridique",
    "segment_client", "prix_vente_unitaire", "nb_clients_mois1",
    "taux_croissance_mensuel", "taux_fidelisation", "saisonnalite",
    "type_activite", "cout_fabrication_unitaire", "cout_infra_numerique",
    "loyer_mensuel", "salaires_equipe", "charges_utilites",
    "licences_logicielles", "budget_marketing", "honoraires_conseil",
    "investissements_initiaux", "certifications", "emprunts",
    "own_capital_invested", "nature_clients_encaissements", "delai_jours",
]

# Minimum fields before triggering pre-creation analysis
PRE_CREATION_MINIMUM = {
    "entity_type", "segment_client", "prix_vente_unitaire",
    "nb_clients_mois1", "taux_croissance_mensuel",
    "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
}

POST_CREATION_PRIORITY = [
    "total_revenue", "cost_of_goods_sold", "operating_expenses",
    "salaries_and_benefits", "total_debt", "total_equity", "total_assets",
]

PHASE_FIELD_PRIORITIES = {
    "pre_creation": PRE_CREATION_PRIORITY,
    "post_creation": POST_CREATION_PRIORITY,
}

# Finance question detection keywords
FINANCE_QUESTION_KEYWORDS = [
    "c'est quoi", "qu'est-ce", "comment", "pourquoi", "expliquez",
    "définir", "définition", "que signifie", "comment calculer",
    "quel est le taux", "quelle est la différence", "?",
]

EXTRACTION_SYSTEM_PROMPT_STRICT = """Tu es un extracteur de données financières Business Plan.

Lis le message et extrait UNIQUEMENT les données financières explicitement mentionnées.
Retourne UNIQUEMENT un JSON valide. Pas de texte autour. Pas de markdown.

Champs possibles :
{
  "entity_name": "string", "entity_type": "corporate"|"government",
  "sector": "string", "statut_juridique": "SARL"|"SA"|"auto-entrepreneur"|"SAS",
  "segment_client": "B2C"|"B2B"|"Mixte",
  "prix_vente_unitaire": number, "abonnement_mensuel": number,
  "nb_clients_mois1": number, "taux_croissance_mensuel": number,
  "taux_fidelisation": number, "saisonnalite": "string",
  "type_activite": "service"|"produit"|"hybride",
  "cout_fabrication_unitaire": number, "cout_infra_numerique": number,
  "loyer_mensuel": number, "salaires_equipe": number,
  "charges_utilites": number, "licences_logicielles": number,
  "budget_marketing": number, "honoraires_conseil": number,
  "investissements_initiaux": number, "certifications": number,
  "emprunts": number, "own_capital_invested": number,
  "nature_clients_encaissements": "B2C"|"B2B"|"Mixte", "delai_jours": number,
  "total_revenue": number, "operating_expenses": number,
  "salaries_and_benefits": number, "cost_of_goods_sold": number,
  "total_debt": number, "total_equity": number, "total_assets": number,
  "current_assets": number, "current_liabilities": number,
  "cash_inflow": number, "cash_outflow": number
}

RÈGLES CRITIQUES :
- NE JAMAIS extraire si ce n'est PAS explicitement dans le message
- Montants toujours en MAD (1 milliard MAD = 1000000000, 1 MMAD = 1000000)
- Une seule lettre ou mot vague ("t", "g", "beaucoup", "pas mal") → {}
- Si rien d'extractible → {}
- UNIQUEMENT le JSON. Rien d'autre."""


# ─────────────────────────────────────────────────────────────────────────────
# INTENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def is_finance_question(text: str) -> bool:
    lower = text.lower().strip()
    if any(kw in lower for kw in FINANCE_QUESTION_KEYWORDS):
        return True
    if re.match(r"^(pourquoi|comment|qu.est|c.est quoi|expliqu|définir|que signifie|kesako)", lower):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# STRICT EXTRACTION + VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_validate(
    user_message: str,
    pending_field: Optional[str],
    conversation_history: list,
) -> Tuple[dict, Optional[str]]:
    """
    Extracts financial data and validates strictly.
    Returns (data, None) on success or ({}, error_string) on failure.

    THE FIX: If pending_field is numeric and user says "t" or "beaucoup"
    → returns a clear error, never stores garbage.
    """
    if not user_message.strip():
        return {}, "Message vide."

    # Detect finance question intent — don't try to extract data from a question
    if pending_field and is_finance_question(user_message):
        return {}, None  # Caller routes to Q&A

    # LLM extraction
    try:
        raw = groq_client.chat(
            system_prompt=EXTRACTION_SYSTEM_PROMPT_STRICT,
            user_message=user_message,
            temperature=0.0,
            max_tokens=400,
        )
        clean = raw.strip().strip("```json").strip("```").strip()
        extracted = json.loads(clean) if clean else {}
    except Exception as e:
        logger.debug(f"Extraction failed: {e}")
        extracted = {}

    if not extracted:
        if pending_field:
            ftype = FIELD_TYPES.get(pending_field, "texte")
            if ftype in ("numerique", "pourcentage"):
                return {}, _type_error(pending_field, user_message, ftype)
        return {}, None

    # Validate extracted values
    clean_extracted = {}
    first_error = None
    for field, value in extracted.items():
        err = _validate_value(field, value)
        if err:
            if not first_error:
                first_error = err
        else:
            clean_extracted[field] = value

    # If the pending field has an error, surface it
    if pending_field and pending_field not in clean_extracted and first_error:
        return {}, first_error

    return clean_extracted, None


def _validate_value(field: str, value) -> Optional[str]:
    ftype = FIELD_TYPES.get(field, "texte")
    if ftype in ("numerique", "pourcentage"):
        return _validate_numeric(field, value)
    if ftype == "choix":
        choices = FIELD_QUESTIONS_FR.get(field, {}).get("choices", [])
        if choices and str(value).lower().strip() not in choices:
            return f"'{value}' invalide pour {field}. Options: {', '.join(choices)}"
    return None


def _validate_numeric(field: str, value) -> Optional[str]:
    try:
        v = float(str(value).replace(",", ".").replace(" ", "").replace("\u202f", ""))
    except (ValueError, TypeError):
        return f"Valeur non numérique pour {field}: '{value}'"

    bounds = {
        "taux_croissance_mensuel": (0, 100),
        "taux_fidelisation":       (0, 100),
        "delai_jours":             (0, 365),
        "nb_clients_mois1":        (1, 100000),
        "prix_vente_unitaire":     (0.01, 100_000_000),
    }
    if field in bounds:
        lo, hi = bounds[field]
        if not (lo <= v <= hi):
            return f"{field}: {v} doit être entre {lo} et {hi}"

    if v < 0 and field not in ("total_equity",):
        return f"{field} ne peut pas être négatif: {v}"

    return None


def _type_error(field: str, raw: str, ftype: str) -> str:
    label = FIELD_QUESTIONS_FR.get(field, {}).get("label", field)
    if ftype == "numerique":
        return f"J'attends un **nombre en MAD** pour **{label}** (ex: 5000, 12500). Vous avez écrit : \"{raw[:50]}\""
    if ftype == "pourcentage":
        return f"J'attends un **pourcentage** pour **{label}** (ex: 8, 15). Vous avez écrit : \"{raw[:50]}\""
    return f"Réponse non reconnue pour **{label}** : \"{raw[:50]}\""


# ─────────────────────────────────────────────────────────────────────────────
# QUESTION SEQUENCING
# ─────────────────────────────────────────────────────────────────────────────

def get_next_question(
    state,
    asked_questions: list,
    phase: str = "pre_creation",
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (field_name, question_text) for next unanswered field."""
    priority = PHASE_FIELD_PRIORITIES.get(phase, PRE_CREATION_PRIORITY)
    filled = state.filled_fields()

    # Expand with H-variable aliases
    alias_map = {
        "segment_client":            ["H1_segment_client"],
        "prix_vente_unitaire":       ["H2_prix_vente_unitaire"],
        "nb_clients_mois1":          ["H4_nb_clients_mois1"],
        "taux_croissance_mensuel":   ["H5_taux_croissance_mensuel"],
        "taux_fidelisation":         ["H6_taux_fidelisation"],
        "type_activite":             ["H8_type_activite"],
        "loyer_mensuel":             ["H13_loyer_mensuel"],
        "salaires_equipe":           ["H14_salaires_equipe", "salaries_and_benefits"],
        "investissements_initiaux":  ["H19_investissements_initiaux"],
        "emprunts":                  ["H21_emprunts", "total_debt"],
        "nature_clients_encaissements": ["H22_nature_clients"],
    }

    known = set(filled.keys())
    for canonical, aliases in alias_map.items():
        if any(a in known for a in aliases):
            known.add(canonical)

    minimum_met = PRE_CREATION_MINIMUM.issubset(known)
    type_activite = str(filled.get("type_activite", "")).lower()
    skip_service = {"cout_fabrication_unitaire", "quantite_min_commande", "delai_fournisseur_jours"}

    for field in priority:
        if field in known:
            continue
        if field in asked_questions:
            continue
        # Skip service-only fields for service businesses
        if type_activite == "service" and field in skip_service:
            continue
        # Skip optional fields once minimum is met
        if minimum_met and field not in PRE_CREATION_MINIMUM and phase == "pre_creation":
            # Only skip truly optional extras
            optional_extras = {"certifications", "licences_logicielles", "honoraires_conseil",
                               "abonnement_mensuel", "saisonnalite", "cout_infra_numerique",
                               "charges_utilites", "quantite_min_commande"}
            if field in optional_extras:
                continue

        q_data = FIELD_QUESTIONS_FR.get(field, {})
        question = q_data.get("question", f"Quelle est la valeur de {field} ?")
        return field, question

    return None, None