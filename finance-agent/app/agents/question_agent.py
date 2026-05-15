"""
agents/question_agent.py — Collects missing enterprise financial data conversationally.

Adapts questions based on entity_type:
  corporate  → revenue, COGS, EBITDA, debt, equity, cash flow
  government → recettes, dépenses, solde, masse salariale, taux d'exécution
"""

import json
import logging
from typing import Optional

from app.core.groq_client import groq_client
from app.core.prompts import EXTRACTION_SYSTEM_PROMPT
from app.schemas.session import BusinessState

logger = logging.getLogger(__name__)

# ── Field priority: most impactful for analysis first ─────────────────────────
# Shared fields asked before we know entity type
COMMON_PRIORITY = [
    "entity_name",
    "entity_type",
    "sector",
    "total_revenue",
]

CORPORATE_PRIORITY = [
    "cost_of_goods_sold",
    "operating_expenses",
    "salaries_and_benefits",
    "depreciation_amortization",
    "interest_expense",
    "tax_expense",
    "total_assets",
    "current_assets",
    "current_liabilities",
    "total_equity",
    "total_debt",
    "cash_inflow",
    "cash_outflow",
    "own_capital_invested",
    "external_funding",
    "revenue_year2",
]

GOVERNMENT_PRIORITY = [
    "tax_revenue",
    "non_tax_revenue",
    "grants_and_transfers",
    "recurrent_expenditure",
    "capital_expenditure",
    "debt_service",
    "subsidies_paid",
    "salaries_and_benefits",
    "total_debt",
    "cash_and_equivalents",
    "investment_budget",
    "investment_executed",
    "revenue_year2",
]

# ── Questions per field ───────────────────────────────────────────────────────
FIELD_QUESTIONS = {
    # Common
    "entity_name": (
        "Quel est le nom de votre organisation ou entreprise ?"
    ),
    "entity_type": (
        "S'agit-il d'une entreprise privée/publique (corporate) "
        "ou d'une entité publique (ministère, collectivité, établissement public) ?"
    ),
    "sector": (
        "Dans quel secteur d'activité opérez-vous ? "
        "(ex : industrie, agroalimentaire, BTP, santé, éducation, finances publiques…)"
    ),
    "total_revenue": (
        "Quel est votre chiffre d'affaires (ou recettes totales) pour la période analysée, en MMAD ?"
    ),
    "revenue_year2": (
        "Quel était votre chiffre d'affaires (ou recettes) l'année précédente, en MMAD ? "
        "(pour calculer le taux de croissance)"
    ),

    # Corporate
    "cost_of_goods_sold": (
        "Quel est votre coût des ventes (coût de revient des produits/services vendus), en MMAD ?"
    ),
    "operating_expenses": (
        "Quelles sont vos charges d'exploitation hors coût des ventes "
        "(loyers, énergie, maintenance, achats divers…), en MMAD ?"
    ),
    "salaries_and_benefits": (
        "Quelle est votre masse salariale totale (salaires + charges sociales) pour la période, en MMAD ?"
    ),
    "depreciation_amortization": (
        "Quel est le montant des dotations aux amortissements et provisions (DAP) pour la période, en MMAD ?"
    ),
    "interest_expense": (
        "Quel est le montant de vos charges financières (intérêts sur emprunts) pour la période, en MMAD ?"
    ),
    "tax_expense": (
        "Quel est le montant de l'impôt sur les sociétés (IS) ou impôt équivalent, en MMAD ?"
    ),
    "total_assets": (
        "Quel est le total de votre bilan (total actif) à la clôture, en MMAD ?"
    ),
    "current_assets": (
        "Quel est le montant de l'actif circulant (stocks + créances + trésorerie actif), en MMAD ?"
    ),
    "current_liabilities": (
        "Quel est le montant du passif circulant (dettes fournisseurs + dettes fiscales + autres CT), en MMAD ?"
    ),
    "total_equity": (
        "Quel est le montant des capitaux propres (situation nette) à la clôture, en MMAD ?"
    ),
    "total_debt": (
        "Quel est le montant total de vos dettes financières (court et long terme), en MMAD ?"
    ),
    "cash_inflow": (
        "Quel est le total des encaissements de la période (recettes effectivement perçues), en MMAD ?"
    ),
    "cash_outflow": (
        "Quel est le total des décaissements de la période (paiements effectivement réalisés), en MMAD ?"
    ),
    "own_capital_invested": (
        "Quel est le montant des fonds propres investis dans le projet ou l'exercice, en MMAD ?"
    ),
    "external_funding": (
        "Quel est le montant des financements externes (emprunts bancaires, obligations, bailleurs…), en MMAD ?"
    ),

    # Government
    "tax_revenue": (
        "Quel est le montant des recettes fiscales (IR, IS, TVA, droits de douane…) de la période, en MMAD ?"
    ),
    "non_tax_revenue": (
        "Quel est le montant des recettes non fiscales (redevances, amendes, recettes domaniales…), en MMAD ?"
    ),
    "grants_and_transfers": (
        "Quel est le montant des subventions et transferts reçus (dotations de l'État, fonds internationaux…), en MMAD ?"
    ),
    "recurrent_expenditure": (
        "Quel est le montant des dépenses de fonctionnement (hors investissement), en MMAD ?"
    ),
    "capital_expenditure": (
        "Quel est le montant des dépenses d'investissement (CAPEX budgétaire exécuté), en MMAD ?"
    ),
    "debt_service": (
        "Quel est le montant du service de la dette (remboursements en principal + intérêts) de la période, en MMAD ?"
    ),
    "subsidies_paid": (
        "Quel est le montant des subventions versées (compensation, soutien aux entreprises…), en MMAD ?"
    ),
    "cash_and_equivalents": (
        "Quel est le niveau de la trésorerie disponible (réserves de l'entité) en fin de période, en MMAD ?"
    ),
    "investment_budget": (
        "Quel est le budget d'investissement initialement approuvé pour la période, en MMAD ?"
    ),
    "investment_executed": (
        "Quel est le montant d'investissement réellement exécuté (mandaté et payé), en MMAD ?"
    ),
}


def extract_business_info(message: str) -> dict:
    """
    Uses the LLM to extract structured financial data from natural language.
    Returns {} if nothing extractable.
    """
    try:
        response = groq_client.chat(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_message=message,
            temperature=0.0,
            max_tokens=500,
        )
        clean = response.strip().strip("```json").strip("```").strip()
        parsed = json.loads(clean)
        result = {k: v for k, v in parsed.items() if v is not None}
        if result:
            logger.info(f"Extracted: {result}")
        return result
    except Exception as e:
        logger.warning(f"Extraction parse failed: {e}")
        return {}


def get_next_question(state: BusinessState, asked_questions: list[str]) -> Optional[str]:
    """
    Returns the next question to ask based on entity type and missing fields.
    Never repeats a question already asked this session.
    """
    if state.is_ready_for_analysis():
        return None

    # Phase 1: always ask common fields first
    for field in COMMON_PRIORITY:
        value = getattr(state, field, None)
        if value is None:
            question = FIELD_QUESTIONS.get(field)
            if question and question not in asked_questions:
                return question

    # Phase 2: entity-type-specific fields
    priority = GOVERNMENT_PRIORITY if state.entity_type == "government" else CORPORATE_PRIORITY
    for field in priority:
        value = getattr(state, field, None)
        if value is None:
            question = FIELD_QUESTIONS.get(field)
            if question and question not in asked_questions:
                return question

    # Fallback
    if state.has_revenue_info() and state.has_cost_info():
        return None

    fallback = (
        "Pouvez-vous me communiquer vos recettes totales et vos charges totales "
        "pour la période analysée, en MMAD ?"
    )
    return fallback if fallback not in asked_questions else None


def should_analyze(state: BusinessState) -> bool:
    return state.is_ready_for_analysis()