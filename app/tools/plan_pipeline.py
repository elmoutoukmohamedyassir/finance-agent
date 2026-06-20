"""
tools/plan_pipeline.py — single source of truth for turning a collected
business_state dict into computed financial plan numbers.

This was extracted out of phase3_analysis_agent.py so that the chat agent
and the PDF-export endpoint (api/routers/chat.py) call the exact same
mapping + calculation code. Two independent copies of this logic is how
the H22_nature_clients mapping bug happened in the first place — keep it
in one place.
"""
from dataclasses import dataclass
from typing import Optional

from app.schemas.hypothesis_output import (
    HypothesisOutput, BlocVentes, BlocAchats, BlocChargesFixes,
    BlocEncaissements, HypothesisMetadata,
)
from app.tools.hypothesis_ingestor import ingest_hypothesis, format_derived_for_prompt
from app.tools.plan_generator import generate_24m_plan, format_plan_for_prompt

# Same set phase_router.py uses to decide a session is ready for phase3 —
# kept here too because compute_plan() below does NOT reliably fail on
# missing data (it falls back to 0/None defaults and produces a degenerate
# all-zero plan instead of raising). Callers that need to know "is there
# really enough data" should check has_minimum_data() explicitly, not just
# "did compute_plan() return non-None".
MINIMUM_FIELDS = {
    "entity_type", "segment_client", "prix_vente_unitaire",
    "nb_clients_mois1", "taux_croissance_mensuel",
    "loyer_mensuel", "salaires_equipe", "investissements_initiaux",
}


def has_minimum_data(business_state: dict) -> bool:
    filled = {k for k, v in (business_state or {}).items() if v is not None and v != ""}
    return MINIMUM_FIELDS.issubset(filled)


def nature_clients_to_literal(value) -> str:
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


def business_state_to_hypothesis(bs: dict) -> HypothesisOutput:
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
            H22_nature_clients=nature_clients_to_literal(g("nature_clients_encaissements")),
            delai_jours=int(g("delai_jours") or 0),
        ),
        metadata=HypothesisMetadata(
            description_projet=g("entity_name"),
            secteur=g("sector"),
            statut_juridique=g("statut_juridique") or "Auto-entrepreneur",
            capital_social=g("own_capital_invested"),
        ),
    )


@dataclass
class ComputedPlan:
    financial_data: object
    derived: object
    proj_inputs: object
    plan: object
    derived_summary: str
    plan_summary: str


def compute_plan(business_state: dict, fiscal_year: int = 2025) -> Optional[ComputedPlan]:
    """
    Run the real deterministic engine on a business_state dict.
    Returns None (never raises) if required fields are missing/invalid —
    callers should treat None as "not enough data yet", not an error.
    """
    try:
        hypothesis = business_state_to_hypothesis(business_state)
        financial_data, derived, proj_inputs = ingest_hypothesis(hypothesis, fiscal_year=fiscal_year)
        derived_summary = format_derived_for_prompt(derived)
        plan = generate_24m_plan(
            proj=proj_inputs,
            derived=derived,
            capital_propre=business_state.get("own_capital_invested") or 0,
            emprunt=business_state.get("emprunts") or 0,
        )
        plan_summary = format_plan_for_prompt(plan)
        return ComputedPlan(
            financial_data=financial_data,
            derived=derived,
            proj_inputs=proj_inputs,
            plan=plan,
            derived_summary=derived_summary,
            plan_summary=plan_summary,
        )
    except Exception:
        return None


def format_full_plan_tables(plan) -> str:
    """
    The complete plan financier as plain text: plan de financement, compte de
    résultat (2 ans), plan de trésorerie, bilan simplifié (2 ans), KPIs.
    Used for the in-chat "full business plan" text response. The PDF export
    (tools/plan_pdf.py) renders the same numbers as proper tables instead.
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