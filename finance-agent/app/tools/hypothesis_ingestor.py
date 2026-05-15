"""
tools/hypothesis_ingestor.py — Bloc 1 → FinancialData translation + Bloc 2 derivation.

This is the bridge between the Hypothesis Agent and the Finance Agent.
Single responsibility: take a validated HypothesisOutput and produce:
  1. A populated FinancialData (for KPIEngine)
  2. A DerivedVariables bag (all Bloc 2 calculated values)
  3. A ProjectionInputs bag (for the 24-month plan generator)

WHY A SEPARATE MODULE:
  The Finance Agent's core (KPIEngine, ScenarioEngine) should not know
  about H-variable naming or Hypothesis Agent specifics. If the Hypothesis
  Agent changes its schema, only THIS file changes.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from app.schemas.hypothesis_output import HypothesisOutput
from app.schemas.financial_variables import FinancialData
from app.tools.fiscal_constants import get_fiscal_constants, FiscalYear

logger = logging.getLogger(__name__)


@dataclass
class DerivedVariables:
    """
    Bloc 2 variables calculated by the Finance Agent from Bloc 1 inputs.
    All monetary values in MAD (same scale as hypothesis inputs, not millions).
    """
    # Salary / HR
    salaire_brut_equipe: float = 0.0           # Net → gross
    cout_total_employeur: float = 0.0           # Gross × employer_charge_multiplier
    masse_salariale_chargee: float = 0.0        # Total monthly HR cost to company

    # Loan repayment (if H21 > 0)
    mensualite_emprunt: float = 0.0             # Monthly annuity (principal + interest)
    duree_emprunt_mois: int = 60                # Default 5 years

    # Amortization
    dotation_amortissement_mensuelle: float = 0.0

    # Break-even
    charges_fixes_mensuelles_totales: float = 0.0
    marge_brute_unitaire: float = 0.0
    taux_marge_brute: Optional[float] = None    # %
    seuil_rentabilite_clients: Optional[float] = None
    seuil_rentabilite_ca: Optional[float] = None

    # BFR
    stock_moyen_initial: float = 0.0
    creances_clients_mad: float = 0.0           # Based on delai_jours
    bfr: float = 0.0                            # Besoin en Fonds de Roulement

    # Cash needed at launch
    tresorerie_initiale_necessaire: float = 0.0
    frais_immatriculation: float = 0.0

    # Tax provisions
    taux_is_applicable: Optional[float] = None
    is_mensuel_provisionne: float = 0.0

    # Revenue structure
    ca_mensuel_mois1: float = 0.0
    ca_annuel_annee1: float = 0.0               # Simplified (no seasonality)

    # Fiscal year used
    fiscal_year: int = 2025


@dataclass
class ProjectionInputs:
    """
    Clean inputs for the 24-month scenario engine.
    No H-variable names — pure financial quantities.
    """
    # Revenue
    nb_clients_mois1: float = 0
    prix_vente_unitaire: float = 0              # Or abonnement_mensuel if subscription
    taux_croissance_mensuel: float = 0          # % per month
    taux_churn_mensuel: float = 0               # % per month (= 100 - fidelisation)
    saisonnalite: Optional[dict] = None

    # Costs
    cout_variable_unitaire: float = 0           # H9 + tech infra share
    charges_fixes_mensuelles: float = 0

    # Cash
    tresorerie_initiale: float = 0
    mensualite_emprunt: float = 0
    dotation_amortissement: float = 0

    # Derived
    marge_brute_unitaire: float = 0
    type_activite: str = "service"
    segment_client: str = "B2C"
    delai_encaissement_jours: int = 0


def ingest_hypothesis(
    hypothesis: HypothesisOutput,
    taux_interet_emprunt: float = 0.06,   # 6% default bank rate Morocco
    duree_emprunt_mois: int = 60,         # 5 years default
    fiscal_year: int = 2025,
) -> tuple[FinancialData, DerivedVariables, ProjectionInputs]:
    """
    Main entry point. Takes a validated HypothesisOutput, returns three objects:
      - FinancialData: for KPIEngine (enterprise KPIs on annual scale)
      - DerivedVariables: all Bloc 2 calculations (for transparency / reporting)
      - ProjectionInputs: for 24-month plan generator

    Raises ValueError if critical fields are missing.
    """
    fc = get_fiscal_constants(fiscal_year)
    v = hypothesis.ventes
    a = hypothesis.achats
    cf = hypothesis.charges_fixes
    enc = hypothesis.encaissements
    meta = hypothesis.metadata

    derived = DerivedVariables(fiscal_year=fiscal_year)

    # ── 1. Salary costs ──────────────────────────────────────────────────
    salaire_net = cf.H14_salaires_equipe or 0.0
    derived.salaire_brut_equipe = _net_to_gross(salaire_net, fc)
    derived.cout_total_employeur = round(
        derived.salaire_brut_equipe * fc.employer_charge_multiplier, 2
    )
    derived.masse_salariale_chargee = derived.cout_total_employeur
    logger.info(f"Masse salariale chargée: {derived.masse_salariale_chargee} MAD/mois")

    # ── 2. Loan repayment ────────────────────────────────────────────────
    if cf.H21_emprunts and cf.H21_emprunts > 0:
        derived.mensualite_emprunt = fc.compute_loan_monthly_payment(
            capital=cf.H21_emprunts,
            taux_annuel=taux_interet_emprunt,
            duree_mois=duree_emprunt_mois,
        )
        derived.duree_emprunt_mois = duree_emprunt_mois
        logger.info(f"Mensualité emprunt: {derived.mensualite_emprunt} MAD/mois")

    # ── 3. Amortization ──────────────────────────────────────────────────
    investissements = cf.H19_investissements_initiaux or 0.0
    # Default: 5-year straight-line (20%/an) — most conservative choice for mixed assets
    derived.dotation_amortissement_mensuelle = round(
        investissements * 0.20 / 12, 2
    )

    # ── 4. Fixed charges total ────────────────────────────────────────────
    charges_fixes_hors_salaires = sum(filter(None, [
        cf.H13_loyer_mensuel,
        cf.H15_charges_utilites,
        cf.H16_licences_logicielles,
        cf.H17_budget_marketing,
        cf.H18_honoraires_conseil,
        a.H11_cout_infra_numerique,
    ]))
    derived.charges_fixes_mensuelles_totales = round(
        derived.cout_total_employeur
        + charges_fixes_hors_salaires
        + derived.mensualite_emprunt
        + derived.dotation_amortissement_mensuelle,
        2
    )
    logger.info(f"Charges fixes totales: {derived.charges_fixes_mensuelles_totales} MAD/mois")

    # ── 5. Revenue and unit economics ─────────────────────────────────────
    prix = v.H2_prix_vente_unitaire or v.H3_abonnement_mensuel or 0.0
    cout_variable = a.H9_cout_fabrication_unitaire or 0.0

    derived.marge_brute_unitaire = round(prix - cout_variable, 2)
    derived.taux_marge_brute = round(
        derived.marge_brute_unitaire / prix * 100, 1
    ) if prix > 0 else None

    clients_mois1 = v.H4_nb_clients_mois1 or 0
    derived.ca_mensuel_mois1 = round(clients_mois1 * prix, 2)
    derived.ca_annuel_annee1 = round(derived.ca_mensuel_mois1 * 12, 2)  # simplified

    # ── 6. Break-even ────────────────────────────────────────────────────
    if derived.marge_brute_unitaire > 0:
        derived.seuil_rentabilite_clients = round(
            derived.charges_fixes_mensuelles_totales / derived.marge_brute_unitaire, 1
        )
        derived.seuil_rentabilite_ca = round(
            derived.seuil_rentabilite_clients * prix, 2
        )
        logger.info(
            f"Seuil rentabilité: {derived.seuil_rentabilite_clients} clients "
            f"({derived.seuil_rentabilite_ca} MAD CA/mois)"
        )

    # ── 7. BFR ───────────────────────────────────────────────────────────
    delai = enc.delai_jours if enc.H22_nature_clients != "comptant" else 0
    qte_min = a.H10_quantite_min_commande or 0
    derived.stock_moyen_initial = round(qte_min * cout_variable, 2)
    derived.creances_clients_mad = round(
        delai * (derived.ca_mensuel_mois1 / 30), 2
    ) if derived.ca_mensuel_mois1 > 0 else 0.0
    derived.bfr = round(
        derived.stock_moyen_initial + derived.creances_clients_mad, 2
    )
    logger.info(f"BFR: {derived.bfr} MAD")

    # ── 8. Initial cash needed ────────────────────────────────────────────
    frais_immat = round(
        (fc.frais_immatriculation_sarl_min + fc.frais_immatriculation_sarl_max) / 2, 0
    )
    certif = cf.H20_certifications or 0.0
    derived.frais_immatriculation = frais_immat
    derived.tresorerie_initiale_necessaire = round(
        derived.bfr
        + derived.charges_fixes_mensuelles_totales  # first month buffer
        + frais_immat
        + certif,
        2
    )

    # ── 9. Tax provision ──────────────────────────────────────────────────
    statut = (meta.statut_juridique or "SARL").upper()
    if "AUTO" in statut:
        derived.taux_is_applicable = fc.ir_auto_entrepreneur_services
    else:
        # IS will be computed on actual result — approximate with midpoint rate
        derived.taux_is_applicable = 0.20  # Most common bracket for SMEs

    # ── 10. Build FinancialData (annual, MAD) ────────────────────────────
    annual_revenue = derived.ca_annuel_annee1
    annual_costs = derived.charges_fixes_mensuelles_totales * 12
    annual_cogs = cout_variable * clients_mois1 * 12 if a.H8_type_activite == "produit" else 0
    annual_salaires = derived.cout_total_employeur * 12

    capital = meta.capital_social or 0.0
    emprunt = cf.H21_emprunts or 0.0
    investissements_total = investissements + (cf.H20_certifications or 0)

    financial_data = FinancialData(
        entity_type="corporate",
        entity_name=meta.description_projet,
        sector=meta.secteur,
        total_revenue=annual_revenue,
        cost_of_goods_sold=annual_cogs,
        operating_expenses=round(annual_costs - annual_salaires - annual_cogs, 2),
        salaries_and_benefits=annual_salaires,
        depreciation_amortization=derived.dotation_amortissement_mensuelle * 12,
        interest_expense=round(
            derived.mensualite_emprunt * 0.3 * 12, 2  # rough: 30% of payment = interest
        ),
        total_assets=round(investissements_total + derived.bfr + capital, 2),
        total_equity=capital,
        total_debt=emprunt,
        cash_inflow=annual_revenue,
        cash_outflow=annual_costs,
        own_capital_invested=capital,
        external_funding=emprunt,
        months=12,
    )

    # ── 11. ProjectionInputs ──────────────────────────────────────────────
    churn_mensuel = round(100 - (v.H6_taux_fidelisation or 97), 2)
    proj = ProjectionInputs(
        nb_clients_mois1=clients_mois1,
        prix_vente_unitaire=prix,
        taux_croissance_mensuel=v.H5_taux_croissance_mensuel or 5.0,
        taux_churn_mensuel=churn_mensuel,
        saisonnalite=v.H7_saisonnalite,
        cout_variable_unitaire=cout_variable,
        charges_fixes_mensuelles=derived.charges_fixes_mensuelles_totales,
        tresorerie_initiale=capital - derived.bfr,
        mensualite_emprunt=derived.mensualite_emprunt,
        dotation_amortissement=derived.dotation_amortissement_mensuelle,
        marge_brute_unitaire=derived.marge_brute_unitaire,
        type_activite=a.H8_type_activite,
        segment_client=v.H1_segment_client,
        delai_encaissement_jours=enc.delai_jours,
    )

    return financial_data, derived, proj


# ── Helpers ──────────────────────────────────────────────────────────────────

def _net_to_gross(salaire_net: float, fc: FiscalYear) -> float:
    """Convert net salary to gross salary."""
    if salaire_net <= 0:
        return 0.0
    deductions = fc.cnss.salarial_part + fc.amo.salariale
    gross = salaire_net / (1 - deductions)
    return round(gross, 2)


def format_derived_for_prompt(derived: DerivedVariables) -> str:
    """
    Compact summary of Bloc 2 variables for injection into LLM prompt.
    Provides grounding — agent sees the full calculation chain.
    """
    lines = [
        f"  Masse salariale chargée: {derived.cout_total_employeur:,.0f} MAD/mois "
        f"(net → brut → charges patronales × {1 + 0.2109 + 0.0411:.3f})",
        f"  Charges fixes totales: {derived.charges_fixes_mensuelles_totales:,.0f} MAD/mois",
        f"  Marge brute unitaire: {derived.marge_brute_unitaire:,.0f} MAD "
        f"({derived.taux_marge_brute:.1f}%)" if derived.taux_marge_brute else
        f"  Marge brute unitaire: {derived.marge_brute_unitaire:,.0f} MAD",
        f"  Seuil de rentabilité: {derived.seuil_rentabilite_clients:.0f} clients/mois "
        f"({derived.seuil_rentabilite_ca:,.0f} MAD CA)" if derived.seuil_rentabilite_clients else
        "  Seuil de rentabilité: non calculable (marge nulle)",
        f"  BFR: {derived.bfr:,.0f} MAD "
        f"(stock: {derived.stock_moyen_initial:,.0f} + créances: {derived.creances_clients_mad:,.0f})",
        f"  Trésorerie initiale nécessaire: {derived.tresorerie_initiale_necessaire:,.0f} MAD",
        f"  Mensualité emprunt: {derived.mensualite_emprunt:,.0f} MAD/mois" if derived.mensualite_emprunt else "",
        f"  Dotation amortissement: {derived.dotation_amortissement_mensuelle:,.0f} MAD/mois",
    ]
    return "\n".join(l for l in lines if l)