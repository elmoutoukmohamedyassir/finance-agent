"""
tools/plan_generator.py — 24-month financial plan. Pure Python, no LLM.

Generates:
  1. Compte de Résultat Prévisionnel (monthly months 1→24, then annual)
  2. Plan de Trésorerie (monthly 24 months)
  3. Plan de Financement Initial (needs vs resources at T0)
  4. Bilan Prévisionnel simplifié (year 1 & year 2)

The LLM receives this structured output and ONLY narrates/interprets it.
It never recalculates. Every number here is deterministic.

All values in MAD (not millions — these are SME-scale figures).
"""

from dataclasses import dataclass, field
from typing import Optional
from app.tools.hypothesis_ingestor import ProjectionInputs, DerivedVariables


@dataclass
class MonthlyRow:
    """One row of the compte de résultat or plan de trésorerie."""
    mois: int
    nb_clients: float
    ca: float
    cout_ventes: float
    marge_brute: float
    charges_fixes: float
    ebitda: float
    amortissements: float
    resultat_avant_is: float
    is_provision: float
    resultat_net: float
    # Trésorerie
    encaissements: float
    decaissements: float
    solde_mensuel: float
    solde_cumule: float


@dataclass
class PlanFinancement:
    """Plan de financement initial (T0 needs vs resources)."""
    # Besoins
    investissements_initiaux: float
    certifications: float
    bfr: float
    frais_immatriculation: float
    reserve_tresorerie: float
    total_besoins: float
    # Ressources
    capital_propre: float
    emprunt_bancaire: float
    aides_subventions: float
    total_ressources: float
    # Equilibre
    solde: float  # total_ressources - total_besoins


@dataclass
class BilanSimplified:
    """Bilan prévisionnel simplifié fin d'année."""
    annee: int
    # Actif
    immobilisations_nettes: float   # investissements - amortissements cumulés
    stocks: float
    creances_clients: float
    tresorerie: float
    total_actif: float
    # Passif
    capital_social: float
    reserves_resultats: float       # cumul résultats nets
    dettes_bancaires: float         # emprunts restants
    dettes_fournisseurs: float
    dettes_fiscales_sociales: float
    total_passif: float


@dataclass
class BusinessPlan24M:
    """Complete 24-month business plan output."""
    # Monthly detail
    compte_resultat: list[MonthlyRow]
    plan_tresorerie: list[MonthlyRow]  # same structure, trésorerie focus

    # Annual summaries
    annee1: dict
    annee2: dict

    # Static documents
    plan_financement: PlanFinancement
    bilan_annee1: BilanSimplified
    bilan_annee2: BilanSimplified

    # KPIs
    seuil_rentabilite_clients: Optional[float]
    mois_point_mort: Optional[int]     # Month when cumulative cash turns positive
    roi_annee1: Optional[float]
    roi_annee2: Optional[float]
    dscr_annee1: Optional[float]

    # Metadata
    scenario_name: str = "Réaliste"
    taux_is: float = 0.20
    hypotheses: dict = field(default_factory=dict)


def generate_24m_plan(
    proj: ProjectionInputs,
    derived: DerivedVariables,
    capital_propre: float = 0.0,
    emprunt: float = 0.0,
    certifications: float = 0.0,
    aides_subventions: float = 0.0,
    taux_is: float = 0.20,
    scenario_name: str = "Réaliste",
    taux_interet: float = 0.06,
) -> BusinessPlan24M:
    """
    Generate the full 24-month financial plan from projection inputs.
    Returns a BusinessPlan24M with all tables populated.
    """
    clients = proj.nb_clients_mois1
    prix = proj.prix_vente_unitaire
    growth = proj.taux_croissance_mensuel / 100
    churn = proj.taux_churn_mensuel / 100
    cout_variable = proj.cout_variable_unitaire
    charges_fixes = proj.charges_fixes_mensuelles
    amort_mensuel = proj.dotation_amortissement
    mensualite = proj.mensualite_emprunt
    delai_j = proj.delai_encaissement_jours

    # Initial cash = capital + emprunt - BFR - frais immat
    cash = capital_propre + emprunt - derived.bfr - derived.frais_immatriculation
    dette_restante = emprunt

    rows: list[MonthlyRow] = []
    cumul_amort = 0.0
    cumul_resultat = 0.0
    mois_point_mort = None

    for m in range(1, 25):
        # Revenue with optional seasonality
        coeff = 1.0
        if proj.saisonnalite and str(((m - 1) % 12) + 1) in proj.saisonnalite:
            coeff = proj.saisonnalite[str(((m - 1) % 12) + 1)]

        ca = round(clients * prix * coeff, 2)
        cout_ventes = round(clients * cout_variable * coeff, 2)
        marge_brute = round(ca - cout_ventes, 2)

        ebitda = round(marge_brute - charges_fixes, 2)
        resultat_avant_is = round(ebitda - amort_mensuel, 2)
        is_prov = round(max(0, resultat_avant_is) * taux_is, 2) if resultat_avant_is > 0 else 0.0
        resultat_net = round(resultat_avant_is - is_prov, 2)

        # Cash flow
        # Encaissements: account for payment delay
        encaissements = ca  # simplified: full CA collected (delay affects BFR, not monthly run-rate after stabilization)
        decaissements = round(charges_fixes + cout_ventes + mensualite + is_prov, 2)
        solde_mensuel = round(encaissements - decaissements, 2)
        cash = round(cash + solde_mensuel, 2)

        if cash > 0 and mois_point_mort is None and m > 1:
            mois_point_mort = m

        cumul_amort += amort_mensuel
        cumul_resultat += resultat_net
        dette_restante = round(max(0, dette_restante - mensualite * 0.7), 2)  # ~70% principal

        row = MonthlyRow(
            mois=m,
            nb_clients=round(clients, 1),
            ca=ca,
            cout_ventes=cout_ventes,
            marge_brute=marge_brute,
            charges_fixes=charges_fixes,
            ebitda=ebitda,
            amortissements=amort_mensuel,
            resultat_avant_is=resultat_avant_is,
            is_provision=is_prov,
            resultat_net=resultat_net,
            encaissements=encaissements,
            decaissements=decaissements,
            solde_mensuel=solde_mensuel,
            solde_cumule=cash,
        )
        rows.append(row)

        # Clients next month
        gained = clients * growth
        lost = clients * churn
        clients = max(0.0, clients + gained - lost)

    annee1 = _summarize_year(rows[:12], 1)
    annee2 = _summarize_year(rows[12:], 2)

    # Plan de financement
    investissements = derived.dotation_amortissement_mensuelle * 12 * 5  # reverse from annual rate
    plan_fin = _build_plan_financement(
        investissements_initiaux=investissements,
        certifications=certifications,
        bfr=derived.bfr,
        frais_immatriculation=derived.frais_immatriculation,
        capital_propre=capital_propre,
        emprunt=emprunt,
        aides_subventions=aides_subventions,
    )

    # Bilans
    immo_brut = investissements + certifications
    bilan1 = BilanSimplified(
        annee=1,
        immobilisations_nettes=round(immo_brut - sum(r.amortissements for r in rows[:12]), 2),
        stocks=derived.stock_moyen_initial,
        creances_clients=derived.creances_clients_mad,
        tresorerie=rows[11].solde_cumule,
        total_actif=0,  # filled below
        capital_social=capital_propre,
        reserves_resultats=annee1["resultat_net"],
        dettes_bancaires=dette_restante,
        dettes_fournisseurs=0,
        dettes_fiscales_sociales=annee1["is_total"],
        total_passif=0,
    )
    bilan1.total_actif = round(
        bilan1.immobilisations_nettes + bilan1.stocks + bilan1.creances_clients + bilan1.tresorerie, 2
    )
    bilan1.total_passif = round(
        bilan1.capital_social + bilan1.reserves_resultats + bilan1.dettes_bancaires + bilan1.dettes_fiscales_sociales, 2
    )

    bilan2 = BilanSimplified(
        annee=2,
        immobilisations_nettes=round(immo_brut - sum(r.amortissements for r in rows), 2),
        stocks=derived.stock_moyen_initial,
        creances_clients=derived.creances_clients_mad * 1.1,
        tresorerie=rows[23].solde_cumule,
        total_actif=0,
        capital_social=capital_propre,
        reserves_resultats=round(annee1["resultat_net"] + annee2["resultat_net"], 2),
        dettes_bancaires=round(max(0, dette_restante - mensualite * 0.7 * 12), 2),
        dettes_fournisseurs=0,
        dettes_fiscales_sociales=annee2["is_total"],
        total_passif=0,
    )
    bilan2.total_actif = round(
        bilan2.immobilisations_nettes + bilan2.stocks + bilan2.creances_clients + bilan2.tresorerie, 2
    )
    bilan2.total_passif = round(
        bilan2.capital_social + bilan2.reserves_resultats + bilan2.dettes_bancaires + bilan2.dettes_fiscales_sociales, 2
    )

    # KPIs
    roi1 = round(
        annee1["resultat_net"] / (capital_propre + emprunt) * 100, 1
    ) if (capital_propre + emprunt) > 0 else None
    roi2 = round(
        annee2["resultat_net"] / (capital_propre + emprunt) * 100, 1
    ) if (capital_propre + emprunt) > 0 else None
    dscr1 = round(
        annee1["ebitda"] / (mensualite * 12), 2
    ) if mensualite > 0 else None

    return BusinessPlan24M(
        compte_resultat=rows,
        plan_tresorerie=rows,
        annee1=annee1,
        annee2=annee2,
        plan_financement=plan_fin,
        bilan_annee1=bilan1,
        bilan_annee2=bilan2,
        seuil_rentabilite_clients=derived.seuil_rentabilite_clients,
        mois_point_mort=mois_point_mort,
        roi_annee1=roi1,
        roi_annee2=roi2,
        dscr_annee1=dscr1,
        scenario_name=scenario_name,
        taux_is=taux_is,
        hypotheses={
            "croissance_mensuelle_pct": proj.taux_croissance_mensuel,
            "churn_mensuel_pct": proj.taux_churn_mensuel,
            "prix_vente": prix,
            "charges_fixes_mensuelles": charges_fixes,
        },
    )


def _summarize_year(rows: list[MonthlyRow], annee: int) -> dict:
    return {
        "annee": annee,
        "ca_total": round(sum(r.ca for r in rows), 2),
        "cout_ventes_total": round(sum(r.cout_ventes for r in rows), 2),
        "marge_brute": round(sum(r.marge_brute for r in rows), 2),
        "charges_fixes_total": round(sum(r.charges_fixes for r in rows), 2),
        "ebitda": round(sum(r.ebitda for r in rows), 2),
        "amortissements_total": round(sum(r.amortissements for r in rows), 2),
        "resultat_avant_is": round(sum(r.resultat_avant_is for r in rows), 2),
        "is_total": round(sum(r.is_provision for r in rows), 2),
        "resultat_net": round(sum(r.resultat_net for r in rows), 2),
        "encaissements_total": round(sum(r.encaissements for r in rows), 2),
        "decaissements_total": round(sum(r.decaissements for r in rows), 2),
        "tresorerie_fin": rows[-1].solde_cumule,
        "nb_clients_fin": rows[-1].nb_clients,
        "marge_nette_pct": round(
            sum(r.resultat_net for r in rows) / sum(r.ca for r in rows) * 100, 1
        ) if sum(r.ca for r in rows) > 0 else None,
    }


def _build_plan_financement(
    investissements_initiaux: float,
    certifications: float,
    bfr: float,
    frais_immatriculation: float,
    capital_propre: float,
    emprunt: float,
    aides_subventions: float,
    reserve: float = 0.0,
) -> PlanFinancement:
    total_besoins = round(
        investissements_initiaux + certifications + bfr + frais_immatriculation + reserve, 2
    )
    total_ressources = round(capital_propre + emprunt + aides_subventions, 2)
    return PlanFinancement(
        investissements_initiaux=investissements_initiaux,
        certifications=certifications,
        bfr=bfr,
        frais_immatriculation=frais_immatriculation,
        reserve_tresorerie=reserve,
        total_besoins=total_besoins,
        capital_propre=capital_propre,
        emprunt_bancaire=emprunt,
        aides_subventions=aides_subventions,
        total_ressources=total_ressources,
        solde=round(total_ressources - total_besoins, 2),
    )


def format_plan_for_prompt(plan: BusinessPlan24M) -> str:
    """
    Compact summary of the 24-month plan for LLM injection.
    The LLM sees the key numbers — not 24 monthly rows.
    """
    a1, a2 = plan.annee1, plan.annee2
    fin = plan.plan_financement
    lines = [
        f"PLAN DE FINANCEMENT INITIAL:",
        f"  Besoins totaux: {fin.total_besoins:,.0f} MAD | Ressources: {fin.total_ressources:,.0f} MAD | Solde: {fin.solde:,.0f} MAD",
        "",
        f"COMPTE DE RÉSULTAT (Scénario {plan.scenario_name}):",
        f"  Année 1 — CA: {a1['ca_total']:,.0f} MAD | Marge brute: {a1['marge_brute']:,.0f} MAD | EBITDA: {a1['ebitda']:,.0f} MAD | Résultat net: {a1['resultat_net']:,.0f} MAD ({a1.get('marge_nette_pct', 'N/A')}%)",
        f"  Année 2 — CA: {a2['ca_total']:,.0f} MAD | Marge brute: {a2['marge_brute']:,.0f} MAD | EBITDA: {a2['ebitda']:,.0f} MAD | Résultat net: {a2['resultat_net']:,.0f} MAD ({a2.get('marge_nette_pct', 'N/A')}%)",
        "",
        f"TRÉSORERIE:",
        f"  Fin année 1: {a1['tresorerie_fin']:,.0f} MAD | Fin année 2: {a2['tresorerie_fin']:,.0f} MAD",
        "",
        f"KPIs CLÉS:",
        f"  Seuil rentabilité: {plan.seuil_rentabilite_clients:.0f} clients/mois" if plan.seuil_rentabilite_clients else "  Seuil rentabilité: N/D",
        f"  Point mort (mois): {plan.mois_point_mort or 'Non atteint sur 24 mois'}",
        f"  ROI an1: {plan.roi_annee1:.1f}% | ROI an2: {plan.roi_annee2:.1f}%" if plan.roi_annee1 else "  ROI: N/D",
        f"  DSCR an1: {plan.dscr_annee1:.2f}x" if plan.dscr_annee1 else "",
    ]
    return "\n".join(l for l in lines if l is not None)