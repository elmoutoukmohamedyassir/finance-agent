"""
test_agent.py — Test every layer of the Enterprise Finance Agent.

Usage:
    python test_agent.py              # run all offline tests (no server needed)
    python test_agent.py fiscal       # fiscal constants + IS calculations
    python test_agent.py metrics      # KPI engine (corporate + government)
    python test_agent.py scenarios    # scenario engine
    python test_agent.py hypothesis   # Hypothesis Agent ingestion (Bloc 1 → Bloc 2)
    python test_agent.py plan         # 24-month plan generator
    python test_agent.py session      # session + BusinessState
    python test_agent.py embedder     # Ollama/sentence-transformers connection
    python test_agent.py rag          # RAG retrieval quality
    python test_agent.py chat         # full HTTP chat flow (server must be running)
    python test_agent.py curl         # print curl command reference

Start server first for chat tests:
    uvicorn app.main:app --reload
"""

import sys
import json
import logging

logging.basicConfig(level=logging.WARNING)

PASS = "  ✓ PASSED"
FAIL = "  ✗ FAILED"


def header(title):
    print(f"\n{'=' * 60}")
    print(f"TEST: {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — Fiscal Constants
# ─────────────────────────────────────────────────────────────────────────────

def test_fiscal():
    header("Fiscal Constants & IS Calculator (no LLM, no network)")

    from app.tools.fiscal_constants import get_fiscal_constants

    fc = get_fiscal_constants(2025)
    print(f"  Fiscal year loaded : {fc.year}")
    print(f"  SMIG mensuel       : {fc.smig_mensuel:,.2f} MAD")
    print(f"  CNSS patronal      : {fc.cnss.patronal_part * 100:.2f}%")
    print(f"  CNSS salarial      : {fc.cnss.salarial_part * 100:.2f}%")
    print(f"  AMO patronale      : {fc.amo.patronale * 100:.2f}%")
    print(f"  Employer multiplier: ×{fc.employer_charge_multiplier}")
    print(f"  TVA standard       : {fc.tva.standard * 100:.0f}%")

    print("\n  IS calculations (progressive brackets):")
    test_cases = [
        (200_000,  "< 300K MAD bracket"),
        (500_000,  "straddles 300K bracket"),
        (2_000_000, "middle bracket"),
        (150_000_000, "top bracket"),
    ]
    for profit, label in test_cases:
        is_due = fc.compute_is(profit)
        effective = is_due / profit * 100
        print(f"    Profit {profit:>15,.0f} MAD → IS {is_due:>12,.2f} MAD  ({effective:.1f}% effective) [{label}]")

    # Assertions
    assert fc.smig_mensuel == 3_111.39
    assert fc.employer_charge_multiplier == 1.252
    assert fc.compute_is(0) == 0
    assert fc.compute_is(200_000) == 20_000.0   # 10% of 200K

    print("\n  Loan monthly payment (6%, 60 months, 100 000 MAD):")
    m = fc.compute_loan_monthly_payment(100_000, 0.06, 60)
    print(f"    Mensualité : {m:,.2f} MAD/mois")
    assert 1900 < m < 2000, f"Expected ~1933 MAD, got {m}"

    print("\n  Salary gross → total employer cost:")
    net = 10_000
    gross = net / (1 - fc.cnss.salarial_part - fc.amo.salariale)
    total = fc.compute_employer_salary_cost(net)
    print(f"    Net 10 000 MAD → Gross {gross:,.2f} → Total employer {total:,.2f} MAD")
    assert total > net * 1.3, "Employer cost should be >30% more than net"

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — KPI Engine: Corporate
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_corporate():
    header("KPI Engine — Corporate Entity")

    from app.schemas.metrics import MetricsInput
    from app.tools.metrics_calculator import calculate_metrics

    # Simulate a medium Moroccan industrial company
    inputs = MetricsInput(
        entity_type="corporate",
        entity_name="Groupe Manufac SA",
        sector="industrie manufacturière",
        total_revenue=87.5,
        revenue_year2=80.0,
        cost_of_goods_sold=42.0,
        operating_expenses=12.0,
        salaries_and_benefits=8.5,
        depreciation_amortization=3.2,
        interest_expense=1.8,
        tax_expense=4.5,
        total_assets=220.0,
        current_assets=45.0,
        current_liabilities=28.0,
        total_equity=110.0,
        total_debt=65.0,
        cash_inflow=95.0,
        cash_outflow=88.0,
        operating_cash_flow=15.0,
        own_capital_invested=30.0,
        external_funding=20.0,
    )

    result = calculate_metrics(inputs)

    print(f"  Entité            : {inputs.entity_name}")
    print(f"  Revenus           : {result.total_revenue_mad_m} MMAD")
    print(f"  Croissance revenus: {result.revenue_growth_rate_pct}%")
    print(f"  Marge brute       : {result.gross_profit_margin_pct}%")
    print(f"  EBITDA            : {result.ebitda_mad_m} MMAD ({result.ebitda_margin_pct}%)")
    print(f"  Marge opérat.     : {result.operating_margin_pct}%")
    print(f"  Résultat net      : {result.net_profit_mad_m} MMAD ({result.net_profit_margin_pct}%)")
    print(f"  Dette/Fonds propres: {result.debt_to_equity}x")
    print(f"  Ratio courant     : {result.current_ratio}x")
    print(f"  ROE               : {result.return_on_equity_pct}%")
    print(f"  ROA               : {result.return_on_assets_pct}%")
    print(f"  Cash flow net     : {result.cash_flow_net_mad_m} MMAD")
    print(f"  Score santé       : {result.health_score}")

    if result.warnings:
        print("\n  Alertes:")
        for w in result.warnings:
            print(f"    ⚠ {w[:100]}...")
    print("\n  Statuts indicateurs:")
    for k, v in result.statuses.items():
        print(f"    {k:<30} {v}")

    # Basic assertions
    assert result.total_revenue_mad_m == 87.5
    assert result.gross_profit_margin_pct is not None
    assert result.ebitda_mad_m is not None
    assert result.health_score is not None
    # Revenue growth: (87.5 - 80) / 80 * 100 = 9.375%
    assert abs(result.revenue_growth_rate_pct - 9.375) < 0.1

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — KPI Engine: Government
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics_government():
    header("KPI Engine — Government / Public Entity")

    from app.schemas.metrics import MetricsInput
    from app.tools.metrics_calculator import calculate_metrics

    # Simulate a Moroccan municipality budget
    inputs = MetricsInput(
        entity_type="government",
        entity_name="Commune de Settat",
        sector="collectivité territoriale",
        total_revenue=0,           # will be derived from sub-items
        tax_revenue=180.0,
        non_tax_revenue=45.0,
        grants_and_transfers=120.0,
        recurrent_expenditure=240.0,
        capital_expenditure=80.0,
        total_expenditure=320.0,
        debt_service=25.0,
        subsidies_paid=15.0,
        salaries_and_benefits=130.0,
        total_debt=200.0,
        cash_and_equivalents=35.0,
        investment_budget=100.0,
        investment_executed=80.0,
        months=12,
    )

    result = calculate_metrics(inputs)

    print(f"  Entité                    : {inputs.entity_name}")
    print(f"  Recettes totales          : {result.total_revenue_mad_m} MMAD")
    print(f"  Solde budgétaire primaire : {result.primary_balance_mad_m} MMAD")
    print(f"  Solde global              : {result.overall_balance_mad_m} MMAD")
    print(f"  Pression fiscale          : {result.fiscal_pressure_pct}%")
    print(f"  Ratio CAPEX               : {result.capex_ratio_pct}%")
    print(f"  Ratio subventions         : {result.subsidies_ratio_pct}%")
    print(f"  Taux exécution budgétaire : {result.budget_execution_rate_pct}%")
    print(f"  Ratio masse salariale     : {result.salary_ratio_pct}%")
    print(f"  DSCR                      : {result.debt_service_coverage}x")
    print(f"  Score santé               : {result.health_score}")

    if result.warnings:
        print("\n  Alertes:")
        for w in result.warnings:
            print(f"    ⚠ {w[:100]}...")

    assert result.total_revenue_mad_m == 345.0  # 180 + 45 + 120
    assert result.budget_execution_rate_pct == 80.0  # 80/100 * 100
    assert result.primary_balance_mad_m is not None

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — Scenario Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_scenarios():
    header("Scenario Engine — 3-year projections (no LLM)")

    from app.tools.scenario_engine import build_standard_scenarios, format_scenarios_for_prompt

    print("  [Corporate — 3 scenarios × 3 years]")
    scenarios = build_standard_scenarios(
        starting_revenue=87.5,
        starting_costs=62.0,
        starting_cash=8.0,
        starting_debt=35.0,
        debt_service_annual=5.0,
        capex_annual=4.0,
        years=3,
        entity_type="corporate",
    )
    for s in scenarios:
        summ = s["summary"]
        print(f"\n  [{s['name']}]")
        for yr in s["yearly_projections"]:
            print(f"    An {yr['year']}: CA {yr['revenue_mad_m']:.1f} MMAD | "
                  f"EBITDA {yr['ebitda_mad_m']:.1f} MMAD ({yr['ebitda_margin_pct']:.1f}%) | "
                  f"Tréso {yr['cumulative_cash_mad_m']:.1f} MMAD")
        print(f"    DSCR an3: {s['yearly_projections'][-1]['dscr']}")

    print("\n\n  [Government — 3 scenarios × 3 years]")
    gov_scenarios = build_standard_scenarios(
        starting_revenue=345.0,
        starting_costs=320.0,
        starting_cash=35.0,
        starting_debt=200.0,
        debt_service_annual=25.0,
        capex_annual=80.0,
        years=3,
        entity_type="government",
    )
    for s in gov_scenarios:
        summ = s["summary"]
        print(f"  [{s['name']}] Recettes an3: {summ['final_revenue_mad_m']:.1f} MMAD | "
              f"EBITDA: {summ['final_ebitda_mad_m']:.1f} MMAD | "
              f"Tréso: {summ['final_cash_mad_m']:.1f} MMAD")

    print("\n  Prompt format:")
    print(format_scenarios_for_prompt(scenarios, years=3))

    assert len(scenarios) == 3
    assert scenarios[0]["name"] == "Pessimiste"
    assert scenarios[2]["name"] == "Optimiste"
    assert len(scenarios[0]["yearly_projections"]) == 3

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — Hypothesis Agent Ingestion (Bloc 1 → Bloc 2)
# ─────────────────────────────────────────────────────────────────────────────

def test_hypothesis_ingestion():
    header("Hypothesis Agent → Finance Agent Ingestion (Bloc 1 + Bloc 2 derivation)")

    from app.schemas.hypothesis_output import (
        HypothesisOutput, BlocVentes, BlocAchats, BlocChargesFixes,
        BlocEncaissements, HypothesisMetadata
    )
    from app.tools.hypothesis_ingestor import ingest_hypothesis, format_derived_for_prompt

    # Simulate a B2B consulting firm in Casablanca
    hypothesis = HypothesisOutput(
        ventes=BlocVentes(
            H1_segment_client="B2B",
            H2_prix_vente_unitaire=2500.0,
            H4_nb_clients_mois1=3.0,
            H5_taux_croissance_mensuel=8.0,
            H6_taux_fidelisation=85.0,
        ),
        achats=BlocAchats(
            H8_type_activite="service",
            H9_cout_fabrication_unitaire=0.0,
            H11_cout_infra_numerique=800.0,
        ),
        charges_fixes=BlocChargesFixes(
            H13_loyer_mensuel=4500.0,
            H14_salaires_equipe=18000.0,
            H15_charges_utilites=600.0,
            H16_licences_logicielles=400.0,
            H17_budget_marketing=3000.0,
            H18_honoraires_conseil=1500.0,
            H19_investissements_initiaux=45000.0,
            H21_emprunts=100000.0,
        ),
        encaissements=BlocEncaissements(
            H22_nature_clients="credit",
            delai_jours=30,
        ),
        metadata=HypothesisMetadata(
            description_projet="Agence de consulting RH pour PME marocaines",
            region="Casablanca",
            secteur="conseil / services aux entreprises",
            statut_juridique="SARL",
            capital_social=50000.0,
        ),
    )

    financial_data, derived, proj_inputs = ingest_hypothesis(hypothesis, fiscal_year=2025)

    print("  Derived variables (Bloc 2):")
    print(format_derived_for_prompt(derived))

    print(f"\n  Masse salariale chargée : {derived.cout_total_employeur:,.0f} MAD/mois")
    print(f"  Mensualité emprunt      : {derived.mensualite_emprunt:,.0f} MAD/mois")
    print(f"  Dotation amortissement  : {derived.dotation_amortissement_mensuelle:,.0f} MAD/mois")
    print(f"  Charges fixes totales   : {derived.charges_fixes_mensuelles_totales:,.0f} MAD/mois")
    print(f"  Marge brute unitaire    : {derived.marge_brute_unitaire:,.0f} MAD")
    print(f"  Seuil rentabilité       : {derived.seuil_rentabilite_clients:.1f} clients/mois")
    print(f"  BFR                     : {derived.bfr:,.0f} MAD")
    print(f"  Trésorerie initiale nec.: {derived.tresorerie_initiale_necessaire:,.0f} MAD")

    print(f"\n  FinancialData (annual):")
    print(f"    CA annuel an1         : {financial_data.total_revenue:,.0f} MAD")
    print(f"    Masse salariale/an    : {financial_data.salaries_and_benefits:,.0f} MAD")

    print(f"\n  ProjectionInputs:")
    print(f"    Clients mois 1        : {proj_inputs.nb_clients_mois1}")
    print(f"    Prix vente            : {proj_inputs.prix_vente_unitaire} MAD")
    print(f"    Croissance mensuelle  : {proj_inputs.taux_croissance_mensuel}%")
    print(f"    Churn mensuel         : {proj_inputs.taux_churn_mensuel}%")

    # Assertions
    assert derived.cout_total_employeur > 18000, "Employer cost must exceed net salary"
    assert derived.mensualite_emprunt > 0, "Loan payment should be calculated"
    assert derived.seuil_rentabilite_clients is not None
    assert derived.seuil_rentabilite_clients > 0
    assert proj_inputs.taux_churn_mensuel == 15.0  # 100 - 85 fidelisation

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6 — 24-Month Plan Generator
# ─────────────────────────────────────────────────────────────────────────────

def test_plan_generator():
    header("24-Month Financial Plan Generator (no LLM)")

    from app.schemas.hypothesis_output import (
        HypothesisOutput, BlocVentes, BlocAchats, BlocChargesFixes,
        BlocEncaissements, HypothesisMetadata
    )
    from app.tools.hypothesis_ingestor import ingest_hypothesis
    from app.tools.plan_generator import generate_24m_plan, format_plan_for_prompt

    hypothesis = HypothesisOutput(
        ventes=BlocVentes(
            H1_segment_client="B2B",
            H2_prix_vente_unitaire=2500.0,
            H4_nb_clients_mois1=3.0,
            H5_taux_croissance_mensuel=8.0,
            H6_taux_fidelisation=85.0,
        ),
        achats=BlocAchats(H8_type_activite="service"),
        charges_fixes=BlocChargesFixes(
            H13_loyer_mensuel=4500.0,
            H14_salaires_equipe=18000.0,
            H15_charges_utilites=600.0,
            H17_budget_marketing=3000.0,
            H19_investissements_initiaux=45000.0,
            H21_emprunts=100000.0,
        ),
        encaissements=BlocEncaissements(H22_nature_clients="credit", delai_jours=30),
        metadata=HypothesisMetadata(
            secteur="conseil",
            region="Casablanca",
            statut_juridique="SARL",
            capital_social=50000.0,
        ),
    )

    _, derived, proj_inputs = ingest_hypothesis(hypothesis, fiscal_year=2025)

    plan = generate_24m_plan(
        proj=proj_inputs,
        derived=derived,
        capital_propre=50000.0,
        emprunt=100000.0,
        taux_is=0.20,
        scenario_name="Réaliste",
    )

    print("  Plan de financement initial:")
    fin = plan.plan_financement
    print(f"    Besoins : {fin.total_besoins:,.0f} MAD")
    print(f"    Ressources: {fin.total_ressources:,.0f} MAD")
    print(f"    Solde   : {fin.solde:,.0f} MAD  {'✓ équilibré' if fin.solde >= 0 else '✗ déficit'}")

    print(f"\n  Compte de résultat (extrait):")
    print(f"  {'Mois':>4} {'Clients':>8} {'CA (MAD)':>12} {'EBITDA':>12} {'Résultat net':>14} {'Tréso cum.':>12}")
    for row in plan.compte_resultat[::4]:  # every 4 months
        print(f"  {row.mois:>4} {row.nb_clients:>8.1f} {row.ca:>12,.0f} {row.ebitda:>12,.0f} {row.resultat_net:>14,.0f} {row.solde_cumule:>12,.0f}")

    print(f"\n  Résumé annuel:")
    a1, a2 = plan.annee1, plan.annee2
    print(f"    Année 1 — CA: {a1['ca_total']:,.0f} MAD | EBITDA: {a1['ebitda']:,.0f} MAD | Résultat net: {a1['resultat_net']:,.0f} MAD | Marge: {a1.get('marge_nette_pct', 'N/A')}%")
    print(f"    Année 2 — CA: {a2['ca_total']:,.0f} MAD | EBITDA: {a2['ebitda']:,.0f} MAD | Résultat net: {a2['resultat_net']:,.0f} MAD | Marge: {a2.get('marge_nette_pct', 'N/A')}%")

    print(f"\n  KPIs clés:")
    print(f"    Seuil rentabilité  : {plan.seuil_rentabilite_clients:.0f} clients/mois" if plan.seuil_rentabilite_clients else "    Seuil rentabilité : N/D")
    print(f"    Point mort (mois)  : {plan.mois_point_mort or 'Non atteint sur 24 mois'}")
    print(f"    ROI an1            : {plan.roi_annee1}%")
    print(f"    ROI an2            : {plan.roi_annee2}%")
    print(f"    DSCR an1           : {plan.dscr_annee1}x" if plan.dscr_annee1 else "    DSCR an1           : N/D")

    print(f"\n  Prompt summary:\n{format_plan_for_prompt(plan)}")

    assert len(plan.compte_resultat) == 24
    assert plan.annee1["ca_total"] > 0
    assert plan.bilan_annee1.total_actif > 0

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 7 — Session + BusinessState
# ─────────────────────────────────────────────────────────────────────────────

def test_session():
    header("Session & BusinessState")

    from app.schemas.session import BusinessState

    # Corporate
    state = BusinessState()
    assert not state.is_ready_for_analysis(), "Empty state should not be ready"
    state.entity_type = "corporate"
    assert not state.is_ready_for_analysis(), "entity_type alone not enough"
    state.total_revenue = 87.5
    assert not state.is_ready_for_analysis(), "Revenue alone not enough"
    state.operating_expenses = 12.0
    state.cost_of_goods_sold = 42.0
    assert state.is_ready_for_analysis(), "Revenue + costs → ready"
    print("  Corporate state progression ✓")

    # Government
    gov_state = BusinessState()
    gov_state.entity_type = "government"
    gov_state.tax_revenue = 180.0
    gov_state.non_tax_revenue = 45.0
    assert not gov_state.is_ready_for_analysis(), "Receipts alone not enough"
    gov_state.recurrent_expenditure = 240.0
    gov_state.capital_expenditure = 80.0
    assert gov_state.is_ready_for_analysis(), "Receipts + expenditures → ready"
    print("  Government state progression ✓")

    filled = state.filled_fields()
    assert "total_revenue" in filled
    assert "entity_type" in filled
    print(f"  filled_fields() returns {len(filled)} fields ✓")

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 8 — Embedder
# ─────────────────────────────────────────────────────────────────────────────

def test_embedder():
    header("Embedder Connection (Ollama or sentence-transformers)")

    from app.core.config import get_settings
    settings = get_settings()

    print(f"  Backend : {settings.embedding_backend}")
    print(f"  Model   : {settings.embedding_model}")
    print(f"  Ollama  : {settings.ollama_base_url}")

    try:
        from app.rag.embedder import get_embedder
        embedder = get_embedder()
        vec = embedder.embed_one("analyse financière entreprise Maroc")
        assert len(vec) > 0
        print(f"  Dimensions : {len(vec)}")
        print(f"  Sample     : {[round(v, 4) for v in vec[:4]]}...")
        print(PASS)
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        print(f"\n  If using Ollama: ollama serve && ollama pull {settings.embedding_model}")
        print(f"  Or switch: EMBEDDING_BACKEND=sentence-transformers in .env")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 9 — RAG Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_rag():
    header("RAG Retrieval — Moroccan financial documents")

    from app.rag.retriever import retrieve_raw, retrieve_context

    queries = [
        "taux IS impôt sur les sociétés Maroc 2025",
        "recettes fiscales TVA budget général",
        "taux d'exécution budgétaire investissement public",
        "marge brute entreprise industrielle Maroc",
        "CNSS cotisations patronales 2024",
    ]

    any_results = False
    for q in queries:
        print(f"\n  Query: '{q}'")
        results = retrieve_raw(q, top_k=2)
        if results:
            any_results = True
            for r in results:
                print(f"    [{r['similarity']:.3f}] {r['source']}: {r['text'][:90]}...")
        else:
            print("    (no results — run: python ingest.py)")

    if not any_results:
        print("\n  ⚠ No documents indexed yet.")
        print("  Run: python ingest.py")
        print("  Then re-run this test.")
    else:
        print(f"\n  Context sample (used in LLM prompt):")
        ctx = retrieve_context("finances publiques budget Maroc 2025")
        print(f"  {ctx[:300]}..." if len(ctx) > 300 else f"  {ctx}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 10 — Full HTTP Chat Flow (server must be running)
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_http():
    header("Full Chat Flow via HTTP (uvicorn app.main:app --reload)")

    import requests
    BASE = "http://localhost:8000/api/v1"

    # ── Test A: conversational corporate flow ────────────────────────────
    print("\n  [A] Conversational corporate flow")
    session_id = None
    turns = [
        "Bonjour, je veux analyser les finances de mon entreprise",
        "C'est une SARL dans l'agroalimentaire à Casablanca",
        "Notre CA annuel est de 45 MMAD",
        "Nos charges d'exploitation sont 18 MMAD, masse salariale 8 MMAD, COGS 22 MMAD",
        "Nous avons 12 MMAD de dettes et 30 MMAD de capitaux propres",
        "Actif circulant 15 MMAD, passif circulant 9 MMAD",
    ]
    for i, msg in enumerate(turns):
        try:
            r = requests.post(
                f"{BASE}/chat",
                json={"session_id": session_id, "message": msg},
                timeout=45,
            )
            r.raise_for_status()
            data = r.json()
            session_id = data["session_id"]
            print(f"  Turn {i+1} [{data['agent_mode']}]: {data['message'][:150]}{'...' if len(data['message']) > 150 else ''}")
            if data.get("metrics_calculated"):
                print(f"    → Métriques calculées: {list(data['metrics_calculated'].keys())[:5]}")
        except requests.exceptions.ConnectionError:
            print("  ✗ Serveur non démarré. Lancer: uvicorn app.main:app --reload")
            return
        except Exception as e:
            print(f"  ✗ Erreur: {e}")
            return

    # ── Test B: direct hypothesis payload ────────────────────────────────
    print("\n  [B] Direct HypothesisOutput ingestion")
    try:
        payload = {
            "message": "Analyse complète du projet",
            "hypothesis_payload": {
                "ventes": {
                    "H1_segment_client": "B2B",
                    "H2_prix_vente_unitaire": 2500,
                    "H4_nb_clients_mois1": 5,
                    "H5_taux_croissance_mensuel": 10,
                    "H6_taux_fidelisation": 88
                },
                "achats": {"H8_type_activite": "service", "H11_cout_infra_numerique": 800},
                "charges_fixes": {
                    "H13_loyer_mensuel": 4500,
                    "H14_salaires_equipe": 20000,
                    "H15_charges_utilites": 700,
                    "H17_budget_marketing": 3000,
                    "H19_investissements_initiaux": 50000,
                    "H21_emprunts": 120000
                },
                "encaissements": {"H22_nature_clients": "credit", "delai_jours": 30},
                "metadata": {
                    "description_projet": "Agence digitale B2B",
                    "region": "Rabat",
                    "secteur": "services numériques",
                    "statut_juridique": "SARL",
                    "capital_social": 60000
                }
            }
        }
        r = requests.post(f"{BASE}/chat", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        print(f"  Mode: {data['agent_mode']}")
        print(f"  Réponse: {data['message'][:300]}...")
        if data.get("metrics_calculated"):
            print(f"  Métriques: {list(data['metrics_calculated'].keys())}")
    except requests.exceptions.ConnectionError:
        print("  ✗ Serveur non démarré.")
    except Exception as e:
        print(f"  ✗ Erreur: {e}")

    print(PASS)


# ─────────────────────────────────────────────────────────────────────────────
# CURL REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

CURL_COMMANDS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURL COMMANDS — Enterprise Finance Agent API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── Health ────────────────────────────────────────────────
curl http://localhost:8000/
curl http://localhost:8000/health

# ── Chat: start conversation (no session_id) ─────────────
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Je veux analyser les finances de ma SARL"}'

# ── Chat: continue (paste session_id from above) ─────────
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"session_id": "PASTE_ID", "message": "Notre CA est 45 MMAD"}'

# ── Chat: direct Hypothesis Agent payload ────────────────
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{
    "message": "Analyse complète",
    "hypothesis_payload": {
      "ventes": {"H1_segment_client": "B2B", "H2_prix_vente_unitaire": 2500,
                 "H4_nb_clients_mois1": 5, "H5_taux_croissance_mensuel": 10,
                 "H6_taux_fidelisation": 88},
      "achats": {"H8_type_activite": "service"},
      "charges_fixes": {"H13_loyer_mensuel": 4500, "H14_salaires_equipe": 20000,
                        "H19_investissements_initiaux": 50000, "H21_emprunts": 120000},
      "encaissements": {"H22_nature_clients": "credit", "delai_jours": 30},
      "metadata": {"secteur": "conseil", "region": "Casablanca",
                   "statut_juridique": "SARL", "capital_social": 60000}
    }
  }'

# ── Metrics: corporate ────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/metrics/calculate \\
  -H "Content-Type: application/json" \\
  -d '{
    "entity_type": "corporate",
    "entity_name": "Groupe Manufac SA",
    "total_revenue": 87.5,
    "cost_of_goods_sold": 42.0,
    "operating_expenses": 12.0,
    "salaries_and_benefits": 8.5,
    "depreciation_amortization": 3.2,
    "interest_expense": 1.8,
    "tax_expense": 4.5,
    "total_assets": 220.0,
    "current_assets": 45.0,
    "current_liabilities": 28.0,
    "total_equity": 110.0,
    "total_debt": 65.0,
    "cash_inflow": 95.0,
    "cash_outflow": 88.0
  }'

# ── Metrics: government ───────────────────────────────────
curl -X POST http://localhost:8000/api/v1/metrics/calculate \\
  -H "Content-Type: application/json" \\
  -d '{
    "entity_type": "government",
    "entity_name": "Commune de Settat",
    "tax_revenue": 180.0,
    "non_tax_revenue": 45.0,
    "grants_and_transfers": 120.0,
    "recurrent_expenditure": 240.0,
    "capital_expenditure": 80.0,
    "total_expenditure": 320.0,
    "debt_service": 25.0,
    "salaries_and_benefits": 130.0,
    "investment_budget": 100.0,
    "investment_executed": 80.0
  }'

# ── Metrics: benchmarks reference ────────────────────────
curl http://localhost:8000/api/v1/metrics/benchmarks

# ── Scenario: corporate 3-year ────────────────────────────
curl -X POST http://localhost:8000/api/v1/scenario/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "entity_type": "corporate",
    "starting_revenue": 87.5,
    "starting_costs": 62.0,
    "starting_cash": 8.0,
    "starting_debt": 35.0,
    "debt_service_annual": 5.0,
    "capex_annual": 4.0,
    "years": 3
  }'

# ── Scenario: government 3-year ───────────────────────────
curl -X POST http://localhost:8000/api/v1/scenario/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "entity_type": "government",
    "starting_revenue": 345.0,
    "starting_costs": 320.0,
    "starting_cash": 35.0,
    "starting_debt": 200.0,
    "debt_service_annual": 25.0,
    "capex_annual": 80.0,
    "years": 3
  }'

# ── RAG: ingest docs ──────────────────────────────────────
curl -X POST http://localhost:8000/api/v1/rag/ingest
curl -X POST "http://localhost:8000/api/v1/rag/ingest?force=true"

# ── RAG: search ───────────────────────────────────────────
curl "http://localhost:8000/api/v1/rag/search?query=IS+impot+societes+Maroc+2025&top_k=3"
curl "http://localhost:8000/api/v1/rag/search?query=taux+execution+budgetaire&top_k=3"
curl "http://localhost:8000/api/v1/rag/search?query=CNSS+cotisations+patronales&top_k=3"

# ── Interactive docs ──────────────────────────────────────
open http://localhost:8000/docs        # Swagger UI (Mac)
start http://localhost:8000/docs       # Swagger UI (Windows)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

OFFLINE_TESTS = [
    ("fiscal",      test_fiscal,               "Fiscal constants & IS"),
    ("metrics",     test_metrics_corporate,     "KPI engine — corporate"),
    ("government",  test_metrics_government,    "KPI engine — government"),
    ("scenarios",   test_scenarios,             "Scenario engine"),
    ("hypothesis",  test_hypothesis_ingestion,  "Hypothesis ingestion"),
    ("plan",        test_plan_generator,        "24-month plan generator"),
    ("session",     test_session,               "Session & BusinessState"),
]

if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "all"

    if cmd == "all":
        print("\n🚀 Running all offline tests (no server needed)...\n")
        failed = []
        for key, fn, label in OFFLINE_TESTS:
            try:
                fn()
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback; traceback.print_exc()
                failed.append(label)
        test_embedder()
        test_rag()
        if failed:
            print(f"\n✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        else:
            print(f"\n✅ All offline tests passed.")
        print(CURL_COMMANDS)

    elif cmd == "chat":
        test_chat_http()

    elif cmd == "curl":
        print(CURL_COMMANDS)

    elif cmd == "embedder":
        test_embedder()

    elif cmd == "rag":
        test_embedder()
        test_rag()

    else:
        match = [fn for key, fn, _ in OFFLINE_TESTS if key == cmd]
        if match:
            match[0]()
        else:
            print(f"Unknown test: '{cmd}'")
            print(f"Available: all | curl | chat | embedder | rag | " +
                  " | ".join(k for k, _, _ in OFFLINE_TESTS))