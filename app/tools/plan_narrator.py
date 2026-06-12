"""
tools/plan_narrator.py — Generate narrative interpretations and action plans for business plans.

Transforms raw financial numbers into clear, actionable insights with risk assessment
and milestone planning.
"""

import logging
from typing import Optional, List, Dict, Any
from app.tools.plan_generator import BusinessPlan24M

logger = logging.getLogger(__name__)


def generate_plan_narrative(plan: BusinessPlan24M, language: str = "fr") -> dict:
    """
    Generate narrative interpretation of the business plan.
    
    Returns:
        {
            "executive_summary": "1-2 paragraph narrative",
            "financial_highlights": {detailed highlights},
            "key_risks": ["Risk 1", "Risk 2", ...],
            "action_plan_6months": ["Month 1: Action", ...],
            "detailed_narrative": "Full interpretation"
        }
    """
    if language == "fr":
        return _generate_narrative_fr(plan)
    else:
        return _generate_narrative_en(plan)


def _generate_narrative_fr(plan: BusinessPlan24M) -> dict:
    """French narrative generation."""
    
    # Executive Summary
    year1_revenue = plan.annee1.get("chiffre_affaires", 0)
    year1_profit = plan.annee1.get("resultat_net", 0)
    breakeven_month = plan.mois_point_mort or "N/A"
    
    executive_summary = f"""
Votre entreprise devrait générer un chiffre d'affaires de {year1_revenue:,.0f} MAD la première année.
À la fin de l'année 1, votre résultat net est estimé à {year1_profit:,.0f} MAD.
Votre seuil de rentabilité devrait être atteint au mois {breakeven_month} (ou progressivement selon la croissance client).
    """.strip()
    
    # Financial Highlights
    highlights = {
        "year1_revenue": f"{year1_revenue:,.0f} MAD",
        "year1_profit": f"{year1_profit:,.0f} MAD",
        "year1_margin": f"{(year1_profit / year1_revenue * 100 if year1_revenue > 0 else 0):.1f}%",
        "breakeven_month": breakeven_month,
        "year2_revenue": f"{plan.annee2.get('chiffre_affaires', 0):,.0f} MAD",
        "cash_position_m12": f"{plan.plan_tresorerie[-1].solde_cumule:,.0f} MAD" if plan.plan_tresorerie else "N/A",
        "debt_ratio": f"{(plan.bilan_annee1.dettes_bancaires / plan.bilan_annee1.total_passif * 100 if plan.bilan_annee1.total_passif > 0 else 0):.1f}%" if plan.bilan_annee1 else "N/A",
    }
    
    # Key Risks
    risks = _identify_risks(plan)
    
    # 6-Month Action Plan
    action_plan = _build_6month_action_plan(plan)
    
    # Detailed Narrative
    detailed_narrative = _build_detailed_narrative_fr(plan, highlights, risks)
    
    return {
        "executive_summary": executive_summary,
        "financial_highlights": highlights,
        "key_risks": risks,
        "action_plan_6months": action_plan,
        "detailed_narrative": detailed_narrative
    }


def _generate_narrative_en(plan: BusinessPlan24M) -> dict:
    """English narrative generation."""
    
    # Executive Summary
    year1_revenue = plan.annee1.get("chiffre_affaires", 0)
    year1_profit = plan.annee1.get("resultat_net", 0)
    breakeven_month = plan.mois_point_mort or "N/A"
    
    executive_summary = f"""
Your business is projected to generate {year1_revenue:,.0f} MAD in revenue in year 1.
By end of year 1, net profit is estimated at {year1_profit:,.0f} MAD.
Break-even point should be reached in month {breakeven_month} (or progressively as customer base grows).
    """.strip()
    
    # Financial Highlights
    highlights = {
        "year1_revenue": f"{year1_revenue:,.0f} MAD",
        "year1_profit": f"{year1_profit:,.0f} MAD",
        "year1_margin": f"{(year1_profit / year1_revenue * 100 if year1_revenue > 0 else 0):.1f}%",
        "breakeven_month": breakeven_month,
        "year2_revenue": f"{plan.annee2.get('chiffre_affaires', 0):,.0f} MAD",
        "cash_position_m12": f"{plan.plan_tresorerie[-1].solde_cumule:,.0f} MAD" if plan.plan_tresorerie else "N/A",
        "debt_ratio": f"{(plan.bilan_annee1.dettes_bancaires / plan.bilan_annee1.total_passif * 100 if plan.bilan_annee1.total_passif > 0 else 0):.1f}%" if plan.bilan_annee1 else "N/A",
    }
    
    # Key Risks
    risks = _identify_risks(plan)
    
    # 6-Month Action Plan
    action_plan = _build_6month_action_plan(plan)
    
    # Detailed Narrative
    detailed_narrative = _build_detailed_narrative_en(plan, highlights, risks)
    
    return {
        "executive_summary": executive_summary,
        "financial_highlights": highlights,
        "key_risks": risks,
        "action_plan_6months": action_plan,
        "detailed_narrative": detailed_narrative
    }


def _identify_risks(plan: BusinessPlan24M) -> List[str]:
    """Identify key financial risks."""
    risks = []
    
    # Check cash flow
    if plan.plan_tresorerie:
        min_cash = min([row.solde_cumule for row in plan.plan_tresorerie])
        if min_cash < 0:
            risks.append(f"⚠️ Cash flow turns negative - minimum position {min_cash:,.0f} MAD")
    
    # Check profitability
    year1_profit = plan.annee1.get("resultat_net", 0)
    if year1_profit < 0:
        risks.append(f"⚠️ Loss in year 1: {year1_profit:,.0f} MAD - review cost structure")
    elif year1_profit < plan.annee1.get("chiffre_affaires", 0) * 0.05:
        risks.append("⚠️ Low profit margin in year 1 - tight operational control required")
    
    # Check debt service
    if plan.dscr_annee1 and plan.dscr_annee1 < 1.5:
        risks.append(f"⚠️ Debt service coverage low ({plan.dscr_annee1:.1f}x) - monitor closely")
    
    # Check client concentration
    if plan.hypotheses.get("nb_clients_mois1", 0) < 5:
        risks.append("⚠️ Few clients in month 1 - revenue concentration risk")
    
    return risks if risks else ["None identified at this stage"]


def _build_6month_action_plan(plan: BusinessPlan24M) -> List[str]:
    """Build 6-month action milestones."""
    milestones = [
        "Month 1: Complete legal setup (CNSS, tax registration, accounting system)",
        "Month 2: Launch first marketing & acquisition campaign",
        "Month 3: Review vs plan - adjust pricing/costs if needed",
        "Month 4: Reach 50% of target client count - evaluate unit economics",
        "Month 5: Confirm product-market fit - plan scaling",
        "Month 6: Prepare for months 7-12 scaling - secure additional financing if needed"
    ]
    return milestones


def _build_detailed_narrative_fr(plan: BusinessPlan24M, highlights: dict, risks: List[str]) -> str:
    """Build detailed French narrative."""
    
    narrative = f"""
RÉSUMÉ FINANCIER DÉTAILLÉ

Année 1:
- Votre chiffre d'affaires attendu: {highlights['year1_revenue']}
- Résultat net: {highlights['year1_profit']}
- Marge nette: {highlights['year1_margin']}

Année 2:
- Chiffre d'affaires projeté: {highlights['year2_revenue']}
- Position de trésorerie en fin année 1: {highlights['cash_position_m12']}

POINT D'ÉQUILIBRE:
Le seuil de rentabilité devrait être atteint au mois {highlights['breakeven_month']}.
Cela signifie que vous commencerez à générer des bénéfices à partir de ce mois.

STRUCTURE DE FINANCEMENT:
Votre ratio d'endettement (dettes bancaires / total passif) est de {highlights['debt_ratio']}.

RISQUES IDENTIFIÉS:
{chr(10).join('- ' + r for r in risks)}

RECOMMANDATIONS:
1. Mettre en place un suivi mensuel des métriques clés (nombre de clients, chiffre d'affaires, trésorerie)
2. Ajuster les charges en fonction de la réalité vs prévisions
3. Maintenir une réserve de trésorerie suffisante pour les imprévus
4. Revoir le plan tous les trimestres et l'adapter selon l'évolution du marché
    """.strip()
    
    return narrative


def _build_detailed_narrative_en(plan: BusinessPlan24M, highlights: dict, risks: List[str]) -> str:
    """Build detailed English narrative."""
    
    narrative = f"""
DETAILED FINANCIAL SUMMARY

Year 1:
- Expected Revenue: {highlights['year1_revenue']}
- Net Profit: {highlights['year1_profit']}
- Net Margin: {highlights['year1_margin']}

Year 2:
- Projected Revenue: {highlights['year2_revenue']}
- Cash Position at Year End: {highlights['cash_position_m12']}

BREAK-EVEN POINT:
Break-even should be reached in month {highlights['breakeven_month']}.
This means you will start generating profits from that month onwards.

FINANCING STRUCTURE:
Your debt ratio (bank debt / total liabilities) is {highlights['debt_ratio']}.

IDENTIFIED RISKS:
{chr(10).join('- ' + r for r in risks)}

RECOMMENDATIONS:
1. Implement monthly tracking of key metrics (clients, revenue, cash)
2. Adjust costs based on actual vs projected performance
3. Maintain sufficient cash reserves for contingencies
4. Review and update the plan quarterly based on market evolution
    """.strip()
    
    return narrative


def format_plan_for_display(plan: BusinessPlan24M, narrative: dict) -> dict:
    """Format entire plan for display/storage."""
    return {
        "executive_summary": narrative["executive_summary"],
        "financial_highlights": narrative["financial_highlights"],
        "key_risks": narrative["key_risks"],
        "action_plan_6months": narrative["action_plan_6months"],
        "detailed_narrative": narrative["detailed_narrative"],
        "raw_data": {
            "annee1": plan.annee1,
            "annee2": plan.annee2,
            "plan_financement": plan.plan_financement.__dict__ if plan.plan_financement else {},
            "kpis": {
                "seuil_rentabilite_clients": plan.seuil_rentabilite_clients,
                "mois_point_mort": plan.mois_point_mort,
                "roi_annee1": plan.roi_annee1,
                "roi_annee2": plan.roi_annee2,
                "dscr_annee1": plan.dscr_annee1,
            }
        }
    }
