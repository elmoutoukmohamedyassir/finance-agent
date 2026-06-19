"""
tools/scenario_engine.py — Enterprise financial projection engine. Pure Python, no LLM.

Projects revenue, costs, profit, debt, and cash over N years for both
corporate and government entities. All values in MAD millions.
"""


def project_scenario(
    starting_revenue: float,
    starting_costs: float,
    starting_cash: float,
    starting_debt: float,
    revenue_growth_pct: float,
    cost_growth_pct: float,
    debt_service_annual: float = 0.0,
    capex_annual: float = 0.0,
    years: int = 3,
) -> dict:
    """
    Project a single scenario over N years.

    Args:
        starting_revenue:    Base year revenue (MAD millions)
        starting_costs:      Base year operating costs (MAD millions)
        starting_cash:       Opening cash position (MAD millions)
        starting_debt:       Total debt at start (MAD millions)
        revenue_growth_pct:  Annual revenue growth rate (%)
        cost_growth_pct:     Annual cost growth rate (%)
        debt_service_annual: Annual principal + interest payments (MAD millions)
        capex_annual:        Annual capital expenditure (MAD millions)
        years:               Projection horizon (1–10)

    Returns dict with yearly snapshots and summary.
    """
    rev_growth = revenue_growth_pct / 100
    cost_growth = cost_growth_pct / 100

    revenue = float(starting_revenue)
    costs = float(starting_costs)
    cash = float(starting_cash)
    debt = float(starting_debt)

    snapshots = []
    cash_negative_year = None

    for year in range(1, years + 1):
        revenue = round(revenue * (1 + rev_growth), 3)
        costs = round(costs * (1 + cost_growth), 3)

        ebitda = round(revenue - costs, 3)
        net_cash_generated = round(ebitda - debt_service_annual - capex_annual, 3)
        cash = round(cash + net_cash_generated, 3)
        debt = round(max(0.0, debt - debt_service_annual), 3)

        ebitda_margin = round((ebitda / revenue * 100), 1) if revenue > 0 else 0.0
        dscr = round(ebitda / debt_service_annual, 2) if debt_service_annual > 0 else None

        if cash < 0 and cash_negative_year is None:
            cash_negative_year = year

        snapshots.append({
            "year": year,
            "revenue_mad_m": revenue,
            "costs_mad_m": costs,
            "ebitda_mad_m": ebitda,
            "ebitda_margin_pct": ebitda_margin,
            "debt_service_mad_m": debt_service_annual,
            "capex_mad_m": capex_annual,
            "net_cash_generated_mad_m": net_cash_generated,
            "cumulative_cash_mad_m": cash,
            "remaining_debt_mad_m": debt,
            "dscr": dscr,
        })

    final = snapshots[-1]
    initial_revenue = starting_revenue * (1 + rev_growth)  # year 1 base
    revenue_growth_total = round(
        (final["revenue_mad_m"] - initial_revenue) / initial_revenue * 100, 1
    ) if initial_revenue > 0 else 0.0

    return {
        "yearly_projections": snapshots,
        "summary": {
            "final_revenue_mad_m": final["revenue_mad_m"],
            "final_ebitda_mad_m": final["ebitda_mad_m"],
            "final_ebitda_margin_pct": final["ebitda_margin_pct"],
            "final_cash_mad_m": final["cumulative_cash_mad_m"],
            "final_debt_mad_m": final["remaining_debt_mad_m"],
            "revenue_growth_total_pct": revenue_growth_total,
            "cash_negative_year": cash_negative_year if cash_negative_year else f"Aucune sur {years} ans",
        }
    }


def build_standard_scenarios(
    starting_revenue: float,
    starting_costs: float,
    starting_cash: float = 10.0,
    starting_debt: float = 0.0,
    debt_service_annual: float = 0.0,
    capex_annual: float = 0.0,
    years: int = 3,
    entity_type: str = "corporate",
) -> list[dict]:
    """
    Build the standard 3-scenario analysis (Pessimiste / Réaliste / Optimiste).

    Corporate growth rates based on Moroccan large enterprise norms.
    Government rates based on budget evolution patterns from Bulletin mensuel.
    """
    if entity_type.lower() == "government":
        configs = [
            ("Pessimiste",  2.0,  5.0),   # low revenue growth, costs rising fast
            ("Réaliste",    5.0,  3.5),   # moderate growth inline with GDP
            ("Optimiste",   9.0,  2.5),   # strong fiscal performance
        ]
    else:
        configs = [
            ("Pessimiste",  3.0,  6.0),   # revenue stagnation, cost pressure
            ("Réaliste",    8.0,  4.0),   # healthy enterprise growth
            ("Optimiste",  15.0,  2.5),   # strong expansion year
        ]

    results = []
    for name, rev_growth, cost_growth in configs:
        data = project_scenario(
            starting_revenue=starting_revenue,
            starting_costs=starting_costs,
            starting_cash=starting_cash,
            starting_debt=starting_debt,
            revenue_growth_pct=rev_growth,
            cost_growth_pct=cost_growth,
            debt_service_annual=debt_service_annual,
            capex_annual=capex_annual,
            years=years,
        )
        data["name"] = name
        data["assumptions"] = {
            "revenue_growth_pct": rev_growth,
            "cost_growth_pct": cost_growth,
            "debt_service_annual_mad_m": debt_service_annual,
            "capex_annual_mad_m": capex_annual,
        }
        results.append(data)
    return results


def format_scenarios_for_prompt(scenarios: list[dict], years: int = 3) -> str:
    """Compact summary injected into the LLM prompt — summaries only, not full yearly data."""
    lines = []
    for s in scenarios:
        summ = s["summary"]
        assum = s["assumptions"]
        lines.append(
            f"  [{s['name']}] "
            f"Croissance revenus {assum['revenue_growth_pct']}%/an · "
            f"Croissance coûts {assum['cost_growth_pct']}%/an → "
            f"Revenus an{years}: {summ['final_revenue_mad_m']:.1f} MMAD · "
            f"EBITDA: {summ['final_ebitda_mad_m']:.1f} MMAD ({summ['final_ebitda_margin_pct']:.1f}%) · "
            f"Trésorerie: {summ['final_cash_mad_m']:.1f} MMAD · "
            f"Dette résiduelle: {summ['final_debt_mad_m']:.1f} MMAD · "
            f"Tréso négative: {summ['cash_negative_year']}"
        )
    return "\n".join(lines)