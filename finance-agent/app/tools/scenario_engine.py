"""
tools/scenario_engine.py — Financial projection engine. Pure Python, no LLM.

Improvement over original: tracks costs, profit, and cash per month —
not just MRR. A scenario that shows MRR growth but negative cash is
a very different story than the original code revealed.
"""


def project_scenario(
    starting_customers: int,
    monthly_price: float,
    monthly_costs: float,
    starting_cash: float,
    monthly_growth_pct: float,
    monthly_churn_pct: float,
    months: int = 12,
    cost_growth_pct: float = 1.5,
) -> dict:
    """Project a single scenario over N months."""
    growth = monthly_growth_pct / 100
    churn = monthly_churn_pct / 100
    cost_growth = cost_growth_pct / 100

    customers = float(starting_customers)
    costs = float(monthly_costs)
    cash = float(starting_cash)
    snapshots = []
    runway_month = None

    for month in range(1, months + 1):
        mrr = round(customers * monthly_price, 2)
        profit = round(mrr - costs, 2)
        cash = round(cash + profit, 2)

        if cash < 0 and runway_month is None:
            runway_month = month - 1

        snapshots.append({
            "month": month,
            "customers": int(round(customers)),
            "mrr": mrr,
            "costs": round(costs, 2),
            "profit": profit,
            "cash": cash,
        })

        gained = customers * growth
        lost = customers * churn
        customers = max(0.0, customers + gained - lost)
        costs *= (1 + cost_growth)

    final = snapshots[-1]
    initial_mrr = starting_customers * monthly_price
    mrr_growth = round((final["mrr"] - initial_mrr) / initial_mrr * 100, 1) if initial_mrr > 0 else 0

    return {
        "monthly_projections": snapshots,
        "summary": {
            "final_mrr": final["mrr"],
            "final_customers": final["customers"],
            "final_cash": final["cash"],
            "mrr_growth_pct": mrr_growth,
            "runway": runway_month if runway_month is not None else f">{months} months",
        }
    }


def build_standard_scenarios(
    starting_customers: int,
    monthly_price: float,
    monthly_costs: float,
    starting_cash: float = 50000,
    months: int = 12,
) -> list[dict]:
    """Build the standard 3-scenario analysis."""
    configs = [
        ("Pessimistic", 5.0, 8.0),
        ("Realistic",   10.0, 4.0),
        ("Optimistic",  20.0, 2.0),
    ]
    results = []
    for name, growth, churn in configs:
        data = project_scenario(
            starting_customers=starting_customers,
            monthly_price=monthly_price,
            monthly_costs=monthly_costs,
            starting_cash=starting_cash,
            monthly_growth_pct=growth,
            monthly_churn_pct=churn,
            months=months,
        )
        data["name"] = name
        data["assumptions"] = {"growth_pct": growth, "churn_pct": churn}
        results.append(data)
    return results


def format_scenarios_for_prompt(scenarios: list[dict]) -> str:
    """Compact summary for LLM injection — sends summaries, not full monthly data."""
    lines = []
    for s in scenarios:
        summ = s["summary"]
        assum = s["assumptions"]
        lines.append(
            f"  [{s['name']}] Growth {assum['growth_pct']}%/mo · Churn {assum['churn_pct']}%/mo "
            f"→ Final MRR: ${summ['final_mrr']:,.0f} · "
            f"Customers: {summ['final_customers']} · "
            f"Cash: ${summ['final_cash']:,.0f} · "
            f"Runway: {summ['runway']}"
        )
    return "\n".join(lines)
