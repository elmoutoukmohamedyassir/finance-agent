"""
tools/scenario_engine.py — SaaS financial scenario projections.

IMPROVEMENTS over the original scenario_analysis.py:
  1. Includes monthly costs in projections (original ignored costs entirely)
  2. Tracks profit/loss per month, not just MRR
  3. Calculates runway per scenario (critical for founders)
  4. Supports custom scenarios, not just fixed pessimistic/realistic/optimistic
  5. Returns structured dicts ready for LLM interpretation

KEY DESIGN: All math is deterministic Python. No LLM involved here.
The LLM only interprets these pre-calculated numbers.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    """Defines a single scenario's growth + churn assumptions."""
    name: str
    monthly_growth_rate_pct: float   # e.g. 10.0 for 10% monthly growth
    monthly_churn_rate_pct: float    # e.g. 5.0 for 5% monthly churn
    cost_growth_rate_pct: float = 2.0  # costs grow slightly each month


def project_scenario(
    starting_customers: int,
    monthly_price: float,
    monthly_costs: float,
    starting_cash: float,
    config: ScenarioConfig,
    months: int = 12,
) -> dict:
    """
    Projects a single scenario over N months.

    Returns a dict with:
      - name: scenario label
      - assumptions: the growth/churn rates used
      - months: list of monthly snapshots
      - summary: final month state + runway info
    """
    growth_rate = config.monthly_growth_rate_pct / 100
    churn_rate = config.monthly_churn_rate_pct / 100
    cost_growth = config.cost_growth_rate_pct / 100

    customers = float(starting_customers)
    costs = float(monthly_costs)
    cash = float(starting_cash)
    monthly_snapshots = []
    runway_month = None  # Month when cash runs out

    for month in range(1, months + 1):
        mrr = round(customers * monthly_price, 2)
        profit = round(mrr - costs, 2)
        cash = round(cash + profit, 2)

        if cash < 0 and runway_month is None:
            runway_month = month - 1  # Ran out before this month

        monthly_snapshots.append({
            "month": month,
            "customers": int(round(customers)),
            "mrr": mrr,
            "costs": round(costs, 2),
            "profit": profit,
            "cash": cash,
        })

        # Apply growth and churn for next month
        gained = customers * growth_rate
        lost = customers * churn_rate
        customers = max(0, customers + gained - lost)
        costs = costs * (1 + cost_growth)

    # Summary stats
    final = monthly_snapshots[-1]
    initial_mrr = starting_customers * monthly_price
    final_mrr = final["mrr"]
    mrr_growth_pct = round(((final_mrr - initial_mrr) / initial_mrr * 100), 1) if initial_mrr > 0 else 0

    summary = {
        "final_customers": final["customers"],
        "final_mrr": final_mrr,
        "final_cash": final["cash"],
        "mrr_growth_over_period_pct": mrr_growth_pct,
        "runway_months": runway_month if runway_month is not None else f">{months}",
        "profitable_in_month_1": monthly_snapshots[0]["profit"] > 0,
    }

    return {
        "name": config.name,
        "assumptions": {
            "monthly_growth_rate_pct": config.monthly_growth_rate_pct,
            "monthly_churn_rate_pct": config.monthly_churn_rate_pct,
        },
        "monthly_projections": monthly_snapshots,
        "summary": summary,
    }


def build_standard_scenarios(
    starting_customers: int,
    monthly_price: float,
    monthly_costs: float,
    starting_cash: float,
    months: int = 12,
) -> list[dict]:
    """
    Builds the standard 3-scenario analysis: pessimistic, realistic, optimistic.
    These are the industry-standard assumptions for early-stage SaaS.
    """
    configs = [
        ScenarioConfig(
            name="Pessimistic",
            monthly_growth_rate_pct=5.0,
            monthly_churn_rate_pct=8.0,
        ),
        ScenarioConfig(
            name="Realistic",
            monthly_growth_rate_pct=10.0,
            monthly_churn_rate_pct=4.0,
        ),
        ScenarioConfig(
            name="Optimistic",
            monthly_growth_rate_pct=20.0,
            monthly_churn_rate_pct=2.0,
        ),
    ]

    return [
        project_scenario(
            starting_customers=starting_customers,
            monthly_price=monthly_price,
            monthly_costs=monthly_costs,
            starting_cash=starting_cash,
            config=cfg,
            months=months,
        )
        for cfg in configs
    ]


def format_scenarios_for_prompt(scenarios: list[dict]) -> str:
    """
    Formats scenario summaries into a clean text block for LLM prompts.
    Only sends summaries (not full monthly data) to save context tokens.
    """
    lines = []
    for s in scenarios:
        summary = s["summary"]
        assumptions = s["assumptions"]
        lines.append(
            f"[{s['name']}] "
            f"Growth {assumptions['monthly_growth_rate_pct']}%/mo, "
            f"Churn {assumptions['monthly_churn_rate_pct']}%/mo → "
            f"Final MRR: ${summary['final_mrr']:,.0f}, "
            f"Final Customers: {summary['final_customers']}, "
            f"Runway: {summary['runway_months']} months, "
            f"MRR Growth: {summary['mrr_growth_over_period_pct']}%"
        )
    return "\n".join(lines)
