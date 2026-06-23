"""
scripts/generate_training_data.py — Synthetic training data generator for the risk classifier.

HOW IT WORKS
────────────
1. Monte-Carlo sample N random business_state dicts across realistic Moroccan SME ranges.
2. Run each through the project's own deterministic engine via compute_plan()
   (the exact same path the real agent uses — label and feature never diverge).
3. Apply the agreed labeling rule → "at_risk" | "viable".
4. Save the raw input features + label to a CSV ready for train_risk_model.py.

LABELING RULE (confirmed in SESSION_SUMMARY_2.md §5.4)
───────────────────────────────────────────────────────
A sample is labeled AT_RISK if ANY of:
  • solde_cumule goes negative at any month across the 24-month window
  • mois_point_mort is None (break-even never reached within 24 months)
  • dscr_annee1 < 1.0  (when the business has debt — otherwise skipped)

Otherwise: VIABLE.

OUTPUT CSV COLUMNS
──────────────────
Features (numeric, categoricals encoded as int):
  segment_client_enc       0=B2C, 1=B2B, 2=Mixte
  type_activite_enc        0=service, 1=produit, 2=hybride
  statut_juridique_enc     0=auto-entrepreneur, 1=SARL, 2=SA
  nature_clients_enc       0=comptant, 1=credit, 2=mixte
  prix_vente_unitaire
  nb_clients_mois1
  taux_croissance_mensuel
  taux_fidelisation
  cout_fabrication_unitaire
  cout_infra_numerique
  loyer_mensuel
  salaires_equipe
  charges_utilites
  budget_marketing
  investissements_initiaux
  emprunts
  own_capital_invested      (capital_social)
  delai_jours

Label:
  label                    "at_risk" | "viable"

USAGE
─────
  # From project root:
  python scripts/generate_training_data.py               # 10 000 rows (default)
  python scripts/generate_training_data.py --n-samples 50000
  python scripts/generate_training_data.py --n-samples 500 --seed 99 --output data/ml/test.csv
  python scripts/generate_training_data.py --verbose     # print per-row debug info
"""

import argparse
import csv
import os
import random
import sys
import time

# ── Path fix so the script can import app.* from the project root ─────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.tools.plan_pipeline import compute_plan  # noqa: E402  (import after sys.path fix)


# ─────────────────────────────────────────────────────────────────────────────
# Sampling ranges — grounded in Moroccan SME reality
# ─────────────────────────────────────────────────────────────────────────────

# Categorical choices + weights (weights don't need to sum to 1 — random.choices normalises)
SEGMENT_CHOICES   = ["B2C",  "B2B",   "Mixte"]
SEGMENT_WEIGHTS   = [0.45,   0.35,    0.20]

ACTIVITE_CHOICES  = ["service", "produit", "hybride"]
ACTIVITE_WEIGHTS  = [0.55,      0.30,      0.15]

STATUT_CHOICES    = ["Auto-entrepreneur", "SARL", "SA"]
STATUT_WEIGHTS    = [0.40,                0.50,   0.10]

NATURE_CHOICES    = ["comptant", "credit", "mixte"]
NATURE_WEIGHTS    = [0.50,       0.30,     0.20]

# Numeric ranges: (min, max) — sampled with uniform distribution unless noted
# Values are in MAD at single-unit scale (not millions)
RANGES = {
    # Revenue side
    "prix_vente_unitaire":      (50,     5_000),   # MAD/unit — covers cheap service to B2B contract
    "nb_clients_mois1":         (1,      200),      # Starting customer base
    "taux_croissance_mensuel":  (1.0,    20.0),     # % monthly — 1% cautious, 20% aggressive
    "taux_fidelisation":        (60.0,   99.0),     # % retention/month

    # Cost side (variable)
    "cout_fabrication_unitaire":(0,      1_000),    # MAD/unit — 0 for pure services

    # Fixed costs (monthly, MAD)
    "cout_infra_numerique":     (0,      3_000),    # Cloud / SaaS tools
    "loyer_mensuel":            (0,      15_000),   # Rent (0 = home office / no premises)
    "salaires_equipe":          (0,      30_000),   # Net salaries total
    "charges_utilites":         (0,      3_000),    # Electricity, water, internet
    "budget_marketing":         (0,      8_000),    # Ads, content, events

    # One-time investment
    "investissements_initiaux": (0,      200_000),  # Equipment, setup

    # Financing
    "emprunts":                 (0,      300_000),  # Bank loan
    "own_capital_invested":     (1_000,  200_000),  # Founder's equity

    # Payment terms
    "delai_jours":              (0,      90),       # Days delay for credit clients
}


def _weighted_choice(choices, weights):
    return random.choices(choices, weights=weights, k=1)[0]


def _uniform(lo, hi):
    """Uniform float in [lo, hi], rounded to 2 decimal places."""
    return round(random.uniform(lo, hi), 2)


def _sample_business_state() -> dict:
    """
    Draw one random business_state dict.

    The dict mirrors the flat format that plan_pipeline.business_state_to_hypothesis()
    expects — same keys, same scale. No translation needed.
    """
    segment  = _weighted_choice(SEGMENT_CHOICES,  SEGMENT_WEIGHTS)
    activite = _weighted_choice(ACTIVITE_CHOICES, ACTIVITE_WEIGHTS)
    statut   = _weighted_choice(STATUT_CHOICES,   STATUT_WEIGHTS)
    nature   = _weighted_choice(NATURE_CHOICES,   NATURE_WEIGHTS)

    prix    = _uniform(*RANGES["prix_vente_unitaire"])
    clients = round(_uniform(*RANGES["nb_clients_mois1"]))       # integer clients

    # For produit/hybride, allow non-zero variable cost; for pure service keep it low
    if activite == "service":
        cout_var = _uniform(0, prix * 0.15)   # service: margin usually high, cap at 15% of price
    else:
        cout_var = _uniform(0, min(_uniform(*RANGES["cout_fabrication_unitaire"]), prix * 0.90))

    emprunts = _uniform(*RANGES["emprunts"])
    # Capital must be at least meaningful relative to the business; allow some undercapitalisation
    capital  = _uniform(*RANGES["own_capital_invested"])

    # Payment delay only makes sense for credit/mixte clients
    delai = 0
    if nature in ("credit", "mixte"):
        delai = round(_uniform(*RANGES["delai_jours"]))

    return {
        # Identifiers for pipeline routing
        "entity_type":                 "startup",
        "entity_name":                 "Synthetic Co",
        "sector":                      "test",

        # Categorical (raw string — pipeline handles encoding)
        "segment_client":              segment,
        "type_activite":               activite,
        "statut_juridique":            statut,
        "nature_clients_encaissements": nature,

        # Revenue
        "prix_vente_unitaire":         prix,
        "nb_clients_mois1":            clients,
        "taux_croissance_mensuel":     _uniform(*RANGES["taux_croissance_mensuel"]),
        "taux_fidelisation":           _uniform(*RANGES["taux_fidelisation"]),

        # Variable cost
        "cout_fabrication_unitaire":   round(cout_var, 2),

        # Fixed costs
        "cout_infra_numerique":        _uniform(*RANGES["cout_infra_numerique"]),
        "loyer_mensuel":               _uniform(*RANGES["loyer_mensuel"]),
        "salaires_equipe":             _uniform(*RANGES["salaires_equipe"]),
        "charges_utilites":            _uniform(*RANGES["charges_utilites"]),
        "budget_marketing":            _uniform(*RANGES["budget_marketing"]),

        # One-time
        "investissements_initiaux":    _uniform(*RANGES["investissements_initiaux"]),

        # Financing
        "emprunts":                    emprunts,
        "own_capital_invested":        capital,

        # Payment terms
        "delai_jours":                 delai,
    }


def _apply_label(plan, business_state: dict) -> str:
    """
    Apply the agreed labeling rule to a computed BusinessPlan24M.

    AT_RISK if ANY of:
      • Any month's solde_cumule (running cash) is negative
      • mois_point_mort is None (never broke even in 24 months)
      • dscr_annee1 < 1.0  — only when there IS debt (mensualité > 0)

    VIABLE otherwise.
    """
    # Rule 1: running cash goes negative at any point
    cash_goes_negative = any(
        row.solde_cumule < 0
        for row in plan.compte_resultat
    )
    if cash_goes_negative:
        return "at_risk"

    # Rule 2: break-even never reached
    if plan.mois_point_mort is None:
        return "at_risk"

    # Rule 3: DSCR < 1.0 when there is a loan
    has_debt = (business_state.get("emprunts") or 0) > 0
    if has_debt and plan.dscr_annee1 is not None and plan.dscr_annee1 < 1.0:
        return "at_risk"

    return "viable"


def _to_csv_row(bs: dict, label: str) -> dict:
    """
    Flatten the business_state dict + label into the CSV row format.
    Categoricals are stored as their raw strings (e.g. "B2C", "service") —
    encoding into integers is the training script's job, not ours.
    """
    return {
        # Categorical features — raw strings
        "segment_client":              bs["segment_client"],
        "type_activite":               bs["type_activite"],
        "statut_juridique":            bs["statut_juridique"],
        "nature_clients_encaissements": bs["nature_clients_encaissements"],

        # Numeric features
        "prix_vente_unitaire":         bs["prix_vente_unitaire"],
        "nb_clients_mois1":            bs["nb_clients_mois1"],
        "taux_croissance_mensuel":     bs["taux_croissance_mensuel"],
        "taux_fidelisation":           bs["taux_fidelisation"],
        "cout_fabrication_unitaire":   bs["cout_fabrication_unitaire"],
        "cout_infra_numerique":        bs["cout_infra_numerique"],
        "loyer_mensuel":               bs["loyer_mensuel"],
        "salaires_equipe":             bs["salaires_equipe"],
        "charges_utilites":            bs["charges_utilites"],
        "budget_marketing":            bs["budget_marketing"],
        "investissements_initiaux":    bs["investissements_initiaux"],
        "emprunts":                    bs["emprunts"],
        "own_capital_invested":        bs["own_capital_invested"],
        "delai_jours":                 bs["delai_jours"],

        # Label
        "label":                       label,
    }


CSV_FIELDNAMES = [
    "segment_client", "type_activite", "statut_juridique", "nature_clients_encaissements",
    "prix_vente_unitaire", "nb_clients_mois1", "taux_croissance_mensuel", "taux_fidelisation",
    "cout_fabrication_unitaire", "cout_infra_numerique", "loyer_mensuel", "salaires_equipe",
    "charges_utilites", "budget_marketing", "investissements_initiaux", "emprunts",
    "own_capital_invested", "delai_jours",
    "label",
]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate(n_samples: int, output_path: str, seed: int, verbose: bool) -> None:
    random.seed(seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counts   = {"viable": 0, "at_risk": 0}
    skipped  = 0
    t_start  = time.time()

    print(f"Generating {n_samples:,} samples → {output_path}")
    print(f"Seed: {seed}\n")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        i = 0
        attempts = 0
        while i < n_samples:
            attempts += 1

            # Safety valve: if the engine rejects too many samples something is wrong
            if attempts > n_samples * 10:
                print(
                    f"\n⚠️  Stopped early: too many failed engine calls "
                    f"({skipped} skipped out of {attempts} attempts). "
                    f"Check compute_plan() and sampling ranges."
                )
                break

            bs = _sample_business_state()
            result = compute_plan(bs)

            if result is None:
                # Engine returned None — missing/invalid fields, skip and try again
                skipped += 1
                if verbose:
                    print(f"  [skip] attempt {attempts}: engine returned None for sample {i+1}")
                continue

            label = _apply_label(result.plan, bs)
            row   = _to_csv_row(bs, label)
            writer.writerow(row)

            counts[label] += 1
            i += 1

            # Progress indicator every 1000 rows (or every row in verbose mode)
            if verbose:
                print(
                    f"  [{i:>5}] label={label:<8} "
                    f"prix={bs['prix_vente_unitaire']:>7.0f} MAD  "
                    f"clients={bs['nb_clients_mois1']:>4}  "
                    f"emprunts={bs['emprunts']:>8.0f}  "
                    f"capital={bs['own_capital_invested']:>7.0f}  "
                    f"mois_pm={result.plan.mois_point_mort or 'None':<5}  "
                    f"dscr={result.plan.dscr_annee1 or 'N/A'}"
                )
            elif i % 1_000 == 0:
                elapsed  = time.time() - t_start
                rate     = i / elapsed
                eta_secs = (n_samples - i) / rate if rate > 0 else 0
                pct_risk = counts["at_risk"] / i * 100
                print(
                    f"  {i:>6,}/{n_samples:,}  "
                    f"at_risk={pct_risk:.1f}%  "
                    f"skipped={skipped}  "
                    f"elapsed={elapsed:.1f}s  ETA={eta_secs:.0f}s"
                )

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────────────
    total = counts["viable"] + counts["at_risk"]
    print(f"\n{'─'*50}")
    print(f"Done in {elapsed:.1f}s  ({total/elapsed:.0f} rows/sec)")
    print(f"")
    print(f"  Rows written : {total:,}")
    print(f"  Skipped      : {skipped}")
    print(f"  Attempts     : {attempts:,}")
    print(f"")
    print(f"  viable       : {counts['viable']:>6,}  ({counts['viable']/total*100:.1f}%)")
    print(f"  at_risk      : {counts['at_risk']:>6,}  ({counts['at_risk']/total*100:.1f}%)")
    print(f"")

    # Warn if class imbalance is extreme (>80/20 split)
    majority_pct = max(counts.values()) / total * 100
    if majority_pct > 80:
        print(
            f"  ⚠️  Class imbalance: majority class is {majority_pct:.0f}% of data.\n"
            f"     Consider adjusting sampling ranges or using class_weight='balanced'\n"
            f"     in train_risk_model.py."
        )

    print(f"  Output: {output_path}")
    print(f"{'─'*50}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for the FinAgent risk classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-samples", type=int, default=10_000,
        help="Number of labeled rows to generate (default: 10 000)",
    )
    parser.add_argument(
        "--output", type=str, default="data/ml/training_data.csv",
        help="Output CSV path relative to project root (default: data/ml/training_data.csv)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print one line per row (slow — use for debugging with small --n-samples)",
    )
    args = parser.parse_args()

    # Resolve output path relative to project root (not wherever the user's cwd is)
    if not os.path.isabs(args.output):
        output_path = os.path.join(PROJECT_ROOT, args.output)
    else:
        output_path = args.output

    generate(
        n_samples=args.n_samples,
        output_path=output_path,
        seed=args.seed,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()