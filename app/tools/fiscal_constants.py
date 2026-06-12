"""
tools/fiscal_constants.py — Moroccan regulatory fiscal constants (2024–2025).

WHY THIS EXISTS:
  Fiscal values (IS rates, CNSS contributions, TVA rates, amortization periods)
  change year to year. They must NOT be hardcoded in calculation logic.
  Centralizing them here means:
    1. A single place to update when laws change
    2. Versioned by year — historical calculations stay reproducible
    3. Any agent (Finance, Legal, Reporting) can import from one source of truth

SOURCE REFERENCES:
  - IS rates: CGI Maroc 2025, Article 19 (tranches progressives depuis 2023)
  - CNSS/AMO: CNSS Maroc taux 2024
  - TVA: CGI Maroc 2025, Article 99
  - Amortissements: CGI Maroc 2025, Annexe des taux d'amortissement
  - SMIG: Ministère du Travail et de l'Insertion Professionnelle, Jan 2025
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ISBracket:
    """Single IS (Impôt sur les Sociétés) tax bracket."""
    lower: float        # MAD lower bound (inclusive)
    upper: float        # MAD upper bound (exclusive, use float('inf') for last bracket)
    rate: float         # Rate as decimal (0.20 = 20%)
    label: str


@dataclass(frozen=True)
class CNSSRates:
    patronal_part: float    # Employer contribution rate (decimal)
    salarial_part: float    # Employee contribution rate (decimal)


@dataclass(frozen=True)
class AMORates:
    patronale: float        # Assurance Maladie Obligatoire — employer (decimal)
    salariale: float        # Assurance Maladie Obligatoire — employee (decimal)


@dataclass(frozen=True)
class AmortizationSchedule:
    """Legal amortization periods and rates in Morocco."""
    # Annual rates (straight-line, taux linéaire)
    materiel_informatique: float = 0.25     # 4 years
    logiciels: float = 0.333               # 3 years
    mobilier_bureau: float = 0.20          # 5 years
    vehicules_tourisme: float = 0.20       # 5 years
    vehicules_utilitaires: float = 0.25    # 4 years
    materiel_industriel: float = 0.10      # 10 years
    amenagements_locaux: float = 0.10      # 10 years
    brevets_licences: float = 0.20         # 5 years


@dataclass(frozen=True)
class TVARates:
    """TVA rates by category (Maroc)."""
    standard: float = 0.20          # Taux normal
    intermediaire: float = 0.14     # Immobilier, transport, banque
    reduit: float = 0.10            # Hôtellerie, restauration, eau, gaz
    super_reduit: float = 0.07      # Produits alimentaires de base
    exonere: float = 0.0            # Exportations, médicaments, éducation


@dataclass(frozen=True)
class FiscalYear:
    """All fiscal constants for a given year."""
    year: int

    # IS brackets (MAD, progressive since finance law 2023)
    is_brackets: list[ISBracket]

    # IR auto-entrepreneur
    ir_auto_entrepreneur_achat_revente: float   # % on CA
    ir_auto_entrepreneur_services: float        # % on CA

    # CNSS
    cnss: CNSSRates

    # AMO
    amo: AMORates

    # Total employer charge multiplier on net salary
    # gross = net / (1 - salarial) → total_employer_cost = gross × (1 + patronal + amo_patronale)
    # Simplified: total_cost_multiplier applied to gross salary
    employer_charge_multiplier: float           # Applied to gross salary

    # Labor
    smig_mensuel: float                         # MAD/month (minimum wage)
    smig_horaire: float                         # MAD/hour

    # TVA
    tva: TVARates

    # Amortization
    amortization: AmortizationSchedule

    # Creation costs
    frais_immatriculation_sarl_min: float       # MAD
    frais_immatriculation_sarl_max: float       # MAD
    capital_minimum_sarl: float                 # MAD (legal, banks may require more)

    def compute_is(self, result_avant_is: float) -> float:
        """
        Calculate IS due on taxable profit using progressive brackets.
        result_avant_is: MAD (full scale, not millions)
        Returns IS amount in MAD.
        """
        if result_avant_is <= 0:
            return 0.0
        remaining = result_avant_is
        total_is = 0.0
        for bracket in self.is_brackets:
            if remaining <= 0:
                break
            bracket_size = (
                (bracket.upper - bracket.lower)
                if bracket.upper != float('inf')
                else remaining
            )
            taxable_in_bracket = min(remaining, bracket_size)
            total_is += taxable_in_bracket * bracket.rate
            remaining -= taxable_in_bracket
        return round(total_is, 2)

    def compute_employer_salary_cost(self, salaire_net: float) -> float:
        """
        Total monthly cost to employer for a given net salary.
        Approximate: gross = net / 0.9326 then × 1.252
        """
        gross = salaire_net / (1 - self.cnss.salarial_part - self.amo.salariale)
        total = gross * (1 + self.cnss.patronal_part + self.amo.patronale)
        return round(total, 2)

    def compute_loan_monthly_payment(
        self,
        capital: float,
        taux_annuel: float,
        duree_mois: int
    ) -> float:
        """
        Monthly annuity (mensualité) for a loan.
        Formula: M = Capital × [r / (1 − (1+r)^−n)]
        """
        if duree_mois <= 0 or capital <= 0:
            return 0.0
        r = taux_annuel / 12
        if r == 0:
            return round(capital / duree_mois, 2)
        m = capital * (r / (1 - (1 + r) ** -duree_mois))
        return round(m, 2)


# ── CONSTANTS: 2025 ──────────────────────────────────────────────────────────

FISCAL_2025 = FiscalYear(
    year=2025,

    # IS progressive brackets (CGI 2025, art. 19)
    # Progressive since Loi de Finances 2023 — full progression from 2026
    is_brackets=[
        ISBracket(0,          300_000,    0.10, "≤ 300 000 MAD"),
        ISBracket(300_000,  1_000_000,    0.20, "300 001 – 1 000 000 MAD"),
        ISBracket(1_000_000, 100_000_000, 0.285, "1 000 001 – 100 000 000 MAD"),
        ISBracket(100_000_000, float('inf'), 0.35, "> 100 000 000 MAD"),
    ],

    # IR auto-entrepreneur (DGI 2024–2025)
    ir_auto_entrepreneur_achat_revente=0.01,   # 1% on CA
    ir_auto_entrepreneur_services=0.02,         # 2% on CA

    # CNSS 2024 (unchanged for 2025 budget)
    cnss=CNSSRates(
        patronal_part=0.2109,   # 21.09%
        salarial_part=0.0674,   # 6.74%
    ),

    # AMO 2024
    amo=AMORates(
        patronale=0.0411,   # 4.11%
        salariale=0.0226,   # 2.26%
    ),

    # Simplified employer cost multiplier on gross salary
    # total_employer_cost = gross × (1 + 0.2109 + 0.0411) = gross × 1.252
    employer_charge_multiplier=1.252,

    # SMIG Jan 2025 (Ministère du Travail)
    smig_mensuel=3_111.39,
    smig_horaire=17.37,

    # TVA (CGI 2025)
    tva=TVARates(
        standard=0.20,
        intermediaire=0.14,
        reduit=0.10,
        super_reduit=0.07,
        exonere=0.0,
    ),

    # Amortissement légaux (CGI 2025)
    amortization=AmortizationSchedule(),

    # Frais de création SARL (OMPIC / CRI 2024)
    frais_immatriculation_sarl_min=3_500,
    frais_immatriculation_sarl_max=5_000,
    capital_minimum_sarl=1,  # Legal minimum; banks often require 10 000+
)

# ── CONSTANTS: 2024 (historical) ─────────────────────────────────────────────
# Same as 2025 for most values — IS brackets were being phased in
FISCAL_2024 = FiscalYear(
    year=2024,
    is_brackets=[
        ISBracket(0,          300_000,    0.10, "≤ 300 000 MAD"),
        ISBracket(300_000,  1_000_000,    0.20, "300 001 – 1 000 000 MAD"),
        ISBracket(1_000_000, 100_000_000, 0.275, "1 000 001 – 100 000 000 MAD"),
        ISBracket(100_000_000, float('inf'), 0.35, "> 100 000 000 MAD"),
    ],
    ir_auto_entrepreneur_achat_revente=0.01,
    ir_auto_entrepreneur_services=0.02,
    cnss=CNSSRates(patronal_part=0.2109, salarial_part=0.0674),
    amo=AMORates(patronale=0.0411, salariale=0.0226),
    employer_charge_multiplier=1.252,
    smig_mensuel=3_111.39,
    smig_horaire=17.37,
    tva=TVARates(),
    amortization=AmortizationSchedule(),
    frais_immatriculation_sarl_min=3_500,
    frais_immatriculation_sarl_max=5_000,
    capital_minimum_sarl=1,
)

# ── REGISTRY ─────────────────────────────────────────────────────────────────

FISCAL_REGISTRY: dict[int, FiscalYear] = {
    2024: FISCAL_2024,
    2025: FISCAL_2025,
}


def get_fiscal_constants(year: int = 2025) -> FiscalYear:
    """
    Returns fiscal constants for the requested year.
    Falls back to most recent year if requested year not available.
    """
    if year in FISCAL_REGISTRY:
        return FISCAL_REGISTRY[year]
    latest = max(FISCAL_REGISTRY.keys())
    return FISCAL_REGISTRY[latest]