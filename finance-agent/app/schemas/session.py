"""
schemas/session.py — Session with explicit Phase enum.

Phase drives the entire agent flow:
  WELCOME → COLLECTING → AWAITING_PLAN_CONFIRM → PRE_CREATION → CREATION → POST_CREATION
"""
from datetime import datetime
from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field


class Phase(str, Enum):
    WELCOME               = "welcome"
    COLLECTING            = "collecting"
    AWAITING_PLAN_CONFIRM = "awaiting_plan_confirm"
    PRE_CREATION          = "pre_creation"
    CREATION              = "creation"
    POST_CREATION         = "post_creation"


class BusinessState(BaseModel):
    # Identity
    entity_name:   Optional[str]   = None
    entity_type:   Optional[str]   = None
    sector:        Optional[str]   = None
    fiscal_year:   Optional[int]   = None
    statut_juridique: Optional[str] = None
    # H1-H7
    segment_client:          Optional[str]   = None
    prix_vente_unitaire:     Optional[float] = None
    abonnement_mensuel:      Optional[float] = None
    nb_clients_mois1:        Optional[float] = None
    taux_croissance_mensuel: Optional[float] = None
    taux_fidelisation:       Optional[float] = None
    saisonnalite:            Optional[str]   = None
    # H8-H12
    type_activite:               Optional[str]   = None
    cout_fabrication_unitaire:   Optional[float] = None
    quantite_min_commande:       Optional[float] = None
    cout_infra_numerique:        Optional[float] = None
    delai_fournisseur_jours:     Optional[float] = None
    # H13-H21
    loyer_mensuel:            Optional[float] = None
    salaires_equipe:          Optional[float] = None
    charges_utilites:         Optional[float] = None
    licences_logicielles:     Optional[float] = None
    budget_marketing:         Optional[float] = None
    honoraires_conseil:       Optional[float] = None
    investissements_initiaux: Optional[float] = None
    certifications:           Optional[float] = None
    emprunts:                 Optional[float] = None
    # H22
    nature_clients_encaissements: Optional[str]   = None
    delai_jours:                  Optional[int]   = None
    own_capital_invested:         Optional[float] = None
    # Enterprise / post-creation
    total_revenue:             Optional[float] = None
    revenue_year2:             Optional[float] = None
    tax_revenue:               Optional[float] = None
    non_tax_revenue:           Optional[float] = None
    grants_and_transfers:      Optional[float] = None
    cost_of_goods_sold:        Optional[float] = None
    operating_expenses:        Optional[float] = None
    salaries_and_benefits:     Optional[float] = None
    depreciation_amortization: Optional[float] = None
    interest_expense:          Optional[float] = None
    tax_expense:               Optional[float] = None
    total_expenditure:         Optional[float] = None
    capital_expenditure:       Optional[float] = None
    recurrent_expenditure:     Optional[float] = None
    debt_service:              Optional[float] = None
    subsidies_paid:            Optional[float] = None
    total_assets:              Optional[float] = None
    current_assets:            Optional[float] = None
    current_liabilities:       Optional[float] = None
    total_equity:              Optional[float] = None
    total_debt:                Optional[float] = None
    cash_and_equivalents:      Optional[float] = None
    cash_inflow:               Optional[float] = None
    cash_outflow:              Optional[float] = None
    operating_cash_flow:       Optional[float] = None
    external_funding:          Optional[float] = None
    investment_budget:         Optional[float] = None
    investment_executed:       Optional[float] = None
    months:                    int             = 12

    def filled_fields(self) -> dict:
        return {k: v for k, v in self.model_dump().items() if v is not None}

    def has_revenue_info(self) -> bool:
        return bool(
            self.total_revenue
            or (self.prix_vente_unitaire and self.nb_clients_mois1)
            or (self.tax_revenue and self.non_tax_revenue)
        )

    def has_cost_info(self) -> bool:
        return any(v is not None for v in [
            self.operating_expenses, self.cost_of_goods_sold,
            self.salaries_and_benefits, self.salaires_equipe,
            self.loyer_mensuel, self.total_expenditure,
        ])

    def is_ready_for_analysis(self) -> bool:
        return bool(self.entity_type and self.has_revenue_info() and self.has_cost_info())


class ConversationSession(BaseModel):
    session_id:   str
    created_at:   datetime = Field(default_factory=datetime.utcnow)
    updated_at:   datetime = Field(default_factory=datetime.utcnow)
    phase:        Phase    = Phase.WELCOME

    conversation_history: list[dict]  = Field(default_factory=list)
    business_state:       BusinessState = Field(default_factory=BusinessState)
    questions_asked:      list[str]   = Field(default_factory=list)
    pending_question:     Optional[str] = None

    # Runtime-only (not persisted to DB)
    cached_plan:       Optional[Any] = Field(default=None, exclude=True)
    cached_metrics:    Optional[dict] = Field(default=None, exclude=True)
    derived_variables: Optional[Any] = Field(default=None, exclude=True)
    projection_inputs: Optional[Any] = Field(default=None, exclude=True)

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})
        self.updated_at = datetime.utcnow()