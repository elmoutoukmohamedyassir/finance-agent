"""
schemas/risk.py — Risk classifier request/response schemas.

The request mirrors the flat business_state dict used everywhere else in
the app (Phase 2 collection output / plan_pipeline input) — same field
names, same scale, so the same dict can be reused across metrics,
scenario, plan, and risk endpoints without translation.
"""

from typing import Optional
from pydantic import BaseModel, Field


class RiskPredictionRequest(BaseModel):
    # Categorical
    segment_client: Optional[str] = Field(default=None, description="'B2C' | 'B2B' | 'Mixte'")
    type_activite: Optional[str] = Field(default=None, description="'service' | 'produit' | 'hybride'")
    statut_juridique: Optional[str] = Field(default=None, description="'Auto-entrepreneur' | 'SARL' | 'SA'")
    nature_clients_encaissements: Optional[str] = Field(default=None, description="'comptant' | 'credit' | 'mixte'")

    # Revenue
    prix_vente_unitaire: Optional[float] = Field(default=None, ge=0, description="MAD/unit")
    nb_clients_mois1: Optional[float] = Field(default=None, ge=0, description="Starting customer base")
    taux_croissance_mensuel: Optional[float] = Field(default=None, description="% monthly growth")
    taux_fidelisation: Optional[float] = Field(default=None, ge=0, le=100, description="% monthly retention")

    # Variable cost
    cout_fabrication_unitaire: Optional[float] = Field(default=None, ge=0, description="MAD/unit")

    # Fixed costs (monthly, MAD)
    cout_infra_numerique: Optional[float] = Field(default=None, ge=0)
    loyer_mensuel: Optional[float] = Field(default=None, ge=0)
    salaires_equipe: Optional[float] = Field(default=None, ge=0)
    charges_utilites: Optional[float] = Field(default=None, ge=0)
    budget_marketing: Optional[float] = Field(default=None, ge=0)

    # One-time
    investissements_initiaux: Optional[float] = Field(default=None, ge=0)

    # Financing
    emprunts: Optional[float] = Field(default=None, ge=0)
    own_capital_invested: Optional[float] = Field(default=None, ge=0)

    # Payment terms
    delai_jours: Optional[float] = Field(default=None, ge=0, description="Days delay for credit clients")

    model_config = {
        "json_schema_extra": {
            "example": {
                "segment_client": "B2B",
                "type_activite": "service",
                "statut_juridique": "SARL",
                "nature_clients_encaissements": "credit",
                "prix_vente_unitaire": 2000,
                "nb_clients_mois1": 10,
                "taux_croissance_mensuel": 5.0,
                "taux_fidelisation": 80.0,
                "cout_fabrication_unitaire": 100,
                "cout_infra_numerique": 500,
                "loyer_mensuel": 8000,
                "salaires_equipe": 20000,
                "charges_utilites": 1000,
                "budget_marketing": 3000,
                "investissements_initiaux": 50000,
                "emprunts": 150000,
                "own_capital_invested": 30000,
                "delai_jours": 60,
            }
        }
    }


class RiskFactorOut(BaseModel):
    feature: str
    label_fr: str
    value: object
    importance: float
    importance_pct: float


class RiskPredictionResponse(BaseModel):
    label: str                          # "at_risk" | "viable" | "unknown"
    confidence: float
    top_factors: list[RiskFactorOut]
    model_info: dict = Field(default_factory=dict)
    error: Optional[str] = None