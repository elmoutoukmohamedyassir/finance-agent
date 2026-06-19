"""
schemas/hypothesis_output.py — Typed contract for Hypothesis Agent → Finance Agent.

This is the EXACT schema the Hypothesis Agent must output.
The Finance Agent validates this on ingestion — wrong types fail at the boundary,
not silently mid-calculation.

Field naming follows the docx specification (H1_... H22_...) exactly
so the two agents stay in sync without a translation layer.

All monetary values in MAD (single units, not millions — these come from
user answers which are typically small business scale).
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Bloc 1: User answers from Hypothesis Agent ──────────────────────────────

class BlocVentes(BaseModel):
    """H1–H7: Sales and revenue assumptions."""
    H1_segment_client: Literal["B2C", "B2B", "Mixte"] = "B2C"
    H2_prix_vente_unitaire: Optional[float] = Field(default=None, ge=0, description="MAD")
    H3_abonnement_mensuel: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H4_nb_clients_mois1: Optional[float] = Field(default=None, ge=0, description="Starting clients")
    H5_taux_croissance_mensuel: Optional[float] = Field(default=None, description="% monthly growth")
    H6_taux_fidelisation: Optional[float] = Field(default=None, ge=0, le=100, description="% retention")
    H7_saisonnalite: Optional[dict] = Field(default=None, description="Monthly coefficients {1: 0.8, 2: 1.0, ...}")


class BlocAchats(BaseModel):
    """H8–H12: Procurement and variable costs."""
    H8_type_activite: Literal["service", "produit", "hybride"] = "service"
    H9_cout_fabrication_unitaire: Optional[float] = Field(default=None, ge=0, description="MAD/unit")
    H10_quantite_min_commande: Optional[float] = Field(default=None, ge=0, description="Units")
    H11_cout_infra_numerique: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H12_delai_fournisseur_jours: Optional[float] = Field(default=None, ge=0, description="Days")


class BlocChargesFixes(BaseModel):
    """H13–H21: Fixed charges and initial investment."""
    H13_loyer_mensuel: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H14_salaires_equipe: Optional[float] = Field(default=None, ge=0, description="MAD/month net")
    H15_charges_utilites: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H16_licences_logicielles: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H17_budget_marketing: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H18_honoraires_conseil: Optional[float] = Field(default=None, ge=0, description="MAD/month")
    H19_investissements_initiaux: Optional[float] = Field(default=None, ge=0, description="MAD one-time")
    H20_certifications: Optional[float] = Field(default=None, ge=0, description="MAD one-time")
    H21_emprunts: Optional[float] = Field(default=None, ge=0, description="MAD total loan")


class BlocEncaissements(BaseModel):
    """H22: Payment collection terms."""
    H22_nature_clients: Literal["comptant", "credit", "mixte"] = "comptant"
    delai_jours: int = Field(default=0, ge=0, le=180, description="Days payment delay for credit clients")


class HypothesisMetadata(BaseModel):
    """Context fields transmitted alongside hypothesis variables."""
    description_projet: Optional[str] = None
    region: Optional[str] = None         # e.g. "Casablanca", "Rabat", "Tanger"
    secteur: Optional[str] = None        # e.g. "restauration", "tech", "BTP"
    statut_juridique: Optional[str] = None  # "SARL", "SA", "auto-entrepreneur", "SAS"
    capital_social: Optional[float] = Field(default=None, ge=0, description="MAD")

    # Pre-computed validation results from Hypothesis Agent
    charges_fixes_mensuelles: Optional[float] = Field(default=None, description="Pre-validated MAD/month")
    marge_unitaire: Optional[float] = Field(default=None, description="Pre-validated MAD/unit")
    seuil_clients_minimum: Optional[float] = Field(default=None, description="Pre-validated break-even clients")


class HypothesisOutput(BaseModel):
    """
    Full output contract from the Hypothesis Agent.
    Finance Agent ingests this as its primary input.

    The Finance Agent MUST validate this with .model_validate() before use.
    Any missing required fields are surfaced as ValidationError at the boundary.
    """
    # Required
    ventes: BlocVentes
    achats: BlocAchats
    charges_fixes: BlocChargesFixes
    encaissements: BlocEncaissements
    metadata: HypothesisMetadata

    # Pre-calculated scenarios (if Hypothesis Agent already ran them)
    scenarios_pre_calcules: Optional[list[dict]] = Field(
        default=None,
        description="If provided, Finance Agent uses these as starting point rather than recalculating"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "ventes": {
                    "H1_segment_client": "B2B",
                    "H2_prix_vente_unitaire": 2500,
                    "H4_nb_clients_mois1": 3,
                    "H5_taux_croissance_mensuel": 8,
                    "H6_taux_fidelisation": 85
                },
                "achats": {
                    "H8_type_activite": "service",
                    "H9_cout_fabrication_unitaire": 0,
                    "H11_cout_infra_numerique": 800
                },
                "charges_fixes": {
                    "H13_loyer_mensuel": 4500,
                    "H14_salaires_equipe": 18000,
                    "H15_charges_utilites": 600,
                    "H17_budget_marketing": 3000,
                    "H19_investissements_initiaux": 45000,
                    "H21_emprunts": 100000
                },
                "encaissements": {
                    "H22_nature_clients": "credit",
                    "delai_jours": 30
                },
                "metadata": {
                    "description_projet": "Agence de consulting RH pour PME marocaines",
                    "region": "Casablanca",
                    "secteur": "conseil / services aux entreprises",
                    "statut_juridique": "SARL",
                    "capital_social": 50000
                }
            }
        }
    }