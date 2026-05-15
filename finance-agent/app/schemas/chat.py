"""
schemas/chat.py — Chat request/response schemas.

Updated to:
  - Remove SaaS example from ChatRequest
  - Add hypothesis_payload for direct agent-to-agent ingestion
  - Add plan_output for 24-month plan results
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        description="Reuse across messages to keep conversation context. Omit for a new session."
    )
    message: str = Field(..., min_length=1, max_length=5000)

    # Optional: direct Hypothesis Agent JSON payload (skips conversation collection)
    hypothesis_payload: Optional[dict] = Field(
        default=None,
        description="Full HypothesisOutput JSON from the Hypothesis Agent. "
                    "When provided, bypasses the conversational data collection phase "
                    "and goes directly to financial analysis."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "title": "Chat conversationnel",
                    "value": {
                        "session_id": "abc-123",
                        "message": "Je gère une SARL dans le secteur BTP à Casablanca. "
                                   "Notre CA annuel est de 12 MMAD."
                    }
                },
                {
                    "title": "Ingestion directe depuis Hypothesis Agent",
                    "value": {
                        "session_id": "abc-456",
                        "message": "Analyse complète de notre projet",
                        "hypothesis_payload": {
                            "ventes": {"H1_segment_client": "B2B", "H2_prix_vente_unitaire": 2500},
                            "achats": {"H8_type_activite": "service"},
                            "charges_fixes": {"H13_loyer_mensuel": 4500, "H14_salaires_equipe": 18000},
                            "encaissements": {"H22_nature_clients": "credit", "delai_jours": 30},
                            "metadata": {"secteur": "conseil", "region": "Casablanca"}
                        }
                    }
                }
            ]
        }
    }


class ChatResponse(BaseModel):
    session_id: str
    message: str
    agent_mode: str = Field(
        description="gathering_info | analyzing | answering"
    )
    metrics_calculated: Optional[dict] = None
    business_state: Optional[dict] = None
    plan_output: Optional[dict] = Field(
        default=None,
        description="24-month business plan output when generated"
    )