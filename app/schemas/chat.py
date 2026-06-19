"""schemas/chat.py"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """Chat request with optional client email and hypothesis data."""
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for multi-turn conversations"
    )
    message: str = Field(
        ..., 
        min_length=1, 
        max_length=5000,
        description="The user's message or question"
    )
    client_email: Optional[str] = Field(
        default=None,
        description="Client email for identification and database tracking (optional)"
    )
    hypothesis_payload: Optional[dict] = Field(
        default=None,
        description="Full financial data from Hypothesis Agent - bypasses manual collection"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "message": "I want to start a consulting business in Casablanca",
                    "description": "Basic chat - agent responds with financial questions"
                },
                {
                    "message": "Calculate my KPIs",
                    "client_email": "entrepreneur@startup.ma",
                    "description": "With client tracking - saves email to database"
                },
                {
                    "message": "Analyze my business plan",
                    "client_email": "user@example.com",
                    "hypothesis_payload": {
                        "revenue_projected_m1": 50000,
                        "revenue_projected_m6": 150000,
                        "costs_monthly": 30000,
                        "employees": 2,
                        "capital_invested": 100000,
                        "sector": "Consulting"
                    },
                    "description": "Fast-track with hypothesis data - skips questions"
                }
            ]
        }


class ChatResponse(BaseModel):
    """Chat response with all new features."""
    session_id: str = Field(
        default="",
        description="Session ID for conversation continuity"
    )
    message: str = Field(
        description="The agent's response message"
    )
    agent_mode: str = Field(
        default="collecting",
        description="Current mode: collecting, analyzing, or planning"
    )
    current_phase: str = Field(
        default="welcome",
        description="Current conversation phase"
    )
    metrics_calculated: Optional[dict] = Field(
        default=None,
        description="Financial metrics calculated"
    )
    business_state: Optional[dict] = Field(
        default=None,
        description="Current business state data"
    )
    plan_output: Optional[dict] = Field(
        default=None,
        description="Generated business plan"
    )
    
    # New fields for enhanced responses
    kpi_suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggested KPIs the agent recommends calculating"
    )
    kpi_details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Calculated KPIs with values, units, and explanations (bilingual FR/EN)"
    )
    action_items: Optional[List[str]] = Field(
        default=None,
        description="Recommended next steps and action items"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata: llm_used (groq/gemini), rag_confidence (0-1)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "session_id": "abc-123",
                    "message": "Pour un consultant en stratégie, je recommande de calculer votre seuil de rentabilité...",
                    "agent_mode": "collecting",
                    "kpi_suggestions": [
                        "Seuil de rentabilité (Break-even)",
                        "Ratio de profitabilité",
                        "Flux de trésorerie mensuel"
                    ],
                    "metadata": {
                        "llm_used": "groq",
                        "rag_confidence": 0.85
                    },
                    "description": "Response with KPI suggestions"
                },
                {
                    "session_id": "abc-123",
                    "message": "Voici vos KPIs calculés basés sur vos données...",
                    "agent_mode": "analyzing",
                    "kpi_details": {
                        "seuil_rentabilite": {
                            "value": 150,
                            "unit": "clients/mois",
                            "explanation": "Nombre de clients nécessaires mensuellement pour couvrir vos coûts fixes et variables"
                        },
                        "ratio_profitabilite": {
                            "value": 32.5,
                            "unit": "%",
                            "explanation": "Pourcentage de chaque euro de chiffre d'affaires qui devient profit après tous les coûts"
                        }
                    },
                    "action_items": [
                        "Confirmer les coûts mensuels estimés",
                        "Planifier votre stratégie marketing pour atteindre 150 clients",
                        "Préparer le plan de financement initial"
                    ],
                    "metadata": {
                        "llm_used": "groq",
                        "rag_confidence": 0.92
                    },
                    "description": "Response with KPI details and actions"
                }
            ]
        }
