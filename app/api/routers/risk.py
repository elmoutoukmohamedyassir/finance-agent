"""
api/routers/risk.py — Risk / viability prediction endpoint.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.risk import RiskFactorOut, RiskPredictionRequest, RiskPredictionResponse
from app.services.risk_predictor_service import get_model_metadata, predict_risk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/risk", tags=["Risk"])


@router.post("/predict", response_model=RiskPredictionResponse)
def predict(request: RiskPredictionRequest):
    """
    Predict whether a business is at_risk or viable from a (possibly
    partial) business_state, and return the top contributing factors.

    Pure ML inference — no LLM involved. If the model artifact hasn't
    been trained yet, returns label="unknown" with an explanatory error
    instead of a 500.
    """
    try:
        business_state = request.model_dump(exclude_none=True)
        result = predict_risk(business_state)

        return RiskPredictionResponse(
            label=result.label,
            confidence=result.confidence,
            top_factors=[
                RiskFactorOut(
                    feature=f.feature,
                    label_fr=f.label_fr,
                    value=f.value,
                    importance=f.importance,
                    importance_pct=f.importance_pct,
                )
                for f in result.top_factors
            ],
            model_info=result.model_info,
            error=result.error,
        )
    except Exception as e:
        logger.error(f"Risk prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prédiction de risque échouée")


@router.get("/model-info")
def model_info():
    """
    Return metadata about the currently loaded model artifact
    (model type, training timestamp, test metrics) — or a 404-style
    payload if no model has been trained yet.
    """
    meta = get_model_metadata()
    if meta is None:
        return {
            "available": False,
            "message": "Aucun modèle entraîné. Exécutez scripts/train_risk_model.py.",
        }
    return {"available": True, **meta}