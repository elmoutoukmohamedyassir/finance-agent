"""
services/risk_predictor_service.py — Risk / viability prediction service.

WHAT THIS DOES
──────────────
Loads the trained ML model artifact (produced by scripts/train_risk_model.py)
once, on first use, then exposes a single public function:

    result = predict_risk(business_state)

It takes the same flat business_state dict the rest of the app already
uses (the Phase 2 collection output), and returns a RiskPrediction with:

  • label          — "at_risk" | "viable"
  • confidence     — probability of the predicted label (0.0 → 1.0)
  • top_factors    — the 5 features that most influenced this specific
                     prediction, ordered by SHAP importance, with human-readable
                     French labels and the actual values from the input
  • model_info     — model type + test metrics from training time

DESIGN NOTES
────────────
• Lazy singleton: the artifact is loaded on first call, not at import time.
  This means the app starts fine even if the model file doesn't exist yet
  (e.g. during development before the first training run), and the 12 MB
  joblib file is only deserialised once per process lifetime.

• No DB dependency: prediction is pure in-memory inference. The router
  can call this from any context — authenticated or not, with or without
  a DB session.

• Feature order must match what train_risk_model.py saved. The artifact's
  own metadata["feature_names"] is used as the source of truth, so the
  service stays in sync with the model automatically after retraining.

• Missing business_state fields default to 0 / the first categorical
  value rather than crashing — the same tolerant pattern compute_plan()
  uses. A partially-filled state still gets a prediction (just less
  accurate), which is intentional: the API can be called mid-conversation
  to give early risk signals.

• SHAP TreeExplainer is used for per-prediction importance. It receives X
  (the fully encoded, ordered matrix the model saw) — NOT the pre-encoding
  feature_row. Display values are recovered from business_state for
  categoricals and from X for numerics.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.config import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Column definitions — must match train_risk_model.py
# ─────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "segment_client",
    "type_activite",
    "statut_juridique",
    "nature_clients_encaissements",
]

NUMERIC_COLS = [
    "prix_vente_unitaire",
    "nb_clients_mois1",
    "taux_croissance_mensuel",
    "taux_fidelisation",
    "cout_fabrication_unitaire",
    "cout_infra_numerique",
    "loyer_mensuel",
    "salaires_equipe",
    "charges_utilites",
    "budget_marketing",
    "investissements_initiaux",
    "emprunts",
    "own_capital_invested",
    "delai_jours",
]

# Default values when a field is absent from business_state
_CATEGORICAL_DEFAULTS = {
    "segment_client":               "B2C",
    "type_activite":                "service",
    "statut_juridique":             "Auto-entrepreneur",
    "nature_clients_encaissements": "comptant",
}

_NUMERIC_DEFAULTS = {col: 0.0 for col in NUMERIC_COLS}

# French display labels for the feature importance section of the response
_FEATURE_LABELS_FR = {
    "segment_client":               "Segment client (B2C/B2B/Mixte)",
    "type_activite":                "Type d'activité",
    "statut_juridique":             "Statut juridique",
    "nature_clients_encaissements": "Nature des encaissements",
    "prix_vente_unitaire":          "Prix de vente unitaire (MAD)",
    "nb_clients_mois1":             "Nombre de clients au démarrage",
    "taux_croissance_mensuel":      "Taux de croissance mensuel (%)",
    "taux_fidelisation":            "Taux de fidélisation (%)",
    "cout_fabrication_unitaire":    "Coût de fabrication unitaire (MAD)",
    "cout_infra_numerique":         "Coût infrastructure numérique (MAD/mois)",
    "loyer_mensuel":                "Loyer mensuel (MAD)",
    "salaires_equipe":              "Salaires équipe — net (MAD/mois)",
    "charges_utilites":             "Charges utilités (MAD/mois)",
    "budget_marketing":             "Budget marketing (MAD/mois)",
    "investissements_initiaux":     "Investissements initiaux (MAD)",
    "emprunts":                     "Emprunts bancaires (MAD)",
    "own_capital_invested":         "Capital propre investi (MAD)",
    "delai_jours":                  "Délai d'encaissement clients (jours)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Public result types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskFactor:
    """One contributing factor in a risk prediction."""
    feature:        str    # internal column name
    label_fr:       str    # human-readable French label
    value:          object # the actual value from this business_state
    importance:     float  # SHAP absolute importance (0→1 normalised)
    importance_pct: float  # same, as a percentage


@dataclass
class RiskPrediction:
    label:       str          # "at_risk" | "viable"
    confidence:  float        # probability of predicted label (0.0 → 1.0)
    top_factors: list         # list[RiskFactor], ordered by importance desc
    model_info:  dict = field(default_factory=dict)
    error:       Optional[str] = None  # set if prediction failed gracefully


# ─────────────────────────────────────────────────────────────────────────────
# Lazy singleton loader
# ─────────────────────────────────────────────────────────────────────────────

_artifact: Optional[dict] = None   # loaded once, reused forever


def _load_artifact() -> Optional[dict]:
    """
    Load the model artifact from disk on first call, cache it globally.
    Returns None (never raises) if the file is missing or corrupt —
    callers surface a graceful error rather than a 500.
    """
    global _artifact
    if _artifact is not None:
        return _artifact

    settings = get_settings()
    model_path = settings.risk_model_path
    if not os.path.isabs(model_path):
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        )))
        model_path = os.path.join(project_root, model_path)

    if not os.path.exists(model_path):
        logger.warning(
            f"Risk model artifact not found at {model_path}. "
            "Run scripts/train_risk_model.py to generate it."
        )
        return None

    try:
        _artifact = joblib.load(model_path)
        meta = _artifact.get("metadata", {})
        logger.info(
            f"Risk model loaded: {meta.get('model_type')} "
            f"(F1={meta.get('test_metrics', {}).get('f1')}, "
            f"trained {meta.get('trained_at', 'unknown')})"
        )
        return _artifact
    except Exception as exc:
        logger.error(f"Failed to load risk model artifact: {exc}")
        return None


def reload_model() -> bool:
    """
    Force a reload of the model artifact from disk.
    Useful after retraining without restarting the server.
    Returns True if the artifact loaded successfully.
    """
    global _artifact
    _artifact = None
    return _load_artifact() is not None


# ─────────────────────────────────────────────────────────────────────────────
# Feature preparation
# ─────────────────────────────────────────────────────────────────────────────

def _business_state_to_feature_row(business_state: dict) -> pd.DataFrame:
    """
    Extract and order the 18 features from a business_state dict into a
    single-row DataFrame ready for the encoder + model.

    Missing keys fall back to safe defaults rather than raising — the
    model handles partial inputs gracefully (just with reduced accuracy).
    """
    row = {}

    for col in CATEGORICAL_COLS:
        row[col] = business_state.get(col) or _CATEGORICAL_DEFAULTS[col]

    for col in NUMERIC_COLS:
        raw = business_state.get(col)
        try:
            row[col] = float(raw) if raw is not None else _NUMERIC_DEFAULTS[col]
        except (TypeError, ValueError):
            row[col] = _NUMERIC_DEFAULTS[col]

    return pd.DataFrame([row])


def _extract_top_factors(
    artifact: dict,
    X: pd.DataFrame,        # fully encoded + ordered — exactly what model.predict() saw
    business_state: dict,   # raw values, used only for readable display of categoricals
    top_n: int = 5,
) -> list[RiskFactor]:
    """
    Per-prediction feature importance via SHAP TreeExplainer.

    X must be the encoded, feature-ordered DataFrame passed to model.predict()
    — NOT the pre-encoding feature_row. Passing raw (pre-encoding) data to
    SHAP causes it to assign all weight to the first few categorical columns
    because their encoded integer values have the highest variance.

    We sum absolute SHAP values across all classes so importance is
    label-agnostic (same ranking whether the prediction is at_risk or viable).
    """
    import shap

    model    = artifact["model"]
    metadata = artifact.get("metadata", {})
    features = metadata.get("feature_names", CATEGORICAL_COLS + NUMERIC_COLS)

    # Unwrap Pipeline → bare estimator (SHAP needs the classifier directly,
    # not the scaler+classifier Pipeline that LogisticRegression uses)
    estimator = model.named_steps["clf"] if isinstance(model, Pipeline) else model

    explainer   = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X)

    # SHAP's return shape for multiclass varies by version:
    #   - older shap: list of (n_samples, n_features) arrays, one per class
    #   - newer shap (>=0.45ish): single (n_samples, n_features, n_classes) ndarray
    # Handle both so importance extraction doesn't silently mis-rank features
    # when the installed shap version differs from whatever wrote this code.
    if isinstance(shap_values, list):
        # Sum |SHAP| across all classes → single importance score per feature
        importances = np.sum([np.abs(sv[0]) for sv in shap_values], axis=0)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes) — sum abs over classes, take row 0
            importances = np.abs(arr[0]).sum(axis=-1)
        elif arr.ndim == 2:
            # (n_samples, n_features) — single-output case
            importances = np.abs(arr[0])
        else:
            raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")

    total = float(importances.sum()) or 1.0
    ranked = sorted(
        zip(features, importances),
        key=lambda pair: pair[1],
        reverse=True,
    )[:top_n]

    factors = []
    for feat_name, importance in ranked:
        # Categoricals: show the original string (e.g. "B2B") not the integer
        # Numerics: show the value directly from X
        if feat_name in CATEGORICAL_COLS:
            display_value = business_state.get(feat_name, X[feat_name].iloc[0])
        else:
            display_value = X[feat_name].iloc[0]

        factors.append(RiskFactor(
            feature=feat_name,
            label_fr=_FEATURE_LABELS_FR.get(feat_name, feat_name),
            value=display_value,
            importance=round(float(importance), 4),
            importance_pct=round(float(importance) / total * 100, 1),
        ))

    return factors


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def predict_risk(business_state: dict) -> RiskPrediction:
    """
    Predict whether a business is at risk or viable.

    Args:
        business_state: the flat dict collected by Phase 2 (same format
                        used everywhere else in the app — prix_vente_unitaire,
                        nb_clients_mois1, loyer_mensuel, etc.)

    Returns:
        RiskPrediction with label, confidence, top_factors, model_info.
        If the model artifact isn't available, returns a graceful error
        result instead of raising.
    """
    artifact = _load_artifact()
    if artifact is None:
        return RiskPrediction(
            label="unknown",
            confidence=0.0,
            top_factors=[],
            error=(
                "Modèle de prédiction non disponible. "
                "Exécutez scripts/train_risk_model.py pour générer l'artefact."
            ),
        )

    try:
        model    = artifact["model"]
        encoder  = artifact["encoder"]
        metadata = artifact.get("metadata", {})

        # 1. Build feature row (pre-encoding)
        feature_row = _business_state_to_feature_row(business_state)

        # 2. Encode categoricals
        feature_row[CATEGORICAL_COLS] = encoder.transform(feature_row[CATEGORICAL_COLS])

        # 3. Reorder to match training feature order
        ordered_features = metadata.get("feature_names", CATEGORICAL_COLS + NUMERIC_COLS)
        X = feature_row[ordered_features]

        # 4. Predict
        label      = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        classes    = metadata.get("label_classes", ["at_risk", "viable"])
        label_idx  = classes.index(label)
        confidence = round(float(proba[label_idx]), 4)

        # 5. Per-prediction SHAP importance — pass X (encoded), not feature_row
        top_factors = _extract_top_factors(artifact, X, business_state)

        # 6. Model info
        model_info = {
            "model_type":   metadata.get("model_type"),
            "trained_at":   metadata.get("trained_at"),
            "test_metrics": metadata.get("test_metrics", {}),
        }

        logger.info(
            f"Risk prediction: {label} (confidence={confidence:.2%}) "
            f"for business_state with "
            f"prix={business_state.get('prix_vente_unitaire')}, "
            f"clients={business_state.get('nb_clients_mois1')}"
        )

        return RiskPrediction(
            label=label,
            confidence=confidence,
            top_factors=top_factors,
            model_info=model_info,
        )

    except Exception as exc:
        logger.error(f"Risk prediction failed: {exc}", exc_info=True)
        return RiskPrediction(
            label="unknown",
            confidence=0.0,
            top_factors=[],
            error=f"Erreur lors de la prédiction : {str(exc)}",
        )


def get_model_metadata() -> Optional[dict]:
    """
    Return the metadata dict from the loaded artifact, or None if unavailable.
    Used by the router to expose model info without re-loading.
    """
    artifact = _load_artifact()
    if artifact is None:
        return None
    return artifact.get("metadata", {})