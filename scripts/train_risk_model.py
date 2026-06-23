"""
scripts/train_risk_model.py — Train and evaluate the FinAgent risk classifier.

WHAT THIS DOES
──────────────
1. Load the CSV produced by generate_training_data.py
2. Encode the 4 categorical columns with OrdinalEncoder (fit on train, apply to test)
3. Split 80/20 train/test, stratified so both halves keep the ~50/50 label balance
4. Train two models: Logistic Regression (interpretable baseline) and Random Forest
5. Evaluate both — accuracy, precision, recall, F1, confusion matrix, feature importances
6. Save the best model + its encoder together as a single joblib artifact

SAVED ARTIFACT  (data/ml/risk_model.joblib)
────────────────────────────────────────────
A dict with three keys — everything the prediction service needs:
  {
    "model":    <fitted classifier>,
    "encoder":  <fitted OrdinalEncoder for the 4 categorical columns>,
    "metadata": {
        "model_type":       "RandomForest" | "LogisticRegression",
        "feature_names":    [...],          # ordered list — must match prediction input
        "categorical_cols": [...],          # which columns go through the encoder
        "numeric_cols":     [...],
        "label_classes":    ["at_risk", "viable"],
        "train_size":       N,
        "test_metrics":     {accuracy, precision, recall, f1},
        "trained_at":       "ISO timestamp",
    }
  }

USAGE
─────
  python scripts/train_risk_model.py                            # defaults
  python scripts/train_risk_model.py --input data/ml/training_data.csv
  python scripts/train_risk_model.py --output data/ml/risk_model.joblib
  python scripts/train_risk_model.py --model logistic           # force logistic regression
"""

import argparse
import os
import sys
from datetime import datetime, timezone

# ── Path fix ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib                                           # noqa: E402
import pandas as pd                                     # noqa: E402
from sklearn.ensemble import RandomForestClassifier     # noqa: E402
from sklearn.linear_model import LogisticRegression     # noqa: E402
from sklearn.metrics import (                           # noqa: E402
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split    # noqa: E402
from sklearn.pipeline import Pipeline                   # noqa: E402
from sklearn.preprocessing import OrdinalEncoder, StandardScaler  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Column definitions — must stay in sync with generate_training_data.py
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

# Feature order for the final matrix — categoricals first, then numerics.
# The prediction service must feed features in this exact order.
ALL_FEATURES = CATEGORICAL_COLS + NUMERIC_COLS

LABEL_COL = "label"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    print(f"\n{'━' * 55}")
    print(f"  {title}")
    print(f"{'━' * 55}")


def _build_feature_matrix(df: pd.DataFrame, encoder: OrdinalEncoder, fit: bool) -> pd.DataFrame:
    """
    Encode categoricals and return a single feature DataFrame with ALL_FEATURES columns.
    fit=True  → fit the encoder on df then transform (training set).
    fit=False → transform only (test set / inference).
    """
    df = df.copy()

    if fit:
        df[CATEGORICAL_COLS] = encoder.fit_transform(df[CATEGORICAL_COLS])
    else:
        df[CATEGORICAL_COLS] = encoder.transform(df[CATEGORICAL_COLS])

    return df[ALL_FEATURES]


def _confusion_matrix_str(cm, class_names) -> str:
    """Pretty-print a 2×2 confusion matrix."""
    w = max(len(n) for n in class_names) + 2
    header = " " * (w + 2) + "  ".join(f"{n:>{w}}" for n in class_names)
    rows = []
    for i, name in enumerate(class_names):
        cells = "  ".join(f"{cm[i][j]:>{w}}" for j in range(len(class_names)))
        rows.append(f"  {name:>{w}}  {cells}")
    return "\n".join([header] + rows)


def _feature_importances_str(model, feature_names: list, top_n: int = 10) -> str:
    """Return a ranked importance table (Random Forest) or coefficient table (LR)."""
    # Unwrap Pipeline to get the actual estimator
    estimator = model["clf"] if isinstance(model, Pipeline) else model

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
        label = "Importance"
    else:
        # Logistic Regression: use absolute coefficient of the positive class
        importances = abs(estimator.coef_[0])
        label = "  |Coef|"

    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    lines = [f"  {'Feature':<35} {label}"]
    lines.append(f"  {'-'*35} {'-'*10}")
    for name, score in ranked[:top_n]:
        bar = "█" * int(score * 40 / ranked[0][1]) if ranked[0][1] > 0 else ""
        lines.append(f"  {name:<35} {score:>8.4f}  {bar}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def train(
    input_path: str,
    output_path: str,
    model_choice: str,   # "auto" | "random_forest" | "logistic"
    test_size: float,
    random_state: int,
) -> None:

    # ── 1. Load ──────────────────────────────────────────────────────────────
    _print_section("1. Loading data")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    label_counts = df[LABEL_COL].value_counts()
    for label, count in label_counts.items():
        print(f"  {label:<12} {count:>6,}  ({count/len(df)*100:.1f}%)")

    X = df[ALL_FEATURES]
    y = df[LABEL_COL]

    # ── 2. Train/test split ───────────────────────────────────────────────────
    _print_section("2. Train / test split")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,   # preserve label balance in both halves
    )
    print(f"  Train: {len(X_train_raw):,} rows  |  Test: {len(X_test_raw):,} rows")
    print(f"  Stratified split — label balance preserved in both halves")

    # ── 3. Encode categoricals ────────────────────────────────────────────────
    _print_section("3. Encoding categorical features")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train = _build_feature_matrix(X_train_raw, encoder, fit=True)
    X_test  = _build_feature_matrix(X_test_raw,  encoder, fit=False)

    for col in CATEGORICAL_COLS:
        cats = encoder.categories_[CATEGORICAL_COLS.index(col)]
        print(f"  {col:<40} {list(cats)}")

    # ── 4. Train models ───────────────────────────────────────────────────────
    _print_section("4. Training models")

    candidates = {}

    if model_choice in ("auto", "random_forest"):
        print("  Training Random Forest …", end=" ", flush=True)
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,         # let trees grow fully
            min_samples_leaf=5,     # light regularisation to avoid memorising noise
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_f1 = f1_score(y_test, rf.predict(X_test), pos_label="at_risk")
        candidates["RandomForest"] = (rf, rf_f1)
        print(f"done  (test F1={rf_f1:.4f})")

    if model_choice in ("auto", "logistic"):
        print("  Training Logistic Regression …", end=" ", flush=True)
        # LR needs scaled features to converge — wrap in a pipeline so the
        # scaler is invisible to the rest of the code (predict/predict_proba
        # still work the same way on raw encoded X)
        lr = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ])
        lr.fit(X_train, y_train)
        lr_f1 = f1_score(y_test, lr.predict(X_test), pos_label="at_risk")
        candidates["LogisticRegression"] = (lr, lr_f1)
        print(f"done  (test F1={lr_f1:.4f})")

    # ── 5. Pick the best model ────────────────────────────────────────────────
    _print_section("5. Model selection")
    best_name, (best_model, best_f1) = max(candidates.items(), key=lambda kv: kv[1][1])
    print(f"  Winner: {best_name}  (F1={best_f1:.4f})")
    if len(candidates) > 1:
        for name, (_, f1) in candidates.items():
            marker = " ✓" if name == best_name else ""
            print(f"    {name:<25} F1={f1:.4f}{marker}")

    # ── 6. Full evaluation of the winner ──────────────────────────────────────
    _print_section("6. Evaluation on test set")
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label="at_risk")
    rec  = recall_score(y_test, y_pred, pos_label="at_risk")
    f1   = f1_score(y_test, y_pred, pos_label="at_risk")

    print(f"\n  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}  (of predicted at_risk, how many really are?)")
    print(f"  Recall    : {rec:.4f}  (of real at_risk cases, how many did we catch?)")
    print(f"  F1        : {f1:.4f}  (harmonic mean of precision & recall)")

    print(f"\n  Confusion matrix (rows=actual, cols=predicted):")
    classes = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(_confusion_matrix_str(cm, classes))

    print(f"\n  Full classification report:")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))

    print(f"  Feature importances (top 10):")
    print(_feature_importances_str(best_model, ALL_FEATURES))

    # ── 7. Save artifact ─────────────────────────────────────────────────────
    _print_section("7. Saving model artifact")
    artifact = {
        "model":   best_model,
        "encoder": encoder,
        "metadata": {
            "model_type":       best_name,
            "feature_names":    ALL_FEATURES,
            "categorical_cols": CATEGORICAL_COLS,
            "numeric_cols":     NUMERIC_COLS,
            "label_classes":    classes,
            "train_size":       len(X_train),
            "test_metrics": {
                "accuracy":  round(acc, 4),
                "precision": round(prec, 4),
                "recall":    round(rec, 4),
                "f1":        round(f1, 4),
            },
            "trained_at": datetime.now(timezone.utc).isoformat(),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(artifact, output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  Saved → {output_path}  ({size_kb:.0f} KB)")

    # ── 8. Smoke-test: reload and predict one row ─────────────────────────────
    _print_section("8. Smoke test — reload artifact and predict one row")
    loaded    = joblib.load(output_path)
    sample_df = X_test_raw.iloc[:1].copy()
    sample_X  = _build_feature_matrix(sample_df, loaded["encoder"], fit=False)
    pred      = loaded["model"].predict(sample_X)[0]
    prob      = loaded["model"].predict_proba(sample_X)[0]
    prob_dict = dict(zip(loaded["metadata"]["label_classes"], prob.round(4)))
    print(f"  Input (first test row):")
    for col in CATEGORICAL_COLS:
        print(f"    {col:<40} {sample_df[col].values[0]}")
    for col in NUMERIC_COLS[:5]:   # just show a few
        print(f"    {col:<40} {sample_df[col].values[0]}")
    print(f"  Prediction : {pred}")
    print(f"  Confidence : {prob_dict}")
    print(f"\n  ✅ Artifact loads and predicts correctly.")

    _print_section("Done")
    print(f"  Model      : {best_name}")
    print(f"  F1 (test)  : {f1:.4f}")
    print(f"  Artifact   : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the FinAgent risk classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", default="data/ml/training_data.csv",
        help="Path to CSV from generate_training_data.py (default: data/ml/training_data.csv)",
    )
    parser.add_argument(
        "--output", default="data/ml/risk_model.joblib",
        help="Where to save the model artifact (default: data/ml/risk_model.joblib)",
    )
    parser.add_argument(
        "--model", choices=["auto", "random_forest", "logistic"], default="auto",
        help="Which model(s) to train. 'auto' trains both and picks the best (default: auto)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data held out for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    def resolve(p):
        return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)

    train(
        input_path=resolve(args.input),
        output_path=resolve(args.output),
        model_choice=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()