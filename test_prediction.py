import sys
sys.path.insert(0, ".")
from app.services.risk_predictor_service import predict_risk

bs = {
    "segment_client": "B2B",
    "type_activite": "service",
    "statut_juridique": "SARL",
    "nature_clients_encaissements": "comptant",
    "prix_vente_unitaire": 5000,    # higher price
    "nb_clients_mois1": 20,         # more clients
    "taux_croissance_mensuel": 8.0,
    "taux_fidelisation": 90.0,
    "cout_fabrication_unitaire": 0,
    "cout_infra_numerique": 500,
    "loyer_mensuel": 3000,
    "salaires_equipe": 8000,
    "charges_utilites": 500,
    "budget_marketing": 1000,
    "investissements_initiaux": 20000,
    "emprunts": 0,
    "own_capital_invested": 50000,
    "delai_jours": 0,
}

result = predict_risk(bs)
print(f"Label      : {result.label}")
print(f"Confidence : {result.confidence:.0%}")
print(f"Error      : {result.error}")
print("Top factors:")
for f in result.top_factors:
    print(f"  {f.label_fr} = {f.value}  ({f.importance_pct:.1f}%)")