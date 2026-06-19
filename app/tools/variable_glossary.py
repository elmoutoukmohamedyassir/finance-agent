"""
tools/variable_glossary.py — Concise explanations for all financial variables and KPIs.

Format: Short 1-2 sentence explanations, not paragraphs.
Languages: French and English
"""

# KPI and Variable Glossary
GLOSSARY = {
    # Sales & Revenue
    "chiffre_affaires": {
        "fr": "Chiffre d'affaires: Revenu total attendu from your products or services",
        "en": "Revenue: Total income expected from your products or services"
    },
    "prix_vente_unitaire": {
        "fr": "Prix de vente unitaire: Price per unit you will charge customers",
        "en": "Unit selling price: Price per unit you will charge customers"
    },
    "segment_client": {
        "fr": "Segment client: Target customer group for your services",
        "en": "Customer segment: Target customer group for your services"
    },
    "nb_clients_mois1": {
        "fr": "Nombre de clients mois 1: Number of customers expected in first month",
        "en": "Number of clients month 1: Expected number of customers in first month"
    },
    "taux_croissance_mensuel": {
        "fr": "Taux de croissance mensuel: Expected monthly percentage increase in customer base",
        "en": "Monthly growth rate: Expected monthly percentage increase in customer base"
    },
    "taux_fidelisation": {
        "fr": "Taux de fidelisation: Percentage of customers who return each month",
        "en": "Customer retention rate: Percentage of customers who return each month"
    },

    # Costs & Expenses
    "type_activite": {
        "fr": "Type d'activité: Main business activity category (product/service/hybrid)",
        "en": "Activity type: Main business activity category (product/service/hybrid)"
    },
    "loyer_mensuel": {
        "fr": "Loyer mensuel: Monthly rent for your business location",
        "en": "Monthly rent: Monthly rent for your business location"
    },
    "salaires_equipe": {
        "fr": "Salaires équipe: Total monthly salary cost for all employees",
        "en": "Team salaries: Total monthly salary cost for all employees"
    },
    "investissements_initiaux": {
        "fr": "Investissements initiaux: Equipment, furniture, technology setup costs at launch",
        "en": "Initial investments: Equipment, furniture, technology setup costs at launch"
    },

    # Financing
    "emprunts": {
        "fr": "Emprunts: Loan amount you plan to borrow to start the business",
        "en": "Loans: Loan amount you plan to borrow to start the business"
    },
    "own_capital_invested": {
        "fr": "Capital investi: Your own money invested in the business",
        "en": "Own capital: Your own money invested in the business"
    },
    "external_funding": {
        "fr": "Financement externe: External funding beyond loans (grants, investors)",
        "en": "External funding: External funding beyond loans (grants, investors)"
    },

    # Calculated KPIs
    "seuil_rentabilite": {
        "fr": "Seuil de rentabilité: Number of clients or units needed monthly to break even",
        "en": "Break-even point: Number of clients or units needed monthly to cover costs"
    },
    "marge_brute": {
        "fr": "Marge brute: Percentage of revenue left after direct product costs",
        "en": "Gross margin: Percentage of revenue left after direct product costs"
    },
    "marge_operationnelle": {
        "fr": "Marge opérationnelle: Percentage of revenue left after all operating costs",
        "en": "Operating margin: Percentage of revenue left after all operating costs"
    },
    "marge_nette": {
        "fr": "Marge nette: Final profit percentage after all expenses and taxes",
        "en": "Net margin: Final profit percentage after all expenses and taxes"
    },
    "besoin_en_fonds_de_roulement": {
        "fr": "Besoin en fonds de roulement: Working capital needed to operate daily business",
        "en": "Working capital needs: Cash needed to operate daily business"
    },
    "cash_flow_mensuel": {
        "fr": "Cash-flow mensuel: Actual cash coming in minus going out each month",
        "en": "Monthly cash flow: Actual cash coming in minus going out each month"
    },
    "point_equilibre_mois": {
        "fr": "Point d'équilibre: Month when cumulative cash flow becomes positive",
        "en": "Break-even month: Month when cumulative cash flow becomes positive"
    },
    "taux_rendement_investissement": {
        "fr": "Taux de rendement: Percentage return on your invested capital annually",
        "en": "Return on investment: Percentage return on your invested capital annually"
    },
    "ratio_endettement": {
        "fr": "Ratio d'endettement: Total debt compared to total capital (lower is safer)",
        "en": "Debt ratio: Total debt compared to total capital (lower is safer)"
    },
    "couverture_interet": {
        "fr": "Couverture d'intérêt: How many times profit can cover interest payments",
        "en": "Interest coverage: How many times profit can cover interest payments"
    },

    # Metadata
    "entity_name": {
        "fr": "Nom de l'entité: Business name or legal entity name",
        "en": "Entity name: Business name or legal entity name"
    },
    "sector": {
        "fr": "Secteur: Industry or business sector",
        "en": "Sector: Industry or business sector"
    },
    "statut_juridique": {
        "fr": "Statut juridique: Legal structure (SARL, EIRL, Association, etc.)",
        "en": "Legal status: Legal structure (SARL, EIRL, Association, etc.)"
    },
}


def get_explanation(variable_name: str, language: str = "fr") -> dict:
    """
    Get KPI/variable explanation.
    
    Args:
        variable_name: Key from glossary (e.g., 'seuil_rentabilite')
        language: 'fr' or 'en'
    
    Returns:
        {"name": "...", "explanation": "..."}
        or {"error": "Unknown variable"} if not found
    """
    if variable_name not in GLOSSARY:
        return {"error": f"Unknown variable: {variable_name}"}
    
    text = GLOSSARY[variable_name].get(language, GLOSSARY[variable_name].get("en", "No explanation available"))
    return {
        "name": variable_name,
        "explanation": text,
        "language": language
    }


def format_kpi_for_display(
    kpi_name: str,
    value: float,
    unit: str = "",
    language: str = "fr"
) -> str:
    """
    Format KPI for display: 'Name: Value [Unit] — Short explanation'
    
    Example: 
        "Seuil de rentabilité: 150 clients — Number of clients needed to break even"
    """
    exp = get_explanation(kpi_name, language)
    if "error" in exp:
        # Fallback if no glossary entry
        return f"{kpi_name}: {value} {unit}".strip()
    
    explanation = exp["explanation"].split(":")[1].strip() if ":" in exp["explanation"] else exp["explanation"]
    return f"{exp['name']}: {value} {unit} — {explanation}".strip()


def get_all_variable_names() -> list[str]:
    """Return list of all available variables."""
    return list(GLOSSARY.keys())


def get_glossary_by_category(category: str = "all", language: str = "fr") -> dict:
    """
    Get glossary filtered by category.
    
    Categories: sales, costs, financing, kpis, metadata
    """
    category_map = {
        "sales": ["chiffre_affaires", "prix_vente_unitaire", "segment_client", "nb_clients_mois1", "taux_croissance_mensuel", "taux_fidelisation"],
        "costs": ["type_activite", "loyer_mensuel", "salaires_equipe", "investissements_initiaux"],
        "financing": ["emprunts", "own_capital_invested", "external_funding"],
        "kpis": ["seuil_rentabilite", "marge_brute", "marge_operationnelle", "marge_nette", "besoin_en_fonds_de_roulement", "cash_flow_mensuel", "point_equilibre_mois", "taux_rendement_investissement", "ratio_endettement", "couverture_interet"],
        "metadata": ["entity_name", "sector", "statut_juridique"],
    }
    
    if category not in category_map and category != "all":
        return {"error": f"Unknown category: {category}"}
    
    result = {}
    vars_to_include = category_map.get(category, list(GLOSSARY.keys()))
    
    for var in vars_to_include:
        if var in GLOSSARY:
            result[var] = GLOSSARY[var].get(language, GLOSSARY[var].get("en"))
    
    return result
