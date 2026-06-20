"""
tools/plan_pdf.py — renders a computed business plan (tools/plan_pipeline.ComputedPlan)
as a downloadable PDF: hypothèses, variables dérivées, compte de résultat,
plan de trésorerie, plan de financement, bilan simplifié, KPIs.

Note: reportlab's built-in fonts don't include checkmarks/emoji/box-drawing
glyphs used in the chat-text version of this report (✅, ❌, ━) — those would
render as black boxes here, so this module uses plain text instead.
"""
from datetime import datetime
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from app.tools.plan_pipeline import ComputedPlan

ACCENT = colors.HexColor("#1f6f4a")
HEADER_BG = colors.HexColor("#eef4f1")
NEGATIVE = colors.HexColor("#b3261e")


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "PlanTitle", parent=styles["Title"], fontSize=22, textColor=ACCENT, spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "PlanSubtitle", parent=styles["Normal"], fontSize=11, textColor=colors.grey,
        alignment=TA_CENTER, spaceAfter=24,
    ))
    styles.add(ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], fontSize=14, textColor=ACCENT,
        spaceBefore=18, spaceAfter=8, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "Note", parent=styles["Normal"], fontSize=8, textColor=colors.grey, spaceBefore=10,
    ))
    return styles


def _table(data, col_widths=None, numeric_cols=None):
    """A clean grid table with a tinted header row and right-aligned numbers."""
    numeric_cols = numeric_cols or []
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for col in numeric_cols:
        style.append(("ALIGN", (col, 0), (col, -1), "RIGHT"))
    t.setStyle(TableStyle(style))
    return t


def _fmt_mad(value) -> str:
    try:
        return f"{value:,.0f} MAD".replace(",", " ")
    except (TypeError, ValueError):
        return "N/D"


def build_plan_pdf(business_state: dict, computed: ComputedPlan) -> bytes:
    """Render the full business plan as PDF bytes, ready to stream to a client."""
    styles = _styles()
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=2 * cm, bottomMargin=2 * cm, leftMargin=2 * cm, rightMargin=2 * cm,
        title=f"Business Plan — {business_state.get('entity_name') or 'Projet'}",
    )
    story = []

    # ── Title page ───────────────────────────────────────────────────────
    entity_name = business_state.get("entity_name") or "Votre projet"
    story.append(Spacer(1, 3 * cm))
    story.append(Paragraph("Business Plan", styles["PlanTitle"]))
    story.append(Paragraph(entity_name, ParagraphStyle("EntityName", parent=styles["Normal"],
                                                        fontSize=16, alignment=TA_CENTER, spaceAfter=6)))
    subtitle_bits = [b for b in [business_state.get("sector"), business_state.get("statut_juridique")] if b]
    story.append(Paragraph(" · ".join(subtitle_bits) or "", styles["PlanSubtitle"]))
    story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y')}",
                            ParagraphStyle("Date", parent=styles["Normal"], alignment=TA_CENTER,
                                           fontSize=10, textColor=colors.grey)))
    story.append(Paragraph(
        "Document généré automatiquement à partir des hypothèses fournies. "
        "Les chiffres sont calculés (seuil de rentabilité, BFR, projections), pas estimés par "
        "intelligence artificielle — ce document reste à titre indicatif et ne remplace pas "
        "l'avis d'un expert-comptable ou d'un conseiller financier.",
        ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=9, textColor=colors.grey,
                        alignment=TA_CENTER, spaceBefore=40)
    ))
    story.append(PageBreak())

    # ── Hypothèses du projet ────────────────────────────────────────────
    story.append(Paragraph("1. Hypothèses du projet", styles["SectionHeading"]))
    hyp_rows = [["Hypothèse", "Valeur"]]
    field_labels = [
        ("entity_type", "Type d'entité"),
        ("segment_client", "Segment client"),
        ("prix_vente_unitaire", "Prix de vente unitaire (MAD)"),
        ("nb_clients_mois1", "Clients prévus, mois 1"),
        ("taux_croissance_mensuel", "Croissance mensuelle (%)"),
        ("taux_fidelisation", "Fidélisation client (%)"),
        ("cout_fabrication_unitaire", "Coût de revient unitaire (MAD)"),
        ("loyer_mensuel", "Loyer mensuel (MAD)"),
        ("salaires_equipe", "Salaires équipe / mois (MAD)"),
        ("charges_utilites", "Charges d'utilités / mois (MAD)"),
        ("budget_marketing", "Budget marketing / mois (MAD)"),
        ("investissements_initiaux", "Investissements initiaux (MAD)"),
        ("emprunts", "Emprunt bancaire (MAD)"),
        ("own_capital_invested", "Capital propre investi (MAD)"),
    ]
    for key, label in field_labels:
        v = business_state.get(key)
        if v is not None and v != "":
            hyp_rows.append([label, str(v)])
    story.append(_table(hyp_rows, col_widths=[10 * cm, 6 * cm], numeric_cols=[1]))

    # ── Variables dérivées ──────────────────────────────────────────────
    d = computed.derived
    story.append(Paragraph("2. Variables dérivées (calculées)", styles["SectionHeading"]))
    derived_rows = [
        ["Indicateur", "Valeur"],
        ["Masse salariale chargée / mois", _fmt_mad(d.masse_salariale_chargee)],
        ["Charges fixes totales / mois", _fmt_mad(d.charges_fixes_mensuelles_totales)],
        ["Marge brute unitaire", f"{d.marge_brute_unitaire:,.0f} MAD ({d.taux_marge_brute:.1f}%)".replace(",", " ") if d.taux_marge_brute is not None else _fmt_mad(d.marge_brute_unitaire)],
        ["Seuil de rentabilité", f"{d.seuil_rentabilite_clients:,.0f} clients/mois ({_fmt_mad(d.seuil_rentabilite_ca)} CA)".replace(",", " ") if d.seuil_rentabilite_clients else "Non calculable"],
        ["BFR (stock + créances)", _fmt_mad(d.bfr)],
        ["Trésorerie initiale nécessaire", _fmt_mad(d.tresorerie_initiale_necessaire)],
        ["Dotation amortissement / mois", _fmt_mad(d.dotation_amortissement_mensuelle)],
    ]
    story.append(_table(derived_rows, col_widths=[10 * cm, 6 * cm], numeric_cols=[1]))

    # ── Plan de financement ──────────────────────────────────────────────
    plan = computed.plan
    fin = plan.plan_financement
    story.append(Paragraph("3. Plan de financement initial", styles["SectionHeading"]))
    fin_rows = [
        ["", "Montant"],
        ["Besoins totaux", _fmt_mad(fin.total_besoins)],
        ["Ressources totales", _fmt_mad(fin.total_ressources)],
        ["Solde", _fmt_mad(fin.solde)],
    ]
    t = _table(fin_rows, col_widths=[10 * cm, 6 * cm], numeric_cols=[1])
    if fin.solde < 0:
        t.setStyle(TableStyle([("TEXTCOLOR", (1, 3), (1, 3), NEGATIVE), ("FONTNAME", (1, 3), (1, 3), "Helvetica-Bold")]))
    story.append(t)
    if fin.solde < 0:
        story.append(Paragraph(
            "Attention : le plan de financement présente un déficit. Un financement complémentaire "
            "(apport personnel, emprunt, subvention) est nécessaire pour couvrir les besoins de démarrage.",
            ParagraphStyle("Warn", parent=styles["Normal"], fontSize=9, textColor=NEGATIVE, spaceBefore=6),
        ))

    # ── Compte de résultat ───────────────────────────────────────────────
    a1, a2 = plan.annee1, plan.annee2
    story.append(Paragraph("4. Compte de résultat prévisionnel", styles["SectionHeading"]))
    cr_rows = [
        ["", "Année 1", "Année 2"],
        ["Chiffre d'affaires", _fmt_mad(a1["ca_total"]), _fmt_mad(a2["ca_total"])],
        ["Marge brute", _fmt_mad(a1["marge_brute"]), _fmt_mad(a2["marge_brute"])],
        ["EBITDA", _fmt_mad(a1["ebitda"]), _fmt_mad(a2["ebitda"])],
        ["Résultat net", _fmt_mad(a1["resultat_net"]), _fmt_mad(a2["resultat_net"])],
        ["Marge nette", f"{a1.get('marge_nette_pct', 'N/D')}%", f"{a2.get('marge_nette_pct', 'N/D')}%"],
    ]
    t = _table(cr_rows, col_widths=[7 * cm, 4.5 * cm, 4.5 * cm], numeric_cols=[1, 2])
    if a1["resultat_net"] < 0:
        t.setStyle(TableStyle([("TEXTCOLOR", (1, 4), (1, 4), NEGATIVE)]))
    if a2["resultat_net"] < 0:
        t.setStyle(TableStyle([("TEXTCOLOR", (2, 4), (2, 4), NEGATIVE)]))
    story.append(t)

    # ── Plan de trésorerie ───────────────────────────────────────────────
    story.append(Paragraph("5. Plan de trésorerie", styles["SectionHeading"]))
    tr_rows = [
        ["", "Année 1", "Année 2"],
        ["Trésorerie en fin de période", _fmt_mad(a1["tresorerie_fin"]), _fmt_mad(a2["tresorerie_fin"])],
    ]
    story.append(_table(tr_rows, col_widths=[7 * cm, 4.5 * cm, 4.5 * cm], numeric_cols=[1, 2]))

    story.append(PageBreak())

    # ── Bilan simplifié ──────────────────────────────────────────────────
    b1, b2 = plan.bilan_annee1, plan.bilan_annee2
    story.append(Paragraph("6. Bilan simplifié", styles["SectionHeading"]))
    bilan_rows = [
        ["ACTIF", "Année 1", "Année 2"],
        ["Immobilisations nettes", _fmt_mad(b1.immobilisations_nettes), _fmt_mad(b2.immobilisations_nettes)],
        ["Stocks", _fmt_mad(b1.stocks), _fmt_mad(b2.stocks)],
        ["Créances clients", _fmt_mad(b1.creances_clients), _fmt_mad(b2.creances_clients)],
        ["Trésorerie", _fmt_mad(b1.tresorerie), _fmt_mad(b2.tresorerie)],
        ["TOTAL ACTIF", _fmt_mad(b1.total_actif), _fmt_mad(b2.total_actif)],
        ["PASSIF", "Année 1", "Année 2"],
        ["Capital social", _fmt_mad(b1.capital_social), _fmt_mad(b2.capital_social)],
        ["Réserves / résultats cumulés", _fmt_mad(b1.reserves_resultats), _fmt_mad(b2.reserves_resultats)],
        ["Dettes bancaires", _fmt_mad(b1.dettes_bancaires), _fmt_mad(b2.dettes_bancaires)],
        ["Dettes fournisseurs", _fmt_mad(b1.dettes_fournisseurs), _fmt_mad(b2.dettes_fournisseurs)],
        ["Dettes fiscales et sociales", _fmt_mad(b1.dettes_fiscales_sociales), _fmt_mad(b2.dettes_fiscales_sociales)],
        ["TOTAL PASSIF", _fmt_mad(b1.total_passif), _fmt_mad(b2.total_passif)],
    ]
    t = _table(bilan_rows, col_widths=[7 * cm, 4.5 * cm, 4.5 * cm], numeric_cols=[1, 2])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 6), (-1, 6), HEADER_BG),
        ("FONTNAME", (0, 5), (-1, 5), "Helvetica-Bold"),
        ("FONTNAME", (0, 6), (-1, 6), "Helvetica-Bold"),
        ("FONTNAME", (0, 12), (-1, 12), "Helvetica-Bold"),
    ]))
    story.append(t)

    # ── KPIs clés ────────────────────────────────────────────────────────
    story.append(Paragraph("7. KPIs clés", styles["SectionHeading"]))
    kpi_rows = [
        ["Indicateur", "Valeur"],
        ["Seuil de rentabilité", f"{plan.seuil_rentabilite_clients:,.0f} clients/mois".replace(",", " ") if plan.seuil_rentabilite_clients else "Non calculable"],
        ["Point mort (mois où la trésorerie cumulée devient positive)",
         f"Mois {plan.mois_point_mort}" if plan.mois_point_mort else "Non atteint sur 24 mois"],
        ["ROI année 1", f"{plan.roi_annee1:.1f}%" if plan.roi_annee1 is not None else "N/D"],
        ["ROI année 2", f"{plan.roi_annee2:.1f}%" if plan.roi_annee2 is not None else "N/D"],
        ["DSCR année 1", f"{plan.dscr_annee1:.2f}x" if plan.dscr_annee1 is not None else "N/D"],
    ]
    story.append(_table(kpi_rows, col_widths=[10 * cm, 6 * cm], numeric_cols=[1]))

    story.append(Paragraph(
        "Document généré par finance_agent.ma — à titre indicatif uniquement.",
        styles["Note"],
    ))

    doc.build(story)
    return buf.getvalue()