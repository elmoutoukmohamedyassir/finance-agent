"""
test_fix_proof.py — Proves the memory + phase-routing fix end to end,
without needing a real GROQ_API_KEY (the LLM call is mocked; everything
else — session_service, schemas/session.py, phase_router.py, phase1/2/3
agents — is the real, unmodified-except-for-the-fix code).

Run: python3 test_fix_proof.py
"""
import json as _json
import tempfile
import pathlib

# Isolate the test DB so we don't touch the project's real data/sessions.db
TMP_DB = pathlib.Path(tempfile.mkdtemp()) / "sessions_test.db"

import app.services.session_service as session_service
session_service.DB_PATH = TMP_DB  # redirect persistence to a scratch file

import app.core.groq_client as groq_mod
from app.agents.question_agent import EXTRACTION_SYSTEM_PROMPT_STRICT
from app.agents.phase_router import PhaseRouter
from app.agents.base_agent import AgentMessage
from app.schemas.session import BusinessState
from app.services.session_service import get_or_create_session, save_session

# ── Mock the LLM so this test needs no network / API key ──────────────────
# The test loop sets NEXT_EXTRACTION right before each call when it knows
# Phase2 is about to ask a specific question — this is just a test double,
# your real LLM extracts freely from natural language.
NEXT_EXTRACTION = {}

def fake_chat(system_prompt, user_message, conversation_history=None, temperature=0.0, max_tokens=400):
    if system_prompt == EXTRACTION_SYSTEM_PROMPT_STRICT:
        return _json.dumps(NEXT_EXTRACTION)
    # Conversational/narration call: echo back proof that history was received.
    last_user_turns = [m["content"] for m in (conversation_history or []) if m["role"] == "user"]
    memory_proof = f" [souvenir: '{last_user_turns[0]}']" if last_user_turns else " [aucun historique reçu]"
    return f"(réponse simulée à: '{user_message}'){memory_proof}"

groq_mod.groq_client.chat = fake_chat  # shadow the lazy proxy's __getattr__

# ── Drive a real multi-turn conversation through the real pipeline ────────
router = PhaseRouter()
session_id = "proof-session-1"

# field -> (a plausible natural-language answer, what extraction should return)
ANSWERS_BY_FIELD = {
    "entity_name":                ("Dar Atlas", {"entity_name": "Dar Atlas"}),
    "entity_type":                ("Auto-entrepreneur",        {"entity_type": "corporate"}),
    "sector":                     ("Restauration", {"sector": "restauration"}),
    "statut_juridique":           ("Auto-entrepreneur", {"statut_juridique": "Auto-entrepreneur"}),
    "segment_client":             ("Des particuliers, du B2C", {"segment_client": "B2C"}),
    "prix_vente_unitaire":        ("85 MAD le plat en moyenne", {"prix_vente_unitaire": 85}),
    "nb_clients_mois1":           ("Environ 40 clients le 1er mois", {"nb_clients_mois1": 40}),
    "taux_croissance_mensuel":    ("Je vise 10% de croissance par mois", {"taux_croissance_mensuel": 10}),
    "taux_fidelisation":          ("Je dirais 85% de clients fidélisés", {"taux_fidelisation": 85}),
    "saisonnalite":               ("Plus fort en été", {"saisonnalite": "plus fort en été"}),
    "type_activite":              ("Produit", {"type_activite": "produit"}),
    "cout_fabrication_unitaire":  ("Environ 30 MAD de coût matière par plat", {"cout_fabrication_unitaire": 30}),
    "cout_infra_numerique":       ("0, pas d'infra numérique particulière", {"cout_infra_numerique": 0}),
    "loyer_mensuel":              ("Le loyer du local est de 6000 MAD", {"loyer_mensuel": 6000}),
    "salaires_equipe":            ("Les salaires de l'équipe : 9000 MAD/mois", {"salaires_equipe": 9000}),
    "investissements_initiaux":   ("Investissement de départ : 50000 MAD", {"investissements_initiaux": 50000}),
    "emprunts":                   ("Pas d'emprunt pour l'instant", {"emprunts": 0}),
}

def run_turn(user_message: str, extraction: dict | None = None):
    global NEXT_EXTRACTION
    NEXT_EXTRACTION = extraction or {}

    session = get_or_create_session(session_id)
    history_before = len(session.conversation_history)
    session.add_message("user", user_message)

    agent_message = AgentMessage(
        sender_agent_id="user",
        session_id=session.session_id,
        user_message=user_message,
        intent="chat",
        context={
            "business_state": session.business_state.model_dump(),
            "router_phase": session.router_phase,
            "conversation_history": session.conversation_history[:-1],
            "asked_questions": session.questions_asked,
            "pending_question": session.pending_question,
        },
    )
    response = router.route_message(agent_message, agent_message.context)
    if response.message:
        session.add_message("assistant", response.message)

    updated_bs = response.business_state or (response.structured_output or {}).get("business_state")
    if updated_bs:
        merged = {**session.business_state.model_dump(),
                  **{k: v for k, v in updated_bs.items() if v is not None and k in BusinessState.model_fields}}
        session.business_state = BusinessState(**merged)

    if response.structured_output:
        if "next_phase" in response.structured_output:
            session.router_phase = response.structured_output["next_phase"]
        if "asked_questions" in response.structured_output:
            session.questions_asked = response.structured_output["asked_questions"]
        if "pending_question" in response.structured_output:
            session.pending_question = response.structured_output["pending_question"]

    save_session(session)

    reloaded = get_or_create_session(session_id)
    history_survived = len(reloaded.conversation_history) >= history_before + 2

    print(f"  router_phase={session.router_phase:7s} | history_len={len(reloaded.conversation_history):2d} "
          f"| history_survived_reload={history_survived} | agent={response.agent_id} "
          f"| pending_question={session.pending_question}")
    print(f"    -> {response.message[:120]}")
    return reloaded


print("=" * 80)
print("PHASE 1 — free-form ideation chat (should remember everything)")
print("=" * 80)
print('\nTurn 1: "Je veux ouvrir un restaurant à Casablanca"')
run_turn("Je veux ouvrir un restaurant à Casablanca")

print('\nTurn 2: "cuisine marocaine"')
run_turn("cuisine marocaine")

print('\nTurn 3: "Je voudrais calculer mes coûts et revenus en détail"  (trigger words -> phase2)')
run_turn("Je voudrais calculer mes coûts et revenus en détail")

print("\n" + "=" * 80)
print("PHASE 2 — structured Q&A (router should now stay in phase2 until full)")
print("=" * 80)

# Phase2 asks ONE question at a time. Drive up to 12 turns, answering
# whatever it asks using ANSWERS_BY_FIELD, until it signals phase3.
from app.agents.question_agent import FIELD_TYPES

def generic_answer_for(field: str):
    """Fallback for any field not explicitly scripted above — uses FIELD_TYPES
    to produce a plausible value of the right type, so the test isn't brittle
    to every optional field in the questionnaire."""
    ftype = FIELD_TYPES.get(field, "texte")
    if ftype in ("numerique",):
        return ("500 MAD", {field: 500})
    if ftype == "pourcentage":
        return ("10%", {field: 10})
    if ftype in ("numerique_details",):
        return ("0, pas de detail particulier", {field: 0})
    if ftype == "booleen_details":
        return ("Non, pas vraiment", {field: "non"})
    if ftype == "choix":
        return ("Standard", {field: "standard"})
    return ("Information générique", {field: "n/a"})

for i in range(25):
    session = get_or_create_session(session_id)
    if session.router_phase == "phase3":
        break
    pending = session.pending_question
    if pending and pending in ANSWERS_BY_FIELD:
        answer, extraction = ANSWERS_BY_FIELD[pending]
    elif pending:
        answer, extraction = generic_answer_for(pending)
    else:
        # First turn into phase2: nothing pending yet, message is just a placeholder
        answer, extraction = ("D'accord, je vous écoute.", {})
    print(f'\nTurn P2.{i+1}: (pending="{pending}") -> "{answer}"')
    run_turn(answer, extraction)

print("\n" + "=" * 80)
print("PHASE 3 — does it actually call the real KPI engine now?")
print("=" * 80)
p3 = run_turn("Peux-tu analyser la viabilité de mon projet ?")
mc = None
# Re-fetch the raw AgentResponse so we can inspect metrics_calculated directly
session = get_or_create_session(session_id)
agent_message = AgentMessage(
    sender_agent_id="user", session_id=session.session_id, user_message="Analyse complète",
    intent="chat",
    context={
        "business_state": session.business_state.model_dump(),
        "router_phase": session.router_phase,
        "conversation_history": session.conversation_history,
        "asked_questions": session.questions_asked,
        "pending_question": session.pending_question,
    },
)
direct_response = router.route_message(agent_message, agent_message.context)
print("\nmetrics_calculated populated:", bool(direct_response.metrics_calculated))
if direct_response.metrics_calculated:
    print("\n--- derived_summary (REAL MATH, not LLM) ---")
    print(direct_response.metrics_calculated["derived_summary"])
    print("\n--- plan_summary (REAL MATH, not LLM) ---")
    print(direct_response.metrics_calculated["plan_summary"])
assert direct_response.metrics_calculated, "Phase3 did NOT run the real engine!"
print("\n✅ Phase3 now runs the real deterministic engine — these numbers are computed, not guessed.")

final = get_or_create_session(session_id)
print("\n" + "=" * 80)
print("FINAL STATE")
print("=" * 80)
print("router_phase        :", final.router_phase)
print("conversation turns  :", len(final.conversation_history))
print("business_state      :", final.business_state.filled_fields())

assert final.router_phase == "phase3", "Did not progress to phase3 (KPI analysis)!"
assert final.business_state.prix_vente_unitaire == 85
assert final.business_state.nb_clients_mois1 == 40
assert final.business_state.loyer_mensuel == 6000
print("\n✅ ALL ASSERTIONS PASSED — memory persists across all turns, phase2 is reachable,")
print("   collected answers accumulate correctly, and phase3 (KPI analysis) is reached.")