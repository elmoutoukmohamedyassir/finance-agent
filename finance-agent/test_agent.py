"""
test_agent.py — Test every part of the Finance Agent from the command line.

Usage:
    python test_agent.py             # run all tests
    python test_agent.py metrics     # test only metrics calculator
    python test_agent.py rag         # test only RAG retrieval
    python test_agent.py chat        # test full chat flow (requires server running)

Before running chat tests, start the server:
    uvicorn app.main:app --reload
"""

import sys
import json
import logging

logging.basicConfig(level=logging.WARNING)  # suppress noisy lib logs during tests


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Metrics Calculator
# ─────────────────────────────────────────────────────────────────────────────

def test_metrics():
    print("\n" + "=" * 60)
    print("TEST: Metrics Calculator (pure Python, no LLM)")
    print("=" * 60)

    from app.schemas.metrics import MetricsInput
    from app.tools.metrics_calculator import calculate_metrics

    inputs = MetricsInput(
        mrr=12000,
        customer_count=80,
        churn_rate=4.0,
        cac=250,
        monthly_costs=7000,
        marketing_budget=1500,
        new_customers_per_month=6,
    )

    result = calculate_metrics(inputs)

    print(f"  MRR:               ${result.mrr:,.2f}")
    print(f"  ARR:               ${result.arr:,.2f}")
    print(f"  ARPU:              ${result.arpu:,.2f}")
    print(f"  Churn:             {result.churn_rate}%")
    print(f"  Retention:         {result.retention_rate}%")
    print(f"  Customer Lifetime: {result.customer_lifetime_months} months")
    print(f"  LTV:               ${result.ltv:,.2f}")
    print(f"  CAC:               ${result.cac:,.2f}")
    print(f"  LTV:CAC Ratio:     {result.ltv_cac_ratio}x")
    print(f"  CAC Payback:       {result.cac_payback_months} months")
    print(f"  Monthly Profit:    ${result.monthly_profit:,.2f}")
    print(f"  Profit Margin:     {result.profit_margin}%")
    print(f"  Health Score:      {result.health_score}")

    if result.warnings:
        print(f"\n  Warnings:")
        for w in result.warnings:
            print(f"    - {w}")

    assert result.mrr == 12000
    assert result.arr == 144000
    assert result.arpu == 150.0
    assert result.ltv is not None
    assert result.health_score is not None
    print("\n  ✓ Metrics calculator test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Scenario Engine
# ─────────────────────────────────────────────────────────────────────────────

def test_scenarios():
    print("\n" + "=" * 60)
    print("TEST: Scenario Engine (pure Python, no LLM)")
    print("=" * 60)

    from app.tools.scenario_engine import build_standard_scenarios, format_scenarios_for_prompt

    scenarios = build_standard_scenarios(
        starting_customers=80,
        monthly_price=150,
        monthly_costs=7000,
        starting_cash=50000,
        months=12,
    )

    for s in scenarios:
        summ = s["summary"]
        print(f"\n  [{s['name']}]")
        print(f"    Final MRR:       ${summ['final_mrr']:,.0f}")
        print(f"    Final Customers: {summ['final_customers']}")
        print(f"    Final Cash:      ${summ['final_cash']:,.0f}")
        print(f"    Runway:          {summ['runway']}")
        print(f"    MRR Growth:      {summ['mrr_growth_pct']}%")

    assert len(scenarios) == 3
    assert scenarios[0]["name"] == "Pessimistic"
    assert scenarios[2]["name"] == "Optimistic"
    print("\n  ✓ Scenario engine test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Business State & Session
# ─────────────────────────────────────────────────────────────────────────────

def test_session():
    print("\n" + "=" * 60)
    print("TEST: Session and BusinessState")
    print("=" * 60)

    from app.schemas.session import BusinessState

    state = BusinessState()
    assert not state.is_ready_for_analysis(), "Empty state should not be ready"

    state.mrr = 12000
    assert not state.is_ready_for_analysis(), "MRR alone is not enough"

    state.monthly_costs = 7000
    assert state.is_ready_for_analysis(), "MRR + costs should be enough"

    print("  Empty state → not ready for analysis ✓")
    print("  MRR only → not ready for analysis ✓")
    print("  MRR + costs → ready for analysis ✓")
    print("\n  ✓ Session test PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: RAG Embedder
# ─────────────────────────────────────────────────────────────────────────────

def test_embedder():
    print("\n" + "=" * 60)
    print("TEST: Embedder connection")
    print("=" * 60)

    from app.core.config import get_settings
    settings = get_settings()

    print(f"  Backend: {settings.embedding_backend}")
    print(f"  Model:   {settings.embedding_model}")

    try:
        from app.rag.embedder import get_embedder
        embedder = get_embedder()
        vec = embedder.embed_one("what is monthly recurring revenue")
        assert len(vec) > 0, "Embedding should have dimensions"
        print(f"  Embedding dimensions: {len(vec)}")
        print(f"  Sample values: {vec[:3]}")
        print(f"\n  ✓ Embedder test PASSED")
    except Exception as e:
        print(f"\n  ✗ Embedder test FAILED: {e}")
        print(f"\n  If using Ollama: make sure it's running (ollama serve)")
        print(f"  and the model is pulled (ollama pull {settings.embedding_model})")
        print(f"\n  To use sentence-transformers fallback, set in .env:")
        print(f"    EMBEDDING_BACKEND=sentence-transformers")
        print(f"    EMBEDDING_MODEL=all-MiniLM-L6-v2")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: RAG Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def test_rag():
    print("\n" + "=" * 60)
    print("TEST: RAG Retrieval")
    print("=" * 60)

    from app.rag.retriever import retrieve_raw, retrieve_context

    queries = [
        "what is churn rate and how to reduce it",
        "LTV CAC ratio SaaS benchmark",
        "how to write a business plan",
    ]

    for q in queries:
        print(f"\n  Query: '{q}'")
        results = retrieve_raw(q, top_k=3)
        if results:
            for r in results:
                print(f"    [{r['similarity']:.3f}] {r['source']}: {r['text'][:80]}...")
        else:
            print("    No results — collection empty or no relevant chunks")
            print("    Run: python ingest.py")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 6: Full Chat Flow (requires server running)
# ─────────────────────────────────────────────────────────────────────────────

def test_chat_http():
    print("\n" + "=" * 60)
    print("TEST: Full Chat Flow via HTTP (server must be running)")
    print("=" * 60)

    import requests

    BASE = "http://localhost:8000/api/v1"
    session_id = None

    conversation = [
        "I'm building a B2B SaaS for HR teams to automate onboarding",
        "We have 45 customers",
        "We charge $120 per month",
        "Our monthly costs are $6000",
        "About 3.5% churn monthly",
        "We spend $1200 on marketing and acquire about 5 new customers a month",
    ]

    for i, message in enumerate(conversation):
        print(f"\n  Turn {i+1}: '{message}'")
        try:
            resp = requests.post(
                f"{BASE}/chat",
                json={"session_id": session_id, "message": message},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            session_id = data["session_id"]
            mode = data["agent_mode"]
            msg = data["message"]
            print(f"  Mode: {mode}")
            print(f"  Response: {msg[:200]}{'...' if len(msg) > 200 else ''}")
            if data.get("metrics_calculated"):
                print(f"  Metrics: {list(data['metrics_calculated'].keys())}")
        except requests.exceptions.ConnectionError:
            print("  ✗ Server not running. Start with: uvicorn app.main:app --reload")
            return
        except Exception as e:
            print(f"  ✗ Error: {e}")
            return

    print(f"\n  ✓ Chat flow test PASSED — session_id: {session_id}")


# ─────────────────────────────────────────────────────────────────────────────
# CURL COMMANDS REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

CURL_COMMANDS = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURL COMMANDS TO TEST THE API
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 1. Health check
curl http://localhost:8000/

# 2. Detailed health (LLM + RAG + sessions)
curl http://localhost:8000/api/v1/health

# 3. Start a conversation (no session_id → creates new session)
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "I want to analyze my SaaS startup"}'

# 4. Continue conversation (use session_id from step 3)
curl -X POST http://localhost:8000/api/v1/chat \\
  -H "Content-Type: application/json" \\
  -d '{"session_id": "PASTE_SESSION_ID_HERE", "message": "We have 50 customers at $99/month"}'

# 5. Direct metrics calculation (no conversation needed)
curl -X POST http://localhost:8000/api/v1/metrics/calculate \\
  -H "Content-Type: application/json" \\
  -d '{
    "mrr": 12000,
    "customer_count": 80,
    "churn_rate": 4.0,
    "cac": 250,
    "monthly_costs": 7000
  }'

# 6. Scenario analysis
curl -X POST http://localhost:8000/api/v1/scenario/analyze \\
  -H "Content-Type: application/json" \\
  -d '{
    "starting_customers": 50,
    "monthly_price": 99,
    "monthly_costs": 6000,
    "starting_cash": 50000,
    "months": 12
  }'

# 7. Ingest PDFs from docs/
curl -X POST http://localhost:8000/api/v1/rag/ingest

# 8. Force re-ingest (useful if you changed embedding model)
curl -X POST "http://localhost:8000/api/v1/rag/ingest?force=true"

# 9. Check RAG status
curl http://localhost:8000/api/v1/rag/status

# 10. Search knowledge base (debug retrieval quality)
curl "http://localhost:8000/api/v1/rag/search?query=churn+reduction+strategies&top_k=3"

# 11. SaaS metric benchmarks reference
curl http://localhost:8000/api/v1/metrics/benchmarks

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if not args or args[0] == "all":
        test_metrics()
        test_scenarios()
        test_session()
        test_embedder()
        test_rag()
        print(CURL_COMMANDS)

    elif args[0] == "metrics":
        test_metrics()

    elif args[0] == "scenarios":
        test_scenarios()

    elif args[0] == "session":
        test_session()

    elif args[0] == "embedder":
        test_embedder()

    elif args[0] == "rag":
        test_embedder()
        test_rag()

    elif args[0] == "chat":
        test_chat_http()

    elif args[0] == "curl":
        print(CURL_COMMANDS)

    else:
        print(f"Unknown test: {args[0]}")
        print("Usage: python test_agent.py [all|metrics|scenarios|session|embedder|rag|chat|curl]")
