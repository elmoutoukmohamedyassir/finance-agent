# AI SaaS Finance Agent — Architecture Guide

## Project Structure

```
finance-agent/
├── app/
│   ├── main.py                    # FastAPI entry point, middleware, router registration
│   ├── api/
│   │   └── routers/
│   │       ├── chat.py            # /chat endpoint — main conversational interface
│   │       ├── metrics.py         # /metrics endpoint — direct SaaS metric calculation
│   │       ├── scenario.py        # /scenario endpoint — what-if financial scenarios
│   │       └── rag.py             # /rag endpoint — PDF ingestion + retrieval
│   ├── core/
│   │   ├── config.py              # All env vars, constants, settings in one place
│   │   ├── groq_client.py         # Groq API wrapper — clean, reusable LLM calls
│   │   └── prompts.py             # All system prompts in one place (easy to tune)
│   ├── agents/
│   │   ├── finance_agent.py       # Main orchestrator: routes, decides, coordinates
│   │   └── question_agent.py      # Determines what info is missing, asks follow-ups
│   ├── tools/
│   │   ├── metrics_calculator.py  # Pure math: MRR, ARR, CAC, LTV, churn, etc.
│   │   ├── scenario_engine.py     # Scenario simulation and projection
│   │   └── finance_guard.py       # Topic validation — rejects off-topic questions
│   ├── rag/
│   │   ├── ingestion.py           # PDF → chunks → embeddings → ChromaDB
│   │   ├── retriever.py           # Query → embed → search → format context
│   │   └── chunker.py             # Smart chunking with overlap and metadata
│   ├── schemas/
│   │   ├── chat.py                # Pydantic models for chat request/response
│   │   ├── metrics.py             # Pydantic models for metrics input/output
│   │   └── session.py             # Session/state models
│   ├── services/
│   │   └── session_service.py     # In-memory session storage + conversation state
│   └── utils/
│       ├── formatting.py          # Response formatters, markdown helpers
│       └── validators.py          # Input validation helpers
├── data/
│   └── chroma/                    # ChromaDB persisted vector store
├── docs/                          # Drop PDFs here for ingestion
├── .env                           # API keys (never commit)
├── .env.example                   # Template for env vars
├── requirements.txt
└── README.md
```

## Component Interaction Flow

```
User → POST /chat
         ↓
    finance_guard.py      → is this a finance question?
         ↓ yes
    session_service.py    → load conversation history + business state
         ↓
    question_agent.py     → what info is still missing?
         ↓
    [if enough info]      → finance_agent.py orchestrates:
         ├── metrics_calculator.py  (if metrics needed)
         ├── scenario_engine.py     (if projection needed)
         ├── retriever.py           (if RAG context needed)
         └── groq_client.py         (LLM reasoning with grounded context)
         ↓
    session_service.py    → save updated state
         ↓
    JSON Response
```

## Key Design Decisions

### Why Groq instead of Ollama?
- No local GPU required — runs anywhere
- Much faster inference (Groq uses LPU hardware)
- Better for student demos and deployment
- Simple REST API, same prompt format

### Why session_service.py (in-memory) instead of a database?
- Zero setup — no Redis, no Postgres needed
- Sufficient for demo/PFE purposes
- Easy to swap for Redis later with same interface
- Session TTL prevents memory bloat

### Why separate question_agent.py?
- Single responsibility: it only decides what to ask next
- Makes the agent feel conversational without complex state machines
- Cleanly separates "what do I know?" from "what should I do with it?"

### Why prompts.py?
- All prompts in one file = easy to tune without touching logic
- Clear separation between prompt engineering and code
- Easier to compare and improve prompts

### Why pure math in metrics_calculator.py?
- No LLM involved in calculations = no hallucinations on numbers
- Deterministic results every time
- LLM only interprets/explains results, never recalculates
