# AI SaaS Finance Agent — v2.0

An AI-powered finance advisor specialized in SaaS business analysis.
Built with FastAPI, Groq, ChromaDB, and sentence-transformers.

---

## What This Does

This backend lets SaaS founders have a conversational financial analysis session with an AI advisor. The agent:

- **Asks intelligent follow-up questions** to progressively collect business data
- **Calculates SaaS metrics** accurately using pure Python math (MRR, ARR, LTV, CAC, churn, runway, etc.)
- **Runs 3-scenario financial projections** (pessimistic / realistic / optimistic)
- **Retrieves knowledge from your PDFs** using RAG (Retrieval-Augmented Generation)
- **Stays specialized** — politely rejects off-topic questions
- **Never hallucinates numbers** — the LLM interprets pre-calculated results, never recalculates

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo>
cd finance-agent
pip install -r requirements.txt
```

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

Get a free Groq API key at: https://console.groq.com

### 3. Add finance documents (optional but recommended)

Drop PDF files into the `docs/` folder. They'll be automatically indexed on startup.

### 4. Run the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open Swagger UI at: http://localhost:8000/docs

---

## API Endpoints

### `POST /api/v1/chat`
Main conversational interface. The agent progressively collects business info,
then performs full financial analysis.

```json
// Request
{
  "session_id": "abc-123",      // reuse this across messages
  "message": "I'm building a project management SaaS for agencies"
}

// Response
{
  "session_id": "abc-123",
  "message": "Great! Who are your target customers exactly?",
  "agent_mode": "gathering_info",
  "metrics_calculated": null,
  "business_state": {
    "business_name": "project management SaaS for agencies"
  }
}
```

**Agent modes:**
- `gathering_info` — collecting business data, asking questions
- `analyzing` — running and interpreting full financial analysis
- `answering` — answering a general finance question
- `off_topic` — rejected non-finance question

### `POST /api/v1/metrics/calculate`
Direct metric calculation — skip the conversation, provide numbers directly.

```json
// Request
{
  "mrr": 15000,
  "customer_count": 120,
  "churn_rate": 3.5,
  "cac": 200,
  "monthly_costs": 8000
}

// Response includes: ARR, ARPU, LTV, LTV:CAC ratio,
//                   CAC payback, profit margin, health_score, warnings
```

### `POST /api/v1/scenario/analyze`
Run 3-scenario projections over 12 months.

```json
{
  "starting_customers": 50,
  "monthly_price": 99,
  "monthly_costs": 8000,
  "starting_cash": 50000,
  "months": 12
}
```

### `POST /api/v1/rag/ingest`
Index PDFs from the `docs/` folder into ChromaDB.
Safe to call multiple times (skips already-indexed files).

### `GET /api/v1/rag/search?query=churn+reduction`
Search the finance knowledge base directly. Good for debugging RAG quality.

### `GET /api/v1/rag/status`
Check how many documents/chunks are indexed.

### `GET /api/v1/metrics/benchmark`
Reference benchmarks for key SaaS metrics (LTV:CAC, churn, payback, etc.)

---

## Architecture

```
app/
├── main.py                    # FastAPI app, routers, startup
├── core/
│   ├── config.py              # All settings from .env
│   ├── groq_client.py         # Groq API wrapper (replaces Ollama)
│   └── prompts.py             # All LLM prompts in one place
├── agents/
│   ├── finance_agent.py       # Main orchestrator
│   └── question_agent.py      # Collects missing business info
├── tools/
│   ├── metrics_calculator.py  # Pure Python SaaS math (no LLM)
│   ├── scenario_engine.py     # Financial projections (no LLM)
│   └── finance_guard.py       # Topic validation
├── rag/
│   ├── chunker.py             # Sentence-aware PDF chunking
│   ├── ingestion.py           # PDF → ChromaDB pipeline
│   └── retriever.py           # Semantic search with relevance filter
├── schemas/
│   ├── chat.py                # Chat request/response models
│   ├── metrics.py             # Metrics input/output models
│   └── session.py             # Session + BusinessState models
├── services/
│   └── session_service.py     # In-memory conversation state
└── api/routers/
    ├── chat.py
    ├── metrics.py
    ├── scenario.py
    └── rag.py
```

### Anti-Hallucination Strategy

The key design decision: **the LLM never does math**.

1. User provides business data through conversation
2. `metrics_calculator.py` calculates all numbers with pure Python
3. `scenario_engine.py` runs projections with pure Python
4. The LLM receives pre-calculated results and **interprets them only**
5. All prompts explicitly say: "Do NOT recalculate. Use the provided numbers."

This eliminates arithmetic hallucinations entirely.

### Conversational State Flow

```
Message → finance_guard → extract_business_info → update session state
                                                        ↓
                                              is_ready_for_analysis?
                                               ↙              ↘
                                          No                  Yes
                                    ask next               calculate metrics
                                    question               build scenarios
                                                           retrieve RAG
                                                           LLM interprets
```

---

## How to Add More Finance PDFs

1. Drop any PDF into the `docs/` folder
2. Either restart the server (auto-ingests on startup) OR call `POST /api/v1/rag/ingest`
3. The content will be available in the next RAG retrieval

---

## Configuration Reference (.env)

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Your Groq API key |
| `GROQ_MODEL` | `llama3-70b-8192` | Groq model to use |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `DOCS_DIR` | `./docs` | Folder to scan for PDFs |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `RAG_TOP_K` | `4` | Number of chunks to retrieve |
| `SESSION_TTL_MINUTES` | `60` | Session expiry time |
| `DEBUG` | `true` | Enable debug logging |

---

## Groq Model Options

| Model | Speed | Context | Best For |
|---|---|---|---|
| `llama3-70b-8192` | Fast | 8K | Best quality, recommended |
| `llama3-8b-8192` | Very fast | 8K | Testing, low latency |
| `mixtral-8x7b-32768` | Fast | 32K | Long conversations |

---

## Running Tests

```bash
# Test metrics calculation
python -c "
from app.schemas.metrics import MetricsInput
from app.tools.metrics_calculator import calculate_metrics
result = calculate_metrics(MetricsInput(mrr=15000, customer_count=120, churn_rate=3.5, cac=200, monthly_costs=8000))
print(result)
"

# Test RAG ingestion
python -m app.rag.ingestion

# Test RAG retrieval
python -c "
from app.rag.retriever import retrieve_context
print(retrieve_context('what is LTV CAC ratio'))
"
```
