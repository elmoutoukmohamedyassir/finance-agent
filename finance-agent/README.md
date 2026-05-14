# AI SaaS Finance Agent v2

An AI finance advisor specialized in SaaS business analysis.
Helps founders evaluate financial health, calculate metrics, project scenarios, and make data-driven decisions.

---

## Quick Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Open .env and add your GROQ_API_KEY
```
Get a free Groq key at https://console.groq.com

### 3. Set up the embedding model

**Recommended — nomic-embed-text via Ollama (better quality):**
```bash
# Install Ollama from https://ollama.com
ollama pull nomic-embed-text
ollama serve          # keep this running in a separate terminal
```
Your `.env` already has `EMBEDDING_BACKEND=ollama` and `EMBEDDING_MODEL=nomic-embed-text`.

**Alternative — sentence-transformers (no Ollama needed):**
```env
EMBEDDING_BACKEND=sentence-transformers
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### 4. Add your finance PDFs
```
docs/
  your_saas_metrics_guide.pdf
  business_plan_template.pdf
  startup_finance_book.pdf
  ... (any finance PDFs you have)
```

### 5. Ingest your PDFs
```bash
python ingest.py
```

### 6. Start the server
```bash
uvicorn app.main:app --reload
```

Open Swagger UI at: **http://localhost:8000/docs**

---

## Testing

```bash
# Test everything (no server needed for most)
python test_agent.py

# Test specific parts
python test_agent.py metrics      # metrics calculator
python test_agent.py scenarios    # scenario projections
python test_agent.py embedder     # check embedding model
python test_agent.py rag          # test retrieval
python test_agent.py chat         # full chat (server must be running)

# Print all curl commands
python test_agent.py curl
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/chat` | Main chat — conversational finance advisor |
| POST | `/api/v1/metrics/calculate` | Direct metric calculation (no chat) |
| GET | `/api/v1/metrics/benchmarks` | SaaS industry benchmark reference |
| POST | `/api/v1/scenario/analyze` | 3-scenario financial projection |
| POST | `/api/v1/rag/ingest` | Ingest PDFs from docs/ |
| GET | `/api/v1/rag/search?query=...` | Search knowledge base |
| GET | `/api/v1/rag/status` | Check how many docs are indexed |
| GET | `/api/v1/health` | System health check |

---

## How to use the chat

```json
// Turn 1: Start (no session_id needed)
POST /api/v1/chat
{ "message": "I'm building a B2B SaaS for HR teams" }

// Response gives you a session_id — copy it!
{ "session_id": "abc-123", "message": "How many paying customers do you have?", "agent_mode": "gathering_info" }

// Turn 2: Continue with same session_id
POST /api/v1/chat
{ "session_id": "abc-123", "message": "45 customers at $120/month, costs $5500/month" }

// After collecting enough info, agent switches to analysis mode:
{ "session_id": "abc-123", "message": "Here's your full financial analysis...", "agent_mode": "analyzing", "metrics_calculated": {...} }
```

**agent_mode values:**
- `gathering_info` — asking questions to collect business data
- `analyzing` — running and interpreting full financial analysis
- `answering` — answering general finance questions
- `off_topic` — question outside finance expertise (handled by prompt, not code)

---

## Why These Design Choices?

### Why `__init__.py` in every folder?
Python requires these files to treat folders as importable packages.
Without them, `from app.core.config import get_settings` throws `ModuleNotFoundError`.
They're empty (or have a comment) — that's normal and correct.

### Why nomic-embed-text instead of all-MiniLM-L6-v2?
- **768 dimensions** vs 384 — 2× richer vector representation
- **8192 token context window** vs 512 — handles long finance document passages without truncation
- Trained on more diverse, longer-form text — better for PDFs
- Consistently outperforms MiniLM on domain-specific retrieval benchmarks
- For finance documents, this means more relevant chunks retrieved = less hallucination

### Why is finance_guard.py gone?
The old approach used keyword matching + a separate LLM API call to classify topics.
Problems: (1) brittle — "profit from cooking" would confuse it, (2) adds latency, (3) redundant.

The new approach puts explicit scope instructions directly in `FINANCE_AGENT_SYSTEM_PROMPT`:
- Lists exactly what the agent covers
- Gives a scripted, friendly refusal for off-topic questions
- The LLM understands conversation context — keyword lists don't

Result: more natural refusals, no extra API call, no brittle keyword matching.

### Why does the LLM never recalculate numbers?
`metrics_calculator.py` uses pure Python math. The LLM receives the pre-calculated
results labeled "CALCULATED METRICS — TREAT AS GROUND TRUTH, DO NOT RECALCULATE."
This eliminates arithmetic hallucinations entirely. LLMs are great at reasoning
about numbers — they're bad at arithmetic.

---

## Adding More PDFs

Recommended finance PDFs for a SaaS Finance Agent:
- SaaS metrics guides (Bessemer's State of the Cloud, OpenView reports)
- Business plan writing guides  
- Financial modeling for startups
- Pricing strategy books
- Investor pitch guides

Steps:
1. Drop PDF into `docs/`
2. Run `python ingest.py` (or `POST /api/v1/rag/ingest`)
3. Done — the agent will use it in next conversations

To re-index after changing embedding model:
```bash
# Delete old ChromaDB data first (switching models = incompatible vectors)
rm -rf data/chroma/
python ingest.py
```

---

## Project Structure

```
finance-agent/
├── app/
│   ├── main.py                    # FastAPI app, startup, routers
│   ├── core/
│   │   ├── config.py              # All settings from .env
│   │   ├── groq_client.py         # Groq API wrapper
│   │   └── prompts.py             # All LLM prompts (easy to tune)
│   ├── agents/
│   │   ├── finance_agent.py       # Main orchestrator
│   │   └── question_agent.py      # Info collection + extraction
│   ├── tools/
│   │   ├── metrics_calculator.py  # Pure Python SaaS math
│   │   └── scenario_engine.py     # Financial projections
│   ├── rag/
│   │   ├── embedder.py            # nomic-embed-text / sentence-transformers
│   │   ├── chunker.py             # Sentence-aware PDF chunking
│   │   ├── ingestion.py           # PDF → ChromaDB pipeline
│   │   └── retriever.py           # Semantic search
│   ├── schemas/
│   │   ├── chat.py                # Request/response models
│   │   ├── metrics.py             # Metrics input/output
│   │   └── session.py             # Session + BusinessState
│   ├── services/
│   │   └── session_service.py     # In-memory session store
│   └── api/routers/
│       ├── chat.py
│       ├── metrics.py
│       ├── scenario.py
│       └── rag.py
├── docs/                          # Drop your PDFs here
├── data/chroma/                   # ChromaDB storage (auto-created)
├── ingest.py                      # Run to embed PDFs
├── test_agent.py                  # All tests + curl commands
├── requirements.txt
└── .env.example
```
