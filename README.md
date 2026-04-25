# ShopBot — Agentic Recommendation System

A personalized product recommendation system powered by a **tool-calling LLM agent**. The agent reasons over a user's purchase history and preferences, searches a vector-embedded product catalog, and returns ranked recommendations with explanations — streaming every reasoning step live to the UI.

---

## How It Works

```
Amazon catalog (HuggingFace)          Synthetic user personas (LLM-generated)
        ↓                                          ↓
  catalog.json                          synthetic_users.json
        ↓                                          ↓
        └──────────────┬────────────────────────────┘
                       ↓
              ChromaDB (cosine similarity)
              items collection + users collection
                       ↓
              RecommendationAgent (tool-calling loop)
                       ↓
         ┌─────────────────────────────┐
         │  Phase 1: fetch_user_history │  → understands preferences, budget, history
         │  Phase 2: search_catalog     │  → semantic search over embeddings
         │  Phase 3: filter_by_attrs    │  → refine by price, rating, category, brand
         │  Phase 4: get_item_details   │  → verify top candidates before finalizing
         └─────────────────────────────┘
                       ↓
         Ranked recommendations + reasoning summary
                       ↓
         ┌─────────────────────────────┐
         │  ShopBot Web UI (Flask/SSE) │  ← primary interface
         │  Streamlit UI               │  ← dev / research interface
         └─────────────────────────────┘
```

---

## Project Structure

```
agentic-rs/
├── src/agentic_rs/
│   ├── agent/
│   │   ├── loop.py        # Agent loop — LLM ↔ tool iteration, SSE streaming
│   │   └── tools.py       # 4 tool definitions + ToolExecutor
│   ├── vectorstore/
│   │   └── build.py       # ChromaDB indexing + VectorStore query class
│   ├── data/
│   │   └── generate.py    # LLM-generated catalog + synthetic user personas
│   ├── ui/
│   │   └── app.py         # Streamlit UI (dev interface)
│   ├── models.py          # Pydantic models: CatalogItem, SyntheticUser, Recommendation
│   └── config.py          # Central config — loads from .env
├── server.py              # Flask backend for ShopBot web app
├── index.html             # ShopBot frontend (vanilla JS + SSE)
├── scripts/
│   ├── ingest_amazon_products.py   # Pull real Amazon metadata from HuggingFace
│   ├── generate_data.py            # Generate synthetic catalog + users via LLM
│   ├── build_vectorstore.py        # Embed and index data into ChromaDB
│   └── run_agent.py                # CLI agent runner
├── data/
│   ├── catalog.json               # Product catalog (Amazon or synthetic)
│   ├── synthetic_users.json       # User personas
│   ├── clicks.db                  # SQLite click-tracking (ShopBot)
│   └── chroma_db/                 # Persisted ChromaDB vector store
├── tests/
├── pyproject.toml
└── .env
```

---

## Setup

```bash
# 1. Create and activate virtual environment (Python 3.10+)
python3.10 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install the package
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Set your API key and preferred provider (see Configuration section)
```

---

## Data Preparation

You can use real Amazon product data or generate a synthetic catalog.

### Option A — Amazon catalog (recommended)

Pulls product metadata from the [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset on HuggingFace. Covers 30+ categories.

```bash
python scripts/ingest_amazon_products.py
# Optional flags:
#   --per-category 100   (default: 100 products per category)
#   --total 2000         (default: 2000 total cap)
```

### Option B — Synthetic catalog

Generates catalog items and user personas via LLM (uses whichever provider is configured).

```bash
agentic-rs-generate
# Produces data/catalog.json and data/synthetic_users.json
```

### Build the vector store

Must be run after either data option above.

```bash
agentic-rs-build-store
# Embeds items and users into ChromaDB (data/chroma_db/)
```

---

## Running the App

### ShopBot Web UI (primary)

A conversational shopping assistant with a product browsing grid, click tracking, and live agent reasoning.

```bash
python server.py
# Open http://localhost:5000
```

Features:
- **Conversational flow** — agent asks one clarifying question before searching
- **Live agent trace** — streams each reasoning phase via SSE
- **Feedback loop** — users can rate recommendations and get refined results
- **Click tracking** — stores product interactions in SQLite
- **Multi-user** — switch between user personas via the nav bar

### Streamlit UI (dev / research)

Simpler interface for inspecting agent behavior step-by-step.

```bash
streamlit run src/agentic_rs/ui/app.py
```

---

## Agent Details

The `RecommendationAgent` in [src/agentic_rs/agent/loop.py](src/agentic_rs/agent/loop.py) runs a structured 4-phase loop:

| Phase | Tool | Purpose |
|-------|------|---------|
| 1 | `fetch_user_history` | Load user profile: purchase history, preferred categories, price sensitivity, wishlist keywords |
| 2 | `search_catalog` | Semantic search over ChromaDB item embeddings |
| 3 | `filter_by_attributes` | Metadata filter: price range, min rating, category, brand |
| 4 | `get_item_details` | Fetch full specs for top candidates before finalizing |

The loop iterates until the LLM emits a final JSON answer (max 10 iterations). Every step yields an `AgentStep` event — these are streamed directly to the UI.

---

## Configuration

All settings are read from `.env` (or environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `groq` | `groq` / `openai` / `anthropic` / `huggingface` |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Model name for the chosen provider |
| `GROQ_API_KEY` | — | Required when `LLM_PROVIDER=groq` |
| `OPENAI_API_KEY` | — | Required when `LLM_PROVIDER=openai` |
| `ANTHROPIC_API_KEY` | — | Required when `LLM_PROVIDER=anthropic` |
| `HF_API_KEY` | — | Required when `LLM_PROVIDER=huggingface` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformer model for embeddings |
| `NUM_RECOMMENDATIONS` | `5` | How many products the agent returns |
| `MAX_AGENT_ITERATIONS` | `10` | Max LLM calls per request |
| `CHROMA_PATH` | `data/chroma_db` | ChromaDB persistence directory |

**Recommended model**: `llama-3.3-70b-versatile` on Groq — reliably uses structured tool calls. Smaller models like `llama-3.1-8b-instant` may generate tool calls in non-standard text formats; the agent handles these automatically but results can be less consistent.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM providers | Groq, OpenAI, Anthropic, HuggingFace (OpenAI-compatible API) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | `chromadb` (persistent, cosine similarity) |
| Web backend | `flask` + SSE streaming |
| Dev UI | `streamlit` |
| Data validation | `pydantic` v2 |
| Amazon dataset | [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) via HuggingFace |
