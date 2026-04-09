# Agentic Recommendation System

An end-to-end agentic recommendation system that combines **LLM-powered data generation**, **vector search**, and a **tool-calling agent** to deliver personalized recommendations with transparent reasoning.

---

## Concept

```
synthetic_users.json   ←  15 personas with history + preferences
        ↓
catalog.json           ←  200–300 items with rich metadata
        ↓
ChromaDB embeddings    ←  items + users embedded via sentence-transformers
        ↓
Agent loop             ←  tool-calling agent with 4 tools
        ↓
Streamlit UI           ←  shows agent reasoning live
```

The agent receives a user request and iterates: it decides which tools to call (search catalog, fetch user history, filter by attributes, get item details), executes them, feeds the results back to the LLM, and repeats until it has enough information to return **three personalized recommendations with explanations**. Every reasoning step is exposed live in the UI.

---

## Project Structure

```
agentic-rs/
├── src/agentic_rs/
│   ├── data/           # Data generation (catalog + synthetic users)
│   ├── vectorstore/    # ChromaDB embedding + indexing
│   ├── tools/          # The 4 agent tools
│   ├── agent/          # Agent loop logic
│   └── ui/             # Streamlit app
├── data/               # Generated JSON files (gitignored after generation)
├── scripts/            # CLI helpers
├── pyproject.toml
└── README.md
```

---

## Setup

```bash
# 1. Create and activate virtual environment (Python 3.10)
python3.10 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install the package
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Add your OPENAI_API_KEY (or other provider) to .env
```

---

## Usage

```bash
# Step 1: Generate catalog + synthetic users
agentic-rs-generate

# Step 2: Embed and index into ChromaDB
agentic-rs-build-store

# Step 3: Launch the Streamlit UI
streamlit run src/agentic_rs/ui/app.py
```

---

## Dataset

Item metadata is inspired by the [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset. Synthetic users are LLM-generated personas with realistic preference profiles and interaction histories.

---

## Tech Stack

| Component | Library |
|---|---|
| LLM | OpenAI API (GPT-4o / GPT-4o-mini) |
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector store | `chromadb` |
| UI | `streamlit` |
| Data validation | `pydantic` |
