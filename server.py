"""
Flask backend for the ShopBot recommendation prototype.

Routes:
  GET  /                  → serve index.html
  GET  /api/products      → 20 products (5 per category)
  GET  /api/users         → 3 hardcoded test users
  POST /api/click         → record a product click to SQLite
  POST /api/recommend     → run agent: click history + ChromaDB → 3 picks
"""

import json
import os
import re
import sqlite3
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_file

load_dotenv()

ROOT = Path(__file__).parent
CATALOG_PATH = ROOT / "data" / "catalog.json"
USERS_PATH   = ROOT / "data" / "synthetic_users.json"
DB_PATH      = ROOT / "data" / "clicks.db"
INDEX_PATH   = ROOT / "index.html"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Boot-time data loading
# ---------------------------------------------------------------------------

def _load_products() -> list[dict]:
    """Return 20 products spread across all categories (5 per category)."""
    catalog: list[dict] = json.loads(CATALOG_PATH.read_text())
    by_cat: dict[str, list] = defaultdict(list)
    for item in catalog:
        by_cat[item["category"]].append(item)
    selected: list[dict] = []
    for items in by_cat.values():
        selected.extend(items[:5])
    return selected[:20]


def _load_test_users() -> list[dict]:
    """Return the first 3 users from synthetic_users.json."""
    users: list[dict] = json.loads(USERS_PATH.read_text())
    return [
        {"user_id": u["user_id"], "name": u["name"],
         "price_sensitivity": u.get("price_sensitivity", "mid-range"),
         "preferred_categories": u.get("preferred_categories", [])}
        for u in users[:3]
    ]


PRODUCTS   = _load_products()
TEST_USERS = _load_test_users()
# Keyed by item_id for fast look-up during recommendations
CATALOG_MAP: dict[str, dict] = {p["item_id"]: p for p in json.loads(CATALOG_PATH.read_text())}

# VectorStore is loaded lazily on the first /api/recommend call so Flask
# can serve the page immediately without blocking on model initialisation.
_STORE = None

def _get_store():
    global _STORE
    if _STORE is None:
        from agentic_rs.vectorstore.build import VectorStore
        _STORE = VectorStore()
    return _STORE

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS clicks (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT    NOT NULL,
                item_id    TEXT    NOT NULL,
                clicked_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


_init_db()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_file(INDEX_PATH)


@app.route("/api/products")
def get_products():
    return jsonify(PRODUCTS)


@app.route("/api/users")
def get_users():
    return jsonify(TEST_USERS)


@app.route("/api/click", methods=["POST"])
def record_click():
    data = request.get_json(force=True)
    user_id = data.get("user_id", "").strip()
    item_id = data.get("item_id", "").strip()
    if not user_id or not item_id:
        return jsonify({"error": "user_id and item_id required"}), 400
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO clicks (user_id, item_id) VALUES (?, ?)",
            (user_id, item_id),
        )
        conn.commit()
    return jsonify({"ok": True})


@app.route("/api/recommend", methods=["POST"])
def recommend():
    data    = request.get_json(force=True)
    user_id = data.get("user_id", "").strip()
    query   = data.get("query", "").strip()

    if not user_id or not query:
        return jsonify({"error": "user_id and query required"}), 400

    # 1. Fetch click history from SQLite
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT item_id FROM clicks WHERE user_id = ? ORDER BY clicked_at DESC LIMIT 10",
            (user_id,),
        ).fetchall()
    clicked_ids   = [r["item_id"] for r in rows]
    clicked_items = [CATALOG_MAP[i] for i in clicked_ids if i in CATALOG_MAP]

    # 2. Semantic search — query only, no click history mixed in.
    # Click history is for LLM personalisation only, not retrieval.
    # Mixing clicks into the vector query biases results toward browsed
    # categories even when the query asks for something completely different.
    candidates: list[dict] = _get_store().search_items(query, n_results=12)

    if not candidates:
        return jsonify({"error": "No candidates found in catalog"}), 404

    # 3. Build LLM prompt — click history is soft context only
    click_ctx = ""
    if clicked_items:
        lines = "\n".join(
            f"  • {p['title']} ({p['category']}, ${p['price_usd']:.2f}, ★{p['avg_rating']})"
            for p in clicked_items
        )
        click_ctx = (
            f"\nUser's recent browsing (use only to personalise explanations, "
            f"do NOT use to override the query intent):\n{lines}\n"
        )

    candidate_lines = "\n".join(
        f"{i+1}. [{c['item_id']}] {c['title']} — "
        f"${c.get('price_usd', 0):.2f} — {c.get('category', '')} — ★{c.get('avg_rating', 0)}"
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are a personalised shopping assistant.
The user asked: "{query}"

Your task:
1. Pick exactly 5 products from the candidate list that BEST MATCH THE QUERY.
   The query is the primary signal — only recommend products clearly relevant to it.
2. Use the browsing history (if provided) solely to write a more personalised explanation.
   Do NOT pick a product just because the user browsed a related category.
{click_ctx}
Candidates (choose only from this list):
{candidate_lines}

Respond with ONLY a JSON object (no markdown fences):
{{
  "recommendations": [
    {{
      "rank": 1,
      "item_id": "...",
      "title": "...",
      "price": 0.00,
      "rating": 0.0,
      "category": "...",
      "explanation": "2-3 sentence personalised reason"
    }}
  ]
}}
"""

    # 5. Call Groq via OpenAI-compatible SDK
    import openai
    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_tokens=1200,
    )
    text = response.choices[0].message.content.strip()

    # 6. Parse JSON — strip markdown fences then try progressively looser strategies
    text = re.sub(r"```[a-z]*\n?", "", text).strip()
    text = re.sub(r"\n?```", "", text).strip()

    payload = None

    # Strategy 1: direct parse
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract outermost {...} block
    if payload is None:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                payload = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    # Strategy 3: truncated JSON — find last complete recommendation object and close arrays/objects
    if payload is None:
        recs = re.findall(
            r'\{\s*"rank"\s*:\s*\d+.*?"explanation"\s*:\s*"[^"]*"\s*\}',
            text, re.DOTALL
        )
        if recs:
            payload = {"recommendations": [json.loads(r) for r in recs]}

    if payload is None:
        return jsonify({"error": "LLM returned unparseable response", "raw": text}), 500

    return jsonify(payload)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
