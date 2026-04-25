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
from flask import Flask, Response, jsonify, request, send_file, stream_with_context

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

# VectorStore and Agent are loaded lazily so Flask can serve the page
# immediately without blocking on model initialisation.
_STORE = None
_AGENT = None

def _get_store():
    global _STORE
    if _STORE is None:
        from agentic_rs.vectorstore.build import VectorStore
        _STORE = VectorStore()
    return _STORE

def _get_agent():
    global _AGENT
    if _AGENT is None:
        from agentic_rs.agent.loop import RecommendationAgent
        _AGENT = RecommendationAgent(_get_store())
    return _AGENT

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
# Conversational helpers
# ---------------------------------------------------------------------------

def _generate_clarification(query: str) -> str:
    """Generate one targeted clarifying question to narrow down the query."""
    import openai
    client = openai.OpenAI(
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    model = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")
    prompt = (
        f'A shopper typed: "{query}"\n\n'
        "Generate ONE short clarifying question to narrow down the best product match. "
        "Pick the single most impactful dimension — for example: budget range, "
        "specific use case, who it's for, or a must-have feature. "
        "Keep it conversational and under 25 words.\n\n"
        "Return ONLY the question text, nothing else."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=80,
    )
    return resp.choices[0].message.content.strip()


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
    """Phase 1 only: generate a single clarifying question."""
    data    = request.get_json(force=True)
    user_id = data.get("user_id", "").strip()
    query   = data.get("query", "").strip()

    if not user_id or not query:
        return jsonify({"error": "user_id and query required"}), 400

    try:
        question = _generate_clarification(query)
        return jsonify({"type": "clarification", "question": question})
    except Exception as exc:
        return jsonify({"error": f"Failed to generate question: {exc}"}), 500


def _rec_to_dict(r) -> dict:
    """Serialize a Recommendation (Pydantic or dataclass) to a plain dict."""
    if hasattr(r, "model_dump"):
        return r.model_dump()
    return {
        "rank": r.rank, "item_id": r.item_id, "title": r.title,
        "price_usd": r.price_usd, "avg_rating": r.avg_rating,
        "explanation": r.explanation,
    }


@app.route("/api/recommend/stream", methods=["POST"])
def recommend_stream():
    """
    Phase 2 / 3: run the full agentic loop and stream AgentStep events as SSE.
    The client sends conversation context (clarification Q&A or feedback) which
    is appended to the query so the agent sees it as part of the request.
    """
    data = request.get_json(force=True)
    user_id                = data.get("user_id", "").strip()
    query                  = data.get("query", "").strip()
    clarification_question = data.get("clarification_question", "").strip()
    clarification_answer   = data.get("clarification_answer", "").strip()
    feedback               = data.get("feedback", "").strip()
    previous_recs          = data.get("previous_recommendations", [])

    if not user_id or not query:
        return jsonify({"error": "user_id and query required"}), 400

    # Build the enriched query the agent will see
    enhanced = query
    if clarification_question and clarification_answer:
        enhanced += (
            f"\n\nPreference clarification:\n"
            f"  Q: {clarification_question}\n"
            f"  A: {clarification_answer}"
        )
    if feedback and previous_recs:
        prev_titles = ", ".join(r.get("title", "") for r in previous_recs[:3])
        enhanced += (
            f"\n\nUser feedback on previous picks ({prev_titles}…): \"{feedback}\"\n"
            "Recommend DIFFERENT products that directly address this feedback."
        )

    agent = _get_agent()

    def generate():
        try:
            for step in agent.run(user_id=user_id, query=enhanced):
                event: dict = {
                    "type":      step.step_type.value,
                    "iteration": step.iteration,
                    "content":   step.content,
                }
                if step.tool_name:
                    event["tool_name"] = step.tool_name
                if step.tool_args:
                    event["tool_args"] = step.tool_args
                if step.tool_result:
                    r = step.tool_result
                    if "error" in r:
                        event["result_summary"] = None   # hidden from trace
                    elif "items" in r:
                        event["result_summary"] = f"{r.get('count', len(r['items']))} items"
                    elif "user" in r:
                        u = r["user"]
                        event["result_summary"] = (
                            f"{u.get('name', '')} — {u.get('price_sensitivity', '')}"
                        )
                    elif "item" in r:
                        event["result_summary"] = r["item"].get("title", "")
                    else:
                        event["result_summary"] = str(r)[:80]
                if step.recommendations:
                    event["recommendations"] = [
                        _rec_to_dict(r) for r in step.recommendations
                    ]
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return Response(
        stream_with_context(generate()),
        content_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        },
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
