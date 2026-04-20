"""
Streamlit UI — Agentic Recommendation System.

Run with:
    streamlit run scripts/app.py
"""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from agentic_rs.agent.loop import AgentStep, RecommendationAgent, StepType
from agentic_rs.config import config
from agentic_rs.vectorstore.build import VectorStore

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Agentic Recommendation System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global styles
# ---------------------------------------------------------------------------

st.markdown("""
<style>
#MainMenu, footer { visibility: hidden; }

.stTextInput > div > div > input {
    border-radius: 12px !important;
    border: 2px solid #e5e7eb !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector store…")
def get_store() -> VectorStore:
    return VectorStore(persist_path=config.CHROMA_PATH)


@st.cache_resource(show_spinner="Initialising agent…")
def get_agent(_store: VectorStore) -> RecommendationAgent:
    return RecommendationAgent(_store)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(text) -> str:
    return str(text).replace("<", "&lt;").replace(">", "&gt;")


def _parse_json_field(value) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return []
    return []


RANK_MEDAL = {1: "🥇", 2: "🥈", 3: "🥉", 4: "#4", 5: "#5"}
RANK_COLOR = {
    1: "#f59e0b",
    2: "#94a3b8",
    3: "#cd7f32",
    4: "#6366f1",
    5: "#8b5cf6",
}


# ---------------------------------------------------------------------------
# Card HTML builder
# ---------------------------------------------------------------------------

def _card_html(rec, details: dict | None = None) -> str:
    accent = RANK_COLOR.get(rec.rank, "#6366f1")
    medal  = RANK_MEDAL.get(rec.rank, f"#{rec.rank}")
    full   = int(rec.avg_rating)
    stars  = "★" * full + "☆" * (5 - full)

    title = _safe(rec.title)
    expl  = _safe(rec.explanation)

    # --- Spec section from vector store details ---
    spec_html = ""
    if details:
        brand       = _safe(details.get("brand", ""))
        category    = _safe(details.get("category", ""))
        subcategory = _safe(details.get("subcategory", ""))
        num_reviews = details.get("num_reviews", 0)
        features    = _parse_json_field(details.get("features", []))[:4]   # top 4
        tags        = _parse_json_field(details.get("tags", []))[:5]        # top 5

        # Spec rows
        spec_rows = ""
        if brand:
            spec_rows += f"""
            <tr>
                <td style="color:#64748b; padding:3px 8px 3px 0; white-space:nowrap; font-size:0.75rem;">Brand</td>
                <td style="color:#e2e8f0; font-size:0.75rem; font-weight:600;">{brand}</td>
            </tr>"""
        if category or subcategory:
            cat_str = f"{category} › {subcategory}" if subcategory else category
            spec_rows += f"""
            <tr>
                <td style="color:#64748b; padding:3px 8px 3px 0; white-space:nowrap; font-size:0.75rem;">Category</td>
                <td style="color:#e2e8f0; font-size:0.75rem;">{cat_str}</td>
            </tr>"""
        if num_reviews:
            spec_rows += f"""
            <tr>
                <td style="color:#64748b; padding:3px 8px 3px 0; white-space:nowrap; font-size:0.75rem;">Reviews</td>
                <td style="color:#e2e8f0; font-size:0.75rem;">{int(num_reviews):,}</td>
            </tr>"""

        # Features list
        features_html = ""
        if features:
            items_li = "".join(
                f'<li style="margin-bottom:3px;">{_safe(f)}</li>'
                for f in features
            )
            features_html = f"""
            <div style="margin-top:10px;">
                <div style="font-size:0.72rem; font-weight:700; color:#64748b;
                            letter-spacing:0.08em; text-transform:uppercase; margin-bottom:5px;">
                    Key Features
                </div>
                <ul style="margin:0; padding-left:16px; color:#cbd5e1; font-size:0.76rem; line-height:1.6;">
                    {items_li}
                </ul>
            </div>"""

        # Tags
        tags_html = ""
        if tags:
            tag_chips = "".join(
                f'<span style="background:rgba(99,102,241,0.18); color:#a5b4fc; '
                f'border-radius:6px; padding:2px 8px; font-size:0.7rem; '
                f'white-space:nowrap;">{_safe(t)}</span>'
                for t in tags
            )
            tags_html = f"""
            <div style="display:flex; flex-wrap:wrap; gap:5px; margin-top:10px;">
                {tag_chips}
            </div>"""

        spec_html = f"""
        <div style="border-top:1px solid {accent}22; margin:12px 0 10px; padding-top:12px;">
            <div style="font-size:0.72rem; font-weight:700; color:#64748b;
                        letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;">
                Specifications
            </div>
            <table style="border-collapse:collapse; width:100%;">
                {spec_rows}
            </table>
            {features_html}
            {tags_html}
        </div>
        """

    return f"""
    <div style="
        background: linear-gradient(145deg, #1e1e2e 0%, #16162a 100%);
        border: 1.5px solid {accent}55;
        border-radius: 18px;
        padding: 0 0 20px;
        overflow: hidden;
        box-shadow: 0 8px 32px {accent}22, 0 2px 8px rgba(0,0,0,0.4);
        display: flex;
        flex-direction: column;
    ">
        <!-- top accent bar -->
        <div style="height:5px; background:linear-gradient(90deg,{accent},{accent}66);"></div>

        <div style="padding:18px 18px 0;">
            <!-- rank row -->
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:12px;">
                <span style="font-size:1.5rem; line-height:1;">{medal}</span>
                <span style="
                    background:{accent}22; color:{accent};
                    border:1px solid {accent}44;
                    border-radius:20px; padding:2px 10px;
                    font-size:0.68rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase;
                ">Pick #{rec.rank}</span>
            </div>

            <!-- title -->
            <div style="font-weight:700; font-size:0.95rem; color:#f1f5f9;
                        line-height:1.45; margin-bottom:12px;">{title}</div>

            <!-- price + rating badges -->
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px;">
                <span style="background:rgba(245,158,11,0.2); color:#fbbf24;
                             border-radius:8px; padding:4px 11px;
                             font-size:0.78rem; font-weight:600;">
                    {stars} {rec.avg_rating:.1f}
                </span>
                <span style="background:rgba(16,185,129,0.18); color:#34d399;
                             border-radius:8px; padding:4px 11px;
                             font-size:0.78rem; font-weight:600;">
                    ${rec.price_usd:.2f}
                </span>
            </div>

            <!-- specs block -->
            {spec_html}

            <!-- divider before explanation -->
            <div style="border-top:1px solid {accent}22; margin:12px 0 10px;"></div>

            <!-- why this item -->
            <div style="font-size:0.72rem; font-weight:700; color:#64748b;
                        letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;">
                Why This Pick
            </div>
            <div style="font-size:0.82rem; color:#94a3b8; line-height:1.65;">{expl}</div>
        </div>
    </div>
    """


# ---------------------------------------------------------------------------
# Results renderer
# ---------------------------------------------------------------------------

def render_results(recs, details_map: dict, reasoning_summary: str = "") -> None:
    if not recs:
        st.warning("No recommendations were returned.")
        return

    st.markdown("### 🎯 Top Picks For You")

    if reasoning_summary:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
            border-left: 4px solid #6366f1;
            border-radius: 10px;
            padding: 13px 18px;
            margin: 8px 0 20px;
            font-size: 0.88rem;
            color: #c4b5fd;
            line-height: 1.6;
        ">
            💡 <strong style="color:#a78bfa;">Agent reasoning:</strong> {_safe(reasoning_summary)}
        </div>
        """, unsafe_allow_html=True)

    row1 = recs[:3]
    row2 = recs[3:]

    cards_row1 = "".join(_card_html(r, details_map.get(r.item_id)) for r in row1)
    cards_row2 = "".join(_card_html(r, details_map.get(r.item_id)) for r in row2)

    grid_html = f"""<!DOCTYPE html>
    <html>
    <head>
    <style>
      * {{ box-sizing:border-box; margin:0; padding:0;
           font-family:'Inter','Segoe UI',sans-serif; }}
      body {{ background:transparent; }}
      .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin-bottom:16px; }}
      .grid2 {{ display:grid; grid-template-columns:repeat(2,1fr); gap:16px;
                width:66.6%; margin:0 auto; }}
    </style>
    </head>
    <body>
      <div class="grid3">{cards_row1}</div>
      {"<div class='grid2'>" + cards_row2 + "</div>" if row2 else ""}
    </body>
    </html>"""

    height = 780 if row2 else 680
    components.html(grid_html, height=height, scrolling=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #9333ea 100%);
        border-radius: 20px;
        padding: 38px 36px 30px;
        margin-bottom: 24px;
        color: white;
    ">
        <div style="font-size:2rem; font-weight:800; margin-bottom:8px;">🤖 Agentic Recommendation System</div>
        <div style="font-size:1rem; opacity:0.88;">
            An AI agent that reasons over your preferences and finds the perfect products —
            <em>personalized just for you.</em>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        store = get_store()
        agent = get_agent(store)
    except Exception as exc:
        st.error(
            f"Failed to load vector store: {exc}\n\n"
            "Run `python scripts/generate_data.py` and `python scripts/build_vectorstore.py` first."
        )
        st.stop()

    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        users = store.list_users()
        if not users:
            st.error("No users found. Generate data first.")
            st.stop()

        user_options = {u["name"]: u["user_id"] for u in users}
        selected_name = st.selectbox("👤 Select user", options=list(user_options.keys()))
        selected_user_id = user_options[selected_name]

        st.markdown("---")
        st.markdown(f"🪪 **User ID:** `{selected_user_id}`")
        st.markdown(f"📦 **Catalog:** {store.item_count()} items")
        st.markdown(f"🧠 **Model:** `{config.LLM_MODEL}`")

    st.markdown(
        "<div style='font-size:0.72rem; font-weight:700; letter-spacing:0.1em; "
        "text-transform:uppercase; color:#8b5cf6; margin-bottom:4px;'>Your Request</div>",
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([5, 1], gap="small")
    with col1:
        query = st.text_input(
            "query",
            placeholder="e.g. 'I need a gift for a tech-savvy teenager under $100'",
            label_visibility="collapsed",
        )
    with col2:
        run_btn = st.button("🚀 Recommend", use_container_width=True, type="primary")

    if run_btn and query:
        steps: list[AgentStep] = []

        with st.spinner("🔍 Agent is analysing your request and searching the catalog…"):
            for step in agent.run(user_id=selected_user_id, query=query):
                steps.append(step)

        final_step = next((s for s in steps if s.step_type == StepType.FINAL), None)
        recs = final_step.recommendations if final_step else []
        reasoning_summary = final_step.content if final_step else ""

        # Enrich each recommendation with full item details from the vector store
        details_map: dict = {}
        for rec in recs:
            item = store.get_item_by_id(rec.item_id)
            if item:
                details_map[rec.item_id] = item

        error_steps = [s for s in steps if s.step_type == StepType.ERROR]
        if error_steps:
            with st.expander("⚠️ Issues encountered", expanded=False):
                for e in error_steps:
                    st.warning(e.content)

        render_results(recs, details_map, reasoning_summary)

    elif run_btn and not query:
        st.warning("Please enter a query before clicking Recommend.")


if __name__ == "__main__":
    main()
