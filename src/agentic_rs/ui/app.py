"""
Step 4 — Streamlit UI.

Displays:
  - User selector (sidebar)
  - Free-text query input
  - Live agent reasoning trace (tool calls + results stream in real time)
  - Final recommendation cards with explanations

Run with:
    streamlit run scripts/app.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

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
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading vector store…")
def get_store() -> VectorStore:
    return VectorStore(persist_path=config.CHROMA_PATH)


@st.cache_resource(show_spinner="Initialising agent…")
def get_agent(store: VectorStore) -> RecommendationAgent:
    return RecommendationAgent(store)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

STEP_ICONS = {
    StepType.THINKING: "🧠",
    StepType.TOOL_CALL: "🔧",
    StepType.TOOL_RESULT: "📦",
    StepType.FINAL: "✅",
    StepType.ERROR: "❌",
}

STEP_COLORS = {
    StepType.THINKING: "#6366f1",   # indigo
    StepType.TOOL_CALL: "#f59e0b",  # amber
    StepType.TOOL_RESULT: "#10b981", # emerald
    StepType.FINAL: "#3b82f6",      # blue
    StepType.ERROR: "#ef4444",      # red
}


def render_step(step: AgentStep, container) -> None:
    icon = STEP_ICONS.get(step.step_type, "•")
    color = STEP_COLORS.get(step.step_type, "#6b7280")
    label = step.step_type.value.upper().replace("_", " ")

    with container:
        st.markdown(
            f"""
            <div style="
                border-left: 3px solid {color};
                padding: 8px 12px;
                margin: 6px 0;
                border-radius: 4px;
                background: rgba(0,0,0,0.03);
            ">
                <span style="color:{color}; font-weight:600; font-size:0.8rem;">
                    {icon} [{step.iteration}] {label}
                </span>
                <div style="margin-top:4px; font-size:0.9rem;">{step.content}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if step.tool_args and step.step_type == StepType.TOOL_CALL:
            with st.expander("Args", expanded=False):
                st.json(step.tool_args)

        if step.tool_result and step.step_type == StepType.TOOL_RESULT:
            with st.expander("Result", expanded=False):
                st.json(step.tool_result)


def render_recommendation_cards(steps: list[AgentStep]) -> None:
    final_step = next((s for s in steps if s.step_type == StepType.FINAL), None)
    if not final_step or not final_step.recommendations:
        return

    st.markdown("---")
    st.subheader("🎯 Recommendations")

    cols = st.columns(len(final_step.recommendations))
    for col, rec in zip(cols, final_step.recommendations):
        with col:
            st.markdown(
                f"""
                <div style="
                    border: 1px solid #e5e7eb;
                    border-radius: 12px;
                    padding: 16px;
                    height: 100%;
                ">
                    <div style="font-size:1.5rem; margin-bottom:4px;">#{rec.rank}</div>
                    <div style="font-weight:600; font-size:0.95rem; margin-bottom:8px;">{rec.title}</div>
                    <div style="color:#6b7280; font-size:0.85rem;">
                        ⭐ {rec.avg_rating} &nbsp;|&nbsp; ${rec.price_usd:.2f}
                    </div>
                    <div style="margin-top:12px; font-size:0.85rem; color:#374151;">
                        {rec.explanation}
                    </div>
                    <div style="margin-top:8px; font-size:0.75rem; color:#9ca3af;">
                        ID: {rec.item_id}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🤖 Agentic Recommendation System")
    st.caption(
        "A tool-calling LLM agent that delivers personalized product recommendations "
        "with a live reasoning trace."
    )

    # --- Load resources ---
    try:
        store = get_store()
        agent = get_agent(store)
    except Exception as exc:
        st.error(
            f"Failed to load vector store: {exc}\n\n"
            "Run `python scripts/generate_data.py` and `python scripts/build_vectorstore.py` first."
        )
        st.stop()

    # --- Sidebar ---
    with st.sidebar:
        st.header("Settings")

        users = store.list_users()
        if not users:
            st.error("No users found. Generate data first.")
            st.stop()

        user_options = {u["name"]: u["user_id"] for u in users}
        selected_name = st.selectbox("Select user", options=list(user_options.keys()))
        selected_user_id = user_options[selected_name]

        st.markdown("---")
        st.markdown(f"**User ID:** `{selected_user_id}`")
        st.markdown(f"**Items in catalog:** {store.item_count()}")
        st.markdown(f"**LLM:** `{config.LLM_PROVIDER}/{config.LLM_MODEL}`")
        st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")

    # --- Query input ---
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g. 'I need a gift for a tech-savvy teenager under $100'",
            label_visibility="collapsed",
        )
    with col2:
        run_btn = st.button("🚀 Get Recommendations", use_container_width=True, type="primary")

    # --- Agent run ---
    if run_btn and query:
        st.markdown("---")
        st.subheader("🔍 Agent Reasoning Trace")

        trace_container = st.container()
        steps: list[AgentStep] = []

        with st.spinner("Agent is working…"):
            for step in agent.run(user_id=selected_user_id, query=query):
                steps.append(step)
                render_step(step, trace_container)
                time.sleep(0.05)  # small delay so each step is visible

        render_recommendation_cards(steps)

        # Check for errors
        error_steps = [s for s in steps if s.step_type == StepType.ERROR]
        if error_steps:
            st.warning(f"{len(error_steps)} error(s) occurred during agent execution.")

    elif run_btn and not query:
        st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
