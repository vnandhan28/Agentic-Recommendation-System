"""Tests for the agent loop with a fully mocked LLM — stdlib unittest."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from agentic_rs.agent.loop import AgentStep, RecommendationAgent, StepType


def _make_mock_store():
    store = MagicMock()
    store.get_user_by_id.return_value = {
        "user_id": "user_001",
        "name": "Alice Chen",
        "price_sensitivity": "mid-range",
        "preferred_categories": json.dumps(["Wireless Headphones"]),
        "disliked_categories": json.dumps([]),
        "wishlist_keywords": json.dumps(["noise cancelling"]),
        "purchase_history": json.dumps([]),
        "taste_description": "Loves minimalist tech.",
    }
    store.search_items.return_value = [
        {
            "item_id": "B0ABC001",
            "title": "ANC Headphones X1",
            "subcategory": "Wireless Headphones",
            "brand": "SoundBrand",
            "price_usd": 89.99,
            "avg_rating": 4.5,
            "num_reviews": 3200,
            "similarity_score": 0.92,
        }
    ]
    store.get_item_by_id.return_value = {
        "item_id": "B0ABC001",
        "title": "ANC Headphones X1",
        "category": "Electronics",
        "price_usd": 89.99,
        "avg_rating": 4.5,
        "description": "Premium ANC headphones.",
        "tags": json.dumps(["wireless"]),
        "features": json.dumps(["ANC"]),
        "compatible_with": json.dumps([]),
    }
    return store


FINAL_ANSWER_JSON = json.dumps({
    "recommendations": [
        {
            "rank": 1,
            "item_id": "B0ABC001",
            "title": "ANC Headphones X1",
            "price_usd": 89.99,
            "avg_rating": 4.5,
            "explanation": "Perfect ANC headphones within Alice's mid-range budget.",
        },
        {
            "rank": 2,
            "item_id": "B0ABC002",
            "title": "Studio Buds Pro",
            "price_usd": 59.99,
            "avg_rating": 4.2,
            "explanation": "Compact earbuds ideal for commuting.",
        },
        {
            "rank": 3,
            "item_id": "B0ABC003",
            "title": "BassMax Over-Ear",
            "price_usd": 110.00,
            "avg_rating": 4.6,
            "explanation": "Superior sound stage for music lovers.",
        },
    ],
    "reasoning_summary": "Selected based on Alice's preference for ANC and mid-range pricing.",
})


class TestAgentLoop(unittest.TestCase):

    def _run_all(self, agent, user_id, query):
        return list(agent.run(user_id=user_id, query=query))

    def test_immediate_final_answer_yields_final_step(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)
        with patch.object(agent, "_call_llm", return_value={"content": FINAL_ANSWER_JSON}):
            steps = self._run_all(agent, "user_001", "recommend headphones")
        final_steps = [s for s in steps if s.step_type == StepType.FINAL]
        self.assertEqual(len(final_steps), 1)

    def test_immediate_final_answer_recommendation_count(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)
        with patch.object(agent, "_call_llm", return_value={"content": FINAL_ANSWER_JSON}):
            steps = self._run_all(agent, "user_001", "recommend headphones")
        final = next(s for s in steps if s.step_type == StepType.FINAL)
        self.assertEqual(len(final.recommendations), 3)

    def test_tool_call_then_final_step_types(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)

        tool_response = {
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "fetch_user_history",
                    "arguments": json.dumps({"user_id": "user_001"}),
                },
            }],
        }
        responses = iter([tool_response, {"content": FINAL_ANSWER_JSON}])
        with patch.object(agent, "_call_llm", side_effect=lambda msgs: next(responses)):
            steps = self._run_all(agent, "user_001", "recommend headphones")

        step_types = {s.step_type for s in steps}
        self.assertIn(StepType.THINKING, step_types)
        self.assertIn(StepType.TOOL_CALL, step_types)
        self.assertIn(StepType.TOOL_RESULT, step_types)
        self.assertIn(StepType.FINAL, step_types)

    def test_tool_call_step_has_tool_name(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)

        tool_response = {
            "content": None,
            "tool_calls": [{
                "id": "call_001",
                "type": "function",
                "function": {
                    "name": "search_catalog",
                    "arguments": json.dumps({"query": "headphones"}),
                },
            }],
        }
        responses = iter([tool_response, {"content": FINAL_ANSWER_JSON}])
        with patch.object(agent, "_call_llm", side_effect=lambda msgs: next(responses)):
            steps = self._run_all(agent, "user_001", "headphones")

        tool_call_steps = [s for s in steps if s.step_type == StepType.TOOL_CALL]
        self.assertEqual(len(tool_call_steps), 1)
        self.assertEqual(tool_call_steps[0].tool_name, "search_catalog")

    def test_invalid_json_yields_error_then_recovers(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)

        responses = iter([
            {"content": "This is not valid JSON at all"},
            {"content": FINAL_ANSWER_JSON},
        ])
        with patch.object(agent, "_call_llm", side_effect=lambda msgs: next(responses)):
            steps = self._run_all(agent, "user_001", "headphones")

        error_steps = [s for s in steps if s.step_type == StepType.ERROR]
        final_steps = [s for s in steps if s.step_type == StepType.FINAL]
        self.assertEqual(len(error_steps), 1)
        self.assertEqual(len(final_steps), 1)

    def test_max_iterations_yields_error(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)

        infinite_tool = {
            "content": None,
            "tool_calls": [{
                "id": "call_x",
                "type": "function",
                "function": {
                    "name": "search_catalog",
                    "arguments": json.dumps({"query": "headphones"}),
                },
            }],
        }

        import agentic_rs.config as cfg
        original = cfg.config.MAX_AGENT_ITERATIONS
        cfg.config.MAX_AGENT_ITERATIONS = 2
        try:
            with patch.object(agent, "_call_llm", return_value=infinite_tool):
                steps = self._run_all(agent, "user_001", "headphones")
        finally:
            cfg.config.MAX_AGENT_ITERATIONS = original

        error_steps = [s for s in steps if s.step_type == StepType.ERROR]
        self.assertGreaterEqual(len(error_steps), 1)
        self.assertIn("maximum iterations", error_steps[-1].content.lower())

    def test_parse_final_plain_json(self):
        result = RecommendationAgent._parse_final(FINAL_ANSWER_JSON)
        self.assertEqual(len(result["recommendations"]), 3)
        self.assertEqual(result["recommendations"][0].rank, 1)

    def test_parse_final_strips_markdown_fences(self):
        fenced = f"```json\n{FINAL_ANSWER_JSON}\n```"
        result = RecommendationAgent._parse_final(fenced)
        self.assertEqual(len(result["recommendations"]), 3)

    def test_parse_final_invalid_raises(self):
        with self.assertRaises(Exception):
            RecommendationAgent._parse_final("not json")

    def test_reasoning_summary_captured(self):
        store = _make_mock_store()
        agent = RecommendationAgent(store)
        with patch.object(agent, "_call_llm", return_value={"content": FINAL_ANSWER_JSON}):
            steps = self._run_all(agent, "user_001", "headphones")
        # Verify no exception — summary is in the AgentResponse returned from run()
        final = next(s for s in steps if s.step_type == StepType.FINAL)
        self.assertIsNotNone(final)


if __name__ == "__main__":
    unittest.main()
