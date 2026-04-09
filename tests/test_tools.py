"""Tests for agent tool execution (uses a mock VectorStore) — stdlib unittest."""
from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

from agentic_rs.agent.tools import ToolExecutor


def _make_mock_store():
    store = MagicMock()
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
        "subcategory": "Wireless Headphones",
        "brand": "SoundBrand",
        "price_usd": 89.99,
        "avg_rating": 4.5,
        "num_reviews": 3200,
        "description": "Premium noise-cancelling headphones.",
        "tags": json.dumps(["wireless", "anc"]),
        "features": json.dumps(["30hr battery", "ANC"]),
        "compatible_with": json.dumps(["iOS", "Android"]),
    }
    store.get_user_by_id.return_value = {
        "user_id": "user_001",
        "name": "Alice Chen",
        "age": 32,
        "occupation": "Software Engineer",
        "price_sensitivity": "mid-range",
        "preferred_categories": json.dumps(["Wireless Headphones"]),
        "disliked_categories": json.dumps([]),
        "wishlist_keywords": json.dumps(["noise cancelling"]),
        "purchase_history": json.dumps([]),
        "taste_description": "Loves minimalist tech.",
    }
    return store


class TestToolExecutor(unittest.TestCase):
    def setUp(self):
        self.store = _make_mock_store()
        self.executor = ToolExecutor(self.store)

    def test_search_catalog_returns_tool_key(self):
        result = self.executor.execute("search_catalog", {"query": "wireless headphones", "n_results": 5})
        self.assertEqual(result["tool"], "search_catalog")

    def test_search_catalog_count(self):
        result = self.executor.execute("search_catalog", {"query": "wireless headphones", "n_results": 5})
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["items"][0]["item_id"], "B0ABC001")

    def test_search_catalog_passes_n_results(self):
        self.executor.execute("search_catalog", {"query": "headphones", "n_results": 5})
        self.store.search_items.assert_called_once_with("headphones", n_results=5)

    def test_search_catalog_default_n_results(self):
        self.executor.execute("search_catalog", {"query": "headphones"})
        self.store.search_items.assert_called_once_with("headphones", n_results=8)

    def test_search_catalog_caps_at_20(self):
        self.executor.execute("search_catalog", {"query": "headphones", "n_results": 999})
        call_kwargs = self.store.search_items.call_args[1]
        self.assertEqual(call_kwargs["n_results"], 20)

    def test_fetch_user_history_tool_key(self):
        result = self.executor.execute("fetch_user_history", {"user_id": "user_001"})
        self.assertEqual(result["tool"], "fetch_user_history")

    def test_fetch_user_history_name(self):
        result = self.executor.execute("fetch_user_history", {"user_id": "user_001"})
        self.assertEqual(result["user"]["name"], "Alice Chen")

    def test_fetch_user_history_json_fields_parsed(self):
        result = self.executor.execute("fetch_user_history", {"user_id": "user_001"})
        user = result["user"]
        self.assertIsInstance(user["preferred_categories"], list)
        self.assertIsInstance(user["purchase_history"], list)

    def test_fetch_user_history_not_found(self):
        self.store.get_user_by_id.return_value = None
        result = self.executor.execute("fetch_user_history", {"user_id": "user_999"})
        self.assertIn("error", result)

    def test_get_item_details_tool_key(self):
        result = self.executor.execute("get_item_details", {"item_id": "B0ABC001"})
        self.assertEqual(result["tool"], "get_item_details")

    def test_get_item_details_title(self):
        result = self.executor.execute("get_item_details", {"item_id": "B0ABC001"})
        self.assertEqual(result["item"]["title"], "ANC Headphones X1")

    def test_get_item_details_json_fields_parsed(self):
        result = self.executor.execute("get_item_details", {"item_id": "B0ABC001"})
        item = result["item"]
        self.assertIsInstance(item["tags"], list)
        self.assertIsInstance(item["features"], list)
        self.assertIsInstance(item["compatible_with"], list)

    def test_get_item_details_not_found(self):
        self.store.get_item_by_id.return_value = None
        result = self.executor.execute("get_item_details", {"item_id": "MISSING"})
        self.assertIn("error", result)

    def test_filter_no_filters(self):
        result = self.executor.execute("filter_by_attributes", {})
        self.assertEqual(result["tool"], "filter_by_attributes")
        self.assertEqual(result["filters_applied"], {})

    def test_filter_with_price_range(self):
        result = self.executor.execute(
            "filter_by_attributes",
            {"min_price": 50.0, "max_price": 150.0},
        )
        self.assertEqual(result["filters_applied"]["min_price"], 50.0)
        self.assertEqual(result["filters_applied"]["max_price"], 150.0)

    def test_filter_omits_none_values(self):
        result = self.executor.execute(
            "filter_by_attributes",
            {"min_price": 50.0},
        )
        self.assertNotIn("max_price", result["filters_applied"])
        self.assertNotIn("min_rating", result["filters_applied"])

    def test_unknown_tool_returns_error(self):
        result = self.executor.execute("nonexistent_tool", {})
        self.assertIn("error", result)
        self.assertIn("Unknown tool", result["error"])

    def test_slim_item_has_required_keys(self):
        result = self.executor.execute("search_catalog", {"query": "headphones"})
        item = result["items"][0]
        for key in ("item_id", "title", "brand", "price_usd", "avg_rating"):
            self.assertIn(key, item)

    def test_slim_item_excludes_description(self):
        result = self.executor.execute("search_catalog", {"query": "headphones"})
        self.assertNotIn("description", result["items"][0])


if __name__ == "__main__":
    unittest.main()
