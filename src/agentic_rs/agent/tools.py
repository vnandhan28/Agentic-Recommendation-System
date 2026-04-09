"""
Step 3a — Agent tools.

Four tools the LLM agent can call:
  1. search_catalog       — semantic search over item embeddings
  2. fetch_user_history   — retrieve a user's purchase history + preferences
  3. filter_by_attributes — filter items by metadata (price, rating, category…)
  4. get_item_details     — fetch full metadata for a specific item ID

Tool schemas use the OpenAI function-calling format (also compatible with Anthropic).
"""
from __future__ import annotations

import json
from typing import Any, Optional

# VectorStore is imported lazily so this module can be imported in test envs
# without chromadb / sentence-transformers installed.

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_catalog",
            "description": (
                "Perform a semantic search over the product catalog. "
                "Returns the top matching items ranked by relevance to the query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "n_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 8, max 20)",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_user_history",
            "description": (
                "Retrieve a user's full profile: purchase history, "
                "taste description, preferred/disliked categories, "
                "price sensitivity, and wishlist keywords. "
                "Always call this first for personalized recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "string",
                        "description": "The user ID, e.g. 'user_001'",
                    },
                },
                "required": ["user_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_attributes",
            "description": (
                "Filter the catalog by metadata: price range, min rating, "
                "category, subcategory, or brand."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category":    {"type": "string"},
                    "subcategory": {"type": "string"},
                    "min_price":   {"type": "number"},
                    "max_price":   {"type": "number"},
                    "min_rating":  {"type": "number"},
                    "brand":       {"type": "string"},
                    "n_results":   {"type": "integer", "default": 10},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_item_details",
            "description": (
                "Fetch complete metadata for a specific item by its item_id. "
                "Use before finalising a recommendation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "item_id": {
                        "type": "string",
                        "description": "The item_id of the product",
                    },
                },
                "required": ["item_id"],
            },
        },
    },
]


class ToolExecutor:
    """Executes tool calls against the vector store and returns structured results."""

    def __init__(self, store: Any):
        self.store = store

    def execute(self, tool_name: str, tool_args: dict) -> dict[str, Any]:
        handlers = {
            "search_catalog":       self._search_catalog,
            "fetch_user_history":   self._fetch_user_history,
            "filter_by_attributes": self._filter_by_attributes,
            "get_item_details":     self._get_item_details,
        }
        handler = handlers.get(tool_name)
        if handler is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return handler(**tool_args)
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _search_catalog(self, query: str, n_results: int = 8) -> dict:
        n_results = min(int(n_results), 20)
        results = self.store.search_items(query, n_results=n_results)
        return {
            "tool": "search_catalog",
            "query": query,
            "count": len(results),
            "items": [self._slim(r) for r in results],
        }

    def _fetch_user_history(self, user_id: str) -> dict:
        user = self.store.get_user_by_id(user_id)
        if user is None:
            return {"error": f"User '{user_id}' not found"}
        for field in ("preferred_categories", "disliked_categories",
                      "wishlist_keywords", "purchase_history"):
            if field in user and isinstance(user[field], str):
                try:
                    user[field] = json.loads(user[field])
                except json.JSONDecodeError:
                    pass
        return {"tool": "fetch_user_history", "user": user}

    def _filter_by_attributes(
        self,
        category: Optional[str] = None,
        subcategory: Optional[str] = None,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        min_rating: Optional[float] = None,
        brand: Optional[str] = None,
        n_results: int = 10,
    ) -> dict:
        conditions = []
        if category:    conditions.append({"category":    {"$eq": category}})
        if subcategory: conditions.append({"subcategory": {"$eq": subcategory}})
        if brand:       conditions.append({"brand":       {"$eq": brand}})
        if min_price is not None: conditions.append({"price_usd":  {"$gte": float(min_price)}})
        if max_price is not None: conditions.append({"price_usd":  {"$lte": float(max_price)}})
        if min_rating is not None: conditions.append({"avg_rating": {"$gte": float(min_rating)}})

        where = ({"$and": conditions} if len(conditions) > 1
                 else (conditions[0] if conditions else None))

        results = self.store.search_items("product", n_results=int(n_results), where=where)

        filters_applied = {k: v for k, v in {
            "category": category, "subcategory": subcategory, "brand": brand,
            "min_price": min_price, "max_price": max_price, "min_rating": min_rating,
        }.items() if v is not None}

        return {
            "tool": "filter_by_attributes",
            "filters_applied": filters_applied,
            "count": len(results),
            "items": [self._slim(r) for r in results],
        }

    def _get_item_details(self, item_id: str) -> dict:
        item = self.store.get_item_by_id(item_id)
        if item is None:
            return {"error": f"Item '{item_id}' not found"}
        for field in ("tags", "features", "compatible_with"):
            if field in item and isinstance(item[field], str):
                try:
                    item[field] = json.loads(item[field])
                except json.JSONDecodeError:
                    pass
        return {"tool": "get_item_details", "item": item}

    @staticmethod
    def _slim(item: dict) -> dict:
        return {
            "item_id":          item.get("item_id"),
            "title":            item.get("title"),
            "subcategory":      item.get("subcategory"),
            "brand":            item.get("brand"),
            "price_usd":        item.get("price_usd"),
            "avg_rating":       item.get("avg_rating"),
            "num_reviews":      item.get("num_reviews"),
            "similarity_score": item.get("similarity_score"),
        }
