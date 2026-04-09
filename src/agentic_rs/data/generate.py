"""
Generate catalog.json and synthetic_users.json using an LLM.
Run: python scripts/generate_data.py
"""

from __future__ import annotations

import json
import os
import random
import re
import uuid
from pathlib import Path

from rich.console import Console
from rich.progress import track

from agentic_rs.config import config
from agentic_rs.models import CatalogItem, PurchaseEvent, SyntheticUser

console = Console()

CATEGORY_SEEDS = [
    ("Electronics", "Headphones"),
    ("Electronics", "Keyboards"),
    ("Electronics", "Monitors"),
    ("Electronics", "Smart Home"),
    ("Electronics", "Cameras"),
    ("Books", "Science Fiction"),
    ("Books", "Self-Help"),
    ("Sports & Outdoors", "Running"),
    ("Sports & Outdoors", "Camping"),
    ("Home & Kitchen", "Coffee Makers"),
    ("Home & Kitchen", "Cookware"),
    ("Beauty & Personal Care", "Skincare"),
    ("Toys & Games", "Board Games"),
    ("Clothing", "Sneakers"),
]


def _get_client():
    from openai import OpenAI
    if config.LLM_PROVIDER == "huggingface":
        return OpenAI(api_key=config.api_key, base_url="https://router.huggingface.co/v1/")
    if config.LLM_PROVIDER == "groq":
        return OpenAI(api_key=config.api_key, base_url="https://api.groq.com/openai/v1")
    return OpenAI(api_key=config.api_key)


def _extract_json(text: str) -> dict | list:
    text = text.strip()
    if "```" in text:
        for part in text.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith(("{", "[")):
                text = part
                break
    # Find first { or [ in case of leading prose
    match = re.search(r"[{\[]", text)
    if match:
        text = text[match.start():]
    return json.loads(text)


def _llm_json(prompt: str, system: str = "") -> dict | list:
    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = dict(
        model=config.LLM_MODEL,
        messages=messages,
        temperature=0.9,
        max_tokens=4096,
    )
    # json_object mode only reliable on OpenAI
    if config.LLM_PROVIDER == "openai":
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return _extract_json(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------

ITEM_SYSTEM = (
    "You are a product catalog generator. "
    "Always respond with valid JSON only, no markdown fences."
)

ITEM_PROMPT_TEMPLATE = """
Generate {n} realistic product catalog entries for the category: {category} > {subcategory}.

Each entry must be a JSON object inside an "items" array with these EXACT fields:
- item_id: use one of the provided IDs (string)
- title: realistic product name (string)
- category: "{category}"
- subcategory: "{subcategory}"
- description: 2-3 sentence product description (string)
- tags: list of 4-6 relevant tags (array of strings)
- price_usd: realistic price in USD as a float
- avg_rating: float between 3.5 and 5.0
- num_reviews: integer between 50 and 50000
- brand: realistic brand name (string)
- features: list of 3-5 key product features (array of strings)

IDs to use: {ids}

Return exactly: {{"items": [...]}}
"""


def _map_item(raw: dict) -> dict:
    """Normalize LLM field names to match CatalogItem."""
    raw.setdefault("item_id", raw.pop("id", str(uuid.uuid4())[:8]))
    raw.setdefault("price_usd", raw.pop("price", 29.99))
    raw.setdefault("avg_rating", raw.pop("rating", 4.0))
    raw.setdefault("asin", None)
    raw.setdefault("compatible_with", [])
    return raw


def generate_catalog(n_items: int = 250) -> list[CatalogItem]:
    console.print(f"[bold blue]Generating catalog ({n_items} items)...[/]")
    items: list[CatalogItem] = []
    categories = CATEGORY_SEEDS * (n_items // len(CATEGORY_SEEDS) + 1)
    random.shuffle(categories)

    batch_size = 10
    batches, remaining, cat_idx = [], n_items, 0
    while remaining > 0:
        size = min(batch_size, remaining)
        batches.append((categories[cat_idx % len(CATEGORY_SEEDS)], size))
        remaining -= size
        cat_idx += 1

    for (category, subcategory), size in track(batches, description="Generating items"):
        ids = [str(uuid.uuid4())[:8] for _ in range(size)]
        prompt = ITEM_PROMPT_TEMPLATE.format(
            n=size, category=category, subcategory=subcategory, ids=json.dumps(ids)
        )
        try:
            data = _llm_json(prompt, ITEM_SYSTEM)
            for raw in data.get("items", []):
                raw = _map_item(raw)
                try:
                    items.append(CatalogItem.model_validate(raw))
                except Exception as e:
                    console.print(f"[yellow]Skipping item: {e}[/]")
        except Exception as exc:
            console.print(f"[red]Batch error: {exc}[/]")

    console.print(f"[green]✓ Generated {len(items)} catalog items[/]")
    return items


# ---------------------------------------------------------------------------
# Synthetic user generation
# ---------------------------------------------------------------------------

USER_SYSTEM = (
    "You are a synthetic user persona generator for a recommendation system. "
    "Always respond with valid JSON only, no markdown fences."
)

USER_PROMPT_TEMPLATE = """
Generate {n} realistic synthetic user personas for a product recommendation system.

Available product categories: {categories}

Each persona must be a JSON object inside a "users" array with these EXACT fields:
- user_id: use one of the provided IDs (string)
- name: realistic full name (string)
- age: integer between 20 and 65
- occupation: realistic job title (string)
- taste_description: 2-3 sentences describing shopping style and preferences
- preferred_categories: 2-4 categories they love (array of strings from available)
- disliked_categories: 1-2 categories they dislike (array of strings)
- price_sensitivity: one of "budget", "mid-range", "premium"
- purchase_history: list of 5-10 past purchases, each with:
    - item_id: random 8-char string
    - title: realistic product title (string)
    - rating_given: float 1.0-5.0
    - review_snippet: 1-sentence review (string)
- wishlist_keywords: 3-6 keywords describing what they want next (array of strings)

IDs to use: {ids}

Return exactly: {{"users": [...]}}
"""

_PRICE_MAP = {"low": "budget", "medium": "mid-range", "high": "premium"}


def _map_user(raw: dict) -> dict:
    """Normalize LLM field names to match SyntheticUser."""
    raw.setdefault("user_id", raw.pop("id", str(uuid.uuid4())[:8]))
    # wishlist_tags → wishlist_keywords
    if "wishlist_tags" in raw and "wishlist_keywords" not in raw:
        raw["wishlist_keywords"] = raw.pop("wishlist_tags")
    raw.setdefault("wishlist_keywords", [])
    # Normalize price_sensitivity
    ps = raw.get("price_sensitivity", "mid-range")
    raw["price_sensitivity"] = _PRICE_MAP.get(ps, ps)
    # Normalize purchase history entries
    ph = []
    for p in raw.get("purchase_history", []):
        p.setdefault("title", p.pop("item_title", "Unknown"))
        p.setdefault("item_id", str(uuid.uuid4())[:8])
        p.pop("category", None)  # not in PurchaseEvent model
        ph.append(p)
    raw["purchase_history"] = ph
    return raw


def generate_users(n_users: int = 12) -> list[SyntheticUser]:
    console.print(f"[bold blue]Generating {n_users} synthetic users...[/]")
    categories = list({c for c, _ in CATEGORY_SEEDS})
    ids = [str(uuid.uuid4())[:8] for _ in range(n_users)]
    prompt = USER_PROMPT_TEMPLATE.format(
        n=n_users, categories=json.dumps(categories), ids=json.dumps(ids)
    )
    data = _llm_json(prompt, USER_SYSTEM)
    users = []
    for raw in data.get("users", []):
        raw = _map_user(raw)
        try:
            users.append(SyntheticUser.model_validate(raw))
        except Exception as e:
            console.print(f"[yellow]Skipping user: {e}[/]")
    console.print(f"[green]✓ Generated {len(users)} users[/]")
    return users


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    items = generate_catalog(config.NUM_CATALOG_ITEMS)
    catalog_data = [item.model_dump() for item in items]
    config.CATALOG_PATH.write_text(json.dumps(catalog_data, indent=2))
    console.print(f"[green]Saved catalog → {config.CATALOG_PATH}[/]")

    users = generate_users(config.NUM_USERS)
    users_data = [u.model_dump() for u in users]
    config.USERS_PATH.write_text(json.dumps(users_data, indent=2))
    console.print(f"[green]Saved users   → {config.USERS_PATH}[/]")


app = main  # backwards-compat alias used by scripts/generate_data.py

if __name__ == "__main__":
    main()
