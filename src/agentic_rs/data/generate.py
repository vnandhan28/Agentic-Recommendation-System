"""
Generate catalog.json (200–300 items) and synthetic_users.json (10–15 personas)
using an LLM. Run directly:  python -m agentic_rs.data.generate
"""

from __future__ import annotations

import json
import random
import uuid
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.progress import track

from agentic_rs import config
from agentic_rs.models import CatalogItem, PurchaseEvent, SyntheticUser

console = Console()
client = OpenAI(api_key=config.OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Category seeds (inspired by Amazon Reviews 2023 product categories)
# ---------------------------------------------------------------------------
CATEGORY_SEEDS = [
    ("Electronics", "Headphones"),
    ("Electronics", "Keyboards"),
    ("Electronics", "Monitors"),
    ("Electronics", "Smart Home"),
    ("Electronics", "Cameras"),
    ("Books", "Science Fiction"),
    ("Books", "Self-Help"),
    ("Books", "History"),
    ("Books", "Technology"),
    ("Sports & Outdoors", "Running"),
    ("Sports & Outdoors", "Yoga"),
    ("Sports & Outdoors", "Camping"),
    ("Home & Kitchen", "Coffee Makers"),
    ("Home & Kitchen", "Cookware"),
    ("Home & Kitchen", "Storage"),
    ("Beauty & Personal Care", "Skincare"),
    ("Beauty & Personal Care", "Hair Care"),
    ("Toys & Games", "Board Games"),
    ("Toys & Games", "LEGO"),
    ("Clothing", "Sneakers"),
]


def _llm_json(prompt: str, system: str = "") -> dict | list:
    """Call the LLM and parse the JSON response."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=messages,
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# Catalog generation
# ---------------------------------------------------------------------------
ITEM_SYSTEM = (
    "You are a product catalog generator. "
    "Always respond with valid JSON only, no markdown fences."
)

ITEM_PROMPT_TEMPLATE = """
Generate {n} realistic product catalog entries for the category: {category} > {subcategory}.

Each entry must be a JSON object inside an "items" array with these exact fields:
- id: unique string (use provided ids)
- title: realistic product name (string)
- category: "{category}"
- subcategory: "{subcategory}"
- description: 2-3 sentence product description (string)
- tags: list of 4-6 relevant tags (array of strings)
- price: realistic price in USD as a float
- rating: float between 3.5 and 5.0
- num_reviews: integer between 50 and 50000
- brand: realistic brand name (string)
- features: list of 3-5 key product features (array of strings)

IDs to use: {ids}

Return exactly: {{"items": [...]}}
"""


def generate_catalog(n_items: int = 250) -> list[CatalogItem]:
    """Generate n_items catalog entries across the category seeds."""
    console.print(f"[bold blue]Generating catalog ({n_items} items)...[/]")
    items: list[CatalogItem] = []
    categories = CATEGORY_SEEDS * (n_items // len(CATEGORY_SEEDS) + 1)
    random.shuffle(categories)

    batch_size = 10
    batches = []
    remaining = n_items
    cat_idx = 0
    while remaining > 0:
        size = min(batch_size, remaining)
        batches.append((categories[cat_idx % len(CATEGORY_SEEDS)], size))
        remaining -= size
        cat_idx += 1

    for (category, subcategory), size in track(batches, description="Generating batches"):
        ids = [str(uuid.uuid4())[:8] for _ in range(size)]
        prompt = ITEM_PROMPT_TEMPLATE.format(
            n=size,
            category=category,
            subcategory=subcategory,
            ids=json.dumps(ids),
        )
        try:
            data = _llm_json(prompt, ITEM_SYSTEM)
            for raw in data.get("items", []):
                raw.setdefault("id", str(uuid.uuid4())[:8])
                items.append(CatalogItem(**raw))
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

Each persona must be a JSON object inside a "users" array with these exact fields:
- id: unique string (use provided ids)
- name: realistic full name (string)
- age: integer between 20 and 65
- occupation: realistic job title (string)
- taste_description: 2-3 sentences describing shopping style and preferences
- preferred_categories: 2-4 categories they love (array of strings from available)
- disliked_categories: 1-2 categories they dislike (array of strings)
- price_sensitivity: one of "low", "medium", "high"
- purchase_history: list of 5-10 past purchases, each with:
    - item_id: random 8-char string
    - item_title: realistic product title
    - category: one of the available categories
    - rating_given: float 1.0-5.0
    - review_snippet: 1-sentence review (string)
- wishlist_tags: 3-6 tags describing what they want next (array of strings)

IDs to use: {ids}

Return exactly: {{"users": [...]}}
"""


def generate_users(n_users: int = 15) -> list[SyntheticUser]:
    """Generate n_users synthetic user personas."""
    console.print(f"[bold blue]Generating {n_users} synthetic users...[/]")
    categories = list({c for c, _ in CATEGORY_SEEDS})
    ids = [str(uuid.uuid4())[:8] for _ in range(n_users)]

    prompt = USER_PROMPT_TEMPLATE.format(
        n=n_users,
        categories=json.dumps(categories),
        ids=json.dumps(ids),
    )
    data = _llm_json(prompt, USER_SYSTEM)
    users = []
    for raw in data.get("users", []):
        raw.setdefault("id", str(uuid.uuid4())[:8])
        # Convert purchase_history dicts → PurchaseEvent objects
        raw["purchase_history"] = [
            PurchaseEvent(**p) for p in raw.get("purchase_history", [])
        ]
        users.append(SyntheticUser(**raw))

    console.print(f"[green]✓ Generated {len(users)} users[/]")
    return users


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Generate catalog
    items = generate_catalog(config.NUM_CATALOG_ITEMS)
    catalog_data = [item.model_dump() for item in items]
    config.CATALOG_PATH.write_text(json.dumps(catalog_data, indent=2))
    console.print(f"[green]Saved catalog → {config.CATALOG_PATH}[/]")

    # Generate users
    users = generate_users(config.NUM_USERS)
    users_data = [u.model_dump() for u in users]
    config.USERS_PATH.write_text(json.dumps(users_data, indent=2))
    console.print(f"[green]Saved users   → {config.USERS_PATH}[/]")


if __name__ == "__main__":
    main()
