#!/usr/bin/env python3
"""
Ingest Amazon product metadata from HuggingFace (McAuley-Lab/Amazon-Reviews-2023)
into the ChromaDB items collection.

Pulls up to PER_CATEGORY products from each category, capped at TOTAL_LIMIT overall.
Merges with existing ChromaDB collection (upsert — won't duplicate).

Usage:
    python scripts/ingest_amazon_products.py
    python scripts/ingest_amazon_products.py --per-category 200 --total 5000
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console
from rich.progress import track

console = Console()

# HuggingFace dataset category names for McAuley-Lab/Amazon-Reviews-2023 (meta_* splits)
AMAZON_CATEGORIES = [
    "All_Beauty",
    "Amazon_Fashion",
    "Appliances",
    "Arts_Crafts_and_Sewing",
    "Automotive",
    "Baby_Products",
    "Beauty_and_Personal_Care",
    "Books",
    "CDs_and_Vinyl",
    "Cell_Phones_and_Accessories",
    "Clothing_Shoes_and_Jewelry",
    "Digital_Music",
    "Electronics",
    "Gift_Cards",
    "Grocery_and_Gourmet_Food",
    "Handmade_Products",
    "Health_and_Household",
    "Health_and_Personal_Care",
    "Home_and_Kitchen",
    "Industrial_and_Scientific",
    "Kindle_Store",
    "Magazine_Subscriptions",
    "Movies_and_TV",
    "Musical_Instruments",
    "Office_Products",
    "Patio_Lawn_and_Garden",
    "Pet_Supplies",
    "Software",
    "Sports_and_Outdoors",
    "Subscription_Boxes",
    "Tools_and_Home_Improvement",
    "Toys_and_Games",
    "Video_Games",
    "Unknown",
]


def _clean_str(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return ", ".join(str(v) for v in val if v)
    return str(val).strip()


def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _map_to_catalog_item(row: dict, category_label: str) -> dict | None:
    title = _clean_str(row.get("title"))
    if not title:
        return None

    avg_rating = _safe_float(row.get("average_rating") or row.get("avg_rating"), 0.0)
    if avg_rating == 0.0:
        avg_rating = 4.0
    avg_rating = max(1.0, min(5.0, avg_rating))

    price_raw = row.get("price")
    price = _safe_float(price_raw, 0.0)

    description_parts = row.get("description") or []
    if isinstance(description_parts, list):
        description = " ".join(description_parts).strip()
    else:
        description = _clean_str(description_parts)

    features_raw = row.get("features") or []
    features = [_clean_str(f) for f in features_raw if f][:10]

    details = row.get("details") or {}
    brand = _clean_str(
        row.get("brand") or (details.get("Brand") if isinstance(details, dict) else None)
    )

    asin = _clean_str(row.get("parent_asin") or row.get("asin"))

    # Generate a stable item_id from asin or title hash
    if asin:
        item_id = f"amz_{asin}"
    else:
        item_id = f"amz_{uuid.uuid5(uuid.NAMESPACE_DNS, title).hex[:8]}"

    human_category = category_label.replace("_", " ")

    return {
        "item_id": item_id,
        "title": title,
        "category": human_category,
        "subcategory": human_category,
        "description": description or title,
        "tags": [],
        "price_usd": price,
        "avg_rating": avg_rating,
        "num_reviews": _safe_int(row.get("rating_number")),
        "brand": brand,
        "asin": asin,
        "features": features,
        "compatible_with": [],
    }


def _embedding_text(item: dict) -> str:
    parts = [
        item["title"],
        item["description"],
        f"Category: {item['category']}",
        f"Brand: {item['brand']}",
        "Features: " + "; ".join(item["features"]),
    ]
    return " | ".join(p for p in parts if p.strip())


def ingest(per_category: int = 200, total_limit: int = 5000) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        console.print("[red]datasets library not found. Run: pip install datasets[/]")
        sys.exit(1)

    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
    from agentic_rs.config import config

    console.print(f"[blue]Loading embedding model: {config.EMBEDDING_MODEL}[/]")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(config.ITEMS_COLLECTION)
        console.print(f"[yellow]Found existing collection with {collection.count()} items — will upsert[/]")
    except Exception:
        collection = client.create_collection(
            name=config.ITEMS_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        console.print("[yellow]Created new items collection[/]")

    total_ingested = 0

    for category in AMAZON_CATEGORIES:
        if total_ingested >= total_limit:
            break

        remaining = total_limit - total_ingested
        take = min(per_category, remaining)

        console.print(f"[cyan]Fetching {take} from {category}...[/]")
        try:
            ds = load_dataset(
                "McAuley-Lab/Amazon-Reviews-2023",
                f"raw_meta_{category}",
                split="full",
                trust_remote_code=True,
                streaming=True,
            )
        except Exception as e:
            console.print(f"[yellow]  Skipping {category}: {e}[/]")
            continue

        items: list[dict] = []
        for row in ds:
            if len(items) >= take:
                break
            item = _map_to_catalog_item(row, category)
            if item:
                items.append(item)

        if not items:
            console.print(f"[yellow]  No valid items in {category}, skipping[/]")
            continue

        batch_size = 64
        for i in track(range(0, len(items), batch_size), description=f"  Embedding {category}"):
            batch = items[i: i + batch_size]
            texts = [_embedding_text(it) for it in batch]
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            metadatas = []
            for it in batch:
                d = dict(it)
                d["tags"] = json.dumps(d["tags"])
                d["features"] = json.dumps(d["features"])
                d["compatible_with"] = json.dumps(d["compatible_with"])
                metadatas.append({k: ("" if v is None else v) for k, v in d.items()})

            collection.upsert(
                ids=[it["item_id"] for it in batch],
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

        total_ingested += len(items)
        console.print(f"[green]  ✓ {len(items)} items ingested (total: {total_ingested}/{total_limit})[/]")

    console.print(f"\n[bold green]Done! {total_ingested} Amazon products added to ChromaDB.[/]")
    console.print(f"[blue]Total items in collection: {collection.count()}[/]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-category", type=int, default=200,
                        help="Max products per Amazon category (default: 200)")
    parser.add_argument("--total", type=int, default=5000,
                        help="Total product limit across all categories (default: 5000)")
    args = parser.parse_args()
    ingest(per_category=args.per_category, total_limit=args.total)
