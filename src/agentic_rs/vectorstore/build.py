"""
Build ChromaDB vector store from catalog.json and synthetic_users.json.
Run: python -m agentic_rs.vectorstore.build
"""

from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

from agentic_rs import config
from agentic_rs.models import CatalogItem, SyntheticUser

console = Console()


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    config.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PERSIST_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_embedding_model() -> SentenceTransformer:
    console.print(f"[blue]Loading embedding model: {config.EMBEDDING_MODEL}[/]")
    return SentenceTransformer(config.EMBEDDING_MODEL)


def build_items_collection(
    client: chromadb.ClientAPI,
    model: SentenceTransformer,
    items: list[CatalogItem],
) -> None:
    """Embed all catalog items and upsert into ChromaDB."""
    console.print(f"[blue]Indexing {len(items)} catalog items...[/]")

    # Drop and recreate for a clean rebuild
    try:
        client.delete_collection(config.ITEMS_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.ITEMS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    batch_size = 64
    for i in track(range(0, len(items), batch_size), description="Embedding items"):
        batch = items[i : i + batch_size]
        texts = [item.embedding_text for item in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        metadatas: list[dict[str, Any]] = []
        for item in batch:
            d = item.model_dump()
            # ChromaDB metadata values must be str/int/float/bool
            d["tags"] = json.dumps(d["tags"])
            d["features"] = json.dumps(d["features"])
            metadatas.append(d)

        collection.upsert(
            ids=[item.id for item in batch],
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    console.print(f"[green]✓ Items collection ready ({collection.count()} docs)[/]")


def build_users_collection(
    client: chromadb.ClientAPI,
    model: SentenceTransformer,
    users: list[SyntheticUser],
) -> None:
    """Embed all user personas and upsert into ChromaDB."""
    console.print(f"[blue]Indexing {len(users)} user personas...[/]")

    try:
        client.delete_collection(config.USERS_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.USERS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [user.embedding_text for user in users]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    metadatas: list[dict[str, Any]] = []
    for user in users:
        d = user.model_dump()
        d["preferred_categories"] = json.dumps(d["preferred_categories"])
        d["disliked_categories"] = json.dumps(d["disliked_categories"])
        d["wishlist_tags"] = json.dumps(d["wishlist_tags"])
        d["purchase_history"] = json.dumps(d["purchase_history"])
        metadatas.append(d)

    collection.upsert(
        ids=[u.id for u in users],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    console.print(f"[green]✓ Users collection ready ({collection.count()} docs)[/]")


def main() -> None:
    if not config.CATALOG_PATH.exists():
        console.print(f"[red]Catalog not found: {config.CATALOG_PATH}[/]")
        console.print("Run `agentic-rs-generate` first.")
        return
    if not config.USERS_PATH.exists():
        console.print(f"[red]Users file not found: {config.USERS_PATH}[/]")
        console.print("Run `agentic-rs-generate` first.")
        return

    items_raw = json.loads(config.CATALOG_PATH.read_text())
    items = [CatalogItem(**i) for i in items_raw]

    users_raw = json.loads(config.USERS_PATH.read_text())
    users = []
    for u in users_raw:
        from agentic_rs.models import PurchaseEvent
        u["purchase_history"] = [PurchaseEvent(**p) for p in u.get("purchase_history", [])]
        users.append(SyntheticUser(**u))

    client = get_chroma_client()
    model = get_embedding_model()

    build_items_collection(client, model, items)
    build_users_collection(client, model, users)

    console.print("[bold green]Vector store build complete![/]")


if __name__ == "__main__":
    main()
