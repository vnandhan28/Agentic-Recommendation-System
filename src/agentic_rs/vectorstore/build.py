"""
Build ChromaDB vector store from catalog.json and synthetic_users.json,
and expose a VectorStore class for querying it at runtime.

Run: python scripts/build_vectorstore.py
"""

from __future__ import annotations

import json
from typing import Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import track

from agentic_rs.config import config
from agentic_rs.models import CatalogItem, SyntheticUser, PurchaseEvent

console = Console()


# ---------------------------------------------------------------------------
# VectorStore — used at query time by the agent
# ---------------------------------------------------------------------------

class VectorStore:
    """Wraps ChromaDB + SentenceTransformer for item/user retrieval."""

    def __init__(self, persist_path=None):
        path = persist_path or config.CHROMA_PATH
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self._model: Optional[SentenceTransformer] = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            console.print(f"[blue]Loading embedding model: {config.EMBEDDING_MODEL}[/]")
            self._model = SentenceTransformer(config.EMBEDDING_MODEL)
        return self._model

    def _embed(self, text: str) -> list:
        return self._get_model().encode([text]).tolist()[0]

    def search_items(self, query: str, n_results: int = 8, where=None) -> list[dict]:
        try:
            collection = self._client.get_collection(config.ITEMS_COLLECTION)
        except Exception:
            return []
        count = collection.count()
        if count == 0:
            return []
        n = min(n_results, count)
        embedding = self._embed(query)
        kwargs: dict = dict(query_embeddings=[embedding], n_results=n)
        if where:
            kwargs["where"] = where
        results = collection.query(**kwargs)
        items = []
        for i, meta in enumerate(results["metadatas"][0]):
            item = dict(meta)
            item["similarity_score"] = 1.0 - results["distances"][0][i]
            items.append(item)
        return items

    def get_item_by_id(self, item_id: str) -> Optional[dict]:
        try:
            collection = self._client.get_collection(config.ITEMS_COLLECTION)
            result = collection.get(ids=[item_id])
            if result["metadatas"]:
                return dict(result["metadatas"][0])
        except Exception:
            pass
        return None

    def get_user_by_id(self, user_id: str) -> Optional[dict]:
        try:
            collection = self._client.get_collection(config.USERS_COLLECTION)
            result = collection.get(ids=[user_id])
            if result["metadatas"]:
                return dict(result["metadatas"][0])
        except Exception:
            pass
        return None

    def list_users(self) -> list[dict]:
        try:
            collection = self._client.get_collection(config.USERS_COLLECTION)
            result = collection.get()
            return [
                {"user_id": m.get("user_id"), "name": m.get("name")}
                for m in result["metadatas"]
            ]
        except Exception:
            return []

    def item_count(self) -> int:
        try:
            return self._client.get_collection(config.ITEMS_COLLECTION).count()
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.ClientAPI:
    config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(config.CHROMA_PATH),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_embedding_model() -> SentenceTransformer:
    console.print(f"[blue]Loading embedding model: {config.EMBEDDING_MODEL}[/]")
    return SentenceTransformer(config.EMBEDDING_MODEL)


def _safe_meta(d: dict) -> dict:
    """Replace None values with empty strings (ChromaDB requirement)."""
    return {k: ("" if v is None else v) for k, v in d.items()}


def build_items_collection(
    client: chromadb.ClientAPI,
    model: SentenceTransformer,
    items: list[CatalogItem],
) -> None:
    console.print(f"[blue]Indexing {len(items)} catalog items...[/]")
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
        batch = items[i: i + batch_size]
        texts = [item.embedding_text for item in batch]
        embeddings = model.encode(texts, show_progress_bar=False).tolist()
        metadatas: list[dict[str, Any]] = []
        for item in batch:
            d = item.model_dump()
            d["tags"] = json.dumps(d["tags"])
            d["features"] = json.dumps(d["features"])
            d["compatible_with"] = json.dumps(d["compatible_with"])
            metadatas.append(_safe_meta(d))

        collection.upsert(
            ids=[item.item_id for item in batch],
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
    console.print(f"[blue]Indexing {len(users)} user personas...[/]")
    try:
        client.delete_collection(config.USERS_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=config.USERS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [user.profile_text for user in users]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    metadatas: list[dict[str, Any]] = []
    for user in users:
        d = user.model_dump()
        d["preferred_categories"] = json.dumps(d["preferred_categories"])
        d["disliked_categories"] = json.dumps(d["disliked_categories"])
        d["wishlist_keywords"] = json.dumps(d["wishlist_keywords"])
        d["purchase_history"] = json.dumps([
            p if isinstance(p, dict) else p.model_dump()
            for p in d["purchase_history"]
        ])
        metadatas.append(_safe_meta(d))

    collection.upsert(
        ids=[u.user_id for u in users],
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    console.print(f"[green]✓ Users collection ready ({collection.count()} docs)[/]")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    if not config.CATALOG_PATH.exists():
        console.print(f"[red]Catalog not found: {config.CATALOG_PATH}[/]")
        console.print("Run `python scripts/generate_data.py` first.")
        return
    if not config.USERS_PATH.exists():
        console.print(f"[red]Users file not found: {config.USERS_PATH}[/]")
        console.print("Run `python scripts/generate_data.py` first.")
        return

    items_raw = json.loads(config.CATALOG_PATH.read_text())
    items = [CatalogItem.model_validate(i) for i in items_raw]

    users_raw = json.loads(config.USERS_PATH.read_text())
    users = []
    for u in users_raw:
        u["purchase_history"] = [
            PurchaseEvent.model_validate(p) if isinstance(p, dict) else p
            for p in u.get("purchase_history", [])
        ]
        users.append(SyntheticUser.model_validate(u))

    client = _get_chroma_client()
    model = _get_embedding_model()

    build_items_collection(client, model, items)
    build_users_collection(client, model, users)

    console.print("[bold green]Vector store build complete![/]")


app = main  # backwards-compat alias

if __name__ == "__main__":
    main()
