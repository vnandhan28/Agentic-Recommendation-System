#!/usr/bin/env python3
"""
CLI script: Embed catalog + users and persist to ChromaDB.

Usage:
    python scripts/build_vectorstore.py
    python scripts/build_vectorstore.py --catalog data/catalog.json --chroma data/chroma_db
"""
from agentic_rs.vectorstore.build import app

if __name__ == "__main__":
    app()
