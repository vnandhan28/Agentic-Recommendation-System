"""Central configuration — loads from .env / environment variables."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ROOT = Path(__file__).parent.parent.parent
    load_dotenv(_ROOT / ".env", override=False)
except ImportError:
    pass  # dotenv optional


class _Config:
    LLM_PROVIDER: str         = os.getenv("LLM_PROVIDER", "groq")
    LLM_MODEL: str            = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    EMBEDDING_MODEL: str      = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    DATA_DIR: Path            = Path(os.getenv("DATA_DIR", "data"))
    CHROMA_PATH: Path         = Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
    CHROMA_PERSIST_DIR: Path  = Path(os.getenv("CHROMA_PATH", "data/chroma_db"))
    CATALOG_PATH: Path        = Path(os.getenv("CATALOG_PATH", "data/catalog.json"))
    USERS_PATH: Path          = Path(os.getenv("USERS_PATH", "data/synthetic_users.json"))
    ITEMS_COLLECTION: str     = os.getenv("ITEMS_COLLECTION", "items")
    USERS_COLLECTION: str     = os.getenv("USERS_COLLECTION", "users")
    NUM_CATALOG_ITEMS: int    = int(os.getenv("NUM_CATALOG_ITEMS", "250"))
    NUM_USERS: int            = int(os.getenv("NUM_USERS", "12"))
    CATALOG_CATEGORY: str     = os.getenv("CATALOG_CATEGORY", "Electronics")
    MAX_AGENT_ITERATIONS: int = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
    NUM_RECOMMENDATIONS: int  = int(os.getenv("NUM_RECOMMENDATIONS", "5"))

    @property
    def api_key(self) -> str:
        if self.LLM_PROVIDER == "anthropic":
            key = "ANTHROPIC_API_KEY"
        elif self.LLM_PROVIDER == "huggingface":
            key = "HF_API_KEY"
        elif self.LLM_PROVIDER == "groq":
            key = "GROQ_API_KEY"
        else:
            key = "OPENAI_API_KEY"
        val = os.getenv(key)
        if not val:
            raise EnvironmentError(f"Required env var '{key}' is not set.")
        return val


config = _Config()
