"""Shared data models.

Uses Pydantic v2 when available; falls back to plain dataclasses for
environments where Pydantic is not installed (e.g. CI without full deps).
"""
from __future__ import annotations

from typing import Any, List, Optional

try:
    from pydantic import BaseModel, Field, field_validator

    class CatalogItem(BaseModel):
        item_id: str
        title: str
        category: str
        subcategory: str
        description: str
        tags: List[str] = Field(default_factory=list)
        price_usd: float
        avg_rating: float
        num_reviews: int
        brand: str
        asin: Optional[str] = None
        features: List[str] = Field(default_factory=list)
        compatible_with: List[str] = Field(default_factory=list)

        @field_validator("avg_rating")
        @classmethod
        def _rating_range(cls, v: float) -> float:
            if not (1.0 <= v <= 5.0):
                raise ValueError(f"avg_rating must be 1.0–5.0, got {v}")
            return v

        @property
        def embedding_text(self) -> str:
            return " | ".join([
                self.title,
                self.description,
                f"Category: {self.category} / {self.subcategory}",
                f"Brand: {self.brand}",
                "Tags: " + ", ".join(self.tags),
                "Features: " + "; ".join(self.features),
            ])

    class PurchaseEvent(BaseModel):
        item_id: str
        title: str
        rating_given: Optional[float] = None
        review_snippet: Optional[str] = None

    class SyntheticUser(BaseModel):
        user_id: str
        name: str
        age: int
        occupation: str
        taste_description: str
        preferred_categories: List[str]
        disliked_categories: List[str] = Field(default_factory=list)
        price_sensitivity: str
        purchase_history: List[PurchaseEvent] = Field(default_factory=list)
        wishlist_keywords: List[str] = Field(default_factory=list)

        @property
        def profile_text(self) -> str:
            return " | ".join([
                f"Name: {self.name}, Age: {self.age}, Occupation: {self.occupation}",
                self.taste_description,
                "Prefers: " + ", ".join(self.preferred_categories),
                "Dislikes: " + ", ".join(self.disliked_categories),
                f"Price sensitivity: {self.price_sensitivity}",
                "Wishlist keywords: " + ", ".join(self.wishlist_keywords),
            ])

    class Recommendation(BaseModel):
        rank: int
        item_id: str
        title: str
        price_usd: float
        avg_rating: float
        explanation: str

    class AgentResponse(BaseModel):
        user_id: str
        user_name: str
        query: str
        recommendations: List[Recommendation]
        reasoning_summary: str

except ImportError:
    # ------------------------------------------------------------------ #
    # Lightweight fallback — no Pydantic required.                        #
    # Supports model_validate(dict) and model_dump() so tests can run.   #
    # ------------------------------------------------------------------ #
    from dataclasses import dataclass, field

    def _validate(cls, data: dict):
        """Recursively instantiate nested models."""
        obj = object.__new__(cls)
        hints = cls.__dataclass_fields__ if hasattr(cls, '__dataclass_fields__') else {}
        for k, v in data.items():
            setattr(obj, k, v)
        for fname, fdef in hints.items():
            if not hasattr(obj, fname):
                if fdef.default_factory is not None:  # type: ignore
                    setattr(obj, fname, fdef.default_factory())
                else:
                    setattr(obj, fname, fdef.default)
        return obj

    class _Base:
        @classmethod
        def model_validate(cls, data: dict):
            return _validate(cls, data)

        def model_dump(self) -> dict:
            return {k: v for k, v in self.__dict__.items()}

    @dataclass
    class PurchaseEvent(_Base):
        item_id: str = ""
        title: str = ""
        rating_given: Optional[float] = None
        review_snippet: Optional[str] = None

    @dataclass
    class CatalogItem(_Base):
        item_id: str = ""
        title: str = ""
        category: str = ""
        subcategory: str = ""
        description: str = ""
        tags: List[str] = field(default_factory=list)
        price_usd: float = 0.0
        avg_rating: float = 4.0
        num_reviews: int = 0
        brand: str = ""
        asin: Optional[str] = None
        features: List[str] = field(default_factory=list)
        compatible_with: List[str] = field(default_factory=list)

        @classmethod
        def model_validate(cls, data: dict):
            rating = float(data.get("avg_rating", 4.0))
            if not (1.0 <= rating <= 5.0):
                raise ValueError(f"avg_rating must be 1.0–5.0, got {rating}")
            return _validate(cls, data)

        @property
        def embedding_text(self) -> str:
            return " | ".join([
                self.title,
                self.description,
                f"Category: {self.category} / {self.subcategory}",
                f"Brand: {self.brand}",
                "Tags: " + ", ".join(self.tags),
                "Features: " + "; ".join(self.features),
            ])

    @dataclass
    class SyntheticUser(_Base):
        user_id: str = ""
        name: str = ""
        age: int = 0
        occupation: str = ""
        taste_description: str = ""
        preferred_categories: List[str] = field(default_factory=list)
        disliked_categories: List[str] = field(default_factory=list)
        price_sensitivity: str = "mid-range"
        purchase_history: List[Any] = field(default_factory=list)
        wishlist_keywords: List[str] = field(default_factory=list)

        @classmethod
        def model_validate(cls, data: dict):
            obj = _validate(cls, data)
            # Coerce purchase_history list of dicts → PurchaseEvent objects
            raw_ph = getattr(obj, "purchase_history", [])
            obj.purchase_history = [
                PurchaseEvent.model_validate(p) if isinstance(p, dict) else p
                for p in raw_ph
            ]
            return obj

        @property
        def profile_text(self) -> str:
            return " | ".join([
                f"Name: {self.name}, Age: {self.age}, Occupation: {self.occupation}",
                self.taste_description,
                "Prefers: " + ", ".join(self.preferred_categories),
                "Dislikes: " + ", ".join(self.disliked_categories),
                f"Price sensitivity: {self.price_sensitivity}",
                "Wishlist keywords: " + ", ".join(self.wishlist_keywords),
            ])

    @dataclass
    class Recommendation(_Base):
        rank: int = 0
        item_id: str = ""
        title: str = ""
        price_usd: float = 0.0
        avg_rating: float = 0.0
        explanation: str = ""

    @dataclass
    class AgentResponse(_Base):
        user_id: str = ""
        user_name: str = ""
        query: str = ""
        recommendations: List[Any] = field(default_factory=list)
        reasoning_summary: str = ""
