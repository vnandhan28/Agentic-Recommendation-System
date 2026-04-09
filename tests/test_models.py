"""Tests for Pydantic data models — stdlib unittest only."""
import unittest
from agentic_rs.models import CatalogItem, SyntheticUser, PurchaseEvent, Recommendation


def make_item(**kwargs) -> dict:
    base = {
        "item_id": "B0TEST12345",
        "title": "Test Headphones Pro",
        "category": "Electronics",
        "subcategory": "Wireless Headphones",
        "description": "Great sound quality for everyday use.",
        "tags": ["wireless", "audio"],
        "price_usd": 79.99,
        "avg_rating": 4.3,
        "num_reviews": 1200,
        "brand": "SoundBrand",
        "features": ["40hr battery", "ANC"],
        "compatible_with": ["iOS", "Android"],
    }
    base.update(kwargs)
    return base


def make_user(**kwargs) -> dict:
    base = {
        "user_id": "user_001",
        "name": "Alice Chen",
        "age": 32,
        "occupation": "Software Engineer",
        "taste_description": "Loves minimalist tech gadgets with great battery life.",
        "preferred_categories": ["Wireless Headphones", "Smart Home"],
        "price_sensitivity": "mid-range",
        "purchase_history": [
            {
                "item_id": "B0ABC123",
                "title": "Old Headphones",
                "rating_given": 4.0,
                "review_snippet": "Good but could be better.",
            }
        ],
        "wishlist_keywords": ["noise cancelling", "compact"],
    }
    base.update(kwargs)
    return base


class TestCatalogItem(unittest.TestCase):
    def test_valid_item(self):
        item = CatalogItem.model_validate(make_item())
        self.assertEqual(item.item_id, "B0TEST12345")
        self.assertAlmostEqual(item.price_usd, 79.99)

    def test_embedding_text_contains_title(self):
        item = CatalogItem.model_validate(make_item())
        self.assertIn("Test Headphones Pro", item.embedding_text)

    def test_embedding_text_contains_category(self):
        item = CatalogItem.model_validate(make_item())
        self.assertIn("Electronics", item.embedding_text)

    def test_rating_too_high_raises(self):
        with self.assertRaises(Exception):
            CatalogItem.model_validate(make_item(avg_rating=5.5))

    def test_rating_too_low_raises(self):
        with self.assertRaises(Exception):
            CatalogItem.model_validate(make_item(avg_rating=0.9))

    def test_optional_asin_defaults_none(self):
        item = CatalogItem.model_validate(make_item())
        self.assertIsNone(item.asin)

    def test_optional_asin_set(self):
        item = CatalogItem.model_validate(make_item(asin="B00XYZ123"))
        self.assertEqual(item.asin, "B00XYZ123")


class TestSyntheticUser(unittest.TestCase):
    def test_valid_user(self):
        user = SyntheticUser.model_validate(make_user())
        self.assertEqual(user.user_id, "user_001")
        self.assertEqual(user.name, "Alice Chen")

    def test_profile_text_contains_name(self):
        user = SyntheticUser.model_validate(make_user())
        self.assertIn("Alice Chen", user.profile_text)

    def test_profile_text_contains_preferences(self):
        user = SyntheticUser.model_validate(make_user())
        self.assertIn("Wireless Headphones", user.profile_text)

    def test_purchase_history_parsed(self):
        user = SyntheticUser.model_validate(make_user())
        self.assertEqual(len(user.purchase_history), 1)
        self.assertIsInstance(user.purchase_history[0], PurchaseEvent)

    def test_empty_disliked_categories(self):
        user = SyntheticUser.model_validate(make_user())
        self.assertEqual(user.disliked_categories, [])

    def test_price_sensitivity_values(self):
        for val in ("budget", "mid-range", "premium"):
            user = SyntheticUser.model_validate(make_user(price_sensitivity=val))
            self.assertEqual(user.price_sensitivity, val)


class TestRecommendation(unittest.TestCase):
    def test_valid_recommendation(self):
        rec = Recommendation.model_validate({
            "rank": 1,
            "item_id": "B0TEST12345",
            "title": "Test Headphones Pro",
            "price_usd": 79.99,
            "avg_rating": 4.3,
            "explanation": "Perfect for Alice's noise-cancelling needs.",
        })
        self.assertEqual(rec.rank, 1)
        self.assertEqual(rec.item_id, "B0TEST12345")


if __name__ == "__main__":
    unittest.main()
