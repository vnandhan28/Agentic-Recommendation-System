#!/usr/bin/env python3
"""
CLI script: Generate synthetic catalog + user data.

Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --items 200 --users 10 --category "Books"
"""
from agentic_rs.data.generate import app

if __name__ == "__main__":
    app()
