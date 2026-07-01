"""Shared pytest fixtures and environment setup."""

import os

# Opt into the insecure default HMAC key for tests. Production runs require
# MODEL_HMAC_KEY to be set or this env var to be set explicitly — see
# src/models/ml_models.py for the rationale.
os.environ.setdefault("BETTING_AGENT_ALLOW_DEFAULT_HMAC", "1")

# SAFETY: tests must NEVER touch a real database. CI (and local dev) set
# DATABASE_URL to the production Postgres, and DatabaseManager._create_engine
# prefers env DATABASE_URL over any sqlite_path passed in config — so a DB-backed
# test would read AND WRITE production. This actually happened: test_briefing_dedup
# seeded fixtures straight into the live Supabase DB during a CI run, which then
# re-briefed those rows. Remove the var at import (before any test constructs a
# DatabaseManager) so every test falls back to a local/temp SQLite database.
os.environ.pop("DATABASE_URL", None)
