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


# SAFETY, the filesystem twin of the DATABASE_URL strip above. This one is
# recorded because it HAPPENED, on 2026-09-10, in the commit that added
# tests/test_odds_credit_gate.py.
#
# `_absorb_quota_headers` calls `_persist_credits`, which writes the provider's
# remaining-credit count to data/models/theodds_credits.json. The new tests feed
# it fabricated header values, so running them overwrote that file with a made-up
# number — and it was committed. The real reading at the time was 154; the file
# shipped saying 100.
#
# That file is not decorative. `TheOddsScraper.update()` HARD-SKIPS the whole
# odds fetch when it reads <= _CREDITS_GATE_THRESHOLD (10):
#
#     "TheOddsAPI: skipping update — only N credits remain"
#
# and two of those tests write remaining=0. A different test ORDER would have
# left 0 on disk and silently disabled pick-time odds fetching on the next
# production run — the exact silent-degradation class this suite exists to
# catch, introduced by the suite.
#
# Redirected for EVERY test rather than stubbed in the one file that noticed,
# because the next author to touch a header-parsing path will not remember.
import pathlib as _pathlib

import pytest as _pytest


@_pytest.fixture(autouse=True)
def _no_test_writes_to_production_credit_state(tmp_path, monkeypatch):
    """No test may write the production credit-state file."""
    try:
        import src.scrapers.theodds_scraper as _tos
    except Exception:
        return
    monkeypatch.setattr(
        _tos, "_CREDITS_STATE_PATH",
        _pathlib.Path(tmp_path) / "theodds_credits.json", raising=False)
