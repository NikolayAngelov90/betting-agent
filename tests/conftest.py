"""Shared pytest fixtures and environment setup."""

import os


# ─────────────── THE `calendar` MARKER (Stage 28, 2026-10-09) ───────────────
#
# A test whose PASS/FAIL CAN CHANGE WITH NO CODE CHANGE does not belong in the
# path that gates price capture. On 2026-10-09 two such tests failed — the
# empty-card suppression's expiry, firing exactly as designed, and three quota
# tests that inherited the allocator's first active day — and together they
# stopped `Run tests`, which stopped `--update`, which lost the first real card
# in nineteen days until it was recovered by hand with 3h01m to spare.
#
# The gate decision is UNCHANGED: capture on an untested tree is still the worse
# risk, because wrong odds rows never backfill. What moves is where a CALENDAR
# assertion lives.
#
# MEMBERSHIP IS DECLARED, NEVER INFERRED. A grep cannot find these: the three
# quota tests that broke contain no clock read at all — `remaining()` reads it.
# So a member carries the marker, and `tests/test_calendar_separation.py`
# asserts the split holds in both directions.
#
# A member asserts a DEPLOYMENT FACT — "this guard has expired, decide" — not a
# correctness property. A correctness test that happens to use a date stays in
# the gating suite.
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "calendar: a test whose outcome depends on today's date rather than on "
        "the code — it asserts a deployment fact and must NOT gate price "
        "capture. Run separately; a failure reds the run without blocking "
        "--update.")

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


# SAFETY, the filesystem twin of the DATABASE_URL strip above — and it covers
# the CLASS, not the one file that was caught.
#
# THE INCIDENT. On 2026-09-10 `tests/test_odds_credit_gate.py` fed fabricated
# quota headers to `_absorb_quota_headers`, which calls `_persist_credits`,
# which writes data/models/theodds_credits.json. The suite overwrote that file
# with a made-up number and `git add -A` committed it. The true reading was 154;
# the repo shipped 100. `TheOddsScraper.update()` HARD-SKIPS the whole odds
# fetch when that file reads <= 10, and two of those tests write 0 — a different
# test ORDER would have silently disabled pick-time odds fetching in production.
#
# THE RULE, which is not file-specific:
#
#     A TEST MUST NOT WRITE TO ANY PATH PRODUCTION READS.
#
# So the paths are enumerated here ONCE rather than discovered one at a time.
# Every module-level `Path("data/...")` / `Path("config/...")` constant in src/
# is redirected into tmp_path for every test. `tests/test_no_test_writes_prod_state.py`
# fails if a new such constant appears in src/ and is not listed here, so the
# class cannot quietly regrow — the enumeration is enforced, not maintained.
#
# Verified by positive control, per path: disable the redirect and the file is
# clobbered; enable it and the md5 is identical across a full run.
import pathlib as _pathlib

import pytest as _pytest

#: (module, attribute, kind). `kind` is "dir" for directory roots, "file"
#: otherwise — a directory must exist before production writes into it.
PRODUCTION_STATE_PATHS = (
    ("src.data.history_mirror", "_DEFAULT_DIR", "dir"),
    ("src.models.ml_models", "MODELS_DIR", "dir"),
    ("src.models.bayesian_weights", "WEIGHTS_PATH", "file"),
    ("src.models.probability_calibration", "DEFAULT_PATH", "file"),
    ("src.reporting.match_briefing", "_SENT_PATH", "file"),
    ("src.reporting.telegram_bot", "_PICKS_SENT_STATE", "file"),
    ("src.reporting.telegram_bot", "_COLD_STREAK_STATE", "file"),
    ("src.scrapers.barren_leagues", "DEFAULT_PATH", "file"),
    ("src.scrapers.historical_loader", "CACHE_FILE", "file"),
    ("src.scrapers.theodds_scraper", "_CREDITS_STATE_PATH", "file"),
)


@_pytest.fixture(autouse=True)
def _no_test_writes_to_production_state(tmp_path, monkeypatch):
    """No test may write any path production reads."""
    import importlib

    root = _pathlib.Path(tmp_path) / "prodstate"
    root.mkdir(parents=True, exist_ok=True)
    for mod_name, attr, kind in PRODUCTION_STATE_PATHS:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            # A module that cannot even import cannot write anything either.
            continue
        current = getattr(mod, attr, None)
        if current is None:
            continue
        target = root / mod_name.replace(".", "_") / attr
        if kind == "dir":
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target = target.with_suffix(_pathlib.Path(str(current)).suffix)
        monkeypatch.setattr(mod, attr, target, raising=False)
