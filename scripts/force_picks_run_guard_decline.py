"""FORCE THE DECLINE BRANCH — the credit-gate-first-refusal practice, applied.

Test-scoped SQLite ledger. No production data, no API key that resolves, no
credit: dry_run=True returns before _fetch_and_persist, and the guard is
evaluated BEFORE that check.
"""
import asyncio, pathlib, re, tempfile
from datetime import datetime, timedelta
from src.utils.logger import setup_logger, get_logger

tmp = pathlib.Path(tempfile.mkdtemp())
setup_logger(log_level="INFO", log_file=str(tmp / "run.log"))   # production's own level

import src.data.database as db_mod
import src.scrapers.theodds_scraper as ts
from src.data.models import Base, Match, SavedPick, Team

NOW = datetime(2026, 9, 19, 10, 47)      # inside the exposure window after 03:00

mgr = db_mod.DatabaseManager(
    config=type("C", (), {"database": {"sqlite_path": str(tmp / "t.db")}})())
Base.metadata.create_all(mgr.engine)

A, B = "england/premier-league", "spain/laliga"
with mgr.get_session() as s:
    s.add_all([Team(id=1, name="Home"), Team(id=2, name="Away")])
    s.commit()
    # League A — a PENDING pick (makes it a candidate) AND an unpicked future
    # fixture (the exposure the guard exists for).
    s.add(Match(id=10, league=A, home_team_id=1, away_team_id=2, is_fixture=True,
                match_date=NOW + timedelta(minutes=60)))
    s.flush()
    s.add(SavedPick(match_id=10, pick_date=(NOW - timedelta(days=1)).date()))
    s.add(Match(id=11, league=A, home_team_id=1, away_team_id=2, is_fixture=True,
                match_date=NOW + timedelta(hours=9)))          # UNPICKED
    # League B — a pending pick and NOTHING left to price.
    s.add(Match(id=20, league=B, home_team_id=1, away_team_id=2, is_fixture=True,
                match_date=NOW + timedelta(minutes=90)))
    s.flush()
    s.add(SavedPick(match_id=20, pick_date=(NOW - timedelta(days=1)).date()))
    s.commit()

sc = ts.TheOddsScraper.__new__(ts.TheOddsScraper)
sc.db = mgr
# `fake-` prefix deliberately: test_no_secrets_in_repo's PLACEHOLDER accepts
# fake.*/dummy/example/test_key and nothing else, and "forced-…" was none of
# them. That literal failed CI for a full day of picks on 2026-09-20.
sc.api_key = "fake-key-never-used-dry-run-only"
sc._remaining_requests = None
sc._used_requests = None
sc._last_league_outcomes = {}

plan = asyncio.run(sc.refresh_imminent(window_minutes=120, min_interval_minutes=0,
                                       quota=None, dry_run=True, now=NOW))
log = (tmp / "run.log").read_text(encoding="utf-8")

import scripts.ci_audit as ci
declines = [l for l in log.splitlines() if "PICKS-RUN GUARD" in l]
print("1. DID THE DECLINE FIRE?           ", "YES" if declines else "NO")
print("2. DOES IT NAME THE LEAGUE?        ",
      "YES — " + A if any(A in l for l in declines) else "NO")
parsed = ci.extract("\n".join(declines))
print("3. DOES ci_audit READ IT?          ",
      f"YES — picks_run_guard_declined={parsed.get('picks_run_guard_declined')}"
      if parsed.get("picks_run_guard_declined") else "NO")
print("4. DID THE RUN CONTINUE?           ",
      f"YES — requested={plan['requested']}" if plan["requested"] else "NO — everything declined")
print()
print("   skipped:", {k: v[:60] for k, v in plan["skipped"].items()})
for l in declines:
    print("   line:", l.split(" - ", 1)[-1].strip()[:150])
