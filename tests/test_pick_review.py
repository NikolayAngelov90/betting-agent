"""Tests for the 'predictions only' Claude pick-review path.

Claude reviews every saved pick and may CHANGE the selection; the in-memory pick
objects must be synced from the DB so the Telegram summary reflects the switch.
No briefing article is posted in this mode.
"""

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.data.database import DatabaseManager
from src.data.models import Team, Match, SavedPick
from src.reporting.match_briefing import MatchBriefingService
from src.utils.config import get_config


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "review_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _rec(match_id, **kw):
    base = dict(match_id=match_id, market="over_under", selection="Over 2.5 Goals",
                odds=2.10, predicted_probability=0.51, expected_value=0.05,
                confidence=0.51, kelly_stake_percentage=1.5, risk_level="medium")
    base.update(kw)
    return SimpleNamespace(**base)


def _svc(db):
    return MatchBriefingService(SimpleNamespace(
        db=db, config=get_config(), telegram=None, agent=None))


def test_sync_recs_reflects_switched_pick(db):
    """After Claude switches the saved pick, the in-memory rec must show it."""
    with db.get_session() as s:
        h, a = Team(name="Djurgarden"), Team(name="Hacken")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 6, 18, 0), league="sweden/allsvenskan",
                  is_fixture=True)
        s.add(m); s.flush()
        # DB pick is "Home Over 1.5" @1.55 (what Claude switched it to).
        s.add(SavedPick(match_id=m.id, pick_date=date.today(),
                        match_name="Djurgarden vs Hacken", league="sweden/allsvenskan",
                        market="team_goals", selection="Home Over 1.5", odds=1.55,
                        predicted_probability=0.6, expected_value=0.08, confidence=0.6,
                        kelly_stake_percentage=2.0, risk_level="low"))
        s.commit()
        mid = m.id

    # rec still holds the pre-switch selection.
    rec = _rec(mid, market="over_under", selection="Over 2.5 Goals", odds=2.10)
    _svc(db)._sync_recs_from_db(mid, [rec])

    assert rec.selection == "Home Over 1.5"
    assert rec.odds == 1.55
    assert rec.market == "team_goals"
    assert rec.expected_value == 0.08


def test_sync_recs_no_saved_pick_leaves_rec_untouched(db):
    rec = _rec(999, selection="Over 2.5 Goals")
    _svc(db)._sync_recs_from_db(999, [rec])
    assert rec.selection == "Over 2.5 Goals"  # unchanged when no DB pick exists


def test_finalize_noop_without_claude_auth(db, monkeypatch):
    """With no Claude auth, review must no-op and leave picks untouched."""
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    rec = _rec(1, selection="Over 2.5 Goals")
    reviewed = asyncio.run(_svc(db).finalize_picks_with_claude([rec]))
    assert reviewed == 0
    assert rec.selection == "Over 2.5 Goals"


def test_finalize_empty_picks(db):
    assert asyncio.run(_svc(db).finalize_picks_with_claude([])) == 0
