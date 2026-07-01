"""Regression test: the briefing loop must not brief the same fixture twice.

Two data sources created separate Match rows for USA vs Bosnia (2026-07-02) —
API-Football "USA"/"Bosnia & Herzegovina" (with odds) and football-data.org
"United States"/"Bosnia-Herzegovina" (no odds). Both got a briefing. The dedup
in MatchBriefingService._wc_fixtures_between must collapse them to the odds-
bearing row.
"""

from datetime import datetime
from types import SimpleNamespace

import pytest

from src.data.database import DatabaseManager
from src.data.models import Team, Match, Odds
from src.reporting.match_briefing import MatchBriefingService
from src.utils.config import get_config


@pytest.fixture
def db(tmp_path, monkeypatch):
    # Defence in depth: conftest already strips DATABASE_URL, but make doubly
    # sure this DB-writing test can never reach production Postgres.
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "dedup_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _seed(db):
    """Two rows for the same fixture (name variants) + one unrelated fixture."""
    kickoff = datetime(2026, 7, 2, 0, 0, 0)
    with db.get_session() as s:
        usa = Team(name="USA")
        united_states = Team(name="United States")
        bih_amp = Team(name="Bosnia & Herzegovina")
        bih_hyphen = Team(name="Bosnia-Herzegovina")
        eng = Team(name="England")
        cod = Team(name="Congo DR")
        s.add_all([usa, united_states, bih_amp, bih_hyphen, eng, cod])
        s.flush()

        af_row = Match(home_team_id=usa.id, away_team_id=bih_amp.id,
                       match_date=kickoff, league="world/fifa-world-cup",
                       is_fixture=True, apifootball_id=1562586)
        fdo_row = Match(home_team_id=united_states.id, away_team_id=bih_hyphen.id,
                        match_date=kickoff, league="world/fifa-world-cup",
                        is_fixture=True, apifootball_id=None)
        other = Match(home_team_id=eng.id, away_team_id=cod.id,
                      match_date=kickoff, league="world/fifa-world-cup",
                      is_fixture=True, apifootball_id=1567307)
        s.add_all([af_row, fdo_row, other])
        s.flush()

        # Only the API-Football row has real odds.
        s.add(Odds(match_id=af_row.id, market_type="1X2", selection="Home",
                   odds_value=1.51, bookmaker="test"))
        s.commit()
        return af_row.id, fdo_row.id, other.id


def test_briefing_dedup_keeps_odds_bearing_row(db):
    af_id, fdo_id, other_id = _seed(db)
    svc = MatchBriefingService(SimpleNamespace(
        db=db, config=get_config(), telegram=None, agent=None))

    ids = svc._wc_fixtures_between(
        datetime(2026, 7, 1, 0, 0, 0), datetime(2026, 7, 3, 0, 0, 0))

    assert af_id in ids, "the odds-bearing row must be briefed"
    assert fdo_id not in ids, "the duplicate row must be dropped"
    assert other_id in ids, "unrelated fixtures must be unaffected"
    assert len(ids) == 2


def test_no_dedup_when_teams_differ(db):
    """Distinct fixtures on the same day must both survive."""
    kickoff = datetime(2026, 7, 2, 0, 0, 0)
    with db.get_session() as s:
        a, b, c, d = (Team(name="Spain"), Team(name="Austria"),
                      Team(name="Portugal"), Team(name="Croatia"))
        s.add_all([a, b, c, d]); s.flush()
        m1 = Match(home_team_id=a.id, away_team_id=b.id, match_date=kickoff,
                   league="world/fifa-world-cup", is_fixture=True, apifootball_id=1)
        m2 = Match(home_team_id=c.id, away_team_id=d.id, match_date=kickoff,
                   league="world/fifa-world-cup", is_fixture=True, apifootball_id=2)
        s.add_all([m1, m2]); s.flush()
        ids_expected = {m1.id, m2.id}
        s.commit()

    svc = MatchBriefingService(SimpleNamespace(
        db=db, config=get_config(), telegram=None, agent=None))
    ids = svc._wc_fixtures_between(
        datetime(2026, 7, 1, 0, 0, 0), datetime(2026, 7, 3, 0, 0, 0))
    assert set(ids) == ids_expected
