"""Tests for football-data.org extra-time/penalty score handling.

FDO v4 `fullTime` is NOT the 90-minute score: it includes extra time, and for
shootout matches even the shootout goals (observed live: Argentina-Switzerland
1-1 aet → fullTime 3-1; Switzerland-Colombia 0-0 aet, 4-3 pens → fullTime 4-3).
update_results must store the final score EXCLUDING shootout goals and persist
score.regularTime into regulation_*goals so 90-minute settlement (C1) never
grades a 90-minute market on a 120-minute score.
"""

import asyncio
from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.data.database import DatabaseManager
from src.data.models import Team, Match
from src.scrapers.footballdataorg_scraper import FootballDataOrgScraper


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "fdo_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _scraper(db, payload):
    s = FootballDataOrgScraper.__new__(FootballDataOrgScraper)
    s.enabled = True
    s.db = db
    s._last_call_at = 0.0

    async def fake_fetch_matches(target):
        return payload

    s.fetch_matches = fake_fetch_matches
    return s


def _seed_match(db, home="Argentina", away="Switzerland", scored=None):
    with db.get_session() as s:
        h, a = Team(name=home), Team(name=away)
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime.combine(date.today(), datetime.min.time()).replace(hour=20),
                  league="world/fifa-world-cup", is_fixture=True)
        if scored:
            m.home_goals, m.away_goals = scored
            m.is_fixture = False
        s.add(m); s.flush()
        mid = m.id
        s.commit()
    return mid


def _wc_payload(home, away, score):
    return [{
        "status": "FINISHED",
        "competition": {"code": "WC"},
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": score,
    }]


def _get(db, mid):
    with db.get_session() as s:
        m = s.get(Match, mid)
        return (m.home_goals, m.away_goals,
                m.regulation_home_goals, m.regulation_away_goals)


def test_extra_time_stores_final_and_regulation(db):
    mid = _seed_match(db)
    scraper = _scraper(db, _wc_payload("Argentina", "Switzerland", {
        "duration": "EXTRA_TIME",
        "fullTime": {"home": 3, "away": 1},      # ET-inclusive
        "regularTime": {"home": 1, "away": 1},   # true 90' score
    }))
    assert asyncio.run(scraper.update_results(days_back=0)) == 1
    assert _get(db, mid) == (3, 1, 1, 1)


def test_penalty_shootout_excludes_shootout_goals(db):
    mid = _seed_match(db, home="Switzerland", away="Colombia")
    scraper = _scraper(db, _wc_payload("Switzerland", "Colombia", {
        "duration": "PENALTY_SHOOTOUT",
        "fullTime": {"home": 4, "away": 3},      # includes shootout goals!
        "penalties": {"home": 4, "away": 3},
        "regularTime": {"home": 0, "away": 0},
    }))
    assert asyncio.run(scraper.update_results(days_back=0)) == 1
    # Final = fullTime minus shootout = 0-0; regulation = 0-0.
    assert _get(db, mid) == (0, 0, 0, 0)


def test_penalty_shootout_without_breakdown_skips_write(db):
    mid = _seed_match(db, home="Switzerland", away="Colombia")
    scraper = _scraper(db, _wc_payload("Switzerland", "Colombia", {
        "duration": "PENALTY_SHOOTOUT",
        "fullTime": {"home": 4, "away": 3},
        # no penalties breakdown → fullTime cannot be decomposed safely
    }))
    assert asyncio.run(scraper.update_results(days_back=0)) == 0
    assert _get(db, mid) == (None, None, None, None)


def test_regulation_backfilled_on_already_scored_match(db):
    # API-Football already wrote the final (3-1) but no regulation score.
    mid = _seed_match(db, scored=(3, 1))
    scraper = _scraper(db, _wc_payload("Argentina", "Switzerland", {
        "duration": "EXTRA_TIME",
        "fullTime": {"home": 3, "away": 1},
        "regularTime": {"home": 1, "away": 1},
    }))
    assert asyncio.run(scraper.update_results(days_back=0)) == 1
    # Final untouched, regulation backfilled.
    assert _get(db, mid) == (3, 1, 1, 1)


def test_regular_match_unchanged_behavior(db):
    mid = _seed_match(db, home="France", away="Morocco")
    scraper = _scraper(db, _wc_payload("France", "Morocco", {
        "duration": "REGULAR",
        "fullTime": {"home": 2, "away": 0},
    }))
    assert asyncio.run(scraper.update_results(days_back=0)) == 1
    assert _get(db, mid) == (2, 0, None, None)
