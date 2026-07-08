"""Tests for coverage-gated forced picks (option C, 2026-07-08).

WC matches always get a forced pick; club fixtures only when the model has
decent data (coverage >= betting.club_pick_min_coverage). A forced pick is
one-per-match-per-day across runs (is_forced guard in _save_picks).
"""

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from src.agent.betting_agent import FootballBettingAgent
from src.data.database import DatabaseManager
from src.data.models import Team, Match, SavedPick


class _FakeConfig:
    def __init__(self, values=None):
        self.values = values or {}

    def get(self, key, default=None):
        return self.values.get(key, default)


def _agent(config_values=None, db=None):
    agent = FootballBettingAgent.__new__(FootballBettingAgent)
    agent.config = _FakeConfig(config_values)
    if db is not None:
        agent.db = db
    return agent


class TestShouldForcePick:
    def test_national_league_always_forced_by_default(self):
        a = _agent()
        assert a._should_force_pick("world/fifa-world-cup", 0.0) is True

    def test_national_league_respects_wc_toggle(self):
        a = _agent({"betting.wc_pick_every_match": False})
        assert a._should_force_pick("world/fifa-world-cup", 1.0) is False

    def test_club_forced_when_coverage_meets_threshold(self):
        a = _agent()
        assert a._should_force_pick("europe/champions-league", 0.75) is True
        assert a._should_force_pick("europe/champions-league", 1.0) is True

    def test_club_not_forced_below_threshold(self):
        a = _agent()
        assert a._should_force_pick("europe/champions-league", 0.5) is False
        assert a._should_force_pick("europe/champions-league", 0.0) is False

    def test_club_disabled_when_zero(self):
        a = _agent({"betting.club_pick_min_coverage": 0})
        assert a._should_force_pick("europe/champions-league", 1.0) is False


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "forced_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    return mgr


def _pick(match_id, selection, league="europe/champions-league", forced=False):
    p = SimpleNamespace(
        match_id=match_id, league=league, market="over_under",
        selection=selection, match="Kairat vs Sutjeska", odds=1.70,
        predicted_probability=0.6, expected_value=0.03, confidence=0.6,
        kelly_stake_percentage=1.0, risk_level="low", used_fallback_odds=False,
        model_agreement="majority",
    )
    if forced:
        p.is_forced = True
    return p


def test_forced_club_pick_is_one_per_match_across_runs(db):
    """A second run's DIFFERENT forced selection must not create a second pick."""
    with db.get_session() as s:
        h, a = Team(name="Kairat Almaty"), Team(name="Sutjeska")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 8, 16, 0),
                  league="europe/champions-league", is_fixture=True)
        s.add(m); s.flush()
        mid = m.id
        s.commit()

    agent = _agent(db=db)
    today = date.today()
    saved1 = agent._save_picks([_pick(mid, "Over 2.5 Goals", forced=True)], today)
    assert len(saved1) == 1
    # Second run: odds moved, forced logic picked a different selection.
    saved2 = agent._save_picks([_pick(mid, "Home Over 1.5", forced=True)], today)
    assert len(saved2) == 0, "forced pick must be one-per-match-per-day"
    with db.get_session() as s:
        assert s.query(SavedPick).filter(SavedPick.match_id == mid).count() == 1


def test_club_value_picks_unaffected_by_forced_guard(db):
    """Ordinary club value picks keep the 2-per-match allowance."""
    with db.get_session() as s:
        h, a = Team(name="Ajax"), Team(name="PSV")
        s.add_all([h, a]); s.flush()
        m = Match(home_team_id=h.id, away_team_id=a.id,
                  match_date=datetime(2026, 7, 8, 18, 0),
                  league="netherlands/eredivisie", is_fixture=True)
        s.add(m); s.flush()
        mid = m.id
        s.commit()

    agent = _agent(db=db)
    today = date.today()
    saved1 = agent._save_picks([_pick(mid, "Over 2.5 Goals")], today)
    saved2 = agent._save_picks([_pick(mid, "Home Over 1.5")], today)
    assert len(saved1) == 1 and len(saved2) == 1
