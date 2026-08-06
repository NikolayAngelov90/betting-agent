"""Equivalence tests for the egress-reducing match-history cache.

``Elo.fit()`` and ``PoissonModel.fit()`` used to run ``session.query(Match)``
— ``SELECT matches.*``, 45 columns, no LIMIT for Elo — on every one of the ~10
``predictor.fit()`` calls a CI day makes. They now read a 9-column projection
from a per-process cache (``src/data/match_history.py``).

That is only a legitimate optimisation if it is invisible: these tests pin the
two properties that make it so.

  1. Same rows in, same ratings/strengths out (vs. a reference implementation
     that replays the original ORM query).
  2. The cache notices new data *within* a process — ``daily_update`` refits
     right after a backfill inserts matches, and settlement turns fixtures into
     completed matches. A cache that served stale rows there would silently
     freeze the models.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from src.data import match_history
from src.data.database import DatabaseManager
from src.data.models import Team, Match


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    mgr = DatabaseManager(config=SimpleNamespace(
        database={"sqlite_path": str(tmp_path / "history_test.db")}
    ))
    assert not mgr.is_postgres, "test DB must be SQLite, not production Postgres"
    mgr.create_tables()
    match_history.invalidate()  # module-level cache is process-wide
    yield mgr
    match_history.invalidate()


def _seed(db, n_teams=6, n_matches=60, start=datetime(2024, 1, 1)):
    """Deterministic league of completed matches + one future fixture."""
    with db.get_session() as s:
        teams = [Team(name=f"Team {i}", league="test/league") for i in range(n_teams)]
        s.add_all(teams)
        s.flush()
        ids = [t.id for t in teams]

        for i in range(n_matches):
            home = ids[i % n_teams]
            away = ids[(i + 1 + i // n_teams) % n_teams]
            if home == away:
                away = ids[(i + 2) % n_teams]
            s.add(Match(
                home_team_id=home, away_team_id=away,
                match_date=start + timedelta(days=i),
                league="test/league", season="2024",
                home_goals=i % 4, away_goals=(i + 1) % 3,
                home_xg=0.5 + (i % 5) * 0.3, away_xg=0.4 + (i % 3) * 0.4,
                is_fixture=False,
            ))
        # A fixture (no result) — must never reach either model.
        s.add(Match(home_team_id=ids[0], away_team_id=ids[1],
                    match_date=start + timedelta(days=n_matches + 10),
                    league="test/league", is_fixture=True))
    return ids


def _reference_rows(db, as_of_date=None, league=None, newest_first=False, limit=None):
    """Replay the pre-optimisation query: full ORM entities, ordered in SQL."""
    with db.get_session() as session:
        q = session.query(Match).filter(
            Match.is_fixture == False,  # noqa: E712
            Match.home_goals.isnot(None),
        )
        if league:
            q = q.filter(Match.league == league)
        if as_of_date is not None:
            q = q.filter(Match.match_date < as_of_date)
        q = q.order_by(
            Match.match_date.desc() if newest_first else Match.match_date.asc()
        )
        if limit:
            q = q.limit(limit)
        return [
            (m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
             m.match_date, m.league, m.home_xg, m.away_xg)
            for m in q.all()
        ]


def _cached_rows(db, **kwargs):
    return [
        (m.home_team_id, m.away_team_id, m.home_goals, m.away_goals,
         m.match_date, m.league, m.home_xg, m.away_xg)
        for m in match_history.get_completed_matches(db, **kwargs)
    ]


class TestRowEquivalence:
    def test_unfiltered_matches_reference(self, db):
        _seed(db)
        assert _cached_rows(db) == _reference_rows(db)

    def test_excludes_fixtures_and_unplayed(self, db):
        _seed(db)
        rows = match_history.get_completed_matches(db)
        assert rows, "seed produced no completed matches"
        assert all(r.home_goals is not None for r in rows)
        assert len(rows) == 60  # the lone is_fixture row is excluded

    def test_as_of_date_cutoff_matches_reference(self, db):
        _seed(db)
        cutoff = datetime(2024, 2, 1)
        assert _cached_rows(db, as_of_date=cutoff) == _reference_rows(db, as_of_date=cutoff)

    def test_as_of_date_accepts_a_plain_date(self, db):
        """tune_ensemble_weights passes SavedPick.pick_date, which is a date."""
        _seed(db)
        cutoff = datetime(2024, 2, 1)
        assert _cached_rows(db, as_of_date=cutoff.date()) == _reference_rows(db, as_of_date=cutoff)

    def test_league_filter_matches_reference(self, db):
        _seed(db)
        assert _cached_rows(db, league="test/league") == _reference_rows(db, league="test/league")
        assert _cached_rows(db, league="nope/nothing") == []

    def test_newest_first_with_limit_matches_reference(self, db):
        """Poisson's order_by(match_date.desc()).limit(n)."""
        _seed(db)
        got = _cached_rows(db, newest_first=True, limit=25)
        assert got == _reference_rows(db, newest_first=True, limit=25)
        assert len(got) == 25

    def test_caller_cannot_mutate_the_cache(self, db):
        _seed(db)
        first = match_history.get_completed_matches(db)
        first.pop()
        assert len(match_history.get_completed_matches(db)) == 60


class TestFreshness:
    def test_newly_inserted_matches_are_picked_up(self, db):
        """daily_update refits right after a backfill inserts history."""
        ids = _seed(db)
        assert len(match_history.get_completed_matches(db)) == 60

        with db.get_session() as s:
            s.add(Match(home_team_id=ids[0], away_team_id=ids[2],
                        match_date=datetime(2023, 6, 1), league="test/league",
                        home_goals=2, away_goals=1, is_fixture=False))

        assert len(match_history.get_completed_matches(db)) == 61

    def test_fixture_gaining_a_result_is_picked_up(self, db):
        """Settlement flips is_fixture rows into the completed set."""
        _seed(db)
        assert len(match_history.get_completed_matches(db)) == 60

        with db.get_session() as s:
            fixture = s.query(Match).filter(Match.is_fixture == True).one()  # noqa: E712
            fixture.is_fixture = False
            fixture.home_goals = 1
            fixture.away_goals = 1

        assert len(match_history.get_completed_matches(db)) == 61


class TestModelEquivalence:
    def test_elo_ratings_match_reference_implementation(self, db, monkeypatch):
        from src.models.elo_system import EloRatingSystem, DEFAULT_ELO

        _seed(db)
        monkeypatch.setattr("src.models.elo_system.get_db", lambda: db)

        elo = EloRatingSystem()
        elo.fit()

        # Reference: same chronological pass, driven by the original ORM query.
        ref = EloRatingSystem()
        ref.ratings, ref.history = {}, {}
        prev_year = None
        for (h, a, hg, ag, md, _lg, _hx, _ax) in _reference_rows(db):
            year = md.year if md else None
            if prev_year is not None and year is not None and year > prev_year:
                for tid in list(ref.ratings.keys()):
                    ref.ratings[tid] = ref.ratings[tid] * (1 - 0.33) + DEFAULT_ELO * 0.33
            if year is not None:
                prev_year = year
            ref._process_match(h, a, hg, ag)

        assert elo.ratings.keys() == ref.ratings.keys()
        for tid in ref.ratings:
            assert elo.ratings[tid] == pytest.approx(ref.ratings[tid], abs=1e-9)

    def test_poisson_strengths_are_stable_across_refits(self, db, monkeypatch):
        """The tuning pipeline fits, re-fits at as_of_date, then restores."""
        from src.models.poisson_model import PoissonModel

        _seed(db)
        model = PoissonModel()
        monkeypatch.setattr(model, "db", db)

        model.fit()
        full = dict(model._team_strengths)
        assert full, "Poisson produced no team strengths"

        model.fit(as_of_date=datetime(2024, 2, 1))
        partial = dict(model._team_strengths)

        model.fit()  # restore
        assert model._team_strengths.keys() == full.keys()
        for tid, vals in full.items():
            assert model._team_strengths[tid]["attack"] == pytest.approx(vals["attack"])
            assert model._team_strengths[tid]["defense"] == pytest.approx(vals["defense"])
        assert partial != full, "as_of_date cutoff had no effect — filter is not applied"

    def test_poisson_respects_num_matches_limit(self, db, monkeypatch):
        from src.models.poisson_model import PoissonModel

        _seed(db)
        model = PoissonModel()
        monkeypatch.setattr(model, "db", db)
        model.fit(num_matches=10)
        # 10 most recent matches involve strictly fewer than all six teams'
        # full history, so the fit must differ from the unlimited one.
        limited = dict(model._team_strengths)
        model.fit()
        assert limited != model._team_strengths
