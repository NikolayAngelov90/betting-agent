"""Tests for feature engineering modules."""

import pytest
from unittest.mock import MagicMock, patch


class TestH2HFeatures:
    """Tests for head-to-head feature calculation."""

    def test_empty_h2h(self):
        from src.features.h2h_features import H2HFeatures
        h2h = H2HFeatures.__new__(H2HFeatures)
        result = h2h._empty_h2h()
        assert result["h2h_total_meetings"] == 0
        assert result["h2h_avg_total_goals"] == 0

    def test_calculate_h2h_all_home_wins(self):
        from src.features.h2h_features import H2HFeatures

        h2h = H2HFeatures.__new__(H2HFeatures)

        # Create mock matches where home_team_id=1 always wins
        matches = []
        for _ in range(5):
            m = MagicMock()
            m.home_team_id = 1
            m.away_team_id = 2
            m.home_goals = 2
            m.away_goals = 0
            matches.append(m)

        result = h2h._calculate_h2h(matches, home_team_id=1, away_team_id=2)
        assert result["h2h_home_wins"] == 5
        assert result["h2h_away_wins"] == 0
        assert result["h2h_draws"] == 0
        assert result["h2h_home_win_pct"] == 1.0

    def test_calculate_h2h_btts(self):
        from src.features.h2h_features import H2HFeatures
        h2h = H2HFeatures.__new__(H2HFeatures)

        matches = []
        for _ in range(4):
            m = MagicMock()
            m.home_team_id = 1
            m.away_team_id = 2
            m.home_goals = 2
            m.away_goals = 1  # Both teams score
            matches.append(m)

        result = h2h._calculate_h2h(matches, 1, 2)
        assert result["h2h_btts_percentage"] == 1.0


class TestInjuryFeatures:
    """Tests for injury impact features."""

    def test_empty_injury_features(self):
        from src.features.injury_features import InjuryFeatures
        inj = InjuryFeatures.__new__(InjuryFeatures)
        result = inj._empty_injury_features()
        assert result["total_injured"] == 0
        assert result["defensive_stability_score"] == 1.0
        assert result["attacking_threat_score"] == 1.0
        assert result["goalkeeper_available"] is True


class TestTeamFeatures:
    """Tests for team form features."""

    def test_empty_form_features(self):
        from src.features.team_features import TeamFeatures
        tf = TeamFeatures.__new__(TeamFeatures)
        result = tf._empty_form_features()
        assert result["matches_played"] == 0
        assert result["points"] == 0
        assert result["form_string"] == ""

    def test_calculate_streak(self):
        from src.features.team_features import TeamFeatures
        tf = TeamFeatures.__new__(TeamFeatures)

        assert tf._calculate_streak(["W", "W", "W", "D", "L"], "W") == 3
        assert tf._calculate_streak(["L", "W", "W"], "L") == 1
        assert tf._calculate_streak(["D", "D", "W"], "D") == 2
        assert tf._calculate_streak(["D", "W", "L"], "W") == 0

    def test_unbeaten_run(self):
        from src.features.team_features import TeamFeatures
        tf = TeamFeatures.__new__(TeamFeatures)

        assert tf._calculate_unbeaten_run(["W", "D", "W", "L", "W"]) == 3
        assert tf._calculate_unbeaten_run(["L", "W", "D"]) == 0
        assert tf._calculate_unbeaten_run(["W", "W", "W"]) == 3


class TestFeatureEngineerPreloadBatch:
    """Tests for FeatureEngineer.preload_batch() — Story 1.1."""

    def _make_fe(self):
        """Return a FeatureEngineer instance without touching the DB."""
        from src.features.feature_engineer import FeatureEngineer
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe._preload_cache = None
        return fe

    def test_cache_is_none_by_default(self):
        fe = self._make_fe()
        assert fe._preload_cache is None

    def test_empty_match_ids_is_noop(self):
        fe = self._make_fe()
        fe.preload_batch([])
        assert fe._preload_cache is None

    def test_cache_populated_with_correct_keys(self, tmp_path, monkeypatch):
        """preload_batch fills every scope, against a real database.

        This used to drive three hand-built MagicMock sessions keyed by call
        order, which meant it passed no matter what SQL was emitted and broke
        the moment the number of queries changed. preload_batch now issues six
        (team, league, referee and H2H scopes were added to fix the standings /
        referee / H2H divergence), so the test runs the real thing on SQLite.
        """
        from datetime import datetime, timedelta
        from types import SimpleNamespace
        from src.data.database import DatabaseManager
        from src.data.models import Team, Match, Odds
        from src.features.feature_engineer import FeatureEngineer

        monkeypatch.delenv("DATABASE_URL", raising=False)
        db = DatabaseManager(config=SimpleNamespace(
            database={"sqlite_path": str(tmp_path / "preload.db")}))
        assert not db.is_postgres
        db.create_tables()

        base = datetime.now() - timedelta(days=120)
        with db.get_session() as s:
            teams = [Team(name=f"T{i}", league="england/premier-league")
                     for i in range(4)]
            s.add_all(teams)
            s.flush()
            ids = [t.id for t in teams]
            for i in range(12):
                s.add(Match(
                    home_team_id=ids[i % 4], away_team_id=ids[(i + 1) % 4],
                    match_date=base + timedelta(days=i * 5),
                    league="england/premier-league", season="2025",
                    home_goals=i % 3, away_goals=(i + 1) % 2,
                    home_xg=1.1, away_xg=0.8, referee="Mike Dean",
                    is_fixture=False,
                ))
            fixture = Match(
                home_team_id=ids[0], away_team_id=ids[1],
                match_date=datetime.now() + timedelta(days=2),
                league="england/premier-league", referee="Mike Dean",
                is_fixture=True,
            )
            s.add(fixture)
            s.flush()
            fid = fixture.id
            s.add(Odds(match_id=fid, bookmaker="Bet365", market_type="1X2",
                       selection="Home", odds_value=1.80, opening_odds=1.85))

        fe = FeatureEngineer()
        fe.db = db
        fe.preload_batch([fid])
        cache = fe._preload_cache

        assert cache is not None
        # Fixture metadata
        assert fid in cache["match_meta"]
        assert cache["match_meta"][fid]["home_team_id"] == ids[0]
        assert cache["match_meta"][fid]["league"] == "england/premier-league"
        # Odds
        assert cache["odds"][fid][0]["bookmaker"] == "Bet365"
        # Per-team history (the fixture's own two teams)
        assert ids[0] in cache["team_history"]
        # League-wide scope — every club in the division, not just those two
        by_team = cache["league_history"]["england/premier-league"]
        assert set(by_team) == set(ids)
        assert "england/premier-league" in cache["league_complete"]
        # Referee scope, with its own rows
        assert cache["referee_history"]["Mike Dean"]
        assert "Mike Dean" in cache["referee_complete"]
        # H2H scope for this pairing
        from src.features.preload_cache import h2h_key
        assert h2h_key(ids[0], ids[1]) in cache["h2h_complete"]

    def test_exception_sets_cache_to_none(self):
        fe = self._make_fe()

        mock_db = MagicMock()
        mock_db.get_session.side_effect = RuntimeError("DB connection lost")
        fe.db = mock_db

        fe.preload_batch([99])
        assert fe._preload_cache is None

    def test_cache_absent_does_not_break_feature_lookup(self):
        """With no preload, _preload_cache stays None — callers can check and fall back."""
        fe = self._make_fe()
        assert fe._preload_cache is None
        # Confirm falsy (not an empty dict that could confuse cache-check logic in Story 1.2)
        assert not fe._preload_cache

    def test_ac4_uncached_match_deferred_to_story_1_2(self):
        """AC4: create_features falls back for a match_id not in the preloaded batch.

        Full coverage requires Story 1.2 to wire _preload_cache into _get_*_features.
        This test documents the deferral and verifies the cache structure does NOT
        contain a match_id that was never preloaded — confirming Story 1.2 can detect
        a cache miss with a simple 'match_id in _preload_cache["match_meta"]' check.
        """
        fe = self._make_fe()
        # Simulate a populated cache that does NOT include match_id=99
        fe._preload_cache = {"match_meta": {1: {}}, "odds": {}, "team_history": {}}
        assert 99 not in fe._preload_cache["match_meta"]  # cache miss → live fallback in 1.2


# ---------------------------------------------------------------------------
# Story 1.2 — Cache-Aware Feature Computation
# ---------------------------------------------------------------------------

def _empty_result_session():
    """A session mock whose query builder returns itself and yields no rows.

    Chained builders (``query().filter().filter().order_by().limit().all()``)
    need every link to answer; a mock stubbed at one fixed depth silently
    returns a fresh MagicMock further down, and a bare MagicMock iterates as
    empty while still being truthy — which reads as "rows exist but there are
    none of them" and blows up on a division. Self-returning avoids the trap.
    """
    from unittest.mock import MagicMock
    session = MagicMock()
    session.__enter__ = lambda s: s
    session.__exit__ = MagicMock(return_value=False)
    builder = MagicMock()
    for method in ("filter", "filter_by", "order_by", "limit", "distinct", "join"):
        getattr(builder, method).return_value = builder
    builder.all.return_value = []
    builder.first.return_value = None
    builder.count.return_value = 0
    builder.__iter__ = lambda s: iter([])
    session.query.return_value = builder
    return session


class TestCacheAwareFeatures:
    """Tests that private _get_* methods read from _preload_cache, not the DB."""

    def _make_fe(self):
        from src.features.feature_engineer import FeatureEngineer
        fe = FeatureEngineer.__new__(FeatureEngineer)
        fe._preload_cache = None
        fe._league_features_cache = {}
        return fe

    # ── _get_bookmaker_features ───────────────────────────────────────────

    def test_bookmaker_features_uses_cache_not_db(self):
        """With odds in cache, _get_bookmaker_features must NOT open a DB session."""
        from unittest.mock import MagicMock
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {},
            "odds": {
                42: [
                    {"market_type": "1X2", "bookmaker": "Bet365", "selection": "Home",
                     "odds_value": 2.0, "opening_odds": 2.1},
                    {"market_type": "1X2", "bookmaker": "Bet365", "selection": "Draw",
                     "odds_value": 3.5, "opening_odds": 3.4},
                    {"market_type": "1X2", "bookmaker": "Bet365", "selection": "Away",
                     "odds_value": 4.0, "opening_odds": 3.9},
                ]
            },
            "team_history": {},
        }
        mock_db = MagicMock()
        fe.db = mock_db

        result = fe._get_bookmaker_features(42)

        mock_db.get_session.assert_not_called()
        assert result["bookmaker_available"] == 1
        assert result["home_implied_prob"] > 0

    def test_bookmaker_features_fallback_when_no_cache(self):
        """With _preload_cache=None, _get_bookmaker_features falls back to DB."""
        from unittest.mock import MagicMock
        fe = self._make_fe()
        fe._preload_cache = None

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.all.return_value = []
        fe.db = MagicMock()
        fe.db.get_session.return_value = mock_session

        result = fe._get_bookmaker_features(42)
        fe.db.get_session.assert_called_once()
        assert result["bookmaker_available"] == 0

    def test_bookmaker_features_cache_miss_falls_back(self):
        """Cache populated but match_id not in odds → DB fallback."""
        from unittest.mock import MagicMock
        fe = self._make_fe()
        fe._preload_cache = {"match_meta": {}, "odds": {99: []}, "team_history": {}}

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.all.return_value = []
        fe.db = MagicMock()
        fe.db.get_session.return_value = mock_session

        fe._get_bookmaker_features(42)   # 42 not in odds cache
        fe.db.get_session.assert_called_once()

    # ── _get_odds_movement_features ───────────────────────────────────────

    def test_odds_movement_uses_cache_not_db(self):
        from unittest.mock import MagicMock
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {},
            "odds": {
                42: [
                    {"market_type": "1X2", "bookmaker": "Bet365", "selection": "Home",
                     "odds_value": 1.90, "opening_odds": 2.00},
                    {"market_type": "1X2", "bookmaker": "Bet365", "selection": "Away",
                     "odds_value": 4.20, "opening_odds": 4.00},
                ]
            },
            "team_history": {},
        }
        fe.db = MagicMock()

        result = fe._get_odds_movement_features(42)

        fe.db.get_session.assert_not_called()
        assert result["home_odds_movement"] != 0.0

    # ── _get_xg_features ─────────────────────────────────────────────────

    def test_xg_features_uses_cache_not_db(self):
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {},
            "odds": {},
            "team_history": {
                1: [
                    {"id": 10, "match_date": date(2026, 3, 1),
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 2, "away_goals": 1,
                     "home_xg": 1.8, "away_xg": 0.9,
                     "home_yellow_cards": 1, "away_yellow_cards": 0,
                     "home_red_cards": 0, "away_red_cards": 0,
                     "home_fouls": 10, "away_fouls": 8,
                     "regulation_home_goals": 2, "regulation_away_goals": 1,
                     "league": "epl", "referee": "Dean"},
                ]
            },
            # Team 1's slice is its COMPLETE history — without this the cache
            # cannot promise that one row is really "the last 10 matches", and
            # the accessor correctly defers to the DB instead of inventing an
            # answer. See test_xg_features_incomplete_slice_falls_back.
            "team_complete": {1},
        }
        fe.db = MagicMock()

        result = fe._get_xg_features(1, "home", as_of_date=None)

        fe.db.get_session.assert_not_called()
        assert result["xg_avg"] == 1.8
        assert result["xg_against_avg"] == 0.9
        assert result["xg_matches"] == 1

    def test_xg_features_incomplete_slice_falls_back(self):
        """A truncated slice that cannot cover the request must hit the DB.

        This is the property that makes preload and live equivalent: one cached
        match is not an answer to "the last 10", so the cache declines.
        """
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {}, "odds": {},
            "team_history": {
                1: [
                    {"id": 10, "match_date": date(2026, 3, 1),
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 2, "away_goals": 1,
                     "home_xg": 1.8, "away_xg": 0.9, "league": "epl"},
                ]
            },
            "team_complete": set(),      # truncated — completeness unknown
        }
        fe.db = MagicMock()
        fe.db.get_session.return_value = _empty_result_session()

        result = fe._get_xg_features(1, "home", as_of_date=None)

        fe.db.get_session.assert_called_once()
        assert result["xg_matches"] == 0

    def test_xg_features_uses_cache_with_as_of_date(self):
        """Training path (as_of_date set) uses cache with Python-side date filter."""
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {}, "odds": {},
            "team_history": {
                1: [
                    {"id": 10, "match_date": date(2024, 6, 1),
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 2, "away_goals": 1,
                     "home_xg": 1.5, "away_xg": 0.8,
                     "home_yellow_cards": 1, "away_yellow_cards": 0,
                     "home_red_cards": 0, "away_red_cards": 0,
                     "home_fouls": 10, "away_fouls": 8,
                     "regulation_home_goals": 2, "regulation_away_goals": 1,
                     "league": "epl", "referee": "Dean"},
                    {"id": 11, "match_date": date(2025, 6, 1),  # after as_of_date, should be filtered
                     "home_team_id": 1, "away_team_id": 3,
                     "home_goals": 1, "away_goals": 0,
                     "home_xg": 2.0, "away_xg": 0.5,
                     "home_yellow_cards": 0, "away_yellow_cards": 0,
                     "home_red_cards": 0, "away_red_cards": 0,
                     "home_fouls": 8, "away_fouls": 7,
                     "regulation_home_goals": 1, "regulation_away_goals": 0,
                     "league": "epl", "referee": "Smith"},
                ]
            },
            "team_complete": {1},
        }
        fe.db = MagicMock()

        result = fe._get_xg_features(1, "home", as_of_date=date(2025, 1, 1))

        fe.db.get_session.assert_not_called()  # cache used, no DB call
        assert result["xg_matches"] == 1  # only the pre-2025 match passes date filter
        assert result["xg_avg"] == 1.5

    # ── _get_referee_features ─────────────────────────────────────────────

    def test_referee_features_uses_cache_not_db(self):
        from unittest.mock import MagicMock
        from datetime import date
        row = {"id": 5, "match_date": date(2026, 3, 10),
               "home_team_id": 1, "away_team_id": 2,
               "home_goals": 2, "away_goals": 2,
               "home_xg": 1.5, "away_xg": 1.5,
               "home_yellow_cards": 3, "away_yellow_cards": 2,
               "home_red_cards": 0, "away_red_cards": 1,
               "home_fouls": 12, "away_fouls": 11,
               "regulation_home_goals": 2, "regulation_away_goals": 2,
               "league": "epl", "referee": "Mike Dean"}
        fe = self._make_fe()
        # Referee stats now come from their own scope. They used to be scraped
        # out of team_history, which only ever held the fixture's own two teams —
        # so an official's "last 30 matches" was really "whichever of these two
        # clubs' games they happened to take", and the live query's 365-day
        # window was skipped entirely.
        fe._preload_cache = {
            "match_meta": {},
            "odds": {},
            "team_history": {1: [row], 2: [row]},
            "referee_history": {"Mike Dean": [row]},
            "referee_complete": {"Mike Dean"},
        }
        fe.db = MagicMock()

        result = fe._get_referee_features("Mike Dean", as_of_date=None)

        fe.db.get_session.assert_not_called()
        assert result["referee_matches"] == 1

    def test_referee_features_unknown_referee_falls_back(self):
        """A referee absent from the scope is a cache miss, not 'no history'."""
        from unittest.mock import MagicMock
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {}, "odds": {}, "team_history": {},
            "referee_history": {}, "referee_complete": set(),
        }
        fe.db = MagicMock()
        fe.db.get_session.return_value = _empty_result_session()

        fe._get_referee_features("Someone Else", as_of_date=None)

        fe.db.get_session.assert_called_once()

    def test_referee_features_uses_cache_with_as_of_date(self):
        """Training path (as_of_date set) uses cache with Python-side date filter."""
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        early_row = {"id": 5, "match_date": date(2024, 6, 1),
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 2, "away_goals": 2,
                     "home_xg": 1.5, "away_xg": 1.5,
                     "home_yellow_cards": 3, "away_yellow_cards": 2,
                     "home_red_cards": 0, "away_red_cards": 1,
                     "home_fouls": 12, "away_fouls": 11,
                     "regulation_home_goals": 2, "regulation_away_goals": 2,
                     "league": "epl", "referee": "Mike Dean"}
        late_row = {**early_row, "id": 6, "match_date": date(2025, 6, 1)}  # after cutoff
        fe._preload_cache = {
            "match_meta": {}, "odds": {},
            "team_history": {1: [early_row, late_row], 2: [early_row, late_row]},
            "referee_history": {"Mike Dean": [late_row, early_row]},
            "referee_complete": {"Mike Dean"},
        }
        fe.db = MagicMock()

        result = fe._get_referee_features("Mike Dean", as_of_date=date(2025, 1, 1))

        fe.db.get_session.assert_not_called()  # cache used, no DB call
        assert result["referee_matches"] == 1  # only pre-2025 match passes filter

    # ── _get_situational_features ─────────────────────────────────────────

    def test_situational_features_uses_cache_not_db(self):
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {
            "match_meta": {},
            "odds": {},
            "team_history": {
                1: [
                    {"id": 10, "match_date": date(2026, 4, 10),
                     "home_team_id": 1, "away_team_id": 2,
                     "home_goals": 1, "away_goals": 0,
                     "home_xg": 1.2, "away_xg": 0.7,
                     "home_yellow_cards": 1, "away_yellow_cards": 1,
                     "home_red_cards": 0, "away_red_cards": 0,
                     "home_fouls": 9, "away_fouls": 8,
                     "regulation_home_goals": None, "regulation_away_goals": None,
                     "league": "epl", "referee": "Dean"},
                ]
            },
            "team_complete": {1},
        }
        fe.db = MagicMock()

        result = fe._get_situational_features(1, date(2026, 4, 19))

        fe.db.get_session.assert_not_called()
        assert result["rest_days"] == 9
        assert result["matches_14d"] == 1
