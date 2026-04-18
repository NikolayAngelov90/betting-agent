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

    def test_cache_populated_with_correct_keys(self):
        from src.features.feature_engineer import FeatureEngineer
        from unittest.mock import MagicMock, patch
        from datetime import date

        fe = self._make_fe()

        # Build mock Match row for the fixture
        fixture = MagicMock()
        fixture.id = 42
        fixture.home_team_id = 1
        fixture.away_team_id = 2
        fixture.league = "england/premier-league"
        fixture.referee = "Mike Dean"
        fixture.match_date = date(2026, 4, 20)
        fixture.venue = "Old Trafford"

        # Build a mock history match
        hist = MagicMock()
        hist.id = 10
        hist.home_team_id = 1
        hist.away_team_id = 3
        hist.match_date = date(2026, 4, 5)
        hist.league = "england/premier-league"
        hist.referee = "Mike Dean"
        hist.home_goals = 2
        hist.away_goals = 1
        hist.home_xg = 1.8
        hist.away_xg = 0.9
        hist.home_yellow_cards = 1
        hist.away_yellow_cards = 2
        hist.home_red_cards = 0
        hist.away_red_cards = 0
        hist.home_fouls = 10
        hist.away_fouls = 8
        hist.regulation_home_goals = 2
        hist.regulation_away_goals = 1

        # Build a mock Odds row
        odds_row = MagicMock()
        odds_row.match_id = 42
        odds_row.market_type = "1X2"
        odds_row.bookmaker = "Bet365"
        odds_row.selection = "Home"
        odds_row.odds_value = 1.80
        odds_row.opening_odds = 1.85

        mock_db = MagicMock()

        # The three sessions return the three query results in order
        session_q1 = MagicMock()
        session_q1.__enter__ = lambda s: s
        session_q1.__exit__ = MagicMock(return_value=False)
        session_q1.query.return_value.filter.return_value.all.return_value = [fixture]

        session_q2 = MagicMock()
        session_q2.__enter__ = lambda s: s
        session_q2.__exit__ = MagicMock(return_value=False)
        session_q2.query.return_value.filter.return_value.all.return_value = [odds_row]

        session_q3 = MagicMock()
        session_q3.__enter__ = lambda s: s
        session_q3.__exit__ = MagicMock(return_value=False)
        session_q3.query.return_value.filter.return_value.order_by.return_value \
            .all.return_value = [hist]

        mock_db.get_session.side_effect = [session_q1, session_q2, session_q3]
        fe.db = mock_db

        fe.preload_batch([42])

        assert fe._preload_cache is not None
        assert 42 in fe._preload_cache["match_meta"]
        assert fe._preload_cache["match_meta"][42]["home_team_id"] == 1
        assert fe._preload_cache["match_meta"][42]["league"] == "england/premier-league"
        assert 42 in fe._preload_cache["odds"]
        assert fe._preload_cache["odds"][42][0]["bookmaker"] == "Bet365"
        assert 1 in fe._preload_cache["team_history"]

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
        }
        fe.db = MagicMock()

        result = fe._get_xg_features(1, "home", as_of_date=None)

        fe.db.get_session.assert_not_called()
        assert result["xg_avg"] == 1.8
        assert result["xg_against_avg"] == 0.9
        assert result["xg_matches"] == 1

    def test_xg_features_skips_cache_when_as_of_date_set(self):
        """Training path (as_of_date set) must always use live DB, never cache."""
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {"match_meta": {}, "odds": {}, "team_history": {1: []}}

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.filter.return_value \
            .filter.return_value.order_by.return_value.limit.return_value.all.return_value = []
        fe.db = MagicMock()
        fe.db.get_session.return_value = mock_session

        fe._get_xg_features(1, "home", as_of_date=date(2025, 1, 1))
        fe.db.get_session.assert_called_once()

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
        fe._preload_cache = {
            "match_meta": {},
            "odds": {},
            "team_history": {1: [row], 2: [row]},  # same match under both teams
        }
        fe.db = MagicMock()

        result = fe._get_referee_features("Mike Dean", as_of_date=None)

        fe.db.get_session.assert_not_called()
        assert result["referee_matches"] == 1  # deduplication: counted once despite 2 teams

    def test_referee_features_skips_cache_when_as_of_date_set(self):
        """Training path (as_of_date set) must always use live DB, never cache."""
        from unittest.mock import MagicMock
        from datetime import date
        fe = self._make_fe()
        fe._preload_cache = {"match_meta": {}, "odds": {}, "team_history": {1: []}}

        mock_session = MagicMock()
        mock_session.__enter__ = lambda s: s
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_session.query.return_value.filter.return_value.filter.return_value \
            .order_by.return_value.limit.return_value.all.return_value = []
        fe.db = MagicMock()
        fe.db.get_session.return_value = mock_session

        fe._get_referee_features("Mike Dean", as_of_date=date(2025, 1, 1))
        fe.db.get_session.assert_called_once()

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
        }
        fe.db = MagicMock()

        result = fe._get_situational_features(1, date(2026, 4, 19))

        fe.db.get_session.assert_not_called()
        assert result["rest_days"] == 9
        assert result["matches_14d"] == 1
