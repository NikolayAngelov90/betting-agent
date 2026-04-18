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
        # Confirm the cache key is absent (not an empty dict that could confuse consumers)
        assert not fe._preload_cache
