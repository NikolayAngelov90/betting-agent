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
