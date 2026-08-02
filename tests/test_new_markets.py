"""Tests for the 2026-08-02 market-coverage expansion:
Double Chance (1X/12/X2), Draw No Bet, match Over/Under 4.5, team "to score"
(over 0.5 per team), and re-enabled Under 2.5/3.5.
"""

from unittest.mock import MagicMock

import pytest


class TestPoissonNewLines:
    def test_over_45_le_over_35(self):
        from src.models.poisson_model import PoissonModel
        m = PoissonModel()
        mat = m._score_matrix(1.9, 1.5)
        assert m._over_under_prob(mat, 4.5) <= m._over_under_prob(mat, 3.5)

    def test_team_over_05_ge_over_15(self):
        from src.models.poisson_model import PoissonModel
        m = PoissonModel()
        mat = m._score_matrix(1.9, 1.5)
        assert (m._team_over_prob(mat, 0.5, side="home")
                >= m._team_over_prob(mat, 1.5, side="home"))

    def test_predict_outputs_new_keys_and_drops_match_over05(self):
        from src.models.poisson_model import PoissonModel
        pred = PoissonModel().predict(1, 2)
        for k in ("over_4.5", "home_over_0.5", "away_over_0.5"):
            assert k in pred and 0.0 <= pred[k] <= 1.0
        # match "over 0.5" is a ~94% non-market — intentionally NOT emitted
        assert "over_0.5" not in pred


_ENS = {
    "home_win": 0.55, "draw": 0.25, "away_win": 0.20,
    "over_1.5": 0.75, "over_2.5": 0.55, "over_3.5": 0.30, "over_4.5": 0.12,
    "under_1.5": 0.25, "under_2.5": 0.45, "under_3.5": 0.70, "under_4.5": 0.88,
    "btts_yes": 0.55, "btts_no": 0.45,
    "home_over_0.5": 0.82, "away_over_0.5": 0.68,
    "home_over_1.5": 0.48, "away_over_1.5": 0.33,
    "dc_1x": 0.80, "dc_12": 0.75, "dc_x2": 0.45,
    "dnb_home": 0.73, "dnb_away": 0.27,
}


def _calc():
    from src.betting.value_calculator import ValueBettingCalculator
    c = ValueBettingCalculator.__new__(ValueBettingCalculator)
    c.min_odds = 1.30; c.max_odds = 10.0; c.min_ev = 0.03
    c.min_confidence = 0.55; c.high_ev_min_confidence = 0.45
    c.min_ev_confidence_score = 0.035; c.kelly_fraction = 0.25
    c.max_stake_pct = 5.0; c.excluded_markets = set(); c.min_kelly_stake = 0.5
    c.config = MagicMock(); c.config.get.return_value = {}
    return c


class TestNewMarketPlumbing:
    def test_market_specs_includes_all_new_keys(self):
        keys = {k for (_, _, _, k) in _calc()._market_specs(_ENS)}
        for k in ("dc_1x", "dc_12", "dc_x2", "dnb_home", "dnb_away",
                  "over_4.5", "under_4.5", "home_over_0.5", "away_over_0.5",
                  "under_2.5", "under_3.5"):
            assert k in keys, f"missing market_key {k}"
        # match over/under 0.5 must NOT be a market (dropped in favour of team 0.5)
        assert "over_0.5" not in keys and "under_0.5" not in keys

    def test_double_chance_odds_matched(self):
        c = _calc()
        odds = [{"bookmaker": "Bet365", "market_type": "double_chance",
                 "selection": "Double Chance 1X", "odds_value": 1.40}]
        assert c._find_best_odds(odds, "Double Chance", "Double Chance 1X") == 1.40

    def test_dnb_odds_matched(self):
        c = _calc()
        odds = [{"bookmaker": "Bet365", "market_type": "draw_no_bet",
                 "selection": "DNB Home", "odds_value": 1.55}]
        assert c._find_best_odds(odds, "Draw No Bet", "DNB Home") == 1.55

    def test_match_over45_matched(self):
        c = _calc()
        odds = [{"bookmaker": "Bet365", "market_type": "over_under",
                 "selection": "Over 4.5", "odds_value": 6.0}]
        assert c._find_best_odds(odds, "Over 4.5", "Over 4.5 Goals") == 6.0

    def test_team_over05_matched(self):
        c = _calc()
        odds = [{"bookmaker": "Bet365", "market_type": "team_goals",
                 "selection": "Home Over 0.5", "odds_value": 1.30}]
        assert c._find_best_odds(odds, "Team Goals", "Home Over 0.5") == 1.30

    def test_market_ev_dnb_applies_draw_refund(self):
        c = _calc()
        ens = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        # DNB conditional prob 0.667 @ 1.60 → generic EV 0.0672; refund factor
        # (1-draw)=0.75 → 0.0504. A plain 1X2 market is unchanged.
        ev_dnb = c._market_ev(0.667, 1.60, "dnb_home", ens)
        assert abs(ev_dnb - 0.0504) < 0.003
        ev_1x2 = c._market_ev(0.55, 2.0, "home_win", ens)
        assert abs(ev_1x2 - 0.10) < 1e-9
        assert ev_dnb < c.calculate_expected_value(0.667, 1.60)  # refund lowers it

    def test_double_chance_produces_value_bet(self):
        """End-to-end: dc_1x 0.80 vs 1.40 implied (0.714) is +EV → a pick."""
        c = _calc()
        odds = [{"bookmaker": "Bet365", "market_type": "double_chance",
                 "selection": "Double Chance 1X", "odds_value": 1.40}]
        recs = c.find_value_bets({"ensemble": _ENS}, odds, match_name="A vs B",
                                 home_team_name="A", away_team_name="B", match_id=1)
        assert any(r.selection == "Double Chance 1X" for r in recs)
