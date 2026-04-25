"""Tests for value-betting decision logic that was previously uncovered.

Targets the most consequential business logic: median odds aggregation,
correlation filter, exposure cap, drawdown breaker, and over-line
monotonicity. These tests use minimal stubs so they don't touch the DB.
"""

from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _make_calc(min_ev=0.03, kelly_fraction=0.25, max_stake_pct=5.0,
               min_confidence=0.58):
    """Return a ValueBettingCalculator without running __init__."""
    from src.betting.value_calculator import ValueBettingCalculator
    calc = ValueBettingCalculator.__new__(ValueBettingCalculator)
    calc.min_odds = 1.30
    calc.max_odds = 10.0
    calc.min_ev = min_ev
    calc.min_confidence = min_confidence
    calc.high_ev_min_confidence = 0.45
    calc.min_ev_confidence_score = 0.035
    calc.kelly_fraction = kelly_fraction
    calc.max_stake_pct = max_stake_pct
    calc.excluded_markets = set()
    calc.min_kelly_stake = 0.5
    calc.config = MagicMock()
    calc.config.get.return_value = {}
    return calc


def _make_rec(match_id, selection, ev, confidence, odds=2.0,
              agreement="unanimous", contrarian_value=1.0,
              market="1X2", match="A vs B", league=""):
    """Build a minimal BetRecommendation for filter/cap tests."""
    from src.betting.value_calculator import BetRecommendation
    return BetRecommendation(
        match=match,
        match_id=match_id,
        market=market,
        selection=selection,
        odds=odds,
        predicted_probability=confidence,
        expected_value=ev,
        confidence=confidence,
        kelly_stake_percentage=2.0,
        recommended_stake=2.0,
        reasoning="",
        risk_level="medium",
        league=league,
        model_agreement=agreement,
        contrarian_value=contrarian_value,
    )


# --------------------------------------------------------------------------- #
# Median odds aggregation
# --------------------------------------------------------------------------- #


class TestFindBestOdds:
    """`_find_best_odds` should return the MEDIAN of available bookies, not max."""

    def test_median_of_three_bookies(self):
        calc = _make_calc()
        odds = [
            {"bookmaker": "Bet365", "market_type": "1X2", "selection": "Home", "odds_value": 2.10},
            {"bookmaker": "Pinnacle", "market_type": "1X2", "selection": "Home", "odds_value": 2.05},
            {"bookmaker": "1xBet", "market_type": "1X2", "selection": "Home", "odds_value": 2.40},
        ]
        # Median of [2.05, 2.10, 2.40] is 2.10
        assert calc._find_best_odds(odds, "1X2", "Home Win") == 2.10

    def test_median_of_two_bookies_averages(self):
        calc = _make_calc()
        odds = [
            {"bookmaker": "Bet365", "market_type": "1X2", "selection": "Home", "odds_value": 2.0},
            {"bookmaker": "Pinnacle", "market_type": "1X2", "selection": "Home", "odds_value": 2.20},
        ]
        # Even count: average of the two
        assert calc._find_best_odds(odds, "1X2", "Home Win") == 2.10

    def test_single_bookmaker_returns_its_price(self):
        calc = _make_calc()
        odds = [{"bookmaker": "Bet365", "market_type": "1X2", "selection": "Home", "odds_value": 2.10}]
        assert calc._find_best_odds(odds, "1X2", "Home Win") == 2.10

    def test_no_odds_returns_zero(self):
        calc = _make_calc()
        assert calc._find_best_odds([], "1X2", "Home Win") == 0.0

    def test_flashscore_only_for_over_under_fallback(self):
        """Flashscore prices are NOT acceptable for 1X2."""
        calc = _make_calc()
        odds = [
            {"bookmaker": "Flashscore", "market_type": "1X2", "selection": "Home", "odds_value": 2.10},
        ]
        # Flashscore 1X2 is composite, not real bookmaker — must be ignored.
        assert calc._find_best_odds(odds, "1X2", "Home Win") == 0.0

    def test_flashscore_over_under_fallback_when_no_real(self):
        calc = _make_calc()
        odds = [
            {"bookmaker": "Flashscore", "market_type": "over_under",
             "selection": "Over 1.5", "odds_value": 1.40},
        ]
        # No real bookmaker → use Flashscore over/under price
        assert calc._find_best_odds(odds, "Over 1.5", "Over 1.5 Goals") == 1.40

    def test_zero_or_invalid_odds_skipped(self):
        calc = _make_calc()
        odds = [
            {"bookmaker": "Bet365", "market_type": "1X2", "selection": "Home", "odds_value": 1.0},
            {"bookmaker": "Pinnacle", "market_type": "1X2", "selection": "Home", "odds_value": 0},
            {"bookmaker": "1xBet", "market_type": "1X2", "selection": "Home", "odds_value": 2.10},
        ]
        # Only the valid one survives
        assert calc._find_best_odds(odds, "1X2", "Home Win") == 2.10


# --------------------------------------------------------------------------- #
# Correlated picks filter
# --------------------------------------------------------------------------- #


class TestCorrelatedPicksFilter:
    """`_filter_correlated_picks` drops the lower-composite-score pick from
    correlated pairs on the same match — using the SAME composite score as
    the main sort (EV * confidence * agreement * contrarian)."""

    def _agent(self):
        from src.agent.betting_agent import FootballBettingAgent
        return FootballBettingAgent.__new__(FootballBettingAgent)

    def test_drops_one_of_correlated_pair(self):
        agent = self._agent()
        winner = _make_rec(1, "Home Win", ev=0.10, confidence=0.65, agreement="unanimous")
        loser = _make_rec(1, "Over 2.5", ev=0.06, confidence=0.55, agreement="majority")
        result = agent._filter_correlated_picks([winner, loser])
        assert len(result) == 1
        assert result[0].selection == "Home Win"

    def test_unanimous_beats_split_with_higher_raw_ev(self):
        """Composite score (with agreement bonus) trumps raw EV."""
        agent = self._agent()
        # Split-models pick has higher raw EV*conf but lower composite.
        split_higher_ev = _make_rec(1, "Over 2.5", ev=0.10, confidence=0.55,
                                    agreement="split")
        unanimous_lower_ev = _make_rec(1, "Home Win", ev=0.085, confidence=0.62,
                                       agreement="unanimous")
        # raw scores: 0.10*0.55=0.055 vs 0.085*0.62=0.0527 (split wins on raw)
        # composite: 0.055*0.85=0.04675 vs 0.0527*1.15=0.0606 (unanimous wins)
        result = agent._filter_correlated_picks([split_higher_ev, unanimous_lower_ev])
        assert len(result) == 1
        assert result[0].selection == "Home Win"

    def test_uncorrelated_pairs_unchanged(self):
        agent = self._agent()
        a = _make_rec(1, "Home Win", ev=0.10, confidence=0.65)
        b = _make_rec(1, "BTTS Yes", ev=0.06, confidence=0.55)  # not in CORRELATED_PAIRS
        result = agent._filter_correlated_picks([a, b])
        assert len(result) == 2

    def test_different_matches_unchanged(self):
        agent = self._agent()
        a = _make_rec(1, "Home Win", ev=0.10, confidence=0.65)
        b = _make_rec(2, "Over 2.5", ev=0.06, confidence=0.55)
        result = agent._filter_correlated_picks([a, b])
        assert len(result) == 2

    def test_three_correlated_drops_to_one(self):
        agent = self._agent()
        a = _make_rec(1, "Home Win", ev=0.10, confidence=0.65, agreement="unanimous")
        b = _make_rec(1, "Over 2.5", ev=0.06, confidence=0.55, agreement="majority")
        c = _make_rec(1, "Home Over 1.5", ev=0.05, confidence=0.50, agreement="split")
        result = agent._filter_correlated_picks([a, b, c])
        # All three are pairwise correlated; the composite winner survives.
        assert len(result) == 1
        assert result[0].selection == "Home Win"


# --------------------------------------------------------------------------- #
# Daily exposure cap
# --------------------------------------------------------------------------- #


class TestExposureCap:
    """`_apply_exposure_cap` trims to ≤ max_pct from the bottom while
    keeping at least the top pick (scaled if it alone exceeds cap)."""

    def _agent(self):
        from src.agent.betting_agent import FootballBettingAgent
        return FootballBettingAgent.__new__(FootballBettingAgent)

    def test_under_cap_keeps_all(self):
        agent = self._agent()
        recs = [_make_rec(i, "Home Win", 0.05, 0.6) for i in range(3)]
        for r in recs:
            r.kelly_stake_percentage = 5.0
        capped, dropped = agent._apply_exposure_cap(recs, 40.0)
        assert len(capped) == 3
        assert len(dropped) == 0

    def test_trims_lowest_when_over_cap(self):
        agent = self._agent()
        recs = []
        for i in range(5):
            r = _make_rec(i, "Home Win", 0.05, 0.6)
            r.kelly_stake_percentage = 10.0  # 5x10% = 50% total
            recs.append(r)
        capped, dropped = agent._apply_exposure_cap(recs, 30.0)
        # Cap = 30%; can only fit 3 picks @ 10% each
        assert len(capped) == 3
        assert len(dropped) == 2
        assert sum(r.kelly_stake_percentage for r in capped) <= 30.0

    def test_first_pick_alone_exceeds_cap_is_scaled(self):
        agent = self._agent()
        big = _make_rec(0, "Home Win", 0.05, 0.6)
        big.kelly_stake_percentage = 50.0  # Single pick > 40% cap
        small = _make_rec(1, "Away Win", 0.04, 0.55)
        small.kelly_stake_percentage = 5.0
        capped, dropped = agent._apply_exposure_cap([big, small], 40.0)
        # Top pick scaled to cap; second dropped
        assert len(capped) == 1
        assert capped[0].kelly_stake_percentage == 40.0
        assert len(dropped) == 1


# --------------------------------------------------------------------------- #
# Drawdown circuit breaker (logic only, no DB)
# --------------------------------------------------------------------------- #


class TestDrawdownMultiplier:
    """`_get_drawdown_multiplier` interpolates between reduce/pause thresholds."""

    def _agent_with_picks(self, picks_data):
        """Return an agent whose DB session yields the supplied SavedPick stubs."""
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.config = MagicMock()
        agent.config.get.side_effect = lambda k, default=None: {
            "models.drawdown_lookback_picks": 30,
            "models.drawdown_reduce_threshold": -0.10,
            "models.drawdown_pause_threshold": -0.30,
        }.get(k, default)

        @dataclass
        class _Pick:
            result: str
            odds: float
            kelly_stake_percentage: float

        picks = [_Pick(**p) for p in picks_data]

        agent.db = MagicMock()
        sess = MagicMock()
        sess.__enter__ = lambda s: s
        sess.__exit__ = MagicMock(return_value=False)
        sess.query.return_value.filter.return_value.order_by.return_value \
            .limit.return_value.all.return_value = picks
        agent.db.get_session.return_value = sess
        return agent

    def test_full_multiplier_when_profitable(self):
        agent = self._agent_with_picks([
            {"result": "win", "odds": 2.0, "kelly_stake_percentage": 2.0},
        ] * 15)
        assert agent._get_drawdown_multiplier() == 1.0

    def test_paused_when_below_pause_threshold(self):
        # 15 losses in a row → ROI = -100%, way below -30% pause
        agent = self._agent_with_picks([
            {"result": "loss", "odds": 2.0, "kelly_stake_percentage": 2.0},
        ] * 15)
        assert agent._get_drawdown_multiplier() == 0.0

    def test_scaled_in_between(self):
        # 10 wins @ 2.0 odds + 5 losses → +10pp profit on 30 staked = +33% ROI
        # Need a mix that lands between -10% and -30% ROI.
        # 2 wins @ 2.0 (+2 each = +4) + 8 losses (-8) = -4 / 10 = -40% ROI → paused
        # Try: 4 wins @ 2.0 (+4) + 6 losses (-6) = -2 / 10 = -20% ROI → between
        agent = self._agent_with_picks(
            [{"result": "win", "odds": 2.0, "kelly_stake_percentage": 1.0}] * 4
            + [{"result": "loss", "odds": 2.0, "kelly_stake_percentage": 1.0}] * 6
        )
        m = agent._get_drawdown_multiplier()
        # ROI = -20%; reduce_at=-10%, pause_at=-30%
        # multiplier = (-0.20 - (-0.30)) / (-0.10 - (-0.30)) = 0.10 / 0.20 = 0.50
        assert 0.0 < m < 1.0
        assert m == pytest.approx(0.50, abs=0.05)

    def test_too_few_picks_returns_full(self):
        agent = self._agent_with_picks([
            {"result": "loss", "odds": 2.0, "kelly_stake_percentage": 2.0},
        ] * 5)  # < 10 picks needed
        assert agent._get_drawdown_multiplier() == 1.0


# --------------------------------------------------------------------------- #
# Over-line monotonicity (regression guard for ensemble.py issue #14)
# --------------------------------------------------------------------------- #


class TestOverLineMonotonicity:
    """Verify ensemble.predict() respects over_1.5 ≥ over_2.5 ≥ over_3.5."""

    def test_monotonicity_with_high_overs(self):
        from src.models.ensemble import EnsemblePredictor

        ep = EnsemblePredictor.__new__(EnsemblePredictor)
        ep.config = MagicMock()
        ep.config.get.side_effect = lambda k, default=None: {
            "models.bookmaker_blend_weight": 0.40,
            "models.goals_ml_blend_weight": 0.25,
            "models.extreme_confidence_ceiling": 0.90,
            "models.intl_goals_dampen": 0.30,
        }.get(k, default)
        ep.calibration_factors = {"poisson": 1.0, "elo": 1.0, "ml": 1.0}
        ep.bayesian_weights = MagicMock()
        ep.bayesian_weights.get_weights.return_value = {
            "poisson": 0.30, "elo": 0.25, "ml": 0.45
        }
        ep.poisson = MagicMock()
        ep.poisson._team_strengths = {1: True, 2: True}
        ep.poisson.predict.return_value = {
            "home_win": 0.60, "draw": 0.25, "away_win": 0.15,
            "over_1.5": 0.95, "over_2.5": 0.75, "over_3.5": 0.40,
            "under_2.5": 0.25, "btts_yes": 0.65, "btts_no": 0.35,
            "home_over_1.5": 0.55, "away_over_1.5": 0.35,
            "home_xg": 2.0, "away_xg": 1.2,
            "most_likely_score": "2-1",
        }
        ep.elo = MagicMock()
        ep.elo.ratings = {1: 1600, 2: 1500}
        ep.elo.predict.return_value = {"home_win": 0.55, "draw": 0.27, "away_win": 0.18}
        ep.ml_models = MagicMock()
        ep.ml_models.is_fitted = False
        ep.goals_model = MagicMock()
        ep.goals_model.is_fitted = False
        ep.INTERNATIONAL_LEAGUES = set()

        result = ep.predict(1, 2)
        ens = result["ensemble"]
        assert ens["over_1.5"] >= ens["over_2.5"] >= ens["over_3.5"]
        # And the corresponding under lines should sum to 1 with their over twin
        assert ens["over_1.5"] + ens["under_1.5"] == pytest.approx(1.0, abs=1e-3)
        assert ens["over_2.5"] + ens["under_2.5"] == pytest.approx(1.0, abs=1e-3)
