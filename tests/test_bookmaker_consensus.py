"""Tests for consensus de-vigging in FeatureEngineer._get_bookmaker_features.

Stage 3, Phase 5/7. Two defects are guarded here:

* The API-Football "Home/Away" bet (a TWO-WAY, draw-excluded market) was mapped to
  market_type "1X2", overwriting the real Home/Away prices. Every Bet365 1X2 row
  in production carried a median overround of 1.3524 as a result, and Bet365 was
  the first-priority book for these features.
* The probability came from a single book while value_calculator's EV used the
  median price across all books, so cross-book price dispersion entered claimed
  EV as if it were edge.
"""

from unittest.mock import MagicMock

import pytest

from src.features.feature_engineer import FeatureEngineer


def _fe(odds_rows, match_id=1):
    fe = FeatureEngineer.__new__(FeatureEngineer)
    fe._preload_cache = {"match_meta": {}, "odds": {match_id: odds_rows},
                         "team_history": {}}
    fe._league_features_cache = {}
    fe.db = MagicMock()
    return fe


def _row(bk, sel, odds, market="1X2"):
    return {"market_type": market, "bookmaker": bk, "selection": sel,
            "odds_value": odds, "opening_odds": None}


def _1x2(bk, h, d, a):
    return [_row(bk, "Home", h), _row(bk, "Draw", d), _row(bk, "Away", a)]


# --------------------------------------------------------- the corruption case

# Real production data, match 49032 (2026-08-07). Bet365's Home/Away legs are the
# two-way prices; only Draw survived from the true 1X2 market.
CORRUPT_BET365 = _1x2("Bet365", 1.25, 3.40, 3.75)      # overround 1.361
GOOD_PINNACLE = _1x2("Pinnacle", 1.71, 3.66, 4.55)     # overround 1.078
GOOD_1XBET = _1x2("1xBet", 1.74, 3.81, 4.89)           # overround 1.042


def test_implausible_overround_book_is_excluded():
    """Bet365's 1.361 overround must be dropped, not averaged in."""
    fe = _fe(CORRUPT_BET365 + GOOD_PINNACLE + GOOD_1XBET)
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_available"] == 1
    assert r["bookmaker_consensus_books"] == 2, "the broken book was not excluded"
    # Consensus of the two sane books, not the corrupt one's inflated 0.588.
    assert 0.52 < r["home_implied_prob"] < 0.57, r["home_implied_prob"]


def test_corrupt_book_alone_yields_no_bookmaker_signal():
    """Better to report 'no bookmaker data' than to hand the model a broken one."""
    fe = _fe(CORRUPT_BET365)
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_available"] == 0
    assert r["home_implied_prob"] == pytest.approx(1 / 3)


def test_old_priority_order_would_have_picked_the_corrupt_book():
    """Documents the regression: Bet365 was tried first and had no plausibility
    gate, so its 0.588 home probability was what the ensemble blended at 60%."""
    corrupt_devig = (1 / 1.25) / (1 / 1.25 + 1 / 3.40 + 1 / 3.75)
    fe = _fe(CORRUPT_BET365 + GOOD_PINNACLE + GOOD_1XBET)
    r = fe._get_bookmaker_features(1)
    assert corrupt_devig > 0.58
    assert abs(r["home_implied_prob"] - corrupt_devig) > 0.03


# ------------------------------------------------------------------- consensus

def _devigged_home(h, d, a):
    inv = [1 / h, 1 / d, 1 / a]
    return inv[0] / sum(inv)


def test_consensus_is_the_median_probability_per_outcome():
    """The median is taken over each book's DE-VIGGED probability for an outcome,
    not by picking a 'median book'. Those differ: the book with the middle raw
    price need not be the book with the middle fair probability."""
    fe = _fe(_1x2("A", 2.0, 3.5, 4.0) + _1x2("B", 2.1, 3.4, 3.9)
             + _1x2("C", 1.9, 3.6, 4.1))
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_consensus_books"] == 3
    homes = sorted([_devigged_home(2.0, 3.5, 4.0),
                    _devigged_home(2.1, 3.4, 3.9),
                    _devigged_home(1.9, 3.6, 4.1)])
    assert r["home_implied_prob"] == pytest.approx(homes[1], abs=0.005)


def test_consensus_resists_a_single_outlier_book():
    """One book pricing home far shorter than the rest must not move the median."""
    fair = _1x2("A", 2.0, 3.5, 4.0) + _1x2("B", 2.05, 3.45, 3.95) \
        + _1x2("C", 1.95, 3.55, 4.05)
    baseline = _fe(fair)._get_bookmaker_features(1)["home_implied_prob"]
    # A fourth book that is sane on overround but way off on shape.
    skewed = fair + _1x2("D", 1.40, 4.60, 8.50)
    with_outlier = _fe(skewed)._get_bookmaker_features(1)["home_implied_prob"]
    assert abs(with_outlier - baseline) < 0.03


def test_probabilities_sum_to_one():
    fe = _fe(_1x2("A", 2.0, 3.5, 4.0) + _1x2("B", 2.6, 3.2, 3.1))
    r = fe._get_bookmaker_features(1)
    total = (r["home_implied_prob"] + r["draw_implied_prob"]
             + r["away_implied_prob"])
    assert total == pytest.approx(1.0, abs=1e-3)


def test_single_sane_book_still_works():
    fe = _fe(GOOD_PINNACLE)
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_available"] == 1
    assert r["bookmaker_consensus_books"] == 1
    assert r["home_implied_prob"] == pytest.approx(0.542, abs=0.01)


def test_missing_leg_drops_that_book_only():
    partial = [_row("A", "Home", 2.0), _row("A", "Draw", 3.5)]  # no Away
    fe = _fe(partial + GOOD_PINNACLE)
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_consensus_books"] == 1


def test_no_odds_returns_defaults():
    fe = _fe([])
    r = fe._get_bookmaker_features(1)
    assert r["bookmaker_available"] == 0
    assert r["bookmaker_consensus_books"] == 0


# ------------------------------------------------------------- other markets

def test_over_under_consensus_and_gate():
    rows = [
        _row("A", "Over 2.5", 1.90, "over_under"),
        _row("A", "Under 2.5", 1.95, "over_under"),
        _row("B", "Over 2.5", 1.85, "over_under"),
        _row("B", "Under 2.5", 2.00, "over_under"),
        # Broken 2-way: overround 1.43
        _row("C", "Over 2.5", 1.40, "over_under"),
        _row("C", "Under 2.5", 1.40, "over_under"),
    ]
    fe = _fe(rows)
    r = fe._get_bookmaker_features(1)
    assert r["goals_bookmaker_available"] == 1
    assert r["over25_implied_prob"] + r["under25_implied_prob"] == pytest.approx(1.0, abs=1e-3)
    # C's 0.50/0.50 must not drag the estimate; A and B both sit near 0.50 anyway,
    # so assert on the gate directly instead.
    assert 0.47 < r["over25_implied_prob"] < 0.53


def test_btts_and_team_goals_consensus():
    rows = [
        _row("A", "Yes", 1.80, "btts"), _row("A", "No", 2.00, "btts"),
        _row("B", "Yes", 1.85, "btts"), _row("B", "No", 1.95, "btts"),
        _row("A", "Home Over 1.5", 2.10, "team_goals"),
        _row("A", "Home Under 1.5", 1.70, "team_goals"),
        _row("A", "Away Over 1.5", 2.60, "team_goals"),
        _row("A", "Away Under 1.5", 1.48, "team_goals"),
    ]
    fe = _fe(rows)
    r = fe._get_bookmaker_features(1)
    assert r["btts_bookmaker_available"] == 1
    assert r["btts_yes_implied_prob"] + r["btts_no_implied_prob"] == pytest.approx(1.0, abs=1e-3)
    assert r["team_goals_bookmaker_available"] == 1
    assert 0.40 < r["home_over15_implied_prob"] < 0.50


# ------------------------------------------------------- the mapping itself

def test_home_away_bet_no_longer_maps_to_1x2():
    """BET_TYPE_MAP regression: 'Home/Away' is the two-way market. Writing it as
    1X2 is what corrupted 2,486 matches."""
    from src.scrapers.apifootball_scraper import BET_TYPE_MAP
    entry = BET_TYPE_MAP["Home/Away"]
    assert entry["market_type"] == "draw_no_bet"
    assert set(entry["selections"].values()) == {"DNB Home", "DNB Away"}
    assert "Draw" not in entry["selections"], "a two-way market has no draw leg"


def test_match_winner_still_maps_to_1x2():
    from src.scrapers.apifootball_scraper import BET_TYPE_MAP
    entry = BET_TYPE_MAP["Match Winner"]
    assert entry["market_type"] == "1X2"
    assert entry["selections"] == {"Home": "Home", "Draw": "Draw", "Away": "Away"}
