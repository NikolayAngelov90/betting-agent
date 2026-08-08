"""Stage 4, Phase 1 — the seven required market-separation tests, plus the
writer-level guards that make the class of bug impossible rather than merely absent.

Incident under test: API-Football's ``"Home/Away"`` bet (TWO-WAY, draw excluded)
was mapped to ``market_type: "1X2"`` with the same Home/Away labels, so it
overwrote genuine three-way prices through the odds table's
ON CONFLICT DO UPDATE. Measured blast radius: 13,274 rows, 2,548 matches, seven
bookmakers, and 92% of all saved picks built on the resulting probabilities.
"""

import pytest

from src.data.market_spec import (
    AUTHORITATIVE_BETS,
    MARKET_SPECS,
    MarketValidationError,
    check_overround,
    devig,
    extract_legs,
    get_spec,
    is_authoritative,
    overround,
    validate_write,
)
from src.scrapers.apifootball_scraper import BET_TYPE_MAP


# ─────────────────────────────────────────── Test 1
def test_1_home_away_must_never_populate_1x2():
    """The exact regression. 'Home/Away' is a two-way market and may not be
    written as 1X2 by any route."""
    entry = BET_TYPE_MAP["Home/Away"]
    assert entry["market_type"] == "draw_no_bet", (
        "'Home/Away' is the draw-excluded two-way market; mapping it to 1X2 "
        "overwrites the real three-way prices")
    assert "Draw" not in entry["selections"]

    # ...and the writer guard refuses it independently of the map.
    ok, why = validate_write("1X2", "Home/Away", ["Home", "Away"])
    assert not ok
    assert "not authoritative" in why

    assert not is_authoritative("1X2", "Home/Away")
    assert is_authoritative("draw_no_bet", "Home/Away")


# ─────────────────────────────────────────── Test 2
def test_2_1x2_must_always_contain_three_outcomes():
    spec = get_spec("1X2")
    assert spec.arity == 3
    assert not spec.overlapping

    full = {"Home": 2.0, "Draw": 3.5, "Away": 4.0}
    assert extract_legs("1X2", full) == [2.0, 3.5, 4.0]

    # Any missing leg yields None — a partial market is never completed silently.
    for missing in ("Home", "Draw", "Away"):
        partial = {k: v for k, v in full.items() if k != missing}
        assert extract_legs("1X2", partial) is None, f"accepted 1X2 without {missing}"

    ok, why = check_overround("1X2", [2.0, 3.5])   # only two legs
    assert not ok and "needs 3 legs" in why


# ─────────────────────────────────────────── Test 3
def test_3_a_bookmaker_may_offer_both_markets_and_both_survive():
    """1X2 and draw_no_bet are different market_types, so their rows occupy
    different keys in the unique index and cannot collide."""
    assert BET_TYPE_MAP["Match Winner"]["market_type"] == "1X2"
    assert BET_TYPE_MAP["Home/Away"]["market_type"] == "draw_no_bet"
    assert BET_TYPE_MAP["Draw No Bet"]["market_type"] == "draw_no_bet"

    # Same fixture, same bookmaker, both markets — distinct storage keys.
    keys_1x2 = {("Bet365", "1X2", s)
                for s in BET_TYPE_MAP["Match Winner"]["selections"].values()}
    keys_dnb = {("Bet365", "draw_no_bet", s)
                for s in BET_TYPE_MAP["Home/Away"]["selections"].values()}
    assert keys_1x2.isdisjoint(keys_dnb)

    # Both remain independently valid.
    ok3, _ = check_overround("1X2", [1.71, 3.66, 4.55])
    ok2, _ = check_overround("draw_no_bet", [1.38, 3.30])
    assert ok3 and ok2


# ─────────────────────────────────────────── Test 4
def test_4_genuine_1x2_overround_is_plausible_and_corrupt_one_is_not():
    # Real Pinnacle/1xBet books from production, match 49032.
    for prices in ([1.71, 3.66, 4.55], [1.74, 3.81, 4.89], [2.0, 3.5, 4.0]):
        ok, why = check_overround("1X2", prices)
        assert ok, f"rejected a genuine book {prices}: {why}"
        assert 1.0 < overround(prices) < 1.15

    # The corrupted Bet365 row from the same match: Home/Away legs are the
    # two-way prices, Draw survived from the real market -> overround 1.361.
    corrupt = [1.25, 3.40, 3.75]
    assert overround(corrupt) == pytest.approx(1.361, abs=0.002)
    ok, why = check_overround("1X2", corrupt)
    assert not ok and "overround" in why


# ─────────────────────────────────────────── Test 5
def test_5_two_way_probabilities_sum_to_one_after_devig():
    for market, prices in [
        ("draw_no_bet", [1.38, 3.30]),
        ("btts", [1.80, 2.00]),
        ("over_under", [1.90, 1.95]),
        ("team_goals", [2.10, 1.70]),
    ]:
        probs = devig(market, prices)
        assert probs is not None, f"{market} rejected a valid book"
        assert len(probs) == 2
        assert sum(probs) == pytest.approx(1.0, abs=1e-9)
        assert all(0 < p < 1 for p in probs)


# ─────────────────────────────────────────── Test 6
def test_6_three_way_probabilities_sum_to_one_after_devig():
    probs = devig("1X2", [1.71, 3.66, 4.55])
    assert probs is not None
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0, abs=1e-9)
    # Sanity against the real market: this is a ~54% home favourite.
    assert 0.53 < probs[0] < 0.56


def test_6b_devig_refuses_a_corrupt_book_instead_of_normalising_it():
    """Normalising a broken market produces a plausible-LOOKING number, which is
    strictly worse than returning nothing."""
    assert devig("1X2", [1.25, 3.40, 3.75]) is None


def test_6c_overlapping_market_is_never_devigged():
    """Double chance legs overlap (1X, 12, X2 each cover two base outcomes), so
    the three inverse prices sum to ~2. Forcing them to sum to 1 would be
    meaningless — and treating that ~1.25 two-leg sum as an overround was a
    false-positive in the first pass of this audit."""
    spec = get_spec("double_chance")
    assert spec.overlapping
    assert devig("double_chance", [1.22, 1.30, 1.83]) is None
    ok, _ = check_overround("double_chance", [1.22, 1.30, 1.83])
    assert ok, "a genuine double-chance book must not be flagged as corrupt"


# ─────────────────────────────────────────── Test 7
def test_7_duplicate_rows_cannot_silently_overwrite_another_market():
    # A single write may not contain the same selection twice.
    ok, why = validate_write("1X2", "Match Winner", ["Home", "Draw", "Home"])
    assert not ok and "duplicate selections" in why

    # Every declared market has an explicit authority list, so no unlisted bet
    # can reach it.
    for market_type in MARKET_SPECS:
        assert market_type in AUTHORITATIVE_BETS, (
            f"{market_type} has no declared authoritative bets — any bet could "
            f"write it")

    # And no bet name is authoritative for two different market types, which is
    # what would let one payload write the same key twice.
    seen = {}
    for market_type, bets in AUTHORITATIVE_BETS.items():
        for bet in bets:
            assert bet not in seen, (
                f"bet {bet!r} is authoritative for both {seen[bet]!r} and "
                f"{market_type!r}")
            seen[bet] = market_type


# ───────────────────────────────── writer-level collision guard
def test_bet_type_map_has_no_market_selection_collisions():
    """Two different bets must not map to the same (market_type, selection).
    This is the invariant whose violation caused the incident."""
    owner = {}
    for bet_name, mapping in BET_TYPE_MAP.items():
        mt = mapping["market_type"]
        for stored in mapping["selections"].values():
            key = (mt, stored)
            if key in owner and owner[key] != bet_name:
                # Same market from two bet names is only acceptable when they are
                # genuinely the same market (Draw No Bet / Home/Away).
                both = {owner[key], bet_name}
                assert both == {"Draw No Bet", "Home/Away"}, (
                    f"{key} written by both {owner[key]!r} and {bet_name!r} — "
                    f"one will overwrite the other")
            owner.setdefault(key, bet_name)


def test_every_mapped_bet_is_authoritative_for_its_market():
    """Guards the map against drifting away from the spec."""
    for bet_name, mapping in BET_TYPE_MAP.items():
        mt = mapping["market_type"]
        assert is_authoritative(mt, bet_name), (
            f"BET_TYPE_MAP maps {bet_name!r} -> {mt!r} but market_spec does not "
            f"list it as authoritative. Update AUTHORITATIVE_BETS deliberately, "
            f"or the mapping is wrong.")


def test_overround_rejects_impossible_prices():
    for bad in ([1.0, 3.5, 4.0], [0.5, 3.5, 4.0], [None, 3.5, 4.0], []):
        with pytest.raises(MarketValidationError):
            overround(bad)


def test_extract_legs_handles_lines_and_sides():
    ou = {"Over 2.5": 1.90, "Under 2.5": 1.95, "Over 1.5": 1.30, "Under 1.5": 3.40}
    assert extract_legs("over_under", ou, line="2.5") == [1.90, 1.95]
    assert extract_legs("over_under", ou, line="1.5") == [1.30, 3.40]
    assert extract_legs("over_under", ou, line="3.5") is None

    tg = {"Home Over 1.5": 2.10, "Home Under 1.5": 1.70,
          "Away Over 1.5": 2.60, "Away Under 1.5": 1.48}
    assert extract_legs("team_goals", tg, line="1.5", side="Home") == [2.10, 1.70]
    assert extract_legs("team_goals", tg, line="1.5", side="Away") == [2.60, 1.48]
