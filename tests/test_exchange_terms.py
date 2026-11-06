"""EXCHANGE TERMS: the size that does not exist, and the band we never bet.

Measured 2026-11-06 by `scripts/exchange_terms.py` on 1,639 settled paper
picks over 70 days.

THE TWO FINDINGS THIS FILE DEFENDS, because both are the kind that get
quietly rounded into something more convenient:

  1. **UNESTABLISHED is not REFUTED.** The exchange price has no measurable
     depth, so its 0.59pp cost is not established as takeable -- the same
     status the best line has, for the same reason, third instance. A later
     reader must not find "the exchange was negative" (it was not measured)
     nor "the exchange was +3.15pp" (that figure has a missing term).

  2. **The anomaly's band is DISJOINT from ours, by configuration.**
     `betting.min_odds: 1.50` is 66.7% implied; the anomaly is documented at
     70-79%. 0 of 1,639 picks sit in it and none can. So the whole netting
     exercise is about a band the deployed config excludes, and harvesting
     the anomaly is a SELECTION change, not a pricing change. That distinction
     is the difference between "there is a path" and "there is a different
     project", and it is derived here from the config rather than asserted.

NOTE ON IMPORTS. `scripts.exchange_terms` keeps every database and
market_spec import INSIDE its functions, so importing it here reaches nothing
on the capture path and this file stays out of the capture gate.
"""
import ast
import pathlib

import pytest
import yaml

from scripts.cost_by_book import DOCUMENTED_ANOMALY, MEASURED_BOOKS
from scripts.exchange_terms import (COMMISSION_PUBLISHED, COMMISSION_RANGE,
                                    MEASURED_TERMS, POWER_DAYS_AT_CURRENT_RATE,
                                    POWER_N_FOR_TARGET_MOVE, commission_cost)


# -- HARD RULE 2: NO MEASURABLE SIZE MEANS NOT ESTABLISHED ----------------

def test_SIZE_IS_UNAVAILABLE_and_the_verdict_says_UNESTABLISHED():
    assert MEASURED_TERMS["size_available"] is False
    v = MEASURED_TERMS["size_verdict"]
    assert "UNAVAILABLE" in v
    assert "NOT ESTABLISHED AS\nTAKEABLE" in v or "NOT ESTABLISHED" in v
    assert "UNESTABLISHED, not refuted" in v, (
        "the size verdict no longer distinguishes UNESTABLISHED from "
        "REFUTED. Only one of those ends the profitability question, and "
        "collapsing them is how a missing measurement becomes a conclusion")
    assert "THIRD instance" in v, (
        "the verdict no longer records this as the third instance of the "
        "not-takeable error class, which is the reason it has a hard rule")


def test_the_SIZE_SEARCH_NAMES_WHERE_IT_LOOKED():
    """An absence claim must say what was searched, or it is unfalsifiable."""
    searched = " ".join(MEASURED_TERMS["size_searched"]).lower()
    for place in ("odds.", "odds_snapshots", "theodds_scraper"):
        assert place.lower() in searched, (
            f"the size search no longer names {place}; a 'no such field' "
            f"claim that does not say where it looked cannot be checked")


def _columns_of(class_name: str) -> set:
    """The COLUMN NAMES of an ORM class, parsed rather than text-matched.

    WHY AST AND NOT RAW TEXT. A substring search over the class body answers
    a different question: a comment or docstring mentioning `liquidity` would
    satisfy it, and the raw-match ratchet refuses new negative text
    assertions for exactly that reason. `code_of` is the register's usual
    instrument, but importing it from `tests.conftest` would pull this file
    into the CAPTURE GATE (conftest reaches `src.data.odds_quota`), and this
    stage has no business on the capture path. Parsing is immune to both.
    """
    tree = ast.parse(
        pathlib.Path("src/data/models.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            cols = set()
            for stmt in node.body:
                if not isinstance(stmt, ast.Assign):
                    continue
                call = stmt.value
                if not (isinstance(call, ast.Call)
                        and getattr(call.func, "id", "") == "Column"):
                    continue
                for t in stmt.targets:
                    if isinstance(t, ast.Name):
                        cols.add(t.id)
            return cols
    raise AssertionError(f"no class {class_name} in src/data/models.py")


def test_KNOWN_WRONG_THE_SCHEMA_REALLY_HAS_NO_SIZE_FIELD():
    """The control for the absence claim: enumerate the columns that DO
    exist, so 'no size field' is a measurement of the schema and not a
    statement about it."""
    cols = _columns_of("Odds")
    for present in ("odds_value", "opening_odds", "first_seen_at"):
        assert present in cols, (
            f"`{present}` is no longer a column on Odds, so this control is "
            f"reading the wrong class and proves nothing. Found: "
            f"{sorted(cols)}")
    sizeish = {c for c in cols
               if any(w in c.lower() for w in
                      ("size", "liquid", "depth", "stake", "matched",
                       "available", "volume"))}
    assert not sizeish, (
        f"these size-like columns now EXIST on Odds: {sorted(sizeish)}. Size "
        f"may be measurable after all -- re-run the stage before relying on "
        f"the UNESTABLISHED verdict")


# -- THE BANDS ARE DISJOINT, AND IT IS DERIVED FROM THE CONFIG -----------

def test_the_ANOMALYS_BAND_IS_BELOW_OUR_CONFIGURED_FLOOR():
    """Derived from the deployed config, not asserted.

    `min_odds` is a price floor, so it is a CEILING on implied probability:
    1/1.50 = 66.7%. The anomaly starts at 70%. The gap is 3.3 points of
    implied probability and nothing we bet can cross it.
    """
    cfg = yaml.safe_load(
        pathlib.Path("config/config.yaml").read_text(encoding="utf-8"))
    live = cfg["betting"]["min_odds"]
    assert live == MEASURED_TERMS["config_min_odds"], (
        f"config/config.yaml now has betting.min_odds = {live}, but this "
        f"stage measured against {MEASURED_TERMS['config_min_odds']}. If the "
        f"floor moved, the disjointness result must be re-measured")
    our_ceiling = 100.0 / live
    band_lo, band_hi = MEASURED_TERMS["anomaly_band_implied_pct"]
    assert our_ceiling == pytest.approx(66.67, abs=0.02)
    assert our_ceiling < band_lo, (
        f"our implied-probability ceiling is {our_ceiling:.1f}% and the "
        f"anomaly's band starts at {band_lo}%. They no longer fail to "
        f"overlap, so the central finding of this stage has changed")
    assert band_lo - our_ceiling == pytest.approx(3.33, abs=0.02)
    assert band_hi > band_lo


def test_WE_HOLD_ZERO_PICKS_IN_THE_ANOMALYS_BAND():
    assert MEASURED_TERMS["picks_in_anomaly_band"] == 0, (
        "we now hold picks in the anomaly's band. That changes the stage's "
        "conclusion from 'a different selection' to 'a measurable exposure'")
    assert MEASURED_TERMS["our_min_odds"] >= 1.50
    assert MEASURED_TERMS["our_median_odds"] > 1.429, (
        "our median price is now inside the anomaly's band; re-measure")


def test_the_BAND_VERDICT_CALLS_IT_A_SELECTION_CHANGE_not_a_pricing_one():
    """The distinction the whole directive turns on."""
    v = MEASURED_TERMS["band_verdict"]
    assert "DISJOINT" in v
    assert "NOT a pricing change" in v
    assert "SELECTION" in v, (
        "the band verdict no longer says that harvesting the anomaly needs a "
        "different selection. That is the difference between a price fix and "
        "a different project")
    assert "min_odds" in v and "66.7" in v, (
        "the verdict no longer carries the mechanism and the figure, so a "
        "reader cannot check it")


# -- HARD RULE 1: A COMMISSION COMES FROM A PUBLISHED SCHEDULE -----------

def test_EVERY_COMMISSION_HAS_A_URL_A_DATE_AND_A_BASIS():
    for venue, c in COMMISSION_PUBLISHED.items():
        assert c["url"].startswith("https://"), venue
        assert "2026-11-06" in c["provenance"], venue
        assert "published" in c["provenance"], venue
        assert "net winnings" in c["basis"], (
            f"{venue}: the commission basis no longer says what it is "
            f"charged on. Per-bet and per-market net winnings are different "
            f"costs and the conversion depends on which it is")
        assert c["caveat"], (
            f"{venue}: the caveat is empty. Every rate here has a limit on "
            f"what it establishes and that limit must travel with it")


def test_the_DIRECTLY_FETCHED_RATE_IS_MARKED_APART_from_the_one_that_is_not():
    """Hard rule 7: provenance is per figure, and these two differ."""
    direct = MEASURED_TERMS["commission_fetched_directly"]
    assert "Betfair Exchange" in direct
    assert "Matchbook" not in direct, (
        "Matchbook is now marked as directly fetched. It was not: every "
        "matchbook.com host fails TLS verification from this environment, "
        "and the figure came from a search of their own domain")
    assert "NOT directly fetched" in COMMISSION_PUBLISHED["Matchbook"][
        "provenance"]
    assert "DIRECTLY FETCHED" in COMMISSION_PUBLISHED["Betfair Exchange"][
        "provenance"]


def test_the_COMMISSION_FORMULA_is_on_NET_WINNINGS_not_on_turnover():
    """A commission on winnings is not a margin. At zero commission the
    effective price must be the quoted price exactly -- that is the control."""
    odds, wr = 1.770, 0.56
    base = commission_cost(odds, wr, 0.0)
    assert base["effective_odds"] == pytest.approx(odds, abs=1e-12)
    assert base["pp"] == pytest.approx(0.0, abs=1e-12)
    assert base["roi"] == pytest.approx(0.0, abs=1e-12)
    # and it must bite only on the profit, never on the stake
    two = commission_cost(odds, wr, 0.02)
    assert two["effective_odds"] == pytest.approx(
        1.0 + (odds - 1.0) * 0.98, abs=1e-12)
    assert two["effective_odds"] > 1.0
    assert 0 < two["pp"] < 1.0
    # THE TWO UNITS DISAGREE ABOUT WHICH PRICE COMMISSION HURTS MOST, and
    # that is not a contradiction -- it is the conversion.
    #
    # In ROI, commission costs `win_rate * c * (o-1)`: strictly increasing in
    # the price, because a longer shot has more profit to take a share of.
    assert (commission_cost(3.0, wr, 0.02)["roi"]
            > commission_cost(1.77, wr, 0.02)["roi"]
            > commission_cost(1.30, wr, 0.02)["roi"])
    # In PROBABILITY POINTS it is about `c*(o-1)/o**2`, which peaks at o=2
    # and falls away on both sides. So commission is worst, in the unit the
    # band costs are in, at EVEN MONEY -- not at long odds.
    peak = commission_cost(2.0, wr, 0.02)["pp"]
    assert peak > commission_cost(1.30, wr, 0.02)["pp"]
    assert peak > commission_cost(5.00, wr, 0.02)["pp"]
    assert peak == pytest.approx(0.5, abs=0.01), (
        f"commission at even money is {peak:.3f}pp, no longer the c/4 = "
        f"0.5pp the algebra gives; the formula has changed")


def test_the_CONCLUSION_HOLDS_ACROSS_THE_WHOLE_COMMISSION_RANGE():
    """The unobtained rate must not be load-bearing.

    The published rates span 2% to 6% and the break-even is 15.6%. If the
    verdict flipped somewhere inside that span, the missing EU rate would
    decide the stage and the stage could not be reported at all.
    """
    assert COMMISSION_RANGE[0] == 0.02 and COMMISSION_RANGE[-1] == 0.156
    oc = MEASURED_TERMS["our_cost_pp"]
    policy = oc["TAKEN_current_policy"]
    odds = 1.770
    for rate in (0.02, 0.04, 0.06):
        allin = oc["TheOddsAPI-matchbook"] + commission_cost(
            odds, 0.56, rate)["pp"]
        assert allin < policy, (
            f"at {100 * rate:.0f}% commission the exchange all-in cost "
            f"{allin:.2f}pp is no longer below the current policy's "
            f"{policy:.2f}pp, so the published-rate range now spans the "
            f"verdict and the missing EU rate becomes decisive")
    # and at the break-even rate it must NOT clear the anomaly -- that is
    # what break-even means, and it is the figure's own definition
    be = MEASURED_TERMS["breakeven_commission_matchbook"]
    band = MEASURED_BOOKS["band_cost_pp"]["TheOddsAPI-matchbook"]
    allin = band + commission_cost(1.370, 0.73, be)["pp"]
    assert allin == pytest.approx(DOCUMENTED_ANOMALY["value_pp"], abs=0.25), (
        f"the break-even commission no longer lands the all-in band cost on "
        f"the anomaly ({allin:.2f}pp vs {DOCUMENTED_ANOMALY['value_pp']}pp), "
        f"so {100 * be:.1f}% is not the break-even rate any more")


# -- HARD RULE 5: COVERAGE, AND WHAT THE SUBSET IS ABOUT -----------------

def test_EVERY_COST_HAS_A_COVERAGE():
    assert set(MEASURED_TERMS["our_cost_pp"]) == set(
        MEASURED_TERMS["takeable_cover_pick_share"])


def test_the_EXCHANGE_SUBSET_IS_A_DIFFERENT_POPULATION_not_a_random_fifth():
    """Coverage alone would read as 'one pick in five'. It is not: the
    exchanges quote NONE of four markets holding over half our picks."""
    absent = MEASURED_TERMS["markets_absent_from_exchanges"]
    full = MEASURED_TERMS["market_mix_full"]
    mb = MEASURED_TERMS["market_mix_matchbook"]
    bf = MEASURED_TERMS["market_mix_betfair"]
    for fam in absent:
        assert fam in full, fam
        assert fam not in mb and fam not in bf, (
            f"{fam} is now quoted by an exchange; the different-population "
            f"finding needs re-measuring")
    share = sum(full[f] for f in absent)
    assert share == pytest.approx(
        MEASURED_TERMS["share_of_picks_in_absent_markets"], abs=0.005), (
        f"the absent markets now hold {share:.3f} of picks, not the recorded "
        f"{MEASURED_TERMS['share_of_picks_in_absent_markets']}")
    assert share > 0.50, (
        "the markets no exchange quotes are no longer most of our picks, "
        "which is the form the finding was recorded in")
    # and the mix within what IS covered is skewed, not proportional
    assert mb["over_under"] > 1.7 * full["over_under"], (
        "matchbook's covered subset is no longer over-weighted toward "
        "over_under, so 'a different population' needs restating")
    assert set(bf) == {"1X2"}, (
        f"Betfair Exchange now covers {sorted(bf)}; it was 1X2 only")


def test_the_MARKET_MIXES_SUM_TO_ONE():
    for key in ("market_mix_full", "market_mix_matchbook",
                "market_mix_betfair"):
        total = sum(MEASURED_TERMS[key].values())
        assert total == pytest.approx(1.0, abs=0.01), (
            f"{key} sums to {total:.3f}; a market mix that does not sum to 1 "
            f"has picks in no market or in two")


def test_the_DAYS_TO_A_DECISION_ARE_DERIVED_from_the_recorded_power_figure():
    """`n=525 picks = 28 days` -> 18.75 picks/day, rescaled by coverage.
    Not a chosen number."""
    assert POWER_N_FOR_TARGET_MOVE == 525
    assert POWER_DAYS_AT_CURRENT_RATE == 28
    rate_then = POWER_N_FOR_TARGET_MOVE / POWER_DAYS_AT_CURRENT_RATE
    assert rate_then == pytest.approx(18.75, abs=0.01)
    d = MEASURED_TERMS["days_for_525"]
    cov = MEASURED_TERMS["takeable_cover_pick_share"]
    # a policy on a fifth of the picks needs multiples of the days
    assert d["TheOddsAPI-matchbook_takeable"] > 5 * d["Bet365"], (
        "the exchange-restricted policy no longer needs multiples of the "
        "days a broad policy needs; the coverage cost has changed shape")
    assert d["TheOddsAPI-betfair_ex_eu_takeable"] > 365, (
        "Betfair-only no longer takes over a year to reach n=525")
    # monotone in coverage: less coverage, more days
    assert cov["Bet365"] > cov["Pinnacle"] > cov[
        "TheOddsAPI-matchbook"] > cov["TheOddsAPI-betfair_ex_eu"]
    assert d["Bet365"] < d["Pinnacle"] < d["TheOddsAPI-matchbook"]


# -- THE VERDICT, AND WHAT IT IS CAREFUL NOT TO CLAIM --------------------

def test_the_VERDICT_keeps_UNESTABLISHED_and_names_what_IS_established():
    v = MEASURED_TERMS["verdict"]
    assert "UNESTABLISHED, not positive" in v
    assert "52.6%" in v and "min_odds" in v
    assert "Pinnacle" in v and "2.53pp" in v and "3.70pp" in v, (
        "the verdict no longer names the one established saving. A stage "
        "that reports only what failed loses the result that stands")


def test_a_COST_SAVING_IS_NOT_AN_EDGE():
    """The arithmetic that must not be misread. Pinnacle saves 1.17pp of
    COST on our own prices. That is not an inefficiency collected, and there
    is no documented inefficiency for our band at all."""
    oc = MEASURED_TERMS["our_cost_pp"]
    saving = oc["TAKEN_current_policy"] - oc["Pinnacle"]
    assert saving == pytest.approx(1.17, abs=0.02)
    # it is a saving, so it cannot exceed what we were paying
    assert saving < oc["TAKEN_current_policy"]
    # and it is smaller than the gap the +3% target needs, which is why it
    # is a cost result and not a profitability result
    assert saving < DOCUMENTED_ANOMALY["value_pp"], (
        "the established saving now exceeds the largest documented "
        "inefficiency; that would be a profitability claim and needs its own "
        "evidence bar, not this stage's")


def test_TWO_BOOKS_ARE_DEARER_THAN_OUR_CURRENT_POLICY():
    """The median policy is not the worst choice available, which is what
    makes 'switch to one book' a trade rather than a free win."""
    oc = MEASURED_TERMS["our_cost_pp"]
    dearer = [b for b, v in oc.items()
              if b != "TAKEN_current_policy" and v > oc["TAKEN_current_policy"]]
    assert set(dearer) == {"Bet365", "1xBet"}, (
        f"the books dearer than the current policy are now {sorted(dearer)}; "
        f"the recorded finding is that the two best-covered books are")
