"""COST BY BOOK: the units, the coverage, and the figure that is NOT a cost.

Measured 2026-11-06 by `scripts/cost_by_book.py` on 1,639 settled paper picks
over 1,629 fixtures, 29 books priced.

WHAT THIS FILE IS FOR. Three of the directive's hard rules are the kind that
fail silently a month later, so each one is an assertion here with a control:

  1. "A figure established as not takeable cannot become a planning cost in a
     different argument." The two cheapest books are EXCHANGES. Their
     commission is charged on net winnings and is NOT in the quoted price, so
     0.59pp is NOT Matchbook's cost -- it is its cost with a term missing.
     The same shape as the 1.1% best-line figure that H-MIX was built on.

  2. "Every cost states its unit, and its conversion if both appear." The
     anomaly is band-specific (70-79% implied = odds 1.27-1.43). Converting a
     band margin at the POPULATION's average odds of 1.627 overstates it by
     about 3pp and INVERTS the sign of the netting. I made exactly that error
     in this stage before catching it, so the correct conversion is pinned
     with the wrong one as its known-wrong control.

  3. "A cost is paired with its coverage." Asserted structurally: every book
     with a cost figure has a coverage figure.

NOTE ON IMPORTS. `scripts.cost_by_book` keeps every database and market_spec
import INSIDE its functions, so importing it here reaches nothing on the
capture path and this file stays out of the capture gate.
"""
import pytest

from scripts.cost_by_book import (ANOMALY_BAND, DOCUMENTED_ANOMALY,
                                  MEASURED_BOOKS, MEDIAN_POLICY, ROI_TARGET,
                                  _pp_at)


# -- HARD RULE 2: THE UNIT, AND THE CONVERSION ----------------------------

def test_the_BAND_COST_is_converted_at_the_BANDS_OWN_PRICE():
    """The conversion that is right, stated as the relation it must satisfy.

    `_pp_at(cost, odds)` = (1/odds) * cost/(1+cost) * 100. It DIVIDES by the
    price, so a band of short prices (1.33) yields FEWER probability points
    than the population average (1.627) for the same relative margin.
    """
    band_odds = 1.33
    for bm, pp in MEASURED_BOOKS["band_cost_pp"].items():
        if bm == "TAKEN_current_policy":
            continue
        rel = MEASURED_BOOKS["cost_rel"][bm]
        # the band margin is not the average margin, so this is a sanity
        # bound rather than an identity: the pp figure must be consistent
        # with SOME relative margin at the band's price, and that margin
        # must be within a factor of two of the book's average.
        implied_rel = None
        lo, hi = 0.0001, 0.60
        for _ in range(60):
            mid = (lo + hi) / 2
            if _pp_at(mid, band_odds) < pp:
                lo = mid
            else:
                hi = mid
            implied_rel = mid
        assert 0.5 * rel < implied_rel < 2.5 * rel, (
            f"{bm}: {pp}pp at odds {band_odds} implies a relative margin of "
            f"{100 * implied_rel:.2f}%, which is not within a factor of ~2 of "
            f"the book's measured average {100 * rel:.2f}%. One of the two "
            f"figures is in the wrong unit.")


def test_KNOWN_WRONG_CONVERTING_THE_BAND_AT_AVERAGE_ODDS_FLIPS_THE_SIGN():
    """The control. If the unit did not matter, pinning it would prove nothing.

    This is the error I actually made: I converted the band margin with
    `cost/(1+cost) * avg_odds * 100`, which both multiplies instead of
    dividing and uses the wrong price. It turned Pinnacle's netting from
    +0.05pp into -4.24pp -- a different conclusion, from the same measurement.
    """
    rel = MEASURED_BOOKS["cost_rel"]["Pinnacle"]
    band_rel = 0.0516          # Pinnacle's measured in-band relative margin
    right = MEASURED_BOOKS["band_cost_pp"]["Pinnacle"]
    wrong = band_rel / (1 + band_rel) * MEASURED_BOOKS["avg_taken_odds"] * 100
    anomaly = DOCUMENTED_ANOMALY["value_pp"]
    assert right == pytest.approx(3.69, abs=0.02)
    assert wrong > 7.0, (
        f"the wrong conversion no longer produces a materially different "
        f"figure ({wrong:.2f}pp), so this control has stopped controlling")
    assert anomaly - right > 0 > anomaly - wrong, (
        f"the unit error no longer flips the sign of the netting "
        f"(right: {anomaly - right:+.2f}pp, wrong: {anomaly - wrong:+.2f}pp), "
        f"so the reason this rule exists is no longer demonstrated")
    assert rel < band_rel, (
        "Pinnacle's in-band margin is no longer above its average margin; "
        "the finding that the favourites band is the DEAREST band needs "
        "re-measuring")


def test_the_BAND_is_the_ANOMALYS_OWN_and_is_DERIVED_from_its_statement():
    """70-79% implied is odds 1/0.79 .. 1/0.70. Not a chosen window."""
    lo, hi = ANOMALY_BAND
    assert lo == pytest.approx(1.0 / 0.79, abs=1e-9)
    assert hi == pytest.approx(1.0 / 0.70, abs=1e-9)
    assert "70-79%" in DOCUMENTED_ANOMALY["label"], (
        "the anomaly's label no longer states the band the ANOMALY_BAND "
        "constant is derived from, so the derivation is unverifiable")


# -- HARD RULE 1: A NOT-TAKEABLE FIGURE IS NOT A PLANNING COST ------------

def test_the_TWO_CHEAPEST_BOOKS_ARE_EXCHANGES_and_are_MARKED():
    """The whole reason line 3 is not the answer."""
    rank = sorted(MEASURED_BOOKS["cost_rel"].items(), key=lambda kv: kv[1])
    cheapest_two = {bm for bm, _ in rank[:2]}
    assert cheapest_two == set(MEASURED_BOOKS["exchanges"]), (
        f"the two cheapest books are now {cheapest_two}, not the two marked "
        f"as exchanges {set(MEASURED_BOOKS['exchanges'])} -- re-check which "
        f"of them quotes a price that already includes its own charge")
    v = MEASURED_BOOKS["verdict"]
    assert "EXCHANGES" in v and "NOT in the quoted price" in v
    assert "UNMEASURED" in v, (
        "the verdict no longer says the exchanges' commission is unmeasured; "
        "without that word 0.59pp reads as a cost")


def test_the_EXCHANGE_FIGURE_IS_NOT_THE_ANSWER_TO_LINE_2():
    """Line 2 asks for the cheapest book whose price is established takeable.

    An exchange's quoted margin is a cost with a term missing, so it cannot
    be the figure a decision is planned against -- that is the second
    instance of this exact move, after the 1.1% best-line figure.
    """
    ex = set(MEASURED_BOOKS["exchanges"])
    eligible = {
        bm: pp for bm, pp in MEASURED_BOOKS["band_cost_pp"].items()
        if bm not in ex and bm != "TAKEN_current_policy"
        and MEASURED_BOOKS["cover_pick_share"].get(bm, 0) >= 0.20
        and MEASURED_BOOKS["later_write_share"].get(bm, 1.0) == 0.0
    }
    assert eligible, "no book qualifies for line 2 at all; restate line 2"
    best = min(eligible, key=eligible.get)
    assert best == "Pinnacle", f"line 2's book is now {best}"
    assert eligible[best] == pytest.approx(3.69, abs=0.02)
    # and it must be DEARER than the exchange figure, or the rule is moot
    assert eligible[best] > min(
        MEASURED_BOOKS["band_cost_pp"][b] for b in ex), (
        "the eligible book is already cheaper than both exchanges, so "
        "excluding them changes nothing and this rule is untested here")


def test_the_BREAKEVEN_COMMISSION_is_reported_INSTEAD_of_an_assumed_rate():
    """Hard rule 6: no number without provenance.

    The exchanges' commission is not in our record, so the stage does not
    assume one. It reports the rate at which the anomaly is exactly erased
    and leaves the comparison to a real rate as Niki's.
    """
    be = MEASURED_BOOKS["breakeven_commission"]
    for bm in MEASURED_BOOKS["exchanges"]:
        assert be[bm] > 0.10, (
            f"{bm}'s break-even commission is {100 * be[bm]:.1f}%, no longer "
            f"far above any real exchange rate -- the claim that the "
            f"exchanges clear the anomaly ON PRICE now depends on the rate "
            f"and must be measured, not reasoned")
    # a book already dearer than the anomaly has a NEGATIVE break-even rate:
    # no commission discount can rescue it. That is the arithmetic's teeth.
    assert be["Pinnacle"] < be["TheOddsAPI-pinnacle"] < min(
        be[b] for b in MEASURED_BOOKS["exchanges"])


# -- HARD RULE 3: A COST IS PAIRED WITH ITS COVERAGE ----------------------

def test_EVERY_COST_FIGURE_HAS_A_COVERAGE_FIGURE():
    cost_keys = set(MEASURED_BOOKS["cost_rel"])
    cover_keys = set(MEASURED_BOOKS["cover_pick_share"])
    assert cost_keys == cover_keys, (
        f"these books have a cost with no coverage beside it: "
        f"{cost_keys - cover_keys}; and coverage with no cost: "
        f"{cover_keys - cost_keys}")
    for bm in cost_keys:
        assert 0.0 < MEASURED_BOOKS["cover_pick_share"][bm] <= 1.0


def test_the_CHEAPEST_BOOKS_ARE_THE_THINNEST_which_is_the_point():
    """Cost and coverage move against each other. Reporting one without the
    other is what makes a cheap book look like a cheap policy."""
    share = MEASURED_BOOKS["cover_pick_share"]
    cost = MEASURED_BOOKS["cost_rel"]
    cheap3 = sorted(cost, key=cost.get)[:3]
    dear3 = sorted(cost, key=cost.get)[-3:]
    assert max(share[b] for b in cheap3) < max(share[b] for b in dear3), (
        "the cheapest books are no longer thinner than the dearest ones, so "
        "the cost/coverage tension this stage reports has changed shape")


# -- THE FINDING: THE ANOMALY IS NEGATIVE AT ITS OWN SOURCE ---------------

def test_the_ANOMALY_IS_NEGATIVE_AT_THE_BOOK_IT_WAS_DOCUMENTED_AT():
    """1xBet is named in the anomaly's own label. Its in-band margin in our
    record is 5.77pp against a documented 3.74pp gap."""
    assert "1xBet" in DOCUMENTED_ANOMALY["label"], (
        "the anomaly's label no longer names the book it was measured at, "
        "which is the premise of this test")
    net = DOCUMENTED_ANOMALY["value_pp"] - MEASURED_BOOKS["band_cost_pp"]["1xBet"]
    assert net < 0, (
        f"the anomaly nets {net:+.2f}pp at 1xBet. The finding recorded for "
        f"2026-11-06 is that it is NEGATIVE at its own source; if that has "
        f"changed, the conclusion changes with it")
    assert net == pytest.approx(-2.03, abs=0.02)


def test_the_BOOKMAKERS_SHADE_THE_FAVOURITES_BAND_and_the_exchanges_do_not():
    """Why the netting fails: the inefficiency lives where the price is worst.

    A bookmaker's margin RISES in the favourites band -- by about a point for
    every real book here. An exchange's does not move at all, because nobody
    is shading anything: the price is two users' orders meeting.

    So the band premium is the measurable signature of bookmaker shading, and
    it lands exactly on the 70-79% band where the documented anomaly sits.
    That is the whole reason the subtraction comes out near zero or negative.
    """
    premium = {}
    for bm, pp in MEASURED_BOOKS["band_cost_pp"].items():
        if bm == "TAKEN_current_policy":
            continue
        avg_pp = _pp_at(MEASURED_BOOKS["cost_rel"][bm],
                        MEASURED_BOOKS["avg_taken_odds"])
        premium[bm] = pp - avg_pp
    ex = set(MEASURED_BOOKS["exchanges"])
    for bm in ex:
        assert abs(premium[bm]) < 0.10, (
            f"{bm} now shades the favourites band by {premium[bm]:+.2f}pp. "
            f"A flat margin across the probability range is what "
            f"distinguishes an exchange here; if it has a band premium, it "
            f"is behaving like a bookmaker and the contrast is gone")
    books = {bm: p for bm, p in premium.items() if bm not in ex}
    assert len(books) >= 4, sorted(books)
    for bm, p in books.items():
        assert p > 0.50, (
            f"{bm}'s band premium is {p:+.2f}pp, no longer the ~1pp of "
            f"shading this stage reports; the explanation for why the "
            f"netting fails needs re-measuring")
    assert min(books.values()) > 10 * max(abs(premium[b]) for b in ex), (
        "the bookmaker/exchange contrast in band premium is no longer an "
        "order of magnitude, which is the form the finding was recorded in")


# -- PART B: A SUBSET ROI IS NOT COMPARABLE; THE PAIRED DELTA IS ----------

def test_EVERY_COUNTERFACTUAL_BOOK_HAS_A_PAIRED_DELTA():
    """A book's ROI on its own covered subset cannot be compared to another's.
    The paired delta -- the same picks at two prices -- can."""
    assert set(MEASURED_BOOKS["paired_delta_pp"]) == set(
        MEASURED_BOOKS["cost_rel"]), (
        "some book is reported with a cost but no paired delta, so its ROI "
        "has nothing it is comparable to")


def test_the_PAIRED_DELTA_SURVIVES_THE_TAKEABILITY_RESTRICTION():
    """The restriction is the load-bearing check, so it must be shown not to
    manufacture the result. The absolute ROIs move a lot under it; the paired
    deltas do not. That is what makes the delta the reportable quantity."""
    full = MEASURED_BOOKS["paired_delta_pp"]
    rest = MEASURED_BOOKS["paired_delta_pp_takeable_only"]
    for bm, d in rest.items():
        assert abs(d - full[bm]) < 0.40, (
            f"{bm}: the paired delta moved from {full[bm]:+.2f} to {d:+.2f} "
            f"under the takeability restriction. If the delta is no longer "
            f"stable, it cannot be reported as the price effect")
    assert rest["TheOddsAPI-matchbook"] > rest["Pinnacle"] > 0


def test_the_LATER_WRITES_ARE_HOURS_not_a_sequencing_artifact():
    """41% of the cheap books' matched prices were written AFTER the pick. If
    that were minutes it would be a scrape-ordering artifact; at a median of
    6.7 hours it is a price that did not exist when the pick was made."""
    assert MEASURED_BOOKS["later_write_median_minutes"] > 120, (
        f"later writes are now a median "
        f"{MEASURED_BOOKS['later_write_median_minutes']:.0f} min late, which "
        f"is close enough to the pick to be an ordering artifact; the "
        f"restriction's justification needs restating")
    oa = [bm for bm in MEASURED_BOOKS["later_write_share"]
          if bm.startswith("TheOddsAPI-")]
    af = [bm for bm in MEASURED_BOOKS["later_write_share"]
          if not bm.startswith("TheOddsAPI-")]
    assert all(MEASURED_BOOKS["later_write_share"][b] > 0.3 for b in oa), oa
    assert all(MEASURED_BOOKS["later_write_share"][b] == 0.0 for b in af), af


def test_a_NAMED_BOOKS_OWN_PRICES_ARE_COHERENT_unlike_the_best_line():
    """The difference between this stage and the execution stage, in one
    number. The best line was 21.28% sub-100% because it was assembled across
    books AND times. A single book's three prices are its own, at one instant."""
    s = MEASURED_BOOKS["sub100_1x2_share"]
    best = s["BEST_LINE_for_contrast"]
    assert best > 0.20
    for bm, v in s.items():
        if bm == "BEST_LINE_for_contrast":
            continue
        assert v < 0.01, (
            f"{bm} now shows {100 * v:.2f}% of 1X2 leg sets implying under "
            f"100%. A single book's own prices are no longer internally "
            f"coherent, which removes the one takeability test this stage "
            f"passed")
    assert best > 20 * max(
        v for bm, v in s.items() if bm != "BEST_LINE_for_contrast")


# -- PART D: THE ARITHMETIC, AND THE TARGET IT IS MEASURED AGAINST --------

def test_the_MEDIAN_POLICY_STATES_BOTH_UNITS():
    """6.40% relative and 3.70pp are the same measurement. They were read as
    a contradiction once, so the record carries both and the conversion."""
    assert MEDIAN_POLICY["cost_rel"] == pytest.approx(0.0640, abs=0.0002)
    assert MEDIAN_POLICY["cost_pp"] == pytest.approx(
        _pp_at(MEDIAN_POLICY["cost_rel"], MEASURED_BOOKS["avg_taken_odds"]),
        abs=0.02), (
        "the policy's two unit figures no longer convert into each other at "
        "our average odds, so one of them is wrong")
    assert "MEASURED" in MEDIAN_POLICY["provenance"]


def test_NO_BOOK_REACHES_THE_TARGET_ON_PRICE_ALONE():
    """The arithmetic that ends the question, stated as what it would take.

    `required = cost/(1+cost) + target`. Even at the cheapest book whose
    price is established takeable, the edge needed for +3% ROI is multiples
    of the largest documented inefficiency expressed as an ROI.
    """
    for bm, rel in MEASURED_BOOKS["cost_rel"].items():
        required = rel / (1.0 + rel) + ROI_TARGET
        assert required > ROI_TARGET
        if bm in MEASURED_BOOKS["exchanges"]:
            continue
        assert required > 0.06, (
            f"{bm} now requires only {100 * required:.2f}% to clear the +3% "
            f"target; the conclusion that the target is unreachable at every "
            f"non-exchange price needs re-measuring")


def test_the_THREE_LINES_ARE_ORDERED_and_only_ONE_is_positive_and_takeable():
    """Part D in assertion form. Line 1 negative, line 2 ~zero, line 3
    positive but at an exchange whose cost is not what is quoted."""
    A = DOCUMENTED_ANOMALY["value_pp"]
    b = MEASURED_BOOKS["band_cost_pp"]
    line1 = A - b["TAKEN_current_policy"]
    line2 = A - b["Pinnacle"]
    line3 = A - b["TheOddsAPI-matchbook"]
    assert line1 < 0, f"line 1 is {line1:+.2f}pp"
    assert 0 <= line2 < 0.5, (
        f"line 2 is {line2:+.2f}pp; the recorded reading is that the "
        f"cheapest takeable book nets approximately ZERO, which is neither "
        f"a path nor a refutation on its own")
    assert line3 > 3.0
    assert line3 > line2 > line1
    assert "TheOddsAPI-matchbook" in MEASURED_BOOKS["exchanges"], (
        "line 3's book is no longer flagged as an exchange, which is the "
        "only reason line 3 is not the answer")
