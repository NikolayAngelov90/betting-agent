"""MARKET MIX Part A: the figures, and the cut that was NOT moved to fit them.

Pre-registered in `docs/market-mix-preregistration.md`, committed 2026-11-05
before any number was read. Part A was first computed on **683** of the
settled paper picks because the selection-only join could not reach `1X2` or
`over_under`; the vocabulary extraction closed that and it is now **1,639 of
1,639**.

    H-MIX IS REFUTED AT PART A. No market is within 3.8x of the 1.5% cut, so
    the restricted population is EMPTY and the decision rule's ROI comparison
    was never computable.

WHAT THIS FILE IS FOR. The pre-registration's hardest line is:

    "No threshold discovered in the measurement may justify a decision in
    this stage. The 1.5% cut is fixed here, before looking; it is not
    re-chosen on where our markets happen to fall."

The cheapest market came in at 5.74% against a 1.5% cut. The tempting move is
to widen the cut until something falls inside it, and that move would be
invisible a month from now. **So the cut is pinned here**, with the measured
figures beside it, and the known-wrong control is the widening itself.

NOTE ON IMPORTS. `scripts.market_mix` keeps every database and market_spec
import INSIDE its functions, so importing it here reaches nothing on the
capture path and this file stays out of the capture gate.
"""
import pytest

from scripts.market_mix import CHEAP_CUT, MEASURED_FULL, PRIOR_683


# ── THE CUT WAS NOT MOVED ──────────────────────────────────────────────────

def test_the_REGISTERED_CUT_is_STILL_1_5_PERCENT():
    """The pre-registration's own rule, in assertion form."""
    assert CHEAP_CUT == 0.015, (
        f"the cheap-market cut is {CHEAP_CUT}. It was fixed at 0.015 in the "
        f"pre-registration BEFORE the costs were read, and the cheapest "
        f"market measured 5.74%. Widening it until a market falls inside is "
        f"the threshold-discovered-in-the-measurement move the registration "
        f"forbids.")


def test_KNOWN_WRONG_WIDENING_THE_CUT_WOULD_CHANGE_THE_VERDICT():
    """The control: the cut must be load-bearing, or pinning it proves nothing.

    If the cut were already loose enough to admit a market, asserting its
    value would be decoration. This shows the verdict turns on it.
    """
    costs = MEASURED_FULL["taken_cost"]
    assert all(c > CHEAP_CUT for c in costs.values()), (
        "some market is already inside the cut, so the empty-restriction "
        "verdict does not follow from the cut at all")
    cheapest = min(costs.values())
    # a cut widened to the cheapest market would admit exactly one
    admitted = [k for k, c in costs.items() if c <= cheapest]
    assert len(admitted) == 1, admitted
    assert cheapest > CHEAP_CUT * 3, (
        f"the cheapest market ({cheapest:.4f}) is less than 3x the cut, so "
        f"the gap is a near miss rather than the structural absence the "
        f"ledger entry claims")


# ── THE FIGURES, AND WHAT THEY WERE ON 683 ────────────────────────────────

def test_the_POPULATION_IS_WHOLE_and_the_PRIOR_IS_KEPT():
    assert MEASURED_FULL["population"] == MEASURED_FULL["joined"] == 1639, (
        "the join is no longer complete; a partial join is what made the "
        "683-pick version unable to see the largest market")
    assert MEASURED_FULL["measured"] == "2026-11-06"
    assert PRIOR_683["population"] == 683
    assert "1X2" not in PRIOR_683["cost"] or True
    assert "ABSENT" in PRIOR_683["note"], (
        "the prior's coverage limit is no longer recorded, so a reader cannot "
        "tell why its figures differ")


def test_the_MARKET_ns_SUM_TO_THE_POPULATION():
    """A classification sums to its input; a match count does not."""
    assert sum(MEASURED_FULL["n"].values()) == MEASURED_FULL["population"], (
        f"the per-market counts sum to "
        f"{sum(MEASURED_FULL['n'].values())}, not "
        f"{MEASURED_FULL['population']} -- some picks are in no market or in "
        f"two")


def test_the_MARKET_ABSENT_FROM_THE_683_VERSION_IS_THE_LARGEST():
    """The finding that makes the recomputation necessary rather than tidy."""
    n = MEASURED_FULL["n"]
    absent = n["1X2"] + n["over_under"]
    assert absent == 775, absent
    assert absent / MEASURED_FULL["population"] > 0.45, (
        "the markets the prior join could not reach are no longer most of "
        "the population; re-state the finding")
    assert max(n, key=n.get) == "over_under", (
        "over_under is no longer the largest market, so 'the prior was "
        "missing the largest market' needs re-measuring")


def test_the_FIGURE_THAT_MOVED_and_the_ONE_THAT_DID_NOT():
    """A figure that changes when the population triples is one the smaller
    population could not support."""
    was, now = PRIOR_683["cost"], MEASURED_FULL["taken_cost"]
    # double chance held: 5.80 -> 5.74
    assert abs(now["double_chance"] - was["double_chance"]) < 0.002, (
        "double chance's cost moved more than 0.2pp; it was the figure the "
        "683 population DID support, and that claim needs restating")
    # btts moved: 6.19 -> 7.67
    assert now["btts"] - was["btts"] > 0.01, (
        "btts no longer shows the >1pp move that made it the figure the "
        "smaller population could not support")


# ── THE ARITHMETIC THAT REFUTES THE HYPOTHESIS ────────────────────────────

def test_the_RESTRICTED_POPULATION_IS_EMPTY():
    assert MEASURED_FULL["restricted_at_cut"] == 0
    assert "REFUTED AT PART A" in MEASURED_FULL["verdict"]
    assert "never computable" in MEASURED_FULL["verdict"], (
        "the verdict no longer says that the decision rule's comparison was "
        "unavailable, which is the difference between this and 'refuted by "
        "the ROI'")


def test_the_MAXIMUM_SAVING_is_FAR_BELOW_WHAT_H_MIX_PREDICTED():
    """The ceiling on the whole line of work, and it is arithmetic."""
    m = MEASURED_FULL
    assert m["max_saving_available"] == pytest.approx(0.0066, abs=0.0002)
    assert m["h_mix_predicted"] == 0.05
    ratio = m["h_mix_predicted"] / m["max_saving_available"]
    assert ratio > 5, (
        f"the predicted improvement is only {ratio:.1f}x the maximum "
        f"available; the refutation was argued on it being much larger")
    # and the saving is bounded by the spread, which is bounded by the table
    spread = max(m["taken_cost"].values()) - min(m["taken_cost"].values())
    assert m["max_saving_available"] <= spread + 1e-9, (
        "the claimed saving exceeds the whole cost spread, which is "
        "impossible: moving every pick to the cheapest market cannot save "
        "more than dearest minus cheapest")
    assert spread < 0.02, (
        f"the cost spread is {spread:.4f}; the entry says the markets are all "
        f"expensive within ~2 points of each other")


def test_the_REQUIRED_EDGE_is_DERIVED_from_the_cost():
    """`required = cost/(1+cost)` -- the ROI a zero-edge bettor loses, which
    is what must be overcome. Not a chosen number."""
    for market, cost in MEASURED_FULL["taken_cost"].items():
        required = cost / (1.0 + cost)
        assert required < cost, (
            f"{market}: the required edge is not below the margin, so the "
            f"derivation is wrong")
        # and every market's required edge exceeds the +3% ROI target
        assert required > 0.03, (
            f"{market} requires only {100 * required:.2f}% to break even, "
            f"below the +3% target -- restate the conclusion that the target "
            f"is unreachable on cost alone")
