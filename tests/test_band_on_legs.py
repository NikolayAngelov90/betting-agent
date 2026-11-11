"""BAND TESTED ON LEGS: the anomaly is real, and it is not collectable.

Measured 2026-11-11 by `scripts/band_legs.py` on **52,737 legs** over ~10,700
fixtures — not on picks. The hypothesis is about MARKET CALIBRATION, so its
unit is a price, and the pick-level version is a selected subset of this one.

    four consecutive bands PRESENT, +1.14 / +1.64 / +1.43 / +2.38 pp,
    every CI above zero, and the LOWEST band NEGATIVE at -1.63pp

That is the favourite–longshot shape, and it agrees in DIRECTION with the
external study in all five comparable bands at 20–130× its n.

**AND THE NET IS NEGATIVE AT EVERY BOOKMAKER IN EVERY BAND**, on the point
estimate and on the CI lower bound alike. One cell survives: 80–90% at an
exchange, +0.48pp before commission — and its depth is unmeasurable, its
coverage 6.1%, and its price band below the floor the operator just restored.

WHAT THIS FILE DEFENDS:

  1. the bands are the EXTERNAL study's, fixed before our data;
  2. de-vigging is PER LEG SET, never pooled — a pooled margin misassigns
     the legs whose own set is wide;
  3. the bootstrap resamples FIXTURES, because one fixture contributes up to
     three legs settled by one scoreline;
  4. existence and collectability stay APART, and the cost is the IN-BAND
     cost at the band's own price;
  5. `s5.19`, `s5.20` and `s5.21` are NEVER pooled, with a known-wrong
     control;
  6. H1's dormancy is a recorded decision with reasons and a mechanism
     review.

NOTE ON IMPORTS. `scripts.band_legs` keeps every database, market_spec and
agent import INSIDE `measure()`, so importing it here reaches nothing on the
capture path and this file stays out of the capture gate.
"""
import ast
import inspect
import pathlib

import pytest

from scripts.band_legs import (BANDS, EXTERNAL, EXTERNAL_LABEL,
                               MEASURED_LEGS, MEASURED_S520, MIN_N_TO_TEST,
                               _band_of)


# ── 1. THE BANDS ARE THE EXTERNAL ONES, FIXED BEFORE OUR DATA ────────────

def test_the_BANDS_ARE_THE_EXTERNAL_STUDYS_BOUNDARIES():
    assert BANDS == ((0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
                     (0.60, 0.70), (0.70, 0.80), (0.80, 0.90)), (
        "the bands moved. They are fixed at the external study's boundaries "
        "and re-banding on our own data is the post-hoc selection "
        "`docs/flb-preregistration.md` forbids")
    # contiguous, half-open, no gaps and no overlaps
    for (a_lo, a_hi), (b_lo, _b_hi) in zip(BANDS, BANDS[1:]):
        assert a_hi == b_lo, f"gap or overlap between {a_hi} and {b_lo}"
    assert _band_of(0.40) == (0.40, 0.50), "the right edge is not exclusive"
    assert _band_of(0.399999) == (0.30, 0.40)
    assert _band_of(0.29) is None and _band_of(0.95) is None
    assert "Footiqo" in EXTERNAL_LABEL and "ONE bookmaker" in EXTERNAL_LABEL


def test_the_EXTERNAL_TABLE_IS_NOT_RE_DERIVED():
    """It is a reference point, not a prior, and it keeps its own n."""
    assert EXTERNAL[(0.70, 0.80)] == {"gap_pp": 3.74, "n": 134}
    assert EXTERNAL[(0.40, 0.50)]["gap_pp"] < 0, (
        "the external study's 40-49 band is no longer negative; our "
        "agreement-in-direction claim rests on it")
    assert set(EXTERNAL) == set(BANDS[1:]), (
        "the external table no longer covers exactly the bands it did; the "
        "30-39 band has no external figure and must stay uncompared")


# ── 2. METHOD: PER-LEG-SET DE-VIGGING, FIXTURE BOOTSTRAP ─────────────────

def test_DEVIGGING_IS_PER_LEG_SET_and_the_source_says_so():
    """Asserted on the call, not on the prose: the margin used for a leg is
    `overround(prices)` of THAT leg set, computed inside the loop."""
    # PARSE THE MODULE FILE, not a dedented function. `inspect.cleandoc` is
    # for docstrings and it stripped the body's indentation into an
    # IndentationError on the first run -- a reminder that a source-shape
    # check has to read the source the way Python does.
    path = pathlib.Path("scripts/band_legs.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "measure"),
              None)
    assert fn is not None, "`measure` is gone from scripts/band_legs.py"
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, "id", "") == "overround"]
    assert calls, "nothing in `measure` calls `overround`, so nothing is "                  "de-vigged"
    src = ast.get_source_segment(path.read_text(encoding="utf-8"), fn) or ""
    assert "p_raw / orr" in src, (
        "the de-vig no longer divides each leg by ITS OWN set's overround. A "
        "pooled or median margin misassigns the legs whose own set is wide, "
        "and the margin spread across books is 0.6pp to 3.9pp")


def test_the_BOOTSTRAP_RESAMPLES_FIXTURES_not_legs():
    from scripts.band_legs import _bootstrap_gap
    src = inspect.getsource(_bootstrap_gap)
    assert "by_fixture" in src and "keys[rng.randrange(len(keys))]" in src, (
        "the bootstrap no longer resamples fixture keys. A fixture "
        "contributes up to three legs settled by ONE scoreline, so a "
        "leg-level resample treats them as independent and understates the "
        "interval")
    # driven: a single fixture cannot produce an interval
    assert _bootstrap_gap({1: [(0.5, 1)]}) is None


def test_KNOWN_WRONG_A_TIED_LEG_SET_WOULD_MISGRADE_without_exact_keys():
    """The control for the key recovery, and it changed a reported figure.

    `extract_legs` returns PRICES, so the selection each price came from must
    be recovered. Matching by VALUE mis-grades a set whose prices tie: `Over
    2.5` and `Under 2.5` both at 1.90 are one number, the two selections
    grade OPPOSITELY, and the de-vigged probability is identical either way —
    so the band would be right and the RESULT wrong. Fixing it moved the
    70-79 gap from +1.66 to +1.43pp.
    """
    from scripts.band_legs import _keys_for
    from src.data.market_spec import get_spec
    spec = get_spec("over_under")
    tied = {"Over 2.5": 1.90, "Under 2.5": 1.90}
    keys = _keys_for(spec, "over_under", tied, "2.5", None)
    assert keys == ["Over 2.5", "Under 2.5"], (
        f"key recovery returned {keys} for a TIED leg set. It must mirror "
        f"`extract_legs`' own key construction, not match prices back")
    # and it refuses rather than guessing when a leg is absent
    assert _keys_for(spec, "over_under", {"Over 2.5": 1.9}, "2.5", None) == []


# ── 3. THE RESULT: FOUR BANDS PRESENT, THE LOWEST NEGATIVE ───────────────

def test_FOUR_CONSECUTIVE_BANDS_ARE_PRESENT():
    b = MEASURED_LEGS["bands"]
    present = [k for k, v in b.items() if v[6] == "PRESENT"]
    assert present == ["0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90"], (
        f"the PRESENT bands are now {present}. 'Several bands moving "
        f"together in the predicted direction' is what "
        f"`docs/flb-preregistration.md` accepts as support for the "
        f"HYPOTHESIS rather than for one band")
    for k in present:
        n, _pred, _obs, gap, lo, hi, _v = b[k]
        assert lo > 0, f"{k}: CI lower bound {lo} is not above zero"
        assert lo < gap < hi, f"{k}: the gap is outside its own interval"
        assert n >= MIN_N_TO_TEST


def test_the_LOWEST_BAND_IS_NEGATIVE_which_is_the_FLB_SHAPE():
    """Longshots overpriced, favourites underpriced. A positive tail with no
    negative head would be a level shift, not a bias."""
    lo_band = MEASURED_LEGS["bands"]["0.30-0.40"]
    assert lo_band[6] == "NEGATIVE" and lo_band[5] < 0, (
        "the 30-39 band is no longer negative, so the measured pattern is a "
        "uniform positive offset rather than a favourite-longshot bias")
    assert MEASURED_LEGS["bands"]["0.40-0.50"][6] == "NOT PRESENT"


def test_the_GAPS_AGREE_IN_DIRECTION_WITH_THE_EXTERNAL_STUDY():
    """Direction, not magnitude. Ours are a third to a half of theirs."""
    b = MEASURED_LEGS["bands"]
    for band, ext in EXTERNAL.items():
        key = f"{band[0]:.2f}-{band[1]:.2f}"
        ours = b[key][3]
        assert (ours > 0) == (ext["gap_pp"] > 0), (
            f"{key}: ours {ours:+.2f}pp disagrees in SIGN with the external "
            f"{ext['gap_pp']:+.2f}pp")
    # and ours are SMALLER wherever both are positive -- stated, not implied
    for band, ext in EXTERNAL.items():
        key = f"{band[0]:.2f}-{band[1]:.2f}"
        if ext["gap_pp"] > 0:
            assert b[key][3] < ext["gap_pp"], (
                f"{key}: our gap is no longer below the external one, so "
                f"'a third to a half of theirs' needs restating")
    # our n dwarfs theirs, which is why ours governs
    assert MEASURED_LEGS["n_legs"] > 20 * sum(
        e["n"] for e in EXTERNAL.values())


# ── 4. COLLECTABILITY IS A SEPARATE QUESTION, AND IT FAILS ───────────────

BOOKMAKERS = ("Pinnacle", "Bet365", "1xBet", "TAKEN_policy")


@pytest.mark.parametrize("band", ["0.50-0.60", "0.60-0.70",
                                  "0.70-0.80", "0.80-0.90"])
def test_the_NET_IS_NEGATIVE_AT_EVERY_BOOKMAKER(band):
    """A gap against a de-vigged price is not money: the margin comes out of
    it, and the IN-BAND margin is the one that applies."""
    gap = MEASURED_LEGS["bands"][band][3]
    ci_lo = MEASURED_LEGS["bands"][band][4]
    costs = MEASURED_LEGS["in_band_cost_pp"][band]
    for bm in BOOKMAKERS:
        assert gap - costs[bm] < 0, (
            f"{band} at {bm}: the POINT estimate now nets "
            f"{gap - costs[bm]:+.2f}pp. That is a profitability claim and "
            f"needs its own evidence bar, not this stage's")
        assert ci_lo - costs[bm] < 0, f"{band} at {bm} on the CI lower bound"


def test_the_IN_BAND_COST_RISES_WITH_THE_BAND_at_the_bookmakers():
    """Why the netting fails: the bookmaker charges most exactly where the
    inefficiency is. Measured, per venue, across the four PRESENT bands."""
    c = MEASURED_LEGS["in_band_cost_pp"]
    for bm in ("Pinnacle", "1xBet", "TAKEN_policy"):
        lo = c["0.50-0.60"][bm]
        hi = c["0.70-0.80"][bm]
        assert hi > lo, (
            f"{bm}'s in-band cost no longer rises from the 50-60 band "
            f"({lo}pp) to the 70-80 band ({hi}pp); the explanation for why "
            f"the netting fails needs re-measuring")
    # and the exchange's is nearly flat, which is the contrast
    ex = "TheOddsAPI-matchbook"
    spread = c["0.80-0.90"][ex] - c["0.50-0.60"][ex]
    assert spread < 0.25, (
        f"the exchange's in-band cost now moves {spread:.2f}pp across the "
        f"bands; a flat margin is what distinguishes it from a bookmaker")


def test_the_ONE_SURVIVING_CELL_is_NOT_A_PLAN():
    s = MEASURED_LEGS["surviving_cell"]
    assert s["band"] == "0.80-0.90"
    assert s["ci_lo_minus_cost_pp"] == pytest.approx(0.48, abs=0.01)
    # it needs a commission below the break-even, and that is narrow
    assert s["breakeven_commission"] == pytest.approx(0.036, abs=0.002)
    assert s["net_at_2pct"] > 0 > s["net_at_6pct"], (
        "the surviving cell no longer straddles the published commission "
        "range, so the verdict no longer depends on an unobtained rate")
    for needed in ("UNAVAILABLE", "6.1%", "NOT directly fetched",
                   "BELOW the restored 1.50 floor"):
        assert needed in s["why_not_a_plan"], (
            f"the surviving cell's record no longer states {needed!r}")
    assert MEASURED_LEGS["exchange_fixture_coverage"] < 0.10


def test_the_VERDICT_keeps_BOTH_halves():
    v = MEASURED_LEGS["verdict"]
    assert "SUPPORTED" in v and "NOT COLLECTABLE" in v, (
        "the verdict no longer carries both halves. The anomaly being real "
        "and our being unable to take it are two findings, and dropping "
        "either one misreports the stage")


# ── 5. THREE COHORTS, NEVER POOLED ───────────────────────────────────────

COHORTS = {
    "s5.19": {"fingerprint": "cdec29", "min_odds": 1.50, "last": "2026-11-07"},
    "s5.20": {"fingerprint": "12f2c7", "min_odds": 1.14, "last": "2026-11-09"},
    "s5.21": {"fingerprint": "52678c", "min_odds": 1.50, "last": None},
}


def test_the_THREE_COHORTS_ARE_DISTINCT_and_may_not_be_pooled():
    """`s5.21` runs the SAME CONFIG as `s5.19` and is a DIFFERENT cohort,
    because the code between them is not the same code. That is exactly why
    restoring a config value does not restore a fingerprint."""
    fps = {c["fingerprint"] for c in COHORTS.values()}
    assert len(fps) == 3, f"two cohorts share a fingerprint: {fps}"
    assert COHORTS["s5.19"]["min_odds"] == COHORTS["s5.21"]["min_odds"] == 1.50
    assert COHORTS["s5.20"]["min_odds"] == 1.14
    # the same config, a different cohort
    assert COHORTS["s5.19"]["fingerprint"] != COHORTS["s5.21"]["fingerprint"], (
        "s5.19 and s5.21 now share a fingerprint. Restoring a config value "
        "would then silently merge two cohorts separated by four days of "
        "code, which is the defect the always-bump rule closes")
    from tests.experiment_pins import (CODE_REVISION_PIN,
                                       FROZEN_MODEL_VERSION,
                                       PREVIOUS_CODE_REVISION,
                                       PREVIOUS_MODEL_VERSION)
    assert CODE_REVISION_PIN == "s5.21"
    assert FROZEN_MODEL_VERSION.endswith(COHORTS["s5.21"]["fingerprint"])
    assert PREVIOUS_CODE_REVISION == "s5.20"
    assert PREVIOUS_MODEL_VERSION.endswith(COHORTS["s5.20"]["fingerprint"])


def test_KNOWN_WRONG_POOLING_THEM_WOULD_HIDE_THE_ROUND_TRIP():
    """The control. If the three were poolable the separation would be
    bookkeeping; this shows what pooling would conceal.

    `s5.20`'s 86 picks average 1.464 against `s5.19`'s 1.607. Pooled into
    the 1,798-pick record the shift is diluted to under a hundredth, so the
    cohort that cost the round trip would become invisible.
    """
    s20_n, s20_avg = MEASURED_S520["n"], MEASURED_S520["odds_mean"]
    s19_n, s19_avg = 1798, 1.607
    pooled = (s19_n * s19_avg + s20_n * s20_avg) / (s19_n + s20_n)
    assert abs(s20_avg - s19_avg) > 0.10, (
        "the two cohorts no longer differ materially in average taken "
        "price, so this control cannot show what pooling hides")
    assert abs(pooled - s19_avg) < 0.01, (
        f"pooling moves the average by {abs(pooled - s19_avg):.4f}, which is "
        f"no longer small enough to call the cohort invisible")


# ── 6. THE s5.20 DISTRIBUTION, AND THE ATTRACTOR QUESTION ────────────────

def test_the_FLOOR_WAS_NOT_AN_ATTRACTOR_and_that_is_measured():
    m = MEASURED_S520
    assert m["floor_was_an_attractor"] is False
    assert m["below_1_15"] == 4 and m["n"] == 86
    assert m["below_1_15"] / m["n"] < 0.10, (
        "more than a tenth of s5.20's picks sat at the floor band, so the "
        "selection WAS taking the shortest admissible price and the "
        "attractor verdict must be restated")
    # the median landed mid-range, not at the floor
    assert m["odds_median"] == pytest.approx(1.440, abs=0.005)
    assert m["odds_min"] < m["odds_median"] < 1.50
    assert "NO." in m["attractor_verdict"]


def test_the_DISTRIBUTION_MOVED_DOWN_WITHOUT_AIMING_AT_THE_BAND():
    """The finding that matters more than the attractor question: the floor
    change made the picks SHORTER without putting them where the anomaly is
    largest."""
    bc = MEASURED_S520["band_counts"]
    high = bc["0.70-0.80"] + bc["0.80-0.90"]
    assert high == 27 and MEASURED_S520["n"] == 86
    assert high / MEASURED_S520["n"] < 0.35, (
        f"{high} of 86 picks reached the high bands; if that is now a "
        f"majority the selection IS aiming at the band and the finding "
        f"inverts")
    assert bc["0.50-0.60"] > bc["0.70-0.80"], (
        "most picks no longer land in the band where the gap is SMALLEST, "
        "which was the point of this measurement")
    assert sum(bc.values()) + MEASURED_S520["UNEVALUATED"] == MEASURED_S520["n"]


def test_the_BREAK_EVEN_WIN_RATE_is_arithmetic_not_a_claim():
    m = MEASURED_S520
    assert m["breakeven_win_rate_at_mean"] == pytest.approx(
        1.0 / m["odds_mean"], abs=0.001), (
        "the break-even rate is no longer 1/odds; it is an identity, not an "
        "estimate")
    # and the settled rate sits BELOW it, which is why the ROI is negative
    assert m["settled_win_rate"] < m["breakeven_win_rate_at_mean"]
    assert m["settled_roi"] < 0
    assert "uninformative" in m["caveat"] and "no conclusion" in m["caveat"], (
        "the caveat no longer says the model's in-band edge estimate is "
        "uninformative and that n=85 over two days supports no conclusion")


# ── 7. H1 IS DORMANT, AS A RECORDED DECISION ─────────────────────────────

def test_H1_DORMANCY_IS_A_DECISION_WITH_ITS_REASONS():
    from src.data.h1_window import DORMANCY, STOPPED_ON
    assert DORMANCY["decision"] == "DORMANT"
    assert DORMANCY["decided"] == "2026-11-11"
    assert STOPPED_ON is not None and DORMANCY["stopped_on_stays"] is True, (
        "STOPPED_ON was cleared. Clearing it is how a NEW WINDOW starts, "
        "which is the other choice, not this one")
    assert "EXCLUDED" in DORMANCY["deletion"]
    assert len(DORMANCY["reasons"]) == 3, (
        "the dormancy record no longer carries three reasons; the guard's "
        "next firing needs something to compare against")
    joined = " ".join(DORMANCY["reasons"])
    for needed in ("s5.20", "MARKET COVERAGE", "VENUE DEPTH"):
        assert needed in joined, f"the reasons no longer name {needed!r}"


def test_the_H1_REVIEW_IS_A_MECHANISM_with_a_DERIVED_anchor():
    import datetime as dt

    from src.data.h1_window import DORMANCY
    assert len(DORMANCY["review_mechanisms"]) == 2
    assert DORMANCY["review_anchor"] == dt.date(2026, 11, 24)
    assert "UEFA" in DORMANCY["review_anchor_basis"], (
        "the anchor no longer names the external fixture fact it is derived "
        "from, so it is a chosen deadline again")
    assert "not a chosen deadline" in DORMANCY["review_anchor_basis"]
    # the anchor must be AFTER the decision, or it is already expired
    assert dt.date.fromisoformat(DORMANCY["decided"]) < DORMANCY[
        "review_anchor"]


def test_the_DORMANCY_CARRIES_WHAT_THE_BAND_TEST_FOUND():
    """Its review is keyed to mechanisms, so the record must say what those
    mechanisms turned up — otherwise the next firing re-derives it."""
    from src.data.h1_window import DORMANCY
    joined = " ".join(DORMANCY["reasons"])
    up = joined.upper()
    assert "PRESENT" in up and "NOT COLLECTABLE" in up, (
        "the dormancy reasons no longer carry the band test's verdict, which "
        "is the measurement that made H1 not the binding question")
    assert "52,737" in DORMANCY["provenance"]
