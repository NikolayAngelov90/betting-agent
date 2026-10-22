"""`mean+3sd` is conservative BY ACCIDENT; the successor is named — 2026-10-22.

THE FINDING. The weekday lag series is BIMODAL under one unchanged cron
(`0 0 * * 1-5` throughout): 255.9-304.9 through 2026-10-09, a ten-day gap,
then 128.4-144.1. `measured` 2026-10-22:

    weekday n=10   mean 220.0   sd 74.1   mean+3sd 442.3   MAX 304.9

**The bound exceeds every observation by 137.5 minutes.** Three sd of a
two-cluster series is mostly the distance BETWEEN the clusters, so the margin
buys over two hours of headroom against an event never seen.

AND THE DIRECTION OF THAT ACCIDENT IS NOT GUARANTEED, which is the part worth
a test rather than a note. It is conservative while the clusters straddle the
mean. Let the lower one dominate and the mean falls toward it while the sd
stays wide; let a third appear ABOVE both and the bound UNDERSTATES the tail.
Both are driven below.

THE SUCCESSOR IS AN EMPIRICAL QUANTILE, and it is NOT SWITCHED TO, because at
n=10 neither statistic is supported. Both print, each labelled.
"""

import pathlib

import pytest

import scripts.ci_audit as ci

Q = ci.QUANTILE_SUCCESSOR
SRC = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")

#: The live weekday lag series, `measured` 2026-10-22. Pinned so the claims
#: above are checkable against the data that produced them.
WEEKDAY = [255.9, 256.8, 304.9, 271.7, 282.7, 285.8, 128.4, 144.1, 137.7, 132.0]


# ── the accident, and that its direction is not guaranteed ────────────────

def test_the_BOUND_EXCEEDS_EVERY_OBSERVATION_which_is_the_accident():
    m = ci.schedule_margin([(None, v) for v in WEEKDAY], 0,
                           deadline_min=570, day_type="weekday")
    assert m["n"] == 10
    assert m["mean"] == pytest.approx(220.0, abs=0.1)
    assert m["sd"] == pytest.approx(74.1, abs=0.1), (
        "sample sd (n-1), which is what schedule_margin uses")
    assert m["mean_plus_3sd"] == pytest.approx(442.3, abs=0.2)
    assert m["max"] == pytest.approx(304.9)
    assert m["mean_plus_3sd"] - m["max"] == pytest.approx(137.5, abs=0.2), (
        "the 137.5-minute excess is the whole finding; re-derive the record "
        "rather than relaxing this")


def test_DRIVEN_the_accident_REVERSES_when_the_LOWER_cluster_DOMINATES():
    """Conservative today, not by construction. Add low points until they
    outnumber the high ones and the bound falls toward the data."""
    skewed = WEEKDAY + [130.0] * 20
    m = ci.schedule_margin([(None, v) for v in skewed], 0,
                           deadline_min=570, day_type="weekday")
    assert m["mean_plus_3sd"] < 442.3, (
        "the bound did not move when the mixture did")
    # still above the max here, but the GAP has collapsed -- the headroom the
    # margin was silently relying on is a function of the mixture, not of risk
    gap_before, gap_after = 137.5, m["mean_plus_3sd"] - m["max"]
    assert gap_after < gap_before / 2, (
        f"the excess over the max only fell {gap_before:.0f} -> "
        f"{gap_after:.0f}; the accident is being treated as stable")


def test_DRIVEN_a_RARE_EXTREME_makes_the_bound_UNDERSTATE_the_tail():
    """The direction that matters, and the shape is not hypothetical.

    A FIRST ATTEMPT GOT THIS WRONG and the correction is the lesson: adding
    THREE points at ~900 to ten ordinary ones left mean+3sd at 1300, still
    above the max, because a cluster big enough to move the max also inflates
    the sd. The understating case needs the extreme to be RARE -- many tight
    observations and one outlier -- so the sd stays small while the maximum
    runs away.

    That is the real shape of this project's own data: `measured` 2026-10-22,
    the gaps between scheduled daily-picks runs are p50 0.97 days with ONE at
    9.18 (the 2026-06-01 outage). `mean+3sd` of that series sits far below its
    maximum, and nothing in the statistic says so.
    """
    rare = [130.0] * 100 + [900.0]
    m = ci.schedule_margin([(None, v) for v in rare], 0,
                           deadline_min=570, day_type="weekday")
    assert m["mean_plus_3sd"] < m["max"], (
        f"mean+3sd {m['mean_plus_3sd']:.0f} is still above the max "
        f"{m['max']:.0f}; the understating case is not being reached")
    # and in that regime the 3sd margin is the LAXER of the two, so the
    # binding choice silently stops protecting against the observed worst case
    assert m["margin_3sd"] > m["margin_max"], (
        "the 3sd margin is no longer the laxer of the two, so the binding "
        "choice has changed meaning")
    # the quantile, by contrast, cannot leave the data
    q = ci.empirical_quantile(rare, 0.95)
    assert q == 130.0, q


# ── the successor, named with its n ───────────────────────────────────────

def test_the_SUCCESSOR_is_NAMED_with_its_statistic_and_its_n():
    assert Q["statistic"] == "p95"
    assert Q["min_n"] == 100
    assert Q["p99_min_n"] == 500
    assert "5/(1-p)" in Q["min_n_rule"]
    assert Q["projected_usable"] == "2027-03-11"


def test_the_MIN_N_RULE_is_DERIVED_from_a_measured_sensitivity():
    """FIVE points above the quantile, because one new point moved mean+3sd
    by 11.4 minutes at n=9 -> n=10. A band chosen without that measurement
    would be a threshold hard-coded from a small sample."""
    i = SRC.index('"min_n": 100')
    head = SRC[max(0, i - 900):i]
    assert "11.4 minutes" in head, (
        "the measured single-point sensitivity the rule is derived from is no "
        "longer recorded beside it")
    assert "5/(1-p)" in head
    assert ci.quantile_is_usable(100, 0.95) and not ci.quantile_is_usable(99, 0.95)
    assert ci.quantile_is_usable(500, 0.99) and not ci.quantile_is_usable(499, 0.99)


def test_p99_is_named_OUT_OF_REACH_rather_than_offered_without_a_date():
    weeks = (Q["p99_min_n"] - 10) / Q["arrival_per_week"]
    assert weeks > 90, weeks
    assert "out of reach" in SRC, (
        "p99 is offered as an option with no date, which is how a plan "
        "becomes a wait")


# ── the quantile itself: UNDEFINED is a state, not a number ──────────────

@pytest.mark.parametrize("n", [1, 5, 10, 19])
def test_the_QUANTILE_is_UNDEFINED_below_1_over_1_minus_p(n):
    """An interpolated p95 of nine points is a number with NO observation
    above it -- the same extrapolation mean+3sd is being retired for.
    Returning `None` makes 'not enough data' a state."""
    assert ci.empirical_quantile(list(range(n)), 0.95) is None


def test_the_QUANTILE_is_WITHIN_THE_DATA_once_defined():
    vals = list(range(100))
    q = ci.empirical_quantile(vals, 0.95)
    assert q in vals, "the quantile left the data, so it is interpolating"
    assert sum(1 for v in vals if v > q) >= 1, (
        "nothing sits above the quantile, so it is the maximum in disguise")


def test_the_QUANTILE_SURVIVES_BIMODALITY_which_is_the_point():
    """The same mixture that inflates mean+3sd by 137.5 minutes moves the
    quantile not at all beyond the data."""
    vals = sorted(WEEKDAY * 10)                 # n=100, same two clusters
    q = ci.empirical_quantile(vals, 0.95)
    assert q is not None
    assert min(vals) <= q <= max(vals)
    m = ci.schedule_margin([(None, v) for v in vals], 0, deadline_min=570,
                           day_type="weekday")
    assert m["mean_plus_3sd"] > max(vals), (
        "the comparison only means something while mean+3sd is outside the "
        "data")
    assert q <= max(vals), "the quantile cannot leave the data"


def test_the_QUANTILE_IGNORES_an_outlier_that_mean_3sd_chases():
    base = [130.0] * 99
    assert ci.empirical_quantile(base + [10000.0], 0.95) == 130.0, (
        "one absurd point moved the p95, so it is not robust after all")


# ── both printed, each labelled, and neither switched ───────────────────

def test_BOTH_are_PRINTED_SIDE_BY_SIDE_and_the_BINDING_one_is_named():
    m = ci.schedule_margin([(None, v) for v in WEEKDAY], 0,
                           deadline_min=570, day_type="weekday")
    out = ci.format_schedule_margin(m)
    assert "mean+3sd=" in out, "the binding statistic stopped printing"
    assert "successor[p95" in out, "the successor is not printed beside it"
    assert "not usable until n>=100" in out
    assert "ABOVE" in out and "observed max" in out, (
        "the line no longer says the bound is outside the data, which is the "
        "reason the successor exists")
    # ── AND IT NOW NAMES THE POPULATION TOO. 2026-10-22. ────────────────
    #
    # The line used to say "mean+3sd BINDS today", which named the STATISTIC.
    # Since the per-cluster report it must also name WHICH POPULATION the
    # binding figure came from, or a reader sees three margins and cannot
    # tell which one the alarm used.
    assert "BIMODAL" in out, "the line no longer says the series splits"
    assert "GOVERNS" in out, "the slow cluster is not marked as governing"
    assert "reported, NOT relied on" in out, (
        "the fast cluster is printed without saying it is not relied on")
    assert f"BINDING {m['binding']:+.0f}m from slow cluster" in out, (
        "the line does not say which POPULATION binds")


def test_the_SWITCH_HAS_NOT_HAPPENED_and_mean_3sd_still_binds():
    """The decision this stage deliberately does NOT take.

    TWO AXES, AND ONLY ONE MOVED. On 2026-10-22 `binding` started coming from
    the SLOW CLUSTER rather than the whole series -- a change of POPULATION.
    The STATISTIC is still `mean+3sd`: the p95 quantile binds on nothing and
    will not until n>=100. Conflating those two would make this test read as
    if the successor had shipped.
    """
    m = ci.schedule_margin([(None, v) for v in WEEKDAY], 0,
                           deadline_min=570, day_type="weekday")
    # the POPULATION moved: this series is bimodal, so the slow cluster binds
    assert m["bimodal"] is True
    assert m["binding_from"] == "slow cluster"
    assert m["binding"] == m["clusters"]["slow"]["binding"]
    # and the pooled figure is kept, labelled, because the ledger quotes it
    assert m["pooled_binding"] == min(m["margin_max"], m["margin_3sd"])
    # the STATISTIC did not move
    assert m["clusters"]["slow"]["binding"] == min(
        m["clusters"]["slow"]["margin_max"],
        m["clusters"]["slow"]["margin_3sd"])
    assert m["quantile_usable"] is False
    assert m["p95"] is None, "n=10 cannot define a p95"


def test_a_SERIES_LONG_ENOUGH_reports_the_quantile_as_USABLE():
    """The positive control: the printed verdict must change when n does, or
    'not usable until n>=100' is a fixed string."""
    m = ci.schedule_margin([(None, v) for v in sorted(WEEKDAY * 10)], 0,
                           deadline_min=570, day_type="weekday")
    assert m["quantile_usable"] is True
    out = ci.format_schedule_margin(m)
    assert "(BINDS)" in out, out
    assert "not usable until" not in out
    # and it STILL does not bind, because switching is a separate decision
    assert m["binding"] == min(m["margin_max"], m["margin_3sd"])


def test_the_ACCIDENT_and_ITS_UNGUARANTEED_DIRECTION_are_RECORDED():
    assert "CONSERVATIVE BY ACCIDENT" in SRC
    assert "DIRECTION OF THAT ACCIDENT IS NOT GUARANTEED" in SRC
    assert "understates the tail" in SRC, (
        "only the conservative direction is recorded, which is the half that "
        "does not need writing down")


# ── ADDENDUM C: the normalisation near-miss, recorded as a rule ──────────

def test_the_NORMALISATION_RULE_is_RECORDED_where_the_detector_lives():
    """> A cosmetic normalisation of text a detector matches is a change to
    > the detector.

    2026-10-22: changing `step(s) FAILED —` to a hyphen while removing an
    unrelated Telegram send would have emptied `steps_failed` and
    `steps_not_run` permanently, with NO test failing -- a regex that stops
    matching reports nothing rather than erroring.
    """
    assert "A COSMETIC NORMALISATION OF TEXT A DETECTOR MATCHES" in SRC, (
        "the rule is not recorded beside the patterns it governs")
    # the pattern accepts either dash now, so the coupling cannot recur
    assert r"step\(s\) FAILED [—-]" in SRC
    assert r"step\(s\) DID NOT RUN [—-]" in SRC
    # and the workflow says its dashes are load-bearing
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    assert "THE EM DASHES ARE LOAD-BEARING" in wf


def test_BOTH_DASHES_are_read_so_the_coupling_cannot_RECUR():
    """Driven on both forms, because the rule is only as good as the pattern
    that implements it."""
    for dash in ("—", "-"):
        log = f"daily picks step states: step(s) FAILED {dash} update, picks"
        assert ci.extract(log)["steps_failed"] == ["update, picks"], dash
        log2 = f"daily picks step states: step(s) DID NOT RUN {dash} picks"
        assert ci.extract(log2)["steps_not_run"] == ["picks"], dash


def test_the_ANCHOR_still_rejects_the_WORKFLOWS_OWN_ECHOED_SOURCE():
    """The reason the pattern requires a letter after the dash: GitHub echoes
    the `run:` block into the log, so the workflow's own source line would
    match a naive pattern and report every run BROKEN."""
    echoed = 'parts.append("step(s) FAILED — " + ", ".join(failed))'
    assert ci.extract(echoed)["steps_failed"] == [], (
        "the workflow's own source matched, which called two DEGRADED runs "
        "BROKEN on 2026-10-13")
