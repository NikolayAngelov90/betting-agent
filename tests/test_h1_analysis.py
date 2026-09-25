"""H1's analysis pipeline, exercised before the data exists.

Written 2026-09-22. The collection check came from the same registration and
carried a defect that would have aborted the collection on its first morning;
it was found by RUNNING it. This file is the equivalent for the analysis.

THE DRY RUN ON PRODUCTION DATA REACHES STEP 2 AND STOPS. Measured 2026-09-22
over 63,651 in-band pre-kickoff observations: every series has ONE or TWO
distinct timestamps and none has three, so steps 3-5 cannot be reached by any
real row at any floor setting. They are exercised here by injection, which is
the only way they can be exercised before 10-01.

    NO DATA is not a null. A null says the effect is smaller than a
    bound; NO DATA says the measurement did not happen.

Keeping those two apart is the distinction the collection check failed on, and
it is the one this file spends the most tests on.
"""

import math
from datetime import datetime, timedelta

import pytest

import scripts.h1_analysis as ha

T0 = datetime(2026, 10, 2, 12, 0)


def _obs(fixture, book, minutes_prices, market="1X2", selection="Home"):
    return [{"fixture": fixture, "bookmaker": book, "market": market,
             "selection": selection, "ts": T0 + timedelta(minutes=m),
             "odds": p, "match_id": fixture, "match_date": T0 + timedelta(days=1)}
            for m, p in minutes_prices]


# ── STEP 2: trajectories ─────────────────────────────────────────────────────

def test_a_clean_series_becomes_one_trajectory():
    got = ha.build_trajectories(
        _obs(1, "TheOddsAPI-pinnacle", [(0, 2.00), (120, 2.20), (240, 2.42)]))
    assert len(got) == 1
    t = got[0]
    assert t["r1"] == pytest.approx(math.log(1.10))
    assert t["r2"] == pytest.approx(math.log(1.10))
    assert t["provider"] == "TheOddsAPI"


def test_an_unseparated_series_produces_NOTHING():
    assert ha.build_trajectories(
        _obs(1, "b", [(0, 2.0), (6, 2.1), (12, 2.2)])) == []


def test_the_move_is_a_LOG_ratio_so_a_round_trip_is_symmetric():
    """2.00 -> 2.20 -> 2.00 must give +x then -x.

    Raw percentages give +10% then -9.09%, which puts a spurious NEGATIVE
    correlation into every round trip — and H1 would then report
    mean-reversion that the transform invented.
    """
    t = ha.build_trajectories(
        _obs(1, "b", [(0, 2.00), (120, 2.20), (240, 2.00)]))[0]
    assert t["r1"] == pytest.approx(-t["r2"])


def test_a_non_positive_price_is_dropped_not_logged():
    assert ha.build_trajectories(
        _obs(1, "b", [(0, 2.0), (120, 0.0), (240, 2.2)])) == []


# ── STEP 3: provider is a STRATUM ────────────────────────────────────────────

def test_both_providers_appear_and_are_labelled_apart():
    got = ha.build_trajectories(
        _obs(1, "TheOddsAPI-pinnacle", [(0, 2.0), (120, 2.1), (240, 2.2)])
        + _obs(2, "Pinnacle", [(0, 2.0), (120, 2.1), (240, 2.2)]))
    assert {t["provider"] for t in got} == {"TheOddsAPI", "API-Football"}


def test_a_series_cannot_span_two_books():
    """The series key includes the book, so splicing is impossible.

    Pinned because 'one price series' is prose, and prose is what this whole
    exercise is converting.
    """
    got = ha.build_trajectories(
        _obs(1, "book-a", [(0, 2.0)]) + _obs(1, "book-b", [(120, 2.1)])
        + _obs(1, "book-c", [(240, 2.2)]))
    assert got == []


# ── STEP 4: AGGREGATION — where H5 lost a factor of fifty ────────────────────

PICK = {1: ("1X2", "Home")}


def test_one_fixture_priced_by_five_books_yields_ONE_row():
    obs = []
    for i, book in enumerate(["TheOddsAPI-a", "TheOddsAPI-b", "TheOddsAPI-c",
                              "TheOddsAPI-d", "TheOddsAPI-e"]):
        obs += _obs(1, book, [(0, 2.0 + i / 100), (120, 2.1), (240, 2.2)])
    trajectories = ha.build_trajectories(obs)
    assert len(trajectories) == 5
    actionable = ha.one_price_per_fixture(trajectories, PICK)
    assert len(actionable) == 1, (
        "five books on one fixture produced five rows — this is the "
        "fifty-fold inflation in n that H5's first sigma died of")


def test_it_takes_the_BEST_line_not_the_mean():
    """The bettor gets a price, not an average of prices."""
    obs = (_obs(1, "TheOddsAPI-a", [(0, 2.00), (120, 2.1), (240, 2.2)])
           + _obs(1, "TheOddsAPI-b", [(0, 2.50), (120, 2.6), (240, 2.7)]))
    got = ha.one_price_per_fixture(ha.build_trajectories(obs), PICK)
    assert len(got) == 1
    assert got[0]["p0"] == 2.50, "took something other than the best line"


def test_a_trajectory_on_an_UNPICKED_selection_is_dropped():
    """A price the pipeline would never have taken is not actionable."""
    obs = _obs(1, "TheOddsAPI-a", [(0, 2.0), (120, 2.1), (240, 2.2)],
               selection="Away")
    assert ha.one_price_per_fixture(ha.build_trajectories(obs), PICK) == []


def test_a_fixture_with_no_pick_is_dropped_and_not_averaged_in():
    obs = _obs(99, "TheOddsAPI-a", [(0, 2.0), (120, 2.1), (240, 2.2)])
    assert ha.one_price_per_fixture(ha.build_trajectories(obs), PICK) == []


# ── STEP 5: the estimator ────────────────────────────────────────────────────

def test_pearson_matches_a_known_value():
    assert ha.pearson([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)
    assert ha.pearson([1, 2, 3, 4], [8, 6, 4, 2]) == pytest.approx(-1.0)


def test_no_variance_is_UNDEFINED_not_zero():
    """A flat series has no correlation; reporting 0.0 would be a result."""
    assert ha.pearson([1, 1, 1, 1], [2, 4, 6, 8]) is None


def test_the_p_value_is_ONE_sided():
    """A strong positive r must be significant and its mirror must not."""
    assert ha.one_sided_p(0.6, 40) < 0.05
    assert ha.one_sided_p(-0.6, 40) > 0.95


def test_the_null_bound_matches_the_registration():
    """n=33 -> 0.292 and n=39 -> 0.267, both below break-even."""
    assert ha.upper_bound_rho(33) == pytest.approx(0.292, abs=0.005)
    assert ha.upper_bound_rho(39) == pytest.approx(0.267, abs=0.005)
    assert ha.captured_gain(ha.upper_bound_rho(39)) < ha.BREAK_EVEN


def test_the_actionable_threshold_is_where_the_gain_meets_the_bar():
    assert ha.captured_gain(ha.RHO_ACTIONABLE) == pytest.approx(2.0, abs=0.05)


# ── THE OUTCOME STATES, AND THE ONE THAT MATTERS MOST ────────────────────────

def test_NO_DATA_is_not_a_null():
    """THE DISTINCTION THE COLLECTION CHECK FAILED ON."""
    state, lines = ha.interpret(0, None, None)
    out = "\n".join(lines)
    assert state == "NO DATA"
    assert "NOT a null result" in out
    assert "NULL" not in state


def test_trajectories_with_no_variance_read_as_APPARATUS_BROKEN():
    """Real prices move. Identical moves are an instrument fault."""
    state, lines = ha.interpret(40, None, None)
    assert state == "APPARATUS BROKEN"
    assert "instrument fault" in "\n".join(lines)
    assert "NO DATA" not in state


def test_NO_DATA_and_APPARATUS_BROKEN_are_different_states():
    """Both produce no answer, and the actions they imply are opposite:
    one says collect, the other says stop and fix."""
    assert ha.interpret(0, None, None)[0] != ha.interpret(40, None, None)[0]


def test_below_the_registered_n_it_REFUSES_to_interpret():
    """Reading the rule early is the optional-stopping error."""
    state, lines = ha.interpret(10, 0.9, 0.001)
    assert state == "INSUFFICIENT"
    assert "NOT interpreted" in "\n".join(lines)
    assert "ACTIONABLE" not in state


def test_a_null_is_reported_WITH_its_bound():
    state, lines = ha.interpret(39, 0.01, 0.48)
    out = "\n".join(lines)
    assert state == "NULL"
    assert "upper bound" in out and "break-even" in out
    assert "RETIRES timing" in out


def test_a_negative_result_is_FALSIFIED_not_weak_support():
    state, lines = ha.interpret(39, -0.5, 0.001)
    assert state == "FALSIFIED"
    assert "SEPARATE finding" in "\n".join(lines)


def test_a_real_but_small_effect_is_NOT_ACTIONABLE():
    state, lines = ha.interpret(39, 0.30, 0.02)
    assert state == "SIGNAL NOT ACTIONABLE"
    assert "below" in "\n".join(lines)


def test_a_large_effect_is_ACTIONABLE():
    state, _ = ha.interpret(39, 0.55, 0.001)
    assert state == "SIGNAL ACTIONABLE"


def test_every_outcome_state_is_distinct():
    """SEVEN states, seven labels. Any collapse makes a report unreadable.

    The count is asserted as well as the distinctness: adding an eighth
    outcome without giving it a label would otherwise pass silently.
    """
    states = {
        ha.interpret(0, None, None)[0],
        ha.interpret(40, None, None)[0],
        ha.interpret(10, 0.9, 0.001)[0],
        ha.interpret(39, 0.01, 0.48)[0],
        ha.interpret(39, -0.5, 0.001)[0],
        ha.interpret(39, 0.30, 0.02)[0],
        ha.interpret(39, 0.55, 0.001)[0],
    }
    assert states == {
        "NO DATA", "APPARATUS BROKEN", "INSUFFICIENT", "NULL",
        "FALSIFIED", "SIGNAL NOT ACTIONABLE", "SIGNAL ACTIONABLE"}, states
    assert len(states) == 7


# ── THE DIAGNOSTIC THAT MAKES A ZERO LEGIBLE ─────────────────────────────────

def test_the_separated_point_distribution_explains_a_zero():
    """A floor that matches nothing and a broken floor both report zero.

    Measured on production 2026-09-22: {1: 63381, 2: 135} — the floor DOES
    match real data, and what is missing is the third observation.
    """
    obs = (_obs(1, "b", [(0, 2.0), (600, 2.1)])          # two separated
           + _obs(2, "b", [(0, 2.0), (3, 2.0), (6, 2.0)]))   # one, thinned
    assert ha.separated_point_distribution(obs) == {2: 1, 1: 1}


def test_two_point_series_are_found_and_counted_by_FIXTURE():
    """135 series across 2 fixtures — the aggregation trap in miniature."""
    obs = (_obs(1, "book-a", [(0, 2.0), (600, 2.1)])
           + _obs(1, "book-b", [(0, 2.0), (600, 2.1)])
           + _obs(1, "book-c", [(0, 2.0), (600, 2.1)]))
    keys = ha.series_keys_with(obs, 2)
    assert len(keys) == 3
    assert len({k[0] for k in keys}) == 1


# ── INFLUENCE ────────────────────────────────────────────────────────────────

def test_the_influence_check_names_the_fixture_it_dropped():
    rows = [{"fixture": i, "r1": float(i), "r2": float(i)} for i in range(6)]
    rows.append({"fixture": 99, "r1": 10.0, "r2": -10.0})
    out = "\n".join(ha.influence_check(rows))
    assert "99" in out, out


def test_the_influence_check_is_silent_when_n_is_tiny():
    assert "too small" in "\n".join(ha.influence_check([]))


# ── THE REGISTERED CONSTANTS ─────────────────────────────────────────────────

def test_constants_match_the_registration():
    assert ha.RHO_ACTIONABLE == 0.42
    assert ha.ALPHA == 0.05
    assert ha.N_MIN == 33
    assert ha.BREAK_EVEN == 1.85
    assert ha.SIGMA_REGISTERED == 5.937


def test_the_analysis_shares_the_checks_predicate_rather_than_restating_it():
    """One prose rule, one implementation. Two diverge silently."""
    import scripts.h1_collection_check as hc
    assert ha.qualifying_triple is hc.qualifying_triple
    assert ha.MIN_GAP_MINUTES == hc.MIN_GAP_MINUTES
    assert ha.load_observations is hc.load_observations


# ── THE NULL-INPUT CONTROL — the positive control's mirror ───────────────────
#
# Added 2026-09-25 at Niki's request, to check a claim I had made: that the raw
# percentage transform "turns every round trip into a spurious negative
# correlation, which H1 would report as mean-reversion the transform invented".
#
#   THE CONTROL REFUTED THE CLAIM. Measured at n = 20,000 with independent
#   multiplicative moves, the percentage transform's bias at true rho = 0 is
#   +0.0005 (sigma=0.06), -0.0004 (sigma=0.30) and -0.0018 (sigma=0.60),
#   against the log transform's +0.0006. It manufactures nothing.
#
# What it actually does is ATTENUATE. At a true rho of 0.42 the log transform
# recovers +0.4238 at every volatility, while the percentage transform gives
# +0.4145 at sigma=0.30 and +0.3824 at sigma=0.60 — a downward bias of up to
# 0.04 at H1's own decision boundary. So the error direction is toward a FALSE
# NULL, not a false signal.
#
# The log ratio is still the correct transform, for the two reasons pinned
# below: exact antisymmetry on a round trip, and unbiased recovery of rho. But
# the "spurious finding" framing was wrong, and the control is what established
# that rather than more argument.

import math
import random


def _simulate(n, seed, transform, rho=0.0, vol=0.06):
    """Independent fixtures with a controllable TRUE serial correlation."""
    rng = random.Random(seed)
    xs, ys = [], []
    for _ in range(n):
        p0 = rng.uniform(1.5, 6.0)
        u1 = rng.gauss(0, vol)
        u2 = rho * u1 + math.sqrt(max(0.0, 1 - rho * rho)) * rng.gauss(0, vol)
        p1, p2 = p0 * math.exp(u1), p0 * math.exp(u1 + u2)
        xs.append(transform(p0, p1))
        ys.append(transform(p1, p2))
    return ha.pearson(xs, ys)


_LOG = lambda a, b: math.log(b / a)
_PCT = lambda a, b: (b - a) / a


def test_NULL_INPUT_CONTROL_zero_true_correlation_reports_zero():
    """THE CONTROL. Feed the estimator noise; it must not find a signal.

    The positive controls above confirm the pipeline reports a correlation
    that IS there. This confirms it does not report one that is not, which is
    the only direction that can produce a false finding.
    """
    r = _simulate(20000, 11, _LOG, rho=0.0)
    assert abs(r) < 0.01, f"noise produced r = {r:+.4f}"


def test_the_null_input_control_reaches_a_NULL_state_not_a_SIGNAL():
    """End of the chain: the decision rule must call this nothing."""
    r = _simulate(400, 11, _LOG, rho=0.0)
    p = ha.one_sided_p(r, 400)
    state, _ = ha.interpret(400, r, p)
    assert state == "NULL", f"noise was reported as {state} (r={r:+.4f}, p={p:.4f})"


def test_the_estimator_RECOVERS_a_planted_correlation():
    """The mirror. A control that only ever says NULL is not a control."""
    r = _simulate(20000, 11, _LOG, rho=0.42)
    assert abs(r - 0.42) < 0.02, f"planted 0.42, recovered {r:+.4f}"
    state, _ = ha.interpret(400, r, ha.one_sided_p(r, 400))
    assert state == "SIGNAL ACTIONABLE"


def test_the_percentage_transform_does_NOT_manufacture_a_signal():
    """THE CLAIM THIS CONTROL WAS BUILT TO CHECK, AND IT FAILED.

    I asserted the percentage transform would produce a spurious negative
    correlation out of nothing. At true rho = 0 it does not, at any volatility
    this market shows. Pinned so the wrong claim cannot come back.
    """
    for vol in (0.06, 0.30, 0.60):
        r = _simulate(20000, 11, _PCT, rho=0.0, vol=vol)
        assert abs(r) < 0.01, (
            f"at vol={vol} the percentage transform gave r={r:+.4f} from pure "
            f"noise — if this ever fires, the spurious-finding claim was right "
            f"after all and the log transform is load-bearing for that reason")


def test_what_the_percentage_transform_ACTUALLY_does_is_attenuate():
    """It biases rho DOWNWARD, so its error points at a false NULL.

    Measured: 0.4238 under log at every volatility; 0.4145 and 0.3824 under
    percentages at sigma 0.30 and 0.60. At H1's 0.42 boundary that is enough
    to move a true signal below the actionable line.
    """
    log_hi = _simulate(20000, 11, _LOG, rho=0.42, vol=0.60)
    pct_hi = _simulate(20000, 11, _PCT, rho=0.42, vol=0.60)
    assert pct_hi < log_hi, "no attenuation observed"
    assert log_hi - pct_hi > 0.02, (log_hi, pct_hi)
    assert abs(log_hi - 0.42) < 0.01, "the log transform should be unbiased"


def test_the_log_transform_is_EXACTLY_antisymmetric_on_a_round_trip():
    """The reason the transform choice is still correct.

    p2 == p0 must give r = -1 exactly. Percentages give -0.9414 on the same
    data: both find mean-reversion, but only one is exact.
    """
    rng = random.Random(3)
    xl, yl, xp, yp = [], [], [], []
    for _ in range(500):
        p0 = rng.uniform(1.5, 6.0)
        p1 = p0 * math.exp(rng.gauss(0, 0.25))
        xl.append(_LOG(p0, p1)); yl.append(_LOG(p1, p0))
        xp.append(_PCT(p0, p1)); yp.append(_PCT(p1, p0))
    assert ha.pearson(xl, yl) == pytest.approx(-1.0, abs=1e-9)
    assert ha.pearson(xp, yp) > -0.99, "percentages are not exact here"
