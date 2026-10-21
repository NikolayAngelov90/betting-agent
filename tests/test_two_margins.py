"""Two margins, not one — the structural fact and the actionable one.

    2026-10-21. Rule 3: every margin prints its basis and its coverage, and
    the pair is never reduced to the optimistic half.

WHY THE SPLIT. `min(reference, floor)` always returned the floor, because the
floor IS the observed earliest kickoff over all 30 configured leagues. So one
number reported a structural statement about the pipeline's duration while
looking like a day-specific deadline. Measured 2026-10-21: floor 09:30 against
today's actual earliest kickoff of 15:30 — six hours pessimistic.

    STRUCTURAL   vs the floor. True every day, no action clears it.
    ACTIONABLE   vs the day's earliest kickoff per the reference. A manual
                 dispatch answers this one, so by CLR-2 it is the one that
                 may alarm.
"""

import datetime as dt

import pytest

import scripts.schedule_deadline as sd

WEEKDAY = dt.date(2026, 10, 22)       # a Thursday
CRON = 0
DELAYS = [200.0, 240.0, 260.0, 280.0, 300.0, 220.0, 230.0, 250.0]
DURATIONS = [30.0, 40.0, 42.0, 45.0, 50.0, 60.0, 35.0, 38.0]


def _ref(minute=None, state="OK"):
    return {"min": minute, "state": state, "league": "europe/champions-league",
            "leagues_present": ["europe/champions-league"],
            "leagues_absent": ["england/premier-league"]}


def test_BOTH_margins_are_returned_and_both_name_their_coverage():
    m = sd.two_margins(WEEKDAY, _ref(16 * 60 + 45), CRON, DELAYS, DURATIONS)
    assert set(m) >= {"structural", "actionable", "residual", "alarms_on"}
    for key in ("structural", "actionable"):
        a = m[key]
        assert a["coverage"], f"{key} has no coverage statement"
        assert a["deadline_basis"], f"{key} has no basis"
        assert a["kind"] == key.upper()
    # the two coverages are DIFFERENT, which is the whole point
    assert m["structural"]["coverage"] != m["actionable"]["coverage"]
    assert "30" in m["structural"]["coverage"]
    assert "9 of 30" in m["actionable"]["coverage"]


def test_the_ALARM_is_on_the_ACTIONABLE_margin():
    """CLR-2: an alarm fires on the quantity its clearing action changes, and
    a dispatch changes only the day-specific one."""
    m = sd.two_margins(WEEKDAY, _ref(16 * 60 + 45), CRON, DELAYS, DURATIONS)
    assert m["alarms_on"] == "actionable"
    assert m["structural"]["acknowledged"]["review"], (
        "the structural breach has no dated review")
    assert m["structural"]["acknowledged"]["action"], (
        "the structural breach names no action")


def test_the_REGISTERED_MINIMUM_is_NOT_edited():
    """Rule: a pre-registered threshold is not refitted to a later
    measurement."""
    assert sd.MIN_ACCEPTABLE_MARGIN_MIN == 60
    m = sd.two_margins(WEEKDAY, _ref(16 * 60 + 45), CRON, DELAYS, DURATIONS)
    assert m["structural"]["min_margin"] == 60
    assert m["actionable"]["min_margin"] == 60


def test_an_ABSENT_reference_is_UNEVALUATED_not_a_clear_margin():
    """THE THIRD STATE. "The reference saw nothing" is not "the margin is
    fine", and the structural half must still be reported."""
    m = sd.two_margins(WEEKDAY, _ref(None, "NO_FIXTURE"), CRON, DELAYS,
                       DURATIONS)
    assert m["actionable"]["verdict"] == "UNEVALUATED"
    assert "NOT a clear margin" in m["actionable"]["reason"]
    assert m["structural"]["verdict"] in ("ALARM", "CLEAR"), (
        "the structural margin went missing when the reference did")
    out = sd.format_two_margins(m)
    assert "STRUCTURAL" in out and "UNEVALUATED" in out


def test_KNOWN_WRONG_a_day_whose_EARLIEST_is_an_UNCOVERED_league():
    """THE CONTROL, drawn from the case that motivated the split.

    2026-10-21: the reference reported 16:45 (champions-league) and the card's
    true earliest kickoff was 15:30, carried by a league the reference does
    not cover. The ACTIONABLE margin is WRONG on such a day — optimistic by
    the residual — and the STRUCTURAL margin must still be reported, so the
    pair is never silently reduced to the optimistic half.
    """
    REPORTED, TRUE = 16 * 60 + 45, 15 * 60 + 30
    m = sd.two_margins(WEEKDAY, _ref(REPORTED), CRON, DELAYS, DURATIONS)
    honest = sd.two_margins(WEEKDAY, _ref(TRUE), CRON, DELAYS, DURATIONS)

    # the actionable margin is OPTIMISTIC by exactly the residual
    over = m["actionable"]["binding"] - honest["actionable"]["binding"]
    assert abs(over - (REPORTED - TRUE)) < 1e-6
    assert over == pytest.approx(75.0), (
        f"the known case is 75 minutes optimistic, got {over}")

    # and the structural half is UNCHANGED by the reference being wrong,
    # which is what makes it the safety claim
    assert m["structural"]["binding"] == honest["structural"]["binding"]
    out = sd.format_two_margins(m)
    assert "STRUCTURAL" in out, "the pair was reduced to the optimistic half"
    assert "UPPER BOUND" in out, (
        "the actionable margin is printed without saying it is an upper bound")


def test_the_RESIDUAL_is_stated_and_it_EXCEEDS_the_threshold():
    """The unsafe direction, handled explicitly rather than assumed away.

    The directive expected the residual to be ~15 minutes and the threshold to
    exceed it. MEASURED, IT DOES NOT: the 15 minutes was the SELF-ANCHORED
    FLOOR's exposure, a different quantity which is now zero by construction.
    The reference's own residual had never been measured. Simulated
    2026-10-21 over n=605 days, it is 75m at the median and exceeds 60m on
    51.6% of days — so an actionable margin clearing 60 is NECESSARY, NOT
    SUFFICIENT, and this test pins that rather than letting it be forgotten.
    """
    r = sd.REFERENCE_RESIDUAL
    assert r["n_days"] == 605
    assert r["p50"] == 75
    assert r["over_60"] / r["n_days"] > 0.5, (
        "the residual no longer exceeds 60m on most days - re-measure, and if "
        "it has genuinely fallen below the threshold, say so and re-state "
        "whether the actionable margin can carry the safety claim alone")
    assert r["exceeds_threshold"] is True
    assert r["p50"] > sd.MIN_ACCEPTABLE_MARGIN_MIN, (
        "the median residual no longer exceeds the registered minimum")
    assert "simulated" in r["provenance"] and "2026-10-21" in r["provenance"]
    out = sd.format_two_margins(
        sd.two_margins(WEEKDAY, _ref(16 * 60 + 45), CRON, DELAYS, DURATIONS))
    assert "NECESSARY, NOT SUFFICIENT" in out
    assert "does NOT exceed the residual" in out


def test_the_EXPOSURE_term_is_still_printed_at_ZERO():
    """A term that disappears when satisfied cannot be seen to have been
    satisfied. The FLOOR's exposure is zero by construction and stays visible;
    it is a different quantity from the REFERENCE's residual above."""
    assert sd.COVERAGE_EXPOSURE["weekday"]["p99"] == 0
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 60, "weekend": 60}
    assert sd.REFERENCE_RESIDUAL["p99"] != sd.COVERAGE_EXPOSURE["weekday"]["p99"], (
        "the two residuals have been conflated; they measure different things")


# ══════════ THE DURATION TERM MEASURES TIME-TO-PICKS ══════════════════════
#
#     2026-10-21. Rule 7: a figure measures what it claims to.

import pathlib
import statistics

import scripts.ci_audit as _ci


def test_the_duration_term_is_TIME_TO_PICKS_not_the_FULL_RUN():
    """The margin asks whether the CARD is saved before the earliest kickoff.

    Everything after the picks step is yesterday's business -- the Flashscore
    results scrape, the late settle, the alerts, the cache saves -- and none
    of it can affect whether today's card beats today's kickoff. The term used
    to charge the card for all of it.
    """
    ttp = _ci.MARGIN_RUN_DURATIONS_BY_DAYTYPE
    full = _ci.MARGIN_RUN_DURATIONS_FULL_BY_DAYTYPE
    for dtype in ("weekday", "weekend"):
        a, b = ttp[dtype], full[dtype]
        m3 = lambda v: statistics.mean(v) + 3 * statistics.pstdev(v)
        assert m3(a) < m3(b), (
            f"{dtype}: the margin's series is no longer tighter than the "
            f"full-run series, so it is measuring the full run again")
        assert max(a) < max(b), f"{dtype}: the maxima no longer differ"
    # the weekday correction, pinned with its figures
    assert abs(m3_ := (statistics.mean(ttp["weekday"])
                       + 3 * statistics.pstdev(ttp["weekday"]))) > 0
    assert 84.0 < m3_ < 87.0, (
        f"the weekday time-to-picks bound moved to {m3_:.1f}; re-measure and "
        f"re-state the structural acknowledgement's figure with it")


def test_the_SUPERSEDED_series_is_RETAINED():
    """A correction that deletes what it corrected cannot be checked."""
    full = _ci.MARGIN_RUN_DURATIONS_FULL_BY_DAYTYPE
    assert len(full["weekday"]) >= 70 and len(full["weekend"]) >= 30
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "WRONG quantity for the margin" in src, (
        "the retained series does not say it is the wrong one for this use, "
        "so the next reader may take it for the current figure")


def test_the_ACKNOWLEDGEMENT_carries_the_RE_MEASURED_figure():
    """A figure in an acknowledgement is the one thing a reader trusts
    without checking, so it must not be allowed to drift."""
    ack = sd.STRUCTURAL_ACK
    assert ack["binding"] == pytest.approx(-7.4), (
        f"the acknowledgement claims {ack['binding']}. It must be the "
        f"MEASURED deficit: -19.8 was computed against a duration term that "
        f"measured the wrong quantity, and -7.0 was a loose transcription of "
        f"60 - 52.6 that the deterioration predicate caught on its first "
        f"exposure")
    assert "INSTRUMENTATION ERROR" in ack["why"], (
        "the acknowledgement does not record that most of the original "
        "shortfall was a measurement error rather than slow code")
    # and the action must name the SPREAD, because that is what decides order
    assert "SPREAD" in ack["action"] and "mean+3sd" in ack["action"], (
        "the action does not say that the spread is what the bound is made "
        "of, so the next person will optimise the mean first")


def test_VARIANCE_BEFORE_MEAN_is_recorded_with_its_arithmetic():
    """Rule 4. The ordering claim is a measurement, not a preference:
    parallelising the review cuts the sd 18.4 -> 10.8 and buys +30.5 min,
    against +3.5 for the retrain move whose mean reduction is comparable."""
    act = sd.STRUCTURAL_ACK["action"]
    for fig in ("+30.5", "18.4", "10.8", "+3.5", "+4.0"):
        assert fig in act, f"{fig} is not recorded in the clearing action"
    assert "n=72" in act, "the model's sample size is not stated"


# ═══════ THE ACKNOWLEDGEMENT, AFTER THE RESTATEMENT — 2026-10-21 ══════════
#
# The structural figure moved -19.8 -> -7.0 when the duration term started
# measuring time-to-picks. An acknowledgement compares a figure, and the LAST
# one silently absorbed a breach because it compared only `binding` and never
# `review`: a deferral with a date nothing reads is not a deferral.


def test_the_ACKNOWLEDGEMENT_matches_the_RESTATED_figure():
    ack = sd.STRUCTURAL_ACK
    assert ack["binding"] == pytest.approx(-7.4), (
        "the figure must be the MEASURED deficit; -7.0 was a loose "
        "transcription of 60 - 52.6 and the predicate caught it on its "
        "first exposure")
    # NOT asserted here: that the LIVE margin sits at that deficit. The first
    # draft did, using this module's synthetic DELAYS/DURATIONS, which give a
    # margin of +102 -- it was checking a fixture and reporting it as a fact
    # about production. That belongs to a `live` test, and conflating the two
    # is the thing the marker exists to prevent.
    assert "7.4" in ack["action"], (
        "the action does not restate the figure it is acknowledging")
    assert "INSTRUMENTATION ERROR" in ack["why"], (
        "the acknowledgement does not record WHY the figure moved from -19.8")


def test_it_ALARMS_ON_DETERIORATION_and_the_drive_is_EXPLICIT():
    """The figures the directive named, driven."""
    today = dt.date(2026, 10, 21)
    for covered in (-7.4, -7.0, -6.9, -6.0, 0.0, 5.0):
        assert sd.structural_breach_is_acknowledged(covered, today), (
            f"deficit {covered:+.1f} is no worse than the acknowledged -7.4 "
            f"and must be COVERED")
    for alarms in (-7.5, -8.0, -20.0, -100.0):
        assert sd.structural_breach_is_acknowledged(alarms, today) == {}, (
            f"deficit {alarms:+.1f} is WORSE than the acknowledged -7.4 and "
            f"must alarm -- deterioration is an event")


def test_it_EXPIRES_and_the_REVIEW_DATE_is_actually_COMPARED():
    """The retired `ACKNOWLEDGED_MARGIN_BREACH` compared only the figure, so
    it absorbed a breach for days past its own review date."""
    review = dt.date.fromisoformat(sd.STRUCTURAL_ACK["review"])
    assert sd.structural_breach_is_acknowledged(-7.4, review), (
        "it expires ON the review date; the date itself is still covered")
    assert sd.structural_breach_is_acknowledged(
        -7.4, review + dt.timedelta(days=1)) == {}, (
        "the acknowledgement outlived its review date")
    assert sd.structural_breach_is_acknowledged(
        -7.4, review + dt.timedelta(days=365)) == {}


def test_an_UNPARSEABLE_or_MISSING_review_date_absorbs_NOTHING():
    """An acknowledgement that cannot be dated is not one."""
    import copy
    for bad in (None, "", "soon", "2026-13-45", "21/11/2026"):
        patched = copy.deepcopy(sd.STRUCTURAL_ACK)
        patched["review"] = bad
        real, sd.STRUCTURAL_ACK = sd.STRUCTURAL_ACK, patched
        try:
            assert sd.structural_breach_is_acknowledged(
                -7.4, dt.date(2026, 10, 21)) == {}, (
                f"review={bad!r} still absorbed the breach")
        finally:
            sd.STRUCTURAL_ACK = real


def test_the_REGISTERED_MINIMUM_is_UNEDITED_and_the_EXPOSURE_PRINTS_AT_ZERO():
    """The acknowledgement answers "has this been seen and recorded", not "is
    it acceptable". It must not have moved the bar instead."""
    assert sd.MIN_ACCEPTABLE_MARGIN_MIN == 60
    assert sd.EFFECTIVE_MIN_MARGIN == {"weekday": 60, "weekend": 60}
    assert sd.COVERAGE_EXPOSURE["weekday"]["p99"] == 0
    # a term that disappears when satisfied cannot be seen to have been
    # satisfied, so it must still be in the rendered output at zero
    out = sd.format_reference_deadline(
        sd.combined_deadline(WEEKDAY, _ref(16 * 60 + 45)))
    assert "p99=0m" in out, (
        "the exposure term vanished from the rendered deadline; at zero it "
        "must still print, or nobody can see it was satisfied")


def test_the_AUDIT_consults_the_predicate_rather_than_printing_unconditionally():
    import ast
    import inspect

    import scripts.ci_audit as _ci
    src = inspect.getsource(_ci.main)
    assert "structural_breach_is_acknowledged" in src, (
        "the audit prints the acknowledgement without checking whether it "
        "still holds, so a deteriorated margin would report 'does NOT alarm'")
    assert "NO LONGER" in src and "DETERIORATED" in src, (
        "there is no branch for an acknowledgement that stopped covering")


# ═════ THE PICKS BOUND: NOT RE-DERIVED, AND THE RULE IS ENFORCED ══════════
#
#     2026-10-21. Rules 1 and 2: regimes are not pooled, and a constant
#     carries its n and its regime with a small n stated as small.


def test_the_PICKS_BOUND_still_describes_the_SERIAL_regime():
    """The review was parallelised in `cccb0e6` at 21:29 on 2026-10-21 and the
    newest daily-picks run is 08:29 the same day: **n = 0 post-change runs.**
    So this constant has NOT been re-derived and still describes the serial
    regime. If it moves, the mover must state the new n and the regime.
    """
    assert _ci.FULL_CARD_PICKS_MAX == 88.1, (
        f"FULL_CARD_PICKS_MAX moved to {_ci.FULL_CARD_PICKS_MAX}. State the "
        f"new n and the REGIME it was measured in, and confirm it is not "
        f"pooled with the serial runs")
    assert _ci.FULL_CARD_PICKS_3SD == 107.0
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "n = 0 post-change runs" in src, (
        "the reason the constant was not re-derived is no longer recorded")


def test_a_RE_DERIVATION_on_a_SMALL_n_is_refused_by_a_STATED_MINIMUM():
    """Replacing an n=4 constant with an n=5 one is not progress, and this
    project measured why: the weekday lateness series moved mean+3sd
    417.5 -> 428.9 on a SINGLE new point, and the margin went +40 -> +37.
    """
    assert _ci.MIN_N_TO_REDERIVE_PICKS_BOUND >= 10, (
        f"the minimum n fell to {_ci.MIN_N_TO_REDERIVE_PICKS_BOUND}; a 3sd "
        f"bound on single digits is not a bound")
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "417.5 -> 428.9" in src, (
        "the measured instability that justifies the minimum is no longer "
        "recorded beside it")


def test_the_FEASIBLE_ROUTE_is_recorded_and_says_NOT_TO_POOL():
    """Full-card weekdays arrive at 0.4/week (6 of 73 days, measured
    2026-10-21), so n=10 full-card runs is ~25 weeks. The route that works is
    all-card time-to-picks, which arrives at ~5/week."""
    plan = _ci.REDERIVE_PICKS_BOUND_PLAN
    assert "ALL-CARD" in plan, "the plan does not name the feasible population"
    assert "do NOT" in plan and "pool" in plan, (
        "the plan does not forbid pooling the serial and parallel regimes")
    assert "max and mean+3sd" in plan, (
        "the plan does not require BOTH figures to be reported")
    assert "MIN_N_TO_REDERIVE_PICKS_BOUND" in plan, (
        "the plan does not reference the minimum it has to clear")
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "0.4" in src and "25 weeks" in src, (
        "the measured full-card rate that makes the full-card route "
        "infeasible is no longer recorded")
