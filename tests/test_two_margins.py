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
