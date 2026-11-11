"""ACKNOWLEDGED AGAINST THE EXPOSURE — 2026-11-07.

A QUANTITY DESCRIBING A RARE EVENT IS ACKNOWLEDGED AGAINST ITS FREQUENCY, NOT
ITS WORST CASE. The structural margin's floor is the earliest kickoff ever
observed over three years, so the margin asks "could we be ready before the
worst day in the record" and the answer is no, every day, for ever. A figure
that cannot improve cannot be acknowledged.

The exposure can:

    exposure(day) = max(0, binding_ready - earliest kickoff THAT DAY)

Measured over 1,015 evaluated days: positive on 10, max 39.3 minutes, and
**ZERO days above the registered 60-minute minimum in any stratum**.

WHAT THIS FILE DEFENDS, because each of these is a way the acknowledgement
could quietly rot:

  1. the threshold CAN be crossed (`over60 > 0`), unlike the margin it
     replaces, and it is DERIVED from the measured single-period movement;
  2. there is NO date review, on purpose — the quantity moves only when a
     mechanism moves, and a date fired on 2026-11-06 against a figure that
     had already improved by 14.8 minutes;
  3. the superseded margin figures are RETIRED WITH their date, floor and n;
  4. `p95 at n>=100` serves the CRON decision and NOT this one, and the two
     are never re-merged.

NOTE ON IMPORTS. `scripts.schedule_deadline` is imported for its constants
only; its database work lives inside functions.
"""
import pytest

from scripts.schedule_deadline import (EXPOSURE_ACK, EXPOSURE_MEASURED,
                                       EXPOSURE_SENSITIVITY,
                                       HEADROOM_NOT_SPENT,
                                       MIN_ACCEPTABLE_MARGIN_MIN,
                                       OBSERVED_EARLIEST_LOCAL_MIN,
                                       exposure_is_acknowledged)


# ── 1. A THRESHOLD THAT CAN BE CROSSED, AND IS DERIVED ───────────────────

@pytest.mark.parametrize("dtype", ["weekday", "weekend"])
def test_the_CURRENT_EXPOSURE_IS_COVERED(dtype):
    assert exposure_is_acknowledged(0, dtype), (
        f"{dtype}: the acknowledgement does not cover its own measured "
        f"figure, so it absorbs nothing and the breach reddens daily")


@pytest.mark.parametrize("dtype", ["weekday", "weekend"])
def test_ONE_DAYS_DETERIORATION_ALARMS(dtype):
    """The driven control. A count's tolerance is zero, so one qualifying day
    is a new fact."""
    assert not exposure_is_acknowledged(1, dtype), (
        f"{dtype}: a single day past the registered minimum is absorbed. "
        f"That is the shape that let a margin breach stand for sixteen days")
    assert not exposure_is_acknowledged(12, dtype), f"{dtype}: materially worse"


@pytest.mark.parametrize("dtype", ["weekday", "weekend"])
def test_UNEVALUATED_ABSORBS_NOTHING(dtype):
    """`could not measure the exposure` is not `the exposure is fine`."""
    assert not exposure_is_acknowledged(None, dtype)


def test_an_UNKNOWN_DAY_TYPE_absorbs_nothing():
    assert not exposure_is_acknowledged(0, "holiday")
    assert not exposure_is_acknowledged(0, "")


def test_the_THRESHOLD_IS_DERIVED_from_the_measured_single_period_movement():
    """Not chosen. The >60m count never moved across nine prefixes."""
    for dtype in ("weekday", "weekend"):
        sv = EXPOSURE_SENSITIVITY[dtype]
        assert sv["largest_over60_move"] == 0, (
            f"{dtype}: the >60m count now moves {sv['largest_over60_move']} "
            f"on a single arriving lateness point, so a tolerance of zero is "
            f"no longer derived and the threshold must be re-derived")
        assert EXPOSURE_ACK[dtype]["tolerance"] == 0
        assert "COUNT" in EXPOSURE_ACK[dtype]["tolerance_basis"].upper()
    assert (EXPOSURE_SENSITIVITY["weekday"]["prefixes"]
            + EXPOSURE_SENSITIVITY["weekend"]["prefixes"]) == 9


def test_KNOWN_WRONG_THE_POSITIVE_DAY_COUNT_WOULD_NOT_WORK():
    """Why the acknowledged quantity is days-over-the-MINIMUM and not
    days-with-any-exposure. One weekend lateness point moved the positive
    count 1 -> 8; the >60m count moved 0. Picking the volatile one would
    have produced an alarm that fires on arrival order."""
    wk = EXPOSURE_SENSITIVITY["weekend"]
    assert wk["largest_pos_move"] >= 7, (
        f"the weekend positive-day count now moves only "
        f"{wk['largest_pos_move']} on one point, so the contrast that "
        f"justified the choice has gone and the choice needs restating")
    assert wk["largest_pos_move"] > wk["largest_over60_move"]
    assert EXPOSURE_ACK["weekend"]["alarm_if"] == "over60 > 0"


# ── 2. NO DATE REVIEW, AND THE MECHANISMS INSTEAD ────────────────────────

@pytest.mark.parametrize("dtype", ["weekday", "weekend"])
def test_there_is_NO_DATE_REVIEW(dtype):
    e = EXPOSURE_ACK[dtype]
    assert "review" not in e, (
        f"{dtype}: a bare `review` date is back. This quantity changes only "
        f"when a mechanism changes, so a date fires on a day when nothing "
        f"has happened -- which is exactly what `margin.schedule` did on "
        f"2026-11-06, against a figure that had improved by 14.8 minutes")
    trig = e["review_trigger"]
    assert "NOT a date" in trig or "as weekday" in trig
    assert "MECHANISM" in trig.upper() or "as weekday" in trig


def test_the_WEEKDAY_TRIGGER_NAMES_EVERY_MECHANISM_that_moves_the_figure():
    trig = EXPOSURE_ACK["weekday"]["review_trigger"]
    for mech in ("cron", "FULL_CARD_PICKS_MAX", "lateness",
                 "flashscore_leagues", "OBSERVED_EARLIEST_LOCAL_MIN"):
        assert mech in trig, (
            f"`{mech}` is no longer named as a review trigger, so a reader "
            f"cannot tell what would make the exposure figure stale")


# ── 3. RETIRED, NOT DELETED ──────────────────────────────────────────────

@pytest.mark.parametrize("dtype,at,floor,n", [
    ("weekday", -39.306530, 750, 11),
    ("weekend", -59.407508, 720, 4),
])
def test_the_SUPERSEDED_MARGIN_keeps_its_date_floor_and_n(dtype, at, floor, n):
    sup = EXPOSURE_ACK[dtype]["superseded"]
    assert sup["measured_at"] == at, "the retired figure was rounded or moved"
    assert sup["floor_min_local"] == floor
    assert sup["lag_n"] == n
    assert sup["measured"] == "2026-11-06"
    assert sup["review_was"] == "2026-11-20"
    assert sup["jitter"] == 11.32
    assert "negative by construction" in sup["why_retired"]
    # the floor it was measured against is the live one
    assert OBSERVED_EARLIEST_LOCAL_MIN[dtype] == floor, (
        f"{dtype}: the live floor is "
        f"{OBSERVED_EARLIEST_LOCAL_MIN[dtype]} but the retired figure was "
        f"measured against {floor}; the retirement record is now misleading")


def test_the_RETIRED_FIGURES_ARE_WORSE_THAN_THE_EXPOSURE_EVER_WAS():
    """The arithmetic that makes the re-specification more than a relabelling.

    The margin reports a 39- and 59-minute breach EVERY DAY. The exposure it
    stands proxy for exceeded nothing: its worst day in three years is 39.3
    minutes and the registered minimum is 60.
    """
    for dtype in ("weekday", "weekend"):
        e = EXPOSURE_ACK[dtype]
        deficit = abs(e["superseded"]["measured_at"])
        assert e["max"] < MIN_ACCEPTABLE_MARGIN_MIN, (
            f"{dtype}: the worst measured exposure {e['max']} now exceeds the "
            f"registered {MIN_ACCEPTABLE_MARGIN_MIN}-minute minimum, so the "
            f"acknowledgement is covering a real breach and must be re-made")
        assert e["over60"] == 0
        if dtype == "weekday":
            assert e["max"] == pytest.approx(deficit, abs=0.1), (
                "the weekday exposure max and the margin deficit are no "
                "longer the same event seen two ways")


def test_the_WEEKEND_ORDERING_INVERTS_and_that_is_recorded():
    """The margin says the weekend is worse; the exposure says it is better.
    An ordering that flips when the quantity is better specified is the
    strongest evidence the old quantity was the wrong one."""
    wd, we = EXPOSURE_ACK["weekday"], EXPOSURE_ACK["weekend"]
    assert abs(we["superseded"]["measured_at"]) > abs(
        wd["superseded"]["measured_at"]), "the margin no longer ranks weekend worse"
    assert we["max"] < wd["max"], "the exposure no longer ranks weekend better"
    assert "inversion" in we["superseded"]["why_retired"].lower()


# ── 4. TWO QUANTITIES, TWO DECISIONS, NEVER RE-MERGED ────────────────────

def test_p95_SERVES_THE_CRON_DECISION_AND_NOT_THE_ACKNOWLEDGEMENT():
    h = HEADROOM_NOT_SPENT
    assert "CRON" in h["p95_serves"].upper()
    assert "ACKNOWLEDGEMENT" in h["p95_does_not_serve"].upper()
    assert h["spendable_when"]["serves"] == (
        "the cron move, NOT the acknowledgement")
    # the cron projection STAYS -- it was never wrong for its own question
    assert h["spendable_when"]["n"] == 100
    assert h["spendable_when"]["weekday_projected"] == "2027-02-27"


def test_the_REASON_p95_IS_NOT_THE_SUCCESSOR_is_the_GRAIN():
    """A count over 674 days is measurable now; a quantile over 11 runs is
    not. That is the whole argument and it must stay legible."""
    why = HEADROOM_NOT_SPENT["p95_does_not_serve"]
    assert "674" in why and "341" in why, (
        "the retirement argument no longer carries the day counts, which is "
        "what makes the exposure measurable at today's n")
    assert "RUNS" in why.upper() and "DAYS" in why.upper()
    assert EXPOSURE_ACK["weekday"]["n"] == 674
    assert EXPOSURE_ACK["weekend"]["n"] == 341


# ── THE MEASUREMENT ITSELF ───────────────────────────────────────────────

def test_the_STRATA_SUM_and_the_HOLIDAY_STRATUM_IS_PRESENT():
    st = EXPOSURE_MEASURED["strata"]
    assert st["weekday"]["n"] + st["weekend"]["n"] == st["ALL"]["n"] == 1015
    assert st["weekday"]["pos"] + st["weekend"]["pos"] == st["ALL"]["pos"]
    for key in ("weekday/holiday", "weekend/holiday"):
        assert key in st, f"the {key} stratum is gone; it failed differently "\
                          f"last time and is reported separately for that reason"
        assert st[key]["over60"] == 0
    # the holiday stratum is a SUBSET, so it cannot exceed its day type
    assert st["weekday/holiday"]["n"] < st["weekday"]["n"]
    assert "assumed" in EXPOSURE_MEASURED["holiday_basis"], (
        "the holiday calendar is no longer labelled `assumed`; no holiday "
        "library is installed and the dates are listed by hand")


def test_UNEVALUATED_DAYS_ARE_REPORTED_and_not_counted_as_zero():
    """Three outcomes, not two. 185 days carry configured-league rows but no
    readable kickoff time, and a day that could not be read is not a day
    with no exposure."""
    assert EXPOSURE_MEASURED["UNEVALUATED_days"] == 185
    assert EXPOSURE_MEASURED["partial_days"] == 678
    total = (EXPOSURE_MEASURED["strata"]["ALL"]["n"]
             + EXPOSURE_MEASURED["UNEVALUATED_days"])
    assert total == 1200, (
        f"evaluated + unevaluated is {total}, not the 1,200 days carrying a "
        f"configured-league row -- some day is in neither bucket")
    assert "NOT counted as zero" in EXPOSURE_MEASURED["UNEVALUATED_why"]


def test_the_POPULATION_IS_THE_CONFIGURED_LEAGUES_and_the_trap_is_recorded():
    """`NOT LIKE 'world/%'` is not the configured set, and the difference was
    a factor of ten on the weekend figure."""
    pop = EXPOSURE_MEASURED["population"]
    assert "flashscore_leagues" in pop
    assert "config.example.yaml" in pop, (
        "the population no longer says it reads the DEPLOYED config; the "
        "local file is overwritten by CI and cannot be the source of truth")
    assert "placeholder" in pop


def test_the_MEASUREMENT_REPRODUCES_THE_FLOORS_OWN_ANCHORS():
    """The check that it measures the right thing. The worst weekday exposure
    equals the margin deficit, and both anchors come from the same event."""
    assert EXPOSURE_MEASURED["binding_ready"]["weekday"] == pytest.approx(
        609.3, abs=0.1)
    assert EXPOSURE_MEASURED["binding_ready"]["weekend"] == pytest.approx(
        605.2, abs=0.1)
    # weekday ready 609.3 against a 09:30 UTC kickoff = 570 -> 39.3
    assert 609.3 - 570 == pytest.approx(
        EXPOSURE_MEASURED["strata"]["weekday"]["max"], abs=0.1)
