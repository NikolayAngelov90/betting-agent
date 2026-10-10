"""The daily-picks cron, split by day of week — 2026-10-10, `s5.17`.

NIKI ASKED FOR ~1h30 LATER. A FLAT SHIFT WOULD HAVE BEEN THE WORSE OF TWO
OPTIONS, because **the deadline is the earliest kickoff and it differs by day**.
Measured against an INDEPENDENT reference — the Flashscore fixtures pages,
probed with `max_days_ahead=8`, converted from Europe/Kyiv to UTC after
verifying that offset against the production `matches` table, which stores UTC:

    weekend  10:15 UTC (Sun 10-11, Sun 10-18), 10:30 (Sat 10-10, Sat 10-17)
    weekday  12:00 UTC (Mon 10-12)  <- BINDING
             15:00 (Fri 10-16), 16:45 (Tue/Wed/Thu)

THE REFERENCE DOES NOT REPRODUCE THE 14:30 WEEKDAY FIGURE that was on file. It
gives 12:00, which is 2h30 **stricter**, and the stricter measured value is what
the weekday cron is sized against. Verifying a figure against the data that
produced it is not verification.

THE OFFSET ERROR IS WORTH KEEPING. The probe's timestamps were taken for UTC
for three hours, and the H1 collection cron was set three hours too late as a
result — its band would have opened AFTER kickoff. Caught by comparing twelve
fixtures across three leagues against the DB: `france/ligue-1` reads
15:15/18:45x4 in UTC against 18:15/21:45x4 in the probe. Second instance in
three days of "verify the timezone of the clock that produced it", and the first
where the error pointed the other way.
"""

import pathlib

import pytest
import yaml

from scripts.ci_audit import (
    DEADLINE_MIN_BY_DAYTYPE,
    MARGIN_ALARM_MIN,
    SCHEDULE_DEADLINE_MIN,
    current_crons_by_daytype,
    format_schedule_margins,
    schedule_margin,
    schedule_margins_by_daytype,
)

#: MEASURED 2026-10-10, regime `0 0 * * *`, scheduled events only, n=9.
DELAY_MIN, DELAY_MAX = 3 * 60 + 58, 5 * 60 + 4
DELAY_MEAN, DELAY_SD = 271.0, 18.2
DUR_MEAN, DUR_SD, DUR_MAX = 45.3, 10.9, 72.0

#: STATED BEFORE COMPUTING, and this literal is the record of that.
MIN_ACCEPTABLE_MARGIN = 60


def _crons():
    d = yaml.safe_load(pathlib.Path(
        ".github/workflows/daily-picks.yml").read_text(encoding="utf-8"))
    on = d[True] if True in d else d["on"]
    return [c["cron"] for c in on["schedule"]]


def test_there_are_exactly_two_crons_and_they_PARTITION_the_week():
    """Every day must be covered exactly once. A gap loses a card silently and
    an overlap runs the pipeline twice against one database."""
    crons = _crons()
    assert len(crons) == 2, crons
    covered = {}
    for expr in crons:
        parts = expr.split()
        dow = parts[4]
        days = set()
        for piece in dow.split(","):
            if "-" in piece:
                a, b = piece.split("-")
                days |= set(range(int(a), int(b) + 1))
            else:
                days.add(int(piece) % 7)
        for d in days:
            assert d not in covered, (
                f"day {d} is covered by BOTH {covered.get(d)} and {expr} — the "
                f"pipeline would run twice against one database")
            covered[d] = expr
    assert set(covered) == set(range(7)), (
        f"the week is not fully covered: {sorted(set(range(7)) - set(covered))} "
        f"have no cron, so those days produce no card at all")


def test_the_weekend_cron_is_UNCHANGED():
    """The weekend carries the bigger card AND the earlier deadline, so it is
    the day type with least to gain and most to lose."""
    crons = current_crons_by_daytype()
    assert crons["weekend"] == 0, (
        f"the weekend cron moved to {crons['weekend']} minutes — it was left at "
        f"00:00 deliberately")


def test_the_weekday_cron_is_the_LATEST_that_clears_the_STATED_margin():
    """`0 4` clears 60 minutes; `0 5` does not. Computed, not asserted.

    The threshold was stated before the computation and lives in
    `MIN_ACCEPTABLE_MARGIN` so it cannot be retrofitted to the answer.
    """
    crons = current_crons_by_daytype()
    assert crons["weekday"] == 4 * 60, crons

    deadline = DEADLINE_MIN_BY_DAYTYPE["weekday"]

    def margins(cron_min):
        ready_max = cron_min + DELAY_MAX + DUR_MAX
        ready_3sd = cron_min + DELAY_MEAN + 3 * DELAY_SD + DUR_MEAN + 3 * DUR_SD
        return deadline - ready_max, deadline - ready_3sd

    m_max, m_3sd = margins(crons["weekday"])
    assert min(m_max, m_3sd) >= MIN_ACCEPTABLE_MARGIN, (
        f"the shipped weekday cron gives {m_max:+.0f}/{m_3sd:+.0f} against a "
        f"stated {MIN_ACCEPTABLE_MARGIN}-minute minimum")

    nxt_max, nxt_3sd = margins(crons["weekday"] + 60)
    assert min(nxt_max, nxt_3sd) < MIN_ACCEPTABLE_MARGIN, (
        f"one hour later would ALSO clear the bar ({nxt_max:+.0f}/"
        f"{nxt_3sd:+.0f}) — the shipped cron is not the latest acceptable one, "
        f"so either ship the later one or restate the minimum")


def test_the_two_deadlines_are_distinct_and_the_WORSE_is_the_single_value():
    """Callers that do not say which day type they mean get the conservative
    answer, never a wrong one."""
    assert DEADLINE_MIN_BY_DAYTYPE["weekend"] == 10 * 60 + 15
    assert DEADLINE_MIN_BY_DAYTYPE["weekday"] == 12 * 60
    assert SCHEDULE_DEADLINE_MIN == min(DEADLINE_MIN_BY_DAYTYPE.values())
    assert DEADLINE_MIN_BY_DAYTYPE["weekend"] < DEADLINE_MIN_BY_DAYTYPE["weekday"], (
        "the weekend deadline is no longer the earlier one — re-derive which "
        "day type binds before trusting the single value")


def test_a_day_type_with_no_series_is_UNEVALUATED_not_safe():
    """The weekday series is EMPTY the day the split lands.

    Membership requires the run's own headSha to carry the cron in force, and
    no weekday run has fired under `0 4` yet. Reporting that as a passing
    margin is exactly "absent instrumentation is not a zero".
    """
    out = schedule_margins_by_daytype(
        {"weekend": [(None, 271.0)] * 3, "weekday": []},
        {"weekend": 0, "weekday": 240})
    assert out["weekday"]["n"] == 0
    assert "weekday" in out["unevaluated"]
    assert out["worse"] == "weekend", (
        "an UNEVALUATED day type was chosen as the worse — a missing series "
        "must not win a comparison it did not enter")
    text = format_schedule_margins(out)
    assert "UNEVALUATED" in text and "not safe" in text, text


def test_the_trigger_fires_on_the_WORSE_of_the_two():
    """One number against one deadline was the old shape. With two deadlines
    the trigger must be the worse, not the mean and not the first."""
    good = [(None, 271.0)] * 5
    bad = [(None, 271.0 + 400)] * 5       # a delay that eats the margin
    out = schedule_margins_by_daytype(
        {"weekend": good, "weekday": bad}, {"weekend": 0, "weekday": 240})
    assert out["weekend"]["alarm"] is False
    assert out["weekday"]["alarm"] is True
    assert out["worse"] == "weekday"
    assert out["alarm"] is True, (
        "the aggregate did not alarm although one day type did — the trigger "
        "is not firing on the worse")


def test_the_two_series_are_NEVER_POOLED():
    """A weekday series and a weekend series are distinct populations under one
    change: different crons, different deadlines, their own n.

    The `390a4be` rule, one axis over — a 04:00 weekday run read as a
    four-hour outlier of a 00:00 series.
    """
    out = schedule_margins_by_daytype(
        {"weekend": [(None, 100.0)] * 2, "weekday": [(None, 300.0)] * 7},
        {"weekend": 0, "weekday": 240})
    assert out["weekend"]["n"] == 2 and out["weekday"]["n"] == 7
    assert out["weekend"]["max"] == 100.0 and out["weekday"]["max"] == 300.0, (
        "a lag from one day type leaked into the other's series")


def test_schedule_margin_defaults_to_the_CONSERVATIVE_deadline():
    """An un-dayed caller must not silently get the laxer number."""
    m = schedule_margin([(None, 271.0)] * 3, 0)
    assert m["deadline_min"] == SCHEDULE_DEADLINE_MIN
    assert m["deadline_min"] == DEADLINE_MIN_BY_DAYTYPE["weekend"]


# ───────────────── the six date-derived sites, re-enumerated ────────────────

def test_the_FRIDAY_to_SATURDAY_cutoff_is_the_binding_site():
    """`max_days_ahead` is `now + 1 day`, one-sided, so a later weekday run
    reaches further into D+1.

    Mon-Thu runs have a D+1 weekday whose earliest kickoff is 15:00-16:45 UTC,
    which is never close. The binding case is a FRIDAY run, whose D+1 is a
    SATURDAY with an earliest kickoff of 10:30 UTC. This is the one site where
    moving the weekday cron later could change the discovered-fixture
    population, so it carries a number rather than a verdict.
    """
    cron = current_crons_by_daytype()["weekday"]
    # the cutoff is set a few minutes into the run, not at completion
    cutoff_max = cron + DELAY_MAX + 5
    saturday_earliest = 10 * 60 + 30
    margin = saturday_earliest - cutoff_max
    assert margin > 0, (
        f"a Friday run's cutoff ({cutoff_max // 60:02d}:{cutoff_max % 60:02d}) "
        f"now reaches past Saturday's earliest kickoff 10:30 — the Friday card "
        f"would start admitting Saturday fixtures, which moves the "
        f"discovered-fixture population and is a cohort event")
    assert margin >= 60, (
        f"only {margin} minutes of margin on the Friday->Saturday cutoff; "
        f"below an hour this needs Friday excluded from the weekday cron")


def test_the_settlement_ordering_clears_the_previous_days_last_match():
    """Stated in minutes, not asserted as 'clears'.

    Latest kickoff 19:30 UTC (22:30 Europe/Kyiv, measured), so D-1's last match
    ends ~21:30.
    """
    cron = current_crons_by_daytype()["weekday"]
    start_min = cron + DELAY_MIN
    last_match_ends = 21 * 60 + 30 - 24 * 60      # on D-1
    clearance = start_min - last_match_ends
    assert clearance >= 240, (
        f"a weekday run starts only {clearance} minutes after the previous "
        f"day's last match ended — settlement could run against unfinished "
        f"matches")


def test_the_settlement_staleness_band_contains_ZERO_kickoffs():
    """The `now - 24h` boundary lands on D-1 07:58-09:04 under `0 4`.

    The earliest kickoff of ANY day type is 10:15 UTC, so the band is empty —
    the same reason the 00:00 move was safe, re-checked at the new band rather
    than assumed to carry over.
    """
    cron = current_crons_by_daytype()["weekday"]
    band_lo, band_hi = cron + DELAY_MIN, cron + DELAY_MAX
    earliest_any = min(DEADLINE_MIN_BY_DAYTYPE.values())
    assert band_hi < earliest_any, (
        f"the staleness band {band_lo}-{band_hi} now reaches the earliest "
        f"kickoff {earliest_any} — settlement's 24h boundary would fall inside "
        f"a match window")


def test_the_alarm_threshold_is_unchanged_and_below_the_stated_minimum():
    """The 30-minute trigger stays as the ALARM; 60 is the DESIGN minimum.

    They are different numbers for different purposes, and conflating them
    would either make the alarm hair-trigger or the design slack.
    """
    assert MARGIN_ALARM_MIN == 30
    assert MIN_ACCEPTABLE_MARGIN > MARGIN_ALARM_MIN, (
        "the design minimum is no longer stricter than the alarm, so a "
        "schedule could be shipped already alarming")
