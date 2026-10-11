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


# ═══════ THE 12:00 / 14:30 DISCREPANCY, SIZED — and neither is the deadline ══

#: MEASURED 2026-10-10 from the production `matches` table, clean rows only
#: (sub-second `match_date` excluded as the known ingestion-timestamp phantom),
#: 2026-02-01 onward, `world/*` excluded. 148 weekdays with fixtures.
#:
#: date         earliest UTC  fixtures  what it is
#: 2026-06-18        09:59        1     finland/veikkausliiga, Midsummer week
#: 2026-04-06        10:15       95     EASTER MONDAY
#: 2026-04-03        11:30       61     GOOD FRIDAY
#: 2026-05-25        12:30       13     Whit Monday
#: 2026-05-01        13:00       26     Labour Day
#: ... 13 more between 13:00 and 14:00 (turkey/greece/romania midweek)
WEEKDAY_EARLIEST_OBSERVED = 9 * 60 + 59
WEEKDAY_EARLIEST_ON_A_FULL_CARD = 10 * 60 + 15      # Easter Monday, 95 fixtures
WEEKDAY_DAYS_MEASURED = 148
WEEKDAY_DAYS_BEFORE_1430 = 18
WEEKDAY_DAYS_BREACHING_READY_3SD = 2


def test_the_1430_figure_is_a_PERCENTILE_not_a_minimum():
    """THE DISCREPANCY'S CAUSE. Neither 12:00 nor 14:30 is the deadline.

    The independent reference gave 12:00 UTC for the weekday earliest kickoff
    and our own data gave 14:30. The weekend figure reproduced EXACTLY (Sunday
    13:15 Kyiv = 10:15 UTC), so the reference is not uniformly stricter and the
    disagreement is localised.

    LOCALISED TO TWO FIXTURES: the reference's only sub-14:30 weekday entries
    are `spain/laliga2` on Monday 2026-10-12 at 12:00 and 14:15 UTC. Everything
    else it lists on a weekday is 15:00 or later.

    AND OUR DATA AGREES WITH IT. We hold clean weekday kickoffs from 09:59 UTC
    — including `spain/laliga2` at 14:15 on Friday 2026-05-01, the same shape as
    the reference's 14:15 entry. So this is NOT "our data is wrong" and NOT "the
    reference covers leagues we do not configure": `spain/laliga2` is one of the
    configured 30.

    IT IS THE THIRD READING. Both are right about different things, because the
    weekday earliest kickoff is BIMODAL:

        ordinary weekday   16:45 UTC (European competition) or 15:00-19:00
        HOLIDAY weekday    10:15-14:15 UTC — it behaves like a weekend

    18 of 148 measured weekdays (12.2%) start before 14:30. So **14:30 is
    roughly the 88th percentile of weekdays, not their minimum** — a figure
    that describes a typical weekday and was used as a bound.
    """
    assert WEEKDAY_DAYS_BEFORE_1430 / WEEKDAY_DAYS_MEASURED > 0.10, (
        "fewer than 10% of weekdays now start before 14:30 — re-measure "
        "before treating this finding as current")
    assert WEEKDAY_EARLIEST_OBSERVED < DEADLINE_MIN_BY_DAYTYPE["weekday"], (
        "the observed weekday minimum is no longer earlier than the configured "
        "weekday deadline, so the deadline may now be a real bound")
    assert WEEKDAY_EARLIEST_OBSERVED < DEADLINE_MIN_BY_DAYTYPE["weekend"], (
        "a weekday has started EARLIER than the weekend deadline — the day-type "
        "split no longer describes the data")


def test_the_weekday_cron_BREACHES_on_a_holiday_weekday():
    """THE MARGIN WE ARE DEFENDING IS NOT THE REAL ONE ON ~1.4% OF WEEKDAYS.

    `0 4` has ready(3sd) at 10:42 UTC. Easter Monday 2026-04-06 carried 95
    fixtures with the earliest at 10:15, so picks would have been delivered
    **27 minutes AFTER the first kickoff**. 2026-06-18 is worse at 09:59
    (−43 min) but carried one fixture.

    This is recorded as an EXPOSURE, not fixed, and it changes no schedule
    today: the next weekday holiday with an early card is **Easter 2027
    (Friday 2027-04-02 and Monday 2027-04-05)**. 2026-12-26 (Boxing Day) falls
    on a SATURDAY and so uses the unchanged weekend cron; 2027-01-01 is a Friday
    but the festive programme's earliest is ~12:30, which clears.

    THE PRINCIPLED REMEDY IS THAT THE DEADLINE IS DATA, NOT A CONSTANT. All
    three figures — 10:15, 12:00, 14:30 — are samples of a quantity that varies
    by date, and the margin should be computed against the DISCOVERED earliest
    kickoff for the upcoming day. That is a design change and it goes to Niki.
    """
    cron = current_crons_by_daytype()["weekday"]
    ready_3sd = cron + DELAY_MEAN + 3 * DELAY_SD + DUR_MEAN + 3 * DUR_SD
    ready_max = cron + DELAY_MAX + DUR_MAX

    margin_full_card = WEEKDAY_EARLIEST_ON_A_FULL_CARD - ready_3sd
    assert margin_full_card < 0, (
        f"Easter Monday's 10:15 kickoff now clears ready(3sd) by "
        f"{margin_full_card:+.0f} min — if the cron moved earlier, update this "
        f"test and the ledger rather than leaving a stale exposure on file")
    assert margin_full_card > -60, (
        f"the breach has grown to {margin_full_card:+.0f} min; below -60 this "
        f"stops being a holiday edge case")

    # And the ordinary-weekday margin is genuinely comfortable - the exposure is
    # the TAIL, not the typical day.
    assert DEADLINE_MIN_BY_DAYTYPE["weekday"] - ready_max >= MIN_ACCEPTABLE_MARGIN
    assert WEEKDAY_DAYS_BREACHING_READY_3SD / WEEKDAY_DAYS_MEASURED < 0.02, (
        "more than 2% of weekdays now breach — this is no longer a tail")


def test_the_phantom_signature_is_what_made_the_raw_count_misleading():
    """Sub-second `match_date` is an ingestion timestamp, not a kickoff.

    The raw sub-14:30 weekday population is dominated by it —
    finland/veikkausliiga 111 of 112, netherlands/eredivisie 20 of 20,
    europe/champions-league 19 of 19 (all at 07:57-07:58). Counting those as
    early kickoffs would have put the weekday deadline at 07:53 and the whole
    schedule would have been sized against an ingestion artefact.

    The clean figures above exclude them, which is why the answer is a
    holiday-shaped tail rather than a contamination finding. The contamination
    census does NOT need re-running against this signature: it is the same
    sub-second `match_date` defect already sized at 510 rows, and these rows
    are its known residue rather than a new population.
    """
    assert WEEKDAY_EARLIEST_OBSERVED > 7 * 60 + 58, (
        "the weekday minimum has dropped into the phantom band (07:57-07:58) — "
        "the clean filter is no longer excluding ingestion timestamps")


# ═════ PART D — mis-attribution produces a figure with the wrong magnitude ══

def test_attribution_is_by_the_runs_OWN_day_of_week():
    """The `closing-lines` censoring problem one level up.

    With two crons on one workflow, nearest-preceding attribution is ambiguous.
    Driven with the two concrete errors rather than asserted:

        a WEEKEND 04:31 start against `0 4`  ->  31 min  (true 271, -240)
        a WEEKDAY 08:31 start against `0 0`  -> 511 min  (true 271, +240)

    Both are "an observation" only if the attribution rule is right; otherwise
    they are an artefact of the rule, and a ceiling or a magnitude imposed by a
    rule is not an observation.
    """
    import datetime as dt

    def lag(start_min, cron_min):
        c = dt.datetime(2026, 10, 12) + dt.timedelta(minutes=start_min)
        sched = c.replace(hour=cron_min // 60, minute=cron_min % 60,
                          second=0, microsecond=0)
        if c < sched:
            sched -= dt.timedelta(days=1)
        return (c - sched).total_seconds() / 60.0

    weekend_start, weekday_start = 4 * 60 + 31, 8 * 60 + 31
    assert lag(weekend_start, 0) == 271.0
    assert lag(weekend_start, 4 * 60) == 31.0, (
        "a weekend run attributed to the weekday cron no longer understates "
        "its lateness by 240 min — re-derive the error before trusting this")
    assert lag(weekday_start, 4 * 60) == 271.0
    assert lag(weekday_start, 0) == 511.0

    # the shipped collector classifies by the run's own weekday
    import inspect

    from scripts.ci_audit import collect_lag_series_by_daytype

    src = inspect.getsource(collect_lag_series_by_daytype)
    assert 'c.date().weekday() >= 5' in src, (
        "the series no longer classifies by the run's own day of week")
    assert 'carried.get(day_type) != cron_now' in src, (
        "membership no longer requires the run's OWN headSha to have carried "
        "the cron — pooling two regimes is the `390a4be` error")


def test_the_attribution_RULE_is_stated_in_the_output():
    """A figure that depends on a rule carries the rule."""
    from scripts.ci_audit import format_series_census

    text = format_series_census({"weekend": [], "weekday": []},
                                {"weekend": 0, "weekday": 240})
    assert "ATTRIBUTION" in text
    assert "DAY OF WEEK" in text
    assert "-240" in text and "+240" in text, (
        "the output does not state the magnitude of the error the rule avoids")


def test_the_weekday_series_is_still_UNEVALUATED_until_a_run_fires():
    """Confirmed 2026-10-11: no weekday run has fired under `0 4`.

    The split landed 2026-10-10 ~17:00 and 10-11 is a Sunday, so the first
    weekday run under the new cron is Monday 2026-10-12. Until then the series
    is n=0 and the report must say UNEVALUATED, not clear and not zero.
    """
    from scripts.ci_audit import (
        format_series_census, schedule_margins_by_daytype,
    )

    out = schedule_margins_by_daytype({"weekend": [(None, 271.0)] * 4,
                                       "weekday": []},
                                      {"weekend": 0, "weekday": 240})
    assert out["weekday"]["n"] == 0
    assert "weekday" in out["unevaluated"]
    census = format_series_census({"weekend": [(None, 271.0)] * 4,
                                   "weekday": []},
                                  {"weekend": 0, "weekday": 240})
    assert "n=0 UNEVALUATED, not clear" in census
