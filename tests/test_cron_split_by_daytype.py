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


def test_the_FALLBACK_is_ONE_cron_covering_EVERY_day():
    """STAGE 34, 2026-11-21. Was: two crons partitioning the week.

    The card is published by ONE external dispatch at 10:30 Europe/Sofia on
    every day, so the cron is a single late fallback for a missed dispatch,
    covering all seven days. Two runs on one day are no longer an overlap
    risk: the workflow's `once` gate stops the second before it starts."""
    crons = _crons()
    assert crons == ["0 11 * * *"], crons
    assert crons[0].split()[4] == "*", "the fallback must cover every day"


def test_the_FALLBACK_starts_only_AFTER_the_dispatched_run_has_FINISHED():
    """STAGE 34. Was: each cron the latest that cleared 60 minutes against a
    lateness series. That sizing is retired with the cron's role: no cron
    hour fixes a delay that DRIFTS (weekday `0 1`, n=21, +5.28 min/day).

    The fallback's EARLIEST possible start is its nominal minute. It must be
    after the 10:30 dispatched run has finished on BOTH seasons -- the
    dispatch is 08:30 UTC in winter, 07:30 UTC in summer -- or a fallback
    could take the card before the dispatch. Longest full run, `measured`
    2026-11-21: 77.0 min (n=38 since 2026-10-22)."""
    import datetime as _d
    from scripts.schedule_deadline import dispatch_utc_min
    fallback = current_crons_by_daytype()["weekday"]
    assert fallback == current_crons_by_daytype()["weekend"] == 660
    RUN_MAX = 77.0
    for day in (_d.date(2026, 12, 15), _d.date(2027, 6, 15)):  # EET, EEST
        clearance = fallback - (dispatch_utc_min(day) + RUN_MAX)
        assert clearance >= 60, (
            f"{day}: the fallback starts only {clearance:.0f} min after the "
            f"dispatched run's longest observed end")


def test_the_HEADROOM_IS_RECORDED_AS_UNSPENT_with_its_n_and_its_DATE():
    """The consumer of `HEADROOM_NOT_SPENT`, and rule 2 in assertion form.

    A record that nothing reads is a producer with no consumer, which is what
    the PNC scan caught when this constant was first written.
    """
    from scripts.schedule_deadline import (HEADROOM_NOT_SPENT as H,
                                           STRUCTURAL_ACK)
    for dtype in ("weekday", "weekend"):
        assert H[dtype]["headroom"] == (
            H[dtype]["latest_clearing"] - H[dtype]["deployed_cron"])
        assert H[dtype]["headroom"] > 0, (
            f"{dtype} has no headroom recorded; if it was spent, say so")

    # RULE 2: headroom against a breached floor is not surplus. The record
    # must name the breach it is being withheld against.
    assert "STRUCTURAL" in H["why_not_spent"].upper()
    for frag in ("39.3", "59.4"):
        assert frag in H["why_not_spent"], (
            f"the record does not name the {frag}-minute deficit it is "
            f"withheld against")
    # and those deficits must still be real
    for dtype, fig in (("weekday", -39.3), ("weekend", -59.4)):
        got = STRUCTURAL_ACK[dtype]["measured_at"]
        assert abs(got - fig) < 0.2, (
            f"{dtype}: the structural deficit is now {got:.2f}, not the "
            f"{fig} the headroom is withheld against. If the floor stopped "
            f"being breached, the headroom becomes spendable -- re-decide.")

    # THE n, AND ITS DATE, DERIVED NOT PICKED
    w = H["spendable_when"]
    assert w["n"] == 100, w
    assert "5/(1-0.95)" in w["basis"] or "quantile_is_usable" in w["basis"], (
        "the n is not tied to the quantile rule that produced it")
    assert w["weekday_projected"] == "2027-02-27"
    assert w["weekend_projected"] == "2027-05-17"
    assert w["weekday_rate_per_week"] > 0 and w["weekend_rate_per_week"] > 0
    # PROVENANCE: a date and a grain, which is what rule 5 asks for -- not
    # the literal word "measured", which my first version checked for and
    # which the string does not contain.
    prov = w["rate_measured"]
    assert "2026-" in prov and "grain=" in prov, (
        f"the arrival rate has no date and grain, so the projected date is a "
        f"guess: {prov!r}")


def test_ONE_dispatch_time_serves_BOTH_day_types_and_the_weekend_cost_is_RECORDED():
    """STAGE 34. Was: the weekend cron earlier than the weekday one.

    THE OPERATOR'S DECISION: one dispatch at 10:30 local every day. The
    weekend's earliest kickoff (12:00 local) is 30 minutes earlier than the
    weekday's, so at 10:30 its margin is below the registered 60 minutes.
    That cost is acknowledged against its EXPOSURE -- a kickoff before picks
    are ready -- never against the margin: ACKNOWLEDGED while picks are ready
    before 12:00, EXPOSED from 12:00."""
    import datetime as _d
    from scripts.schedule_deadline import (DISPATCH_EXPOSURE_MEASURED,
                                           dispatch_exposure_ack)
    import scripts.ci_audit as _ci
    sunday = _d.date(2026, 11, 22)
    ok = dispatch_exposure_ack(sunday, 2.0, _ci.FULL_CARD_PICKS_3SD, None)
    assert ok["state"] == "ACKNOWLEDGED", ok
    # known-wrong control: a duration that puts picks at 12:00 local is EXPOSED
    bad = dispatch_exposure_ack(sunday, 0.0, 90.0, None)
    assert bad["state"] == "EXPOSED", bad
    # and the measured exposure that the acknowledgement quotes
    assert DISPATCH_EXPOSURE_MEASURED["exposed_at_audit_term"] == {
        "weekend": 0, "weekday": 0}
    assert dispatch_exposure_ack(_d.date(2026, 11, 23), 2.0, 64.59,
                                 None)["state"] == "NOT_APPLICABLE"


@pytest.mark.live
def test_the_DISPATCH_is_RUNNING_and_the_weekday_margin_CLEARS_or_is_ACKNOWLEDGED():
    """STAGE 34, 2026-11-21. Was: the weekday CRON clears the margin against
    the series it produces. The cron no longer publishes the card; the 10:30
    dispatch does, so the margin is read from the DISPATCHED runs' delays.

    A LIVE CHECK: it reads `gh run list --event workflow_dispatch`.
      * no dispatched run measured past the grace date -> FAILS: the
        external dispatch job is not dispatching, and the card is being left
        to the late fallback. That is the failure this exists to catch.
      * measured -> the weekday margin (dispatch minute + delays + the SERIAL
        duration term, the audit's own definition) clears 60, or the
        mechanism acknowledges it while the no-review series is below n=10.
    """
    import datetime as _d
    import scripts.ci_audit as _ci
    from scripts.schedule_deadline import DISPATCH_SINCE, dispatch_utc_min
    disp = _ci.collect_dispatch_series()
    assert not disp["unreadable"], (
        "UNEVALUATED, not clear: `gh run list` FAILED (no auth, or a dead "
        "GH_TOKEN), so the dispatched runs could not be listed -- 'could not "
        "look', not a fact about the dispatch. CLEARING ACTION: give this job "
        "a live token")
    GRACE = DISPATCH_SINCE + _d.timedelta(days=3)
    if not disp["series"]:
        # THE TWO EMPTIES ARE NOT THE SAME EMPTY (the 2026-11-05 lesson). A
        # `gh` with no auth or a dead GH_TOKEN now makes the reader return
        # `unreadable` -- its RETURNCODE decides, closing the residual the
        # previous version of this reminder named -- so reaching here means
        # the runs WERE listed and none is a dispatch yet.
        assert _d.date.today() <= GRACE, (
            f"no dispatched run measured since {DISPATCH_SINCE} and the grace "
            f"ended {GRACE}: the external 10:30 dispatch job is NOT "
            f"dispatching, so every card is the late fallback. CLEARING "
            f"ACTION: check the job is enabled and its token valid")
        pytest.skip(
            f"UNEVALUATED, not clear: the runs were listed and no dispatched run "
            f"exists yet, within the grace to {GRACE} while the operator "
            f"enables the external job")
    delays = [d for _day, d, _r in disp["series"]]
    if len(delays) < 2:
        delays = delays * 2          # n=1: max and mean coincide; stated
    tomorrow = _d.date.today() + _d.timedelta(days=1)
    binding = _ci.serial_term_weekday_margin(delays, dispatch_utc_min(tomorrow))
    regime_n = (_ci.removal_regime_picks_series().get("n")
                if binding < MIN_ACCEPTABLE_MARGIN else 0)
    ack = _ci.weekday_margin_shortfall_is_acknowledged(binding, regime_n)
    assert ack["state"] in ("CLEAR", "ACKNOWLEDGED"), (
        f"the weekday margin on the dispatched start is {binding:+.0f}m "
        f"against {MIN_ACCEPTABLE_MARGIN} and the acknowledgement is "
        f"{ack['state']}: {ack['why']}")


def test_the_two_deadlines_are_distinct_and_the_WORSE_is_the_single_value():
    """Callers that do not say which day type they mean get the conservative
    answer, never a wrong one."""
    # ── THE VALUES MOVED 2026-10-19: percentiles -> DERIVED FLOORS. ───────
    #
    # 12:00 / 10:15 were percentiles and the fallback on 1087 of 1090 days,
    # the weekday one two and a half hours laxer than our own measured
    # earliest kickoff. That gap WAS the coverage-exposure tail. The fallback
    # is now `DEADLINE_FLOOR`, derived per day type as
    # `ceil_15(ready_binding + 60 registered)` from each day type's OWN
    # series: 09:45 weekday, 09:15 weekend.
    # RE-ANCHORED 2026-10-20: the deadline is the OBSERVED EARLIEST KICKOFF (weekday 09:30, weekend 09:45), not ceil_15(ready+60). The self-anchored floor was a TAUTOLOGY -- it made margin>=60 true by construction -- and freezing it against a live series breached 60 within a day when mean+3sd drifted +11.4m on one new observation.
    # ── LOCALISED 2026-10-22, SO THE VALUES ARE A FUNCTION OF THE DAY. ───
    #
    # The anchors are LOCAL times (weekday 12:30, weekend 12:00) converted
    # per day, so there is no single UTC pair to assert. BOTH SIDES, as the
    # DST work established: a one-sided assertion cannot tell a derived value
    # from a literal that happens to match today.
    import datetime as _d

    from scripts.schedule_deadline import observed_earliest_min
    for day, wd, we in ((_d.date(2026, 10, 24), 9 * 60 + 30, 9 * 60 + 0),
                        (_d.date(2026, 10, 26), 10 * 60 + 30, 10 * 60 + 0)):
        assert observed_earliest_min("weekday", day) == wd, day
        assert observed_earliest_min("weekend", day) == we, day
    assert SCHEDULE_DEADLINE_MIN == min(DEADLINE_MIN_BY_DAYTYPE.values())
    # ── AND THE ORDERING INVERTED, WHICH IS WHAT THIS TEST ASKED FOR. ────
    #
    # Its old message said "re-derive which day type binds before trusting
    # the single value", and localising did exactly that. In LOCAL time the
    # weekend minimum is 12:00 against the weekday's 12:30, so the WEEKEND
    # is the earlier floor -- on both sides, permanently, not seasonally.
    # The retired UTC pair said the opposite (09:30 < 09:45) because both
    # figures were measured in SUMMER on days that happened to differ.
    #
    # The MARGIN still fires on the weekday, because the weekday DELAY is
    # worse. Which deadline is earlier and which margin is tighter are two
    # different questions, and conflating them is what the old ordering did.
    for day in (_d.date(2026, 10, 24), _d.date(2026, 10, 26)):
        assert observed_earliest_min("weekend", day) <             observed_earliest_min("weekday", day), (
            f"on {day} the weekend deadline is no longer the earlier one — "
            f"re-derive which day type binds before trusting the single value")


def test_a_day_type_with_no_series_is_NOT_COMPUTED_and_says_it_does_not_alarm():
    """The weekday series is EMPTY the day the split lands.

    Membership requires the run's own headSha to carry the cron in force, and
    no weekday run has fired under `0 4` yet. Reporting that as a passing
    margin is exactly "absent instrumentation is not a zero".

    RENAMED 2026-10-22. The old name ended in `UNEVALUATED_not_safe` and the
    line it pinned said "not safe" while alarming on nothing -- a safety
    claim from a line that cannot evaluate safety. It was reworded rather
    than promoted to an alarm, because this state is cleared by the next
    scheduled run firing, and an alarm cleared by waiting teaches waiting.
    Its two siblings (an unreadable `gh workflow list`, a missing cron) are
    cleared by an EDIT and do now alarm.
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
    assert "margin NOT COMPUTED" in text, text
    assert "neither clear nor breached" in text, text
    assert "not safe" not in text, (
        "the safety claim is back in a line that alarms on nothing")


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
    # THE CONSERVATIVE DEFAULT IS THE WEEKEND AGAIN, as of the 2026-10-22
    # localisation. It has now moved twice: the retired percentiles made the
    # weekend bind, re-anchoring to the observed earliest kickoff made the
    # weekday bind, and localising the anchors moved it back -- because in
    # LOCAL time the weekend minimum (12:00) is earlier than the weekday's
    # (12:30), on both sides of a transition. The DEFAULT is still `min(...)`,
    # so it is correct by construction; what keeps moving is which stratum
    # that picks out, which is why this asserts the relation and not a value.
    assert m["deadline_min"] == DEADLINE_MIN_BY_DAYTYPE["weekend"]
    assert m["deadline_min"] == min(DEADLINE_MIN_BY_DAYTYPE.values())


# ───────────────── the six date-derived sites, re-enumerated ────────────────

def test_the_FRIDAY_to_SATURDAY_cutoff_on_the_DISPATCHED_run():
    """`max_days_ahead` is `now + 1 day`, so a Friday run's cutoff reaches
    into Saturday. STAGE 34: the Friday card is dispatched at 10:30 local --
    08:30 UTC in winter -- and Saturday's earliest kickoff is 12:00 local
    (10:00 UTC in winter). The run's cutoff, a few minutes in, clears it.

    THE FALLBACK'S COST, STATED: on a missed-dispatch Friday the run starts
    11:00 UTC plus GitHub's lateness (up to 7h38m seen), so its cutoff
    reaches Saturday ~18:43 UTC and admits Saturday fixtures. That is a
    fallback-day cost, and it is why the fallback is a safety net only."""
    import datetime as _d
    from scripts.schedule_deadline import (dispatch_utc_min,
                                           observed_earliest_min)
    friday = _d.date(2026, 11, 27)
    cutoff = dispatch_utc_min(friday) + 5
    saturday_earliest = observed_earliest_min("weekend",
                                              friday + _d.timedelta(days=1))
    assert saturday_earliest - cutoff >= 60, (saturday_earliest, cutoff)
    fallback_cutoff = 660 + 458.8 + 5
    assert fallback_cutoff > saturday_earliest, (
        "the stated fallback cost no longer holds -- re-derive it")


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


def test_the_settlement_staleness_band_on_the_DISPATCHED_run():
    """The `now - 24h` boundary lands on D-1 at the run's start. STAGE 34:
    the dispatched run starts 08:30 UTC (winter) / 07:30 (summer) plus
    minutes, before the earliest kickoff of either day type (10:00 UTC
    winter weekend), so the band holds no kickoff.

    THE FALLBACK'S COST, STATED: a fallback run starts ~15:00-18:40 UTC, a
    band that DOES contain kickoffs; on a missed-dispatch day the settlement
    boundary falls inside a match window. Safety net only."""
    import datetime as _d
    from scripts.schedule_deadline import (DISPATCH_WINDOW_MIN,
                                           dispatch_utc_min,
                                           observed_earliest_min)
    for day in (_d.date(2026, 12, 12), _d.date(2027, 6, 12)):   # Saturdays
        band_hi = dispatch_utc_min(day) + 30      # realistic dispatch delay
        assert band_hi < observed_earliest_min("weekend", day), day
    assert DISPATCH_WINDOW_MIN == 180


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
    # ── RE-POINTED 2026-10-19 AT THE RE-MEASURED MINIMUM. ─────────────────
    #
    # `WEEKDAY_EARLIEST_OBSERVED` is 09:59, measured over 148 days from
    # 2026-02-01. Over the FULL record (n=715 weekday days, 2022-07-05
    # onward) the weekday minimum is **09:30** -- 2023-10-26,
    # `bulgaria/efbet-league`. A minimum is a function of its window and a
    # longer window can only lower it.
    #
    # The derived floor is 09:45, which is BELOW the old 09:59 figure and
    # ABOVE the re-measured 09:30. So the claim this test makes -- the
    # deadline is not a real bound -- still holds, but only against the
    # current measurement, and the 15-minute gap is the residual exposure
    # recorded in `COVERAGE_EXPOSURE`.
    # BOTH SIDES. The anchor is 12:30 LOCAL since 2026-10-22, so its UTC
    # minute is 09:30 in summer and 10:30 in winter; asserting one of them
    # pins the season, not the anchor.
    import datetime as _d

    from scripts.schedule_deadline import (OBSERVED_EARLIEST_LOCAL_MIN,
                                           observed_earliest_min)
    assert OBSERVED_EARLIEST_LOCAL_MIN["weekday"] == 12 * 60 + 30, (
        "the re-measured weekday minimum moved; re-derive the floor, which is "
        "computed from the ready time and checked against this value")
    assert observed_earliest_min("weekday", _d.date(2026, 10, 24)) == 9 * 60 + 30
    assert observed_earliest_min("weekday", _d.date(2026, 10, 26)) == 10 * 60 + 30
    # IT IS NOW EXACTLY THE BOUND, by construction: the deadline IS the
    # observed minimum, so the coverage exposure is zero and the margin
    # carries the whole trade. That is the point of the re-anchoring, and it
    # holds per day rather than per constant since the localisation.
    #
    # ON FIXED DAYS, NOT `date.today()`. The first version of this line read
    # the clock, and the capture gate refused the whole gating selection for
    # it -- correctly: a clock-reading test in that path has cost this
    # project seven cards. The property is "the floor IS the observed
    # minimum", which holds for any day, so two named days prove it.
    from scripts.schedule_deadline import deadline_floor_for
    for _day in (_d.date(2026, 10, 24), _d.date(2026, 10, 26)):
        assert observed_earliest_min("weekday", _day) ==             deadline_floor_for("weekday", _day)[0]
    assert WEEKDAY_EARLIEST_OBSERVED > deadline_floor_for(
        "weekday", _d.date(2026, 6, 18))[0], (
        "the 2026-06-18 kickoff at 09:59 is no longer later than that day's "
        "localized deadline; the full-record minimum is 09:30")


def test_the_DISPATCH_clears_a_holiday_weekday_EASTER_2027():
    """STAGE 34. Easter Monday 2027-04-05 (EEST): an early holiday card,
    earliest kickoff 10:15 UTC. The dispatch is 07:30 UTC; with a few
    minutes' delay and the audit's duration term (64.59) picks are ready
    ~08:37 UTC, ninety-odd minutes before the first kickoff."""
    import datetime as _d
    import scripts.ci_audit as _ci
    from scripts.schedule_deadline import dispatch_utc_min
    easter_monday = _d.date(2027, 4, 5)
    ready = dispatch_utc_min(easter_monday) + 2.0 + _ci.FULL_CARD_PICKS_3SD
    assert (10 * 60 + 15) - ready >= MIN_ACCEPTABLE_MARGIN, ready


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


# ─────────────────────────────────────────────────────────────────────────────
# THE MEMBERSHIP RULE DEGRADED SILENTLY — measured 2026-10-11
#
# `UNEVALUATED: no runs under the current cron yet` was printed in CI for a
# window that contained FOURTEEN scheduled runs. The rule reads each run's cron
# from its own headSha with `git show <sha>:path`, `_sh` returns "" when that
# fails, and `actions/checkout` defaults to `fetch-depth: 1` — so outside HEAD
# every sha was unreadable and every run was dropped.
#
# MEASURED on one variable, same repo and same `gh` query:
#
#     full clone     weekend n=4   0 runs excluded
#     --depth 1      weekend n=0   14 runs excluded, reported as "no runs yet"
#
# It had looked healthy for weeks because every run in the window shared the
# then-HEAD sha; five commits in two days was enough to empty it. An exclusion
# wearing the label of an absence is worse than either one alone.
# ─────────────────────────────────────────────────────────────────────────────


def test_the_audit_checks_out_FULL_history_and_the_reason_is_recorded():
    """`fetch-depth: 0`, pinned where it is needed and not where it is not.

    This is the guard that replaces a runtime alarm: the degradation cannot
    recur without this test failing, which is tighter than noticing a line.
    """
    import pathlib

    import yaml

    wf = yaml.safe_load(pathlib.Path(".github/workflows/ci-audit.yml")
                        .read_text(encoding="utf-8"))
    steps = list(wf["jobs"].values())[0]["steps"]
    checkout = [s for s in steps if "checkout" in str(s.get("uses", ""))]
    assert len(checkout) == 1, checkout
    assert (checkout[0].get("with") or {}).get("fetch-depth") == 0, (
        "ci-audit is back on a shallow clone. The scheduler series reads each "
        "run's cron from its own headSha, so a shallow clone makes every "
        "non-HEAD run unreadable and the series reports itself as empty.")


def test_an_UNREADABLE_sha_is_not_a_DIFFERENT_cron():
    """The third state, at the only place it can be distinguished.

    `{'weekend': None, 'weekday': None}` says "this sha carried no recognisable
    cron". `None` says "this sha could not be read at all". Collapsing them is
    what made the exclusion invisible.
    """
    from scripts.ci_audit import current_crons_by_daytype

    assert current_crons_by_daytype("0" * 40) is None, (
        "an unreadable sha again returns a dict, which the membership rule "
        "cannot tell from a sha that carried a different cron")
    # the control: HEAD is readable and yields real crons
    import subprocess

    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          encoding="utf-8", errors="replace").stdout.strip()
    carried = current_crons_by_daytype(head)
    assert isinstance(carried, dict) and carried.get("weekend") is not None


def test_the_EXCLUDED_runs_are_COUNTED_and_carry_a_clearing_action():
    """Hard rule: an exclusion imposed by an attribution rule is not an
    observation, and every alarm names an action a person can take."""
    from scripts.ci_audit import UNREADABLE_KEY, format_series_census

    degraded = format_series_census(
        {"weekend": [], "weekday": [], UNREADABLE_KEY: ["a" * 40, "b" * 40]},
        {"weekend": 0, "weekday": 240})
    assert "2 run(s) EXCLUDED" in degraded
    assert "DEGRADED census, not a small one" in degraded
    assert "fetch-depth: 0" in degraded, "no clearing action in the text"

    # POSITIVE CONTROL. Without this, the line above would also print on a
    # healthy census and carry no information — `fixtures_zero_active` again.
    clean = format_series_census({"weekend": [], "weekday": []},
                                 {"weekend": 0, "weekday": 240})
    assert "EXCLUDED" not in clean
    assert "READABLE sha" in clean


def test_the_series_census_HAS_A_CONSUMER():
    """PNC-1, found 2026-10-11. NOT "the thirteenth instance": that ordinal
    was accreted, and the 2026-10-12 tally audit put the verified count at
    11 of 14 claimed (docs/rule-tallies.md).

    `format_series_census` carried the attribution rule and the per-series n,
    and nothing in `main` called it — it was exercised only by its own test
    above, so the rule had never been printed in CI even once. A formatter
    whose only caller is its test is a definition, not output.
    """
    import inspect

    import scripts.ci_audit as ci

    src = inspect.getsource(ci)
    calls = [l for l in src.splitlines()
             if "format_series_census(" in l and "def " not in l]
    assert any("print(" in l for l in calls), (
        "format_series_census is no longer printed; the attribution rule and "
        f"the per-series n are produced and discarded. Sites: {calls!r}")


# ── THE LIVE LABEL WAS A LITERAL AND WENT STALE IN ONE DAY ──────────────────
#
#     A label that contradicts its own data is worse than no label.
#
# `format_series_census` printed `LIVE  0 4 * * 1-5` from a hard-coded ternary.
# The weekday cron was reverted to `0 0 * * 1-5` at 566c927, so from that
# commit the report named `0 4` on the same line as a margin (+280m) measured
# under `0 0`. The series ATTRIBUTION was never wrong - it matches each run to
# the cron its own headSha carried - so the only incorrect field was the one a
# reader uses to decide which schedule the numbers describe.
#
# Found 2026-10-18 by reading the audit's own output against the tree.

def test_the_LIVE_cron_label_is_READ_FROM_THE_TREE_not_written_down():
    import pathlib
    import re as _re

    from scripts.ci_audit import format_series_census

    text = format_series_census({"weekend": [(None, 1.0)], "weekday": [(None, 1.0)]},
                                {"weekend": 0, "weekday": 0})
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    tree = set(_re.findall(r"- cron: '([^']+)'", wf))
    live = set(_re.findall(r"^  LIVE\s+(\S+ \S+ \S+ \S+ \S+)\s", text, _re.M))
    assert live == tree, (
        f"the census labels {sorted(live)} as LIVE and the workflow holds "
        f"{sorted(tree)} -- the label is not derived from the schedule it "
        f"claims to describe")


def test_the_label_CANNOT_be_a_LITERAL_in_the_source():
    """The remedy, enforced. A derived label that someone re-hard-codes next
    week is the same defect with a newer date."""
    import inspect

    from scripts.ci_audit import format_series_census

    src = inspect.getsource(format_series_census)
    body = src.split('for day_type in ("weekend", "weekday"):', 1)[1]
    bad = [l for l in body.splitlines()
           if "expr" in l and "* *" in l and not l.strip().startswith("#")]
    assert not bad, (
        f"a cron expression is written into the census body again: {bad}")
    assert "live_exprs" in body, "the label is no longer read from the parse"


def test_the_EXPRESSION_READER_shares_the_PARSE_with_the_minutes():
    """One rule, one definition. A second parser for the same five fields is
    THE HABIT, and it drifts."""
    import scripts.ci_audit as ci

    plain = ci.current_crons_by_daytype()
    pair = ci.current_crons_by_daytype(want_exprs=True)
    assert isinstance(plain, dict) and isinstance(pair, tuple) and len(pair) == 2
    assert pair[0] == plain, "the two call forms disagree about the minutes"
    for day_type, expr in pair[1].items():
        if expr is None:
            assert plain[day_type] is None
            continue
        parts = expr.split()
        assert int(parts[1]) * 60 + int(parts[0]) == plain[day_type], (
            f"{day_type}: the expression {expr!r} and the minutes "
            f"{plain[day_type]} come from one parse and still disagree")


def test_the_EXPRS_are_NOT_a_SENTINEL_KEY_in_a_dict_others_iterate():
    """The landmine the first draft of this fix planted.

    `schedule_margins_by_daytype` iterates `crons.items()` and divides the
    value by 60. A `__exprs__` key holding a dict would have reached that
    division. Caught before shipping, pinned so it cannot be re-introduced.
    """
    import scripts.ci_audit as ci

    assert "__exprs__" not in ci.current_crons_by_daytype()
    m, _e = ci.current_crons_by_daytype(want_exprs=True)
    assert "__exprs__" not in m
    assert all(v is None or isinstance(v, int) for v in m.values()), (
        "a non-integer reached the minutes mapping that other functions "
        "divide by 60")


def test_an_UNREADABLE_tree_returns_None_in_BOTH_call_forms():
    """The third state must survive the new parameter, or the shallow-clone
    fix is undone for one of the two readers."""
    import scripts.ci_audit as ci

    assert ci.current_crons_by_daytype(sha="deadbeefdeadbeef") is None
    assert ci.current_crons_by_daytype(sha="deadbeefdeadbeef",
                                       want_exprs=True) is None
