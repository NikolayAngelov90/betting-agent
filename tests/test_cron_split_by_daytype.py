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


def test_BOTH_crons_are_the_LATEST_that_CLEAR_the_registered_60():
    """MOVED 2026-10-22, and the weekend moved too.

    It used to assert the weekend cron was UNCHANGED at 00:00, on the
    reasoning that the weekend carries the bigger card and the earlier
    deadline so it had least to gain. That reasoning was right while the two
    deadlines were stored UTC constants. Localising them inverted which floor
    is earlier -- the weekend's is 12:00 local against the weekday's 12:30 --
    and once each day type is sized against its OWN floor and its OWN spread,
    the weekend has 30 minutes to give and the weekday 60.

    THE DERIVATION, and it is re-run here rather than quoted: each cron is the
    latest that clears `MIN_ACCEPTABLE_MARGIN_MIN` on BOTH pairings the margin
    takes the worse of, at the SUMMER floor (which binds, because one UTC cron
    serves both seasons and summer's floor is the earlier one).
    """
    import datetime as _d
    import math

    import scripts.ci_audit as _ci
    from scripts.schedule_deadline import (MIN_ACCEPTABLE_MARGIN_MIN,
                                           observed_earliest_min)

    #: The SLOW cluster per day type, `measured` 2026-10-22 under the retired
    #: `0 0` cron. Recorded here because moving the cron empties the live
    #: series: membership keys on the run's own headSha carrying the cron in
    #: force, so this derivation cannot be re-run from live data until the new
    #: crons have fired.
    SLOW = {"weekday": {"max": 304.9, "p3sd": 332.6},
            "weekend": {"max": 293.7, "p3sd": 342.3}}
    summer = _d.date(2026, 10, 24)

    crons = current_crons_by_daytype()
    for dtype, expect in (("weekday", 60), ("weekend", 30)):
        floor = observed_earliest_min(dtype, summer)
        cap = min(
            floor - MIN_ACCEPTABLE_MARGIN_MIN - SLOW[dtype]["max"]
            - _ci.FULL_CARD_PICKS_MAX,
            floor - MIN_ACCEPTABLE_MARGIN_MIN - SLOW[dtype]["p3sd"]
            - _ci.FULL_CARD_PICKS_3SD)
        latest15 = int(math.floor(cap / 15) * 15)
        assert crons[dtype] == expect, (
            f"the {dtype} cron is {crons[dtype]}, expected {expect}")
        assert crons[dtype] == latest15, (
            f"the {dtype} cron {crons[dtype]} is no longer the latest "
            f"15-minute boundary that clears {MIN_ACCEPTABLE_MARGIN_MIN}m "
            f"(cap {cap:.1f} -> {latest15}). Either an input moved or the "
            f"cron was set by hand; re-derive before trusting it.")
        # and it must actually clear, not merely be the latest candidate
        for delay, picks in ((SLOW[dtype]["max"], _ci.FULL_CARD_PICKS_MAX),
                             (SLOW[dtype]["p3sd"], _ci.FULL_CARD_PICKS_3SD)):
            assert floor - (crons[dtype] + delay + picks) >= \
                MIN_ACCEPTABLE_MARGIN_MIN, (dtype, delay, picks)


def test_the_WEEKEND_cron_is_EARLIER_than_the_weekday_one():
    """Because its floor is earlier. The relation, not the values, is what
    carries the reason -- and it is the inverse of what the retired UTC
    constants implied."""
    crons = current_crons_by_daytype()
    assert crons["weekend"] < crons["weekday"], (
        f"weekend {crons['weekend']} is no longer earlier than weekday "
        f"{crons['weekday']}; the weekend floor is the earlier one (12:00 "
        f"local against 12:30), so its cron must be too")


@pytest.mark.live
def test_the_weekday_cron_CLEARS_the_stated_margin_and_LATEST_IS_RETIRED():
    # ── `live` ADDED 2026-10-21, AFTER THIS TEST KILLED A CARD ──────────────
    #
    # It calls `collect_lag_series_by_daytype()`, which queries the GitHub
    # API. That is the `live` marker's definition verbatim: "a test that
    # queries LIVE EXTERNAL STATE ... its outcome changes with no code
    # change". The test step runs `-m "not calendar and not live"` and gets no
    # GH credentials, so the series came back EMPTY and the n>=8 guard fired:
    #
    #     run 39532208006, 2026-10-21 02:17Z, headSha b8d91c1
    #     "the weekday series shrank to n=0; the +40m figure was measured at n=8"
    #
    # It passes locally (n=8, `gh` is authed) and fails in CI. First CI
    # exposure was today; the assertion was added yesterday in 269aea9.
    #
    # THE GUARD ITSELF WAS RIGHT -- n=0 is not a measurement, and refusing to
    # compute a margin from an empty series is the correct behaviour. What was
    # wrong is WHERE it ran: `daily-picks.yml` carries a comment four lines
    # above its pytest command recording that calendar assertions were moved
    # out of this gate on 2026-10-18 "at a cost of SIX CARDS", and a live test
    # went straight back into it four days later. Mine.
    """REWRITTEN 2026-10-18. Was: "`0 4` is the LATEST that clears".

    THAT RULE IS RETIRED, AND ITS FAILURE IS THE REASON. `0 4` was chosen by
    maximising lateness against the margin computed from `DELAY_MEAN/SD/MAX` —
    constants measured under the `0 0` regime, n=9, sd 18.2 minutes. The `0 4`
    regime's own distribution then turned out to be **4.5x wider**:

        weekday `0 0`   n=6  mean 276.3  sd 18.8  max 305  (measured 2026-10-18)
        weekday `0 4`   n=5  mean 357.5  sd 85.6  max 458  (measured 2026-10-18)

    so the shipped cron's real binding margin was **-241m**, not the +75m the
    selection had computed.

        A cron chosen as "the latest that clears" using ANOTHER cron's spread
        is an extrapolation presented as a bound.

    What replaces it is not a different optimum. It is the requirement that the
    shipped cron clear the margin against the series **it actually produces**,
    which can only be checked after it runs — so the selection stops being an
    optimisation and becomes a measurement with a revert path.
    """
    # ── MOVED AGAIN 2026-10-22 (`s5.19`), AND THIS TEST'S OWN MESSAGE SAID
    #    WHAT TO DO: "the series for the NEW cron must be measured before this
    #    number is trusted". It cannot be yet.
    #
    # Membership keys on the run's own headSha carrying the cron in force, so
    # the move EMPTIED the series. That is the reset recorded as a cost of the
    # move in `test_cron_moved.py`, and the requirement this test exists to
    # enforce -- that the shipped cron clears the margin against the series IT
    # ACTUALLY PRODUCES -- is therefore UNEVALUATED until the new cron runs.
    #
    # So the assertion becomes: the cron is where the derivation put it, and
    # the series is EMPTY, which is a state and not a pass. Once runs
    # accumulate this branch stops being taken and the margin below is checked
    # against real data -- the same shape as the `0 4` revert's requirement,
    # applied to its successor.
    crons = current_crons_by_daytype()
    assert crons["weekday"] == 60, (
        f"the weekday cron is at {crons['weekday']} minutes, expected 60 "
        f"(`0 1`, derived 2026-10-22 from the SLOW cluster against the "
        f"localised floor). If it has moved again, the series for the NEW "
        f"cron must be measured before any margin is trusted.")

    deadline = DEADLINE_MIN_BY_DAYTYPE["weekday"]

    def margins(cron_min):
        ready_max = cron_min + DELAY_MAX + DUR_MAX
        ready_3sd = cron_min + DELAY_MEAN + 3 * DELAY_SD + DUR_MEAN + 3 * DUR_SD
        return deadline - ready_max, deadline - ready_3sd

    m_max, m_3sd = margins(crons["weekday"])
    # THE PRE-SPLIT CONSTANTS still clear, and that is now a WARNING rather
    # than a reassurance: they are the figures the `0 4` selection used, and
    # the live series does NOT clear (asserted below). A check that passes on
    # stale constants while the measurement fails is how `0 4` shipped.
    assert min(m_max, m_3sd) >= MIN_ACCEPTABLE_MARGIN, (
        f"even the pre-split constants no longer clear: {m_max:+.0f}/"
        f"{m_3sd:+.0f} against {MIN_ACCEPTABLE_MARGIN}")

    # AND THE OLD ASSERTION IS DELIBERATELY NOT INVERTED. "One hour later
    # would also clear, so ship the later one" is precisely the reasoning that
    # produced `0 4`, and it used these same pre-split constants. The margin
    # bought by an earlier cron is not spent just because the arithmetic says
    # it could be: the 2026-10-18 series shows the spread is a function of the
    # regime, so headroom under one cron is not headroom under another.
    #
    # What IS asserted is that the measured weekday series - whatever cron it
    # carries - clears the bar, which is the only form of the claim that
    # survived contact with the data.
    import statistics

    # ── THE MEASURED SERIES NO LONGER CLEARS 60, AND THAT IS THE FINDING ───
    #
    # RE-POINTED 2026-10-20. This asserted `>= MIN_ACCEPTABLE_MARGIN` against
    # the live series and passed while the deadline was SELF-ANCHORED at
    # `ceil_15(ready + 60)` -- which made the claim a tautology. With the
    # deadline anchored to the OBSERVED EARLIEST KICKOFF (weekday 09:30) the
    # same arithmetic reports the truth:
    #
    #     measured 2026-10-20, weekday n=8:  binding +40m against a stated 60
    #
    # So the assertion is INVERTED and the breach is recorded with its number
    # and its clearing action, rather than a passing check standing where a
    # failing measurement belongs. If it ever clears again, this test fails and
    # says to re-state the finding -- which is the right direction for a
    # breach to surprise someone in.
    from scripts.ci_audit import collect_lag_series_by_daytype

    by_type, _crons2 = collect_lag_series_by_daytype()
    lag = [l for _d, l in (by_type.get("weekday") or [])]

    # ── "COULD NOT LOOK" IS NOT "THE SERIES SHRANK". 2026-10-21, second fix.
    #
    # Marking this `live` took it out of the capture gate, and capture then
    # succeeded -- but the test still FAILED in the non-gating step, because
    # that step has no GH credentials either, so the series came back empty
    # there too (run 39569612160). An empty series with no way to ask is
    # UNEVALUATED; my assertion called it a shrinkage and reported a finding
    # about football from the absence of a credential. That is the same
    # third-state defect this project keeps paying for, in a test I wrote to
    # guard against defects.
    #
    # A daily red that no action clears is also an alarm that always fires.
    # So: refuse to measure, and SAY which it is.
    if not lag:
        import os
        pytest.skip(
            "UNEVALUATED, not clear: the weekday lag series is empty and "
            "there is no way to ask here (GH_TOKEN present: "
            f"{bool(os.environ.get('GH_TOKEN') or os.environ.get('GITHUB_TOKEN'))}). "
            "n=0 is not a measurement, and the absence of a credential is not "
            "a fact about the schedule.")
    # ── A SERIES TOO THIN TO MEASURE IS UNEVALUATED, NOT FAILED. ─────────
    #
    # The cron move of 2026-10-22 emptied this series by design, and it is
    # refilling: n=1 as of 2026-10-23. `len(lag) >= 8` turned "the series is
    # refilling exactly as the entry said it would" into a red — a cost that
    # was recorded and then asserted against.
    #
    # The same third state the margin itself now reports: below
    # `LAG_CLUSTER_MIN_N` the series cannot be separated into clusters and
    # mean+3sd is not a bound. It SAYS SO rather than passing or failing.
    from scripts.ci_audit import LAG_CLUSTER_MIN_N, cluster_state
    if len(lag) < LAG_CLUSTER_MIN_N:
        pytest.skip(
            f"UNEVALUATED, not clear: n={len(lag)} is below "
            f"{LAG_CLUSTER_MIN_N}, so the series cannot be separated and "
            f"mean+3sd is not a bound. The cron moved on 2026-10-22 and the "
            f"series is refilling; the conservative reading until then is the "
            f"retired cron's SLOW cluster (mean+3sd 332.6).")
    if len(lag) < 8:
        pytest.skip(
            f"UNEVALUATED: n={len(lag)} is separable ("
            f"{cluster_state(lag)}) but below the n=8 the +40m figure was "
            f"measured at, so the comparison below has no standing yet.")
    lm, lsd = statistics.mean(lag), statistics.pstdev(lag)
    from scripts.ci_audit import MARGIN_RUN_DURATIONS_BY_DAYTYPE as _DUR
    du = _DUR["weekday"]
    dm, dsd = statistics.mean(du), statistics.pstdev(du)
    ready = max(max(lag) + max(du), (lm + 3 * lsd) + (dm + 3 * dsd))
    binding = deadline - ready
    assert binding < MIN_ACCEPTABLE_MARGIN, (
        f"the weekday margin is {binding:+.0f}m and now CLEARS the "
        f"{MIN_ACCEPTABLE_MARGIN}-minute minimum. That is better than the "
        f"state recorded on 2026-10-20 (+40m) -- re-measure and re-state the "
        f"finding rather than leaving this inverted assertion in place.")
    assert -20 <= binding <= 55, (
        f"the weekday margin moved to {binding:+.0f}m, outside the range this "
        f"finding was recorded over. Re-measure before quoting 19.8 minutes "
        f"as the shortfall.")
    # THE CLEARING ACTION IS NOT A DEADLINE CHOICE
    from scripts.ci_audit import MARGIN_REMEDY

    assert "earlier cron" in MARGIN_REMEDY and "faster run" in MARGIN_REMEDY, (
        "the remedy no longer names the only two levers")
    assert "14:30" not in MARGIN_REMEDY, (
        "the remedy cites the retired 14:30 percentile again")


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


def test_the_weekday_cron_CLEARS_a_holiday_weekday_since_the_revert():
    """RESOLVED 2026-10-18 BY THE CRON REVERT. Was: BREACHES.

    Under `0 4`, ready(3sd) was 10:42 UTC and Easter Monday 2026-04-06 — 95
    fixtures, earliest 10:15 — would have had picks delivered **27 minutes
    AFTER the first kickoff**. Under `0 0` the same arithmetic clears it by
    **+211 min**, and the 09:59 observed minimum clears too.

    The exposure below is kept as HISTORY because it is why the revert
    happened, and it changed no schedule at the time: the next weekday holiday with an early card is **Easter 2027
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

    # ── THE EXPOSURE IS RESOLVED, 2026-10-18, and this test did exactly what
    # it asked to do: its own message said "if the cron moved earlier, update
    # this test and the ledger rather than leaving a stale exposure on file".
    # The cron moved to `0 0` and the holiday breach became a CLEARANCE:
    #
    #     Easter Monday 10:15 vs ready(3sd)   -27 min  ->  +211 min
    #
    # So the assertion INVERTS rather than being deleted: the holiday that
    # breached must now clear, and if it ever breaches again the cron has moved
    # back or the distribution has widened, either of which is a finding.
    margin_full_card = WEEKDAY_EARLIEST_ON_A_FULL_CARD - ready_3sd
    assert margin_full_card > 0, (
        f"Easter Monday's 10:15 kickoff BREACHES ready(3sd) again by "
        f"{margin_full_card:+.0f} min. Under `0 0` it cleared by +211. Either "
        f"the weekday cron moved later or the delay distribution widened — "
        f"re-measure the series before changing this number.")
    assert margin_full_card >= 60, (
        f"only {margin_full_card:+.0f} min of holiday clearance; below an hour "
        f"the 1.4% of weekday holidays stop being comfortably covered")
    # 2026-06-18's 09:59 is the harder case and carried ONE fixture; recorded
    # here so the thin end of the distribution is not forgotten.
    assert WEEKDAY_EARLIEST_OBSERVED - ready_3sd > 0, (
        f"the observed weekday minimum 09:59 breaches ready(3sd) by "
        f"{WEEKDAY_EARLIEST_OBSERVED - ready_3sd:+.0f} min")

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
