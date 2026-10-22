"""The deadline is data — the prospective check, verified on dates that failed.

> A deadline derived from a sample of a varying quantity is a PERCENTILE, not a
> bound. The name "the earliest kickoff" hides which.

`MEASURED` 2026-10-10 from the production `matches` table, clean rows only
(phantom exclusion applied), 2026-02-01 onward, `world/*` excluded — earliest
kickoff per day:

    day type   days    p0      p05     p25     p50     p88     p100
    weekday     148   09:59   13:00   15:00   16:00   18:00   19:30
    weekend      66   10:00   10:15   10:30   11:00   12:00   14:45

**14:30 is the ~12th percentile of weekdays, not the 88th** — I had the tail
inverted when this was first reported. 87.8% of weekdays start LATER than
14:30; the error is that 18 of 148 start EARLIER. And **10:15 is the ~5th
percentile of weekends**, which also occur at 10:00.

THE TWO FAIL DIFFERENTLY, which is the whole reason one is urgent:

    weekend `0 0`  ready(3sd) 06:42  vs observed min 10:00  -> +198  harmless
    weekday `0 4`  ready(3sd) 10:42  vs observed min 10:15  ->  -27  BREACH

A CHECK THAT ALARMS ON EVERYTHING IS THE SAME AS ONE THAT ALARMS ON NOTHING, so
this file carries four positives and a negative control.
"""

import datetime as dt

import pytest

from scripts.schedule_deadline import (
    CLEARING_ACTION,
    DEADLINE_PERCENTILES,
    MIN_ACCEPTABLE_MARGIN_MIN,
    day_type,
    format_prospective,
    projected_ready,
    prospective_margin,
)

#: `MEASURED` 2026-10-10, regime `0 0 * * *`, scheduled events only, n=9.
#: The live series the projection uses; both spreads, never the mean alone.
DELAYS = [4 * 60 + 31, 4 * 60 + 45, 4 * 60 + 42, 4 * 60 + 31, 5 * 60 + 4,
          4 * 60 + 16, 4 * 60 + 31, 3 * 60 + 58, 4 * 60 + 15]
DURATIONS = [72.3, 53.3, 29.0, 41.8, 47.5, 37.0, 40.8, 45.4, 43.2, 42.3]

WEEKDAY_CRON = 4 * 60      # `0 4 * * 1-5`, shipped 2026-10-10
WEEKEND_CRON = 0           # `0 0 * * 6,0`, unchanged

#: The four dates that already failed, with their MEASURED earliest kickoffs.
#: 2027 carries the same holidays; its kickoff time is `assumed` from the 2026
#: observation of the same holiday and is labelled as such.
GOOD_FRIDAY_2026 = (dt.date(2026, 4, 3), 11 * 60 + 30, "measured, 61 fixtures")
EASTER_MONDAY_2026 = (dt.date(2026, 4, 6), 10 * 60 + 15, "measured, 95 fixtures")
GOOD_FRIDAY_2027 = (dt.date(2027, 4, 2), 11 * 60 + 30, "assumed from 2026-04-03")
EASTER_MONDAY_2027 = (dt.date(2027, 4, 5), 10 * 60 + 15, "assumed from 2026-04-06")

#: The NEGATIVE CONTROL: an ordinary weekday binding on European competition.
ORDINARY_WEEKDAY = (dt.date(2026, 10, 13), 16 * 60 + 45, "measured, champions-league")


def _margin(day, kickoff_min):
    cron = WEEKEND_CRON if day_type(day) == "weekend" else WEEKDAY_CRON
    return prospective_margin(kickoff_min, cron, DELAYS, DURATIONS, serves=day)


# ═══════════════════════ B1 — the four positives ════════════════════════════

@pytest.mark.parametrize("day,kickoff,provenance", [
    GOOD_FRIDAY_2026, EASTER_MONDAY_2026,
    GOOD_FRIDAY_2027, EASTER_MONDAY_2027,
])
def test_the_check_ALARMS_on_every_date_that_already_failed(day, kickoff, provenance):
    """Replayed against the known breaches, not against today.

    Easter Monday 2026-04-06 carried 95 fixtures with the earliest at 10:15 and
    `0 4` yields -27 min: the prospective check must have alarmed. Good Friday
    yields +48, positive but BELOW the stated 60-minute minimum, so it must
    alarm too — a margin that is merely positive is not a margin that was
    accepted.
    """
    m = _margin(day, kickoff)
    assert m["verdict"] == "ALARM", (
        f"{day} ({provenance}) earliest {kickoff // 60:02d}:{kickoff % 60:02d} "
        f"did not alarm: {m}")
    assert m["binding"] < MIN_ACCEPTABLE_MARGIN_MIN
    assert m["deadline_min"] == kickoff, (
        "the margin did not carry the deadline it was measured against")
    assert "ACTION REQUIRED" in format_prospective(m)


def test_EASTER_MONDAY_is_NEGATIVE_not_merely_short():
    """The worst measured case, pinned with its number.

    95 fixtures, earliest 10:15, ready(3sd) 10:42. Picks land AFTER the first
    kickoff, which is the failure the deadline exists to prevent.
    """
    day, kickoff, _ = EASTER_MONDAY_2026
    m = _margin(day, kickoff)
    assert m["margin_3sd"] < 0, m
    assert -40 < m["margin_3sd"] < -10, (
        f"the Easter Monday breach is {m['margin_3sd']:+.0f} min, not the "
        f"measured ~-27 — re-derive before trusting this file")


def test_GOOD_FRIDAY_is_POSITIVE_but_below_the_stated_minimum():
    """The case that proves the minimum is doing work.

    +48 min would pass a bare "is it positive" check and fails the stated bar.
    """
    day, kickoff, _ = GOOD_FRIDAY_2026
    m = _margin(day, kickoff)
    assert m["margin_3sd"] > 0, m
    assert m["margin_3sd"] < MIN_ACCEPTABLE_MARGIN_MIN, m
    assert m["verdict"] == "ALARM"


# ═══════════════════════ B1 — the negative control ══════════════════════════

def test_the_check_is_SILENT_on_an_ordinary_weekday():
    """THE CONTROL. A check that alarms on everything carries no information.

    2026-10-13 binds on champions-league at 16:45 UTC, which is 87.8% of
    weekdays' shape.
    """
    day, kickoff, _ = ORDINARY_WEEKDAY
    m = _margin(day, kickoff)
    assert m["verdict"] == "CLEAR", m
    assert m["binding"] >= MIN_ACCEPTABLE_MARGIN_MIN
    assert "ACTION REQUIRED" not in format_prospective(m)


def test_an_ordinary_WEEKEND_is_silent_too():
    """The weekend cron has three hours of slack, so even its observed minimum
    clears. That is why the weekend percentile costs nothing."""
    # ── THE ANCHOR IS ASKED FOR THE DAY THIS SERVES. 2026-10-22. ────────
    #
    # It read `OBSERVED_EARLIEST_MIN["weekend"]`, which resolves for TOMORROW,
    # while `serves=` named 2026-10-11 -- two different days, and after the
    # localisation two different UTC minutes whenever they straddle a
    # transition. The anchor is now asked for the day under test.
    serves = dt.date(2026, 10, 11)
    from scripts.schedule_deadline import observed_earliest_min
    m = prospective_margin(observed_earliest_min("weekend", serves),
                           WEEKEND_CRON, DELAYS, DURATIONS, serves=serves)
    assert m["verdict"] == "CLEAR", m
    # 182 -> 137 ON 2026-10-22, and it is a CORRECTION, not a regression. The
    # weekend anchor localised from 09:45 UTC (a summer-subset minimum) to
    # 12:00 local, which in summer is 09:00 UTC -- 45 minutes EARLIER. The
    # margin is asserted against the REGISTERED minimum rather than against
    # the old reading, because the old reading was the thing that was wrong.
    assert m["binding"] > MIN_ACCEPTABLE_MARGIN_MIN, m
    assert m["binding"] == pytest.approx(137.3, abs=1.0), (
        f"the weekend binding moved from the measured 137.3; re-derive "
        f"(got {m['binding']:.1f})")


# ═════════════════ UNKNOWN is a third state, not a safe one ═════════════════

def test_an_UNDISCOVERED_card_is_UNKNOWN_and_never_substituted():
    """`None` must not become a constant.

    An undiscovered card is an unknown deadline. Falling back to 14:30 would
    turn "we do not know" into "we have until 14:30", which is the failure this
    whole stage exists to remove.
    """
    m = prospective_margin(None, WEEKDAY_CRON, DELAYS, DURATIONS,
                           serves=dt.date(2026, 10, 12))
    assert m["verdict"] == "UNKNOWN"
    assert m["deadline_min"] is None
    assert "not discovered" in m["reason"]
    assert "NOT safe" in format_prospective(m)
    assert m.get("binding") is None, (
        "an UNKNOWN verdict produced a binding margin — it has nothing to bind "
        "against")


def test_a_missing_SERIES_is_also_UNKNOWN_not_clear():
    """No runs under the cron in force means the projection has no basis.

    This is the live state of the weekday series the day the split landed.
    """
    m = prospective_margin(16 * 60, WEEKDAY_CRON, [], [],
                           serves=dt.date(2026, 10, 13))
    assert m["verdict"] == "UNKNOWN"
    assert "series" in m["reason"]


# ═══════════════════ CLR-2 — the action is named in the alert ═══════════════

def test_the_clearing_action_is_NAMED_and_says_WHO():
    """An alarm nobody can act on is `fixtures_zero_active` again.

    There is exactly one action available and it is manual, which is acceptable
    only because it is named — including that no automated alternative exists.
    """
    assert "DISPATCH" in CLEARING_ACTION
    assert "daily-picks" in CLEARING_ACTION
    assert "operator" in CLEARING_ACTION, (
        "the action does not say WHO performs it")
    assert "no automated alternative" in CLEARING_ACTION, (
        "the absence of an automated remedy is not stated, so a reader will "
        "assume one exists")


def test_the_named_action_ACTUALLY_CHANGES_the_predicate():
    """CLR-2 proper: confirm the remedy clears the alarm.

    A dispatched run is not subject to the scheduler's measured 3h58m-5h04m
    lateness, so its ready-time is the duration alone. Driven rather than
    asserted: the same Easter Monday that breaches on the cron must CLEAR when
    dispatched at 06:00.
    """
    day, kickoff, _ = EASTER_MONDAY_2026
    on_cron = _margin(day, kickoff)
    assert on_cron["verdict"] == "ALARM"

    dispatched_at = 6 * 60
    cleared = prospective_margin(kickoff, dispatched_at, [0.0] * len(DURATIONS),
                                 DURATIONS, serves=day)
    assert cleared["verdict"] == "CLEAR", (
        f"the only named clearing action does NOT clear the alarm: {cleared} — "
        f"then the alarm is unactionable and must not be shipped")
    assert cleared["binding"] >= MIN_ACCEPTABLE_MARGIN_MIN


# ═══════════ PART C — the constants are retired AS BOUNDS ══════════════════

def test_the_constants_are_labelled_PERCENTILES_with_which_one():
    """Each retired constant carries its percentile and its provenance."""
    for dtp, (minutes, pct, prov) in DEADLINE_PERCENTILES.items():
        assert pct.startswith("~"), (dtp, pct)
        assert "measured" in prov and "2026-10-10" in prov, (dtp, prov)
        # BOTH SIDES. The anchor is LOCAL since 2026-10-22, so the retired
        # percentile must be later than its conversion in EITHER season --
        # the binding comparison is against the WINTER one, which is later.
        from scripts.schedule_deadline import observed_earliest_min
        for _day in (dt.date(2026, 10, 24), dt.date(2026, 10, 26)):
            _obs = observed_earliest_min(dtp, _day)
            assert minutes > _obs, (
                f"on {_day} the {dtp} constant {minutes} is not later than "
                f"the observed minimum {_obs} - then it is a bound after all "
                f"and this label is wrong")
    # The weekday fallback must be the figure the cron was SIZED against, never
    # the laxer 14:30 - a fallback that loosens what justified the schedule is
    # worse than no fallback.
    assert DEADLINE_PERCENTILES["weekday"][0] == 12 * 60, (
        "the weekday fallback moved off 12:00, the value `0 4` was sized "
        "against; 14:30 is our ~12th percentile and would loosen it silently")
    assert "09:59" in DEADLINE_PERCENTILES["weekday"][2], (
        "the weekday provenance no longer records the observed minimum, so a "
        "reader cannot see that even 12:00 is not a bound")


def test_no_margin_computation_uses_a_deadline_LITERAL():
    """The guard that keeps the constants retired.

    14:30 / 12:00 / 10:15 as minute literals (870 / 720 / 615) must not appear
    in a margin computation. `DEADLINE_PERCENTILES` holds them as labelled
    data; `schedule_deadline.py` is where that label lives, so it is the one
    file allowed to name them.
    """
    import pathlib
    import re

    offenders = []
    for path in (pathlib.Path("scripts/ci_audit.py"),):
        txt = path.read_text(encoding="utf-8")
        # strip comments: a literal in prose is provenance, not a computation
        code = "\n".join(l.split("#", 1)[0] for l in txt.splitlines())
        # WORD-BOUNDED for the bare numbers. `870` as a substring matched the
        # RUN ID `38708288664` on 2026-10-18 and reported a deadline literal
        # in a `RESOLVED_RUNS` comment. A digit run inside a longer digit run
        # is not a literal.
        import re as _re
        for lit in ("14 * 60 + 30", "12 * 60", "10 * 60 + 15"):
            for m in re.finditer(re.escape(lit), code):
                line = code[:m.start()].count("\n") + 1
                offenders.append(f"{path}:{line} uses {lit!r}")
        for bare in ("870", "720", "615"):
            if _re.search(r"(?<![0-9])" + bare + r"(?![0-9])", code):
                offenders.append(f"{path}:{code.index(bare)} uses {bare!r}")
    assert not offenders, (
        "a deadline literal is back in a margin computation:\n  "
        + "\n  ".join(offenders)
        + "\nUse `discovered_deadline()`; if a fallback is unavoidable, take it "
          "from DEADLINE_PERCENTILES so it carries its label.")


def test_the_WEEKEND_distribution_is_UNIMODAL_which_is_why_it_may_stay():
    """The directive's condition, measured rather than assumed.

    weekend p0 10:00 -> p50 11:00 -> p88 12:00 -> p100 14:45: one tight mode
    with a right tail. The weekday distribution has a GAP — p0 09:59 then p05
    13:00 — because its early mass is public holidays behaving like weekends
    while ordinary weekdays sit at 15:00-19:30.

    So the weekend constant survives as a labelled percentile, and it survives
    SAFELY for a second reason: `0 0` is ready by 06:42, so even 10:00 clears
    by 198 minutes.
    """
    weekend_p0, weekend_p50, weekend_p88 = 10 * 60, 11 * 60, 12 * 60
    weekday_p0, weekday_p05 = 9 * 60 + 59, 13 * 60
    assert weekend_p88 - weekend_p0 <= 120, (
        "the weekend spread p0..p88 exceeded two hours — re-test unimodality "
        "before leaving its constant in place")
    assert weekday_p05 - weekday_p0 > 120, (
        "the weekday p0..p05 gap closed, so the bimodality claim no longer "
        "holds and the weekday treatment should be re-derived")
    m = prospective_margin(weekend_p0, WEEKEND_CRON, DELAYS, DURATIONS,
                           serves=dt.date(2026, 10, 11))
    assert m["binding"] > 150, (
        "the weekend minimum no longer clears comfortably, so its percentile "
        "has started to cost something")


def test_the_ALERT_TEXT_carries_the_clearing_action_not_just_the_ledger():
    """Rule 4. `fixtures_zero_active` was noise because the remedy lived in the
    ledger while the alert said only that something was wrong.

    THE PROPERTY MOVED on 2026-10-11 and this test moved with it. The
    prospective alarm was briefly built inside `ci_audit`, where it could never
    evaluate - that step passes no `DATABASE_URL` and installs no driver, so it
    read `UNKNOWN` on every run. It now lives in
    `scripts/fixture_horizon_probe.py`, which has both.
    """
    import inspect
    import pathlib as _p

    from scripts.fixture_horizon_probe import main as probe_main

    src = inspect.getsource(probe_main)
    assert "CLEARING_ACTION" in src, (
        "the prospective alarm no longer interpolates the clearing action, so "
        "the alert says a margin is short without saying what to do")
    assert "binding" in src, (
        "the alarm text omits the margin itself")
    # The literal moved OUT of this function on 2026-10-12, deliberately:
    # `ci_audit.SELF_EMITTED_ANNOTATIONS` pins each registered pattern to the
    # file that prints it, and its guard requires the LITERAL rather than an
    # f-string assembly. So the property is checked in two halves - the
    # function uses the constant, and the constant is an `::error::` literal.
    import scripts.fixture_horizon_probe as _pr

    assert "ANNOTATION_ALARM" in src, (
        "the alarm is not surfaced as an annotation")
    assert _pr.ANNOTATION_ALARM.startswith("::error::"), _pr.ANNOTATION_ALARM
    assert _pr.ANNOTATION_REFUSED.startswith("::error::")
    assert _pr.ANNOTATION_ALARM != _pr.ANNOTATION_REFUSED, (
        "a designed refusal and a finding share one annotation, so the log "
        "cannot tell them apart")

    # ANCHOR MOVED 2026-10-12, PROPERTY STRENGTHENED. The date and the
    # deadline used to be asserted by scanning this function's SOURCE for the
    # tokens `serves` and `deadline_min`. They now live in
    # `format_dispatch_alert`, so the check is made against the RENDERED BODY
    # instead - which is strictly stronger, and is the lesson of this stage:
    # an alert body never rendered in a test is a formatter with no consumer.
    import datetime as _dt

    import scripts.fixture_horizon_probe as _probe

    body = _probe.format_dispatch_alert(
        _dt.date(2026, 10, 13), 11 * 60 + 30, "x/y",
        basis="deadline DISCOVERED by the probe", margin=-37.0)
    assert "2026-10-13" in body, "the body omits the date it serves"
    assert "11:30 UTC" in body, "the body omits the deadline"
    assert "LAST SAFE DISPATCH" in body, (
        "the body omits the last safe dispatch time, so the operator still "
        "has to subtract the runtime himself")
    assert "gh workflow run daily-picks.yml" in body
    assert "no automated alternative" in body
    assert "operator" in body
    assert "DISCOVERED by the probe" in body, "the body omits its basis"


def test_the_prospective_alarm_only_fires_for_TOMORROW():
    """Today's margin is retrospective: the run has already started, so an
    alarm on it cannot be acted on and would fire alongside every real one.

    The probe evaluates exactly one day - the one the NEXT run serves.
    """
    import inspect

    from scripts.fixture_horizon_probe import main as probe_main

    src = inspect.getsource(probe_main)
    assert "timedelta(days=1)" in src, (
        "the probe no longer targets tomorrow specifically")
    assert "tomorrow" in src
