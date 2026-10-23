"""The schedule deadline, derived from the card rather than from a constant.

    from scripts.schedule_deadline import discovered_deadline, prospective_margin

WHY THIS EXISTS. `daily-picks` is sized against "the earliest kickoff", and that
was a per-day-type constant. **The quantity varies by date**, so a constant is a
PERCENTILE wearing the name of a bound.

MEASURED 2026-10-10 from the production `matches` table, clean rows only,
2026-02-01 onward, `world/*` excluded — the earliest kickoff per day:

    day type   days    p0      p05     p25     p50     p88     p100
    weekday     148   09:59   13:00   15:00   16:00   18:00   19:30
    weekend      66   10:00   10:15   10:30   11:00   12:00   14:45

So the constants were:

    14:30 weekday  -> the ~12th PERCENTILE. 87.8% of weekdays start LATER, and
                      18 of 148 start EARLIER. Conservative for most days and
                      simply wrong for the tail.
    10:15 weekend  -> the ~5th PERCENTILE. Weekends at 10:00 occur.

**AND THE TWO FAIL DIFFERENTLY.** The weekend cron `0 0` is ready by 06:42
(mean+3sd), so even the observed 10:00 minimum clears by 198 minutes — the
weekend constant is a percentile that costs nothing. The weekday cron `0 4` is
ready at 10:42, so the observed 10:15 holiday minimum is a **-27 minute**
margin. Same kind of error, opposite consequence.

THE WEEKDAY DISTRIBUTION IS BIMODAL, which is why no single percentile works:
p0 09:59 then p05 13:00 is a gap, because the early mass is PUBLIC HOLIDAYS
(Good Friday, Easter Monday, 1 May, Whit Monday) behaving like weekends, while
ordinary weekdays sit at 15:00-19:30. The weekend distribution is unimodal
(10:00 -> 11:00 -> 12:00, tail to 14:45).

THE PHANTOM EXCLUSION IS NOT OPTIONAL. Sub-second `match_date` is an ingestion
timestamp, not a kickoff. Uncorrected, the weekday deadline reads **07:53** and
111 of 112 sub-14:30 rows in one league are phantoms. Every query here that
computes a kickoff extremum applies it; Rule 3 has no fixed point.
"""

from __future__ import annotations

import datetime as _dt
from typing import Optional, Sequence

#: The stated minimum acceptable margin, in minutes, on mean+3sd. Stated before
#: any computation on 2026-10-10 and kept here so it cannot be retrofitted.
MIN_ACCEPTABLE_MARGIN_MIN = 60

#: Legacy constants, RETIRED AS BOUNDS and kept only as labelled percentiles so
#: a reader who meets one knows what it is. Nothing may compute a margin from
#: these — see `test_no_margin_computation_uses_a_deadline_LITERAL`.
#:
#: THE WEEKDAY ENTRY IS THE STRICTER OF TWO SOURCES ON PURPOSE. Three weekday
#: figures exist and none is a bound:
#:
#:     09:59  our observed MINIMUM over 148 days  -> a bound for that window,
#:            but `0 4` fails it by -43, so using it as a daily constant would
#:            make the alarm permanent, which is `fixtures_zero_active` again
#:     12:00  the independent reference's minimum over ONE WEEK (n=5 days) —
#:            what `0 4` was actually sized against, clearing it by +76
#:     14:30  our own ~12th percentile over 148 days
#:
#: The table carries **12:00**, because a fallback must never be laxer than the
#: figure the schedule was justified with. 14:30 would have loosened it
#: silently.
DEADLINE_PERCENTILES = {
    # day type: (minutes, what it is, provenance)
    "weekday": (12 * 60, "~5th-percentile proxy",
                "independent reference minimum over one week (n=5 days), "
                "measured 2026-10-10; our own ~12th percentile is 14:30 and "
                "our observed minimum is 09:59"),
    "weekend": (10 * 60 + 15, "~5th percentile",
                "measured 2026-10-10, n=66 days; observed minimum 10:00"),
}

#: Observed MINIMA, which are bounds for the measured window and nothing more.
#:
#: RE-MEASURED 2026-10-19 over the FULL record -- n=1090 days, 2022-07-05 ..
#: 2026-10-19, the 30 configured leagues, phantom exclusion applied. Both moved
#: EARLIER than the 2026-10-10 figures, which were taken over 2026-02-01
#: onward: a minimum is a function of its window, and a longer window can only
#: lower it. The earlier values are kept beside each one so the movement is
#: visible rather than silent.
#: ── THE ANCHOR IS A LOCAL TIME. LOCALISED 2026-10-22. ────────────────────
#:
#: A kickoff is scheduled by a human in a local clock, so its UTC minute moves
#: twice a year and a UTC anchor measured in one season is wrong in the other.
#: `measured` 2026-10-22 as the minimum **LOCAL** kickoff over the whole
#: record, Europe/Sofia, the 30 configured leagues, phantoms excluded:
#:
#:   weekday  12:30 local  n=718 days   set 2023-10-26 Thu, bulgaria/efbet-league
#:   weekend  12:00 local  n=375 days   set 2024-11-02 Sat, bulgaria/efbet-league
#:
#: THE WEEKDAY ANCHOR IS THE SAME EVENT AS BEFORE; THE WEEKEND ANCHOR IS NOT.
#: The retired weekend constant (09:45 UTC, 2024-10-19) was the minimum of the
#: SUMMER subset only. The true local minimum is 12:00, and it comes from a
#: WINTER day -- four of the five earliest local kickoffs are winter dates. So
#: localising the weekend floor makes it **45 minutes TIGHTER in summer**, not
#: looser. A correction is not a gift, and this one runs both ways.
#:
#: WHAT EACH ANCHOR PRODUCES, both sides of 2026-10-25:
#:
#:   day type   local    2026-10-24 (UTC+3)   2026-10-26 (UTC+2)   vs retired
#:   weekday    12:30    09:30 UTC            10:30 UTC            0 / +60
#:   weekend    12:00    09:00 UTC            10:00 UTC          -45 / +15
OBSERVED_EARLIEST_LOCAL_MIN = {
    "weekday": 12 * 60 + 30,    # 2023-10-26 Thu, bulgaria/efbet-league
    "weekend": 12 * 60 + 0,     # 2024-11-02 Sat, bulgaria/efbet-league
}

#: The clock those anchors are expressed in. Named, never assumed, and the
#: offset is read from the zone database per day -- see `operator_offset_min`.
OPERATOR_TZ = "Europe/Sofia"


def operator_offset_min(day) -> int:
    """The operator's UTC offset in MINUTES on `day`, DERIVED per day.

    Measured at 12:00 local so the transition hour itself can never straddle,
    and it RAISES rather than guessing when the zone database is unavailable:
    a deadline computed from an assumed offset is a deadline that is an hour
    wrong for half the year, and silently.
    """
    from zoneinfo import ZoneInfo
    return int(_dt.datetime.combine(
        day, _dt.time(12), tzinfo=ZoneInfo(OPERATOR_TZ))
        .utcoffset().total_seconds() // 60)


def observed_earliest_min(day_type: str, day) -> int:
    """The anchor as a UTC minute ON `day`. The only correct way to ask."""
    return OBSERVED_EARLIEST_LOCAL_MIN[day_type] - operator_offset_min(day)




#: RETIRED 2026-10-22, KEPT FOR THE RECORD. The stored UTC pair, both of them
#: summer figures, which is the defect. Nothing computes from this.
RETIRED_OBSERVED_EARLIEST_UTC_MIN = {
    "weekday": 9 * 60 + 30,     # 2023-10-26, three days before that year's
                                # DST transition (2023-10-29)
    "weekend": 9 * 60 + 45,     # 2024-10-19, eight days before 2024-10-27
}

# ─────────────────────────────────────────────────────────────────────────────
# THE FALLBACK IS A DERIVED FLOOR, NOT A PERCENTILE — 2026-10-19
#
#     Raising the minimum margin accommodates a wrong constant. Fixing the
#     constant removes the exposure.
#
# The 12:00 weekday constant was the fallback on 1087 of 1090 days and it was
# **two and a half hours laxer than our own measured earliest kickoff**. That
# gap WAS the exposure tail: max 150m, above 60m on 8 of 715 weekday days. It
# never belonged to the reference, which can only ever lower the deadline.
#
# THE MEASURED EARLIEST KICKOFF, `measured` 2026-10-19 from production
# `matches`, the 30 CONFIGURED leagues, phantom exclusion applied (sub-second
# `match_date` is an ingestion stamp; 00:00 rows are placeholders), n=1090
# days over 2022-07-05 .. 2026-10-19:
#
#   stratum              n     MIN     p01     p05     p50   <10:00  <11:00
#   weekday            715   09:30   10:45   13:00   17:00        2        9
#     ordinary         693   09:30   11:00   14:00   17:00        2        7
#     holiday           22   10:15   10:15   10:30   13:00        0        2
#   weekend            375   09:45   10:00   10:15   11:30        1       96
#     ordinary         373   09:45   10:00   10:15   11:30        1       96
#     holiday            2   13:00   13:00   13:00   13:00        0        0
#
# THE HOLIDAY STRATUM IS NOT THE EARLIEST ONE, and the earlier analysis implied
# it was. Holidays shift the whole weekday distribution forward (p50 13:00
# against 17:00) but the MINIMUM is an ordinary Thursday: 2023-10-26 09:30,
# `bulgaria/efbet-league`. The weekday minima are 09:30, then 09:59
# (2026-06-18, `finland/veikkausliiga`), then 10:15 (Easter Monday 2026).
#
# THE READY TIME AT `0 0`, each day type from ITS OWN series -- which is the
# thing `0 4` got wrong by borrowing the other's spread:
#
#   day type   lateness (attributed by headSha)   duration (by day type)
#   weekday    n=7  max 305  mean+3sd 417.5       n=78  max 100.8  m+3sd 100.9
#   weekend    n=6  max 294  mean+3sd 334.5       n=34  max 129.2  m+3sd 148.8
#
#   weekday  ready max 06:45   mean+3sd 08:38   BINDING 08:38
#   weekend  ready max 07:02   mean+3sd 08:03   BINDING 08:03
#
# CAVEAT, STATED: the binding figure is a mean+3sd projection on a lateness
# series of n=7 / n=6, and it exceeds the observed max by 112 / 40 minutes. The
# sd is poorly determined at that n, so THE FLOOR IS RE-DERIVABLE and will move
# EARLIER as the series grows and the sd tightens. Re-derive it, do not defend
# it. Lateness is attributed per run by its own `headSha` cron, because pooling
# regimes reads a `37 9` start as 21 hours late; duration is NOT cron-dependent
# and is pooled across regimes by day type, which is both better powered and
# more conservative than the current regime alone (weekday n=7 m+3sd 59.4).
#
# THE DERIVATION, which is an arithmetic and not a choice:
#
#     floor = ceil_to_15min( ready_binding + MIN_ACCEPTABLE_MARGIN_MIN )
#
# the TIGHTEST value the measured pipeline can support while clearing the
# pre-registered 60 minutes. Tighter is safer against an unobserved early
# kickoff -- which is the only thing a fallback is for -- and the margin
# requirement is what stops it going lower.
#
#   weekday  ceil_15(518.4 + 60) = 09:45   margin +66.6m   exposure 15m worst
#   weekend  ceil_15(483.3 + 60) = 09:15   margin +71.7m   exposure  0m
#
# AND THE WEEKDAY CANNOT HAVE BOTH, BY 8.4 MINUTES. Zero exposure needs
# `floor <= 09:30` (the observed minimum); the 60-minute margin needs
# `floor >= 09:38`. The window is empty by **8.4 minutes**, so the weekday
# keeps a 15-minute residual after rounding -- one day in 715, far inside the
# 60-minute budget. The weekend windows overlap and its exposure is ZERO.
#
# THE ALTERNATIVE, NOT TAKEN: `floor = observed minimum` maximises the margin
# among zero-exposure choices (weekend 09:45, +101.7m). It was rejected
# because the observed minimum is a bound for the MEASURED WINDOW and nothing
# more -- a fixture 20 minutes earlier next season would reintroduce exposure
# silently, whereas the derived floor already sits 30 minutes below it.
#
# WHAT IT COSTS IN ALERTS: `measured` by replaying all 1090 days with each
# day's own `min(covered-league minimum, floor)` as the deadline and each day
# type's own binding ready time --
#
#   constants            exposure>0   max    >60m   ALARMS   per week
#   current 12:00/10:15   21 days    150m   8 days     0      0.000
#   derived 09:45/09:15    1 day      15m   0 days     0      0.000
#
# **ZERO additional alerts, and eight fewer days above the 60-minute budget.**
# For scale, the DISCOVERED-card basis -- the 0.69/wk figure measured under
# `0 4` -- alarms once in 1090 days under `0 0`, 0.0045/wk; the regime change
# did that, not this.
#
# NOT A COHORT EVENT: the deadline is read only by margin reporting and
# alerting. No pick, price or probability depends on it.

# `_ceil15` was DELETED 2026-10-20 with the self-anchored floor: it existed
# only to round `ready_binding + 60` up to a quarter hour, and the
# world-anchored deadline is a measured kickoff, which needs no rounding.


#: `READY_BINDING_MIN` was DELETED 2026-10-20 with the self-anchored floor.
#: It held 518.4 / 483.3 -- the ready times measured on 2026-10-19 -- and the
#: world-anchored deadline does not read them. Keeping a stale measured
#: constant that nothing consumes is the defect this project deletes: by the
#: next morning the weekday figure was already 529.8, and a frozen 518.4
#: sitting in the module would have been quoted by someone.
#: The live figures come from `collect_lag_series_by_daytype()` at use.


# ─────────────────────────────────────────────────────────────────────────────
# RE-ANCHORED 2026-10-20: THE DEADLINE IS A FACT ABOUT FOOTBALL, NOT ABOUT US
#
#     A deadline derived from our own readiness makes the margin a tautology.
#
# WHAT I SHIPPED YESTERDAY WAS WRONG, and it failed on its first real exposure.
# `floor = ceil_15(ready_binding + MIN_ACCEPTABLE_MARGIN_MIN)` has two defects:
#
#   1. IT IS A TAUTOLOGY. If the deadline is defined as `ready + 60`, then
#      `margin >= 60` is true BY CONSTRUCTION at the instant of derivation.
#      Re-deriving it each run would make the prospective alarm **unable to
#      fire** -- a vacuous filter, the shape this project names VAC-1.
#   2. IT WAS FROZEN AGAINST A MOVING INPUT. Deriving once from a live series
#      and then hard-coding the result is the stale-literal defect fixed three
#      days ago in the `LIVE 0 4 * * 1-5` cron label. Measured: the weekday
#      lateness series went n=7 -> n=8 overnight and `mean+3sd` moved
#      417.5 -> 428.9, because the new point LOWERED the mean (255.2 -> 241.3)
#      and RAISED the sd (54.1 -> 62.5), and 3sd dominates. `ready_binding`
#      drifted +11.4m and the margin fell +66.6 -> **+55.2m**, breaching the
#      registered 60 and reddening the audit three times.
#
#      I had flagged the instability and got the DIRECTION wrong: the note said
#      the floor "will move EARLIER as the series grows". It moved later.
#
# SO THE DEADLINE IS ANCHORED TO THE OBSERVED EARLIEST KICKOFF -- the world --
# and the margin becomes a real measurement again:
#
#   day type   deadline            ready (measured)   margin
#   weekday    09:30 (2023-10-26)  08:49              **+40.2m  FAILS 60**
#   weekend    09:45 (2024-10-19)  08:03              +101.7m   clears
#
# AND THAT BREACH IS THE TRUTH, NOT A BUG. Under the self-anchored floor the
# margin read +55.2m; world-anchored it reads **+40.2m, 19.8 minutes short of
# the registered 60.** The self-anchoring was hiding 15 minutes of it. The
# registered 60 is NOT edited -- it was stated on 2026-10-10 before any of
# this -- so the breach stands as a finding.
#
# THE CLEARING ACTION IS NOT A DEADLINE CHOICE. No anchoring clears it: the
# earliest kickoff is a fact and the ready time is ours. The levers are an
# EARLIER CRON or a FASTER RUN, and the duration profile already sized the
# second: ~7 min in the sequential injury loop, ~7 min in the sequential Claude
# review, up to ~9 min in the AF upsert loop. **Any two of those close 19.8
# minutes.** That is the decision, and it is Niki's.
#
# EXPOSURE BECOMES ZERO BY CONSTRUCTION. The deadline now IS the observed
# minimum, so `max(0, deadline - true_earliest)` is 0 for every day in the
# measured window -- the coverage-exposure term is satisfied rather than
# traded, and it is still printed.
#: Provenance, per day type. The MINUTE is no longer stored here -- it is a
#: function of the day, so storing it is the defect that was just fixed.
DEADLINE_FLOOR_PROVENANCE = {
    "weekday": (
        "observed earliest LOCAL kickoff",
        "12:30 local on 2023-10-26, bulgaria/efbet-league; measured "
        "2026-10-22 over n=718 weekday days, the 30 configured leagues, "
        "phantoms excluded. Converted to UTC per day from Europe/Sofia. A "
        "bound for the measured window and nothing more -- re-measure as the "
        "record grows, and a longer window can only lower it"),
    "weekend": (
        "observed earliest LOCAL kickoff",
        "12:00 local on 2024-11-02, bulgaria/efbet-league; measured "
        "2026-10-22 over n=375 weekend days, same population and exclusions. "
        "NOT the same event as the retired 09:45 UTC constant, which was the "
        "minimum of the SUMMER subset only -- four of the five earliest local "
        "kickoffs are winter dates, and localising this floor TIGHTENS it by "
        "45 minutes in summer"),
}


def deadline_floor_for(day_type: str, day):
    """`(utc_minutes, what, provenance)` for `day_type` ON `day`.

    THE DAY IS REQUIRED. The retired `DEADLINE_FLOOR` dict held a UTC minute
    with no day attached, so it answered with a summer figure all winter.
    """
    what, prov = DEADLINE_FLOOR_PROVENANCE[day_type]
    return (observed_earliest_min(day_type, day), what, prov)


class _DeadlineFloorForToday(dict):
    """`DEADLINE_FLOOR[dtype]` kept working, resolved for TOMORROW.

    WHY THE SHAPE SURVIVES. Fourteen call sites and four test files subscript
    this name, and most of them are asking about the margin for TOMORROW --
    which is the day every schedule question in this project concerns. So the
    mapping still exists and still yields `(minutes, what, provenance)`, but
    the minutes are DERIVED on each lookup from tomorrow's offset rather than
    stored.

    ANY CALLER THAT KNOWS ITS DAY SHOULD USE `deadline_floor_for` INSTEAD, and
    the two functions in this module that serve a specific day now do.
    """

    def __getitem__(self, day_type):
        tomorrow = _dt.date.today() + _dt.timedelta(days=1)
        return deadline_floor_for(day_type, tomorrow)

    def get(self, day_type, default=None):
        try:
            return self[day_type]
        except KeyError:
            return default

    def keys(self):
        return DEADLINE_FLOOR_PROVENANCE.keys()

    def items(self):
        return ((k, self[k]) for k in self.keys())

    def values(self):
        return (self[k] for k in self.keys())

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(DEADLINE_FLOOR_PROVENANCE)


DEADLINE_FLOOR = _DeadlineFloorForToday()

#: RETAINED for the record: what the self-anchored derivation produced, and
#: what it cost. Nothing computes from this -- it is here so the next reader
#: meets the rejected design rather than re-deriving it.
REJECTED_SELF_ANCHORED_FLOOR = {
    "weekday": 9 * 60 + 45, "weekend": 9 * 60 + 15,
    "why_rejected": "ceil_15(ready_binding + 60) is a tautology -- it makes "
                    "the margin >= 60 by construction -- and freezing it "
                    "against a live series breached 60 within one day when "
                    "mean+3sd drifted +11.4m on a single new observation",
}

#: Kept as documentation of what the filter means; the filter itself runs in
#: Python because `EXTRACT(MICROSECONDS ...)` is PostgreSQL-only.
_PHANTOM_RULE = ("match_date with sub-second precision is a `datetime.now()` "
                 "ingestion stamp, not a kickoff; 00:00 rows are placeholders")


def as_datetime(v):
    """Coerce a raw-SQL DATETIME to `datetime`, or return it unchanged.

    THIRD INSTANCE TODAY of the same defect. Raw `text()` SQL returns DATETIME
    columns as **strings** on SQLite and as `datetime` on psycopg2, so any
    arithmetic on them raises `TypeError: unsupported operand type(s) for -`
    in tests while working in production. It bit `load_observations` this
    morning and then `verify_offset` an hour later. One coercion, imported by
    both, so there is no fourth.
    """
    if isinstance(v, _dt.datetime) or v is None:
        return v
    if isinstance(v, str):
        try:
            return _dt.datetime.fromisoformat(
                v.replace("Z", "+00:00").strip()).replace(tzinfo=None)
        except ValueError:
            return None
    return v


def day_type(day: _dt.date) -> str:
    """`'weekend'` for Sat/Sun, else `'weekday'`."""
    return "weekend" if day.weekday() >= 5 else "weekday"


def discovered_deadline(session, day: _dt.date) -> Optional[int]:
    """Earliest kickoff IN MINUTES for `day`, from the discovered card.

    `None` when the card for that day is not discovered — which is **not** a
    late deadline and must never be replaced by a constant. An undiscovered
    card is an unknown deadline, and the caller says so.

    The phantom exclusion is applied here rather than left to the caller,
    because the one query that forgets it reads 07:53.
    """
    from sqlalchemy import text

    # THE PHANTOM FILTER IS APPLIED IN PYTHON, NOT IN SQL, and that is a
    # portability decision with a precedent: `EXTRACT(MICROSECONDS FROM ...)`
    # is PostgreSQL-only and SQLite has no equivalent, so a SQL-side filter
    # works in production and raises in every test. Same shape as the
    # `greatest()` / `max()` lesson in `api_budget`. A day holds at most ~120
    # fixtures, so fetching them and filtering here costs nothing.
    rows = session.execute(text("""
        SELECT m.match_date FROM matches m
        WHERE m.is_fixture = true
          AND m.match_date >= :lo AND m.match_date < :hi
          AND m.league NOT LIKE 'world/%'
    """), {"lo": _dt.datetime.combine(day, _dt.time.min),
           "hi": _dt.datetime.combine(day + _dt.timedelta(days=1),
                                      _dt.time.min)}).fetchall()
    clean = [as_datetime(r[0]) for r in rows]
    clean = [d for d in clean
             if d is not None
             and d.microsecond == 0        # not a phantom
             and d.hour >= 6]              # not a 00:00 placeholder
    if not clean:
        return None
    ko = min(clean)
    return ko.hour * 60 + ko.minute


def projected_ready(cron_minutes: int, delays: Sequence[float],
                    durations: Sequence[float]) -> dict:
    """`{'max': m, 'mean_plus_3sd': m}` picks-ready times, in minutes past
    midnight, from the LIVE delay and duration series.

    Both, never the mean alone — the mean is what hid the 09:45 deadline for a
    month.
    """
    import statistics

    if not delays or not durations:
        return {}
    d_mean = statistics.mean(delays)
    d_sd = statistics.pstdev(delays) if len(delays) > 1 else 0.0
    u_mean = statistics.mean(durations)
    u_sd = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    return {
        "max": cron_minutes + max(delays) + max(durations),
        "mean_plus_3sd": cron_minutes + d_mean + 3 * d_sd + u_mean + 3 * u_sd,
        "n_delays": len(delays),
        "n_durations": len(durations),
    }


def prospective_margin(earliest_kickoff_min: Optional[int],
                       cron_minutes: Optional[int],
                       delays: Sequence[float],
                       durations: Sequence[float],
                       serves: _dt.date = None,
                       min_margin: int = MIN_ACCEPTABLE_MARGIN_MIN,
                       deadline_basis: str = "discovered card") -> dict:
    """The margin against the DISCOVERED deadline, with its own verdict.

    Returns a dict carrying the deadline it was measured against, because a
    margin without its deadline is a number without a referent.

    THREE OUTCOMES, kept apart:
      `UNKNOWN`  the card is not discovered, or no series exists for the cron.
                 NOT safe, and it does not alarm either — it says so.
      `ALARM`    the projected margin is below the stated minimum.
      `CLEAR`    it is not.
    """
    out = {
        "serves": serves.isoformat() if serves else None,
        "day_type": day_type(serves) if serves else None,
        "deadline_min": earliest_kickoff_min,
        # ── THE BASIS IS A PARAMETER, NOT A LITERAL. 2026-10-19. ────────
        #
        # It was the fixed string `"discovered card"`, and the first caller to
        # pass a deadline from somewhere else printed
        # `deadline=12:00 (discovered card)` for a value that came from the
        # REFERENCE combined with a CONSTANT and never touched the card. Same
        # shape as the `LIVE 0 4 * * 1-5` label fixed the day before: a label
        # pinned to its original producer, still printing after the producer
        # changed. The caller names its own source.
        "deadline_basis": deadline_basis,
        "min_margin": min_margin,
    }
    if earliest_kickoff_min is None or cron_minutes is None:
        out["verdict"] = "UNKNOWN"
        out["reason"] = ("the card for that day is not discovered"
                         if earliest_kickoff_min is None
                         else "no cron resolved for that day type")
        return out
    ready = projected_ready(cron_minutes, delays, durations)
    if not ready:
        out["verdict"] = "UNKNOWN"
        out["reason"] = "no delay/duration series under the cron in force"
        return out
    out["ready_max"] = ready["max"]
    out["ready_3sd"] = ready["mean_plus_3sd"]
    out["n"] = ready["n_delays"]
    out["margin_max"] = earliest_kickoff_min - ready["max"]
    out["margin_3sd"] = earliest_kickoff_min - ready["mean_plus_3sd"]
    out["binding"] = min(out["margin_max"], out["margin_3sd"])
    out["verdict"] = "ALARM" if out["binding"] < min_margin else "CLEAR"
    return out


#: THE CLEARING ACTION, named here so the alert can carry it. CLR-2: an alarm
#: must be clearable by an action a person can take, and recording the action in
#: the ledger alone is what made `fixtures_zero_active` noise.
#:
#: There is exactly ONE action available today and it is manual, which is
#: acceptable only because it is NAMED: dispatch `daily-picks` by hand that
#: morning, early enough that picks complete before the discovered kickoff.
#: `workflow_dispatch` exists on the workflow and a dispatched run is not
#: subject to the scheduler's 3h58m-5h04m lateness, so it clears the predicate
#: directly rather than by waiting.
CLEARING_ACTION = (
    "DISPATCH daily-picks MANUALLY that morning "
    "(gh workflow run daily-picks.yml). A dispatched run skips the scheduler's "
    "measured 3h58m-5h04m lateness, so it is the one action that changes this "
    "margin. Nobody else performs it: it is the operator's, and there is no "
    "automated alternative today."
)


def format_prospective(m: dict) -> str:
    """One line, carrying the deadline it was measured against."""
    def hm(v):
        v = int(round(v))
        return f"{v // 60:02d}:{v % 60:02d}"

    head = (f"prospective_margin[serves={m.get('serves')} "
            f"{m.get('day_type')}")
    if m["verdict"] == "UNKNOWN":
        return head + f" UNKNOWN: {m['reason']} - NOT safe]"
    return (head + f" deadline={hm(m['deadline_min'])} "
            f"({m.get('deadline_basis', 'UNSTATED BASIS')}) "
            f"ready max={hm(m['ready_max'])} 3sd={hm(m['ready_3sd'])} "
            f"| margin max={m['margin_max']:+.0f}m 3sd={m['margin_3sd']:+.0f}m "
            f"binding={m['binding']:+.0f}m min={m['min_margin']}m n={m['n']}"
            + ("  ** ACTION REQUIRED **]" if m["verdict"] == "ALARM" else "]"))


# ─────────────────────────────────────────────────────────────────────────────
# THE COVERAGE RESIDUAL IS A DISTRIBUTION, AND IT IS NOT THE ONE THAT BITES
#
#     A residual is a distribution until its tail is measured. One observation
#     is not a bound.
#
# Easter Monday 2026 measured a 15-minute gap. That was one case, and it was
# not even a typical one.
#
# THE RESIDUAL, `measured` 2026-10-19 over production `matches`, phantoms and
# `world/*` excluded, configured leagues only, n=585 days carrying BOTH a
# covered and an uncovered fixture. Defined as
# `min(kickoff | 21 uncovered) - min(kickoff | 9 covered)`, so NEGATIVE means
# the uncovered card is earlier and the reference is LATE by that much:
#
#   stratum             n    negative      median   p90   p99   max   >60m
#   ALL               585   388 (66.3%)      60     240   450   585   45.6%
#   weekday           305   235 (77.0%)     120     285   510   585   59.0%
#   weekend           280   153 (54.6%)      30     180   330   360   31.1%
#   holiday            14    11 (78.6%)      60     360   510   510   42.9%
#
# So the raw residual's tail is NINE AND THREE-QUARTER HOURS and it is
# negative on two days in three. Taken at face value that would say the
# combination is unsafe on half of all days, and the needed minimum is 570
# minutes. Both conclusions are wrong, and the reason is worth stating.
#
# THE RESIDUAL IS NOT WHAT THE MARGIN BEARS. The deadline is
# `min(reference, constant)` and the CONSTANT binds on 1087 of 1090 days --
# 12:00 weekday and 10:15 weekend both sit far below the first covered
# kickoff. What the margin bears is the EXPOSURE:
#
#     exposure = max(0, deadline - min(kickoff | all configured leagues))
#              = minutes our deadline sits AFTER the first kickoff we bet
#
# `measured` 2026-10-19, same population, n=1090 days:
#
#   stratum      exposure>0     median  p90   p99   max   >60m
#   ALL          21 (1.9%)        0      0     45   150   8 (0.7%)
#   weekday      14 (2.0%)        0      0     75   150   8 (1.1%)
#   weekend       7 (1.9%)        0      0     15    30   0 (0.0%)
#   wd/holiday    3 (13.6%)       0     15     90    90   1 (4.5%)
#
# AND IT IS NOT INTRODUCED BY THE REFERENCE. Under the constant ALONE the
# exposure is max 150m on 23 of 1090 days; under the combination it is max
# 150m on 21. `min()` can only lower the deadline, so exposure(combination)
# <= exposure(constant) ALWAYS -- measured, 3 days improved (median 30m, max
# 90m), ZERO days made worse. The tail is a property of the 12:00 weekday
# constant, which the 2026-10-10 analysis already recorded as the ~12th
# percentile with an observed minimum of 09:59.
#
# WHAT DRIVES THE TAIL: genuine early kickoffs in a few leagues, not a
# timezone artefact. `bulgaria/efbet-league` is the earliest uncovered league
# on 7 of the 10 worst days (its own distribution: min 09:30, p05 10:30, p50
# 15:30 UTC -- i.e. 11:30/12:30 local, normal for Bulgaria), with
# `finland/veikkausliiga` at 09:59 and `italy/serie-b` at 10:30. Checked
# against three other UTC+2/+3 leagues and against the Premier League, whose
# own minimum is 11:30 UTC with 24 fixtures before 12:00 -- which is why the
# reference binds at all.

#: `measured` 2026-10-19. The exposure distribution per day type: how many
#: minutes the COMBINED deadline sits after the true first kickoff.
#:
#: STRATIFIED BY DAY TYPE, not by holiday, because the deadline itself is keyed
#: by day type -- and the weekday figure DOMINATES its own holiday subset
#: (weekday/holiday max 90m against weekday max 150m), so using the day-type
#: number is the conservative choice rather than a convenient one.
COVERAGE_EXPOSURE = {
    "weekday": {"n": 715, "pos": 0, "p99": 0, "max": 0, "over60": 0},
    "weekend": {"n": 375, "pos": 0, "p99": 0, "max": 0, "over60": 0},
}

#: RECOMPUTED 2026-10-20 WHEN THE DEADLINE WAS RE-ANCHORED TO THE WORLD.
#:
#:   constants                     exposure>0   p99    max    >60m
#:   12:00 / 10:15 (percentiles)    21 days      45m   150m   8 days
#:   09:45 / 09:15 (self-anchored)   1 day        0m    15m   0 days
#:   09:30 / 09:45 (observed min)    0 days       0m     0m   0 days
#:
#: **ZERO BY CONSTRUCTION**, because the deadline now IS the observed earliest
#: kickoff: `max(0, deadline - true_earliest)` cannot be positive over the
#: measured window. The 15-minute weekday residual that the self-anchored floor
#: carried is gone -- it existed only because that floor sat 15 minutes ABOVE
#: the observed minimum in order to manufacture a 60-minute margin.
#:
#: THE TERM IS STILL PRINTED AT ZERO. It is satisfied, not absent, and the next
#: person to widen the reference's coverage or re-measure the minimum needs to
#: find the figure already on the line. A term that disappears when satisfied
#: cannot be seen to have been satisfied.
#:
#: AND THE TRADE MOVED, IT DID NOT VANISH: the exposure is zero and the MARGIN
#: now fails instead -- +40.2m against the registered 60. That is the honest
#: accounting. The self-anchored floor bought a passing margin by accepting
#: exposure and by defining the margin in terms of itself.
EFFECTIVE_MIN_MARGIN = {
    dt: MIN_ACCEPTABLE_MARGIN_MIN + v["p99"]
    for dt, v in COVERAGE_EXPOSURE.items()
}


def coverage_exposure(day: _dt.date) -> dict:
    """The measured exposure for `day`'s stratum, with its basis.

    Returned rather than inlined so the figure printed beside a deadline and
    the figure the verdict uses are THE SAME NUMBER. The `LIVE 0 4 * * 1-5`
    label is what happens when they are two.
    """
    dtype = day_type(day)
    e = COVERAGE_EXPOSURE[dtype]
    return {
        "day_type": dtype,
        "p99": e["p99"],
        "max": e["max"],
        "n": e["n"],
        "over60": e["over60"],
        "effective_min": EFFECTIVE_MIN_MARGIN[dtype],
        "basis": (f"measured 2026-10-19, n={e['n']} {dtype} days, production "
                  f"`matches`, phantoms and world/* excluded; exposure = "
                  f"max(0, deadline - first kickoff over the 30 configured "
                  f"leagues); p99={e['p99']}m max={e['max']}m, exceeds 60m on "
                  f"{e['over60']} day(s)"),
    }

# ─────────────────────────────────────────────────────────────────────────────
# THE DAY-AHEAD DEADLINE COMES FROM THE EXTERNAL REFERENCE — 2026-10-19
#
#     A minimum over a subset is an UPPER BOUND on the true minimum.
#
# WHY NOT FROM US. `max_days_ahead` is a rolling 24 hours from the run, so our
# own `matches` has NEVER held tomorrow's early card: `measured` 2026-10-18
# over n=1,131 rows, the maximum write-to-kickoff lead is 23.62h and rows known
# more than 24h ahead number ZERO. The Flashscore probe built to fetch it paid
# 240 page-loads for 8 refusals and zero deadlines, and was disabled.
#
# WHY FROM THE REFERENCE. football-data.org answers `/matches?date=D+1`
# directly. `measured` 2026-10-19 for 2026-10-20 it returned **16:45**
# (europe/champions-league) while our own table held **0 rows** for that day.
# Free tier: 10 calls/minute, **no daily limit**, so one call a run costs
# neither an API-Football request nor a TheOddsAPI credit nor a Flashscore
# page-load.
#
# AND IT IS NOT A BOUND. `measured` 2026-10-19 it covers **9 of the 30
# configured leagues**. A minimum over 9 errs LATE, which is the unsafe
# direction, so the reference is COMBINED with the conservative constant and
# never substituted for it:
#
#     deadline = min(reference over covered leagues, the day-type constant)
#
# This can only TIGHTEN. It is still an upper bound on the true minimum --
# `measured` on Easter Monday 2026 the truth was 10:15 in `poland/ekstraklasa`,
# which the reference does not cover, and the combination returned 10:30: 90
# minutes better than the constant and 15 minutes short of the truth. The
# residual is reported, not hidden.

#: The reference, named so no figure derived from it is quoted without a source.
REFERENCE_NAME = "football-data.org"
REFERENCE_BASE = "https://api.football-data.org/v4"

#: Where the key comes from. Absent is a THIRD STATE, not an empty answer.
REFERENCE_KEY_ENV = "FOOTBALL_DATA_ORG_KEY"

#: `reference_earliest` states, enumerated so the set is read, not inferred.
REF_OK = "OK"                       #: a covered fixture was found for that day
REF_NO_FIXTURE = "NO_FIXTURE"       #: rows came back, NONE for the asked date
REF_EMPTY_RESPONSE = "EMPTY_RESPONSE"  #: the provider returned no rows at all
REF_NO_KEY = "NO_KEY"               #: we could not ask
REF_FETCH_FAILED = "FETCH_FAILED"   #: we asked and could not read the answer
#: FOUR STATES WHERE THERE ARE FOUR FACTS. `NO_FIXTURE` and `EMPTY_RESPONSE`
#: were one state until 2026-10-19, and they are not the same claim: an empty
#: response says the provider has nothing anywhere near that query, while
#: NO_FIXTURE says it answered with rows and not one of them was for the day
#: we asked about -- which is what 2027-04-02 did, returning 39 matches all
#: dated 2027-04-03. Conflating them hides exactly the case where the provider
#: answered a different question.
REFERENCE_STATES = (REF_OK, REF_NO_FIXTURE, REF_EMPTY_RESPONSE, REF_NO_KEY,
                    REF_FETCH_FAILED)

#: The two that mean "could not look", which is NOT "found nothing". Both fall
#: to the constant, and they print differently because their remedies differ:
#: one needs a secret, the other needs the provider to answer.
REF_UNEVALUATED = (REF_NO_KEY, REF_FETCH_FAILED)


def _reference_fetch(day: _dt.date, api_key: str, timeout: float = 15.0):
    """GET `/matches?date=day`. Returns the decoded payload, or raises.

    STDLIB ONLY, and that is not a style choice: `ci-audit.yml` installs no
    packages ("No pip install: ci_audit imports the standard library only"),
    and this is reached from the audit. `httpx` here would kill that job at
    import.
    """
    import json as _json
    import urllib.error  # noqa: F401  (imported for the caller's except)
    import urllib.request

    url = f"{REFERENCE_BASE}/matches?date={day.isoformat()}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": api_key})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return _json.loads(resp.read().decode("utf-8"))


def reference_earliest(day: _dt.date, fetch=None, api_key: str = None) -> dict:
    """Earliest covered kickoff for `day` per the reference, with its coverage.

    `fetch` is injectable so every branch is testable without a network, and
    so the known-wrong cases below are driven rather than waited for.

    `?date=X` IS A TWO-DAY WINDOW, AND THE PROVIDER SAYS SO. `measured`
    2026-10-19: asked for 2027-04-02 it returned **39 matches, every one dated
    2027-04-03**, and its own echo read

        filters: {dateFrom: 2027-04-02, dateTo: 2027-04-03, ...}

    An earlier note here said the provider "does not honour `date`". That was
    imprecise and its own echo disproves it: `date=X` expands to `[X, X+1]`
    INCLUSIVE, so those 39 rows are the honest second day of the window. The
    defect would be OURS -- attributing a two-day window's minimum to one day.

    So the response is validated against the request TWICE: the echoed
    `filters.dateFrom` is compared against what was asked, and every row is
    checked against its own `utcDate`. Mismatched rows are COUNTED rather than
    merely dropped, because a silent filter and a provider with nothing to say
    look identical from the outside.
    """
    import os

    from src.scrapers.footballdata_competitions import (COMPETITION_MAP,
                                                        STATUS_NOT_PLAYABLE)

    out = {
        "day": day.isoformat(),
        "reference": REFERENCE_NAME,
        "min": None,
        "league": None,
        "n_usable": 0,
        "n_raw": 0,
        "leagues_present": [],
        "leagues_absent": [],
        "dropped": {"other_day": 0, "unmapped": 0, "world": 0,
                    "not_playable": 0, "unparseable": 0},
    }
    key = api_key if api_key is not None else (
        os.environ.get(REFERENCE_KEY_ENV, "") or "").strip()
    if not key and fetch is None:
        out["state"] = REF_NO_KEY
        out["reason"] = (f"{REFERENCE_KEY_ENV} is not set, so the reference was "
                         f"NOT asked. This is 'could not look', not 'nothing "
                         f"tomorrow' - the deadline falls to the constant and "
                         f"says so.")
        return out
    try:
        payload = (fetch or _reference_fetch)(day, key)
    except Exception as exc:                      # noqa: BLE001 - any failure
        out["state"] = REF_FETCH_FAILED
        out["reason"] = (f"{REFERENCE_NAME} did not answer: "
                         f"{type(exc).__name__}. 'Could not look', not "
                         f"'nothing tomorrow'.")
        return out

    # VALIDATE AGAINST THE REQUEST USING THE PROVIDER'S OWN ECHO. It returns
    # `filters: {dateFrom, dateTo}`; if `dateFrom` is not the day we asked
    # about, the response answers a different question and nothing below may
    # be attributed to `day`.
    out["echo"] = dict((payload or {}).get("filters") or {})
    _from = out["echo"].get("dateFrom")
    out["echo_matches_request"] = (_from == day.isoformat()) if _from else None
    if out["echo_matches_request"] is False:
        out["state"] = REF_FETCH_FAILED
        out["reason"] = (
            f"{REFERENCE_NAME} echoed dateFrom={_from!r} for a request of "
            f"{day.isoformat()} - it answered a DIFFERENT question, so no row "
            f"in this response may be attributed to {day}. Treated as a "
            f"failed fetch: we did not get the answer we asked for.")
        return out

    matches = (payload or {}).get("matches") or []
    out["n_raw"] = len(matches)
    if not matches:
        out["state"] = REF_EMPTY_RESPONSE
        out["reason"] = (f"{REFERENCE_NAME} returned NO ROWS AT ALL for "
                         f"{day}. Distinct from 'rows came back and none was "
                         f"for the day we asked about' - that is NO_FIXTURE "
                         f"and it is how a provider answers a different "
                         f"question.")
        from src.scrapers.footballdata_competitions import REFERENCE_LEAGUES
        out["leagues_absent"] = sorted(REFERENCE_LEAGUES)
        return out
    rows = []
    for m in matches:
        code = ((m or {}).get("competition") or {}).get("code", "")
        league = COMPETITION_MAP.get(code)
        if not league:
            out["dropped"]["unmapped"] += 1
            continue
        if league.startswith("world/"):
            out["dropped"]["world"] += 1
            continue
        if ((m or {}).get("status") or "") in STATUS_NOT_PLAYABLE:
            out["dropped"]["not_playable"] += 1
            continue
        ko = as_datetime((m or {}).get("utcDate") or "")
        if ko is None:
            out["dropped"]["unparseable"] += 1
            continue
        if ko.date() != day:
            # THE TRAP, counted. See the docstring.
            out["dropped"]["other_day"] += 1
            continue
        rows.append((ko, league))

    from src.scrapers.footballdata_competitions import REFERENCE_LEAGUES

    out["n_usable"] = len(rows)
    out["leagues_present"] = sorted({lg for _k, lg in rows})
    out["leagues_absent"] = sorted(REFERENCE_LEAGUES - set(out["leagues_present"]))
    if not rows:
        out["state"] = REF_NO_FIXTURE
        out["reason"] = (
            f"{REFERENCE_NAME} answered with {out['n_raw']} row(s) and NOT "
            f"ONE is a covered fixture on {day}: {out['dropped']}. "
            + (f"{out['dropped']['other_day']} of them were for a DIFFERENT "
               f"DATE - the provider answered a different question than it "
               f"was asked, which is why they are counted rather than merely "
               f"dropped. "
               if out["dropped"]["other_day"] else "")
            + "The reference LOOKED and found nothing for this day - distinct "
              "from not having looked, and from having returned nothing at "
              "all.")
        return out
    ko, league = min(rows, key=lambda t: t[0])
    out["state"] = REF_OK
    out["min"] = ko.hour * 60 + ko.minute
    out["league"] = league
    return out


def combined_deadline(day: _dt.date, ref: dict) -> dict:
    """`min(reference, constant)` for `day`, carrying its basis and coverage.

    RULE 1, MECHANICAL. The reference is a minimum over a SUBSET, so it is an
    upper bound on the true minimum and errs LATE. `min` with the day-type
    constant means the answer can only ever be TIGHTER than what the schedule
    was already justified against -- it can never loosen it, whatever the
    reference says or fails to say.

    `basis` names which side bound the answer, because a deadline without its
    basis is a number without a referent, and because the two sides fail
    differently: a reference that goes dark leaves the constant standing, while
    a constant that is wrong is wrong on every day at once.
    """
    from src.scrapers.footballdata_competitions import \
        REFERENCE_COVERAGE_MEASURED

    dtype = day_type(day)
    # THE FLOOR, not the percentile record. Repointed 2026-10-19 when the
    # fallback stopped being a percentile and became a derived value, and
    # again 2026-10-22 when it stopped being a UTC CONSTANT: this function
    # serves a specific `day`, so it asks for that day's conversion rather
    # than taking tomorrow's from the mapping.
    const_min, const_what, const_prov = deadline_floor_for(dtype, day)
    nc, nt = REFERENCE_COVERAGE_MEASURED
    out = {
        "serves": day.isoformat(),
        "day_type": dtype,
        "constant_min": const_min,
        "constant_basis": f"{const_what} ({const_prov})",
        "reference": REFERENCE_NAME,
        "reference_state": ref.get("state"),
        "reference_min": ref.get("min"),
        "reference_league": ref.get("league"),
        "coverage": f"{nc} of {nt} configured leagues",
        "leagues_present": ref.get("leagues_present") or [],
        "leagues_absent": ref.get("leagues_absent") or [],
        # `ref.get(k, 0)`, not `ref.get(k) or 0`: the fallback happens to be
        # the same value here, but the idiom is the one that killed
        # `missing_scheduled_runs`, and it reads as if zero were absent.
        "n_usable": ref.get("n_usable", 0),
        "dropped": ref.get("dropped") or {},
    }
    # THE RESIDUAL TRAVELS WITH THE DEADLINE. An improved deadline must never
    # read as a safe one: `min(reference, constant)` is still up to
    # `exposure["max"]` minutes LATER than the first kickoff we bet, because
    # the reference sees 9 of 30 leagues.
    out["exposure"] = coverage_exposure(day)
    rmin = ref.get("min")
    if rmin is None:
        out["deadline_min"] = const_min
        out["bound_by"] = "CONSTANT"
        out["basis"] = (
            f"CONSTANT {const_min // 60:02d}:{const_min % 60:02d} - the "
            f"reference supplied NO value ({ref.get('state')}). "
            + ("NOT reference-verified: " if ref.get("state") in REF_UNEVALUATED
               else "")
            + (ref.get("reason") or ""))
        out["reference_verified"] = ref.get("state") in (REF_NO_FIXTURE,
                                                         REF_EMPTY_RESPONSE)
        return out
    out["deadline_min"] = min(rmin, const_min)
    out["bound_by"] = "REFERENCE" if rmin < const_min else "CONSTANT"
    out["reference_verified"] = True
    if rmin < const_min:
        out["basis"] = (
            f"REFERENCE {rmin // 60:02d}:{rmin % 60:02d} in "
            f"{ref.get('league')} ({REFERENCE_NAME}, {out['coverage']}), "
            f"TIGHTER than the {const_min // 60:02d}:{const_min % 60:02d} "
            f"constant by {const_min - rmin} min")
    else:
        out["basis"] = (
            f"CONSTANT {const_min // 60:02d}:{const_min % 60:02d} - the "
            f"reference's {rmin // 60:02d}:{rmin % 60:02d} in "
            f"{ref.get('league')} is LATER, and a minimum over "
            f"{out['coverage']} is an UPPER BOUND on the true minimum, so it "
            f"may not relax the constant")
    return out


def format_reference_deadline(c: dict) -> str:
    """One line carrying the deadline, its basis, and its coverage."""
    def hm(v):
        v = int(round(v))
        return f"{v // 60:02d}:{v % 60:02d}"

    absent = c.get("leagues_absent") or []
    return (
        f"day_ahead_deadline[serves={c['serves']} {c['day_type']} "
        f"deadline={hm(c['deadline_min'])} bound_by={c['bound_by']} "
        f"ref={c['reference_state']}"
        + (f"@{hm(c['reference_min'])}" if c.get("reference_min") is not None
           else "")
        + f" coverage={c['coverage']} present={len(c.get('leagues_present') or [])}"
        f" absent={len(absent)} usable={c.get('n_usable')} "
        f"dropped={c.get('dropped')}]"
        + (f"{chr(10)}  residual[coverage exposure {e['day_type']}: "
           f"p99={e['p99']}m max={e['max']}m over60={e['over60']}/{e['n']}d "
           f"-> effective minimum {e['effective_min']}m = "
           f"{MIN_ACCEPTABLE_MARGIN_MIN}m registered + {e['p99']}m p99 | "
           f"basis: {e['basis']}]"
           if (e := c.get("exposure")) else ""))


# ══════════════════════ TWO MARGINS, NOT ONE ══════════════════════════════
#
#     2026-10-21. "Separate the structural margin from the actionable one."
#
# `combined_deadline` returns `min(reference, floor)`, and the floor sits AT
# the earliest kickoff any covered league has ever had -- so the floor binds
# every day and one number was carrying two different facts. Measured
# 2026-10-21: floor 09:30 against today's actual earliest kickoff of 15:30,
# six hours pessimistic.
#
#   STRUCTURAL  against the conservative floor. True every day, NO action
#               clears it, and it is a statement about the pipeline's duration
#               against the worst case football can produce. It is the
#               duration problem, already sized at 16.7 minutes.
#
#   ACTIONABLE  against THE DAY'S earliest kickoff per the reference. This is
#               the one a manual dispatch answers, so by CLR-2 it is the one
#               that may alarm.
#
# ─────────────────────── AND THE UNSAFE DIRECTION ─────────────────────────
#
# THE ACTIONABLE MARGIN IS AN UPPER BOUND, AND ITS RESIDUAL EXCEEDS THE
# REGISTERED MINIMUM ON MOST DAYS. The reference covers 9 of 30 configured
# leagues, so its minimum errs LATE whenever an uncovered league opens the
# card. `simulated` 2026-10-21 over n=605 days with a covered fixture, using
# the reference's own league set as the proxy for its coverage:
#
#     p50   75 min      <- today's case exactly: 16:45 reported, 15:30 true
#     p75  210 min
#     p90  360 min
#     p95  480 min
#     p99  592 min
#     max 1185 min
#     residual > 60m on 312 of 605 days (51.6%); == 0 on 212
#
# SO THE ALARM'S THRESHOLD DOES *NOT* EXCEED THE RESIDUAL. It is exceeded by
# it on 51.6% of days. That is the opposite of what a safety threshold needs,
# and it is why the structural margin is still printed and still carries the
# safety claim: an actionable margin clearing 60 is NECESSARY, not SUFFICIENT.
# Raising the threshold to p99 would make it fire every day, which rule 4
# forbids, and subtracting the residual reproduces the floor. Neither is an
# improvement, so the residual is REPORTED beside the margin instead of being
# folded into it.
REFERENCE_RESIDUAL = {
    "n_days": 605,
    "p50": 75, "p75": 210, "p90": 360, "p95": 480, "p99": 592, "max": 1185,
    "over_60": 312, "zero": 212,
    "provenance": ("simulated 2026-10-21 from our own card over 2024-01-01.., "
                   "comparing the earliest kickoff among the reference's 9 "
                   "covered league keys against the earliest over all 30 "
                   "configured; the phantom 00:00 ingestion stamps excluded"),
    "exceeds_threshold": True,
    "today_case": "2026-10-21: reference 16:45, true earliest 15:30 = 75m",
}

#: THE STRUCTURAL BREACH, ACKNOWLEDGED WITH A DATE. The registered 60 is NOT
#: edited -- it was stated 2026-10-10 before any of this -- and the exposure
#: stays its own term at zero rather than being folded in.
STRUCTURAL_ACK = {
    "since": "2026-10-21",
    # RE-MEASURED the same day. It read -19.8 against a duration term that
    # charged the card for work done AFTER it was delivered; the term now
    # measures time-to-picks and the shortfall is -7.0. A figure in an
    # acknowledgement is the one thing a reader trusts without checking, so
    # it is re-stated rather than left to drift -- the stale-literal defect.
    # -7.4, NOT -7.0, AND THE CHECK CAUGHT THAT ON ITS FIRST EXPOSURE.
    # I computed the margin at +52.6 against a required 60 and wrote -7.0,
    # which is 60 - 52.6 rounded the wrong way. The deterioration predicate
    # refused to cover the live -7.4 and alarmed, which is exactly what it is
    # for -- and it is the same shape as the self-anchored floor: a figure I
    # derived and then transcribed loosely.
    "binding": -7.4,
    "why": ("the structural margin is the DURATION problem, not a deadline "
            "choice: the floor IS the observed earliest kickoff, so no "
            "anchoring clears it. 14.6 of the original 19.8 minutes were an "
            "INSTRUMENTATION ERROR (the term measured run start to run END, "
            "including the post-picks results scrape: mean 12.8m, max 49.6m, "
            "n=111) and are gone without any step getting faster. 7.0 "
            "remain, and they are real"),
    "action": ("~7.4 min of time-to-picks remain. MEASURED 2026-10-21 over "
               "n=72 runs: parallelising the Claude review behind an atomic "
               "marker buys +30.5m by itself (it cuts the SPREAD, sd 18.4 -> "
               "10.8, and the bound is mean+3sd); moving the ML retrain off "
               "the picks path buys +3.5m; the shipped request-pacing change "
               "buys +4.0m once it reaches the series. Retrain + pacing alone "
               "leaves 1.3m. Or dispatch manually that morning"),
    "review": "2026-11-21",
}

#: THE FIELD `binding` IS A SIGNED DEFICIT: `-7.0` means seven minutes SHORT
#: of the registered minimum, which is the same arithmetic as the margin
#: itself (margin +53.0 against a required 60 is a deficit of -7.0). Stated
#: because the previous acknowledgement stored a figure whose convention a
#: reader had to infer.


def structural_breach_is_acknowledged(binding: float, today=None) -> dict:
    """`STRUCTURAL_ACK` when it covers this figure, `{}` when it does not.

    TWO COMPARISONS, AND THE LAST ACKNOWLEDGEMENT MADE ONLY ONE. The retired
    `ACKNOWLEDGED_MARGIN_BREACH` compared `binding` and never `review`, so it
    went on absorbing a breach for days after its own review date had passed —
    a deferral with a date nothing reads is not a deferral. Both are compared
    here, expiry first.

      EXPIRES      past its review date, or an unparseable date, absorbs
                   NOTHING. An acknowledgement that cannot be dated is not one.
      DETERIORATES a figure WORSE than the acknowledged one alarms, because
                   deterioration is an event and the acknowledgement was made
                   against a measurement, not against the condition in general.

    The registered `MIN_ACCEPTABLE_MARGIN_MIN` is NOT consulted and NOT
    edited: this answers "has this been seen and recorded", not "is it
    acceptable".
    """
    today = today or _dt.date.today()
    ack = STRUCTURAL_ACK
    review = ack.get("review")
    if not review:
        return {}
    try:
        if today > _dt.date.fromisoformat(review):
            return {}
    except (ValueError, TypeError):
        return {}
    if binding < ack["binding"]:
        return {}
    return dict(ack)


def two_margins(day, ref: dict, cron_minutes,
                delays, durations) -> dict:
    """Both margins for `day`, each with its basis and its coverage.

    THE PAIR IS NEVER REDUCED TO THE OPTIMISTIC HALF. `structural` is always
    present; `actionable` carries a `verdict` of `UNEVALUATED` when the
    reference supplied no value, which is a different fact from a clear
    margin and says so.
    """
    from src.scrapers.footballdata_competitions import \
        REFERENCE_COVERAGE_MEASURED

    dtype = day_type(day)
    # Per-day conversion, as above: this function also serves one named day.
    floor_min, floor_what, floor_prov = deadline_floor_for(dtype, day)
    nc, nt = REFERENCE_COVERAGE_MEASURED
    coverage_all = f"ALL {nt} configured leagues (the observed minimum)"
    coverage_ref = f"{nc} of {nt} configured leagues"

    # THE MINIMUM COMES FROM THE SAME EXPOSURE DICT THAT IS PRINTED, so the
    # two cannot diverge -- the contract
    # `test_the_exposure_the_VERDICT_uses_is_the_one_PRINTED` pins. The
    # exposure belongs to THIS margin and not to the actionable one: it
    # measures how far the FLOOR can sit after the true earliest kickoff, and
    # it is zero by construction because the floor IS that kickoff. The
    # reference's own residual is a different quantity, reported separately as
    # REFERENCE_RESIDUAL rather than folded into a threshold.
    _exposure = coverage_exposure(day)
    structural = prospective_margin(
        floor_min, cron_minutes, delays, durations, serves=day,
        min_margin=_exposure["effective_min"],
        deadline_basis=f"FLOOR {floor_what} - {coverage_all}")
    structural["kind"] = "STRUCTURAL"
    structural["coverage"] = coverage_all
    structural["exposure"] = _exposure
    structural["clearing_action"] = STRUCTURAL_ACK["action"]
    structural["acknowledged"] = STRUCTURAL_ACK

    rmin = ref.get("min")
    actionable = prospective_margin(
        rmin, cron_minutes, delays, durations, serves=day,
        deadline_basis=(f"REFERENCE {REFERENCE_NAME} - {coverage_ref}"))
    actionable["kind"] = "ACTIONABLE"
    actionable["coverage"] = coverage_ref
    actionable["reference_state"] = ref.get("state")
    actionable["residual"] = REFERENCE_RESIDUAL
    actionable["clearing_action"] = CLEARING_ACTION
    if rmin is None:
        actionable["verdict"] = "UNEVALUATED"
        actionable["reason"] = (
            f"the reference supplied no value ({ref.get('state')}), so there "
            f"is no day-specific kickoff to measure against. NOT a clear "
            f"margin.")
    return {
        "serves": day.isoformat(),
        "day_type": dtype,
        "structural": structural,
        "actionable": actionable,
        "residual": REFERENCE_RESIDUAL,
        # WHICH ONE MAY ALARM. CLR-2: an alarm fires on the quantity its
        # clearing action changes, and a manual dispatch changes only the
        # day-specific one. The structural margin is reported and
        # acknowledged; it is the duration problem and nobody clears it by
        # dispatching.
        "alarms_on": "actionable",
    }


def format_two_margins(m: dict) -> str:
    """Both lines, every run, each naming its basis and its coverage."""
    out = [f"PROSPECTIVE MARGINS for {m['serves']} ({m['day_type']}) - "
           f"TWO FACTS, NOT ONE"]
    for key in ("structural", "actionable"):
        a = m[key]
        dl = a.get("deadline_min")
        dls = (f"{dl // 60:02d}:{dl % 60:02d}" if dl is not None else "--:--")
        # ── KEYED ON THE ABSENCE OF THE VALUE, NOT ON A SPELLING. ─────────
        #
        # This read `a.get("verdict") == "UNEVALUATED"` and crashed the whole
        # audit on 2026-10-23 with `KeyError: 'binding'`, exit 2, nothing
        # audited. TWO SPELLINGS FOR ONE THIRD STATE: `prospective_margin`
        # returns `verdict="UNKNOWN"` when no series exists under the cron,
        # `two_margins` sets `"UNEVALUATED"` on the actionable half, and this
        # branch recognised only the second -- so `UNKNOWN` fell through to
        # `a['binding']`, which is exactly the key such a dict does not have.
        #
        # IT WAS TRIGGERED BY THE CRON MOVE OF 2026-10-22: moving a cron
        # empties the lag series by design, and the reset was recorded as a
        # cost WITHOUT checking every consumer of the margin dict for the
        # n=0 shape. A state recorded as expected is not a state that is
        # handled.
        #
        # So the test is "is there a binding figure", which no later spelling
        # can break.
        if a.get("binding") is None:
            out.append(f"  {a['kind']:10s} "
                       f"{a.get('verdict', 'UNEVALUATED'):11s} deadline {dls}"
                       f"  [{a.get('coverage', 'coverage unknown')}]")
            out.append(f"             {a.get('reason', 'no binding figure')}")
            continue
        out.append(
            f"  {a['kind']:10s} {a['binding']:+6.0f}m  deadline {dls}  "
            f"verdict {a['verdict']}  [{a['coverage']}]")
        out.append(f"             basis: {a['deadline_basis']}")
    r = m["residual"]
    out.append(
        f"  RESIDUAL on the ACTIONABLE margin: it is an UPPER BOUND. "
        f"p50={r['p50']}m p99={r['p99']}m max={r['max']}m, "
        f"> 60m on {r['over_60']}/{r['n_days']} days "
        f"({r['over_60'] / r['n_days']:.1%}).")
    out.append(
        f"             so clearing {MIN_ACCEPTABLE_MARGIN_MIN}m on the "
        f"actionable margin is NECESSARY, NOT SUFFICIENT - the threshold does "
        f"NOT exceed the residual. {r['provenance']}")
    ack = m["structural"]["acknowledged"]
    out.append(
        f"  STRUCTURAL is ACKNOWLEDGED since {ack['since']} at "
        f"{ack['binding']:+.1f}m against the registered "
        f"{MIN_ACCEPTABLE_MARGIN_MIN}m (NOT edited), review {ack['review']}.")
    out.append(f"             {ack['why']}")
    out.append(f"  ALARMS ON: {m['alarms_on']} (CLR-2 - the only one a "
               f"dispatch changes)")
    return "\n".join(out)
