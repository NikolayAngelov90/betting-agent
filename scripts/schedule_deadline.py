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
OBSERVED_EARLIEST_MIN = {
    "weekday": 9 * 60 + 59,     # 2026-06-18 Thu, finland/veikkausliiga, 1 fx
    "weekend": 10 * 60,         # 2026-02-01..2026-10-10
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
                       min_margin: int = MIN_ACCEPTABLE_MARGIN_MIN) -> dict:
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
        "deadline_basis": "discovered card",
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
    return (head + f" deadline={hm(m['deadline_min'])} (discovered card) "
            f"ready max={hm(m['ready_max'])} 3sd={hm(m['ready_3sd'])} "
            f"| margin max={m['margin_max']:+.0f}m 3sd={m['margin_3sd']:+.0f}m "
            f"binding={m['binding']:+.0f}m min={m['min_margin']}m n={m['n']}"
            + ("  ** ACTION REQUIRED **]" if m["verdict"] == "ALARM" else "]"))
