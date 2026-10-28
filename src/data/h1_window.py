"""THE H1 COLLECTION WINDOW. One definition, imported, never restated.

FOUR SITES WERE HALF-MOVED IN ONE WEEK, each found individually:

    the reservation itself      `odds_quota.H1_RESERVATION_FROM/UNTIL`
    the snapshot consumer       `h1_collection_check._H1_WINDOW_FROM`
                                and `H1_WINDOW_TO`, re-typed as strings
    the collect loop's fixtures pinned to the October dates, so the loop
                                halted `NO_DATA` instead of
                                `RESERVATION_EXHAUSTED` — the halt was real
                                and the REASON was wrong
    `h1_analysis --since`       still defaulted to OCTOBER's credit reset, so
                                the default population would have been
                                2026-10-01..11-09: five weeks of
                                normal-pricing snapshots counted as H1
                                observations

AND A FIFTH MISSED SITE DURING COLLECTION CORRUPTS THE RESULT RATHER THAN
REDDENING A RUN. That asymmetry is why this module exists: every one of the
four above was caught by a test, and the only reason they were caught is that
they were moved while nobody was collecting. A date left behind on 2026-11-03
produces a plausible number with no error anywhere.

    NO CONSUMER COMPUTES A DATE FROM THE WINDOW.

If a consumer needs a derived value, the derivation lives HERE and is imported
too. `month_key()` off the wall clock was exactly this defect: the credit
ledger is keyed by month, two callers each derived that month independently,
and the moment the window and the clock sat in different months the stop
reported `credits=0/168` with 168 credits spent.

── THE GRAIN RULE, STATED HERE BECAUSE EVERY FIGURE BELOW FEEDS ONE ─────────

    Every H1 figure states its grain before its value.

GRAIN HAS BEEN THE DECIDING AXIS TWICE, and both times the wrong grain pointed
at the more dramatic conclusion:

  * KEY against FIXTURE. A count at (fixture, book, market, selection) grain
    overstated `n` a hundredfold — in the direction that licenses an
    analysis. The unit is the FIXTURE.
  * FIXTURE-DAY against SERIES. In-band `odds_snapshots` showed 67-103
    distinct instants per fixture-day, which read as serious contamination of
    H1's observations. At SERIES grain — the unit `qualifying_triple` actually
    sees — all 1,939 in-band series-days carry EXACTLY ONE instant and ZERO
    carry three, so an existing writer cannot manufacture a trajectory.

Neither error was arithmetic. Both were a correct count of the wrong thing.

── THIS MODULE IMPORTS NOTHING ──────────────────────────────────────────────

Only `datetime`, deliberately: `scripts/h1_collection_check.py` must be
importable as a script as well as a module, and a window definition that drags
in the database layer could not be imported from everywhere that needs it.
"""

from __future__ import annotations

from datetime import date as _date
from datetime import datetime as _datetime
from typing import Tuple

#: ── THE WINDOW. CHANGE IT HERE AND NOWHERE ELSE. ─────────────────────────
#:
#: BOTH DATES ARE DERIVED, not picked:
#:   FROM  the TheOddsAPI monthly credit reset, which is the 1st. A
#:         reservation carved out of a monthly pool must open when that pool
#:         does; opening later spends it from a pool normal pricing has
#:         already drawn on.
#:   UNTIL FROM + 8 days, EXCLUSIVE, preserving the pre-registered EIGHT-DAY
#:         length exactly. Only the position moved, so the re-run is a
#:         reschedule and not an amendment to what is collected.
FROM: _date = _date(2026, 11, 1)
UNTIL: _date = _date(2026, 11, 9)

#: The retired October window. Kept because a figure measured inside it is not
#: comparable with one measured inside this one, and the first attempt's `n=4`
#: is quoted often enough that its dates must stay readable.
RETIRED: Tuple[_date, _date] = (_date(2026, 10, 9), _date(2026, 10, 17))

#: Collection days, inclusive of FROM and exclusive of UNTIL.
DAYS: int = (UNTIL - FROM).days

#: The last day on which collection can happen.
LAST_DAY: _date = _date.fromordinal(UNTIL.toordinal() - 1)

#: The day the rate decision is taken, and WHY it is that day: it must fall
#: where a projection can still change the plan, which is the midpoint.
RATE_DECISION: _date = _date.fromordinal(FROM.toordinal() + DAYS // 2 - 1)


def contains(day: _date) -> bool:
    """Whether `day` is a collection day. Half-open, like the constants."""
    return FROM <= day < UNTIL


def iso() -> Tuple[str, str]:
    """`(from, until)` as ISO strings, for the consumers that need text.

    HERE RATHER THAN AT THE CALL SITE. `h1_collection_check` held these two as
    hand-typed string literals — the exact half-move this module exists to
    stop — and a caller writing `str(FROM)` is the same defect one line later.
    """
    return FROM.isoformat(), UNTIL.isoformat()


def month_keys() -> Tuple[_date, ...]:
    """Every credit-ledger month key the window touches, in order.

    THE DERIVATION LIVES HERE. `month_key()` is the ledger's key for a
    calendar month (its first day), and two callers each deriving it off the
    WALL CLOCK is what made the stop read `credits=0/168` with 168 credits
    spent. A window that straddles a month boundary touches two keys, and a
    consumer that assumes one would silently read the wrong row.
    """
    keys, cur = [], FROM.replace(day=1)
    while cur < UNTIL:
        keys.append(cur)
        cur = _next_month(cur)
    return tuple(keys)


def _next_month(first_of_month: _date) -> _date:
    y, m = first_of_month.year, first_of_month.month
    return _date(y + 1, 1, 1) if m == 12 else _date(y, m + 1, 1)


def straddles_a_month() -> bool:
    """Whether the window crosses a credit reset.

    It does not today — FROM is itself a reset — and a consumer that needs to
    know asks here rather than comparing `.month` itself.
    """
    return len(month_keys()) > 1


def as_datetimes() -> Tuple[_datetime, _datetime]:
    """`(from, until)` at midnight, for the observation-window queries.

    `load_observations` compares against `observed_at`, a datetime, and a
    caller doing `datetime.combine(FROM, time.min)` is one more derivation
    outside this module.
    """
    return (_datetime.combine(FROM, _datetime.min.time()),
            _datetime.combine(UNTIL, _datetime.min.time()))
