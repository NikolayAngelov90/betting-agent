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
from typing import Optional, Tuple

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

#: ── THE STOP IS RECORDED, AND IT IS *NOT* A FUNDING SWITCH. 2026-11-06. ──
#:
#: `H1 WINDOW FAILED` on 2026-11-05 (day-5 rule, n=12 FIXTURE-grain against
#: 18, NOT extended). A `STOPPED_ON` date plus an `is_collecting()` predicate
#: was built here and then REVERTED the same day. The reasoning is kept
#: because the next person will have the same idea.
#:
#: WHAT IT WOULD HAVE BOUGHT: `h1_reservation()` returning 0 from 11-05
#: instead of 11-09, i.e. 168 credits back to normal pricing for THREE DAYS.
#:
#: WHAT IT COST, measured: 13 tests in the CAPTURE GATE flipped --
#: `test_two_month_counter`, `test_h1_reservation_is_an_allocation`,
#: `test_h1_enforced_stop` -- because they call quota methods with no
#: explicit `today=`. Those are the LATENT MEMBERS that
#: `test_calendar_separation` already counts and warns about. Worse,
#: `reservation_is_expressible()` refused the window as
#: `BUDGET_UNREADABLE`: it asks whether a 168-credit reservation can be
#: CARVED on every day of the window, which is a DESIGN property, and an
#: unfunded day is indistinguishable from a mis-declared one. Reporting a
#: stopped experiment as an unreadable budget is a third-state collapse --
#: the exact defect class this project keeps paying for.
#:
#: WHY IT IS UNNECESSARY. The reservation is calendar-driven BY DESIGN and
#: goes inert on UNTIL with no code change: for any day >= UNTIL,
#: `h1_reservation()` is 0 and `allocate()["h1_ceiling"]` is 0. The successor
#: contract is therefore a PROPERTY OF THE DEPLOYED DESIGN and not something
#: that had to be built -- it simply arrives on 2026-11-09.
#:
#: THE ACCEPTED COST, stated rather than hidden: 168 credits stay reserved
#: for 2026-11-05..11-08 for an experiment that has stopped. Nothing is
#: collecting, so nothing spends them; the only effect is that normal
#: pricing's ceiling is 182 rather than 350 on those four days, against a
#: measured normal usage that has never approached either. Four days of
#: unused headroom is a smaller cost than a funding switch inside a
#: capture-path module.
STOPPED_ON: Optional[_date] = _date(2026, 11, 5)   #: RECORDED, read by nothing

#: ═════════ H1 IS DORMANT. A DECISION, RECORDED — 2026-11-11 ═════════════
#:
#: The expiry guard fired on 2026-11-09 and offered two choices: start a new
#: window, or record dormancy and move the review to what is intended.
#: **DORMANCY IS CHOSEN**, authorised 2026-11-11, and the reasons are recorded
#: here so the guard's next firing has something to compare against.
#:
#: Deletion remains excluded: 274 references / 61 assertions, 51 of which
#: encode live facts a future attempt needs. `STOPPED_ON` stays.
#:
#: WHY NOT NOW — three reasons, each measured rather than asserted:
#:
#:  1. THE PICK POPULATION JUST CHANGED TWICE IN FOUR DAYS. `min_odds` went
#:     1.50 -> 1.14 on 11-07 (cohort s5.20) and back on 11-11 (s5.21).
#:     `s5.20` is 86 picks over two days at an average taken price of 1.464
#:     against 1.59..1.67 for every cohort before it. Running H1 across a
#:     composition that has just moved and moved back measures a moving
#:     target, and the three cohorts may not be pooled.
#:
#:  2. THE REGISTERED LIMIT IS MARKET COVERAGE AND IT NEEDS RE-MEASURING.
#:     `docs/h1-design-finding.md` sizes the reachable ceiling at 57.1% --
#:     42.9% of picks sit in markets whose trajectories cannot be collected.
#:     The band change altered which markets picks fall in, so that figure is
#:     stale until it is re-measured on the `s5.21` composition. Sizing a
#:     window against a stale ceiling is how the first attempt overdrew 288
#:     credits against a 168 reservation.
#:
#:  3. AND THE QUESTION H1 WAS FOR HAS PARTLY ANSWERED ITSELF ELSEWHERE.
#:     The leg-level band test (2026-11-11, n=52,737 legs over ~10,700
#:     fixtures) finds the favourite-longshot bias PRESENT in four
#:     consecutive bands -- +1.14 / +1.64 / +1.43 / +2.38 pp, every CI above
#:     zero -- and NOT COLLECTABLE at any venue whose price is established
#:     takeable: the net is negative at every bookmaker in every band, on the
#:     CI lower bound AND on the point estimate. The one surviving cell is
#:     the 80-90% band at an EXCHANGE, +0.48pp on the lower bound before
#:     commission, needing a rate below 3.6%, on 6.1% fixture coverage, with
#:     DEPTH UNAVAILABLE. **What blocks acting is venue depth, not price
#:     history** -- so a price-path collection is not the binding measurement
#:     today.
#:
#: THE REVIEW IS A MECHANISM, NOT A DEADLINE. Two things must report before a
#: window is sized, and the date below is an ANCHOR for the first opportunity
#: rather than a promise:
#:
#:     (a) the market-coverage ceiling re-measured on the `s5.21` pick
#:         composition, replacing the 57.1% / 42.9% figures; and
#:     (b) a venue with MEASURABLE DEPTH, since that is what blocks acting on
#:         a gap the band test has already established.
#:
#: The anchor is the next UEFA matchday cluster, **MD5 on 24-25 November
#: 2026** -- an external fixture fact, not a chosen deadline. A window can be
#: anchored to it at no cost today, and anchoring to a real cluster is what
#: the first attempt's sizing lacked.
DORMANCY = {
    "decided": "2026-11-11",
    "decision": "DORMANT",
    "authorised_by": "operator, 2026-11-11, in response to the 11-09 expiry",
    "deletion": "EXCLUDED - 274 refs / 61 assertions, 51 protect a retry",
    "stopped_on_stays": True,
    "reasons": (
        "the pick population changed twice in four days (min_odds 1.50 -> "
        "1.14 -> 1.50, cohorts s5.20 then s5.21; s5.20 is 86 picks over two "
        "days at avg 1.464) and the three cohorts may not be pooled",
        "the registered limit is MARKET COVERAGE (57.1% reachable / 42.9% "
        "not) and the band change altered which markets picks fall in, so "
        "the ceiling is stale until re-measured on s5.21",
        "the leg-level band test already establishes the anomaly (four bands "
        "PRESENT, +1.14..+2.38pp) and establishes it is NOT collectable at "
        "any venue with an established price; what blocks acting is VENUE "
        "DEPTH, not price history, so a price-path collection is not the "
        "binding measurement today",
    ),
    "review_mechanisms": (
        "the market-coverage ceiling re-measured on the s5.21 composition",
        "a venue with MEASURABLE DEPTH becoming available",
    ),
    "review_anchor": _date(2026, 11, 24),
    "review_anchor_basis": (
        "the next UEFA matchday cluster, MD5 on 24-25 November 2026 -- an "
        "external fixture fact, not a chosen deadline. It is the first "
        "opportunity a window could be anchored to, and anchoring to a real "
        "cluster is what the first attempt's sizing lacked"),
    "provenance": "recorded 2026-11-11; the band figures are measured, "
                  "grain=LEG, n=52,737",
}


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


def days_list() -> Tuple[_date, ...]:
    """Every DAY of the window, in order. The derivation lives here too.

    A consumer that needs to ask a per-day question -- "is a reservation
    declared for this day?" -- was otherwise building the range itself from
    FROM and UNTIL, which is the computing-a-date-from-the-window defect the
    one-source rule exists to stop.
    """
    return tuple(_date.fromordinal(o)
                 for o in range(FROM.toordinal(), UNTIL.toordinal()))


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
