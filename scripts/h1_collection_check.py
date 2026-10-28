"""H1's daily collection check — SEPARATED trajectories, never raw rows.

Stage 26, Part B, requirement 3, in runnable form. Written 2026-09-22, nine
days before collection starts on 10-01, because the first morning of a
two-day purchase is not when to discover the check was a paragraph.

    A purchase can burn its ceiling producing duplicates and report success.

THE MEASURED FAILURE THIS EXISTS FOR: fourteen fixtures appeared to carry
three pre-kickoff points and were six-minute pairs written by a single run. A
raw count of observations says "collecting"; the experiment needs trajectories,
and a trajectory needs SEPARATION.

WHAT QUALIFIES. A fixture qualifies when some three of its observations, on one
bookmaker / market / selection, satisfy BOTH:

    consecutive gaps  >= 30 minutes   (separation)
    the two intervals differ by < 2x  (comparability)

The second is the within-fixture control: H1 compares t0->t1 against t1->t2, and
if the intervals are not alike the comparison is drift against autocorrelation.
Both thresholds come from the pre-registration and are NOT tuneable here.

The triple is SEARCHED FOR rather than taken as the first three points. A
six-minute duplicate simply never gets chosen, so dense noise cannot
manufacture a trajectory and cannot destroy one either.

CONTAMINATION CONTROLS, applied per query rather than argued once — each one
killed a prior positive result in this project:

    phantom class      training_exclusion_reason IS NULL, on every query
    overround band     (1.005, 1.25) per (fixture, book, market, timestamp);
                       H2 read +0.705% and was the two-way draw-excluded trap
    provider stratum   TheOddsAPI-* and everything else counted APART, never
                       pooled: API-Football's median movement was +37.3%
                       against TheOddsAPI's -0.61%
    fixture identity   one series per FIXTURE GROUP, not per match_id; three
                       known guarantee violations came from that distinction

THREE OUTCOMES, THREE LINES — a collection that finds nothing and a collection
that never ran are different facts, and L2 shipped inert reporting
"implemented, 0 saved":

    H1 COLLECTION: n of 39 qualifying trajectories
    H1 COLLECTION PRODUCED NOTHING - the apparatus is not working
    H1 COLLECTION DID NOT RUN

READ-ONLY. It opens a session, runs SELECTs and prints. It spends no credit,
writes no row and changes no config.
"""

from __future__ import annotations

import argparse
import os
import sys

# ── THE REPO ROOT, SO A MODULE-LEVEL `src.` IMPORT IS SAFE HERE ───────────
#
# This module deliberately imported NOTHING from `src.` at module level: the
# entry-point guard runs every script with `--help`, which argparse answers
# before any work, and `python scripts/x.py` puts `scripts/` on sys.path[0]
# rather than the repo root. So a `src.` import at module level broke the
# script form, and two entry points are pinned in `SCRIPT_FORM_LATENT` for
# exactly that.
#
# THE SUCCESSOR CONTRACT WRITTEN ON 2026-10-11 SAID: "both get the same
# `sys.path` bootstrap `ci_audit.py` carries, AFTER 2026-10-17." That date has
# passed and this file needs the import, so the bootstrap arrives here first.
# `test_the_SCRIPT_form_is_either_WORKING_or_PINNED` keeps both forms honest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import h1_window as _h1_window
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# ── REGISTERED CONSTANTS. Changing one changes the experiment. ───────────────

TARGET_N = 39                 #: stop condition, upper end of the 33-39 band
CREDIT_CEILING = 200          #: the other stop condition, whichever comes first
#: ── THESE TWO MIRROR `odds_quota`'s WINDOW AND CANNOT IMPORT IT ───────────
#:
#: This module imports NOTHING from `src.` at module level, on purpose: the
#: entry-point guard runs every script with `--help`, which argparse answers
#: before any work, and a module-level `src.` import would need the sys.path
#: bootstrap to survive the script form. Every `src.` import here lives inside
#: a function for that reason.
#:
#: So the dates are RE-TYPED, which is a hand-maintained coupling and the kind
#: this project refuses to leave on trust. It is made mechanical from the
#: other side instead: `test_the_ANALYSIS_WINDOW_MATCHES_THE_RESERVATION`
#: asserts these two against `odds_quota.H1_RESERVATION_FROM/UNTIL`, so the
#: window cannot be moved in one place and not the other. Half-moving it is
#: the live failure mode: the reservation would open on 11-01 while the
#: analysis still clamped the population to the October dates, and every
#: trajectory collected would be excluded with no error anywhere.
_H1_WINDOW_FROM = _h1_window.iso()[0]   #: the window's start, for the DID_NOT_START message
#: THE WINDOW'S END, AND IT CLOSES THE POPULATION. Added 2026-10-20 after a
#: scheduled collection run fired THREE DAYS PAST the window and wrote rows.
#:
#: `load_observations` had only a lower bound, so `h1_analysis` -- whose
#: `--since` defaults to 2026-10-01 -- would have counted anything collected
#: afterwards. **n is 4 and the analysis is closed**; a trajectory collected
#: after the pre-registered window is not evidence for it, and post-window data
#: entering a closed experiment's count is how a dead result becomes a live one
#: by accident. The analysis now passes this as `until`, always.
#:
#: EXCLUSIVE, matching the pre-registration: the window is
#: 2026-11-01 -> 2026-11-09 exclusive (moved 2026-10-28 with the reservation;
#: the retired October window is kept at
#: `odds_quota.RETIRED_H1_RESERVATION_WINDOW`).
H1_WINDOW_TO = _h1_window.iso()[1]
MIN_GAP_MINUTES = 30          #: separation
MAX_INTERVAL_RATIO = 2.0      #: comparability of the two intervals
OVERROUND_LO = 1.005          #: below this a book is not pricing a real market
OVERROUND_HI = 1.25           #: above it, a trap or a stale line

#: A market is only scored when its complete set of outcomes is present at that
#: instant — an overround computed from two of three legs is not a small
#: error, it is a different number. Incomplete markets are counted as
#: NOT COMPUTABLE, which is neither in-band nor out-of-band.
COMPLETE_MARKET_SIZES = {"1X2": 3, "over_under": 2, "btts": 2}

#: THE LEGS OF ONE MARKET ARE NOT WRITTEN AT ONE TIMESTAMP, and assuming they
#: were made this check useless before it ever ran.
#:
#: Measured 2026-09-22 over 87,380 pre-kickoff rows: grouping 1X2 legs on the
#: EXACT timestamp yields three legs in ~0 cases and one leg in 32,279.
#: Grouping to the minute yields three legs in 10,735. The writer emits Home,
#: Draw and Away seconds apart, so an exact-timestamp key never assembles a
#: complete market — every instant scores NOT COMPUTABLE, the band drops
#: everything, and the check prints "PRODUCED NOTHING - the apparatus is not
#: working" on day one of collection no matter what was collected.
#:
#: A false alarm that is indistinguishable from the real one is worse than no
#: alarm, so legs within this many seconds are one instant.
#:
#: IT MUST STAY FAR BELOW THE SEPARATION FLOOR. At 120s against a 1800s floor
#: there is no width at which assembling one market could swallow two genuine
#: observations. A test pins that relationship, not just the value.
MARKET_ASSEMBLY_SECONDS = 120


def provider_of(bookmaker: str) -> str:
    """`TheOddsAPI-pinnacle` -> TheOddsAPI; `Pinnacle` -> API-Football.

    The provider is encoded in the bookmaker string and nowhere else. Named
    here once so no caller re-derives it, because merging the two strata is
    the failure H2 died of.
    """
    return "TheOddsAPI" if str(bookmaker).startswith("TheOddsAPI-") else "API-Football"


# ── THE PREDICATE, PURE AND TESTABLE WITHOUT A DATABASE ──────────────────────

def qualifying_triple(
    times: Sequence[datetime],
    min_gap_minutes: int = MIN_GAP_MINUTES,
    max_ratio: float = MAX_INTERVAL_RATIO,
) -> Optional[Tuple[datetime, datetime, datetime]]:
    """The first triple that is BOTH separated and comparable, or None.

    Exhaustive over the sorted distinct timestamps. Series are a handful of
    points, so the cost does not matter and the correctness does: a greedy
    left-to-right scan can take a point that makes the ratio fail when a later
    one would have passed.
    """
    ts = sorted(set(times))
    gap = timedelta(minutes=min_gap_minutes)
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            first = ts[j] - ts[i]
            if first < gap:
                continue
            for k in range(j + 1, len(ts)):
                second = ts[k] - ts[j]
                if second < gap:
                    continue
                lo, hi = sorted((first.total_seconds(), second.total_seconds()))
                if lo > 0 and hi / lo < max_ratio:
                    return ts[i], ts[j], ts[k]
    return None


def count_qualifying_fixtures(
    observations: Iterable[dict],
) -> Tuple[Dict[str, int], Dict[str, set]]:
    """Group to series, test each, and report FIXTURES per provider.

    `observations` carry `fixture`, `bookmaker`, `market`, `selection`, `ts`.

    THE UNIT IS THE FIXTURE, not the series and not the row. One fixture with
    four qualifying (book, market, selection) series is ONE trajectory for
    sizing — H5's sigma came back fifty-fold wrong on exactly this distinction.
    """
    series: Dict[tuple, List[datetime]] = defaultdict(list)
    for o in observations:
        key = (provider_of(o["bookmaker"]), o["fixture"],
               o["bookmaker"], o["market"], o["selection"])
        series[key].append(o["ts"])

    fixtures_by_provider: Dict[str, set] = defaultdict(set)
    series_by_provider: Dict[str, int] = defaultdict(int)
    for (provider, fixture, _bk, _mk, _sel), times in series.items():
        if qualifying_triple(times) is not None:
            fixtures_by_provider[provider].add(fixture)
            series_by_provider[provider] += 1
    return dict(series_by_provider), dict(fixtures_by_provider)


def assemble_instants(
    keys: Iterable[Tuple[int, str, str, datetime]],
    tolerance_seconds: int = MARKET_ASSEMBLY_SECONDS,
) -> Dict[Tuple[int, str, str, datetime], datetime]:
    """Map every (match, book, market, ts) onto the INSTANT it belongs to.

    Within one (match, book, market), timestamps are sorted and consecutive
    ones closer than `tolerance_seconds` join the same cluster, which is
    labelled by its earliest write.

    Chaining is intended: legs arriving 40s apart across three writes are one
    market, and at 120s against a 1800s separation floor a chain cannot reach
    far enough to swallow two genuine observations.
    """
    by_market: Dict[Tuple[int, str, str], List[datetime]] = defaultdict(list)
    for match_id, book, market, ts in keys:
        by_market[(match_id, book, market)].append(ts)

    out: Dict[Tuple[int, str, str, datetime], datetime] = {}
    tol = timedelta(seconds=tolerance_seconds)
    for (match_id, book, market), stamps in by_market.items():
        anchor = None
        previous = None
        for ts in sorted(set(stamps)):
            if anchor is None or (ts - previous) > tol:
                anchor = ts
            out[(match_id, book, market, ts)] = anchor
            previous = ts
    return out


def overround(prices: Dict[str, float], market: str) -> Optional[float]:
    """Sum of implied probabilities, or None when the market is incomplete.

    None is the third state and it is deliberate: "could not compute" is not
    "out of band", and folding them together is how a partial market gets
    scored as a trap.
    """
    need = COMPLETE_MARKET_SIZES.get(market)
    if need is None or len(prices) < need:
        return None
    try:
        total = sum(1.0 / float(v) for v in prices.values() if float(v) > 0)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return total if total > 0 else None


def in_band(value: Optional[float]) -> bool:
    return value is not None and OVERROUND_LO < value < OVERROUND_HI


# ── THE QUERY ────────────────────────────────────────────────────────────────

def load_observations(session, since: datetime,
                      until: datetime = None) -> Tuple[List[dict], dict]:
    """Pre-kickoff, non-phantom observations that clear the overround band.

    Returns `(observations, dropped)` — the counts of what was excluded and
    why, because a check that silently drops rows can report zero for a reason
    it never names.
    """
    from sqlalchemy import text

    # ── READS `odds_snapshots`, NOT `odds` (changed 2026-10-10) ─────────────
    #
    # `odds` carries a UNIQUE index on
    # (match_id, bookmaker, market_type, selection) — verified in production —
    # and is OVERWRITTEN in place. It holds at most ONE instant per series, a
    # qualifying triple needs THREE, so `n` was structurally ZERO on every
    # input, forever. Measured 2026-10-10: 10,066 pre-kickoff series in `odds`
    # since 10-09, instants-per-series histogram {1: 10066} — not one series
    # ever observed twice.
    #
    # `odds_snapshots` is append-only, has been written on every refresh since
    # 2026-08-25, and was built in Stage 18 for exactly this question. It was
    # unread for 46 days while the analysis queried the table that cannot hold
    # its input. PNC-1: a producer with no consumer looks like a working
    # feature, and a test asserting nothing consumed it pinned that in place as
    # a specification.
    #
    # The column is `observed_at`, not `timestamp`. Position 4 stays the
    # instant so every downstream index is unchanged.
    rows = session.execute(text("""
        SELECT o.match_id, o.bookmaker, o.market_type, o.selection,
               o.observed_at, o.odds_value, m.match_date
        FROM odds_snapshots o
        JOIN matches m ON m.id = o.match_id
        WHERE o.observed_at >= :since
          AND (:until IS NULL OR o.observed_at < :until)
          AND m.training_exclusion_reason IS NULL
          AND o.observed_at < m.match_date
    """), {"since": since, "until": until}).fetchall()

    # ── COERCE THE INSTANT TO A DATETIME (added 2026-10-10) ─────────────────
    #
    # Raw `text()` SQL on SQLite returns DATETIME columns as STRINGS; psycopg2
    # returns `datetime`. `qualifying_triple` subtracts two of them, so on
    # SQLite the check died with
    # `TypeError: unsupported operand type(s) for -: 'str' and 'str'` and
    # `refresh_imminent` fail-closed to UNEVALUATED — a halt for a type error,
    # reported as "could not evaluate the stop".
    #
    # IT WAS LATENT, AND HIDDEN BY THE DEFECT ABOVE. The subtraction only runs
    # on a series with THREE instants, and `odds` could never hold two, so no
    # input reached it. Repointing the read at the append-only table made
    # trajectories possible and exposed this on the same day. A crash that
    # needs correct data to fire is invisible while the data is wrong.
    def _as_dt(v):
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00").strip()
                                          ).replace(tzinfo=None)
        return v

    rows = [(r[0], r[1], r[2], r[3], _as_dt(r[4]), r[5], _as_dt(r[6]))
            for r in rows]

    # Overround is a property of a whole market at one INSTANT, and an instant
    # is a cluster of writes — not a timestamp. See MARKET_ASSEMBLY_SECONDS.
    instant_of = assemble_instants(
        [(r[0], r[1], r[2], r[4]) for r in rows])

    books: Dict[tuple, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        key = (r[0], r[1], r[2], instant_of[(r[0], r[1], r[2], r[4])])
        books[key][r[3]] = r[5]

    verdict = {k: overround(v, k[2]) for k, v in books.items()}
    dropped = {
        "not_computable": sum(1 for v in verdict.values() if v is None),
        "out_of_band": sum(1 for v in verdict.values()
                           if v is not None and not in_band(v)),
        "kept": sum(1 for v in verdict.values() if in_band(v)),
    }

    # `odds` and `match_date` ride along so the ANALYSIS reads the same rows
    # through the same controls. Two implementations of one predicate diverge;
    # this project has paid for that more than once.
    obs = [
        {"match_id": r[0], "bookmaker": r[1], "market": r[2],
         "selection": r[3], "ts": r[4], "odds": r[5], "match_date": r[6]}
        for r in rows
        if in_band(verdict[(r[0], r[1], r[2],
                            instant_of[(r[0], r[1], r[2], r[4])])])
    ]
    return obs, dropped


def attach_fixture_identity(session, observations: List[dict]) -> None:
    """Map every `match_id` onto its FIXTURE GROUP, in place.

    Falls back to the match_id when resolution is unavailable, and says so —
    silently degrading to row identity is the defect s5.9 exists for.
    """
    match_ids = {o["match_id"] for o in observations}
    groups: Dict[int, int] = {}
    if match_ids:
        try:
            from src.data.fixture_identity import resolve_fixture_groups
            groups = resolve_fixture_groups(session, match_ids)
        except Exception as exc:                       # pragma: no cover
            print(f"!! fixture-group resolution unavailable ({exc}) — falling "
                  f"back to match_id. Duplicate rows of one fixture will be "
                  f"counted as separate trajectories, which OVERSTATES n.",
                  file=sys.stderr)
    for o in observations:
        o["fixture"] = groups.get(o["match_id"], o["match_id"])


# ── REQUIREMENT 1: THE ENFORCED STOP (Stage 26, un-suspended 2026-10-01) ─────
#
# `TARGET_N` and `CREDIT_CEILING` lived HERE and nowhere else, so the check could
# report "stop condition met" while the runner kept spending. An advisory stop on
# a MONTHLY budget is not a stop: H1 is sized at 106-168 credits of a 450-credit
# month, and an unbounded loop empties it in one run with no second chance until
# 11-01.
#
# THE OUTCOMES ARE REGISTERED BEFORE THE DISCRIMINATOR IS USED, in the type that
# emits them. Three TERMINAL states, and they are not interchangeable:
#
#   COMPLETE     n >= TARGET_N. The experiment has its data. Exit clean.
#   CEILING_HIT  credits exhausted BEFORE the target. Exit and ALARM — the
#                collection stopped short and the budget will not return until
#                the next reset.
#   NO_DATA      a halt with ZERO in-band observations. The budget went and the
#                apparatus produced nothing. This REFINES a halt rather than
#                triggering one, because "spent and collected nothing" must never
#                be filed as either of the other two.
#
# THE REVERT PLAN, per the original registration, shipped in the same commit as
# the stop rather than written after it fires:
#
#   1. stop passing `h1_collection=True` to `refresh_imminent`
#   2. restore the registered normal values: `window_minutes=120`,
#      `min_interval_minutes=180`. These are `refresh_imminent`'s DEFAULTS, so
#      the revert is the removal of two arguments, not an edit to a constant —
#      which is why the H1 values are passed and never defaulted.
#   3. leave `TARGET_N`, `CREDIT_CEILING` and this module untouched: the
#      registration fixes them, and a revert that edits them is a new experiment
#      wearing the old one's name.
#
# ON EACH TERMINAL STATE:
#   COMPLETE     revert, and the analysis may run. Exit clean.
#   CEILING_HIT  revert, and the analysis may NOT run at n < TARGET_N without
#                saying so — an underpowered result reported as a result is the
#                failure H2 died of. ALARM.
#   NO_DATA      revert, and do NOT re-run collection before finding out why the
#                apparatus produced nothing. ALARM. Re-running costs the next
#                month's budget to repeat a known failure.
#
#: Non-terminal. Named so that "still collecting" is a state and not an absence.
COLLECTION_RUNNING = "COLLECTING"
#: FIVE, not four — and the fifth is a SPLIT, not an addition. `CEILING_HIT` was
#: carrying two different facts: the registered bound was reached, and the ledger
#: could not be read. The second is not a ceiling event, and folding it in meant
#: a fail-closed halt was reported as if a bound had been hit.
COLLECTION_TERMINAL_STATES = ("COMPLETE", "RESERVATION_EXHAUSTED", "CEILING_HIT",
                              "NO_DATA", "BUDGET_UNREADABLE")
COLLECTION_STATES = (COLLECTION_RUNNING,) + COLLECTION_TERMINAL_STATES


def effective_bound(reservation: int) -> int:
    """THE ONE CREDIT BOUND. `CREDIT_CEILING` caps the reservation; it is not a
    parallel limb.

    CEILINGS COLLAPSED 2026-10-02. Two independent bounds over one exhaustible
    pool are not a budget, and the weaker one was DEAD: with a 168 reservation
    against a registered 200, nothing could ever reach 200 from the runner, so
    `CEILING_HIT` was a terminal state that could not occur on a readable
    ledger.

    Now there is one number. Which of the two supplies it is reported by the
    STATE, so the registered ceiling still has a voice — and it becomes the
    binding one by a STATED INPUT: a reservation raised above it, which is a
    misconfiguration the state names rather than silently honours.
    """
    return min(int(reservation), CREDIT_CEILING)


class CollectionStop(NamedTuple):
    """The stop decision. `state` is always one of `COLLECTION_STATES`."""

    halt: bool
    state: str
    reason: str

    def __str__(self) -> str:
        return (f"H1_STOP state={self.state} halt={self.halt} "
                f"reason={self.reason}")


def collection_stop(*, n_fixtures: int, credits_spent: Optional[int],
                    raw_rows: int,
                    reservation: Optional[int] = None) -> CollectionStop:
    """THE ONLY DEFINITION of the stop. Imported by the runner, not re-typed.

    FAILS CLOSED on an unreadable ledger: `credits_spent is None` halts. A
    collection that cannot see its own spend must not keep spending — the
    alternative is `None` compared against an int, which is how a guard becomes
    a no-op.

    `reservation` is the BUDGET bound and comes from `odds_quota.allocate()` —
    the single allocator both consumers read. It is smaller than
    `CREDIT_CEILING` (168 against 200), so it binds first and
    `RESERVATION_EXHAUSTED` is the state that actually fires in practice.
    CEILING_HIT remains the REGISTERED experiment stop and is not redefined:
    the two answer different questions, and H1 cannot borrow past either.
    """
    if reservation is None:
        from src.data.odds_quota import H1_RESERVED_CREDITS
        reservation = H1_RESERVED_CREDITS

    # ── A WINDOW THAT STRADDLES A CREDIT RESET IS REFUSED. 2026-10-28. ───
    #
    # `credits_spent` comes from the ATTRIBUTED counter, which `api_budget`
    # keys by CALENDAR MONTH. A window crossing a reset touches TWO keys and
    # every reader here looks at one, so the spend would be understated by
    # whatever was spent in the other month -- and understating spend against
    # a bound means the stop does not fire. Silent, and in the direction that
    # keeps collecting.
    #
    # The window does not straddle today (FROM is itself a reset), which is
    # exactly why this is worth asserting rather than assuming: the property
    # holds by construction now and nothing would notice if it stopped.
    if _h1_window.straddles_a_month():
        return CollectionStop(
            True, "BUDGET_UNREADABLE",
            f"the collection window {_h1_window.iso()} CROSSES a credit "
            f"reset, touching ledger months "
            f"{[str(k) for k in _h1_window.month_keys()]}. The attributed "
            f"counter is keyed by one month, so the spend read against the "
            f"bound is incomplete and this stop would not fire when it "
            f"should. This is the bound being UNKNOWABLE, not reached.")

    bound = effective_bound(reservation)
    #: Which of the two supplied the bound. The registered ceiling binds only
    #: when the reservation exceeds it — a misconfiguration, named rather than
    #: silently honoured.
    bound_is_ceiling = CREDIT_CEILING < reservation
    budget_hit = credits_spent is not None and credits_spent >= bound
    target_hit = n_fixtures >= TARGET_N

    # FAIL CLOSED, AND SAY WHICH FACT IT IS. An unreadable ledger is not a
    # ceiling event; reporting it as one made a halt-for-ignorance look like a
    # halt-at-a-bound.
    if credits_spent is None:
        return CollectionStop(
            True, "BUDGET_UNREADABLE",
            f"credit ledger UNREADABLE — failing closed with n={n_fixtures}; "
            f"this is NOT a bound being reached, it is the bound being "
            f"unknown, and a collection that cannot see its spend must not "
            f"keep spending")

    if not (budget_hit or target_hit):
        return CollectionStop(
            False, COLLECTION_RUNNING,
            f"n={n_fixtures}/{TARGET_N} credits={credits_spent}/{bound} "
            f"(bound = {'registered ceiling' if bound_is_ceiling else 'reservation'}"
            f"; reservation {reservation}, registered ceiling {CREDIT_CEILING})")

    # NO_DATA refines a halt: the budget is gone and nothing was collected.
    if raw_rows == 0:
        return CollectionStop(
            True, "NO_DATA",
            f"halting with ZERO in-band observations after "
            f"credits={credits_spent}/{CREDIT_CEILING} — the collection spent "
            f"and collected nothing; the apparatus is not working")
    if target_hit:
        return CollectionStop(
            True, "COMPLETE",
            f"n={n_fixtures} >= TARGET_N={TARGET_N} "
            f"(credits={credits_spent}/{bound})")
    if bound_is_ceiling:
        return CollectionStop(
            True, "CEILING_HIT",
            f"credits={credits_spent} reached the REGISTERED ceiling "
            f"{CREDIT_CEILING} with n={n_fixtures} < TARGET_N={TARGET_N}. The "
            f"ceiling bound rather than the reservation because the reservation "
            f"({reservation}) EXCEEDS it — a misconfiguration: the registration "
            f"fixes {CREDIT_CEILING} and an allocation may not raise it")
    return CollectionStop(
        True, "RESERVATION_EXHAUSTED",
        f"credits={credits_spent} reached the H1 RESERVATION of {reservation} "
        f"with n={n_fixtures} < TARGET_N={TARGET_N} — the allocation is spent "
        f"and H1 may NOT borrow from the global pool; the registered ceiling "
        f"{CREDIT_CEILING} was never reached")


# ──────────────── THE APPARATUS STATE (Stage 27, 2026-10-02) ────────────────
#
# `render()` printed "H1 COLLECTION PRODUCED NOTHING - the apparatus is not
# working" whenever there were observations and no trajectories. On 2026-10-02
# that fired with 103 observations and 0 trajectories — and it was WRONG:
# collection had not started, the window opens 10-09, and ordinary pricing
# refreshes at most once per league per 180 min, so no series can accumulate
# three separated points. A correct zero was reported as a broken apparatus.
#
# NOT-STARTED and STARTED-AND-FAILING are the third-state family again, in the
# check's own headline. Registered here, in the type that emits them.

APPARATUS_STATES = ("DID_NOT_START", "RUNNING_AND_FAILING", "RUNNING_AND_FINE")


class ApparatusState(NamedTuple):
    state: str
    alarms: bool
    message: str

    def __str__(self) -> str:
        return f"H1_APPARATUS state={self.state} alarms={self.alarms}"


def apparatus_state(*, in_window: bool, raw_rows: int,
                    n_fixtures: int) -> ApparatusState:
    """Which of the three the collection is in. `in_window` comes from the
    reservation window — one definition, not a second calendar."""
    if not in_window:
        return ApparatusState(
            "DID_NOT_START", False,
            f"H1 COLLECTION HAS NOT STARTED — the window opens "
            f"{_H1_WINDOW_FROM}. {raw_rows} observation(s) and {n_fixtures} "
            f"trajectories is the EXPECTED state: ordinary pricing refreshes a "
            f"league at most once per 180 min, so no series accumulates three "
            f"separated points. This is not a null and not a fault.")
    if n_fixtures == 0:
        return ApparatusState(
            "RUNNING_AND_FAILING", True,
            f"H1 COLLECTION PRODUCED NOTHING — the apparatus is not working. "
            f"{raw_rows} observation(s) written and ZERO separated "
            f"trajectories: inside the window, the collection is spending and "
            f"not collecting.")
    return ApparatusState(
        "RUNNING_AND_FINE", False,
        f"H1 COLLECTION: {n_fixtures} of {TARGET_N} qualifying trajectories")


def collection_state(session, since: datetime, today=None) -> CollectionStop:
    """Evaluate the stop against the database. One call, so the runner does not
    re-assemble the check's four steps and drift from them.

    `today` is passed straight to `h1_credits_used` and defaults to the clock,
    so the runner is unchanged and a test can state its regime.
    """
    observations, _dropped = load_observations(session, since)
    attach_fixture_identity(session, observations)
    _series, by_provider = count_qualifying_fixtures(observations)
    return collection_stop(
        # H1's OWN spend, not the month's. `credits_used` is month-to-date
        # TOTAL and was being compared against the 168 reservation: on
        # 2026-10-09 that charged H1 with 38 credits spent by normal pricing
        # before H1 had run at all. See `h1_credits_used`.
        n_fixtures=len(by_provider.get("TheOddsAPI", set())),
        credits_spent=h1_credits_used(session, today),
        raw_rows=len(observations))


def h1_credits_used(session, today=None) -> Optional[int]:
    """Credits spent BY H1 this month, or None when the ledger cannot answer.

    THE ONLY NUMBER COMPARABLE TO THE RESERVATION. `credits_used` below is the
    month's total across every consumer; this reads the attributed counter that
    only a `for_h1=True` quota increments.

    `today` IS A SEAM, ADDED 2026-10-28, and the reason is the morning's
    card loss. This function read `month_key()` off the WALL CLOCK while the
    claim writes under `month_key(today)` — the claim's date. In production
    both read the same clock on the same day, so they agree and nothing was
    wrong. In a test they do not: driving a claim at 2026-11-01 from an
    October clock wrote the November row and read the October one, and the
    stop reported `credits=0/168` with 168 credits spent.

    A function that reads the clock itself and takes no `now` cannot be
    driven at a date. That is the shape that cost a card on 2026-10-27, and
    the remedy is the seam rather than a frozen fixture on the other side.

    A MISSING ROW IS ZERO, NOT UNKNOWN, and that is safe here only because
    `OddsApiQuota.claim_requests` REFUSES to spend an H1 credit it cannot
    attribute. Without that refusal this zero would be the silent kind — a
    bound compared against an unwritten counter never binds. An exception
    still returns None and the stop fails closed.
    """
    try:
        from src.data.odds_quota import PROVIDER_H1, month_key
        from sqlalchemy import text
        row = session.execute(text(
            "SELECT used FROM api_budget WHERE day = :d AND provider = :p"),
            {"d": month_key(today), "p": PROVIDER_H1}).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


def credits_used(session, today=None) -> Optional[int]:
    """Credits spent this month BY EVERY CONSUMER, or None if unreadable.

    NOT H1's spend — see `h1_credits_used`. This is the pool's total and it is
    reported, never compared against the reservation or the ceiling.

    `today` IS A SEAM for the same reason as `h1_credits_used`'s, added the
    same day: this read `month_key(today)` off the wall clock while a claim writes
    under the claim's date, so the two disagree whenever they are not the same
    month. Defaults to the clock, so no caller changes.

    `api_budget` is keyed (day, provider) where the credit ledger stores the
    MONTH in `day` — see `odds_quota.month_key`. The column is `used`, not
    `credits_used`; getting that wrong printed UNKNOWN against a ledger that
    was perfectly readable, which is a quieter version of the same failure
    this whole check is about.
    """
    try:
        from src.data.odds_quota import PROVIDER, month_key
        from sqlalchemy import text
        row = session.execute(text(
            "SELECT used FROM api_budget WHERE day = :d AND provider = :p"),
            {"d": month_key(today), "p": PROVIDER}).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


# ── REPORT ───────────────────────────────────────────────────────────────────

def render(n_fixtures: int, by_provider: Dict[str, set],
           series_by_provider: Dict[str, int], dropped: dict,
           raw_rows: int, used: Optional[int],
           h1_used: Optional[int] = None,
           month_to_date_only: bool = False,
           in_window: Optional[bool] = None) -> List[str]:
    out: List[str] = []

    from src.data.odds_quota import h1_reservation

    if in_window is None:
        in_window = h1_reservation() > 0

    if raw_rows == 0:
        out.append("H1 COLLECTION DID NOT RUN")
        out.append("  no pre-kickoff observation was written in the window — "
                   "the slots made no requests, so this is NOT a null result")
    else:
        _app = apparatus_state(in_window=in_window, raw_rows=raw_rows,
                               n_fixtures=n_fixtures)
        out.append(str(_app))
        out.append("  " + _app.message)

    out.append("")
    out.append(f"  raw pre-kickoff observations (in band) : {raw_rows}")
    out.append(f"  market-instants kept / out-of-band / not computable : "
               f"{dropped.get('kept', 0)} / {dropped.get('out_of_band', 0)} / "
               f"{dropped.get('not_computable', 0)}")

    # STRATIFIED, NEVER POOLED.
    out.append("  qualifying FIXTURES by provider (never summed):")
    if by_provider:
        for provider in sorted(by_provider):
            out.append(f"     {provider:14s} fixtures={len(by_provider[provider]):4d} "
                       f"series={series_by_provider.get(provider, 0)}")
    else:
        out.append("     (none)")
    if len(by_provider) > 1:
        out.append("  !! MORE THAN ONE PROVIDER QUALIFIED. The registration "
                   "analyses them as separate strata or drops the second — "
                   "they are different instruments, not different samples.")

    # TWO NUMBERS, NEVER ONE. The pool's total and H1's own spend were the
    # same variable until 2026-10-09, and the comparison that mattered used the
    # wrong one. The caveat that used to live here ("month-to-date is not
    # collection-to-date") was CORRECT and was ALSO DISARMED: it was gated on
    # `since < month_key()`, and with --since defaulting to the 10-01 reset the
    # flag read False — precisely while the collection window started 10-09 with
    # 38 credits already spent. A caveat that knows the right thing and cannot
    # fire is not a safeguard. It is replaced by measuring the quantity.
    if used is None:
        out.append("  pool credits used this month: UNKNOWN (ledger unreadable)")
    else:
        out.append(f"  pool credits used this month: {used} "
                   f"(EVERY consumer — not H1's spend, not comparable to the "
                   f"reservation or the ceiling)")

    if h1_used is None:
        out.append(f"  H1 ATTRIBUTED credits: UNKNOWN (ledger unreadable) — the "
                   f"stop FAILS CLOSED on this, see BUDGET_UNREADABLE")
    else:
        _res = h1_reservation()
        _bound = effective_bound(_res) if _res else CREDIT_CEILING
        out.append(f"  H1 ATTRIBUTED credits: {h1_used} / bound {_bound} "
                   f"(reservation {_res}, registered ceiling {CREDIT_CEILING})")
        if h1_used >= _bound:
            out.append("  !! H1 CREDIT BOUND REACHED — the stop condition is "
                       "met regardless of n. Restore the window and interval "
                       "now.")
    if n_fixtures >= TARGET_N:
        out.append(f"  !! TARGET REACHED ({n_fixtures} >= {TARGET_N}) — stop "
                   f"condition met. Restore the window and interval now.")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="ISO date; default 2026-10-01, the reset")
    ap.add_argument("--days", type=int,
                    help="look back this many days instead of --since")
    ap.add_argument("--allow-sqlite", action="store_true",
                    help="answer from the local SQLite file (NOT production)")
    args = ap.parse_args(argv)

    # `.env` is read HERE, not at import. A script that calls load_dotenv() at
    # import undoes conftest's DATABASE_URL strip and lets the test suite reach
    # production — that has happened in this repo, so the loader stays inside
    # main() where the tests never go.
    try:
        from dotenv import load_dotenv
        import pathlib
        env = pathlib.Path(__file__).resolve().parent.parent / ".env"
        if env.is_file():
            load_dotenv(env)
    except Exception:
        pass

    if args.days is not None:
        since = datetime.utcnow() - timedelta(days=args.days)
    elif args.since:
        since = datetime.fromisoformat(args.since)
    else:
        since = datetime(2026, 10, 1)

    from src.data.database import get_db
    from src.data.odds_quota import month_key
    db = get_db()

    # ANSWERED FROM THE WRONG DATABASE MUST NOT LOOK LIKE ANSWERED. Without
    # DATABASE_URL, `get_db()` falls back to a local SQLite file that is months
    # stale and lacks columns this query needs. A fallback that substitutes
    # silently turns "could not look" into a number — the collapse this project
    # has closed four times. So it refuses instead.
    if not db.is_postgres and not args.allow_sqlite:
        print("REFUSING: connected to local SQLite, not the production "
              "database. DATABASE_URL is unset and no .env supplied it, so "
              "any count printed here would describe a stale local file.\n"
              "  Set DATABASE_URL, or pass --allow-sqlite to accept a "
              "local-only answer deliberately.", file=sys.stderr)
        return 2

    with db.get_session() as session:
        observations, dropped = load_observations(session, since)
        attach_fixture_identity(session, observations)
        series_by_provider, by_provider = count_qualifying_fixtures(observations)
        used = credits_used(session)
        h1_used = h1_credits_used(session)

    # The TRIGGER is TheOddsAPI's count: the collection is single-provider by
    # construction (142 of 142 existing separated series), and a stray
    # API-Football fixture is reported above rather than counted here.
    n = len(by_provider.get("TheOddsAPI", set()))
    for line in render(n, by_provider, series_by_provider, dropped,
                       len(observations), used, h1_used,
                       month_to_date_only=since.date() < month_key()):
        print(line)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
