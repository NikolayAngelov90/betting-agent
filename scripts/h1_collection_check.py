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
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

# ── REGISTERED CONSTANTS. Changing one changes the experiment. ───────────────

TARGET_N = 39                 #: stop condition, upper end of the 33-39 band
CREDIT_CEILING = 200          #: the other stop condition, whichever comes first
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

def load_observations(session, since: datetime) -> Tuple[List[dict], dict]:
    """Pre-kickoff, non-phantom observations that clear the overround band.

    Returns `(observations, dropped)` — the counts of what was excluded and
    why, because a check that silently drops rows can report zero for a reason
    it never names.
    """
    from sqlalchemy import text

    rows = session.execute(text("""
        SELECT o.match_id, o.bookmaker, o.market_type, o.selection,
               o.timestamp, o.odds_value, m.match_date
        FROM odds o
        JOIN matches m ON m.id = o.match_id
        WHERE o.timestamp >= :since
          AND m.training_exclusion_reason IS NULL
          AND o.timestamp < m.match_date
    """), {"since": since}).fetchall()

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
COLLECTION_TERMINAL_STATES = ("COMPLETE", "CEILING_HIT", "NO_DATA",
                              "RESERVATION_EXHAUSTED")
COLLECTION_STATES = (COLLECTION_RUNNING,) + COLLECTION_TERMINAL_STATES


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

    ceiling_hit = credits_spent is None or credits_spent >= CREDIT_CEILING
    reservation_hit = credits_spent is not None and credits_spent >= reservation
    target_hit = n_fixtures >= TARGET_N

    if not (ceiling_hit or reservation_hit or target_hit):
        return CollectionStop(
            False, COLLECTION_RUNNING,
            f"n={n_fixtures}/{TARGET_N} credits={credits_spent}/{reservation} "
            f"reserved (registered ceiling {CREDIT_CEILING})")

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
            f"(credits={credits_spent}/{reservation} reserved)")
    if credits_spent is None:
        return CollectionStop(
            True, "CEILING_HIT",
            f"credit ledger UNREADABLE — failing closed with n={n_fixtures}")
    # THE RESERVATION BINDS BEFORE THE REGISTERED CEILING, and the two are not
    # the same event: this one says the ALLOCATION ran out, which is a budget
    # fact; CEILING_HIT says the EXPERIMENT's registered bound was reached.
    if reservation_hit and credits_spent < CREDIT_CEILING:
        return CollectionStop(
            True, "RESERVATION_EXHAUSTED",
            f"credits={credits_spent} reached the H1 RESERVATION of "
            f"{reservation} with n={n_fixtures} < TARGET_N={TARGET_N} — the "
            f"allocation is spent and H1 may NOT borrow from the global pool; "
            f"the registered ceiling {CREDIT_CEILING} was never reached")
    return CollectionStop(
        True, "CEILING_HIT",
        f"credits={credits_spent}/{CREDIT_CEILING} reached the REGISTERED "
        f"ceiling with n={n_fixtures} < TARGET_N={TARGET_N} — stopped SHORT of "
        f"the target; the budget does not return until the next monthly reset")


def collection_state(session, since: datetime) -> CollectionStop:
    """Evaluate the stop against the database. One call, so the runner does not
    re-assemble the check's four steps and drift from them."""
    observations, _dropped = load_observations(session, since)
    attach_fixture_identity(session, observations)
    _series, by_provider = count_qualifying_fixtures(observations)
    return collection_stop(
        n_fixtures=len(by_provider.get("TheOddsAPI", set())),
        credits_spent=credits_used(session),
        raw_rows=len(observations))


def credits_used(session) -> Optional[int]:
    """Credits spent this month, or None when the ledger cannot answer.

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
            {"d": month_key(), "p": PROVIDER}).fetchone()
        return int(row[0]) if row else 0
    except Exception:
        return None


# ── REPORT ───────────────────────────────────────────────────────────────────

def render(n_fixtures: int, by_provider: Dict[str, set],
           series_by_provider: Dict[str, int], dropped: dict,
           raw_rows: int, used: Optional[int],
           month_to_date_only: bool = False) -> List[str]:
    out: List[str] = []

    if raw_rows == 0:
        out.append("H1 COLLECTION DID NOT RUN")
        out.append("  no pre-kickoff observation was written in the window — "
                   "the slots made no requests, so this is NOT a null result")
    elif n_fixtures == 0:
        out.append("H1 COLLECTION PRODUCED NOTHING - the apparatus is not working")
        out.append(f"  {raw_rows} observation(s) written and ZERO separated "
                   f"trajectories: the collection is spending and not collecting")
    else:
        out.append(f"H1 COLLECTION: {n_fixtures} of {TARGET_N} qualifying trajectories")

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

    if used is None:
        out.append(f"  credits: UNKNOWN (ledger unreadable) / ceiling {CREDIT_CEILING}")
    elif month_to_date_only:
        # The ceiling counts credits spent BY THE COLLECTION. Month-to-date is
        # the same number only because collection starts on the reset day —
        # a coincidence the design leans on, so it is named rather than
        # assumed. Before the reset the two differ and the comparison is void.
        out.append(f"  credits used this month: {used} — NOT comparable to the "
                   f"{CREDIT_CEILING} ceiling: this window starts before the "
                   f"month reset, so month-to-date is not collection-to-date")
    else:
        out.append(f"  credits used this month: {used} / ceiling {CREDIT_CEILING}")
        if used >= CREDIT_CEILING:
            out.append("  !! CREDIT CEILING REACHED — the stop condition is met "
                       "regardless of n. Restore the window and interval now.")
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

    # The TRIGGER is TheOddsAPI's count: the collection is single-provider by
    # construction (142 of 142 existing separated series), and a stray
    # API-Football fixture is reported above rather than counted here.
    n = len(by_provider.get("TheOddsAPI", set()))
    for line in render(n, by_provider, series_by_provider, dropped,
                       len(observations), used,
                       month_to_date_only=since.date() < month_key()):
        print(line)
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
