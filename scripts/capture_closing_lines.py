"""Capture the closing line for pending picks, so CLV becomes computable.

    python -m scripts.capture_closing_lines [--within-minutes 90] [--dry-run]
                                            [--refresh-odds] [--stats]

Stage 5, Phases 5-7 and 10. Run shortly before kickoff; a scheduled entry
60-90 minutes out is the practical compromise on free data.

Design rules, each of which exists because of something that went wrong
----------------------------------------------------------------------

**Never invent a price.** A pick that could not be priced is recorded as
``closing_capture_status = 'missing'``, not left ambiguously NULL and not filled
with the nearest available number. A CLV series built from invented closes would
be worse than no series.

**Never mix markets.** Selection-to-market resolution goes through
``src.data.market_spec``, the same declaration the scraper and feature builder
use. Matching a Home/Away price to a 1X2 pick is the exact bug that corrupted
2,548 matches, and it is now impossible by construction rather than by care.

**Validate per (match, bookmaker, market).** Seven bookmakers were corrupt on
2026-02-01..08-05, Pinnacle among them at 26%. Dropping whole bookmakers would
discard the majority of Pinnacle's good data, so each book's market is gated on
its own overround.

**Late is not closing.** A capture at or after kickoff is marked ``'late'`` and
excluded from CLV — it may contain in-play information. A capture far *before*
kickoff is not a close either; ``clv.validate_pair`` enforces that at read time.

**Idempotent.** Only picks with ``closing_capture_status = 'pending'`` are
considered, so a second run in the same window is a no-op and a captured price
is never overwritten.

**Cheap.** Two reads per run regardless of fixture count: one for the pending
picks in the window, one for their odds rows, both column-projected. Picks whose
kickoff has already passed are separated out *before* the odds read, so a stale
backlog costs one status UPDATE and no odds egress. Write-back is one UPDATE per
distinct status plus one per genuinely captured price — the captured set is
bounded by the capture window, not by history. No ``SELECT *``. ``--stats``
prints the measured cost.
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

from src.data.database import get_db
from src.data.market_spec import check_overround, devig, extract_legs
from src.data.models import Match, Odds, SavedPick
from src.utils.logger import get_logger, utcnow

logger = get_logger()

# Capture outcomes written to SavedPick.closing_capture_status.
STATUS_PENDING = "pending"
STATUS_CAPTURED = "captured"
STATUS_MISSING = "missing"    # no usable price existed
STATUS_LATE = "late"          # kickoff already passed
STATUS_INVALID = "invalid"    # selection has no market mapping

#: Bookmakers whose prices are display composites, not bettable quotes.
EXCLUDED_BOOKMAKERS = frozenset({"Flashscore"})

#: SavedPick.selection -> (market_type, line, side, leg index within the spec).
#: The leg index is what makes "which of this market's outcomes is our pick?"
#: explicit, instead of re-deriving it from string prefixes at each call site.
SELECTION_SPEC: Dict[str, Tuple[str, Optional[str], Optional[str], int]] = {
    "Home Win":         ("1X2", None, None, 0),
    "Draw":             ("1X2", None, None, 1),
    "Away Win":         ("1X2", None, None, 2),
    "Over 1.5 Goals":   ("over_under", "1.5", None, 0),
    "Under 1.5 Goals":  ("over_under", "1.5", None, 1),
    "Over 2.5 Goals":   ("over_under", "2.5", None, 0),
    "Under 2.5 Goals":  ("over_under", "2.5", None, 1),
    "Over 3.5 Goals":   ("over_under", "3.5", None, 0),
    "Under 3.5 Goals":  ("over_under", "3.5", None, 1),
    "Over 4.5 Goals":   ("over_under", "4.5", None, 0),
    "Under 4.5 Goals":  ("over_under", "4.5", None, 1),
    "BTTS Yes":         ("btts", None, None, 0),
    "BTTS No":          ("btts", None, None, 1),
    "Home Over 0.5":    ("team_goals", "0.5", "Home", 0),
    "Away Over 0.5":    ("team_goals", "0.5", "Away", 0),
    "Home Over 1.5":    ("team_goals", "1.5", "Home", 0),
    "Away Over 1.5":    ("team_goals", "1.5", "Away", 0),
    "DNB Home":         ("draw_no_bet", None, None, 0),
    "DNB Away":         ("draw_no_bet", None, None, 1),
    # Double chance outcomes OVERLAP, so market_spec refuses to de-vig them and
    # closing_fair_probability stays NULL for these. The raw price comparison —
    # which is what price_clv uses — is unaffected.
    "Double Chance 1X": ("double_chance", None, None, 0),
    "Double Chance 12": ("double_chance", None, None, 1),
    "Double Chance X2": ("double_chance", None, None, 2),
}

#: Every selection the value calculator can produce must be mappable here, or a
#: pick silently becomes uncapturable. Guarded by
#: tests/test_closing_capture.py::test_every_tradeable_selection_is_mappable.



def _chunks(items: List[int], size: int):
    """Split ids into batches so an IN (...) list never grows unbounded."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def _median(values: List[float]) -> Optional[float]:
    s = sorted(values)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def consensus_close(odds_rows, market_type: str, line: Optional[str],
                    side: Optional[str], leg: int,
                    not_before=None) -> Tuple[Optional[float], Optional[float],
                                              int, Optional[object]]:
    """(median price for our leg, median de-vigged probability, books used,
    observed_at).

    Each bookmaker's market is validated on its own before it contributes, so
    one broken book cannot define the close.

    ``not_before`` is EXCLUSIVE: a row must be observed strictly after it. A
    price the odds table has been holding since this morning is not a closing
    price, and the caller cannot tell the difference from the value alone — the
    row's own ``timestamp`` is the only evidence of when it was true.

    Exclusive rather than inclusive because the caller passes the pick's own
    creation time here (Stage 8, Phase 8). The row the pick was priced from
    carries exactly that timestamp, so an inclusive bound would hand it straight
    back as the "closing" price and CLV would read 0.00%.

    ``observed_at`` is the OLDEST contributing row's timestamp, i.e. the most
    conservative claim we can make about how fresh the consensus is. It becomes
    ``closing_odds_captured_at`` so that ``clv.validate_pair``'s lead check
    measures the price's age rather than the script's run time.
    """
    by_book: Dict[str, Dict[str, float]] = defaultdict(dict)
    seen_at: Dict[str, object] = {}
    for r in odds_rows:
        if r.bookmaker in EXCLUDED_BOOKMAKERS or r.market_type != market_type:
            continue
        ts = getattr(r, "timestamp", None)
        if not_before is not None and ts is not None and ts <= not_before:
            continue
        if r.odds_value and r.odds_value > 1.0:
            by_book[r.bookmaker][r.selection] = r.odds_value
            prev = seen_at.get(r.bookmaker)
            if ts is not None and (prev is None or ts < prev):
                seen_at[r.bookmaker] = ts

    prices, fairs, stamps = [], [], []
    for book, sels in by_book.items():
        legs = extract_legs(market_type, sels, line=line, side=side)
        if legs is None or leg >= len(legs):
            continue
        ok, _ = check_overround(market_type, legs)
        if not ok:
            continue
        probs = devig(market_type, legs)
        prices.append(legs[leg])
        if seen_at.get(book) is not None:
            stamps.append(seen_at[book])
        if probs is not None:
            fairs.append(probs[leg])

    if not prices:
        return None, None, 0, None
    return (_median(prices), (_median(fairs) if fairs else None), len(prices),
            min(stamps) if stamps else None)


def resolve_close(odds_for_match, selection: str, not_before):
    """Apply every closing rule to one selection. The single resolver.

    Returns ``(status, price, fair_prob, n_books, observed_at)``.

    Stage 10 gave the pipeline a second attribution series, and both series must
    be judged by identical rules — a MODEL observation held to a looser standard
    than a FINAL one would make the two CLV numbers incomparable, which is the
    only thing the paired comparison measures. So both paths call this, rather
    than each re-implementing the checks.
    """
    spec = SELECTION_SPEC.get(selection)
    if spec is None:
        return STATUS_INVALID, None, None, 0, None

    market_type, line, side, leg = spec
    price, fair, n_books, observed_at = consensus_close(
        odds_for_match, market_type, line, side, leg, not_before=not_before)
    if price is None:
        return STATUS_MISSING, None, None, 0, None
    return STATUS_CAPTURED, price, fair, n_books, observed_at


def _load_observations(session, pick_ids, stats: dict) -> list:
    """Pending MODEL/FINAL observation rows for the picks in this window.

    Loaded BEFORE the odds query so their markets can widen `needed_markets`:
    after a Claude CHANGE the model observation sits in a different market from
    the pick, and querying only the pick's market would leave it permanently
    `missing` for want of rows nobody asked for.

    Returns [] when migration 006 is not applied — the FINAL series keeps
    working off `saved_picks.closing_*` exactly as before.
    """
    from src.data.models import PickObservation

    if not pick_ids:
        return []
    try:
        obs = session.query(PickObservation).filter(
            PickObservation.pick_id.in_(pick_ids),
            PickObservation.closing_odds.is_(None),
            PickObservation.closing_status == STATUS_PENDING,
        ).all()
        stats["db_queries"] += 1
        return obs
    except Exception as e:
        logger.debug(f"pick_observations unavailable ({e}) — skipping the "
                     f"dual-attribution pass. Is migration 006 applied?")
        return []


def _capture_observations(session, obs, live_rows, by_match, max_lead, now,
                          dry_run: bool, stats: dict) -> None:
    """Resolve closing prices for the MODEL and FINAL attribution rows.

    Stage 10, sections 10, 11 and 16.

    Reads the odds snapshot the caller already loaded — no extra odds query, and
    no extra Odds API request, because capture never calls the API at all. Where
    MODEL and FINAL name the same (market, selection) the close is resolved
    ONCE and written to both rows: one underlying observation, two attributions.
    """
    if not obs:
        return

    by_pick = {r.id: r for r in live_rows}
    # (pick_id, market, selection) -> resolved outcome. MODEL and FINAL on an
    # unchanged pick share a key and therefore share one resolution.
    resolved: Dict[tuple, tuple] = {}

    for o in obs:
        row = by_pick.get(o.pick_id)
        if row is None:
            continue

        # Same causal boundary as the FINAL series: the observation's own
        # taken_at, which _update_final_observation deliberately does not move.
        not_before = (row.match_date - max_lead) if row.match_date else None
        if o.taken_at is not None:
            not_before = (max(not_before, o.taken_at) if not_before
                          else o.taken_at)

        key = (o.pick_id, o.market, o.selection)
        if key not in resolved:
            resolved[key] = resolve_close(
                by_match.get(row.match_id, []), o.selection, not_before)
            stats["observations_resolved"] += 1

        status, price, fair, n_books, observed_at = resolved[key]
        stats[f"obs_{o.attribution}_{status}"] = (
            stats.get(f"obs_{o.attribution}_{status}", 0) + 1)

        if dry_run:
            continue
        o.closing_status = status
        if status == STATUS_CAPTURED:
            o.closing_odds = float(price)
            o.closing_fair_prob = float(fair) if fair else None
            o.closing_book_count = int(n_books)
            o.closing_captured_at = observed_at or now

    stats["observations_considered"] = len(obs)
    if not dry_run:
        session.flush()


def capture(within_minutes: int = 90, dry_run: bool = False,
            max_retries: int = 3,
            max_lead_minutes: Optional[int] = None) -> dict:
    """Capture closing prices for pending picks kicking off inside the window.

    ``max_lead_minutes`` is how old a price may be and still count as closing.
    It defaults to the same value ``clv.validate_pair`` enforces at read time,
    so capture and validation cannot drift apart: anything this function is
    willing to store is something the CLV layer is willing to use.
    """
    from src.evaluation.clv import DEFAULT_MAX_CAPTURE_LEAD

    max_lead = (timedelta(minutes=max_lead_minutes)
                if max_lead_minutes is not None else DEFAULT_MAX_CAPTURE_LEAD)
    db = get_db()
    now = utcnow()
    horizon = now + timedelta(minutes=within_minutes)
    stats = {
        "considered": 0, "captured": 0, "missing": 0, "late": 0, "invalid": 0,
        "db_queries": 0, "odds_rows_read": 0, "elapsed_s": 0.0,
        # Stage 10 dual attribution. `observations_resolved` counts DISTINCT
        # (pick, market, selection) resolutions — an unchanged pick contributes
        # 1, not 2, which is how "one underlying observation" is verifiable
        # rather than merely asserted.
        "observations_considered": 0, "observations_resolved": 0,
    }
    started = time.monotonic()

    for attempt in range(1, max_retries + 1):
        try:
            with db.get_session() as session:
                # Query 1 — pending picks in the window. Column-projected and
                # joined to matches so kickoff comes back in the same round trip.
                rows = session.query(
                    SavedPick.id, SavedPick.match_id, SavedPick.selection,
                    SavedPick.market, SavedPick.odds, SavedPick.match_name,
                    SavedPick.created_at, Match.match_date,
                ).join(Match, Match.id == SavedPick.match_id).filter(
                    SavedPick.closing_odds.is_(None),
                    SavedPick.closing_capture_status == STATUS_PENDING,
                    Match.match_date <= horizon,
                ).all()
                stats["db_queries"] += 1

                if not rows:
                    logger.info(
                        f"capture_closing_lines: no pending picks kick off in "
                        f"the next {within_minutes} minutes")
                    stats["elapsed_s"] = round(time.monotonic() - started, 2)
                    return stats

                stats["considered"] = len(rows)

                # Kickoff already passed → there is no closing price to look up,
                # only a status to record. Split those off BEFORE the odds query
                # instead of inside the capture loop: their match_ids would
                # otherwise widen Query 2 and ship rows straight to /dev/null.
                # This is not hypothetical — migration 003 backfilled all 1,070
                # historical picks to closing_capture_status='pending', and the
                # window filter is `match_date <= now + 90min` with no lower
                # bound, so the first production run sweeps every one of them.
                live_rows, late_rows = [], []
                for r in rows:
                    is_late = r.match_date is not None and now >= r.match_date
                    (late_rows if is_late else live_rows).append(r)
                stats["late"] = len(late_rows)

                updates: List[dict] = [
                    {"id": r.id, "status": STATUS_LATE} for r in late_rows
                ]

                match_ids = {r.match_id for r in live_rows}

                # Stage 10 — load the attribution rows before the odds query.
                # A Claude CHANGE leaves the MODEL observation in a DIFFERENT
                # market from the pick, so its market has to widen the filter
                # below or it stays 'missing' forever for want of rows nobody
                # asked for.
                observations = _load_observations(
                    session, [r.id for r in live_rows], stats)

                # Only the market types the pending picks actually need. Without
                # this the query returns every market on every match — measured
                # at 104,117 rows on a full-history run, against ~7,600 for the
                # markets in use. The window keeps production small either way,
                # but there is no reason to ship rows nothing will read.
                needed_markets = {
                    SELECTION_SPEC[r.selection][0]
                    for r in live_rows if r.selection in SELECTION_SPEC
                }
                needed_markets |= {
                    SELECTION_SPEC[o.selection][0]
                    for o in observations if o.selection in SELECTION_SPEC
                }
                if not needed_markets:
                    needed_markets = {"1X2"}   # nothing mappable; keep the query valid

                # Query 2 — the relevant odds rows for those matches, once.
                odds_rows = []
                if match_ids:
                    odds_rows = session.query(
                        Odds.match_id, Odds.bookmaker, Odds.market_type,
                        Odds.selection, Odds.odds_value, Odds.timestamp,
                    ).filter(
                        Odds.match_id.in_(match_ids),
                        Odds.market_type.in_(needed_markets),
                        Odds.bookmaker.notin_(tuple(EXCLUDED_BOOKMAKERS)),
                    ).all()
                    stats["db_queries"] += 1
                stats["odds_rows_read"] = len(odds_rows)

                by_match = defaultdict(list)
                for o in odds_rows:
                    by_match[o.match_id].append(o)

                for r in live_rows:
                    spec = SELECTION_SPEC.get(r.selection)
                    if spec is None:
                        logger.warning(
                            f"capture_closing_lines: no market mapping for "
                            f"selection {r.selection!r} (pick {r.id}) — marking "
                            f"invalid. Add it to SELECTION_SPEC.")
                        updates.append({"id": r.id, "status": STATUS_INVALID})
                        stats["invalid"] += 1
                        continue

                    market_type, line, side, leg = spec
                    # Only prices observed inside the closing window count. The
                    # odds table holds whatever was last written for a match,
                    # and for markets the pre-kickoff refresh does not cover
                    # (BTTS, team goals, double chance — 36% of recent picks)
                    # that is the SAME row the pick was priced from. Accepting
                    # it would manufacture CLV of exactly 0.00% and read as
                    # "we get closing-line parity".
                    not_before = (r.match_date - max_lead) if r.match_date else None

                    # Stage 8, Phase 8 — the SAME-SNAPSHOT rule.
                    #
                    # The window rule above is about time; this one is about
                    # identity. A pick taken 90 minutes before kickoff is priced
                    # from an odds row that is itself inside the closing window,
                    # so the window alone happily hands that very row back as
                    # the "closing" price. CLV would then be taken/closing - 1 =
                    # exactly 0.00% — not a measurement, an echo.
                    #
                    # A closing observation must be an observation the market
                    # made AFTER we took our price. `Odds.timestamp` is refreshed
                    # on every upsert (ON CONFLICT ... SET timestamp), so a book
                    # re-quoted at the same number still counts — that is a
                    # genuine unchanged close. Only a row nobody looked at again
                    # is excluded.
                    if r.created_at is not None:
                        not_before = (max(not_before, r.created_at)
                                      if not_before else r.created_at)

                    price, fair, n_books, observed_at = consensus_close(
                        by_match.get(r.match_id, []), market_type, line, side, leg,
                        not_before=not_before)

                    if price is None:
                        updates.append({"id": r.id, "status": STATUS_MISSING})
                        stats["missing"] += 1
                        continue

                    clv = (r.odds / price - 1) if r.odds else None
                    lead = (
                        (r.match_date - observed_at).total_seconds() / 60.0
                        if (observed_at is not None and r.match_date is not None)
                        else None
                    )
                    logger.info(
                        f"closing line: {r.match_name} {r.selection} "
                        f"taken @ {r.odds} closing @ {price:.2f} "
                        f"({n_books} books"
                        + (f", observed {lead:.0f}min pre-KO" if lead is not None
                           else ", observation time unknown")
                        + ")"
                        + (f" CLV {clv:+.2%}" if clv is not None else ""))
                    updates.append({
                        "id": r.id, "status": STATUS_CAPTURED,
                        "closing_odds": float(price),
                        "closing_fair_probability": float(fair) if fair else None,
                        "closing_bookmaker_count": int(n_books),
                        # When the PRICE was true, not when this script ran.
                        # Stamping `now` made validate_pair's lead check
                        # vacuous: the script only runs inside the window, so
                        # every capture passed no matter how old the row was.
                        "captured_at": observed_at or now,
                    })
                    stats["captured"] += 1

                # Stage 10 — the dual-attribution pass, over the same odds
                # snapshot. Runs whether or not the FINAL loop found anything.
                _capture_observations(session, observations, live_rows,
                                      by_match, max_lead, now, dry_run, stats)

                if not dry_run and updates:
                    # Status-only outcomes (late/missing/invalid) carry no
                    # per-row payload, so they go back as one UPDATE per status
                    # rather than one SELECT+UPDATE per pick. The previous
                    # session.get() loop was an N+1: the rows were loaded as
                    # column tuples, so nothing was in the identity map and a
                    # 1,070-pick legacy sweep meant 1,070 round trips to
                    # Supabase — against a docstring promising two queries.
                    status_only = defaultdict(list)
                    captured = []
                    for u in updates:
                        if u["status"] == STATUS_CAPTURED:
                            captured.append(u)
                        else:
                            status_only[u["status"]].append(u["id"])

                    for status, ids in status_only.items():
                        for chunk in _chunks(ids, 500):
                            session.query(SavedPick).filter(
                                SavedPick.id.in_(chunk)
                            ).update(
                                {SavedPick.closing_capture_status: status},
                                synchronize_session=False,
                            )
                            stats["db_queries"] += 1

                    # Captured rows each carry their own price, so they stay
                    # individual. They are bounded by the capture window (picks
                    # kicking off in the next `within_minutes`), not by history.
                    for u in captured:
                        pick = session.get(SavedPick, u["id"])
                        if pick is None:
                            continue
                        pick.closing_capture_status = u["status"]
                        pick.closing_odds = u["closing_odds"]
                        pick.closing_fair_probability = u["closing_fair_probability"]
                        pick.closing_bookmaker_count = u["closing_bookmaker_count"]
                        pick.closing_odds_captured_at = u["captured_at"]
                        stats["db_queries"] += 1

                    session.commit()
            break

        except Exception as exc:
            # Transient failures (Supabase cold start, pooler hiccup) retry with
            # backoff. The work is idempotent, so a retry cannot double-capture.
            if attempt >= max_retries:
                logger.error(
                    f"capture_closing_lines: giving up after {attempt} "
                    f"attempt(s): {exc}")
                raise
            wait = 2 ** attempt
            logger.warning(
                f"capture_closing_lines: attempt {attempt} failed ({exc}) — "
                f"retrying in {wait}s")
            time.sleep(wait)

    stats["elapsed_s"] = round(time.monotonic() - started, 2)
    logger.info(
        f"capture_closing_lines: {stats['captured']} captured, "
        f"{stats['missing']} missing, {stats['late']} late, "
        f"{stats['invalid']} invalid (of {stats['considered']} pending)"
        + ("  [DRY RUN — nothing written]" if dry_run else ""))
    if stats["missing"] or stats["invalid"]:
        logger.warning(
            f"capture_closing_lines: {stats['missing']} pick(s) had no usable "
            f"price and {stats['invalid']} had no market mapping. These are "
            f"recorded explicitly, NOT left pending, so CLV coverage stays honest.")
    return stats


def print_coverage() -> None:
    """Current CLV coverage, straight from the database."""
    from src.evaluation.clv import coverage_report

    db = get_db()
    with db.get_session() as session:
        rows = session.query(
            SavedPick.odds, SavedPick.closing_odds,
            SavedPick.closing_odds_captured_at, SavedPick.market,
            SavedPick.selection, SavedPick.closing_capture_status,
            Match.match_date,
        ).join(Match, Match.id == SavedPick.match_id).all()

    class _P:
        def __init__(self, r):
            self.odds = r.odds
            self.closing_odds = r.closing_odds
            self.closing_odds_captured_at = r.closing_odds_captured_at
            self.market = r.market
            self.selection = r.selection
            self.kickoff = r.match_date

    cov = coverage_report([_P(r) for r in rows])
    print(cov.render())
    by_status = defaultdict(int)
    for r in rows:
        by_status[r.closing_capture_status or "pending"] += 1
    print("  capture status breakdown:")
    for k, v in sorted(by_status.items()):
        print(f"      {k:<12} {v}")


def _load_env() -> None:
    """Load .env — called from main(), never at import.

    Import-time load_dotenv() is a trap: tests/conftest.py deliberately pops
    DATABASE_URL so nothing can reach production, and importing this module put
    it straight back. That turned a SQLite unit test into a live write against
    the production database (caught by an IntegrityError, but only by luck).
    Environment mutation is a side effect and belongs in an entry point.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # optional dependency; env may already be exported
        pass


def main():
    _load_env()
    from src.utils.config import get_config

    default_window = 90
    try:
        default_window = int(get_config().get(
            "betting.clv_capture_window_minutes", 90))
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--within-minutes", type=int, default=None,
                    help="capture picks whose kickoff is inside this window "
                         "(default: betting.clv_capture_window_minutes, else 90)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stats", action="store_true",
                    help="print CLV coverage and exit")
    args = ap.parse_args()

    if args.stats:
        print_coverage()
        return

    window = args.within_minutes if args.within_minutes is not None else default_window
    stats = capture(window, args.dry_run)
    print(
        f"capture: considered={stats['considered']} captured={stats['captured']} "
        f"missing={stats['missing']} late={stats['late']} "
        f"invalid={stats['invalid']}\n"
        f"cost:    {stats['db_queries']} db queries, "
        f"{stats['odds_rows_read']} odds rows read, {stats['elapsed_s']}s"
    )


if __name__ == "__main__":
    main()
