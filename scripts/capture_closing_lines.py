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

**Cheap.** Two queries per run regardless of fixture count: one for the pending
picks in the window, one for their odds rows, both column-projected. No N+1, no
``SELECT *``. ``--stats`` prints the measured cost.
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



def _median(values: List[float]) -> Optional[float]:
    s = sorted(values)
    n = len(s)
    if not n:
        return None
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def consensus_close(odds_rows, market_type: str, line: Optional[str],
                    side: Optional[str], leg: int) -> Tuple[Optional[float],
                                                            Optional[float], int]:
    """(median price for our leg, median de-vigged probability, books used).

    Each bookmaker's market is validated on its own before it contributes, so
    one broken book cannot define the close.
    """
    by_book: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in odds_rows:
        if r.bookmaker in EXCLUDED_BOOKMAKERS or r.market_type != market_type:
            continue
        if r.odds_value and r.odds_value > 1.0:
            by_book[r.bookmaker][r.selection] = r.odds_value

    prices, fairs = [], []
    for sels in by_book.values():
        legs = extract_legs(market_type, sels, line=line, side=side)
        if legs is None or leg >= len(legs):
            continue
        ok, _ = check_overround(market_type, legs)
        if not ok:
            continue
        probs = devig(market_type, legs)
        prices.append(legs[leg])
        if probs is not None:
            fairs.append(probs[leg])

    if not prices:
        return None, None, 0
    return _median(prices), (_median(fairs) if fairs else None), len(prices)


def capture(within_minutes: int = 90, dry_run: bool = False,
            max_retries: int = 3) -> dict:
    """Capture closing prices for pending picks kicking off inside the window."""
    db = get_db()
    now = utcnow()
    horizon = now + timedelta(minutes=within_minutes)
    stats = {
        "considered": 0, "captured": 0, "missing": 0, "late": 0, "invalid": 0,
        "db_queries": 0, "odds_rows_read": 0, "elapsed_s": 0.0,
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
                    Match.match_date,
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
                match_ids = {r.match_id for r in rows}

                # Only the market types the pending picks actually need. Without
                # this the query returns every market on every match — measured
                # at 104,117 rows on a full-history run, against ~7,600 for the
                # markets in use. The window keeps production small either way,
                # but there is no reason to ship rows nothing will read.
                needed_markets = {
                    SELECTION_SPEC[r.selection][0]
                    for r in rows if r.selection in SELECTION_SPEC
                }
                if not needed_markets:
                    needed_markets = {"1X2"}   # nothing mappable; keep the query valid

                # Query 2 — the relevant odds rows for those matches, once.
                odds_rows = session.query(
                    Odds.match_id, Odds.bookmaker, Odds.market_type,
                    Odds.selection, Odds.odds_value,
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

                updates: List[dict] = []
                for r in rows:
                    kickoff = r.match_date
                    # Late capture is not a closing price. Mark and exclude.
                    if kickoff is not None and now >= kickoff:
                        updates.append({"id": r.id, "status": STATUS_LATE})
                        stats["late"] += 1
                        continue

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
                    price, fair, n_books = consensus_close(
                        by_match.get(r.match_id, []), market_type, line, side, leg)

                    if price is None:
                        updates.append({"id": r.id, "status": STATUS_MISSING})
                        stats["missing"] += 1
                        continue

                    clv = (r.odds / price - 1) if r.odds else None
                    logger.info(
                        f"closing line: {r.match_name} {r.selection} "
                        f"taken @ {r.odds} closing @ {price:.2f} "
                        f"({n_books} books)"
                        + (f" CLV {clv:+.2%}" if clv is not None else ""))
                    updates.append({
                        "id": r.id, "status": STATUS_CAPTURED,
                        "closing_odds": float(price),
                        "closing_fair_probability": float(fair) if fair else None,
                        "closing_bookmaker_count": int(n_books),
                        "captured_at": now,
                    })
                    stats["captured"] += 1

                if not dry_run and updates:
                    for u in updates:
                        pick = session.get(SavedPick, u["id"])
                        if pick is None:
                            continue
                        pick.closing_capture_status = u["status"]
                        if u["status"] == STATUS_CAPTURED:
                            pick.closing_odds = u["closing_odds"]
                            pick.closing_fair_probability = u["closing_fair_probability"]
                            pick.closing_bookmaker_count = u["closing_bookmaker_count"]
                            pick.closing_odds_captured_at = u["captured_at"]
                    session.commit()
                    stats["db_queries"] += 1
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
