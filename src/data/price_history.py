"""Stage 18 C1/C2 — record the price path, once, for every writer.

THE HABIT, PRE-EMPTED. Two scrapers persist odds (`theodds_scraper` and
`apifootball_scraper`) and both need identical snapshot behaviour. Writing it
twice is how `(1.005, 1.25)` came to exist three times. It is written here once
and imported.

WHAT THIS IS NOT. It does not change `odds`. That table keeps its unique
constraint, its columns and its exact contents, so every current-price consumer
reads what it read yesterday and Stage 3's egress work is untouched. This only
appends to a table nothing reads yet.

FAIL-OPEN, DELIBERATELY. History is a research nicety; pricing picks is the job.
If the snapshot insert fails, the run must continue and the odds write must
stand. The alternative — a failed history write breaking pick generation — trades
a working system for a research convenience, which is the wrong way round.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.data.models import InjuryObservation, OddsSnapshot
from src.utils.logger import get_logger, utcnow

logger = get_logger()


#: A quote is FRESH when the book changed it no longer ago than this before
#: we observed it. Registered in docs/confident-value-preregistration.md
#: section 5, before any fresh subset existed to tune it on.
FRESH_MAX_AGE_S = 600.0

QUOTE_FRESH = "FRESH"
QUOTE_STALE = "STALE"
#: No `last_update` came with the quote. NOT fresh and NOT stale: unknown.
QUOTE_UNEVALUATED = "UNEVALUATED"


def parse_book_last_update(value):
    """The feed's `last_update` as a naive-UTC datetime, or None.

    NONE IS THE ONLY FALLBACK. Absent, empty, the wrong type or unparseable
    all return None -- never `utcnow()`. A quote stamped with our own fetch
    time would have age zero, and "we do not know how old this is" would be
    stored as "this is as fresh as a price can be".
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def quote_age_seconds(observed_at, book_last_update):
    """Seconds between the book's last change and our observation, or None.

    None when either stamp is missing. A NEGATIVE age (the book's stamp is
    after our own) is returned as it is: it is a clock disagreement, and
    clamping it to zero would manufacture the freshest quote in the table.
    """
    if observed_at is None or book_last_update is None:
        return None
    return (observed_at - book_last_update).total_seconds()


def quote_freshness(observed_at, book_last_update,
                    max_age_s: float = FRESH_MAX_AGE_S) -> str:
    """FRESH, STALE or UNEVALUATED. A price with no known age is never fresh."""
    age = quote_age_seconds(observed_at, book_last_update)
    if age is None:
        return QUOTE_UNEVALUATED
    return QUOTE_FRESH if 0.0 <= age <= max_age_s else QUOTE_STALE


def record_price(session, *, match_id: int, bookmaker: str, market_type: str,
                 selection: str, odds_value: float, observed_at=None,
                 book_last_update=None) -> None:
    """Append one price observation. Never raises.

    `book_last_update` is the BOOK's own stamp for this quote, or None when
    the feed gave none. It is stored as given and never defaulted.
    """
    try:
        session.add(OddsSnapshot(
            match_id=int(match_id), bookmaker=str(bookmaker),
            market_type=str(market_type), selection=str(selection),
            # float() at the DB boundary: a numpy scalar leaking into SQL
            # surfaces as "schema X does not exist", which is unreadable.
            odds_value=float(odds_value),
            observed_at=observed_at or utcnow(),
            book_last_update=parse_book_last_update(book_last_update),
        ))
    except Exception as exc:                      # pragma: no cover - defensive
        logger.warning(f"price snapshot skipped for match {match_id}: {exc}")


def stamp_first_seen(row, observed_at=None) -> None:
    """Set `first_seen_at` once, on a row that has never carried it.

    IT SETS IT ONLY WHEN ABSENT, AND IT IS **NOT** THE ONLY SITE THAT SETS IT.
    That is a correction, made 2026-10-12. This docstring used to claim "this
    is the only site that sets it", and the claim was false from the day it was
    written:

        src/scrapers/apifootball_scraper.py:2174   "first_seen_at": now
        src/scrapers/theodds_scraper.py:981        first_seen_at=_now

    set the column inline, and **nothing in production calls this function at
    all** - it is TEST-ONLY, which `scripts/pnc_scan.py` reports. So the
    guarantee was being asserted by a definition no execution passed through.
    One rule, two definitions, and the docstring is part of the definition.

    WRITE-ONCE NEVERTHELESS HOLDS, and it was checked rather than assumed:
    `first_seen_at` is **absent from the `on_conflict_do_update` set_ clause**
    in `_upsert_odds_rows`, so a refresh cannot turn first-seen into last-seen.
    The property is real; the stated reason for it was not.

    WHY THIS IS A DOCSTRING FIX AND NOT A STRUCTURAL ONE. The two honest
    repairs are to wire this function and drop the inline writes, or to delete
    it. Both touch the odds write path, and H1 is collecting until 2026-10-17 -
    this function is in `pnc_scan.DEFERRED` for exactly that reason. The choice
    between wiring and deleting is part of that re-triage. Until then the
    description matches the behaviour, which is the part that was lying.

    Existing rows are NEVER backfilled - a guessed first-sight time would look
    like evidence, and Stage 17 established that `created_at` is a backfill
    stamp for 53.5% of matches.
    """
    try:
        if getattr(row, "first_seen_at", None) is None:
            row.first_seen_at = observed_at or utcnow()
    except Exception as exc:                      # pragma: no cover - defensive
        logger.warning(f"first_seen_at not stamped: {exc}")


def record_injury(session, *, team_id: int, player_id: Optional[int] = None,
                  injury_type: Optional[str] = None, status: Optional[str] = None,
                  start_date=None, source: Optional[str] = None,
                  observed_at=None) -> None:
    """Append one injury observation. `observed_at` is the whole point."""
    try:
        session.add(InjuryObservation(
            team_id=int(team_id),
            player_id=int(player_id) if player_id is not None else None,
            injury_type=injury_type, status=status, start_date=start_date,
            source=source, observed_at=observed_at or utcnow(),
        ))
    except Exception as exc:                      # pragma: no cover - defensive
        logger.warning(f"injury observation skipped for team {team_id}: {exc}")
