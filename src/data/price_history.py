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

from typing import Optional

from src.data.models import InjuryObservation, OddsSnapshot
from src.utils.logger import get_logger, utcnow

logger = get_logger()


def record_price(session, *, match_id: int, bookmaker: str, market_type: str,
                 selection: str, odds_value: float, observed_at=None) -> None:
    """Append one price observation. Never raises."""
    try:
        session.add(OddsSnapshot(
            match_id=int(match_id), bookmaker=str(bookmaker),
            market_type=str(market_type), selection=str(selection),
            # float() at the DB boundary: a numpy scalar leaking into SQL
            # surfaces as "schema X does not exist", which is unreadable.
            odds_value=float(odds_value),
            observed_at=observed_at or utcnow(),
        ))
    except Exception as exc:                      # pragma: no cover - defensive
        logger.debug(f"price snapshot skipped for match {match_id}: {exc}")


def stamp_first_seen(row, observed_at=None) -> None:
    """Set `first_seen_at` once, on a row that has never carried it.

    Write-once by construction rather than by validator: this is the only site
    that sets it, and it sets it only when absent. Existing rows are NEVER
    backfilled — a guessed first-sight time would look like evidence and Stage
    17 established that `created_at` is a backfill stamp for 53.5% of matches.
    """
    try:
        if getattr(row, "first_seen_at", None) is None:
            row.first_seen_at = observed_at or utcnow()
    except Exception as exc:                      # pragma: no cover - defensive
        logger.debug(f"first_seen_at not stamped: {exc}")


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
        logger.debug(f"injury observation skipped for team {team_id}: {exc}")
