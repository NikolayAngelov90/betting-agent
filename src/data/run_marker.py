"""Has today's picks run finished? A fact the refresh must not guess.

THE PATH THIS CLOSES. `refresh_and_capture` rewrites `odds` for fixtures near
kickoff. The picks run reads `odds`. They are ordered by cron — picks at 03:00,
first refresh at 10:47 — so under normal delay the picks are written long
before any refresh fires and the two never interact.

**That ordering is a scheduling coincidence, not a guarantee.** The observed
maximum scheduler delay is 11h21m and it has occurred twice (2026-08-27 at
19:58, 2026-08-28 at 20:58). **Beyond about 7h40m of delay the picks run reads
odds a refresh has already rewritten, and the run becomes selection-affecting
without announcing it** — a cohort break with no marker, which is the worst
shape this ledger has catalogued.

WHY A WRITTEN MARKER AND NOT AN INFERENCE. The obvious signal is "does a
`saved_picks` row exist for today". It does not work: a legitimate zero-pick day
and a picks run that never happened both produce zero rows. That is precisely
the ambiguity this project has now found five times — `[]` meaning both
measured-and-clean and never-ran, a fallback degrading silently, a guard that
logs only when it acts. **The completion of a run is not derivable from its
output; it has to be recorded by the run.**

`api_budget` is reused rather than adding a table: it is already keyed
`(day, provider)`, already migrated, and `closing-lines` never writes an
api-football row, so a reserved provider key cannot collide.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger()

#: Reserved `api_budget.provider` value. Not a provider — a completion marker.
PICKS_MARKER = "__picks_run_complete__"


def mark_picks_complete(db, day: Optional[_date] = None) -> bool:
    """Record that today's picks run finished writing picks.

    Called at the END of the picks path, after picks are persisted. Never
    raises: failing to write the marker must not fail a run that has already
    done its work — but it is logged at WARNING, because the consequence is
    that the next refresh will decline and a day's captures will be lost.
    """
    day = day or _date.today()
    try:
        from sqlalchemy import text
        with db.get_session() as s:
            s.execute(text("""
                INSERT INTO api_budget (day, provider, used, limit_)
                VALUES (:d, :p, 1, 1)
                ON CONFLICT (day, provider) DO UPDATE SET used = api_budget.used + 1
            """), {"d": day, "p": PICKS_MARKER})
            s.commit()
        logger.info(f"PICKS RUN MARKER written for {day} — the odds refresh may "
                    f"now run without racing pick-time prices")
        return True
    except Exception as exc:
        logger.warning(
            f"PICKS RUN MARKER could not be written for {day} ({exc}). The next "
            "odds refresh will decline, and a day of closing-line captures will "
            "be lost. This is the safe direction, but it is not free.")
        return False


def picks_completed(db, day: Optional[_date] = None) -> Optional[bool]:
    """True / False / None — and None is NOT False.

    ``None`` means the question could not be answered (no table, no database).
    The caller must treat that differently from a confident "not yet": absence
    of evidence is not evidence, which is the distinction `_parse_match_date`
    and `coverage_checks` both draw.
    """
    day = day or _date.today()
    try:
        from sqlalchemy import text
        with db.get_session() as s:
            n = s.execute(text(
                "SELECT count(*) FROM api_budget WHERE day = :d AND provider = :p"
            ), {"d": day, "p": PICKS_MARKER}).scalar()
        return bool(n)
    except Exception as exc:
        logger.warning(
            f"PICKS RUN MARKER could not be read ({exc}) — the refresh guard is "
            "UNAVAILABLE for this run, not satisfied.")
        return None


def refresh_may_run(db, day: Optional[_date] = None) -> tuple:
    """(may_run, reason). Announces itself on every path, including the quiet one.

    THE THREE CASES, and only one of them blocks:

      * marker present  -> RUN. The picks are written; a refresh cannot reach
        back and change the price they were taken at.
      * marker absent   -> DECLINE. Today's picks run has not finished. A
        refresh now can rewrite odds the picks run is about to read.
      * unanswerable    -> RUN, loudly. The guard is unavailable; declining
        every capture on a database hiccup costs more than the exposure it
        would prevent, and the exposure is the pre-existing status quo rather
        than something this guard introduced.

    A guard that can silently do nothing must say it ran — so this logs on all
    three paths, not only when it blocks.
    """
    day = day or _date.today()
    done = picks_completed(db, day)
    if done is None:
        logger.warning(
            f"PICKS-RUN GUARD UNAVAILABLE for {day} — proceeding. This run has "
            "no evidence either way that today's picks are written.")
        return True, "guard unavailable"
    if done:
        logger.info(f"PICKS-RUN GUARD: picks for {day} are complete — refresh may run")
        return True, "picks complete"
    logger.warning(
        f"PICKS-RUN GUARD: DECLINING the odds refresh — today's ({day}) picks "
        "run has not recorded completion. Refreshing now would rewrite odds the "
        "picks run is about to read, which makes the run selection-affecting "
        "without a cohort bump. Waiting is the cheap side of this trade.")
    return False, "picks run not complete"
