"""Cross-process daily request budget for external APIs.

The problem
-----------
``ApiFootballScraper`` counted spend in ``self._requests_today``, an instance
attribute initialised to 0 and never persisted. The daily workflow runs seven
separate Python processes, each constructing its own scraper, so a "100
requests/day" cap was really 100 *per process* — today's true ceiling is ~700.
With any sharding it becomes N_workers x 100, and every number derived from the
counter (``remaining_budget()``, ``BUDGET_XG``, ``BUDGET_RESERVE``, the odds
semaphore's capacity) is computed from a quantity that does not mean anything.

The claim protocol
------------------
Spend is *claimed before it is made*, with one conditional UPDATE:

    UPDATE api_budget SET used = used + :n
     WHERE day = :day AND provider = :p AND used + :n <= limit_
 RETURNING used

PostgreSQL takes a row lock for the UPDATE, so concurrent claimants serialise on
it and two workers can never both spend the last request. No row returned means
the claim was refused — the budget is gone.

Claiming before spending means a crashed process leaks its claim for the rest of
the day. That is the safe direction to fail: under-spending an API quota costs
freshness, over-spending costs money and can get the key suspended.

Degradation
-----------
If the ``api_budget`` table is missing (migration 002 not applied, or rolled
back) every method reports failure and the caller falls back to its in-process
counter — the previous behaviour, no crash.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Optional

from sqlalchemy import case, inspect, select, update

from src.data.models import ApiBudget
from src.utils.logger import get_logger, utcnow

logger = get_logger()


class ApiBudgetStore:
    """Atomic daily quota, shared by every process using the same database."""

    def __init__(self, db, provider: str, daily_limit: int):
        self.db = db
        self.provider = provider
        self.daily_limit = daily_limit
        self._available: Optional[bool] = None

    # ------------------------------------------------------------- capability

    def available(self) -> bool:
        """Whether the backing table exists. Probed once per instance."""
        if self._available is None:
            try:
                self._available = inspect(self.db.engine).has_table("api_budget")
            except Exception as e:
                logger.debug(f"api_budget probe failed: {e}")
                self._available = False
            if not self._available:
                logger.info(
                    "api_budget table not found — falling back to per-process "
                    "request accounting (migration 002 not applied?)"
                )
        return self._available

    # ------------------------------------------------------------------ state

    def _ensure_row(self, session, day: _date) -> None:
        """Create today's row if absent. Race-safe via ON CONFLICT DO NOTHING."""
        from sqlalchemy.dialects import postgresql, sqlite

        dialect = session.bind.dialect.name if session.bind is not None else ""
        insert = postgresql.insert if dialect == "postgresql" else sqlite.insert
        session.execute(
            insert(ApiBudget.__table__)
            .values(day=day, provider=self.provider, used=0,
                    limit_=self.daily_limit)
            .on_conflict_do_nothing(index_elements=["day", "provider"])
        )

    def used(self, day: Optional[_date] = None) -> int:
        """Requests already claimed today across all processes."""
        if not self.available():
            return 0
        day = day or _date.today()
        try:
            with self.db.get_session() as session:
                row = session.execute(
                    select(ApiBudget.used).where(
                        ApiBudget.day == day,
                        ApiBudget.provider == self.provider,
                    )
                ).first()
            return int(row[0]) if row else 0
        except Exception as e:
            logger.warning(f"api_budget read failed ({e}) — assuming 0 used")
            return 0

    def remaining(self, reserve: int = 0, day: Optional[_date] = None) -> int:
        return max(0, self.daily_limit - self.used(day) - reserve)

    # ------------------------------------------------------------------ claim

    def claim(self, n: int = 1, reserve: int = 0,
              day: Optional[_date] = None) -> bool:
        """Reserve ``n`` requests. True only if the whole claim fit.

        ``reserve`` keeps headroom for later stages (the scraper reserves room
        for the injury fetcher), so the effective ceiling is
        ``daily_limit - reserve``.
        """
        if n <= 0:
            return True
        if not self.available():
            return True   # caller keeps its own accounting

        day = day or _date.today()
        ceiling = max(0, self.daily_limit - reserve)
        try:
            with self.db.get_session() as session:
                self._ensure_row(session, day)
                row = session.execute(
                    update(ApiBudget.__table__)
                    .where(
                        ApiBudget.day == day,
                        ApiBudget.provider == self.provider,
                        ApiBudget.used + n <= ceiling,
                    )
                    .values(used=ApiBudget.used + n, updated_at=utcnow())
                    .returning(ApiBudget.__table__.c.used)
                ).first()
            return row is not None
        except Exception as e:
            # Never let budget bookkeeping break a scrape. Failing open here is
            # deliberate: the caller's in-process counter still applies.
            logger.warning(f"api_budget claim failed ({e}) — allowing the request")
            return True

    def release(self, n: int = 1, day: Optional[_date] = None) -> None:
        """Give back an unused claim (e.g. the request was never sent)."""
        if n <= 0 or not self.available():
            return
        day = day or _date.today()
        try:
            with self.db.get_session() as session:
                # CASE rather than greatest(): greatest() is PostgreSQL-only
                # and SQLite spells it max(). Keep one portable statement.
                session.execute(
                    update(ApiBudget.__table__)
                    .where(
                        ApiBudget.day == day,
                        ApiBudget.provider == self.provider,
                    )
                    .values(
                        used=case((ApiBudget.used - n < 0, 0),
                                  else_=ApiBudget.used - n),
                        updated_at=utcnow(),
                    )
                )
        except Exception as e:
            logger.debug(f"api_budget release failed: {e}")
