"""Monthly credit budget for The Odds API — claim before you spend.

Stage 6, Phase 6.

The Odds API free tier grants **500 credits per calendar month**, and charges
``1 credit x regions x markets`` per request. The scraper asks for `regions=eu`
and `markets=h2h,totals`, so every league request costs **2 credits**.

Why a ledger and not a counter
------------------------------
The existing protection reads ``x-requests-remaining`` from the last response and
persists it to ``data/models/theodds_credits.json``. That is *reactive* and it
has two holes:

* it only knows what the API told it **after** a request, so a burst of
  concurrent requests (the scraper fires them with ``asyncio.gather``) can
  overshoot before any header comes back;
* the file lives in the GitHub Actions cache. Lose the cache and the guard
  silently resets to "unknown", which reads as "spend freely".

This module claims spend **before** the request, in the database, with the same
conditional-UPDATE protocol ``ApiBudgetStore`` already uses for API-Football:

    UPDATE api_budget SET used = used + n
     WHERE day = <first of month> AND provider = 'theoddsapi'
       AND used + n <= limit_

PostgreSQL takes a row lock, so concurrent claimants serialise and two workers
can never both spend the last credit. No row returned means the budget is gone
and the caller must not call the API.

Monthly period on a daily table
-------------------------------
``ApiBudget`` is keyed ``(day, provider)``. Rather than add a table for a second
period type — and a production migration with it — a month is represented by its
**first day**. ``date(2026, 8, 1)`` *is* the August ledger. The row's meaning is
carried by the provider name, and the reuse costs nothing.

Failing safe
------------
Claiming before spending means a crashed process leaks its claim for the rest of
the month. That is the right direction to fail: under-spending costs freshness,
over-spending costs the ability to collect any closing lines at all until the
quota resets.

If the ``api_budget`` table is missing, ``available()`` is False and the caller
falls back to the pre-existing header-based guard — degraded, never crashing.
"""

from __future__ import annotations

from datetime import date as _date
from typing import Optional

from src.data.api_budget import ApiBudgetStore
from src.utils.logger import get_logger

logger = get_logger()

PROVIDER = "theoddsapi"

#: Free tier. Not a target — see DEFAULT_MONTHLY_BUDGET.
FREE_TIER_CREDITS = 500

#: What we allow ourselves to spend. Deliberately below the free tier so an
#: unusually busy fixture month cannot exhaust it: Stage 6's simulation put
#: normal operation at ~256 credits/month with a worst observed day of 46.
DEFAULT_MONTHLY_BUDGET = 400

#: Credits held back from the budget for manual/diagnostic calls.
DEFAULT_SAFETY_MARGIN = 50

#: Ceiling on what ONE workflow execution may spend, independent of the monthly
#: budget. The monthly guard alone cannot stop a single pathological run — a
#: fixture-data glitch that made 27 leagues look imminent would spend 54 credits
#: in one go and still be "within budget" until the month ran dry.
#:
#: Sized from measurement, not intuition: across 313 charged runs in the
#: simulated history the per-run cost was mean 3.7, p95 10, p99 14, max 20
#: credits. A cap of 24 clips 0% of observed runs while bounding a runaway to
#: 12 leagues.
DEFAULT_MAX_CREDITS_PER_RUN = 24

#: Cost of one league request: 1 credit x regions x markets.
REGIONS = 1
MARKETS = 2
CREDITS_PER_REQUEST = REGIONS * MARKETS


def _credit_account() -> str:
    """Which scheduled job is spending. GITHUB_WORKFLOW in CI, else 'local'.

    Deliberately an env read rather than a parameter: every call site would
    otherwise have to thread it through, and a call site that forgot would be
    indistinguishable from one that spent nothing.
    """
    import os
    return (os.environ.get("GITHUB_WORKFLOW") or "local").replace(" ", "-")


def month_key(today: Optional[_date] = None) -> _date:
    """The ledger key for a calendar month: its first day."""
    d = today or _date.today()
    return d.replace(day=1)


def credits_for(n_requests: int) -> int:
    """Credit cost of n league requests."""
    return max(0, int(n_requests)) * CREDITS_PER_REQUEST


class OddsApiQuota:
    """Monthly claim-before-spend budget for The Odds API."""

    def __init__(self, db, monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
                 safety_margin: int = DEFAULT_SAFETY_MARGIN,
                 max_credits_per_run: int = DEFAULT_MAX_CREDITS_PER_RUN):
        self.monthly_budget = int(monthly_budget)
        self.safety_margin = int(safety_margin)
        self.max_credits_per_run = int(max_credits_per_run)
        self._store = ApiBudgetStore(db, PROVIDER, self.monthly_budget)
        #: Credits claimed by THIS process, for the per-run ceiling.
        self.spent_this_run = 0

    # ------------------------------------------------------------- capability

    def available(self) -> bool:
        """Whether the durable ledger can be used at all."""
        return self._store.available()

    # ----------------------------------------------------------------- state

    def used(self, today: Optional[_date] = None) -> int:
        return self._store.used(day=month_key(today))

    def remaining(self, today: Optional[_date] = None) -> int:
        """Spendable credits, after the safety margin."""
        return max(0, self._store.remaining(
            reserve=self.safety_margin, day=month_key(today)))

    def max_requests(self, today: Optional[_date] = None) -> int:
        """How many league requests the remaining budget allows."""
        return self.remaining(today) // CREDITS_PER_REQUEST

    # ----------------------------------------------------------------- claim

    def claim_requests(self, n_requests: int,
                       today: Optional[_date] = None) -> int:
        """Claim budget for ``n_requests`` league calls.

        Returns the number of requests actually granted, which may be fewer than
        asked for and may be zero. The caller must make **at most** that many
        API calls — the credits are already spent from the ledger's point of
        view whether or not the calls happen.

        Never raises: a quota failure must degrade to "make no request", not to
        an exception in the middle of a scheduled job.
        """
        n_requests = max(0, int(n_requests))
        if n_requests == 0:
            return 0

        # Per-run ceiling first: it applies even when the durable ledger is
        # unavailable, so a degraded run still cannot spend without bound.
        if self.max_credits_per_run > 0:
            run_room = self.max_credits_per_run - self.spent_this_run
            allowed_by_run = max(0, run_room // CREDITS_PER_REQUEST)
            if allowed_by_run < n_requests:
                logger.warning(
                    f"OddsApiQuota: per-run ceiling limits this execution to "
                    f"{allowed_by_run} of {n_requests} request(s) "
                    f"({self.spent_this_run}/{self.max_credits_per_run} credits "
                    f"already spent this run)")
                n_requests = allowed_by_run
            if n_requests == 0:
                return 0

        if not self.available():
            logger.warning(
                "OddsApiQuota: api_budget table unavailable — falling back to "
                "the header-based credit guard; the monthly ledger is NOT "
                "enforcing a budget this run")
            self.spent_this_run += credits_for(n_requests)
            return n_requests

        day = month_key(today)
        granted = 0
        # Claim the whole block first; if refused, walk down so a partially
        # available budget still buys the most valuable requests (the caller
        # orders them by priority).
        want = n_requests
        while want > 0:
            if self._store.claim(credits_for(want), reserve=self.safety_margin,
                                 day=day):
                granted = want
                break
            want -= 1

        self.spent_this_run += credits_for(granted)

        # Stage 15 instrumentation. The 213/144 split between pick-time pricing
        # and closing capture was RECONSTRUCTED by inference over CI logs and
        # reconciled to within 2.3%; it drove the whole frontier calculation and
        # nothing measured it directly. One structured line per claim, tagged
        # with the workflow that spent it, makes the next stage's version of
        # that number a measurement instead of an argument.
        logger.info(
            f"CREDITS_CLAIMED account={_credit_account()} "
            f"credits={credits_for(granted)} requests={granted} "
            f"asked={n_requests} month={day:%Y-%m}")

        if granted < n_requests:
            logger.warning(
                f"OddsApiQuota: budget limited this run — asked for "
                f"{n_requests} league request(s) ({credits_for(n_requests)} "
                f"credits), granted {granted} ({credits_for(granted)}). "
                f"Month {day:%Y-%m}: {self.used(today)}/{self.monthly_budget} "
                f"used, {self.remaining(today)} spendable after a "
                f"{self.safety_margin}-credit safety margin."
            )
        else:
            logger.info(
                f"OddsApiQuota: claimed {credits_for(granted)} credits for "
                f"{granted} league request(s). Month {day:%Y-%m}: "
                f"{self.used(today)}/{self.monthly_budget} used."
            )
        return granted

    def reconcile(self, provider_used: Optional[int],
                  today: Optional[_date] = None) -> int:
        """Raise the ledger to match what the provider says has been spent.

        The ledger and The Odds API's own counter are independent. The ledger
        starts each month at zero and only knows about spend that went through
        it; the provider counts everything on the key. Measured 2026-08-10: the
        provider reported **95 used** while the ledger read **0**. Something
        outside this pipeline — a manual call, an MCP client, a second
        deployment — had already spent a fifth of the month's tier, and every
        budget decision here was being made against a number that was wrong in
        the dangerous direction.

        Only ever raises, never lowers. A provider count *below* the ledger
        means a month boundary or a different key, and in both cases spending
        more on the strength of it is the wrong bet.

        Returns the ledger's `used` after reconciliation.
        """
        current = self.used(today)
        if provider_used is None or not self.available():
            return current
        try:
            provider_used = int(provider_used)
        except (TypeError, ValueError):
            return current

        if provider_used <= current:
            return current

        logger.warning(
            f"OddsApiQuota: provider reports {provider_used} credits used this "
            f"month but the ledger recorded {current} — adopting the provider's "
            f"number ({provider_used - current} credit(s) were spent outside "
            f"this pipeline). Budget decisions were being made against a stale "
            f"count.")
        # raise_used_to, not claim: the spend has already happened, so a
        # refusal above our own budget would leave the wrong number in place.
        return self._store.raise_used_to(provider_used, day=month_key(today))

    def release_requests(self, n_requests: int,
                         today: Optional[_date] = None) -> None:
        """Give back credits for requests that were claimed but never made.

        Used when a request is skipped after claiming (e.g. the league turned
        out to be unmapped). Best-effort: a lost release only under-spends.
        """
        n_requests = max(0, int(n_requests))
        if n_requests and self.available():
            self._store.release(credits_for(n_requests), day=month_key(today))
            self.spent_this_run = max(
                0, self.spent_this_run - credits_for(n_requests))
            logger.debug(
                f"OddsApiQuota: released {credits_for(n_requests)} credits for "
                f"{n_requests} unmade request(s)")

    # ------------------------------------------------------------ diagnostics

    def describe(self, today: Optional[_date] = None) -> str:
        day = month_key(today)
        if not self.available():
            return ("OddsApiQuota: ledger UNAVAILABLE (api_budget table missing) "
                    "— header-based guard only")
        return (
            f"OddsApiQuota {day:%Y-%m}: {self.used(today)}/{self.monthly_budget} "
            f"credits used, {self.remaining(today)} spendable "
            f"(safety margin {self.safety_margin}, free tier {FREE_TIER_CREDITS}), "
            f"= {self.max_requests(today)} more league request(s); "
            f"this run has spent {self.spent_this_run}/{self.max_credits_per_run}"
        )
