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

from src.data import h1_window as _h1_window
from src.data.api_budget import ApiBudgetStore
from src.utils.logger import get_logger

logger = get_logger()

PROVIDER = "theoddsapi"

#: THE ATTRIBUTED COUNTER for H1's own spend (added 2026-10-09).
#:
#: WHY A SECOND ROW AND NOT A SECOND COLUMN: `api_budget` is keyed
#: (day, provider) and this needs no migration, which matters because the
#: alternative was shipping a schema change into a live eight-day window.
#:
#: WHY IT EXISTS AT ALL — the defect it repairs, measured 2026-10-09: the stop
#: read `credits_used()`, which is MONTH-TO-DATE TOTAL for the provider, and
#: compared it against H1's 168 reservation. 38 credits spent by NORMAL pricing
#: were therefore charged to H1 before H1 had spent anything, cutting its usable
#: budget to 130 (21 league-days instead of 28). Worse than the arithmetic: had
#: normal pricing alone reached 168, H1 would have halted
#: `RESERVATION_EXHAUSTED` having spent ZERO — a terminal state whose stated
#: reason ("the allocation is spent") would have been false.
#:
#: The report path already knew the difference and said so ("month-to-date is
#: not collection-to-date"); the ENFORCING path did not. One quantity, two
#: definitions, and the caveat lived on the side that only printed.
#:
#: This row counts ONLY credits claimed by a `for_h1=True` instance. The
#: authoritative total still counts them too, so the free-tier guard is
#: unchanged — H1 spend is counted twice ON PURPOSE, once for the pool and once
#: for the account.
PROVIDER_H1 = "theoddsapi:h1"

#: Free tier. Not a target — see DEFAULT_MONTHLY_BUDGET.
FREE_TIER_CREDITS = 500

#: What we allow ourselves to spend. Deliberately below the free tier so an
#: unusually busy fixture month cannot exhaust it: Stage 6's simulation put
#: normal operation at ~256 credits/month with a worst observed day of 46.
DEFAULT_MONTHLY_BUDGET = 400

#: Credits held back from the budget for manual/diagnostic calls.
DEFAULT_SAFETY_MARGIN = 50

# ─────────────────── THE ALLOCATION (Stage 26, 2026-10-01) ───────────────────
#
# TWO INDEPENDENT CEILINGS OVER ONE EXHAUSTIBLE POOL ARE NOT A BUDGET.
# `CREDIT_CEILING = 200` bounded H1; `monthly_budget - safety_margin = 400`
# bounded everything; nothing allocated between them, so first-come won — and H1
# runs LAST by construction, after the card has been priced.
#
# MEASURED 2026-10-01, and it is worse than first-come: September's 400 credits
# were spent in ELEVEN DAYS (09-01..09-11) and the pipeline then ran the
# remaining nine days of in-season football with NO Odds API pricing at all. The
# pool is not consumed evenly; 55% of it went on 09-04/05/06.
#
#   pick-time pricing   113 requests = 226 credits (57%)   hours 07-08
#   pre-kickoff refresh  85 requests = 170 credits (43%)   hours 13-18
#
# (Inferred from append-only `odds_snapshots` as distinct (league, minute) pairs
# and CROSS-CHECKED against the ledger: 198 x 2 = 396 against a recorded 400.)
#
# So the reservation is deducted from the pool BEFORE normal operation sees it,
# and normal operation's ceiling is the REMAINDER rather than a second constant.

#: Reserved for H1 collection. 168 is H1's upper sizing (106-168), and it is
#: affordable because the 170-credit pre-kickoff refresh path buys CLV
#: observations on an axis Stage 16 already resolved — upper bound +0.107%
#: against a +1.85% best-line break-even, recorded as ALREADY SATISFIED.
#:
#: NOT the same number as `CREDIT_CEILING = 200`, and deliberately smaller: the
#: ceiling is the REGISTERED experiment stop and must not move, while this is a
#: budget allocation. Because 168 < 200 the reservation binds first, which is
#: why `RESERVATION_EXHAUSTED` exists as its own terminal state.
H1_RESERVED_CREDITS = 168

#: The reservation turns itself on and off BY CALENDAR, never by a flag. A switch
#: someone must remember to unset is how SUP-1 happens, and this one would
#: silently throttle normal pricing for every month after the experiment.
#:
#: ── MOVED TO THE RE-RUN WINDOW. 2026-10-28, AUTHORISED. ───────────────────
#:
#: WAS 2026-10-09 .. 2026-10-17. The first attempt returned n=4 because the
#: INPUT starved, not because the method failed, and the re-run is chosen.
#:
#: BOTH DATES ARE DERIVED, not picked:
#:   FROM  = the TheOddsAPI monthly credit reset, which is the 1st. Starting
#:           anywhere later spends the reservation out of a pool normal
#:           pricing has already drawn on, and the whole point of an
#:           allocation is that neither consumer depends on the other's
#:           timing.
#:   UNTIL = FROM + 8 days, exclusive, preserving the pre-registered
#:           EIGHT-DAY length exactly (10-09..10-16 inclusive was 8; so is
#:           11-01..11-08). The LENGTH is pre-registered; only its position
#:           moves, so this is a reschedule and not an amendment to what is
#:           collected.
#:
#: The 2026-11-04 rate decision therefore falls on DAY 4 OF 8 — the midpoint,
#: which is where a projection can still change the plan.
#:
#: AND IT WAS INERT UNTIL NOW, which is why this had to move before anything
#: else: with the old window past, `h1_reservation()` returned 0 on 11-01, so
#: `h1_ceiling` was 0 and H1 would have been refused ENTIRELY. That is the
#: safe direction and it is still a non-start.
#: ── IMPORTED, NOT RESTATED. 2026-10-28. ──────────────────────────────────
#: These were two literals here and two more in `h1_collection_check`, and
#: four sites were half-moved in one week. `src/data/h1_window.py` is now the
#: only place either date appears; these names stay as the published spelling
#: every existing consumer already imports.
H1_RESERVATION_FROM = _h1_window.FROM
H1_RESERVATION_UNTIL = _h1_window.UNTIL          #: exclusive — eight days

#: THE RETIRED WINDOW, kept because a figure measured inside it is not
#: comparable with one measured inside the new one, and the first attempt's
#: `n=4` is quoted often enough that the dates it belongs to must stay
#: readable.
RETIRED_H1_RESERVATION_WINDOW = _h1_window.RETIRED


def h1_reservation(today: Optional[_date] = None) -> int:
    """Credits reserved for H1 TODAY. Zero outside the window."""
    today = today or _date.today()
    if H1_RESERVATION_FROM <= today < H1_RESERVATION_UNTIL:
        return H1_RESERVED_CREDITS
    return 0


def allocate(monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
             safety_margin: int = DEFAULT_SAFETY_MARGIN,
             today: Optional[_date] = None) -> dict:
    """THE ONE ALLOCATION. Both consumers read this; neither keeps a copy.

    `normal_ceiling` is computed as the remainder, so it cannot drift away from
    the pool it is carved out of — which is precisely what two independent
    constants did.
    """
    pool = max(0, int(monthly_budget) - int(safety_margin))
    reserved = min(max(0, h1_reservation(today)), pool)
    return {"pool": pool,
            "h1_reserved": reserved,
            "normal_ceiling": pool - reserved,
            #: H1'S OWN CEILING, ADDED 2026-10-23 — AND IT IS THE SAME NUMBER,
            #: not a second one. Stage 26 collapsed two independent constants
            #: into this remainder; naming H1's side here adds an ENFORCEMENT
            #: POINT for the existing allocation, which is not the same thing
            #: as adding a definition. Both consumers still read one function.
            #:
            #: IT EXISTS BECAUSE THE ALLOCATION WAS ONE-SIDED. Measured
            #: 2026-10-23 on a throwaway ledger: inside the window, a
            #: `for_h1=True` instance asked for 144 league requests and was
            #: GRANTED ALL 144 = 288 credits against a 168 reservation,
            #: overdrawing by 120 and leaving normal pricing with `remaining()
            #: == 0` and a claim of 50 refused outright. Normal operation
            #: could not touch H1's share; H1 could take normal operation's.
            "h1_ceiling": reserved,
            "window": (H1_RESERVATION_FROM.isoformat(),
                       H1_RESERVATION_UNTIL.isoformat())}

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
                 max_credits_per_run: int = DEFAULT_MAX_CREDITS_PER_RUN,
                 for_h1: bool = False):
        #: `for_h1` decides which side of the allocation this instance spends.
        #: Normal operation (the default) has H1's reservation WITHHELD from it,
        #: so it halts at the remainder with the reservation intact. H1's own
        #: collection sets it True — the reservation is its to spend, and its
        #: ceiling is enforced by `collection_stop`, not here.
        self.for_h1 = bool(for_h1)
        self.monthly_budget = int(monthly_budget)
        self.safety_margin = int(safety_margin)
        self.max_credits_per_run = int(max_credits_per_run)
        self._store = ApiBudgetStore(db, PROVIDER, self.monthly_budget)
        #: H1's ATTRIBUTED spend. A pure counter: its `daily_limit` is the free
        #: tier so it can never refuse a write, because the bound lives in
        #: `collection_stop` and a ledger ceiling here would be a second
        #: definition of it.
        self._h1_store = ApiBudgetStore(db, PROVIDER_H1, FREE_TIER_CREDITS)
        #: Credits claimed by THIS process, for the per-run ceiling.
        self.spent_this_run = 0

    # ------------------------------------------------------------- capability

    def available(self) -> bool:
        """Whether the durable ledger can be used at all."""
        return self._store.available()

    # ----------------------------------------------------------------- state

    def used(self, today: Optional[_date] = None) -> int:
        return self._store.used(day=month_key(today))

    def h1_used(self, today: Optional[_date] = None) -> Optional[int]:
        """Credits spent BY H1 this month, or None when unattributable.

        NOT `used()`. `used()` is the pool's total and includes normal pricing;
        this is the only number comparable to `H1_RESERVED_CREDITS`.
        """
        if not self._h1_store.available():
            return None
        return self._h1_store.used(day=month_key(today))

    def reserved_from_me(self, today: Optional[_date] = None) -> int:
        """Credits this instance may NOT spend: the margin, plus — for normal
        operation inside the H1 window — H1's **unspent** reservation.

        One source for both numbers, so the remainder cannot drift from the
        pool.

        UNSPENT, NOT THE WHOLE RESERVATION, and the difference strands
        credits. Measured 2026-10-23, immediately after the H1 side was
        bound: with H1's 168 fully spent, normal operation's `remaining()`
        read **64** instead of its 232 ceiling, because the reservation was
        withheld from the pool a second time after H1 had already drawn it.
        Total spendable across both consumers was 168 + 64 = 232 of a
        400-credit pool — **168 credits unspendable by anyone**, which is the
        "two independent ceilings over one exhaustible pool" defect Stage 26
        closed, re-entering through the other door.

        FAIL-CLOSED DIRECTION: when the attributed counter cannot be read,
        withhold the FULL reservation. Not knowing how much H1 has spent must
        protect H1's share, never release it.
        """
        if self.for_h1:
            return self.safety_margin
        reservation = h1_reservation(today)
        if reservation <= 0:
            return self.safety_margin
        spent = self.h1_used_for_reserve(today)
        if spent is None:
            return self.safety_margin + reservation
        return self.safety_margin + max(0, reservation - spent)

    def h1_used_for_reserve(self, today: Optional[_date] = None):
        """H1's attributed spend as the NORMAL side needs it, or None.

        Separate from `h1_used` only so a normal instance may read the H1
        counter without being `for_h1` — the attribution store is built by
        every instance, and reading it is not spending from it.
        """
        if not self._h1_store.available():
            return None
        return self._h1_store.used(day=month_key(today))

    def h1_headroom(self, today: Optional[_date] = None) -> Optional[int]:
        """Credits H1 may still spend under ITS OWN allocation, or None.

        `None` means the attributed counter could not be read, and the caller
        must treat that as "spend nothing" rather than as full headroom:
        an H1 credit that cannot be attributed is refused anyway (see
        `claim_requests`), and a bound compared against an unreadable counter
        never binds.

        MEASURED AGAINST `h1_used()`, NEVER `used()`. That distinction is
        Stage 29's second defect: `used()` is the pool's month-to-date total
        across every consumer, and comparing it to the 168 reservation once
        charged H1 with 38 credits spent by normal pricing before H1 had run
        at all.
        """
        if not self.for_h1:
            return None
        ceiling = allocate(self.monthly_budget, self.safety_margin,
                           today)["h1_ceiling"]
        used = self.h1_used(today)
        if used is None:
            return None
        return max(0, ceiling - used)

    def remaining(self, today: Optional[_date] = None) -> int:
        """Spendable credits for THIS instance, and the number that is enforced.

        Normal operation: the pool after the safety margin AND H1's
        reservation. Withholding the reservation here is what makes normal
        operation halt at its remainder.

        H1: the SMALLER of its own allocation headroom and what is physically
        left in the pool after the margin. Both limbs are needed — the first
        stops H1 overdrawing its share, the second stops it spending credits
        that do not exist.

        `claim_requests` CLAMPS TO THIS NUMBER, so there is one definition and
        no second ceiling to drift.

        THIS DOCSTRING HAS BEEN FALSE TWICE, in the same way each time, and
        the second time was the mirror of the first:

        * UNTIL 2026-10-09 it said `claim_requests` "walks down from this
          number" — it walked down from what the CALLER asked for and claimed
          against `safety_margin` alone, so the reported ceiling bound
          nothing on the NORMAL side.
        * UNTIL 2026-10-23 the H1 side was still unbound here. `for_h1=True`
          made `reserved_from_me` drop the reservation, which is correct —
          the reservation is H1's to spend — but nothing then capped H1 AT
          the reservation, so `remaining()` reported the whole 400-credit
          pool to an instance allocated 168.

        Verified by a consumer being REFUSED, never by reading a printed
        number: see `test_H1_cannot_spend_past_its_RESERVATION` and
        `test_normal_pricing_cannot_spend_the_H1_RESERVATION`, which drive
        the two directions against each other.
        """
        pool_room = max(0, self._store.remaining(
            reserve=self.reserved_from_me(today), day=month_key(today)))
        if not self.for_h1:
            return pool_room
        head = self.h1_headroom(today)
        if head is None:
            # FAIL CLOSED. An unreadable attribution counter is not headroom.
            return 0
        return min(head, pool_room)

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
            # THE ASYMMETRY IS DELIBERATE. Normal pricing fails OPEN because a
            # missing ledger must not cost the card its prices — stale odds are
            # worse than unbudgeted ones. H1 fails CLOSED because it is a
            # REGISTERED experiment with a credit bound: credits it cannot
            # account for produce observations it cannot defend, and
            # `collection_stop` would read an attributed zero and never halt.
            # Spent credits that produce unusable evidence are a defect, not a
            # cost.
            if self.for_h1:
                logger.error(
                    "OddsApiQuota: api_budget unavailable and this is an H1 "
                    "instance — REFUSING to spend. H1 credits must be "
                    "attributable; an unmeasurable spend cannot be reported "
                    "against the 168 reservation and would read as zero.")
                return 0
            logger.warning(
                "OddsApiQuota: api_budget table unavailable — falling back to "
                "the header-based credit guard; the monthly ledger is NOT "
                "enforcing a budget this run")
            self.spent_this_run += credits_for(n_requests)
            return n_requests

        day = month_key(today)

        # ── THE ALLOCATION IS ENFORCED HERE, FOR BOTH CONSUMERS ───────────
        #
        # `remaining()` is the reported ceiling, so it is the one clamped to.
        # Before 2026-10-23 only the NORMAL side was bound, by the pool-level
        # `reserve=` below; H1's own 168 was reported by `remaining()` and
        # enforced nowhere, so an H1 instance asking for 144 requests got all
        # 144 (288 credits) and left normal pricing at zero.
        #
        # For normal operation this is belt to the pool-level reserve's
        # braces and changes no outcome — it reaches the same number without
        # the walk-down. For H1 it is the only enforcement there is.
        allowed = self.remaining(today) // CREDITS_PER_REQUEST
        if allowed < n_requests:
            logger.warning(
                f"OddsApiQuota: the allocation limits this "
                f"{'H1' if self.for_h1 else 'normal'} instance to {allowed} of "
                f"{n_requests} request(s) — {self.remaining(today)} credit(s) "
                f"spendable"
                + (f", H1 headroom {self.h1_headroom(today)} of "
                   f"{allocate(self.monthly_budget, self.safety_margin, today)['h1_ceiling']}"
                   if self.for_h1 else ""))
            n_requests = max(0, allowed)
        if n_requests == 0:
            return 0

        granted = 0
        # Claim the whole block first; if refused, walk down so a partially
        # available budget still buys the most valuable requests (the caller
        # orders them by priority).
        want = n_requests
        while want > 0:
            # `reserved_from_me`, NOT `safety_margin`. MEASURED 2026-10-09:
            # with the margin alone, a non-H1 instance inside the window was
            # granted 200 of 200 requests = 400 credits against a reported
            # `normal_ceiling` of 232 — it ate all 168 of H1's reservation. The
            # allocation was computed in `remaining()` and ENFORCED NOWHERE,
            # because this is the only claim site and it reserved the smaller
            # of the two numbers. A ceiling that is reported but not enforced
            # is not a ceiling.
            if self._store.claim(credits_for(want),
                                 reserve=self.reserved_from_me(today),
                                 day=day):
                granted = want
                break
            want -= 1

        # ATTRIBUTE BEFORE RETURNING, AND REFUSE IF IT CANNOT BE DONE.
        # This is what makes `h1_used()` equal to H1's spend BY CONSTRUCTION
        # rather than by convention: an unattributed H1 credit is impossible,
        # so a counter reading 0 means zero spent — not broken attribution.
        if self.for_h1 and granted > 0:
            attributed = self._h1_store.record_attributed(
                credits_for(granted), day=day)
            if attributed is None:
                self._store.release(credits_for(granted), day=day)
                logger.error(
                    f"OddsApiQuota: H1 claim of {credits_for(granted)} credit(s) "
                    f"COULD NOT BE ATTRIBUTED — the claim has been RELEASED and "
                    f"the request is refused. Nothing was spent.")
                return 0
            logger.info(
                f"H1_CREDITS_ATTRIBUTED credits={credits_for(granted)} "
                f"h1_total={attributed}/{h1_reservation(today)} month={day:%Y-%m}")

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
