"""H1 collection: one job, three requests, in-job waits.

    python -m scripts.h1_collect --rounds 3 --gap-minutes 120 --max-credits 6
    python -m scripts.h1_collect --dry-run            # spends nothing

WHY IN-JOB WAITS AND NOT A CRON. Measured 2026-10-10 from
`gh run list --workflow=closing-lines.yml --limit 200` (`event=schedule`): the
nominal 8-slot, 120-minute cron fires a **median of 4 slots/day**, shaped as a
**12-24 minute pair** followed by ~200-minute gaps. The pair is below
`MIN_GAP_MINUTES = 30`, so it counts as ONE separated instant and the second
slot buys nothing. Simulated against those measured slots and real fixtures,
**every** `min_interval` from 120 down to 30 yields ZERO qualifying
trajectories. The scheduler cannot deliver the spacing, so the spacing is
produced here instead, where it is deterministic and verifiable per fixture.

A nominal schedule is not an observed cadence.

THE SPACING, AND WHY `min_interval` IS 90 AGAINST A 120-MINUTE GAP. A
qualifying triple needs each gap >= 30 minutes and `hi/lo` **strictly** < 2.0,
inside the 360-minute window; 120/120 qualifies with the widest safety margin.
`min_interval` must sit strictly below the achieved spacing or the per-league
dedup rejects the very observations this loop creates — that is the amended
pre-registration's condition, and 90 against a deterministic 120 satisfies it
with 30 minutes of slack.

THE STOP IS CHECKED BEFORE EACH ROUND, NOT AFTER THE WAITS. An exhausted
reservation must halt the loop mid-sequence rather than after spending again.
`refresh_imminent` re-checks it internally before any request, so the bound is
enforced twice and the outer check additionally skips the remaining waits — a
job that has halted must not hold a runner for two more hours.

COST. 3 requests x 2 credits = 6 credits per league-day, so ~22 credits/day is
3-4 leagues and seven days fits inside the 168-credit reservation.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timedelta

from src.utils.logger import utcnow

#: H1's registered window. Passed, never defaulted.
H1_WINDOW_MINUTES = 360

#: ZERO, and that is a robustness decision rather than a relaxation.
#:
#: The amended pre-registration requires `min_interval` strictly below the
#: achieved spacing; 0 satisfies that with the whole gap as slack. The reason
#: to go all the way to 0 is CROSS-WORKFLOW COUPLING, found by reading
#: `_leagues_refreshed_since`: it filters on
#: `Odds.bookmaker LIKE 'TheOddsAPI%'` and a timestamp, and **nothing else** —
#: it is WRITER-AGNOSTIC. So a `closing-lines` run, which fires at `17 11,13,
#: 15,17,19,21,23` and has been observed starting 16:56-17:25, would mark a
#: shared league "refreshed within the last 90 min" and H1's NEXT ROUND WOULD
#: BE SILENTLY SKIPPED — truncating the trajectory to two instants, which is
#: exactly the shape that yields nothing.
#:
#: H1 does not need the dedup. The dedup exists so a 2-hourly cron does not pay
#: twice for the same freshness; H1's spacing is enforced by its own in-job
#: waits, so the only thing the dedup can do here is let another workflow's
#: timing decide whether H1 collects. The per-job `--max-credits` cap bounds
#: the spend either way.
H1_MIN_INTERVAL_MINUTES = 0

#: The in-job gap between requests. 120 puts the two intervals at 1.0 ratio.
H1_GAP_MINUTES = 120

#: Three instants is what `qualifying_triple` needs.
H1_ROUNDS = 3


def _load_env() -> None:
    """Read `.env` from main() only, never at import — an import-time load
    re-introduces DATABASE_URL after conftest strips it."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def _stop_now(since: datetime):
    """Evaluate the shipped stop against the database. One definition."""
    from scripts.h1_collection_check import collection_state
    from src.data.database import get_db

    try:
        with get_db().get_session() as s:
            return collection_state(s, since)
    except Exception as exc:
        from scripts.h1_collection_check import CollectionStop

        # FAIL CLOSED, and say which fact it is: a stop that cannot be
        # evaluated is not a stop that passed.
        return CollectionStop(
            True, "UNEVALUATED",
            f"could not evaluate the stop ({type(exc).__name__}: {exc}) — "
            f"refusing to spend")


async def _one_round(window: int, min_interval: int, dry_run: bool,
                     max_credits: int, leagues=None):
    from src.data.odds_quota import OddsApiQuota
    from src.data.database import get_db
    from src.scrapers.theodds_scraper import TheOddsScraper
    from src.utils.config import get_config

    config = get_config()
    quota = OddsApiQuota(
        get_db(),
        monthly_budget=int(config.get("odds_api.monthly_credit_budget", 400)),
        safety_margin=int(config.get("odds_api.safety_margin_credits", 50)),
        for_h1=True,
        max_credits_per_run=max_credits,
    )
    scraper = TheOddsScraper(config)
    try:
        return await scraper.refresh_imminent(
            window_minutes=window,
            min_interval_minutes=min_interval,
            quota=None if dry_run else quota,
            # H1 needs repeated observations of the same fixture ACROSS hours,
            # including after its pick is taken, so the picks-run guard would
            # truncate every trajectory at the pick.
            require_pending_pick=False,
            dry_run=dry_run,
            h1_collection=True,
            leagues=leagues,
        ), quota
    finally:
        close = getattr(scraper, "close", None)
        if close:
            try:
                await close()
            except Exception:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rounds", type=int, default=H1_ROUNDS)
    ap.add_argument("--gap-minutes", type=int, default=H1_GAP_MINUTES)
    ap.add_argument("--window", type=int, default=H1_WINDOW_MINUTES)
    ap.add_argument("--min-interval", type=int, default=H1_MIN_INTERVAL_MINUTES)
    ap.add_argument("--max-credits", type=int, required=True,
                    help="per-JOB credit ceiling, stated as a number before "
                         "the run (the trial's cap is 6)")
    ap.add_argument("--leagues", default="",
                    help="comma-separated league keys to PIN. A trajectory is "
                         "three observations of the same fixture, so without a "
                         "pin the credit cap picks a different league each "
                         "round and scatters single observations")
    ap.add_argument("--allow-sqlite", action="store_true",
                    help="accept a local SQLite database deliberately (tests "
                         "and local inspection only; production is Postgres)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print each round's plan; spend nothing")
    args = ap.parse_args(argv)

    _load_env()
    from src.data.database import get_db
    from src.data.odds_quota import h1_reservation
    from src.utils.logger import get_logger

    logger = get_logger()
    db = get_db()
    if not db.is_postgres and not (args.dry_run or args.allow_sqlite):
        print("REFUSING: not connected to the production database.",
              file=sys.stderr)
        return 2

    # The observation window starts when H1's reservation does, NOT at the
    # month reset: counting pre-H1 observations toward `n` would credit the
    # experiment with trajectories it did not collect.
    from src.data.odds_quota import H1_RESERVATION_FROM

    since = datetime(H1_RESERVATION_FROM.year, H1_RESERVATION_FROM.month,
                     H1_RESERVATION_FROM.day)

    print("=" * 78)
    print("H1 COLLECTION — one job, "
          f"{args.rounds} request(s), {args.gap_minutes}-minute in-job waits"
          + ("  [DRY RUN]" if args.dry_run else ""))
    print("=" * 78)
    print(f"  window {args.window} min, min-interval "
          f"{args.min_interval} min (0 = the in-job waits are the only "
          f"spacing; the per-league dedup is writer-agnostic and another "
          f"workflow could otherwise skip a round)")
    print(f"  per-job credit cap  : {args.max_credits}")
    print(f"  reservation today   : {h1_reservation()}")
    # ── WHERE IN THE WINDOW THIS RUN SITS. 2026-10-28. ─────────────────────
    #
    # The operator reading this log needs to know whether a thin day still has
    # days left to recover in, and whether TODAY is the rate decision. Both
    # are facts about the window, so both come from it rather than being
    # counted off a calendar by whoever is reading.
    from src.data import h1_window as _hw
    _today = utcnow().date()
    if _hw.contains(_today):
        print(f"  window day          : {(_today - _hw.FROM).days + 1} of "
              f"{_hw.DAYS}  (last day {_hw.LAST_DAY})")
        if _today == _hw.RATE_DECISION:
            print(f"  RATE DECISION TODAY : {_hw.RATE_DECISION} is the "
                  f"midpoint. If the projection is below the floor of 33 it "
                  f"goes to Niki TODAY, not at the end of the window.")
    else:
        print(f"  window day          : OUTSIDE {_hw.iso()} — collection is "
              f"not due today, and the reservation above says so too")
    print(f"  observations since  : {since:%Y-%m-%d}")
    pinned = [x.strip() for x in args.leagues.split(",") if x.strip()] or None
    print(f"  pinned leagues      : {pinned if pinned else 'ALL (not pinned)'}")

    # ── THE INPUT PRECONDITION, BEFORE ANY SPEND. 2026-10-28. ──────────────
    #
    # IT WAS NEVER WIRED. `h1_precondition.evaluate` existed, had four states
    # and its own tests, and the collection loop did not call it — so the
    # first attempt spent credits on days whose card was 8, 4 and 1 fixtures
    # and skipped even those, and the REASON was never recorded. PNC-1 again:
    # a producer with no consumer looks like a working feature.
    #
    # THE REASON IS CAPTURED HERE, AT THE INSTANT OF THE DECISION, and that is
    # the whole point. `is_fixture` is flipped later by the post-picks scrape,
    # so a reason re-derived afterwards is not recoverable — which is exactly
    # why the first attempt's answer was lost.
    #
    # FOUR STATES, PRINTING DIFFERENTLY, and "nothing to do" is not one of
    # them: NO_CARD (daily-picks has not written the card), NO_FIXTURE_IN_BAND
    # (there is a card and nothing in it can carry three instants), READY, and
    # UNEVALUATED (the question could not be asked, which is NOT a clean
    # precondition).
    from scripts.h1_precondition import (PRECONDITION_STATES, STATE_READY,
                                         evaluate as _pre_evaluate,
                                         format_precondition)
    from src.data.database import get_db as _get_db
    try:
        with _get_db().get_session() as _s:
            pre = _pre_evaluate(_s, pinned or [])
    except Exception as exc:                            # noqa: BLE001
        print(f"  H1_PRECONDITION state=UNEVALUATED — "
              f"{type(exc).__name__}: "
              f"{str(exc).splitlines()[0][:120]}")
        print("  REFUSING to spend: a precondition that could not be "
              "evaluated is not a precondition that passed.")
        return 1
    print()
    print(format_precondition(pre))
    assert pre.state in PRECONDITION_STATES, pre.state
    if pre.state != STATE_READY:
        print(f"  NOT COLLECTING — {pre.state}. No credit claimed, no "
              f"request made. The reason above was captured NOW, before the "
              f"post-picks scrape can flip `is_fixture` and make it "
              f"unrecoverable.")
        return 0

    spent_total = 0
    instants: list[datetime] = []
    for rnd in range(1, args.rounds + 1):
        # ── THE STOP, BEFORE THE SPEND AND BEFORE THE NEXT WAIT ────────────
        stop = _stop_now(since)
        print(f"\n  round {rnd}/{args.rounds}  {utcnow():%H:%M:%S} UTC  "
              f"{stop}")
        if stop.halt:
            print(f"  HALTED before round {rnd} — {stop.state}. "
                  f"{stop.reason}")
            print(f"  Remaining {args.rounds - rnd + 1} round(s) and "
                  f"{(args.rounds - rnd) * args.gap_minutes} minute(s) of "
                  f"waiting SKIPPED — a halted job must not hold a runner.")
            break

        plan, quota = asyncio.run(_one_round(
            args.window, args.min_interval, args.dry_run,
            max(0, args.max_credits - spent_total), leagues=pinned))
        claimed = int(plan.get("credits_claimed") or 0)
        spent_total += claimed
        now = utcnow()
        instants.append(now)
        print(f"      requested {len(plan.get('requested') or [])} league(s), "
              f"credits claimed {claimed}, odds rows "
              f"{plan.get('odds_written')}, job total {spent_total}/"
              f"{args.max_credits}")
        for lg in (plan.get("requested") or []):
            print(f"        -> {lg}")
        if plan.get("h1_stop"):
            print(f"      inner stop: {plan['h1_stop']}")

        if rnd < args.rounds:
            if spent_total >= args.max_credits:
                print(f"  per-job cap reached ({spent_total}/"
                      f"{args.max_credits}) — remaining rounds SKIPPED rather "
                      f"than waited through.")
                break
            # ── THE STOP AGAIN, BETWEEN THE SPEND AND THE WAIT ──────────
            #
            # Checking only at the TOP of the next round is checking it AFTER
            # the wait. Caught by
            # `test_the_stop_halts_MID_SEQUENCE_and_skips_the_remaining_waits`:
            # round 1 exhausted the reservation and the loop slept 120 minutes
            # before noticing. The bound was never at risk — `refresh_imminent`
            # re-checks before every request — but the runner was held for two
            # hours to learn something already known. "Between waits" means
            # HERE.
            mid = _stop_now(since)
            if mid.halt:
                print(f"      HALTED after round {rnd} — {mid.state}. "
                      f"{mid.reason}")
                print(f"      Remaining {args.rounds - rnd} round(s) and "
                      f"{(args.rounds - rnd) * args.gap_minutes} minute(s) of "
                      f"waiting SKIPPED — the wait is not entered.")
                break
            wait_s = args.gap_minutes * 60
            print(f"      waiting {args.gap_minutes} min "
                  f"(resumes ~{(now + timedelta(minutes=args.gap_minutes)):%H:%M} UTC)")
            sys.stdout.flush()
            if not args.dry_run:
                time.sleep(wait_s)

    print()
    print("=" * 78)
    print("JOB SUMMARY")
    print("=" * 78)
    print(f"  rounds executed : {len(instants)}")
    print(f"  credits spent   : {spent_total}/{args.max_credits}")
    if len(instants) >= 2:
        gaps = [round((instants[i + 1] - instants[i]).total_seconds() / 60, 1)
                for i in range(len(instants) - 1)]
        print(f"  ACTUAL request gaps (min) : {gaps}")
    print(f"  final stop      : {_stop_now(since)}")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
