"""Refresh imminent odds, then capture closing lines — one scheduled run.

    python -m scripts.refresh_and_capture [--dry-run] [--window 120]
                                          [--min-interval 180] [--status]

Stage 6, Phases 8-10. This is the only job the closing-line experiment needs on
a schedule.

Why refresh and capture must share a process
--------------------------------------------
``capture_closing_lines`` reads odds already in the database and
``clv.validate_pair`` rejects any snapshot taken more than 180 minutes before
kickoff. The daily 09:37 UTC pipeline leaves odds 8-10 hours stale by an evening
kickoff, so every capture would be rejected: the experiment could run forever and
collect nothing. Refreshing immediately before capturing is what makes a stored
price a *closing* price.

What it costs
-------------
Only leagues with a fixture kicking off inside the window AND a pending pick
awaiting a closing line are refreshed, subject to a per-league minimum interval
and a hard monthly credit budget claimed before any HTTP call. Measured on
production history: ~256 credits/month at 2-hourly runs with a 120-minute
window, against a 500-credit free tier.

Safety
------
* ``--dry-run`` spends nothing and prints the exact plan.
* The budget is claimed BEFORE the request, so a crash cannot overspend.
* Nothing here relaxes closing-line validation. If odds are not fresh enough,
  the capture records ``missing``/``late`` and CLV coverage stays honestly low —
  which is the correct outcome, not a problem to engineer around.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Optional


def _load_env() -> None:
    """Load .env from main() only — never at import.

    An import-time load re-introduces DATABASE_URL after conftest strips it,
    which once let a SQLite unit test write to production.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


async def _run(window: int, min_interval: int, dry_run: bool,
               require_pending_pick: bool, h1: bool = False,
               max_credits: int = 0) -> int:
    from src.data.database import get_db
    from src.data.odds_quota import OddsApiQuota
    from src.scrapers.theodds_scraper import TheOddsScraper
    from src.utils.config import get_config
    from src.utils.logger import get_logger

    logger = get_logger()
    config = get_config()
    db = get_db()

    quota = OddsApiQuota(
        db,
        monthly_budget=int(config.get("odds_api.monthly_credit_budget", 400)),
        safety_margin=int(config.get("odds_api.safety_margin_credits", 50)),
        # H1 SPENDS ITS OWN RESERVATION, so the 168 is not withheld from it.
        # Its bound is `collection_stop`, enforced inside `refresh_imminent`
        # before any request — not this ledger.
        for_h1=h1,
        # THE TRIAL'S CAP. A per-run ceiling below the reservation, so a loop
        # defect cannot consume the window on day one. 0 keeps the default.
        max_credits_per_run=max_credits if max_credits else 24,
    )
    print(quota.describe())
    if h1:
        from src.data.odds_quota import h1_reservation
        print(f"  H1 COLLECTION: reservation {h1_reservation()} credits, "
              f"per-run cap {quota.max_credits_per_run}")

    scraper = TheOddsScraper(config)
    try:
        plan = await scraper.refresh_imminent(
            window_minutes=window,
            min_interval_minutes=min_interval,
            quota=None if dry_run else quota,
            require_pending_pick=require_pending_pick,
            dry_run=dry_run,
            h1_collection=h1,
        )
    finally:
        close = getattr(scraper, "close", None)
        if close:
            try:
                await close()
            except Exception:
                pass

    print()
    print("=" * 80)
    print("ODDS REFRESH PLAN" + ("  [DRY RUN — no credits spent]" if dry_run else ""))
    print("=" * 80)
    print(f"  window            : {plan['window_minutes']} min")
    print(f"  min interval      : {plan['min_interval_minutes']} min")
    print(f"  candidate leagues : {len(plan['candidates'])}")
    for c in plan["candidates"]:
        print(f"      {c['league']:<40}{c['fixtures']:>3} fixture(s), "
              f"next {c['next_kickoff']}")
    print(f"  WOULD REQUEST     : {len(plan['requested'])} league(s) "
          f"= {plan['credits_estimated']} credits")
    for lg in plan["requested"]:
        print(f"      -> {lg}")
    if plan["skipped"]:
        print(f"  skipped           : {len(plan['skipped'])}")
        for lg, why in sorted(plan["skipped"].items()):
            print(f"      {lg or '(no league)':<40}{why}")
    if not dry_run:
        print(f"  odds rows written : {plan['odds_written']}")
        print(f"  credits claimed   : {plan['credits_claimed']}")

    if dry_run:
        print("\n  Dry run complete — no API call was made, no credit spent.")
        return 0

    # ---- capture immediately, while the odds we just wrote are freshest ----
    from scripts.capture_closing_lines import capture

    cap_window = max(window, 30)
    stats = capture(within_minutes=cap_window, dry_run=False)
    print()
    print("=" * 80)
    print("CLOSING CAPTURE")
    print("=" * 80)
    print(f"  considered : {stats['considered']}")
    print(f"  captured   : {stats['captured']}")
    print(f"  missing    : {stats['missing']}")
    print(f"  late       : {stats['late']}")
    print(f"  invalid    : {stats['invalid']}")
    print(f"  cost       : {stats['db_queries']} db queries, "
          f"{stats['odds_rows_read']} odds rows, {stats['elapsed_s']}s")
    print()
    print(quota.describe())
    return 0


def main() -> None:
    _load_env()
    from src.utils.config import get_config

    cfg = get_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=None,
                    help="minutes ahead to look for imminent fixtures "
                         "(default: betting.odds_refresh_window_minutes)")
    ap.add_argument("--min-interval", type=int, default=None,
                    help="minimum minutes between refreshes of the same league "
                         "(default: betting.odds_refresh_min_interval_minutes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan; make no API call and spend no credit")
    ap.add_argument("--any-fixture", action="store_true",
                    help="refresh imminent fixtures even without a pending pick "
                         "(costs more credits; off by default)")
    ap.add_argument("--status", action="store_true",
                    help="print the quota ledger and CLV coverage, then exit")
    # ── H1 COLLECTION (Stage 29, 2026-10-09) ────────────────────────────────
    #
    # THE CALLER. The stop, the allocator, the reservation and five terminal
    # states shipped across three stages, all green, and NOTHING CALLED THEM.
    # PNC-1 at the top level: a producer with no consumer looks exactly like a
    # working feature, and the more thoroughly the producer is tested the more
    # convincingly it does.
    #
    # THE REGISTERED VALUES ARE PASSED HERE, NEVER DEFAULTED — that was the
    # registration's own condition and it is what makes the revert a removal of
    # two arguments rather than an edit to a constant.
    ap.add_argument("--h1", action="store_true",
                    help="H1 collection: window 360, min-interval 120, the "
                         "enforced stop ON. Spends the H1 reservation.")
    ap.add_argument("--max-credits", type=int, default=0,
                    help="per-run credit ceiling (H1 trial: state it as a "
                         "number so a loop defect cannot consume the window)")
    args = ap.parse_args()

    if args.status:
        from src.data.database import get_db
        from src.data.odds_quota import OddsApiQuota
        from scripts.capture_closing_lines import print_coverage

        db = get_db()
        print(OddsApiQuota(
            db,
            monthly_budget=int(cfg.get("odds_api.monthly_credit_budget", 400)),
            safety_margin=int(cfg.get("odds_api.safety_margin_credits", 50)),
        ).describe())
        print()
        print_coverage()
        return

    # ── H1's REGISTERED VALUES, PASSED EXPLICITLY ───────────────────────────
    #
    # THE REVERT PLAN, shipped in the same commit:
    #   1. stop passing `--h1` to `refresh_and_capture`
    #   2. that is the whole revert. The values live HERE, in the `--h1` branch,
    #      and the defaults below are the NORMAL ones — so removing the flag
    #      restores normal pricing without editing a constant.
    #   3. leave TARGET_N, CREDIT_CEILING, H1_RESERVED_CREDITS and the window
    #      pair untouched; the registration fixes them, and a revert that edits
    #      them is a new experiment wearing the old one's name.
    #
    # WHY `require_pending_pick=False` FOR H1, and what it costs: the guard
    # exists so a refresh cannot rewrite a price a pending picks run will read.
    # H1 needs REPEATED observations of the same fixture across hours, including
    # after its pick is taken, so the guard would truncate every trajectory at
    # the pick. Setting it False is therefore required — and the exposure is
    # real: a refresh between discovery and picks could move a price the picks
    # run then reads. It is acceptable here only because the H1 window (360) and
    # the picks run (00:00 cron, delivered ~04:30) do not overlap in practice —
    # measured 2026-10-09, the daily-picks runs land 03:58-05:04 and H1's slots
    # are the 2-hourly closing-lines times.
    if args.h1:
        window = args.window if args.window is not None else 360
        min_interval = args.min_interval if args.min_interval is not None else 120
        require_pending_pick = False
    else:
        window = args.window if args.window is not None else int(
            cfg.get("betting.odds_refresh_window_minutes", 120))
        min_interval = args.min_interval if args.min_interval is not None else int(
            cfg.get("betting.odds_refresh_min_interval_minutes", 180))
        require_pending_pick = not args.any_fixture

    raise SystemExit(asyncio.run(_run(
        window, min_interval, args.dry_run,
        require_pending_pick=require_pending_pick,
        h1=args.h1, max_credits=args.max_credits)))


if __name__ == "__main__":
    main()
