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
               require_pending_pick: bool) -> int:
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
    )
    print(quota.describe())

    scraper = TheOddsScraper(config)
    try:
        plan = await scraper.refresh_imminent(
            window_minutes=window,
            min_interval_minutes=min_interval,
            quota=None if dry_run else quota,
            require_pending_pick=require_pending_pick,
            dry_run=dry_run,
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

    window = args.window if args.window is not None else int(
        cfg.get("betting.odds_refresh_window_minutes", 120))
    min_interval = args.min_interval if args.min_interval is not None else int(
        cfg.get("betting.odds_refresh_min_interval_minutes", 180))

    raise SystemExit(asyncio.run(_run(
        window, min_interval, args.dry_run,
        require_pending_pick=not args.any_fixture)))


if __name__ == "__main__":
    main()
