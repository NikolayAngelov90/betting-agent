"""READ-ONLY fixture-horizon probe: tomorrow's earliest kickoff, for the margin.

    python -m scripts.fixture_horizon_probe            # probe + prospective check
    python -m scripts.fixture_horizon_probe --report-only

WHY IT EXISTS. The prospective schedule margin needs tomorrow's earliest
kickoff, and the database cannot supply it. `max_days_ahead=1` is a **rolling
24 hours from the moment of the run**, not the next calendar day: measured
2026-10-10, the production table held 117 fixtures for that day and **ZERO for
the next**, the furthest `match_date` in the whole table was 19:30 that evening,
and **no fixture has ever been known more than 14.6 hours before kickoff**.

So the input is fetched instead — and fetched WITHOUT touching the discovered
fixture population, because widening `max_days_ahead` in the pipeline would
change which fixtures enter the card and that is a cohort event.

WHAT IT MUST NOT DO. It persists nothing. `_scrape_fixtures_page` returns parsed
dicts and this module never calls a store, never opens a write session, and
never passes its rows to anything that does. Proved by observation in
`tests/test_fixture_horizon_probe.py`, which counts rows in every table before
and after and carries a positive control that fails when a write is introduced.

WINDOW: **2 DAYS**, which is the requirement. One day ahead of the run is about
two days from the probe, and the 8-day window used for the density
demonstration on 2026-10-10 was four times what the margin needs.

ALL 30 LEAGUES, and that is not negotiable: the deadline is a MINIMUM, so any
league can set it. Probing a subset would substitute an assumption about which
leagues can bind for a measurement of which do.

THE TIMEZONE IS ESTABLISHED, NEVER ASSUMED — and this is the fifth instance of
that rule in three days. Flashscore renders in the BROWSER's timezone. The
2026-10-10 density probe ran on a Europe/Kyiv machine and returned UTC+3; a
GitHub runner is UTC and returns +0. An assumed offset is how the H1 collection
cron was set three hours late. So this probe CROSS-CHECKS itself against the
production `matches` table, which stores UTC, and **refuses to emit a deadline
unless the offset it measures is zero** — anything else is reported as UNKNOWN
rather than corrected by a guessed shift.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
import time
from typing import Optional

#: The requirement, not the demonstration. One day ahead of the run is about
#: two days from the probe.
PROBE_DAYS_AHEAD = 2

#: Tolerance on the self-verified timezone offset, in minutes. Flashscore
#: renders to the minute and kickoffs are on 15-minute boundaries, so a
#: non-zero offset is a timezone, not noise.
OFFSET_TOLERANCE_MIN = 5


def _load_env() -> None:
    """`.env` from main() only — an import-time load re-introduces DATABASE_URL
    after conftest strips it, which once let a unit test write to production."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


def probe_fixtures(leagues, days_ahead: int = PROBE_DAYS_AHEAD) -> dict:
    """`{league: [datetime, ...]}` as the PAGES render them. Persists nothing.

    Returns the raw parsed kickoffs in the browser's own timezone. Converting
    them is `verify_offset`'s job, and it refuses rather than guesses.
    """
    from src.scrapers.flashscore_scraper import (
        FLASHSCORE_BASE_URL, FlashscoreScraper,
    )
    from src.utils.config import get_config
    from src.utils.logger import get_logger

    logger = get_logger()
    scraper = FlashscoreScraper(get_config())
    out, pages, t0 = {}, 0, time.time()
    for league in leagues:
        url = f"{FLASHSCORE_BASE_URL}/football/{league}/fixtures/"
        try:
            rows = scraper._scrape_fixtures_page(url, max_days_ahead=days_ahead)
            pages += 1
        except Exception as exc:
            logger.warning(f"probe: {league} FAILED ({type(exc).__name__}: {exc})")
            out[league] = None          # None is NOT an empty list
            continue
        kos = []
        for r in rows:
            d = r.get("match_date")
            if isinstance(d, str):
                try:
                    d = _dt.datetime.fromisoformat(d)
                except ValueError:
                    continue
            if isinstance(d, _dt.datetime):
                kos.append(d)
        out[league] = sorted(kos)
    out["__meta__"] = {
        "pages_fetched": pages,
        "leagues_asked": len(leagues),
        "wall_seconds": round(time.time() - t0, 1),
        "days_ahead": days_ahead,
        "refused": sorted(k for k, v in out.items()
                          if v is None and k != "__meta__"),
    }
    return out


def verify_offset(session, probed: dict) -> Optional[int]:
    """Minutes the probe's clock sits ahead of UTC, or `None` if unverifiable.

    Cross-checked against the production `matches` table, which stores UTC, on
    fixtures the probe and the database BOTH hold. `None` means the offset could
    not be established — which is not zero, and the caller must not emit a
    deadline on it.
    """
    from sqlalchemy import text

    from scripts.schedule_deadline import as_datetime

    deltas = []
    for league, kos in probed.items():
        if league == "__meta__" or not kos:
            continue
        for ko in kos[:4]:
            rows = session.execute(text("""
                SELECT m.match_date FROM matches m
                WHERE m.league = :lg
                  AND m.is_fixture = true
                  AND m.match_date BETWEEN :lo AND :hi
            """), {"lg": league,
                   "lo": ko - _dt.timedelta(hours=13),
                   "hi": ko + _dt.timedelta(minutes=1)}).fetchall()
            # Phantoms filtered HERE, not in SQL: EXTRACT(MICROSECONDS ...) is
            # PostgreSQL-only and would raise on SQLite.
            for (raw,) in rows:
                db_ko = as_datetime(raw)
                if db_ko is None or db_ko.microsecond:
                    continue
                delta = round((ko - db_ko).total_seconds() / 60.0)
                # Only whole-hour deltas are timezones; anything else is a
                # different fixture and is not evidence either way.
                if delta % 60 == 0 and -720 <= delta <= 720:
                    deltas.append(delta)
    if not deltas:
        return None
    return max(set(deltas), key=deltas.count)      # the modal offset


def earliest_for(probed: dict, day: _dt.date,
                 offset_min: int) -> Optional[int]:
    """Earliest kickoff on `day` IN UTC MINUTES, or `None`."""
    best = None
    for league, kos in probed.items():
        if league == "__meta__" or not kos:
            continue
        for ko in kos:
            utc = ko - _dt.timedelta(minutes=offset_min)
            if utc.date() == day:
                m = utc.hour * 60 + utc.minute
                best = m if best is None else min(best, m)
    return best


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--days-ahead", type=int, default=PROBE_DAYS_AHEAD)
    ap.add_argument("--report-only", action="store_true",
                    help="print the probe and the margin; never exit non-zero")
    args = ap.parse_args(argv)

    _load_env()
    from scripts.ci_audit import (
        MARGIN_RUN_DURATIONS, collect_lag_series_by_daytype,
    )
    from scripts.schedule_deadline import (
        CLEARING_ACTION, day_type, format_prospective, prospective_margin,
    )
    from src.data.database import get_db
    from src.utils.config import get_config

    cfg = get_config()
    leagues = [l for l in (cfg.get("scraping.flashscore_leagues", []) or [])
               if l not in set(cfg.get("scraping.flashscore_skip_leagues", []) or [])]
    print("=" * 78)
    print(f"FIXTURE HORIZON PROBE - {len(leagues)} leagues, "
          f"{args.days_ahead} day(s) ahead, READ-ONLY")
    print("=" * 78)

    probed = probe_fixtures(leagues, args.days_ahead)
    meta = probed["__meta__"]
    print(f"  pages fetched : {meta['pages_fetched']}/{meta['leagues_asked']}")
    print(f"  wall time     : {meta['wall_seconds']}s")
    if meta["refused"]:
        print(f"  REFUSED       : {len(meta['refused'])} league(s) "
              f"{meta['refused']}")
    else:
        print(f"  REFUSED       : 0")

    db = get_db()
    tomorrow = _dt.date.today() + _dt.timedelta(days=1)
    with db.get_session() as session:
        offset = verify_offset(session, probed)
    if offset is None:
        print("  timezone      : UNVERIFIABLE - no fixture is held by both the "
              "probe and the database")
    else:
        print(f"  timezone      : probe clock is {offset:+d} min from UTC "
              f"(verified against `matches`, which stores UTC)")

    if offset is None or abs(offset) > OFFSET_TOLERANCE_MIN:
        print()
        print(f"  prospective_margin[serves={tomorrow} UNKNOWN: the probe's "
              f"timezone is "
              + ("unverifiable" if offset is None else f"{offset:+d} min, not UTC")
              + " - NOT safe]")
        print("  A deadline is NOT emitted on an unestablished clock. Setting "
              "TZ=UTC in the job is what fixes this; a guessed shift is how the "
              "H1 cron was set three hours late.")
        return 0 if args.report_only else 1

    deadline = earliest_for(probed, tomorrow, offset)
    by_type, crons = collect_lag_series_by_daytype()
    dtp = day_type(tomorrow)
    delays = [lag for _d, lag in (by_type.get(dtp) or [])]
    m = prospective_margin(deadline, crons.get(dtp), delays,
                           MARGIN_RUN_DURATIONS, serves=tomorrow)
    print()
    print("  " + format_prospective(m))
    if m["verdict"] == "ALARM":
        print()
        print(f"::error::PROSPECTIVE schedule margin {m['binding']:+.0f}m for "
              f"{m['serves']} (probed kickoff "
              f"{m['deadline_min'] // 60:02d}:{m['deadline_min'] % 60:02d}) is "
              f"below {m['min_margin']}m. {CLEARING_ACTION}")
        return 0 if args.report_only else 1
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
