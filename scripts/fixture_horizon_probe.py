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
import os as _os
import pathlib
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


# ── EXIT CODES: A DESIGNED REFUSAL IS NOT A CRASH ───────────────────────────
#
# Until 2026-10-12 all three exited 1, and the workflow's `if: failure()` step
# fired one fixed sentence for every one of them. The first and only probe run
# crashed on a missing dependency and sent the operator "Tomorrow's picks may
# land AFTER the first kickoff" - a MARGIN FINDING - from a job that had
# computed no margin at all. The text carried a disjunction ("if instead the
# probe refused...") that the reader had no way to resolve.
#
#     A designed refusal must be distinguishable from a crash, in the exit
#     status and in the log.
#
# Same taxonomy as `ci_audit`, one code wider, because this tool has a fourth
# outcome that tool does not: a legitimate UNKNOWN.
EXIT_CLEAR = 0        # a deadline was discovered and the margin is adequate
EXIT_ALARM = 1        # a deadline was discovered and the margin is SHORT
EXIT_CRASH = 2        # MACHINERY. Nothing was probed and nothing is known.
EXIT_REFUSED = 3      # BY DESIGN: the clock is unverified, or a league is None

#: Printed so the log is self-classifying too, not only the exit status.
MARKER = {
    EXIT_ALARM: "PROBE ALARM",
    EXIT_CRASH: "PROBE CRASHED",
    EXIT_REFUSED: "PROBE REFUSED",
}

#: THE ANNOTATION PREFIXES, AS LITERALS. `ci_audit.SELF_EMITTED_ANNOTATIONS`
#: pins each registered pattern to the file that prints it, and a test asserts
#: the emitter CONTAINS the literal — so building these with an f-string makes
#: the registry "drift into guessing from the text", which is what that test
#: says and what it caught here on 2026-10-12.
ANNOTATION_ALARM = "::error::PROBE ALARM"
ANNOTATION_REFUSED = "::error::PROBE REFUSED"

#: Time from run start to picks complete on a FULL card.
#: MEASURED 2026-10-01, n=4: 67.4-88.1 min, mean 76.4, sd 10.2.
#: THE UPPER BOUND, never the mean - a dispatch sized on 76 minutes misses a
#: card that takes 88, and the whole point of the figure is to not be late.
PICKS_UPPER_BOUND_MIN = 88.1

#: Subtracted on top of the upper bound. Stated as a number rather than folded
#: into the bound, so the two are separable: one is measured, one is a choice.
DISPATCH_SAFETY_MARGIN_MIN = 15

OPERATOR_TZ = "Europe/Sofia"


def _in_tz(day, minutes_utc: int, tz_name: str = OPERATOR_TZ):
    """`(HH:MM, label)` in `tz_name`, or `(None, reason)`.

    EVERY TIME NAMES ITS CLOCK. Five instances of "establish the clock" rest
    on a single ledger site, and an alert printing a bare hour is the next one.
    """
    import datetime as _d
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(tz_name)
    except Exception as exc:                        # noqa: BLE001
        return None, f"{tz_name} unavailable ({type(exc).__name__})"
    aware = _d.datetime.combine(
        day, _d.time(minutes_utc // 60, minutes_utc % 60),
        tzinfo=_d.timezone.utc).astimezone(z)
    return aware.strftime("%H:%M"), aware.strftime("%Z%z")


def earliest_league_for(probed: dict, day, offset_min: int):
    """The LEAGUE holding the earliest kickoff, so the alert can name it.

    The fixture itself is NOT available: `probe_fixtures` returns kickoff
    datetimes only, so naming home-vs-away would need the scrape to carry team
    names through. Stated rather than guessed - see the ledger entry for what
    that would take.
    """
    import datetime as _d
    best, best_league = None, None
    for league, kos in probed.items():
        if league == "__meta__" or not kos:
            continue
        for ko in kos:
            utc = ko - _d.timedelta(minutes=offset_min)
            if utc.date() != day:
                continue
            m = utc.hour * 60 + utc.minute
            if best is None or m < best:
                best, best_league = m, league
    return best_league


def format_dispatch_alert(day, deadline_min, league, basis: str,
                          margin=None, run_url: str = "") -> str:
    """The alert body, WITH THE TWO NUMBERS the operator had to work out.

    Before this, the text said "dispatch daily-picks manually tomorrow
    morning" and left the reader to find the kickoff and subtract the runtime.
    """
    hhmm = f"{deadline_min // 60:02d}:{deadline_min % 60:02d}"
    local, label = _in_tz(day, deadline_min)
    last = int(deadline_min - PICKS_UPPER_BOUND_MIN
               - DISPATCH_SAFETY_MARGIN_MIN)
    day_before = last < 0
    last %= 24 * 60
    last_hhmm = f"{last // 60:02d}:{last % 60:02d}"
    last_local, last_label = _in_tz(day, last)

    lines = [
        f"PICKS MAY LAND AFTER THE FIRST KICKOFF on {day} ({basis}).",
        "",
        f"EARLIEST KICKOFF  {hhmm} UTC"
        + (f" / {local} {label}" if local else f" / {label}")
        + (f"   league: {league}" if league else ""),
        f"LAST SAFE DISPATCH {last_hhmm} UTC"
        + (f" / {last_local} {last_label}" if last_local else "")
        + (" (THE DAY BEFORE)" if day_before else ""),
        f"  = kickoff - {PICKS_UPPER_BOUND_MIN:.0f}m time-to-picks "
        f"(MEASURED upper bound, n=4, 67.4-88.1, NOT the mean) "
        f"- {DISPATCH_SAFETY_MARGIN_MIN}m safety margin",
    ]
    if margin is not None:
        lines.append(f"  projected margin {margin:+.0f}m")
    lines += [
        "",
        "ACTION: gh workflow run daily-picks.yml -- before the time above. A "
        "dispatched run skips the measured 3h58m-5h04m scheduler lateness and "
        "is the ONLY action that changes this margin. It is the operator's; "
        "there is no automated alternative today.",
    ]
    if run_url:
        lines.append(f"Logs: {run_url}")
    return chr(10).join(lines)


# ── WHICH REFUSAL IS IT? THE PROVENANCE, MEASURED. 2026-10-18. ──────────────
#
# `verify_offset` returns None for two different worlds and the log said
# `UNVERIFIABLE` for both:
#
#   the scrape ran and wrote rows the verifier could not match
#   the scrape had not run, so the population could not exist yet
#
# While the second is ALWAYS true the conflation is harmless - the refusal is
# a constant. It stops being a constant the moment the ordering can invert.
#
#     An always-empty verifier is honest. An intermittently-populated one is
#     worse, because its silence and its answer become indistinguishable in
#     provenance - unless the output says which.
#
# MEASURED 2026-10-18 over all nine probe runs: the ordering was INVERTED in
# 9 of 9, separation +7.33h to +33.49h against the target day's scrape
# completion, so ZERO runs fell below the 6.0h floor and the intermittency has
# not started. But the projection has thinned: with the probe's observed
# maximum lateness (18:00 -> 00:21) against the weekday scrape at 00:00 plus
# its observed 256-305 min, the worst case is **+3.92h** - and pre-revert it
# was already **+4.37h**, so the nominal 10h -> 6h change moved the real
# worst case by 27 minutes, not four hours. The nominal gap overstated the
# margin in both regimes.
#
# So the ordering is now OBSERVED, from the database, not derived from a cron.
ORDERING_POPULATION_EXISTS = "POPULATION_EXISTS"
ORDERING_INVERTED = "ORDERING_INVERTED"
ORDERING_UNKNOWN = "ORDERING_UNKNOWN"

ORDERING_STATES = (ORDERING_POPULATION_EXISTS, ORDERING_INVERTED,
                   ORDERING_UNKNOWN)

#: The two refusals, printing differently, so a reader never has to infer
#: which world produced a `None`.
REFUSAL_EMPTY_POPULATION = "UNVERIFIABLE: EMPTY_POPULATION"
REFUSAL_ORDERING_INVERTED = "UNVERIFIABLE: ORDERING_INVERTED"


def observed_ordering(session, day) -> tuple:
    """`(state, detail)` for `day`, MEASURED from the rows themselves.

    The question is not "what do the crons say" - the crons are nominal and
    both jobs run hours late. It is "had the scrape written this day's
    population when the probe looked", and `matches.updated_at` answers it.
    """
    from sqlalchemy import text

    from scripts.schedule_deadline import as_datetime
    try:
        row = session.execute(text(
            "SELECT count(*) AS n, max(updated_at) AS last_write, "
            "       sum(CASE WHEN is_fixture THEN 1 ELSE 0 END) AS fixtures "
            "FROM matches WHERE match_date >= :a AND match_date < :b"),
            {"a": day.isoformat(),
             "b": (day + _dt.timedelta(days=1)).isoformat()}).mappings().first()
    except Exception as exc:                        # noqa: BLE001
        return ORDERING_UNKNOWN, (f"the ordering could not be measured "
                                  f"({type(exc).__name__}) - that is not an "
                                  f"ordering, it is an absence of evidence")
    n = int(row["n"] or 0)
    fx = int(row["fixtures"] or 0)
    last = as_datetime(row["last_write"])
    if n == 0:
        return ORDERING_INVERTED, (
            f"no row exists for {day} at all, so the scrape that writes this "
            f"day's population has NOT run yet. The probe reached the question "
            f"before the answer could exist - this is ORDERING, not an empty "
            f"result.")
    sep = ""
    if last is not None:
        sep = (f", last written {last:%Y-%m-%d %H:%M} UTC, "
               f"{(_dt.datetime.now(_dt.timezone.utc).replace(tzinfo=None) - last).total_seconds() / 3600.0:+.2f}h "
               f"before this probe")
    return ORDERING_POPULATION_EXISTS, (
        f"{n} row(s) for {day}, {fx} of them fixtures{sep}. The population "
        f"EXISTS, so a failure to match is about the match, not the ordering.")


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
        ordering, ord_detail = observed_ordering(session, tomorrow)
    print(f"  ordering      : {ordering} (MEASURED from the rows, not from "
          f"the cron)")
    print(f"                  {ord_detail}")
    if offset is None:
        print("  timezone      : UNVERIFIABLE - no fixture is held by both the "
              "probe and the database")
    else:
        print(f"  timezone      : probe clock is {offset:+d} min from UTC "
              f"(verified against `matches`, which stores UTC)")

    if offset is None or abs(offset) > OFFSET_TOLERANCE_MIN:
        # WHICH REFUSAL. A `None` offset has two causes and they are not the
        # same finding: one is about the clock, the other about the schedule.
        if offset is None:
            which = (REFUSAL_ORDERING_INVERTED
                     if ordering == ORDERING_INVERTED
                     else REFUSAL_EMPTY_POPULATION
                     if ordering == ORDERING_POPULATION_EXISTS
                     else "UNVERIFIABLE: ORDERING_UNKNOWN")
        else:
            which = f"UNVERIFIABLE: CLOCK_OFFSET {offset:+d}min"
        print()
        print(f"  prospective_margin[serves={tomorrow} {which} - NOT safe]")
        print(f"  ordering basis: {ord_detail}")
        print("  A deadline is NOT emitted on an unestablished clock. Setting "
              "TZ=UTC in the job is what fixes this; a guessed shift is how the "
              "H1 cron was set three hours late.")
        print(f"{ANNOTATION_REFUSED} [{which}]: the clock is not established, so the "
              f"margin is UNKNOWN and NOT safe. This is a DESIGNED REFUSAL, "
              f"not a crash and not a finding. CLEARING ACTION: confirm TZ=UTC "
              f"on the runner; the probe verifies rather than trusting it.")
        return EXIT_CLEAR if args.report_only else EXIT_REFUSED

    # AN INVERTED ORDERING MUST NEVER SILENTLY PRODUCE A DEADLINE.
    #
    # I WROTE "it cannot reach here today" AND IT REACHED HERE ON THE NEXT
    # RUN. Run 39131837165, 2026-10-18 13:31, minutes after this guard
    # shipped:
    #
    #     timezone : probe clock is +0 min from UTC (verified against
    #                `matches`, which stores UTC)
    #     ordering : ORDERING_INVERTED - no row exists for 2026-10-19
    #
    # `verify_offset` SUCCEEDED for the first time in the probe's life,
    # because the morning's dispatch had populated TODAY's card and the
    # rolling window overlaps it - so the clock verified against today while
    # the deadline would have been computed for TOMORROW, whose population
    # does not exist. The old code would have emitted a deadline from the
    # probe's pages with nothing to check them against, and it would have
    # looked exactly like a real one.
    #
    # The two conditions are INDEPENDENT: the clock is verified against
    # whatever day overlaps, and the deadline is computed for the target day.
    # Assuming `offset is None` covers the absent-population case was the
    # error, and "it cannot happen" is how the H1 cron was set three hours
    # wrong.
    if ordering != ORDERING_POPULATION_EXISTS:
        print()
        print(f"{ANNOTATION_REFUSED} [{REFUSAL_ORDERING_INVERTED}]: the clock "
              f"verified ({offset:+d} min) but the population for {tomorrow} "
              f"does not exist, so any deadline would be computed from the "
              f"probe's pages with nothing to check them against. "
              f"{ord_detail}")
        return EXIT_CLEAR if args.report_only else EXIT_REFUSED

    deadline = earliest_for(probed, tomorrow, offset)
    by_type, crons = collect_lag_series_by_daytype()
    dtp = day_type(tomorrow)
    delays = [lag for _d, lag in (by_type.get(dtp) or [])]
    m = prospective_margin(deadline, crons.get(dtp), delays,
                           MARGIN_RUN_DURATIONS, serves=tomorrow)
    print()
    print("  " + format_prospective(m))
    if m["verdict"] == "UNKNOWN":
        print()
        print(f"{ANNOTATION_REFUSED}: no deadline could be discovered for "
              f"{tomorrow} - a league returned None, or the card is not yet "
              f"published. The margin is UNKNOWN and NOT safe. DESIGNED "
              f"REFUSAL. CLEARING ACTION: none today; the card for a day is "
              f"not discoverable earlier than the provider publishes it.")
        return EXIT_CLEAR if args.report_only else EXIT_REFUSED

    if m["verdict"] == "ALARM":
        league = earliest_league_for(probed, tomorrow, offset)
        body = format_dispatch_alert(
            tomorrow, m["deadline_min"], league,
            basis=f"deadline DISCOVERED by the probe; ordering {ordering}",
            margin=m["binding"],
            run_url=_os.environ.get("PROBE_RUN_URL", ""))
        print()
        print(f"{ANNOTATION_ALARM}: a discovered deadline, and the "
              f"margin is {m['binding']:+.0f}m against {m['min_margin']}m. "
              f"This is a FINDING, not a malfunction. {CLEARING_ACTION}")
        print(body)
        _write_alert_body(body)
        return EXIT_CLEAR if args.report_only else EXIT_ALARM
    return EXIT_CLEAR


def _write_alert_body(body: str) -> None:
    """Hand the rendered body to the workflow, so the ALERT carries the
    numbers rather than a fixed sentence written months earlier."""
    path = _os.environ.get("PROBE_ALERT_BODY")
    if not path:
        return
    try:
        pathlib.Path(path).write_text(body, encoding="utf-8")
    except OSError as exc:                          # pragma: no cover
        print(f"could not write the alert body: {exc}")


def _main_guarded(argv=None) -> int:
    """THE CRASH BOUNDARY. Exit 2 is machinery and says so.

    The first and only probe run died on `ModuleNotFoundError: aiohttp` and
    the workflow sent a margin finding. A crash must announce itself as one.
    """
    try:
        return main(argv)
    except SystemExit:
        raise
    except BaseException as exc:                    # noqa: BLE001
        import traceback
        traceback.print_exc()
        body = (f"{MARKER[EXIT_CRASH]} ({type(exc).__name__}): this is "
                f"MACHINERY, not a finding. NOTHING was probed, so NOTHING is "
                f"known about tomorrow's margin - do not read the absence of "
                f"a deadline as a safe one. CLEARING ACTION: fix the probe; "
                f"until it runs, the prospective check is UNEVALUATED.")
        print(body)
        _write_alert_body(body)
        return EXIT_CRASH


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(_main_guarded())
