"""Fixtures that are UNPRICED, which is indistinguishable from UNWANTED.

THE DEFECT CLASS. A fixture with no odds produces no pick — and so does a
fixture nobody wanted to bet. The two are identical in every output the pipeline
produces, which is the shape every defect in this project has had: a failure
presenting as a plausible normal state rather than as an error.

MEASURED 2026-08-31, the two known causes, both from ONE wrong integer each:

  * Leuven vs St. Liege, 2026-08-29 — the Stage 20 identity-gate regression
    refused the API-Football path. The row survived via Flashscore and carried
    ZERO odds while its three same-day league peers carried 77, 79 and 175.
  * Every Cracovia fixture — row 411 ("Rakow") holds API-Football id 350, which
    is Cracovia's, so row 420 ("Cracovia") can never adopt it. 12 of 12 August
    fixtures carried zero odds.

THE CHECK IS SELF-CALIBRATING, with no threshold and no maintained league list:
a fixture is compared against its OWN same-league, same-day peers. Zero odds
where peers are priced is an anomaly; zero odds where nobody is priced is a
quiet league, an off-season, or a competition the provider does not cover.

WHY THE PEER COMPARISON RATHER THAN "has a Flashscore id but no API-Football
id". That narrower signature describes the Leuven case exactly and MISSES 11 of
the 12 known Cracovia fixtures, which carry neither id. The id columns are
reported as context because they name the likely cause; they are not the trigger,
because the trigger has to catch the class rather than the instance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()

#: Emitted once per affected fixture. `ci_audit.py` greps this exact prefix.
ALARM_PREFIX = "UNPRICED FIXTURE"


def find_unpriced_fixtures(session, day_from, day_to) -> Optional[List[Tuple]]:
    """Fixtures with zero odds whose same-league, same-day peers have odds.

    Returns (match_id, league, kickoff, home, away, has_af_id, has_fs_id,
    peers_priced, peer_median_odds), newest first. Never raises: a coverage
    check that breaks a run is worse than the gap it reports.

    RETURNS None WHEN THE QUERY ITSELF FAILED, which is NOT the same as an
    empty list, and the distinction was earned. The first version returned []
    on any exception; run against a dead credential it printed
    "2026-08-29: 0 unpriced" — a reassuring number produced by a check that had
    not run. A check that reports "nothing wrong" when it is broken is worse
    than no check, because it is trusted. Empty means measured-and-clean; None
    means unmeasured, and the caller says so out loud.

    THE PEER GROUPING IS DONE IN PYTHON, DELIBERATELY. The first version
    expressed it in SQL with `match_date::date`, `percentile_cont` and
    `FILTER` — all Postgres-only. On the SQLite fallback every one of those
    raises, the `except` below swallows it, and the function returns an empty
    list: the check would have reported "no unpriced fixtures" on exactly the
    database where nobody would look. That is the silent-substitution shape
    this check exists to catch, reproduced inside the check itself. The
    portable form is also the testable one.
    """
    from sqlalchemy import text
    try:
        rows = session.execute(text("""
            SELECT m.id, m.league, m.match_date,
                   COALESCE(ht.name, '?'), COALESCE(at2.name, '?'),
                   m.apifootball_id, m.flashscore_id,
                   (SELECT count(*) FROM odds o WHERE o.match_id = m.id)
            FROM matches m
            LEFT JOIN teams ht  ON ht.id  = m.home_team_id
            LEFT JOIN teams at2 ON at2.id = m.away_team_id
            WHERE m.match_date >= :lo AND m.match_date < :hi
            ORDER BY m.match_date DESC
        """), {"lo": day_from, "hi": day_to}).fetchall()
    except Exception as exc:                       # pragma: no cover - defensive
        logger.debug(f"unpriced-fixture query failed: {exc}")
        return None

    # Group by (league, calendar day) and calibrate each fixture against the
    # peers it actually has. No threshold, no league list.
    priced_counts: dict = {}
    priced_odds: dict = {}
    parsed = []
    for mid, league, ko, home, away, af_id, fs_id, n_odds in rows:
        if isinstance(ko, str):                    # SQLite returns text
            from datetime import datetime as _dt
            try:
                ko = _dt.fromisoformat(ko)
            except ValueError:
                continue
        key = (league, ko.date())
        n_odds = int(n_odds or 0)
        if n_odds > 0:
            priced_counts[key] = priced_counts.get(key, 0) + 1
            priced_odds.setdefault(key, []).append(n_odds)
        parsed.append((mid, league, ko, home, away, af_id, fs_id, n_odds, key))

    out = []
    for mid, league, ko, home, away, af_id, fs_id, n_odds, key in parsed:
        if n_odds > 0 or priced_counts.get(key, 0) == 0:
            continue
        vals = sorted(priced_odds[key])
        mid_i = len(vals) // 2
        med = vals[mid_i] if len(vals) % 2 else (vals[mid_i - 1] + vals[mid_i]) / 2
        out.append((mid, league, ko, home, away,
                    af_id is not None, fs_id is not None,
                    priced_counts[key], med))
    return out


def report_unpriced_fixtures(session, day_from, day_to) -> int:
    """Log the unpriced fixtures, SPLIT BY CAUSE. Returns the alarm count.

    THE SPLIT IS THE DIFFERENCE BETWEEN AN ALARM AND NOISE, and it was added
    after measuring: the unsplit check fired 18 times on 2026-08-29 and 10 on
    08-30. That is the `fixtures_zero_active` failure again — a check that fires
    every day is a check that gets ignored — and the two populations are not the
    same problem:

      * NO API-FOOTBALL ID  (3 of 18 on 08-29: Leuven/St. Liege,
        Radomiak/Cracovia, Avellino/Vicenza). The team could not be resolved, so
        the fixture was never priced. This is the identity-corruption class and
        it ALARMS.
      * HAS AN API-FOOTBALL ID  (15 of 18). The fixture resolved and simply was
        not covered by the odds budget — a known, priced constraint, not a
        defect. This is counted at INFO.

    The `fs_id` column is reported, not filtered on: requiring it would have
    matched Leuven and MISSED 11 of the 12 known Cracovia fixtures.
    """
    found = find_unpriced_fixtures(session, day_from, day_to)
    if found is None:
        logger.warning(
            f"{ALARM_PREFIX} CHECK DID NOT RUN — the query failed, so this run "
            "has NO evidence either way about unpriced fixtures. Not a clean "
            "result.")
        return 0
    unresolved = [f for f in found if not f[5]]
    covered = [f for f in found if f[5]]

    for (mid, league, ko, home, away, has_af, has_fs, priced, med) in unresolved:
        logger.warning(
            f"{ALARM_PREFIX} {home} vs {away} ({league}, "
            f"{ko:%Y-%m-%d %H:%M}) has 0 odds while {priced} same-day peer(s) "
            f"in the same league carry a median of {int(med or 0)}, and it has "
            f"NO API-FOOTBALL ID — the team was never resolved, so the fixture "
            f"was never priced [match_id={mid} fs_id={has_fs}]")
    if unresolved:
        logger.warning(
            f"{ALARM_PREFIX}S: {len(unresolved)} fixture(s) unpriced because a "
            "team could not be resolved. An unpriced fixture is indistinguishable "
            "from one nobody wanted to bet — that is why this alarms.")
    if covered:
        logger.info(
            f"Unpriced but resolved: {len(covered)} fixture(s) have an "
            "API-Football id and no odds — odds-budget coverage, not identity.")
    return len(unresolved)
