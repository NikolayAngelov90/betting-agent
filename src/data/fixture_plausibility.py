"""A club's fixtures must lie in AT MOST ONE domestic country's competitions.

ING-1, STEP 1 — and it is the only step in that chain where a fix is cheap.

    ingestion attributes a fixture to the wrong team row
        -> the fixture persists, unmarked
            -> the fixture-derived path reads it as EVIDENCE
                -> it writes `apifootball_team_id` PERMANENTLY
                    -> `resolve_team` step 1 treats that column as PROOF

Ordering, verification, clearing and merging all act on steps 3-5. This acts on
step 1. A wrong attribution caught here costs one marked fixture; the same
attribution caught after step 4 has already become a stored identity, and
clearing that identity returns the row to step 3 with the evidence intact
(CLR-1).

WHY THIS RULE AND NOT A COUNTRY COLUMN. `teams.country` is 44% populated and
records where a club was FIRST SEEN rather than where it plays — Levski Sofia is
stored as "Europe" because a Conference League tie created it. This rule needs
no per-team data at all: it reads only the competitions a row's own fixtures are
in, and asks whether a single club could be in all of them.

MEASURED BEFORE IT WAS BUILT, 2026-09-16, the same discipline the country check
used — measure what it would refuse before it refuses anything:

    teams with any fixtures                 1,493
    teams it refuses                            8   (0.5%)
    of those, LEGITIMATE                        0
    minority-country fixtures implicated       46
    already marked `corrupt_team_identity`     27
    NOT marked — what it ADDS                  19

**One mechanical rule reproduces 27 of Stage 13's hand-built 29, refuses nothing
legitimate, and finds 19 the hand missed.** It is a better instrument than the
one it supplements, not a cheap approximation of it.

THE SIGNATURE IT KEEPS FINDING — an OPPONENT-SEASON BACKFILL. Four of the eight
are the same shape: a season's worth of ONE foreign league against ONE dominant
opponent. `Telstar` v Hapoel Beer Sheva x4, `Aris` v Omonia Nicosia x4,
`Kocaelispor` v Omonia Nicosia x3, `NK Varazdin` v Neftchi Baku x4. Backfilling
one club's season resolves each opponent name in turn, and a single name that
resolves to the wrong row deposits that club's whole season on it. Recognise the
shape on sight rather than re-deriving it: **one foreign league, one dominant
opponent, a contiguous season.**

TWO LIMITS, AND THEY ARE STATED HERE RATHER THAN ONLY IN THE LEDGER BECAUSE
THIS IS WHERE THEY WILL BE READ:

  1. IT FLAGS THE ROW, NOT THE SIDE. `York City` is refused with `usa=37,
     england=4` and it is the MAJORITY that is wrong there — York United's
     Canadian fixtures sit on the English club's row. Never assume the minority
     is the contamination; the check says the row is impossible, not which half
     to detach. That judgement is a separate step and needs a human.
  2. IT CANNOT SEE SAME-COUNTRY CORRUPTION — one club's fixtures on another
     club's row inside a single league. That is exactly the 2 of Stage 13's 29
     it does not reach, and it is a blind spot by construction. This rule is
     necessary and not sufficient, and nothing here should be read as covering
     the class.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger

logger = get_logger()

#: Emitted once per affected team. `ci_audit.py` greps this exact prefix.
ALARM_PREFIX = "IMPLAUSIBLE ATTRIBUTION"

#: Competitions that are NEUTRAL GROUND: a club legitimately appears in them
#: whatever its domestic pyramid, so they can never make a row impossible.
NEUTRAL_LEAGUE_PREFIXES = ("europe/", "other/world")


def country_of_league(league: Optional[str]) -> Optional[str]:
    """The domestic country a league belongs to, or None when it is exempt.

    THE NORMALISATION IS THE WHOLE RULE. This project spells one country two
    ways — `netherlands/eredivisie` puts it in the HEAD, `other/israel` puts it
    in the TAIL — and without folding them together a club with fixtures in
    `other/netherlands` and `netherlands/eredivisie` reads as two countries and
    the check fires on 1,400 legitimate rows.

    Returns None for neutral ground and for national-team competitions, which
    are a different identity space entirely (`_partition_filter`'s boundary).
    """
    if not league:
        return None
    try:
        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        if league in NATIONAL_TEAM_LEAGUES:
            return None
    except Exception:                                # pragma: no cover
        pass
    if any(league.startswith(p) for p in NEUTRAL_LEAGUE_PREFIXES):
        return None
    head, _, tail = league.partition("/")
    return (tail if head == "other" else head) or None


def find_implausible_attributions(session) -> Optional[List[dict]]:
    """Team rows whose fixtures span more than one domestic country.

    Returns one dict per offending row, or None WHEN THE QUERY ITSELF FAILED.

    None is not an empty list, and the distinction is the same one
    `find_unpriced_fixtures` earned: a check that answers "nothing wrong" when
    it is broken is worse than no check, because it is trusted. Empty means
    measured-and-clean.

    The grouping is done in PYTHON rather than SQL for the same reason as the
    unpriced check: the normalisation above is not expressible portably, and a
    dialect-specific expression that raises on SQLite would be swallowed and
    return an empty list — reporting "no implausible rows" on exactly the
    database where nobody would look.
    """
    from sqlalchemy import text
    try:
        rows = session.execute(text("""
            SELECT t.id AS tid, t.name AS tname, m.league AS mleague,
                   m.id AS mid, m.training_exclusion_reason AS excl
            FROM teams t
            JOIN matches m
              ON m.home_team_id = t.id OR m.away_team_id = t.id
        """)).fetchall()
    except Exception as exc:                         # pragma: no cover - defensive
        logger.debug(f"implausible-attribution query failed: {exc}")
        return None

    per_team: Dict[int, Dict[str, List[Tuple[int, Optional[str]]]]] = {}
    names: Dict[int, str] = {}
    for r in rows:
        c = country_of_league(r.mleague)
        if not c:
            continue
        names[r.tid] = r.tname
        per_team.setdefault(r.tid, {}).setdefault(c, []).append((r.mid, r.excl))

    out: List[dict] = []
    for tid, by_country in per_team.items():
        if len(by_country) < 2:
            continue
        ordered = sorted(by_country.items(), key=lambda kv: -len(kv[1]))
        majority, rest = ordered[0], ordered[1:]
        minority_fixtures = [f for _, fx in rest for f in fx]
        out.append({
            "team_id": tid,
            "team_name": names.get(tid, "?"),
            "countries": {c: len(fx) for c, fx in ordered},
            "majority_country": majority[0],
            "majority_count": len(majority[1]),
            # NAMED `minority_*`, NOT `wrong_*`. See limit 1 in the module
            # docstring: `York City` is refused with the MAJORITY wrong.
            "minority_fixture_ids": [m for m, _ in minority_fixtures],
            "minority_unmarked_ids": [m for m, x in minority_fixtures if not x],
        })
    return sorted(out, key=lambda d: -d["majority_count"])


def report_implausible_attributions(session) -> int:
    """Log each impossible row and return how many there are. Never raises."""
    found = find_implausible_attributions(session)
    if found is None:
        logger.warning(
            f"{ALARM_PREFIX} CHECK DID NOT RUN — the query failed, so this run "
            "has NO evidence either way about fixture attribution. Not a clean "
            "result.")
        return 0

    for d in found:
        spread = ", ".join(f"{c}={n}" for c, n in d["countries"].items())
        logger.warning(
            f"{ALARM_PREFIX} team {d['team_id']} {d['team_name']!r} has fixtures "
            f"in {len(d['countries'])} domestic countries [{spread}] — a club "
            f"plays in one domestic pyramid, so one of these groups belongs to a "
            f"different club. {len(d['minority_unmarked_ids'])} of "
            f"{len(d['minority_fixture_ids'])} minority fixture(s) are NOT yet "
            f"marked. WHICH GROUP IS WRONG IS NOT DECIDED HERE — the majority "
            f"can be the contaminated one.")
    if found:
        unmarked = sum(len(d["minority_unmarked_ids"]) for d in found)
        logger.warning(
            f"{ALARM_PREFIX}S: {len(found)} team row(s) carry fixtures from more "
            f"than one domestic country; {unmarked} implicated fixture(s) are "
            f"unmarked and remain readable as evidence by the identity-writing "
            f"path. Blind to same-country corruption by construction.")
    return len(found)
