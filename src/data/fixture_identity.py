"""Fixture identity, as distinct from row identity.

THE DISTINCTION THIS MODULE EXISTS FOR. `matches.id` identifies a ROW.
`max_picks_per_match` is a guarantee about a FIXTURE. Those are different
things, and on 2026-08-30 the difference cost a violation of the guarantee
Stage 13 broke a cohort to establish:

    50920  RC Deportivo La Coruña v Valencia  (API-Football)  -> 1 pick
    50927  Dep. A Coruna v Valencia           (Flashscore)    -> 1 pick

One real fixture, two rows, two `match_id`s, two independent pick slots. The
cap grouped by `match_id` and saw two matches. The correlation filter keyed on
the match and never compared the pair, so a positively-correlated Double
Chance X2 / Under 2.5 pair survived — the lower-EV member is exactly what that
filter exists to drop.

WHY THIS IS NOT A BETTER MATCHER. The ingest-time matcher cannot be made
sufficient: `Vitória SC` and `Guimaraes` are the same club and share zero
tokens, a residual already documented as unreachable by any lexical test. An
alias closes that pair and not the next one. This asks a different question —
not "do these names look alike" but "are these the same fixture" — and answers
it from provider identity, so it holds when the matcher fails.

WHY PICK TIME IS A BETTER POSITION, MEASURED. At ingest the comparison is the
provider's raw text against a stored name. By pick time both sides have been
through team resolution, so both are canonical stored names carrying provider
ids. All five live duplicate pairs pass `team_names_similar` on BOTH sides
when the two STORED rows are compared, though every one failed at ingest:

    ingest     "Vitória SC"          vs stored "Guimaraes"   -> False
    pick time  stored "Guimaraes"    vs stored "Guimaraes"   -> True

THE THREE BRANCHES, and their evidence is not equal — see `same_fixture`.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from src.utils.logger import get_logger
from src.utils.team_names import team_names_similar

logger = get_logger()


class _Row:
    """The identity-relevant projection of a match row."""

    __slots__ = ("mid", "league", "ko", "home", "away", "home_af", "away_af")

    def __init__(self, mid, league, ko, home, away, home_af, away_af):
        self.mid = mid
        self.league = league
        # SQLite hands back a string where Postgres hands back a datetime, and
        # the bucket key is (league, kickoff) — mixing the two types would put
        # one fixture's rows in two buckets and silently group nothing. The
        # same portability trap `coverage_checks` hit; normalise at the
        # boundary, once.
        if isinstance(ko, str):
            from datetime import datetime as _dt
            try:
                ko = _dt.fromisoformat(ko)
            except ValueError:
                pass
        self.ko = ko
        self.home = home or ""
        self.away = away or ""
        self.home_af = home_af
        self.away_af = away_af


def _shares_resolved_club(a: Optional[int], b: Optional[int]) -> bool:
    return a is not None and b is not None and a == b


def same_fixture(a: _Row, b: _Row) -> Optional[str]:
    """Are these two rows the same real fixture? Returns the deciding branch.

    Callers must already have established that the rows share a league and a
    kickoff minute; this decides identity within that bucket.

    BRANCH 1 — PROVABLE. At least one club resolves to the same provider id.
    A club cannot play two fixtures in one competition at the same minute, so
    a shared resolved club plus a shared minute IS identity, whatever the
    other slot says. This is an impossibility argument, not a threshold: it
    has no tunable part and cannot drift. It catches all five live duplicate
    pairs and both known violations.

    BRANCH 2 — HEURISTIC, and MEASURED rather than assumed. No shared provider
    id, but both stored name pairs are similar. Measured 2026-08-31 against
    every same-league same-minute pair in the database (60,976 candidates,
    77,542 branch-2-eligible at a wider ±4h): it would refuse 89 pairs, and
    ZERO of those are provably different fixtures under any available decider
    — not by team provider id, not by `matches.apifootball_id`, not by
    `flashscore_id`. Measured false-positive rate 0 of 89. The bound is
    [0.0%, 40.4%] only if every pair where NEITHER row carries a provider
    fixture id is also wrong, which inspection contradicts (`Man United v Man
    City` ‖ `Manchester Utd v Manchester City`).

    BRANCH 3 — RESIDUAL, declared. No shared id and dissimilar names: not the
    same fixture as far as this can tell. Smaller at pick time than at ingest
    because of the stored-vs-stored positioning above, but NOT empty and not
    claimed to be.

    NOTE ON FAILING CLOSED. This refuses on identity EVIDENCE, never on its
    absence. Refusing every same-league same-minute pair that cannot be
    resolved would reject genuinely simultaneous fixtures, which are the norm
    on a final matchday — 60,141 of the 60,976 candidates. The same
    distinction `_parse_match_date` draws: unknown is not the same as wrong.
    """
    if _shares_resolved_club(a.home_af, b.home_af) or \
            _shares_resolved_club(a.away_af, b.away_af):
        return "provider_id"
    if team_names_similar(a.home, b.home) and team_names_similar(a.away, b.away):
        return "stored_names"
    return None


def _load(session, match_ids: Sequence[int]) -> List[_Row]:
    """Every row sharing a (league, kickoff) with any of `match_ids`.

    THE TWIN IS OFTEN NOT A CANDIDATE, so loading only `match_ids` would miss
    it. On 2026-08-31 row 50990 carried 78 odds and produced no pick while its
    twin 50969 produced one; a resolver that saw only candidates would have
    found nothing to group. It matters more for the cross-run case: a pick
    saved earlier today on the twin must still consume this fixture's slot.
    """
    from sqlalchemy import bindparam, text
    if not match_ids:
        return []
    # `expanding=True` is required for an IN over a bound list. Without it the
    # driver receives a single tuple parameter and raises, which the caller's
    # except turns into a warning and a degrade to row identity — the feature
    # would have been a no-op in production while logging that it was fine.
    stmt = text("""
        WITH seed AS (
            SELECT DISTINCT league, match_date FROM matches WHERE id IN :ids
        )
        SELECT m.id, m.league, m.match_date,
               ht.name, at2.name,
               ht.apifootball_team_id, at2.apifootball_team_id
        FROM matches m
        JOIN seed s ON s.league = m.league AND s.match_date = m.match_date
        LEFT JOIN teams ht  ON ht.id  = m.home_team_id
        LEFT JOIN teams at2 ON at2.id = m.away_team_id
    """).bindparams(bindparam("ids", expanding=True))
    rows = session.execute(stmt, {"ids": list(match_ids)}).fetchall()
    return [_Row(*r) for r in rows]


def resolve_fixture_groups(session, match_ids: Iterable[int]) -> Dict[int, int]:
    """Map each match_id to a canonical id shared by every row of its fixture.

    The canonical id is the smallest `match_id` in the cluster, so the mapping
    is stable across runs and independent of iteration order — the same
    property `_rank_key` needed for reproducibility.

    Returns an identity mapping for anything it cannot group, so callers can
    apply it unconditionally. Never raises: a resolver that breaks the picks
    run is worse than the duplicate it would have caught.
    """
    ids = sorted({int(m) for m in match_ids})
    if not ids:
        return {}
    try:
        rows = _load(session, ids)
    except Exception as exc:                       # pragma: no cover - defensive
        logger.warning(
            f"fixture-identity resolution failed, falling back to row identity "
            f"({exc}). The per-fixture cap degrades to per-row for this run.")
        return {i: i for i in ids}

    buckets: Dict[tuple, List[_Row]] = {}
    for r in rows:
        buckets.setdefault((r.league, r.ko), []).append(r)

    parent: Dict[int, int] = {r.mid: r.mid for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            lo, hi = (rx, ry) if rx < ry else (ry, rx)
            parent[hi] = lo

    for group in buckets.values():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                branch = same_fixture(group[i], group[j])
                if branch:
                    union(group[i].mid, group[j].mid)
                    logger.info(
                        f"SAME FIXTURE {group[i].mid} and {group[j].mid} "
                        f"({group[i].league}, {group[i].ko}) — "
                        f"'{group[i].home} v {group[i].away}' and "
                        f"'{group[j].home} v {group[j].away}' are one fixture "
                        f"on two rows [branch={branch}]")

    return {i: find(i) if i in parent else i for i in ids}
