"""Identity must not be decided by whatever the database returned first.

    A SELECTION AMONG SEVERAL MATCHING ROWS IS A DECISION. Made without an
    ORDER BY, it is made by the query planner, and it can differ between two
    runs of the same code over the same data.

THE PRECEDENT IS s5.2. The ranking defect resolved ties by iteration order, so
the day's picks depended on fixture iteration order and changed systematically
under sharding. The remedy was a TOTAL ORDER, deliberately chosen, and it was a
cohort event. This is that defect in identity resolution.

WHY IT IS NOT COSMETIC. `resolve_team` step 1 treats a provider id as PROOF, and
until 2026-09-16 thirty-two provider ids matched two team rows under an
unordered `.first()`. Elo and Poisson key on `team_id`, so a club's history
could split across two rows DIFFERENTLY BETWEEN RUNS, and each run's models
trained on whichever split that run happened to produce. The Stage 24 merge
removed the population; it did not remove the pattern, and the next duplicate
from any path still creating rows restores it silently.

MEASURED 2026-09-16, keys that matched more than one row at the time of the fix:

    apifootball._find_match_id (home, away, +/-1 day)   3,882 rows had a rival
    fdo._ensure_fixture (league, date, home name)       2,156 keys
    flashscore exact (home, away, match_date)             773 keys
    Match.apifootball_id                                   264 keys
    fdo._find_team_by_prefix (name ILIKE prefix%)      29 of 889 names
    apifootball resolve-from-any-fixture                20 of 133 teams
    Team.name                                                1 key

THE ORDER IS `id` ASCENDING, EVERYWHERE, and that is a choice rather than a
default: the lowest id is the oldest row, the one other tables already reference
most, and the same survivor rule s5.10's OP1 and Stage 24 used. One rule across
every site beats a locally clever rule at each.

    THIS BUYS DETERMINISM, NOT CORRECTNESS, AND THE DIFFERENCE MATTERS.

`id` ascending has a KNOWN PATHOLOGY in this codebase. It is exactly what walked
s5.10's survivors into the 2.9% of rows carrying `league IS NULL` — low ids
predate the column being populated, so "oldest" selected precisely the rows the
league-scoped lookups were blind to, and the repair walked its own output into
the blind spot. **Oldest is not a proxy for correct.**

So where the oldest row is the WRONG row, ordering makes the wrongness
reproducible rather than removing it. That is still a large improvement — **a
defect that behaves the same way every time is findable; one that flips per
query is not** — but nobody should read the presence of an ORDER BY as evidence
that the right row won. Ordering fixes arbitrary-among-candidates. Only
verification fixes wrong-source, and the two have different remedies.

WHAT THIS DOES NOT PIN. An EXISTENCE PROBE is not a selection: `if existing:
continue` cares only whether a row is there, so which one comes back cannot
change an outcome. Those are listed below with their reason, exactly as
`TEAM_CONSTRUCTION_ALLOWED` lists its exemptions.
"""

import pathlib
import re

#: Files where a `.first()` decides an IDENTITY, not merely existence.
IDENTITY_MODULES = (
    "src/data/team_resolution.py",
    "src/data/fixture_identity.py",
    "src/scrapers/flashscore_scraper.py",
    "src/scrapers/apifootball_scraper.py",
    "src/scrapers/footballdataorg_scraper.py",
    "src/scrapers/historical_loader.py",
)

#: `file:line` -> why an unordered selection is correct there.
EXISTENCE_PROBES = {
    # Odds de-duplication on re-run. The query asks "is there already a row for
    # this (match, bookmaker, market, selection)" and the caller's only response
    # is `continue`. WHICH row comes back cannot change the outcome, so an
    # ORDER BY would be noise pinned by a test.
    "src/scrapers/historical_loader.py:475",
}

#: How many lines above a `.first()` / `.all()` to look for its `.order_by(`.
_LOOKBEHIND = 8


def _unordered_selections():
    out = []
    for rel in IDENTITY_MODULES:
        p = pathlib.Path(rel)
        if not p.exists():
            continue
        lines = p.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines, 1):
            if line.strip().startswith("#"):
                continue
            if not re.search(r"\.(first|one_or_none|one)\(\)", line):
                continue
            window = "\n".join(lines[max(0, i - _LOOKBEHIND):i])
            if ".order_by(" in window:
                continue
            key = f"{rel}:{i}"
            if key in EXISTENCE_PROBES:
                continue
            out.append(f"{key}: {line.strip()[:72]}")
    return out


def test_no_identity_selection_is_left_to_the_query_planner():
    offenders = _unordered_selections()
    assert not offenders, (
        "these selections pick one row out of possibly several with no ORDER "
        "BY, so which row wins is whatever the database returned:\n  "
        + "\n  ".join(offenders)
        + "\n\nAdd `.order_by(<Model>.id)`, or — if the query only asks whether "
          "a row EXISTS — add it to EXISTENCE_PROBES with the reason. "
          "Thirty-two provider ids matched two team rows under an unordered "
          "`.first()`, and Elo and Poisson key on team_id.")


def test_the_exemption_list_does_not_rot():
    """An exemption naming a line that moved is worse than no exemption.

    The list is keyed by line number, so an edit above it silently re-points the
    exemption at unrelated code — and the check it was meant to suppress comes
    back unnoticed. Same failure as any hand-maintained list in this project.
    """
    for key in EXISTENCE_PROBES:
        rel, ln = key.rsplit(":", 1)
        lines = pathlib.Path(rel).read_text(encoding="utf-8").splitlines()
        assert 0 < int(ln) <= len(lines), f"{key} is past the end of the file"
        assert re.search(r"\.(first|one_or_none|one)\(\)", lines[int(ln) - 1]), (
            f"{key} no longer names a selection — the line moved, and the "
            f"exemption is now suppressing something else. Re-point it.")


def test_resolve_team_orders_every_one_of_its_lookups():
    """The five steps are where a wrong choice becomes a stored identity."""
    src = pathlib.Path("src/data/team_resolution.py").read_text(encoding="utf-8")
    assert src.count(".order_by(Team.id)") >= 3, (
        "resolve_team has an unordered lookup — step 1 treats a provider id as "
        "PROOF, and proof chosen arbitrarily among candidates is not proof")
