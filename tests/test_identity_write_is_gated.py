"""ING-1 steps 3 and 4: what a fixture is allowed to turn into.

The chain, and where these two act:

    1. ingestion attributes a fixture to the wrong team row
    2. the fixture persists                 <- the plausibility invariant marks it
    3. the id path READS it as evidence     <- GATED HERE (exclusion honoured)
    4. it WRITES apifootball_team_id        <- REFUSED HERE (no collisions)
    5. resolve_team step 1 calls it PROOF

**Step 3 without step 4 is not enough, and neither is the reverse.** The
exclusion cannot see `1531 Telstar 1963`, whose five fixtures are all legitimate
Dutch league matches — nothing marks them, and the write would be a CORRECT id
onto a SECOND row. The collision guard cannot see `Telstar`/`other/israel`,
where the id is simply wrong and no other row holds it. Two halves, two
mechanisms, and each is blind where the other sees.

MEASURED, the number that justifies the collision guard: of the 32
shared-provider-id components merged on 2026-09-16, **8 had the DUPLICATE's id
written by this path**, and it never wrote both sides — so it completes
collisions rather than creating them alone, and refusing is what breaks the
completion.
"""

import re
import pathlib

import src.data.database as db_mod
from src.data.models import Base, Match, Team
from datetime import datetime

SRC = pathlib.Path("src/scrapers/apifootball_scraper.py")


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "g.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


# ── step 3: the exclusion gates the identity-writing path ─────────────────

def test_the_id_path_filters_on_the_exclusion_mark():
    """THE ENFORCEMENT. Without it the invariant is a detector wired to nothing.

    Asserted on the source because the surrounding function needs a live API
    client and a budget. What must not regress is the PREDICATE, and that is
    readable without either.
    """
    src = SRC.read_text(encoding="utf-8")
    block = src[src.index("missing_api_id = ["):]
    block = block[:block.index("# Persist resolved IDs")]
    assert "Match.apifootball_id.isnot(None)" in block
    assert "Match.training_exclusion_reason.is_(None)" in block, (
        "the identity-writing path no longer honours the exclusion mark — it "
        "DERIVES PERSISTENT STATE from a match, so it is inside the gate, and "
        "without it the plausibility invariant marks fixtures that this path "
        "goes on reading anyway")


def test_a_marked_fixture_is_not_a_candidate(tmp_path):
    """The query's own semantics, exercised against a database."""
    from sqlalchemy import or_ as _or
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="Telstar"), Team(id=2, name="Hapoel Beer Sheva")])
        s.commit()
        s.add(Match(id=10, home_team_id=1, away_team_id=2, league="other/israel",
                    match_date=datetime(2026, 3, 1), apifootball_id=1214904,
                    training_exclusion_reason="corrupt_team_identity"))
        s.add(Match(id=11, home_team_id=1, away_team_id=2, league="other/israel",
                    match_date=datetime(2026, 3, 2), apifootball_id=1214995))
        s.commit()

        def _candidate(tid, honour_exclusion):
            q = s.query(Match).filter(
                _or(Match.home_team_id == tid, Match.away_team_id == tid),
                Match.apifootball_id.isnot(None))
            if honour_exclusion:
                q = q.filter(Match.training_exclusion_reason.is_(None))
            return q.order_by(Match.id).first()

        assert _candidate(1, False).id == 10, "precondition"
        assert _candidate(1, True).id == 11, (
            "a marked fixture was still offered as evidence")

        s.query(Match).filter(Match.id == 11).update(
            {"training_exclusion_reason": "corrupt_team_identity"})
        s.commit()
        assert _candidate(1, True) is None, (
            "with every candidate marked the path must find NOTHING and leave "
            "the row unidentified — that is the intended outcome, not a failure")


# ── step 4: a provider id is an identity claim ────────────────────────────

def test_the_write_refuses_an_id_another_row_already_holds():
    """THE 1531 CASE. A correct id on a second row is still a collision."""
    src = SRC.read_text(encoding="utf-8")
    block = src[src.index("# Persist resolved IDs"):]
    block = block[:block.index("# Rebuild low_coverage")]
    assert "PROVIDER ID COLLISION REFUSED" in block, (
        "the identity write no longer refuses a collision — `1531 Telstar "
        "1963` has five LEGITIMATE fixtures and `1528 Telstar` already holds "
        "af=427, so the plausibility invariant cannot reach this and only the "
        "refusal can")
    assert "Team.apifootball_team_id == api_id" in block
    assert "Team.id != tid" in block, (
        "the collision check must exclude the row being written, or every "
        "re-run refuses itself")


def test_the_collision_predicate_holds_against_a_database(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([
            Team(id=1528, name="Telstar", apifootball_team_id=427),
            Team(id=1531, name="Telstar 1963"),
            Team(id=9, name="Lorient"),
        ])
        s.commit()

        def _holder(tid, api_id):
            return (s.query(Team)
                    .filter(Team.apifootball_team_id == api_id, Team.id != tid)
                    .order_by(Team.id).first())

        assert _holder(1531, 427) is not None, (
            "writing 427 onto 1531 would produce the 33rd shared-provider-id "
            "component hours after 32 were merged")
        assert _holder(9, 999) is None, "a free id must still be writable"
        assert _holder(1528, 427) is None, (
            "the row that already holds the id must not refuse itself")


def test_a_refused_row_is_not_carried_forward_as_resolved():
    """The refusal must survive the line after it.

    `resolved` is the proposal and `written` is what the database accepted. The
    backfill below reads one of them; if it reads the proposal, a refused row is
    treated as identified and the refusal is invisible one line after it is
    logged.
    """
    src = SRC.read_text(encoding="utf-8")
    block = src[src.index("# Persist resolved IDs"):]
    block = block[:block.index("# Rebuild low_coverage")]
    assert re.search(r"resolved\s*=\s*\{.*for t, a in resolved\.items\(\)", block), (
        "`resolved` is not narrowed to what was actually written, so a refused "
        "collision is carried into the backfill as though it had an id")


# ── the two halves are genuinely different ────────────────────────────────

def test_neither_half_subsumes_the_other(tmp_path):
    """Each is blind exactly where the other sees.

    Telstar/other/israel : the id is WRONG and no row holds it
                           -> exclusion catches it, collision guard does not
    Telstar 1963         : the id is RIGHT and another row holds it
                           -> collision guard catches it, exclusion does not
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1528, name="Telstar", apifootball_team_id=427),
                   Team(id=1531, name="Telstar 1963"),
                   Team(id=2, name="Hapoel Beer Sheva")])
        s.commit()
        # 1531's fixtures are legitimate -> nothing to mark.
        s.add(Match(id=20, home_team_id=1531, away_team_id=2,
                    league="netherlands/eredivisie",
                    match_date=datetime(2026, 3, 1), apifootball_id=1552118))
        s.commit()
        unmarked = s.query(Match).filter(
            Match.training_exclusion_reason.isnot(None)).count()
        assert unmarked == 0, (
            "the plausibility invariant has nothing to mark here, which is why "
            "the collision guard is not redundant with it")
        holder = (s.query(Team)
                  .filter(Team.apifootball_team_id == 427, Team.id != 1531)
                  .order_by(Team.id).first())
        assert holder is not None and holder.id == 1528
