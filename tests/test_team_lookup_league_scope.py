"""A NULL-league team was invisible to the duplicate check. Rule 1, again.

MEASURED 2026-09-11, the day after s5.10 merged 129 team rows: 26 of them were
re-created within ~24 hours — 20% of the merge undone in a day, the whole merge
in about five at that rate.

Every resurrected row duplicated a survivor carrying `league IS NULL`, and
`flashscore._get_or_create_team` scanned `filter_by(league=<scraped league>)`.
So the survivor was never in the candidate set and `same_team_strict` was never
called against it. `same_team_strict("PSG", "Paris SG")` is True and always
was — it was simply never asked.

    A lookup is only as good as its earliest decision point. The league filter
    decided the candidate set BEFORE the comparator was consulted.

WHY THE SURVIVORS WERE THE NULL ONES. s5.10's OP1 kept the LOWEST ID, and low
ids are old rows created before `league` was populated. Only 2.9% of teams carry
a NULL league — and the merge concentrated its survivors into exactly that 2.9%.
The repair walked its own output into the blind spot.

THIS FIX IS NECESSARY AND NOT SUFFICIENT. Of the 26 resurrections it prevents 4
outright (strict already True, only the filter hid them). The other 18 need an
alias as well, because `same_team_strict("Lens", "Racing Club de Lens")` is
False even when the two are compared. The alias half is BLOCKED: the set
generated from the merge log introduces 369 new symmetric-canonicalisation
hazards against the Stage 20 pin, 75 of 110 being token-deleting because OP1
kept the lowest id rather than the longest name.
"""

import src.data.database as db_mod
from src.data.models import Base, Team
from src.scrapers.flashscore_scraper import FlashscoreScraper


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "t.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _scraper():
    return FlashscoreScraper.__new__(FlashscoreScraper)


def test_a_null_league_twin_is_found(tmp_path):
    """THE REGRESSION. The survivor of a merge carries no league."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Paris SG", league=None, apifootball_team_id=85))
        s.commit()
        got = _scraper()._get_or_create_team(s, "PSG", "france/ligue-1")
        s.flush()
        assert got.name == "Paris SG", (
            f"created {got.name!r} instead of matching the NULL-league "
            "survivor — this is the 26-resurrections defect")
        assert got.apifootball_team_id == 85
        assert s.query(Team).count() == 1, "a duplicate row was created"


def test_a_same_league_twin_still_matches(tmp_path):
    """The existing behaviour must survive the widening."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Malmo FF", league="sweden/allsvenskan"))
        s.commit()
        got = _scraper()._get_or_create_team(s, "Malmö FF", "sweden/allsvenskan")
        s.flush()
        assert got.name == "Malmo FF"
        assert s.query(Team).count() == 1


def test_a_genuinely_new_club_is_still_created(tmp_path):
    """Widening the scan must not start matching everything."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Paris SG", league=None))
        s.commit()
        got = _scraper()._get_or_create_team(s, "Lorient", "france/ligue-1")
        s.flush()
        assert got.name == "Lorient"
        assert s.query(Team).count() == 2


def test_distinct_same_city_clubs_are_still_kept_apart(tmp_path):
    """The conservatism that makes same_team_strict worth consulting.

    Widening WHICH rows are compared must not widen WHAT counts as a match.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Sheffield Wednesday", league=None))
        s.commit()
        got = _scraper()._get_or_create_team(
            s, "Sheffield United", "england/championship")
        s.flush()
        assert got.name == "Sheffield United", (
            "Sheffield United was matched to Sheffield Wednesday — the "
            "widening changed the comparator, which it must not")
        assert s.query(Team).count() == 2


def test_the_strict_scan_is_not_widened_beyond_null_league(tmp_path):
    """Only NULL is added to the STRICT scan, not every league.

    Uses non-identical names on purpose, because the exact-name lookup above
    the strict scan short-circuits first — see the test below.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Arsenal FC", league="argentina/primera"))
        s.commit()
        got = _scraper()._get_or_create_team(s, "Arsenal", "england/premier-league")
        s.flush()
        assert got.league == "england/premier-league", (
            "strict-matched Arsenal (England) to Arsenal FC (Argentina) — the "
            "scan was widened beyond NULL-league rows")
        assert s.query(Team).count() == 2


def test_the_exact_name_lookup_is_global_and_that_is_PRE_EXISTING(tmp_path):
    """Found while testing the fix, and NOT caused by it. Recorded, not changed.

    `_get_or_create_team` short-circuits on
    `session.query(Team).filter_by(name=team_name).first()` BEFORE the strict
    scan, and that query has no league filter at all. Two clubs sharing an exact
    name in different countries therefore collapse into one row, and always did.

    This test pins the CURRENT behaviour so the next reader meets it as a known
    property rather than a surprise. Changing it is a separate decision with its
    own blast radius — every existing row that collapsed this way would need
    splitting, which is the opposite of a merge and much harder.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Arsenal", league="argentina/primera"))
        s.commit()
        got = _scraper()._get_or_create_team(s, "Arsenal", "england/premier-league")
        s.flush()
        assert got.league == "argentina/primera", (
            "the exact-name short-circuit no longer matches across leagues — "
            "that is a behaviour CHANGE and needs its own decision")
        assert s.query(Team).count() == 1
