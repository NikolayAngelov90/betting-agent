"""A club plays in one domestic pyramid. ING-1 step 1.

Measured before it was built, on 2026-09-16: it refuses **8 of 1,493 teams
(0.5%)**, **none of them legitimate**, and reproduces **27 of Stage 13's
hand-built 29** `corrupt_team_identity` marks while adding **19 the hand
missed** — including the four `Telstar` / `other/israel` rows that begin the
documented chain.

Same discipline as the country check: measure what a rule would refuse before it
refuses anything.

THE TWO LIMITS ARE TESTED, NOT JUST DOCUMENTED, because a limit nobody exercises
is a claim rather than a property.
"""

import src.data.database as db_mod
from src.data.fixture_plausibility import (
    country_of_league,
    find_implausible_attributions,
)
from src.data.models import Base, Match, Team
from datetime import datetime


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "p.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _fx(s, home, away, league, day=1, excl=None):
    m = Match(home_team_id=home, away_team_id=away, league=league,
              match_date=datetime(2026, 3, day, 15, 0),
              training_exclusion_reason=excl)
    s.add(m)
    s.flush()
    return m


# ── the normalisation IS the rule ─────────────────────────────────────────

def test_one_country_spelled_two_ways_is_one_country():
    """`other/netherlands` and `netherlands/eredivisie` must fold together.

    Without this the check fires on essentially every row, because this project
    spells a country in the HEAD for covered leagues and in the TAIL under
    `other/`.
    """
    assert country_of_league("netherlands/eredivisie") == "netherlands"
    assert country_of_league("other/netherlands") == "netherlands"
    assert country_of_league("other/israel") == "israel"
    assert country_of_league("israel/ligat-haal") == "israel"


def test_neutral_ground_is_exempt():
    """A club in a European tie is not thereby implausible."""
    assert country_of_league("europe/champions-league") is None
    assert country_of_league("europe/europa-conference-league") is None
    assert country_of_league("other/world") is None


def test_national_team_competitions_are_exempt():
    from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
    for lg in list(NATIONAL_TEAM_LEAGUES)[:3]:
        assert country_of_league(lg) is None, lg


# ── the rule ──────────────────────────────────────────────────────────────

def test_a_club_in_one_pyramid_is_clean(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="Telstar 1963"), Team(id=2, name="Cambuur"),
                   Team(id=3, name="Heerenveen")])
        s.commit()
        _fx(s, 1, 2, "netherlands/eredivisie", 1)
        _fx(s, 3, 1, "netherlands/eredivisie", 2)
        s.commit()
        assert find_implausible_attributions(s) == [], (
            "a club with fixtures in one domestic country was refused")


def test_european_ties_do_not_make_a_club_implausible(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="Levski Sofia"), Team(id=2, name="CSKA Sofia"),
                   Team(id=3, name="Rangers")])
        s.commit()
        _fx(s, 1, 2, "bulgaria/efbet-league", 1)
        _fx(s, 1, 3, "europe/europa-conference-league", 2)
        _fx(s, 1, 3, "other/world", 3)
        s.commit()
        assert find_implausible_attributions(s) == [], (
            "continental and neutral fixtures were counted as a second "
            "domestic country — that would refuse every club that plays in "
            "Europe")


def test_THE_TELSTAR_CASE(tmp_path):
    """THE REGRESSION. A Dutch club with a season of Israeli league fixtures.

    Four fixtures against one dominant opponent — the opponent-season backfill
    signature. This is the attribution that begins ING-1: it became evidence,
    the evidence wrote af=604 (Maccabi Tel Aviv) onto the row, and step 1 of
    `resolve_team` treats that column as proof.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="Telstar"), Team(id=2, name="Cambuur"),
                   Team(id=3, name="Hapoel Beer Sheva")])
        s.commit()
        for d in (1, 2, 3, 4, 5, 6):          # the real row carries 48
            _fx(s, 1, 2, "netherlands/eredivisie", d)
        for d in (7, 8, 9, 10):               # and exactly 4 Israeli
            _fx(s, 1, 3, "other/israel", d)
        s.commit()
        got = find_implausible_attributions(s)
        assert len(got) == 1, got
        d = got[0]
        assert d["team_id"] == 1
        assert d["majority_country"] == "netherlands"
        assert d["countries"] == {"netherlands": 6, "israel": 4}
        assert len(d["minority_fixture_ids"]) == 4
        assert len(d["minority_unmarked_ids"]) == 4, (
            "all four are unmarked in production and therefore still readable "
            "as evidence by the identity-writing path")


def test_unmarked_fixtures_are_counted_separately_from_marked(tmp_path):
    """Only UNMARKED fixtures are still readable as evidence by the id path."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="St. Pauli"), Team(id=2, name="Bayern"),
                   Team(id=3, name="Troyes")])
        s.commit()
        for d in (1, 2, 3, 4):
            _fx(s, 1, 2, "germany/bundesliga", d)
        _fx(s, 1, 3, "france/ligue-2", 5, excl="corrupt_team_identity")
        _fx(s, 1, 3, "france/ligue-2", 6)
        s.commit()
        d = find_implausible_attributions(s)[0]
        assert len(d["minority_fixture_ids"]) == 2
        assert len(d["minority_unmarked_ids"]) == 1, (
            "a marked fixture was counted as still-readable evidence")


# ── LIMIT 1: it flags the ROW, not the SIDE ───────────────────────────────

def test_the_MAJORITY_can_be_the_contaminated_side(tmp_path):
    """`York City`: usa=37, england=4, and the ENGLISH four are the real club.

    The check must not name a side. A caller that assumes "minority = wrong"
    would detach the only legitimate fixtures this row has.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="York City"), Team(id=2, name="Forge FC"),
                   Team(id=3, name="Exeter")])
        s.commit()
        for d in (1, 2, 3, 4, 5):
            _fx(s, 1, 2, "usa/mls", d)
        _fx(s, 1, 3, "england/league-two", 6)
        s.commit()
        d = find_implausible_attributions(s)[0]
        assert d["majority_country"] == "usa"
        assert "wrong_country" not in d and "wrong_fixture_ids" not in d, (
            "the check named a side — it establishes that the ROW is "
            "impossible, never which half to detach")


# ── LIMIT 2: blind to same-country corruption ─────────────────────────────

def test_same_country_corruption_is_NOT_detected_and_that_is_KNOWN(tmp_path):
    """Two clubs in one league, one's fixtures on the other's row: invisible.

    This is the 2 of Stage 13's 29 the rule does not reach. Pinned so the gap
    is a known property rather than a surprise, and so nobody reports this rule
    as covering the class.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="Rakow"), Team(id=2, name="Cracovia"),
                   Team(id=3, name="Lech Poznan")])
        s.commit()
        for d in (1, 2, 3):
            _fx(s, 1, 3, "poland/ekstraklasa", d)
        _fx(s, 2, 1, "poland/ekstraklasa", 4)
        s.commit()
        assert find_implausible_attributions(s) == [], (
            "this rule appeared to detect same-country corruption — it cannot, "
            "and a test claiming otherwise would overstate its coverage")


# ── the third state, kept ─────────────────────────────────────────────────

def test_a_failed_query_returns_None_not_an_empty_list():
    """Empty means measured-and-clean. None means unmeasured."""
    class _Boom:
        def execute(self, *a, **k):
            raise RuntimeError("no such table")

    got = find_implausible_attributions(_Boom())
    assert got is None and got != [], (
        "a broken check reported 'nothing wrong' — the failure mode this "
        "project has closed four times")
