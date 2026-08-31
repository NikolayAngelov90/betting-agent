"""s5.9 — the per-match cap keys on FIXTURE identity, not ROW identity.

THE VIOLATION THIS PREVENTS, replayed from production:

    2026-08-30 17:30, spain/laliga, Deportivo La Coruña v Valencia
      row 50920 (API-Football) -> Double Chance X2 @1.515  (settled loss)
      row 50927 (Flashscore)   -> Under 2.5 @1.56, EV -0.1924

One real fixture, two rows, two `match_id`s, two independent pick slots under
`max_picks_per_match: 1`. The correlation filter keys on the match too, so the
positively-correlated pair was never compared.

THE THREE BRANCHES HAVE UNEQUAL EVIDENCE and the tests say which is which:

  1. PROVABLE — a shared provider club id at a shared league and kickoff
     minute. A club cannot play twice in one competition at one minute, so
     this is an impossibility argument with no tunable part.
  2. HEURISTIC — no shared id, both stored name pairs similar. MEASURED
     2026-08-31 across every same-league same-minute pair in the database:
     fires on 89, of which ZERO are provably different fixtures under any
     decider available (team provider id, `matches.apifootball_id`,
     `flashscore_id`). 0/89.
  3. RESIDUAL — declared, not empty. `Vitória SC` / `Guimaraes` shares zero
     tokens and no lexical test reaches it.

REFUSING ON EVIDENCE, NEVER ON ITS ABSENCE. 60,141 of 60,976 same-league
same-minute pairs are genuinely different simultaneous fixtures. A rule that
failed closed on unresolvable pairs would reject a normal matchday.
"""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data.fixture_identity import resolve_fixture_groups
from src.data.models import Base, Match, Team

KO = datetime(2026, 8, 30, 17, 30)


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    s = sessionmaker(bind=engine)()
    yield s
    s.close()


def _team(s, tid, name, af=None):
    s.add(Team(id=tid, name=name, apifootball_team_id=af))


def _match(s, mid, home, away, league="spain/laliga", ko=KO, af_fix=None):
    s.add(Match(id=mid, league=league, match_date=ko,
                home_team_id=home, away_team_id=away, apifootball_id=af_fix))


# ── branch 1: provable ──────────────────────────────────────────────────────
def test_the_2026_08_30_violation_is_now_one_fixture(session):
    """Valencia resolves to 532 on both rows; that alone settles it."""
    _team(session, 1, "RC Deportivo La Coruña", af=544)
    _team(session, 2, "Valencia", af=532)
    _team(session, 3, "Dep. A Coruna", af=None)      # unresolved, as in production
    _team(session, 4, "Valencia", af=532)
    _match(session, 50920, 1, 2)
    _match(session, 50927, 3, 4)
    session.commit()

    groups = resolve_fixture_groups(session, [50920, 50927])
    assert groups[50920] == groups[50927], (
        "these two rows are one fixture — Valencia cannot play two matches in "
        "one competition at 17:30. Grouping them apart is the s5.3 violation")
    assert groups[50920] == 50920, "canonical id is the smallest in the cluster"


def test_one_shared_club_is_enough_even_when_the_other_side_is_unresolved(session):
    """The 2026-08-31 Braga shape: away shares afid 224, home resolves on neither."""
    _team(session, 1, "Sporting Clube de Braga", af=None)
    _team(session, 2, "Guimaraes", af=224)
    _team(session, 3, "Braga", af=217)
    _team(session, 4, "Guimaraes", af=224)
    _match(session, 50969, 1, 2, league="portugal/primeira-liga")
    _match(session, 50990, 3, 4, league="portugal/primeira-liga")
    session.commit()
    g = resolve_fixture_groups(session, [50969, 50990])
    assert g[50969] == g[50990]


# ── branch 2: the heuristic, and it must still fire ────────────────────────
def test_stored_names_group_a_pair_no_provider_id_can(session):
    """`Man United v Man City` ‖ `Manchester Utd v Manchester City`.

    Neither slot resolves on both rows, so branch 1 cannot see it. This is the
    89-pair population whose measured false-positive rate is 0.
    """
    _team(session, 1, "Man United", af=33)
    _team(session, 2, "Man City", af=None)
    _team(session, 3, "Manchester Utd", af=None)
    _team(session, 4, "Manchester City", af=50)
    _match(session, 100, 1, 2, league="england/premier-league")
    _match(session, 101, 3, 4, league="england/premier-league")
    session.commit()
    g = resolve_fixture_groups(session, [100, 101])
    assert g[100] == g[101], "branch 2 must still fire where branch 1 cannot"


# ── the property that keeps a normal matchday working ──────────────────────
def test_simultaneous_DIFFERENT_fixtures_are_left_alone(session):
    """60,141 of 60,976 same-league same-minute pairs are exactly this."""
    for tid, name, af in ((1, "Barcelona", 529), (2, "Sevilla", 536),
                          (3, "Real Madrid", 541), (4, "Villarreal", 533)):
        _team(session, tid, name, af)
    _match(session, 200, 1, 2)
    _match(session, 201, 3, 4)
    session.commit()
    g = resolve_fixture_groups(session, [200, 201])
    assert g[200] != g[201], (
        "two different fixtures kicking off together in one league must keep "
        "their own pick slots — refusing on the absence of evidence would "
        "reject a normal final matchday")


def test_a_different_kickoff_minute_is_a_different_bucket(session):
    _team(session, 1, "Valencia", af=532)
    _team(session, 2, "Getafe", af=546)
    _team(session, 3, "Valencia", af=532)
    _team(session, 4, "Getafe", af=546)
    _match(session, 300, 1, 2, ko=datetime(2026, 8, 30, 17, 30))
    _match(session, 301, 3, 4, ko=datetime(2026, 8, 30, 19, 30))
    session.commit()
    g = resolve_fixture_groups(session, [300, 301])
    assert g[300] != g[301], (
        "identical kickoff is the precision lever; sources that disagree on "
        "time are OUT OF REACH by construction, which is why 750 is a floor")


def test_a_different_league_is_a_different_bucket(session):
    _team(session, 1, "Valencia", af=532)
    _team(session, 2, "Getafe", af=546)
    _team(session, 3, "Valencia", af=532)
    _team(session, 4, "Getafe", af=546)
    _match(session, 400, 1, 2, league="spain/laliga")
    _match(session, 401, 3, 4, league="europe/champions-league")
    session.commit()
    g = resolve_fixture_groups(session, [400, 401])
    assert g[400] != g[401]


# ── the twin is often not a candidate ──────────────────────────────────────
def test_the_twin_is_found_even_when_it_is_not_in_the_input(session):
    """On 2026-08-31 the twin carried 78 odds and produced no pick.

    A resolver seeing only candidates would have found nothing to group. It
    matters most across runs: a pick saved earlier today on the twin must
    still consume this fixture's slot.
    """
    _team(session, 1, "RC Deportivo La Coruña", af=544)
    _team(session, 2, "Valencia", af=532)
    _team(session, 3, "Dep. A Coruna", af=None)
    _team(session, 4, "Valencia", af=532)
    _match(session, 50920, 1, 2)
    _match(session, 50927, 3, 4)
    session.commit()

    g = resolve_fixture_groups(session, [50927])       # only the twin asked for
    assert g[50927] == 50920, (
        "the canonical id must come from the whole cluster, including rows "
        "that were never candidates")


# ── never raises ───────────────────────────────────────────────────────────
def test_a_broken_session_degrades_to_row_identity_and_does_not_raise():
    class Dead:
        def execute(self, *a, **k):
            raise RuntimeError("connection lost")

    g = resolve_fixture_groups(Dead(), [1, 2, 3])
    assert g == {1: 1, 2: 2, 3: 3}, (
        "a resolver that breaks the picks run is worse than the duplicate it "
        "would have caught — degrade to the pre-s5.9 behaviour, loudly")


def test_empty_input_is_empty_output(session):
    assert resolve_fixture_groups(session, []) == {}
