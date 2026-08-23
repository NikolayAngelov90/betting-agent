"""Stage 13 Part B — the team-identity gate.

The defect: on 2026-08-13 API-Football returned `CSKA Sofia vs Maccabi Tel Aviv`
for fixture 1607568. The row holding Maccabi's API-Football team id (604) was
named `Telstar` — a Dutch second-tier club — so the away side was priced against
Eredivisie history, the published pick named the wrong opponent, and the first
leg's result was written into Telstar's record.

The correct name was present in the same response that chose the wrong row. The
gate verifies against that payload, which is why it needs no extra request and
why the check can never drift from the data it validates.
"""

from datetime import datetime

import pytest

import src.data.database as db_mod
from src.data.models import Base, Team
from src.scrapers.apifootball_scraper import names_share_an_anchor


# ─────────────────────────────────────────── the discriminator, on real data

# Every name pair below was taken from CI logs: the string API-Football sent at
# fixture-creation time, against the name stored on the row that was resolved.
ALIASES = [
    ("Union St. Gilloise", "St. Gilloise"),
    ("NEC Nijmegen", "Nijmegen"),
    ("Mjallby AIF", "Mjallby"),
    ("Fenerbahçe", "Fenerbahce"),
    ("Universitatea Craiova", "Univ. Craiova"),
    ("FC Thun", "Thun"),
    ("Ferencvarosi TC", "Ferencvaros"),
    ("Heart Of Midlothian", "Hearts"),
    ("Red Bull Salzburg", "Salzburg"),
    ("SC Braga", "Braga"),
    ("IFK Goteborg", "Goteborg"),
    ("Hammarby FF", "Hammarby"),
    ("FC Midtjylland", "Midtjylland"),
    ("FC Nordsjaelland", "Nordsjaelland"),
    ("NSI Runavik", "NSI Runavik"),
    ("FC Lugano", "Lugano"),
    ("FC Sion", "Sion"),
    ("FC Noah", "Noah"),
    ("Rapid Vienna", "SK Rapid"),
    ("FC ST. Gallen", "St. Gallen"),
    ("CFR 1907 Cluj", "CFR Cluj"),
]


@pytest.mark.parametrize("api_name,db_name", ALIASES)
def test_benign_aliases_are_not_refused(api_name, db_name):
    """19 of 20 logged disagreements were the same club under another label.

    A gate that refuses these would block most of the fixture list, be switched
    off within a week, and take the real protection with it.
    """
    assert names_share_an_anchor(api_name, db_name), (
        f"benign alias refused: {api_name!r} vs {db_name!r}")


def test_the_actual_defect_is_refused():
    """The one case in 65 that was a different club."""
    assert not names_share_an_anchor("Maccabi Tel Aviv", "Telstar")


def test_gate_is_symmetric():
    assert (names_share_an_anchor("Maccabi Tel Aviv", "Telstar")
            is names_share_an_anchor("Telstar", "Maccabi Tel Aviv"))


def test_short_tokens_do_not_anchor():
    """`FC`, `SK`, `AC` appear everywhere and identify nothing.

    Without this, every club prefixed `FC` would match every other one and the
    gate would never refuse anything — passing vacuously, which is the failure
    mode this stage has found three times.
    """
    assert not names_share_an_anchor("FC Porto", "FC Basel")
    assert not names_share_an_anchor("SK Rapid", "SK Brann")


def test_the_name_check_alone_cannot_separate_two_clubs_sharing_a_token():
    """Why the name check is not sufficient on its own.

    Rapid Vienna and Rapid Bucuresti are different clubs that share an anchor.
    This is not a hypothetical: row 467 (SK Rapid, Austria) carries ten Romanian
    matches against Dinamo Bucuresti, CFR Cluj and U. Cluj. The name check
    cannot separate them — which is why the country check exists.
    """
    assert names_share_an_anchor("Rapid Vienna", "Rapid Bucuresti")


def test_country_disagreement_is_refused_unconditionally(tmp_path):
    """The Rapid case, closed. No ratio, nothing to tune.

    A club plays in exactly one domestic league, so a Romanian domestic fixture
    cannot involve an Austrian club. Refusal here does not depend on how similar
    the names are — it is a contradiction in a field that admits only one value.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=467, name="SK Rapid", country="Austria",
                   league="austria/bundesliga", apifootball_team_id=781))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "Rapid Bucuresti", "romania/liga-1", apifootball_team_id=781,
        country="Romania")

    assert got is None, "an Austrian club was resolved for a Romanian fixture"


def test_st_pauli_pau_class_is_refused(tmp_path):
    """The second measured contamination: 14 Ligue 2 matches on a German row."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=66, name="St. Pauli", country="Germany",
                   league="germany/bundesliga", apifootball_team_id=186))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "Pau FC", "france/ligue-2", apifootball_team_id=186, country="France")
    assert got is None


def test_continental_fixtures_do_not_trigger_the_country_check(tmp_path):
    """The false positive this must not create.

    A Europa League fixture reports country "World", and a club first seen in
    one is stored as "Europe". Refusing on either would reject legitimate
    fixtures wholesale. Missing information falls through to the name check —
    which is what still catches Telstar.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=1255, name="CSKA Sofia", country="Europe",
                   league="europe/europa-league", apifootball_team_id=853))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "CSKA Sofia", "europe/europa-league", apifootball_team_id=853,
        country="World")
    assert got == 1255, "a legitimate continental fixture was refused"


def test_residual_class_two_clubs_same_country_sharing_a_token(tmp_path):
    """What remains open, pinned so it is not forgotten.

    Two clubs in the SAME country whose names share an anchor still pass both
    checks — country agrees and the names anchor. Nothing in this stage measured
    such a case, but the gate does not close it, and saying so is the point.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=900, name="Sporting Lisbon", country="Portugal",
                   league="portugal/primeira-liga", apifootball_team_id=228))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "Sporting Braga", "portugal/primeira-liga", apifootball_team_id=228,
        country="Portugal")
    assert got == 900, (
        "if this now refuses, the gate got stronger — update the claim")


# ─────────────────────────────────────────── the gate inside team resolution

def _mgr(tmp_path):
    from sqlalchemy import event

    mgr = db_mod.DatabaseManager(config=type("C", (), {
        "database": {"sqlite_path": str(tmp_path / "gate.db")}})())

    @event.listens_for(mgr.engine, "connect")
    def _fk_on(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    Base.metadata.create_all(mgr.engine)
    return mgr


def _scraper(mgr):
    from src.scrapers.apifootball_scraper import APIFootballScraper
    s = APIFootballScraper.__new__(APIFootballScraper)
    s.db = mgr
    return s


def test_resolution_refuses_a_row_whose_name_contradicts_the_payload(tmp_path):
    """Reproduces 2026-08-13 exactly, then asserts the fix.

    Team 124 holds API-Football id 604 under the name `Telstar`. The payload
    says id 604 is `Maccabi Tel Aviv`. Resolution must refuse rather than return
    the Dutch club's row.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=124, name="Telstar", league="netherlands/eredivisie",
                   apifootball_team_id=604))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "Maccabi Tel Aviv", "europe/europa-league", apifootball_team_id=604)

    assert got is None, (
        "resolution returned a team whose stored name contradicts the payload")

    with mgr.get_session() as s:
        row = s.get(Team, 124)
        assert row.name == "Telstar", "the suspect row must not be renamed"
        assert row.apifootball_team_id == 604, "and must not be re-keyed"
        assert s.query(Team).count() == 1, "nor may a parallel row be created"


def test_resolution_still_returns_the_row_for_a_benign_alias(tmp_path):
    """The gate must not break ordinary ingestion."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(id=202, name="Hearts", league="scotland/premiership",
                   apifootball_team_id=254))
        s.commit()

    got = _scraper(mgr)._get_or_create_team_id(
        "Heart Of Midlothian", "scotland/premiership", apifootball_team_id=254)

    assert got == 202
