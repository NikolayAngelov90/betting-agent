"""Stage 23: every team row comes through ONE function, and the list is enforced.

Three creation paths with three matching regimes produced three blind spots, and
the patch for one was itself blind in the reverse direction. A fourth site would
produce a fourth blind spot, so a fourth site must fail the suite instead.

Same move as `test_no_test_writes_prod_state` (the path list) and
`test_overround_band_is_one_definition` (the band): the guarantee is
unenforceable by care, so pin the claim and make changing it deliberate.
"""

import pathlib
import re

import src.data.database as db_mod
from src.data.models import Base, Team
from src.data.team_resolution import resolve_team

#: Files allowed to construct a Team directly, with the reason.
TEAM_CONSTRUCTION_ALLOWED = {
    # THE one resolution function. Its step 5 is the only sanctioned create.
    "src/data/team_resolution.py",
    # The ORM declaration and its __repr__. `class Team(Base)` and
    # f"<Team(name=...)>" are not construction sites.
    "src/data/models.py",
}

_CONSTRUCT = re.compile(r"(?<![\w.])Team\s*\(")


def test_no_site_constructs_a_team_outside_the_resolution_function():
    offenders = []
    for p in sorted(pathlib.Path("src").rglob("*.py")):
        rel = str(p).replace("\\", "/")
        if rel in TEAM_CONSTRUCTION_ALLOWED:
            continue
        text = p.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            s = line.strip()
            if s.startswith("#") or s.startswith("*"):
                continue
            if _CONSTRUCT.search(line):
                offenders.append(f"{rel}:{i}: {s[:70]}")
    assert not offenders, (
        "these sites construct a Team directly instead of calling "
        "resolve_team():\n  " + "\n  ".join(offenders)
        + "\n\nThree such sites produced three different blind spots and 44 "
          "resurrections in three days. Route it through resolve_team(), or "
          "add the file to TEAM_CONSTRUCTION_ALLOWED with the reason.")


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "r.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


# ── the five steps, in order ──────────────────────────────────────────────

def test_step1_provider_id_wins(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Racing Club de Lens", league=None,
                   apifootball_team_id=116))
        s.commit()
        got = resolve_team(s, "Lens", league="france/ligue-1", provider_id=116)
        assert got.name == "Racing Club de Lens"
        assert s.query(Team).count() == 1


def test_step3_exact_name_is_not_league_scoped(tmp_path):
    """The survivor carries league=None; the scrape carries a real league."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Paris SG", league=None, apifootball_team_id=85))
        s.commit()
        got = resolve_team(s, "Paris SG", league="france/ligue-1")
        assert got.apifootball_team_id == 85
        assert s.query(Team).count() == 1


def test_step4_strict_match_without_a_league_filter(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Malmo FF", league=None))
        s.commit()
        got = resolve_team(s, "Malmö FF", league="sweden/allsvenskan")
        assert got.name == "Malmo FF"
        assert s.query(Team).count() == 1


def test_step5_creates_only_when_everything_refuses(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Paris SG", league=None))
        s.commit()
        got = resolve_team(s, "Lorient", league="france/ligue-1")
        assert got.name == "Lorient"
        assert s.query(Team).count() == 2


def test_distinct_same_city_clubs_stay_apart(tmp_path):
    """Widening WHICH rows are compared must not widen WHAT counts as a match."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Sheffield Wednesday", league=None))
        s.commit()
        got = resolve_team(s, "Sheffield United", league="england/championship")
        assert got.name == "Sheffield United"
        assert s.query(Team).count() == 2


def test_a_missing_former_names_table_fails_open(tmp_path):
    """Migration 010 unapplied must degrade, not break.

    The pipeline falls back to pre-Stage-23 behaviour — the resurrections
    return, which is the observable cost, and it is preferable to a crash.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Racing Club de Lens", league=None))
        s.commit()
        got = resolve_team(s, "Lens", league="france/ligue-1")
        assert got is not None          # created; no table, so step 2 is silent


# ── the country check, applied from the AF-id gate ────────────────────────

def test_the_country_check_refuses_a_cross_country_name_match(tmp_path):
    """Arsenal (England) must not become Arsenal FC (Argentina).

    THE SAME CHECK THE AF-ID GATE USES, applied to close an asymmetry rather
    than to fix an observed failure: that gate refused a cross-country match
    while this name path permitted one. Measured before applying — of the 25
    name-path joins resolve_team could then make, it refuses ZERO.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Arsenal FC", league="argentina/primera",
                   country="Argentina"))
        s.commit()
        got = resolve_team(s, "Arsenal", league="england/premier-league",
                           country="England")
        assert got.country == "England", (
            "strict-matched across a real country boundary — the AF-id gate "
            "refuses this and the name path must too")
        assert s.query(Team).count() == 2


def test_the_country_check_fails_OPEN_on_a_missing_country(tmp_path):
    """`teams.country` records where a club was FIRST SEEN, not where it plays.

    Levski Sofia is stored as "Europe" because a Conference League tie created
    it. Refusing on a missing or continental value would reject legitimate
    fixtures wholesale, so absence must fall through — 56% of rows carry no
    real country.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Malmo FF", league=None, country=None))
        s.commit()
        got = resolve_team(s, "Malmö FF", league="sweden/allsvenskan",
                           country="Sweden")
        assert got.name == "Malmo FF", (
            "a missing stored country caused a refusal — the check must fail "
            "open, or it rejects the 56% of rows that carry no real country")
        assert s.query(Team).count() == 1


def test_a_continental_country_value_also_falls_through(tmp_path):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add(Team(name="Levski Sofia", league=None, country="Europe"))
        s.commit()
        got = resolve_team(s, "Levski Sofia", league="bulgaria/efbet-league",
                           country="Bulgaria")
        assert got.country == "Europe"
        assert s.query(Team).count() == 1
