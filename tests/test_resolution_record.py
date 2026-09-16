"""Every resolution records the INCOMING NAME beside the row it resolved to.

    ZERO ATTEMPTS AND ZERO FAILURES ARE THE SAME OBSERVATION.

On 2026-09-14 and 09-15 both the new team rows AND `resolve_team` step 2's
interceptions were zero. Under one exposure model they cannot both be — so
either the fix worked perfectly or the cards contained nothing to fix, and
nothing in the database could tell the two apart. The row a name RESOLVED TO is
stored; the name that CAME IN never was.

That made "how many resurrection attempts were there" unanswerable — not just
that week, but on every future card. One line per resolution fixes it for all of
them:

    attempts    = lines whose `name` appears in `team_former_names`
    intercepted = those with `step=former_name`
    residual    = those with `step=create`

It also closes the gap Stage 24's registration named: only step 2 announced
itself, so attribution for steps 1, 3 and 4 had to be argued from `league IS
NULL` in the data rather than read from a log.

REGISTERED BEFORE THE EXPOSURE, deliberately. The 2026-09-16 `daily-picks` run
had not fired when this landed, so the measurement exists before the card it is
meant to measure — rather than being added afterwards to explain a number.
"""

import pathlib
import re

import src.data.database as db_mod
from src.data.models import Base, Team
from src.data.team_resolution import resolve_team
from src.utils.logger import get_logger

#: A quoted value may CONTAIN SPACES ('Malmo FF'), so `\S+` stops at the first
#: one and the match fails on exactly the names this record exists to capture.
_Q = r"(?:'[^']*'|\"[^\"]*\"|None)"
_LINE = re.compile(
    rf"TEAM_RESOLVE name=({_Q}) step=(\w+) team=(\S+) "
    rf"resolved=({_Q}) league=({_Q})")


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "r.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _capture(fn):
    sink = []
    h = get_logger().add(lambda m: sink.append(str(m)), level="DEBUG")
    try:
        fn()
    finally:
        get_logger().remove(h)
    return [_LINE.search(s) for s in sink if "TEAM_RESOLVE" in s]


def test_every_step_emits_exactly_one_record(tmp_path):
    mgr = _mgr(tmp_path)

    def run():
        with mgr.get_session() as s:
            s.add(Team(name="Paris SG", league=None, apifootball_team_id=85))
            s.commit()
            resolve_team(s, "Paris SG", league="france/ligue-1")
            resolve_team(s, "PSG", league="france/ligue-1", provider_id=85)
            resolve_team(s, "Lorient", league="france/ligue-1")

    hits = _capture(run)
    assert len(hits) == 3, f"expected one record per resolution, got {len(hits)}"
    assert [h.group(2) for h in hits] == ["exact_name", "provider_id", "create"]


def test_the_INCOMING_name_is_recorded_not_only_the_resolved_one(tmp_path):
    """THE WHOLE POINT. `PSG` resolving to `Paris SG` must show BOTH."""
    mgr = _mgr(tmp_path)

    def run():
        with mgr.get_session() as s:
            s.add(Team(name="Paris SG", league=None, apifootball_team_id=85))
            s.commit()
            resolve_team(s, "PSG", league="france/ligue-1", provider_id=85)

    hit = _capture(run)[0]
    assert "PSG" in hit.group(1), (
        "the incoming name was not recorded — without it an interception is "
        "indistinguishable from a name that never arrived")
    assert "Paris SG" in hit.group(4)
    assert hit.group(3) != "None"


def test_a_create_is_distinguishable_from_an_interception(tmp_path):
    """The two counts the 09-14/09-15 ambiguity needed and did not have."""
    mgr = _mgr(tmp_path)

    def run():
        with mgr.get_session() as s:
            s.add(Team(name="Malmo FF", league=None))
            s.commit()
            resolve_team(s, "Malmö FF", league="sweden/allsvenskan")  # strict
            resolve_team(s, "Brommapojkarna", league="sweden/allsvenskan")

    steps = [h.group(2) for h in _capture(run)]
    assert steps == ["strict", "create"], steps


def test_a_refused_lookup_records_no_match(tmp_path):
    """`create=False` returning None must still leave a record.

    A caller that asks "does this resolve?" and gets None produces no row and
    no log line under a naive implementation — so the attempt vanishes, which
    is the exact absence this file exists to remove.
    """
    mgr = _mgr(tmp_path)

    def run():
        with mgr.get_session() as s:
            resolve_team(s, "Nobody FC", league="x/y", create=False)

    hits = _capture(run)
    assert len(hits) == 1 and hits[0].group(2) == "no_match"


def test_the_audit_parses_the_line_it_is_given():
    """The producer and the consumer must agree, and nothing else enforces it.

    `ci_audit.resolution_summary` reads these lines with its own regex. A format
    change here that the audit cannot parse degrades silently to "no resolution
    data", which reads identically to a run that resolved nothing.
    """
    import scripts.ci_audit as ci

    log = ("TEAM_RESOLVE name='Lens' step=former_name team=576 "
           "resolved='Racing Club de Lens' league='france/ligue-1'\n"
           "TEAM_RESOLVE name='Lorient' step=create team=9 "
           "resolved='Lorient' league='france/ligue-1'\n")
    facts = ci.extract(log)
    assert facts.get("team_resolve_steps") == {"former_name": 1, "create": 1}, (
        "the audit cannot parse the line resolve_team emits — a format drift "
        "here degrades to 'no resolution data', which reads exactly like a run "
        "that resolved nothing")
    assert "former_name=1" in ci.resolution_summary(facts)


def test_the_record_is_emitted_from_the_one_resolution_function():
    """If a second site started resolving, its resolutions would be invisible."""
    src = pathlib.Path("src/data/team_resolution.py").read_text(encoding="utf-8")
    assert src.count("_record(") >= 6, (
        "a step of resolve_team returns without recording — that step's "
        "resolutions become uncountable")
