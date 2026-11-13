"""The standard rule keeps being computed and stored -- never sent. s5.23.

The switch to `confident_value` was decided on a PAIRED comparison: on the
same fixtures, the standard rule won 58.3% at 1.63 and lost 5.5%; the new
rule at P_min 0.54 won 57.4% at 1.73 and lost 0.5%. For that comparison to
continue prospectively the standard rule's pick has to exist for every
fixture the new rule sees -- so it is computed and stored in `shadow_picks`:

    never in `saved_picks`   so no report, settlement or card reads it
    never sent               the client card is built from the real picks
    never staked             there is no stake to read
    graded on read           settlement is untouched

DRIVEN on scratch SQLite through `_recommendations_for_fixture` and
`_save_standard_shadow`, the two methods the picks path calls, with the
standard mode as the known-wrong control (it stores nothing, because its own
picks ARE the standard rule).
"""
import ast
import datetime as dt
import pathlib
from datetime import datetime, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, SavedPick, ShadowPick, Team
from tests.test_zero_pick_states import (NOTHING, OVER35, PRED, _agent, _cfg,
                                         _odds_data)

D = dt.date(2026, 11, 14)
T0 = datetime(2026, 11, 14, 18, 0)


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(config=type(
        "C", (), {"database": {"sqlite_path": str(tmp_path / "s.db")}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A"))
        s.add(Team(id=2, name="B"))
        s.flush()
        for mid in (7, 8):
            s.add(Match(id=mid, home_team_id=1, away_team_id=2, league="x/y",
                        match_date=T0))
        s.commit()
    return mgr


def _run(agent, rows, match_id):
    return agent._recommendations_for_fixture(
        PRED, _odds_data(rows), "A vs B", {}, home_team_name="A",
        away_team_name="B", match_id=match_id,
        league="england/premier-league", kickoff=T0 - timedelta(hours=1),
        home_id=None, away_id=None, coverage_score=0.95, low_coverage=False)


def _shadow_rows(mgr):
    with mgr.get_session() as s:
        return [(r.pick_date, r.match_id, r.rule, r.selection, r.odds,
                 r.is_forced) for r in s.query(ShadowPick)
                .order_by(ShadowPick.match_id)]


def test_DRIVEN_the_NEW_MODE_stores_the_STANDARD_pick_and_saves_NO_pick(
        tmp_path):
    """Fixture 7: the new rule picks Over 3.5 and the standard rule picks
    something else (it excludes over_3.5). Fixture 8: the new rule picks
    nothing and the standard rule forces its pick. Both standard picks are
    stored; `saved_picks` is untouched."""
    mgr = _mgr(tmp_path)
    agent = _agent(_cfg(), OVER35)            # the deployed config: switched
    agent.db = mgr
    agent._should_force_pick = lambda league, cov: True
    agent._standard_shadow = {}
    picked, confident = _run(agent, OVER35, 7)
    assert confident is True
    assert [r.selection for r in picked] == ["Over 3.5 Goals"]
    agent._odds_rows_with_time = lambda match_id, kickoff=None: list(NOTHING)
    none, _ = _run(agent, NOTHING, 8)
    assert none == []

    assert agent._save_standard_shadow(D, 1) == 2
    rows = _shadow_rows(mgr)
    assert [(m, rule) for _d, m, rule, *_ in rows] == [(7, "standard"),
                                                      (8, "standard")]
    by = {m: (sel, forced) for _d, m, _r, sel, _o, forced in rows}
    assert by[7][0] != "Over 3.5 Goals", (
        "the shadow for fixture 7 is the NEW rule's pick, not the standard's")
    assert by[8][1] is True, "fixture 8's standard pick is a forced pick"
    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 0
    # idempotent: a re-run the same day writes nothing new
    assert agent._save_standard_shadow(D, 1) == 0
    assert len(_shadow_rows(mgr)) == 2


def test_KNOWN_WRONG_the_STANDARD_mode_stores_NO_shadow(tmp_path):
    mgr = _mgr(tmp_path)
    agent = _agent(_cfg({"betting.selection_mode": "standard"}), OVER35)
    agent.db = mgr
    agent._should_force_pick = lambda league, cov: True
    agent._standard_shadow = {}
    recs, confident = _run(agent, OVER35, 7)
    assert confident is False and recs, "the standard rule picked"
    assert agent._standard_shadow == {}
    assert agent._save_standard_shadow(D, 1) == 0
    assert _shadow_rows(mgr) == []


def test_the_SHADOW_ranks_by_the_SAME_KEY_as_finalize_picks():
    """One definition, module level. From the AST: `finalize_picks` binds
    `_rank_key = standard_rank_key` and `_save_standard_shadow` sorts by
    `standard_rank_key` -- so the shadow's 'best' is the standard rule's
    'best' and cannot drift from it."""
    src = pathlib.Path("src/agent/betting_agent.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef)}
    fin = ast.get_source_segment(src, fns["finalize_picks"]) or ""
    sav = ast.get_source_segment(src, fns["_save_standard_shadow"]) or ""
    assert "_rank_key = standard_rank_key" in fin
    assert "key=standard_rank_key" in sav
    module_level = {n.name for n in tree.body
                    if isinstance(n, ast.FunctionDef)}
    assert "standard_rank_key" in module_level


def test_the_RANK_KEY_is_the_one_finalize_picks_always_used():
    """The extracted key, driven: EV x confidence x agreement x contrarian,
    ties broken by (match_id, selection)."""
    from types import SimpleNamespace

    from src.agent.betting_agent import standard_rank_key

    def r(ev, conf, agree, cv=0.0, mid=1, sel="X"):
        return SimpleNamespace(expected_value=ev, confidence=conf,
                               model_agreement=agree, contrarian_value=cv,
                               match_id=mid, selection=sel)
    a = r(0.10, 0.60, "unanimous")                 # 0.06 x 1.15
    b = r(0.10, 0.60, "split")                     # 0.06 x 0.85
    c = r(0.10, 0.60, "unanimous", cv=1.4)         # x 1.10 contrarian
    assert standard_rank_key(a)[0] == pytest.approx(-0.10 * 0.60 * 1.15)
    assert standard_rank_key(c)[0] == pytest.approx(-0.10 * 0.60 * 1.15 * 1.10)
    assert sorted([b, a, c], key=standard_rank_key) == [c, a, b]
    tie1, tie2 = r(0.1, 0.5, "solo", mid=2, sel="B"), r(0.1, 0.5, "solo",
                                                         mid=2, sel="A")
    assert sorted([tie1, tie2], key=standard_rank_key) == [tie2, tie1]


def test_a_SHADOW_FAILURE_cannot_cost_the_REAL_pick(tmp_path):
    """The shadow's computation is caught: a standard rule that raises
    leaves the new rule's pick exactly as it was."""
    agent = _agent(_cfg(), OVER35)
    agent._should_force_pick = lambda league, cov: True

    def _boom(*_a, **_k):
        raise RuntimeError("standard rule broke")
    agent.value_calculator.find_value_bets = _boom
    agent._standard_shadow = {}
    recs, confident = _run(agent, OVER35, 7)
    assert confident is True
    assert [r.selection for r in recs] == ["Over 3.5 Goals"]
    assert agent._standard_shadow == {}

    class _Broken:
        def get_session(self):
            raise RuntimeError("database gone")
    agent.db = _Broken()
    agent._standard_shadow = {7: recs}
    assert agent._save_standard_shadow(D, 1) == 0
