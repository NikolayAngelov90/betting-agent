"""Stage 10.3 — a pick and its attribution rows commit together, or not at all.

The invariant:

    A newly created pick must never commit without both its `model` and `final`
    observation rows. If observation persistence fails, the entire pick-save
    transaction fails and rolls back.

Why losing the pick is the right failure. The MODEL observation is the only
place the frozen model's taken price will exist once the Claude review runs:
`_apply_decision` overwrites `saved_picks.odds`, and the odds table keeps one
row per (match, bookmaker, market, selection) which every refresh overwrites, so
nothing can reconstruct it afterwards. A pick saved without observations looks
identical in the report to one that was never measurable — the experiment loses
rows it does not know it lost. A missing pick is visible; a silently
unattributable one is not.

Every test asserts the DATABASE STATE after the failure, not merely that an
exception was raised.

Temp SQLite throughout; conftest strips DATABASE_URL.
"""

from datetime import date, datetime
from types import SimpleNamespace

import pytest
from sqlalchemy import event

import src.agent.betting_agent as ba
import src.data.database as db_mod
from src.agent.betting_agent import (FootballBettingAgent,
                                     PickObservationsUnavailable)
from src.data.models import Base, Match, PickObservation, SavedPick, Team


@pytest.fixture(autouse=True)
def _reset_preflight_cache():
    """The preflight caches per process; each test needs a clean slate."""
    ba._PICK_OBSERVATIONS_READY = False
    yield
    ba._PICK_OBSERVATIONS_READY = False


def _mgr(tmp_path, name, n_matches=2, with_table=True):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())

    @event.listens_for(mgr.engine, "connect")
    def _fk_on(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys = ON")
        cur.close()

    Base.metadata.create_all(mgr.engine)
    if not with_table:
        with mgr.engine.begin() as c:
            c.exec_driver_sql("DROP TABLE pick_observations")
    with mgr.get_session() as s:
        s.add(Team(id=1, name="Home FC"))
        s.add(Team(id=2, name="Away FC"))
        for i in range(1, n_matches + 1):
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.commit()
    return mgr


def _agent(mgr):
    a = FootballBettingAgent.__new__(FootballBettingAgent)
    a.db = mgr
    a.config = SimpleNamespace(get=lambda k, d=None: d)
    return a


def _pick(match_id, market="1X2", selection="Home Win", odds=2.0):
    return SimpleNamespace(
        match_id=match_id, match=f"H vs A {match_id}", league="x/y",
        market=market, selection=selection, odds=odds,
        predicted_probability=0.55, expected_value=0.10, confidence=0.55,
        kelly_stake_percentage=1.0, risk_level="medium",
        used_fallback_odds=False, model_agreement="unanimous",
        market_probability=0.50, market_books=8)


def _counts(mgr):
    with mgr.get_session() as s:
        picks = s.query(SavedPick).count()
        try:
            obs = s.query(PickObservation).count()
        except Exception:
            obs = None                      # table absent
    return picks, obs


# ══════════════════════════════ Test A — table missing

def test_A_missing_table_aborts_before_any_pick_is_inserted(tmp_path):
    mgr = _mgr(tmp_path, "a.db", with_table=False)

    with pytest.raises(PickObservationsUnavailable) as excinfo:
        _agent(mgr)._save_picks([_pick(1), _pick(2)], date(2026, 9, 1))

    msg = str(excinfo.value)
    assert "pick_observations" in msg
    assert "006" in msg, "the error does not name the migration to apply"

    picks, _ = _counts(mgr)
    assert picks == 0, "picks were inserted despite the missing table"


def test_A2_preflight_runs_before_any_analysis_or_quota_spend():
    """Placement matters as much as existence: everything between the start of
    get_daily_picks and the save spends API-Football quota. Discovering the
    problem afterwards would burn a day's budget producing picks we refuse to
    store."""
    import inspect

    src = inspect.getsource(FootballBettingAgent.get_daily_picks)
    body = src[src.index("target = target_date or date.today()"):]
    guard = body.index("_require_pick_observations(self.db)")
    # Nothing that touches the network or the analysis pipeline may precede it.
    preceding = body[:guard]
    for forbidden in ("await ", "scrape", "create_features", "predict",
                      "find_value_bets"):
        assert forbidden not in preceding, (
            f"{forbidden!r} runs before the pick_observations preflight")


# ═══════════════════════ Test B — observation constraint failure

def test_B_constraint_failure_rolls_back_the_whole_save(tmp_path, monkeypatch):
    """A NOT NULL violation on the observation must take the pick with it."""
    mgr = _mgr(tmp_path, "b.db")
    real = ba._write_pick_observations
    monkeypatch.setattr(
        ba, "_write_pick_observations",
        lambda session, pick_id, **kw: real(
            session, pick_id, **{**kw, "market": None}))

    with pytest.raises(Exception):
        _agent(mgr)._save_picks([_pick(1)], date(2026, 9, 1))

    assert _counts(mgr) == (0, 0), (
        "a partial pick survived a failed observation write")


def test_B2_arbitrary_exception_also_rolls_back(tmp_path, monkeypatch):
    """Not only database errors. Any failure in the observation write must
    abort — the swallow this replaced caught everything."""
    mgr = _mgr(tmp_path, "b2.db")
    monkeypatch.setattr(ba, "_write_pick_observations",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("boom")))

    with pytest.raises(RuntimeError):
        _agent(mgr)._save_picks([_pick(1)], date(2026, 9, 1))

    assert _counts(mgr) == (0, 0)


# ═══════════════════ Test C — heterogeneous batch (the Stage 10.1 B3 case)

def test_C_second_pick_failing_rolls_back_the_first_too(tmp_path, monkeypatch):
    """The measured Stage 10.1 failure: pick 1 attributable, pick 2 silently
    not, both committed. The batch is one transaction and must behave like it.
    """
    mgr = _mgr(tmp_path, "c.db")
    real = ba._write_pick_observations
    calls = {"n": 0}

    def flaky(session, pick_id, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            kw = {**kw, "market": None}         # second pick fails
        return real(session, pick_id, **kw)

    monkeypatch.setattr(ba, "_write_pick_observations", flaky)

    with pytest.raises(Exception):
        _agent(mgr)._save_picks([_pick(1), _pick(2)], date(2026, 9, 1))

    picks, obs = _counts(mgr)
    assert picks == 0, f"{picks} pick(s) survived a failed batch"
    assert obs == 0, f"{obs} observation(s) survived a failed batch"
    assert calls["n"] == 2, "the batch did not reach the failing pick"


def test_C2_delta_is_zero_against_a_populated_table(tmp_path, monkeypatch):
    """Deltas, not absolutes: a failed batch must leave PRE-EXISTING rows alone
    and add nothing."""
    mgr = _mgr(tmp_path, "c2.db", n_matches=3)
    _agent(mgr)._save_picks([_pick(1)], date(2026, 9, 1))
    before = _counts(mgr)
    assert before == (1, 2)

    real = ba._write_pick_observations
    monkeypatch.setattr(
        ba, "_write_pick_observations",
        lambda session, pick_id, **kw: real(
            session, pick_id, **{**kw, "market": None}))

    with pytest.raises(Exception):
        _agent(mgr)._save_picks([_pick(2), _pick(3)], date(2026, 9, 1))

    assert _counts(mgr) == before, (
        "a failed batch changed the database; delta must be zero")


# ═══════════════════════════════ Test D — the normal path

def test_D_success_writes_one_pick_and_two_observations(tmp_path):
    mgr = _mgr(tmp_path, "d.db")
    _agent(mgr)._save_picks(
        [_pick(1, "Over 2.5", "Over 2.5 Goals", 1.85)], date(2026, 9, 1))

    assert _counts(mgr) == (1, 2)
    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}
    assert set(rows) == {"model", "final"}
    for o in rows.values():
        assert (o.market, o.selection) == ("Over 2.5", "Over 2.5 Goals")
        assert o.taken_odds == pytest.approx(1.85)
        assert o.closing_status == "pending"
    # KEEP semantics: identical rows, so capture resolves ONE close for both.
    assert rows["model"].taken_at == rows["final"].taken_at


def test_D2_change_still_yields_two_independent_observations(tmp_path):
    """Stage 10 behaviour preserved: after a switch the two series carry
    different markets, selections and prices, and taken_at does not move."""
    from src.reporting.match_briefing import _update_final_observation

    mgr = _mgr(tmp_path, "d2.db")
    _agent(mgr)._save_picks(
        [_pick(1, "Over 2.5", "Over 2.5 Goals", 1.85)], date(2026, 9, 1))
    with mgr.get_session() as s:
        taken_at = s.query(PickObservation).first().taken_at
        p = s.query(SavedPick).one()
        p.market, p.selection, p.odds = "1X2", "Home Win", 2.10
        p.review_action = "CHANGE"
        _update_final_observation(s, p.id, market="1X2",
                                  selection="Home Win", taken_odds=2.10)
        s.commit()

    with mgr.get_session() as s:
        rows = {o.attribution: o for o in s.query(PickObservation).all()}
    assert (rows["model"].market, rows["model"].selection) == \
        ("Over 2.5", "Over 2.5 Goals")
    assert rows["model"].taken_odds == pytest.approx(1.85)
    assert (rows["final"].market, rows["final"].selection) == ("1X2", "Home Win")
    assert rows["final"].taken_odds == pytest.approx(2.10)
    assert rows["model"].taken_at == rows["final"].taken_at == taken_at


def test_D3_batch_of_several_picks_all_get_observations(tmp_path):
    mgr = _mgr(tmp_path, "d3.db", n_matches=3)
    _agent(mgr)._save_picks([_pick(1), _pick(2), _pick(3)], date(2026, 9, 1))
    assert _counts(mgr) == (3, 6)


# ═════════════════════ Test E — the preflight is per run, not per pick

def test_E_preflight_queries_once_per_run(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "e.db", n_matches=3)

    from sqlalchemy import inspect as _real_inspect
    calls = {"n": 0}

    def counting_inspect(target):
        calls["n"] += 1
        return _real_inspect(target)

    # _require_pick_observations does `from sqlalchemy import inspect` inside
    # the function, so it resolves through the module at call time and this
    # patch is what it will use.
    import sqlalchemy
    monkeypatch.setattr(sqlalchemy, "inspect", counting_inspect)

    _agent(mgr)._save_picks([_pick(1), _pick(2), _pick(3)], date(2026, 9, 1))

    assert calls["n"] == 1, (
        f"the table check ran {calls['n']} times for 3 picks — it must be "
        f"once per run")
    assert _counts(mgr) == (3, 6)


def test_E2_cache_is_not_consulted_before_the_first_check(tmp_path):
    """A stale cache from a previous process must not let a run start against a
    database that lacks the table."""
    assert ba._PICK_OBSERVATIONS_READY is False
    mgr = _mgr(tmp_path, "e2.db", with_table=False)
    with pytest.raises(PickObservationsUnavailable):
        _agent(mgr)._save_picks([_pick(1)], date(2026, 9, 1))
    assert ba._PICK_OBSERVATIONS_READY is False, (
        "a failed preflight marked the table as ready")


# ═════════════════════════ the frozen model is untouched

def test_stage103_changed_no_model_identity():
    from src.models.model_version import CODE_REVISION, model_version
    from src.utils.config import Config

    from tests.test_config_identity import FROZEN_MODEL_VERSION

    assert CODE_REVISION == "s5.2"
    assert model_version(Config("config/config.example.yaml")) == \
        FROZEN_MODEL_VERSION
