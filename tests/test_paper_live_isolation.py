"""Stage 7, section 15 — prove paper picks are isolated from the live record.

`is_paper` was written by Stage 5 but never READ. Enabling paper mode without
these guards would have pooled measurement-only picks into live ROI, the
Telegram performance report, and the loops that change future predictions —
letting the experiment's own output rewrite the model it is measuring.

Every test runs against temp SQLite; conftest strips DATABASE_URL.
"""

from datetime import date, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, SavedPick


def _mgr(tmp_path, name="iso.db"):
    return db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())


def _agent_with(mgr):
    """A FootballBettingAgent shell wired to a temp DB, no network or models."""
    from src.agent.betting_agent import FootballBettingAgent

    a = FootballBettingAgent.__new__(FootballBettingAgent)
    a.db = mgr
    return a


def _seed(mgr, n_live=6, n_paper=6, live_win=True, paper_win=False):
    """Live picks all win at 2.0; paper picks all lose — so any leakage moves
    ROI by an unmissable amount."""
    Base.metadata.create_all(mgr.engine)
    today = date(2026, 9, 1)
    with mgr.get_session() as s:
        pid = 0
        for i in range(n_live + n_paper):
            paper = i >= n_live
            pid += 1
            s.add(Match(id=pid, home_team_id=1, away_team_id=2,
                        match_date=date(2026, 9, 1), league="x/y",
                        is_fixture=False, home_goals=1, away_goals=0))
            s.add(SavedPick(
                id=pid, match_id=pid, pick_date=today - timedelta(days=i % 5),
                market="1X2", selection="Home Win", odds=2.0,
                predicted_probability=0.55, expected_value=0.10, confidence=0.55,
                kelly_stake_percentage=1.0,
                result=("win" if (paper_win if paper else live_win) else "loss"),
                actual_home_goals=1, actual_away_goals=0,
                is_paper=paper, model_version="stage7_test",
            ))
        s.commit()
    return mgr


# ═══════════════════════════════════════════ live ROI excludes paper

def test_get_stats_excludes_paper_picks(tmp_path):
    """The headline guard. Live picks all win, paper picks all lose; if paper
    leaked in, ROI would collapse from +100% to 0%."""
    mgr = _seed(_mgr(tmp_path, "s1.db"))
    agent = _agent_with(mgr)

    class _Pred:
        def coverage_summary(self):
            return {"poisson_teams": 0, "elo_teams": 0, "ml_fitted": False,
                    "goals_ml_fitted": False}

    agent.predictor = _Pred()
    stats = agent.get_stats()

    all_time = stats["all_time"]
    assert all_time["total"] == 6, f"paper picks leaked into live ROI: {all_time}"
    assert all_time["wins"] == 6 and all_time["losses"] == 0
    assert all_time["roi"] == pytest.approx(1.0), "live ROI was contaminated"


def test_rolling_backtest_excludes_paper_picks(tmp_path, capsys):
    mgr = _seed(_mgr(tmp_path, "s2.db"))
    agent = _agent_with(mgr)
    agent.rolling_backtest()
    out = capsys.readouterr().out
    # 6 live wins, 0 losses. A leak would show 12 decided picks.
    assert "     6     6     0" in out or "6      6     0" in out.replace("  ", " "), out


# ═════════════════════════════ paper cannot change future predictions

def _run_ev_calibration(tmp_path, name, *, n_live, n_paper, persisted_ev,
                        monkeypatch, live_win=True):
    """Run _auto_calibrate_ev_threshold against an ISOLATED models directory.

    Stage 12.1, Defect 1. The previous version of this test read the real
    `data/models/ev_threshold.json`, which CI restores from the ML cache — so it
    passed locally (that file happens to hold 0.05) and failed in CI with the
    thoroughly misleading message "paper losses moved the live EV threshold".
    Nothing about paper picks was involved: `_auto_calibrate_ev_threshold`
    assigns `min_ev` from the persisted file BEFORE it queries the database.

    Redirecting MODELS_DIR makes the persisted value an input of the test rather
    than an accident of the environment.
    """
    import json

    import src.models.ml_models as mlm
    from src.betting.value_calculator import ValueBettingCalculator

    models_dir = tmp_path / f"models_{name}"
    models_dir.mkdir()
    (models_dir / "ev_threshold.json").write_text(json.dumps(
        {"min_ev": persisted_ev, "roi": -0.25, "n_picks": 40}))
    # _auto_calibrate_ev_threshold imports MODELS_DIR inside the function, so
    # patching the module attribute is picked up at call time.
    monkeypatch.setattr(mlm, "MODELS_DIR", models_dir)

    mgr = _mgr(tmp_path, f"{name}.db")
    _seed(mgr, n_live=n_live, n_paper=n_paper, live_win=live_win,
          paper_win=False)
    agent = _agent_with(mgr)

    class _Cfg:
        betting = {"min_expected_value": 0.05}

        def get(self, key, default=None):
            return {"models.ev_calibration_lookback": 40}.get(key, default)

    agent.config = _Cfg()
    agent.value_calculator = ValueBettingCalculator.__new__(ValueBettingCalculator)
    agent.value_calculator.min_ev = 0.05
    agent._auto_calibrate_ev_threshold()
    return agent


@pytest.mark.parametrize("persisted_ev", [0.05, 0.07])
def test_ev_threshold_calibration_ignores_paper_picks(tmp_path, monkeypatch,
                                                      persisted_ev):
    """The invariant, stated so the persisted value cannot decide the outcome.

    With no LIVE settled picks the calibration has nothing to recompute from,
    so the threshold must come out exactly as persisted — whatever that is.
    Parametrised over two values precisely because the old test only passed
    when the environment happened to supply 0.05.
    """
    agent = _run_ev_calibration(tmp_path, f"paper{int(persisted_ev * 100)}",
                                n_live=0, n_paper=40,
                                persisted_ev=persisted_ev,
                                monkeypatch=monkeypatch)

    # 40 paper losses are a -100% ROI cold streak. If they counted, min_ev
    # would be tightened well above the persisted value.
    assert agent.value_calculator.min_ev == persisted_ev, (
        "paper losses moved the live EV threshold")
    assert getattr(agent, "_recent_roi", None) is None, (
        "paper picks were counted as recent live ROI")


@pytest.mark.parametrize("persisted_ev", [0.05, 0.07])
def test_ev_threshold_is_identical_with_and_without_paper_picks(
        tmp_path, monkeypatch, persisted_ev):
    """The differential form: paper picks change nothing at all."""
    with_paper = _run_ev_calibration(
        tmp_path, f"w{int(persisted_ev * 100)}", n_live=0, n_paper=40,
        persisted_ev=persisted_ev, monkeypatch=monkeypatch)
    without = _run_ev_calibration(
        tmp_path, f"n{int(persisted_ev * 100)}", n_live=0, n_paper=0,
        persisted_ev=persisted_ev, monkeypatch=monkeypatch)

    assert with_paper.value_calculator.min_ev == without.value_calculator.min_ev, (
        "adding 40 paper picks changed the EV threshold")
    assert getattr(with_paper, "_recent_roi", None) is None
    assert getattr(without, "_recent_roi", None) is None


def test_ev_threshold_calibration_does_respond_to_live_picks(tmp_path,
                                                             monkeypatch):
    """Guards against the whole suite passing vacuously.

    The tests above prove "nothing happened". They would also pass if
    `_auto_calibrate_ev_threshold` were a no-op, or if its query returned
    nothing for an unrelated reason. This one shows the machinery is live and
    that `_live_only()` is what excludes paper picks: the same 40 losing picks,
    flagged LIVE instead of paper, must move the threshold.
    """
    agent = _run_ev_calibration(tmp_path, "livecold", n_live=40, n_paper=0,
                                persisted_ev=0.05, monkeypatch=monkeypatch,
                                live_win=False)

    assert agent.value_calculator.min_ev > 0.05, (
        "40 LIVE losses did not tighten min_ev — the calibration is inert, so "
        "the paper-isolation tests above prove nothing")
    assert getattr(agent, "_recent_roi", None) is not None, (
        "live picks were not counted as recent ROI")


def test_tune_ensemble_weights_ignores_paper_picks(tmp_path):
    """Paper picks must not move the weight learner: the frozen model would
    drift while model_version stayed constant."""
    import asyncio

    mgr = _mgr(tmp_path, "s4.db")
    _seed(mgr, n_live=0, n_paper=30)
    agent = _agent_with(mgr)
    result = asyncio.run(agent.tune_ensemble_weights())
    # With no LIVE settled picks it must bail out before touching any weights.
    assert result is None


def test_pick_calibration_ignores_paper_picks(tmp_path, monkeypatch, tmp_path_factory):
    mgr = _mgr(tmp_path, "s5.db")
    _seed(mgr, n_live=0, n_paper=40)
    agent = _agent_with(mgr)

    import src.agent.betting_agent as ba
    written = {}
    monkeypatch.setattr(ba.Path, "write_text",
                        lambda self, txt, *a, **k: written.update({str(self): txt}))
    monkeypatch.setattr(ba.Path, "mkdir", lambda self, *a, **k: None)
    monkeypatch.setattr(ba.Path, "exists", lambda self: False)

    class _Pred:
        pick_calibration = {}

    agent.predictor = _Pred()
    factors = agent.calibrate_from_pick_outcomes()
    assert factors == {}, f"paper picks produced calibration factors: {factors}"


# ═════════════════════════════════ paper IS included in the experiment

def test_paper_picks_are_visible_to_the_experiment_report(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "s6.db")
    _seed(mgr, n_live=3, n_paper=7)
    import scripts.paper_trading_report as rep
    monkeypatch.setattr(rep, "load_picks", rep.load_picks)
    import src.data.database as dbm
    monkeypatch.setattr(dbm, "_db_manager", mgr, raising=False)
    monkeypatch.setattr(dbm, "get_db", lambda: mgr)

    picks = rep.load_picks(days=3650, include_live=False, model_version=None)
    assert len(picks) == 7, "paper picks are not visible to the experiment report"
    assert all(p.is_paper for p in picks)

    both = rep.load_picks(days=3650, include_live=True, model_version=None)
    assert len(both) == 10


def test_historical_non_paper_picks_are_untouched(tmp_path):
    """Stage 7 must not rewrite the 1,070 historical rows."""
    mgr = _seed(_mgr(tmp_path, "s7.db"))
    with mgr.get_session() as s:
        live = s.query(SavedPick).filter(SavedPick.is_paper == False).all()  # noqa: E712
        assert len(live) == 6
        for p in live:
            assert p.result == "win"
            assert p.is_paper is False


def test_live_only_treats_null_is_paper_as_live(tmp_path):
    """A deployment whose column was added without a default reads NULL. Those
    rows are real history and must not vanish from the live record."""
    from sqlalchemy import text

    mgr = _mgr(tmp_path, "s8.db")
    _seed(mgr, n_live=2, n_paper=0)
    with mgr.get_session() as s:
        s.execute(text("UPDATE saved_picks SET is_paper = NULL"))
        s.commit()

    agent = _agent_with(mgr)

    class _Pred:
        def coverage_summary(self):
            return {"poisson_teams": 0, "elo_teams": 0, "ml_fitted": False,
                    "goals_ml_fitted": False}

    agent.predictor = _Pred()
    assert agent.get_stats()["all_time"]["total"] == 2, (
        "NULL is_paper rows were dropped from the live record")


# ═══════════════════════════════════════════ config safety

def test_paper_mode_is_explicit_and_currently_on():
    """Stage 7 section 11 turned paper trading ON after the safety checklist.

    The value is asserted rather than left free so that flipping it back to
    real-money operation is a deliberate act that breaks a test, not a quiet
    config edit. If you intend to leave paper mode, change this test in the
    same commit and say why.
    """
    import pathlib

    import yaml

    cfg = yaml.safe_load(
        pathlib.Path("config/config.example.yaml").read_text(encoding="utf-8"))
    value = cfg["betting"]["paper_trading_mode"]
    assert isinstance(value, bool), "paper_trading_mode must be an explicit bool"
    assert value is True, (
        "paper_trading_mode is off. Stages 5-7 concluded PAPER TRADING ONLY "
        "until 500 valid closing lines exist; turning it off needs a documented "
        "decision, not a config tweak.")


def test_saving_a_pick_stamps_is_paper_from_config(tmp_path, monkeypatch):
    """The flag must come from config, not be hardcoded."""
    import inspect

    from src.agent.betting_agent import FootballBettingAgent

    src = inspect.getsource(FootballBettingAgent._save_picks)
    assert "betting.paper_trading_mode" in src
    assert "is_paper=_paper_mode" in src


# ═══════════════════════ paper picks are not disguised as live bets

def test_telegram_message_carries_a_paper_banner():
    """This message IS the betting action — the reader places bets from it by
    hand. A measurement-only pick that looks identical to a live recommendation
    is a money-safety problem."""
    import asyncio
    from types import SimpleNamespace

    from src.reporting.telegram_bot import TelegramNotifier

    sent = []
    n = TelegramNotifier.__new__(TelegramNotifier)
    n.enabled = True
    n._get_bot = lambda: object()
    n._send_message = lambda msg, **kw: sent.append(msg) or asyncio.sleep(0)

    pick = SimpleNamespace(
        match="A vs B", market="1X2", selection="Home Win", odds=2.0,
        predicted_probability=0.55, expected_value=0.10, confidence=0.55,
        kelly_stake_percentage=1.0, recommended_stake=1.0, reasoning="",
        risk_level="medium", league="x/y", match_date=None, model_agreement="",
        models_for="", models_against="", injury_impact="", h2h_insight="",
        form_insight="", xg_edge="", predicted_xg="", contrarian_value=0.0,
        home_xg_avg=0.0, away_xg_avg=0.0, match_id=1, opening_odds=0.0,
        used_fallback_odds=False, models_agree=None, market_probability=0.0,
        market_books=0,
    )

    asyncio.run(n.send_daily_picks([pick], stats={}, paper_mode=True, force=True))
    assert sent, "no message was produced"
    assert "PAPER TRADING" in sent[0]
    assert "DO NOT BET REAL MONEY" in sent[0]

    sent.clear()
    asyncio.run(n.send_daily_picks([pick], stats={}, paper_mode=False, force=True))
    assert sent and "PAPER TRADING" not in sent[0], (
        "the live message must not carry the paper banner")


def test_picks_send_passes_the_config_flag_through():
    import inspect

    import src.agent.betting_agent as ba

    src = inspect.getsource(ba.main)
    assert "paper_mode=bool(agent.config.get(" in src
    assert '"betting.paper_trading_mode"' in src
