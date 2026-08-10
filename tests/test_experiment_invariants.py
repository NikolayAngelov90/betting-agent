"""Stage 8, Phase 11 — the ten invariants the prospective experiment rests on.

These are not unit tests of convenience. Each one corresponds to a way the
100/200/500-observation experiment could produce a confident, wrong answer.
If one of these fails, the data collected after that point is not evidence.

Every test runs against temp SQLite; conftest strips DATABASE_URL.
"""

from datetime import date, datetime, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, SavedPick, Team


def _mgr(tmp_path, name):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _rec(match_id, selection, market="1X2", ev=0.10, conf=0.55,
         agreement="unanimous", match="A vs B"):
    """A BetRecommendation-shaped object for the portfolio-phase filters."""
    from types import SimpleNamespace
    return SimpleNamespace(
        match=match, match_id=match_id, market=market, selection=selection,
        odds=2.0, predicted_probability=conf, expected_value=ev,
        confidence=conf, kelly_stake_percentage=1.0, model_agreement=agreement,
        contrarian_value=0.0, league="x/y",
    )


def _agent():
    from src.agent.betting_agent import FootballBettingAgent
    return FootballBettingAgent.__new__(FootballBettingAgent)


# ══════════════════════════════ Invariant 1 — no double persistence

def test_invariant_1_same_prediction_cannot_be_persisted_twice(tmp_path):
    """The DB unique index is the last line of defence and must actually hold."""
    from src.agent.betting_agent import _insert_pick_if_absent

    mgr = _mgr(tmp_path, "i1.db")
    from sqlalchemy import Index
    # The dedup index ships as migration 002; create it here so the test proves
    # the guard rather than the absence of a constraint.
    with mgr.engine.begin() as conn:
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS ix_saved_picks_dedup "
            "ON saved_picks (match_id, selection, pick_date)")
    assert Index is not None

    values = dict(match_id=1, pick_date=date(2026, 9, 1), market="1X2",
                  selection="Home Win", odds=2.0, predicted_probability=0.55,
                  expected_value=0.1, confidence=0.55,
                  kelly_stake_percentage=1.0)
    with mgr.get_session() as s:
        first = _insert_pick_if_absent(s, dict(values))
        second = _insert_pick_if_absent(s, dict(values))
        s.commit()
    assert first is not None, "the first insert should have created a row"
    assert second is None, "the same prediction was persisted twice"

    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 1


def test_invariant_1b_inmemory_dedup_keys_on_identity_not_display_name():
    """The in-memory gate must agree with the DB index about what is 'the same
    pick'. Keying on the rendered match name means two fixture rows for one game
    collapse a legitimate pick, and one fixture whose name changed between
    shards slips a duplicate through."""
    import inspect

    from src.agent.betting_agent import FootballBettingAgent

    src = inspect.getsource(FootballBettingAgent.finalize_picks)
    assert "key = (rec.match_id, rec.market, rec.selection)" in src, (
        "the duplicate key is not the normalized identity")


# ══════════════════ Invariant 2 — correlated selections are not two observations

def test_invariant_2_correlated_pair_is_filtered_before_persistence():
    agent = _agent()
    picks = [_rec(1, "Over 2.5 Goals", "Over 2.5", ev=0.20),
             _rec(1, "Under 3.5 Goals", "Under 3.5", ev=0.10)]
    kept = agent._filter_correlated_picks(picks)
    assert len(kept) == 1, (
        "Over 2.5 + Under 3.5 on one fixture survived as two observations")
    assert kept[0].selection == "Over 2.5 Goals", "the wrong leg was dropped"


@pytest.mark.parametrize("a,b", [
    ("Over 2.5 Goals", "Under 3.5 Goals"),
    ("Over 1.5 Goals", "Under 2.5 Goals"),
    ("Over 3.5 Goals", "Under 4.5 Goals"),
    ("Home Win", "Over 2.5 Goals"),
    ("Home Win", "Double Chance 1X"),
    ("BTTS Yes", "Over 2.5 Goals"),
])
def test_invariant_2b_declared_pairs_are_symmetric_and_caught(a, b):
    from src.agent.betting_agent import FootballBettingAgent as A
    assert A.selections_are_correlated(a, b)
    assert A.selections_are_correlated(b, a), "the predicate is not symmetric"


def test_invariant_2c_genuinely_different_markets_are_kept():
    """The policy must not over-collapse. Two uncorrelated selections on one
    fixture are two real observations and must both survive."""
    agent = _agent()
    picks = [_rec(1, "Over 2.5 Goals", "Over 2.5"),
             _rec(1, "Double Chance 1X", "Double Chance")]
    assert len(agent._filter_correlated_picks(picks)) == 2


def test_invariant_2d_claude_change_cannot_create_a_correlated_pair():
    """The gap that produced every correlated pair in production.

    _filter_correlated_picks runs inside get_daily_picks; the Claude review runs
    after it and rewrites selections. Its only guard was exact equality, so a
    switch onto a correlated selection was accepted — including onto pairs the
    filter table already declared.
    """
    import inspect

    from src.reporting.match_briefing import MatchBriefingService

    src = inspect.getsource(MatchBriefingService._apply_decision)
    assert "selections_are_correlated" in src, (
        "the CHANGE path does not re-check correlation")
    # And it must reject the switch, not delete the model's other pick.
    idx = src.index("selections_are_correlated")
    after = src[idx:idx + 700]
    assert 'review_action = "KEEP"' in after, (
        "a correlated switch must fall back to KEEP, not drop the other pick")
    assert "session.delete" not in after, (
        "the correlation branch deletes a pick — it must only reject the switch")


# ═══════════ Invariant 3 — paper picks cannot influence fitting/calibration

def test_invariant_3_calibration_fit_ignores_paper_rows(tmp_path):
    from src.models.probability_calibration import ProbabilityCalibrator

    mgr = _mgr(tmp_path, "i3.db")
    with mgr.get_session() as s:
        # 200 paper picks, wildly miscalibrated (predict 90%, always lose).
        for i in range(1, 201):
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=date(2026, 9, 1), league="x/y"))
            s.add(SavedPick(id=i, match_id=i, pick_date=date(2026, 9, 1),
                            market="1X2", selection="Home Win", odds=2.0,
                            predicted_probability=0.90, expected_value=0.1,
                            confidence=0.9, kelly_stake_percentage=1.0,
                            result="loss", is_paper=True))
        s.commit()

    pc = ProbabilityCalibrator()
    fitted = pc.fit_from_db(mgr)
    assert not fitted, (
        "the calibrator fitted on paper picks — the experiment can now "
        "recalibrate the model it is measuring")


def test_invariant_3b_every_learning_path_filters_paper():
    """Structural check: the learning/reporting sites must all carry the
    predicate. A new call site added without it is the regression this
    catches."""
    import inspect

    import src.agent.betting_agent as ba

    for fn_name in ("get_stats", "rolling_backtest", "_auto_calibrate_ev_threshold",
                    "tune_ensemble_weights", "calibrate_from_pick_outcomes"):
        fn = getattr(ba.FootballBettingAgent, fn_name)
        src = inspect.getsource(fn)
        assert "_live_only()" in src, f"{fn_name} does not exclude paper picks"


# ═════════════════ Invariant 4 — a stale snapshot cannot become a closing line

def test_invariant_4_stale_odds_are_rejected(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "i4.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=45)
    stale = kickoff - timedelta(minutes=600)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.1), ("Draw", 3.4), ("Away", 3.6)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=stale))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(), market="1X2",
                        selection="Home Win", odds=2.2,
                        closing_capture_status="pending",
                        created_at=stale - timedelta(minutes=1)))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=90)
    assert stats["captured"] == 0 and stats["missing"] == 1


# ════ Invariant 5 — the close cannot be the same observation as the taken price

def test_invariant_5_same_snapshot_is_not_a_closing_line(tmp_path, monkeypatch):
    """The subtle one. A pick taken 90 minutes before kickoff is priced from an
    odds row that is ITSELF inside the closing window, so the time rule alone
    hands that very row back as the close. CLV would be exactly 0.00% — an echo
    of our own price, indistinguishable in the data from genuine parity.
    """
    mgr = _mgr(tmp_path, "i5.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=40)
    snapshot = now - timedelta(minutes=20)        # inside the 180-min window
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.20), ("Draw", 3.40), ("Away", 3.60)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=snapshot))
        # The pick was created FROM that snapshot, at the same instant.
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(), market="1X2",
                        selection="Home Win", odds=2.20,
                        closing_capture_status="pending", created_at=snapshot))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=90)

    assert stats["captured"] == 0, (
        "the pick's own pricing snapshot was recorded as its closing line — "
        "CLV would read exactly 0.00% and look like closing-line parity")
    assert stats["missing"] == 1
    with mgr.get_session() as s:
        assert s.get(SavedPick, 1).closing_odds is None


def test_invariant_5b_a_reobserved_price_still_counts(tmp_path, monkeypatch):
    """The rule is identity, not value. A book re-quoted after the pick — even
    at the same number — is a genuine closing observation with CLV 0."""
    mgr = _mgr(tmp_path, "i5b.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=40)
    taken_at = now - timedelta(minutes=30)
    reobserved = now - timedelta(minutes=5)       # refreshed AFTER the pick
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.20), ("Draw", 3.40), ("Away", 3.60)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=reobserved))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(), market="1X2",
                        selection="Home Win", odds=2.20,
                        closing_capture_status="pending", created_at=taken_at))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    assert cap.capture(within_minutes=90)["captured"] == 1


# ═══════════════ Invariants 6 & 7 — version moves iff the prediction population does

def test_invariant_6_prediction_affecting_change_moves_the_version():
    """CODE_REVISION must feed the fingerprint. If it did not, a pure-code
    change would leave two genuinely different models sharing one version."""
    from src.models import model_version as mv

    cfg = type("C", (), {"get": staticmethod(lambda k, d=None: d)})()
    before = mv.fingerprint(cfg)
    original = mv.CODE_REVISION
    try:
        mv.CODE_REVISION = original + "-probe"
        after = mv.fingerprint(cfg)
    finally:
        mv.CODE_REVISION = original
    assert before != after, "CODE_REVISION does not affect model_version"


def test_invariant_6b_stage8_bumped_the_revision():
    """Stage 8 changed which picks get persisted. That is a different population
    of predictions and therefore a different experiment, even though no
    probability changed. Pinning the value here means a future selection change
    that forgets to bump has to argue with a test."""
    from src.models.model_version import CODE_REVISION

    assert CODE_REVISION == "s5.2", (
        "Stage 8 made three selection-affecting changes (over/under cross "
        "pairs, the post-Claude correlation re-check, the normalized dedup "
        "key). Any further selection change needs its own bump.")


def test_invariant_7_evaluation_only_change_does_not_move_the_version():
    """The cluster bootstrap and the effective-n display changed how results are
    REPORTED, not which predictions exist. Tracking them would fragment cohorts
    for no reason."""
    from src.models.model_version import TRACKED_KEYS

    for key in TRACKED_KEYS:
        assert not key.startswith("reporting."), key
        assert "bootstrap" not in key and "checkpoint" not in key, key
    # And the tracked list is config-only: no evaluation module feeds it.
    assert all(k.startswith(("models.", "betting.")) for k in TRACKED_KEYS)


# ═════════════════ Invariant 8 — the checkpoint counter's statistical unit

def test_invariant_8_checkpoints_expose_the_effective_sample_size():
    """The checkpoint DEFINITION stays 100/200/500 valid closing lines. What
    must not happen is reading 100 picks as 100 independent observations when
    18.9% of fixtures carry two."""
    import inspect

    from scripts import paper_trading_report as rep

    src = inspect.getsource(rep.section_checkpoints)
    assert "_effective_n" in src, (
        "the checkpoint counter does not report an effective sample size")
    assert "independent fixtures" in src
    assert [t for t, _ in rep.CHECKPOINTS] == [100, 200, 500], (
        "the checkpoint definition was altered")


def test_invariant_8b_cluster_bootstrap_is_wider_than_iid():
    """The property that makes clustering matter: resampling fixtures must not
    produce a narrower interval than resampling picks."""
    import numpy as np

    from scripts.paper_trading_report import _boot, _effective_n

    rng = np.random.default_rng(7)
    # 60 fixtures, each contributing two perfectly correlated picks.
    values, clusters = [], []
    for f in range(60):
        v = float(rng.normal())
        values += [v, v]
        clusters += [f, f]

    lo_i, hi_i = _boot(values, iters=1500, seed=1)
    lo_c, hi_c = _boot(values, clusters=clusters, iters=1500, seed=1)
    assert (hi_c - lo_c) > (hi_i - lo_i), (
        "the cluster bootstrap is not wider than the i.i.d. one on perfectly "
        "correlated pairs — the clustering is not taking effect")

    n, k, deff, n_eff = _effective_n(clusters)
    assert (n, k) == (120, 60)
    assert deff == pytest.approx(2.0), "duplicated picks must halve effective n"
    assert n_eff == pytest.approx(60.0)


def test_invariant_8c_singleton_clusters_behave_like_iid():
    import numpy as np

    from scripts.paper_trading_report import _effective_n

    n, k, deff, n_eff = _effective_n(list(range(40)))
    assert (n, k) == (40, 40)
    assert deff == pytest.approx(1.0)
    assert n_eff == pytest.approx(40.0)
    assert np is not None


# ══════════ Invariant 9 — pre-Stage-5 history cannot enter the experiment

def test_invariant_9_unversioned_history_is_excluded_by_model_version(tmp_path,
                                                                     monkeypatch):
    """The 1,070 historical picks carry model_version = NULL. Filtering the
    report by the frozen version must exclude every one of them."""
    mgr = _mgr(tmp_path, "i9.db")
    with mgr.get_session() as s:
        for i in range(1, 11):
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=date(2026, 9, 1), league="x/y"))
            s.add(SavedPick(id=i, match_id=i, pick_date=date.today(),
                            market="1X2", selection="Home Win", odds=2.0,
                            predicted_probability=0.5, expected_value=0.1,
                            confidence=0.5, kelly_stake_percentage=1.0,
                            result="win",
                            model_version=None if i <= 7 else "frozen_v1",
                            is_paper=i > 7))
        s.commit()

    import scripts.paper_trading_report as rep
    import src.data.database as dbm
    monkeypatch.setattr(dbm, "get_db", lambda: mgr)

    scoped = rep.load_picks(days=3650, include_live=True,
                            model_version="frozen_v1")
    assert len(scoped) == 3, "unversioned history leaked into the experiment"
    assert all(p.model_version == "frozen_v1" for p in scoped)


# ═════════════ Invariant 10 — paper can never enter the real-money ROI path

def test_invariant_10_paper_never_reaches_live_roi(tmp_path):
    """Live picks all win at 2.0; paper picks all lose. Any leak moves ROI from
    +100% to something else, so this cannot pass by accident."""
    mgr = _mgr(tmp_path, "i10.db")
    with mgr.get_session() as s:
        for i in range(1, 21):
            paper = i > 10
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=date(2026, 9, 1), league="x/y",
                        home_goals=1, away_goals=0))
            s.add(SavedPick(id=i, match_id=i, pick_date=date(2026, 9, 1),
                            market="1X2", selection="Home Win", odds=2.0,
                            predicted_probability=0.55, expected_value=0.1,
                            confidence=0.55, kelly_stake_percentage=1.0,
                            result="loss" if paper else "win", is_paper=paper))
        s.commit()

    agent = _agent()
    agent.db = mgr

    class _Pred:
        def coverage_summary(self):
            return {"poisson_teams": 0, "elo_teams": 0, "ml_fitted": False,
                    "goals_ml_fitted": False}

    agent.predictor = _Pred()
    all_time = agent.get_stats()["all_time"]
    assert all_time["total"] == 10 and all_time["wins"] == 10
    assert all_time["roi"] == pytest.approx(1.0)


def test_invariant_10b_paper_mode_is_on_and_pinned():
    import pathlib

    import yaml

    cfg = yaml.safe_load(
        pathlib.Path("config/config.example.yaml").read_text(encoding="utf-8"))
    assert cfg["betting"]["paper_trading_mode"] is True


def test_teams_table_import_is_used():
    """Keeps the Team import meaningful for schema creation in this module."""
    assert Team.__tablename__ == "teams"
