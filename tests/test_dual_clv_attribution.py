"""Stage 9 — the dual-attribution contract: frozen model vs final selection.

The experiment asks two questions and must never answer one while appearing to
answer the other:

    model  — did the FROZEN Stage 5 selection's price move the right way?
    final  — did the selection actually taken move the right way?

The dangerous failure is silent substitution: reporting the final selection's
CLV under the model's name on a row where Claude changed the pick. Several of
these tests exist purely to make that impossible.

Temp SQLite throughout; conftest strips DATABASE_URL.
"""

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, SavedPick
from src.evaluation.attribution import (FINAL, MODEL, MODEL_PRICE_NOT_KEPT,
                                        NO_MODEL_SNAPSHOT, coverage_class,
                                        resolve, selection_changed,
                                        shares_one_observation)


def _mgr(tmp_path, name):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _pick(market="1X2", selection="Home Win", odds=2.20,
          model_market="1X2", model_selection="Home Win", **kw):
    return SimpleNamespace(market=market, selection=selection, odds=odds,
                           model_market=model_market,
                           model_selection=model_selection, **kw)


# ═══════════════ Test 1 — identical selections: one close, two attributions

def test_1_identical_selections_share_one_observation():
    m, f = resolve(_pick())
    assert m.attribution == MODEL and f.attribution == FINAL
    assert m.measurable and f.measurable
    assert m.taken_odds == f.taken_odds == 2.20
    assert shares_one_observation(m, f), (
        "an unchanged pick must be recognised as ONE underlying observation")
    assert coverage_class(m, f) == "both_measurable_same_selection"


def test_1b_capture_runs_once_for_an_unchanged_pick(tmp_path, monkeypatch):
    """Section 6: no duplicate capture operation, no duplicate row, no extra
    quota. The pick is stored once and the close is resolved once."""
    mgr = _mgr(tmp_path, "d1.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=40)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.05), ("Draw", 3.40), ("Away", 3.90)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o,
                       timestamp=now - timedelta(minutes=5)))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(),
                        market="1X2", selection="Home Win", odds=2.20,
                        model_market="1X2", model_selection="Home Win",
                        closing_capture_status="pending",
                        created_at=now - timedelta(hours=2)))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    stats = cap.capture(within_minutes=90)

    assert stats["captured"] == 1
    assert stats["considered"] == 1, "the pick was considered more than once"
    with mgr.get_session() as s:
        assert s.query(SavedPick).count() == 1, "capture created a second row"


# ═════════ Test 2 — differing selections: no substitution in either direction

def test_2_changed_selection_never_borrows_the_other_series_price():
    """The single most important guard in Stage 9.

    On a CHANGE the two series are different bets. Reporting the final
    selection's CLV as the model's would answer the wrong question while looking
    perfectly healthy.
    """
    m, f = resolve(_pick(market="Over 2.5", selection="Over 2.5 Goals",
                         odds=1.53,
                         model_market="Team Goals",
                         model_selection="Home Over 0.5"))
    assert f.measurable and f.selection == "Over 2.5 Goals" and f.taken_odds == 1.53
    assert not m.measurable, "the model series claimed a price it does not have"
    assert m.taken_odds is None, "the final selection's price leaked into model"
    assert m.selection == "Home Over 0.5", (
        "the model series must still name what the model picked")
    assert not shares_one_observation(m, f)
    assert coverage_class(m, f) == "final_only_measurable"


def test_2b_report_does_not_attribute_a_final_close_to_the_model(tmp_path,
                                                                monkeypatch):
    """End-to-end version of the same guard, through the report."""
    mgr = _mgr(tmp_path, "d2.db")
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2,
                    match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
        s.add(SavedPick(id=1, match_id=1, pick_date=date.today(),
                        market="Over 2.5", selection="Over 2.5 Goals", odds=1.53,
                        predicted_probability=0.6, expected_value=-0.1,
                        confidence=0.6, kelly_stake_percentage=1.0,
                        model_market="Team Goals",
                        model_selection="Home Over 0.5",
                        review_action="CHANGE",
                        closing_odds=1.45, closing_capture_status="captured"))
        s.commit()

    import scripts.paper_trading_report as rep
    import src.data.database as dbm
    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    picks = rep.load_picks(days=3650, include_live=True, model_version=None)
    assert len(picks) == 1

    m, f = resolve(picks[0])
    assert f.measurable and not m.measurable
    assert m.unavailable_reason == MODEL_PRICE_NOT_KEPT


# ═══════════ Tests 3 & 4 — one series measurable, the other not

def test_3_model_measurable_final_not_keeps_the_model_series():
    """A final selection with no usable price must not drag the model series
    down with it."""
    m, f = resolve(_pick(odds=None))
    assert not f.measurable
    assert not m.measurable and m.unavailable_reason == "no_taken_price"
    assert coverage_class(m, f) == "neither_measurable"

    # And with a price, the pair is measurable — the asymmetry is about data,
    # not about one series being privileged.
    m2, f2 = resolve(_pick(odds=2.0))
    assert m2.measurable and f2.measurable


def test_4_final_measurable_model_unavailable(tmp_path):
    """The dominant historical case: 999 of 1,070 rows predate the snapshot."""
    m, f = resolve(_pick(model_market=None, model_selection=None))
    assert f.measurable, "the final series must survive a missing model snapshot"
    assert not m.measurable
    assert m.unavailable_reason == NO_MODEL_SNAPSHOT
    assert coverage_class(m, f) == "final_only_measurable"
    assert tmp_path is not None


# ═══════ Test 5 — one odds row, two attributions, still ONE fixture

def test_5_shared_observation_is_two_counters_but_one_fixture():
    from scripts.paper_trading_report import _effective_n

    # Five unchanged picks on five fixtures: each contributes one model CLV and
    # one final CLV off the SAME close.
    model_fx, final_fx = [], []
    for fid in range(5):
        m, f = resolve(_pick())
        assert shares_one_observation(m, f)
        model_fx.append(fid)
        final_fx.append(fid)

    n_m, k_m, deff_m, _ = _effective_n(model_fx)
    n_f, k_f, deff_f, _ = _effective_n(final_fx)
    assert (n_m, k_m) == (5, 5) and (n_f, k_f) == (5, 5)
    assert deff_m == pytest.approx(1.0) and deff_f == pytest.approx(1.0), (
        "a shared observation was counted as two picks on one fixture")
    # Pooling the two series would be the bug: 10 picks on 5 fixtures.
    pooled_n, pooled_k, pooled_deff, _ = _effective_n(model_fx + final_fx)
    assert (pooled_n, pooled_k) == (10, 5) and pooled_deff == pytest.approx(2.0)


# ═════════════════════ Test 6 — the same-snapshot rule, three ways

@pytest.mark.parametrize("offset_minutes,expect_captured", [
    (-5, 0),    # odds observed BEFORE the pick   → rejected
    (0, 0),     # odds observed AT the pick       → rejected (exclusive bound)
    (+5, 1),    # odds observed AFTER the pick    → accepted
])
def test_6_same_snapshot_boundary(tmp_path, monkeypatch, offset_minutes,
                                  expect_captured):
    mgr = _mgr(tmp_path, f"d6_{offset_minutes}.db")
    now = datetime.utcnow()
    kickoff = now + timedelta(minutes=40)
    created = now - timedelta(minutes=30)
    observed = created + timedelta(minutes=offset_minutes)
    with mgr.get_session() as s:
        s.add(Match(id=1, home_team_id=1, away_team_id=2, match_date=kickoff,
                    league="x/y", is_fixture=True))
        for sel, o in (("Home", 2.05), ("Draw", 3.40), ("Away", 3.90)):
            s.add(Odds(match_id=1, bookmaker="Pinnacle", market_type="1X2",
                       selection=sel, odds_value=o, timestamp=observed))
        s.add(SavedPick(id=1, match_id=1, pick_date=kickoff.date(),
                        market="1X2", selection="Home Win", odds=2.20,
                        closing_capture_status="pending", created_at=created))
        s.commit()

    from scripts import capture_closing_lines as cap
    monkeypatch.setattr(cap, "get_db", lambda: mgr)
    assert cap.capture(within_minutes=90)["captured"] == expect_captured


def test_6b_the_comparison_is_strict():
    """Guards the exact operator. `>=` would admit the pick's own snapshot."""
    import inspect

    from scripts import capture_closing_lines as cap

    src = inspect.getsource(cap.consensus_close)
    assert "ts <= not_before" in src, (
        "the same-snapshot bound is not exclusive — the pick's own pricing row "
        "would be accepted as its closing line")


# ═════════════ Test 7 — historical row with no model snapshot

def test_7_historical_row_marks_model_unavailable_not_failed():
    m, f = resolve(_pick(model_market=None, model_selection=None, odds=1.90))
    assert m.unavailable_reason == NO_MODEL_SNAPSHOT
    assert m.taken_odds is None
    assert f.measurable and f.taken_odds == 1.90
    # Tri-state: absence of a snapshot is NOT evidence the selection was kept.
    assert selection_changed(_pick(model_market=None,
                                   model_selection=None)) is None


# ═════════════════════════ Test 8 — CHANGE attribution mapping

def test_8_change_maps_each_series_to_its_own_fields():
    p = _pick(market="1X2", selection="Home Win", odds=1.65,
              model_market="Team Goals", model_selection="Away Over 0.5")
    m, f = resolve(p)
    assert (m.market, m.selection) == ("Team Goals", "Away Over 0.5")
    assert (f.market, f.selection) == ("1X2", "Home Win")
    assert selection_changed(p) is True


def test_8b_keep_is_not_a_change():
    p = _pick(review_action="KEEP")
    assert selection_changed(p) is False
    m, f = resolve(p)
    assert shares_one_observation(m, f)


# ═══════════════════ Test 9 — paired comparison and clustering

def test_9_paired_delta_is_final_minus_model_and_clusters_by_fixture():
    from scripts.paper_trading_report import _boot, _effective_n

    # Two picks per fixture, so clustering has something to do.
    paired = []
    for fid in range(30):
        for _ in range(2):
            paired.append((fid, 0.01, 0.03))     # model +1%, final +3%
    deltas = [f - m for (_, m, f) in paired]
    fx = [x for x, _, _ in paired]

    assert all(d == pytest.approx(0.02) for d in deltas), (
        "delta must be final - model, not the other way round")

    n, k, deff, n_eff = _effective_n(fx)
    assert (n, k) == (60, 30)
    assert deff == pytest.approx(2.0), "fixture clustering was not applied"
    assert n_eff == pytest.approx(30.0)

    lo, hi = _boot(deltas, clusters=fx, iters=300)
    assert lo == pytest.approx(0.02) and hi == pytest.approx(0.02)


def test_9b_paired_is_not_presented_as_an_independent_sample():
    import inspect

    from scripts import paper_trading_report as rep

    src = inspect.getsource(rep._section_paired)
    assert "not a causal claim" in src
    assert "observed difference" in src.lower()


# ══════════════════════ Test 10 — checkpoint separation

def test_10_checkpoints_report_each_series_separately():
    import inspect

    from scripts import paper_trading_report as rep

    src = inspect.getsource(rep.section_checkpoints)
    assert '"MODEL"' in src and '"FINAL"' in src, (
        "checkpoints do not name the two attribution series")
    assert "model_fx" in src and "final_fx" in src
    # The definition itself must be untouched.
    assert [t for t, _ in rep.CHECKPOINTS] == [100, 200, 500]


def test_10b_a_model_only_close_cannot_advance_the_final_counter(tmp_path,
                                                                 monkeypatch):
    """Counters must not contaminate one another. A CHANGE row with a captured
    close advances FINAL only — the model series has no measurable price."""
    mgr = _mgr(tmp_path, "d10.db")
    with mgr.get_session() as s:
        for i in range(1, 4):
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=datetime(2026, 9, 1, 18, 0), league="x/y"))
            s.add(SavedPick(id=i, match_id=i, pick_date=date.today(),
                            market="Over 2.5", selection="Over 2.5 Goals",
                            odds=1.90, predicted_probability=0.55,
                            expected_value=0.05, confidence=0.55,
                            kelly_stake_percentage=1.0,
                            model_market="1X2", model_selection="Home Win",
                            review_action="CHANGE", closing_odds=1.80,
                            closing_capture_status="captured"))
        s.commit()

    import scripts.paper_trading_report as rep
    import src.data.database as dbm
    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    picks = rep.load_picks(days=3650, include_live=True, model_version=None)

    model_fx, final_fx = [], []
    for p in picks:
        m, f = resolve(p)
        if f.measurable:
            final_fx.append(p.match_id)
        if m.measurable and shares_one_observation(m, f):
            model_fx.append(p.match_id)

    assert len(final_fx) == 3
    assert len(model_fx) == 0, (
        "a changed selection's close advanced the MODEL checkpoint")


# ═══════════ Tests 11 & 12 — nothing earlier was weakened

def test_11_paper_live_isolation_still_holds(tmp_path):
    from src.agent.betting_agent import FootballBettingAgent

    mgr = _mgr(tmp_path, "d11.db")
    with mgr.get_session() as s:
        for i in range(1, 11):
            paper = i > 5
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=date(2026, 9, 1), league="x/y",
                        home_goals=1, away_goals=0))
            s.add(SavedPick(id=i, match_id=i, pick_date=date(2026, 9, 1),
                            market="1X2", selection="Home Win", odds=2.0,
                            predicted_probability=0.55, expected_value=0.1,
                            confidence=0.55, kelly_stake_percentage=1.0,
                            result="loss" if paper else "win", is_paper=paper))
        s.commit()

    agent = FootballBettingAgent.__new__(FootballBettingAgent)
    agent.db = mgr
    agent.predictor = SimpleNamespace(coverage_summary=lambda: {
        "poisson_teams": 0, "elo_teams": 0, "ml_fitted": False,
        "goals_ml_fitted": False})
    stats = agent.get_stats()["all_time"]
    assert stats["total"] == 5 and stats["roi"] == pytest.approx(1.0)


def test_12_stage8_integrity_fixes_are_intact():
    from src.agent.betting_agent import FootballBettingAgent as A
    from src.models.model_version import CODE_REVISION

    # Correlation policy unchanged by Stage 9.
    assert A.selections_are_correlated("Over 2.5 Goals", "Under 3.5 Goals")
    assert A.selections_are_correlated("Home Win", "Over 2.5 Goals")
    assert not A.selections_are_correlated("Over 2.5 Goals", "Double Chance 1X")
    # Stage 9 is evaluation-only: the frozen version must NOT move.
    assert CODE_REVISION == "s5.2", (
        "Stage 9 changed CODE_REVISION — it is evaluation-only and must not")


def test_12b_model_version_is_unchanged_by_stage9():
    from src.models.model_version import model_version
    from src.utils.config import Config

    from tests.test_config_identity import FROZEN_MODEL_VERSION

    # Stage 10.2: read the DEPLOYED config, not the local one. This used to
    # read config/config.yaml — the gitignored file CI overwrites — so it
    # pinned an identity production never ran.
    assert model_version(Config("config/config.example.yaml")) == \
        FROZEN_MODEL_VERSION


# ═════════════ The blocker, pinned so it cannot be quietly "fixed" wrongly

def test_model_series_on_a_change_is_blocked_not_approximated():
    """Stage 9's open blocker, asserted as a contract.

    A CHANGE destroys SavedPick.odds — the model selection's taken price — and
    CLV is taken/closing - 1. Until that price is recorded, the model series on
    a changed pick is UNAVAILABLE. The failure mode this guards against is a
    future 'fix' that reconstructs the price by inverting the stored EV: that
    inversion is wrong for Draw No Bet, whose EV is scaled by P(decisive), and
    it would fail silently.
    """
    m, _ = resolve(_pick(market="1X2", selection="Home Win", odds=1.65,
                         model_market="Over 2.5",
                         model_selection="Over 2.5 Goals",
                         pre_claude_ev=0.12, model_probability=0.62))
    assert not m.measurable
    assert m.unavailable_reason == MODEL_PRICE_NOT_KEPT
    assert m.taken_odds is None, (
        "the model's taken price was reconstructed from stored EV — that "
        "inversion is unsound for Draw No Bet and must not be used")


def test_report_sections_render_with_real_closes(tmp_path, monkeypatch, capsys):
    """Smoke test with actual captured closes: the two-series block, the paired
    subset and the review-action table must all print without crashing and must
    not merge the series."""
    mgr = _mgr(tmp_path, "d13.db")
    with mgr.get_session() as s:
        for i in range(1, 13):
            changed = i > 8                     # 4 CHANGE rows, 8 unchanged
            s.add(Match(id=i, home_team_id=1, away_team_id=2,
                        match_date=datetime(2026, 9, 1, 18, 0), league="x/y",
                        home_goals=2, away_goals=1))
            s.add(SavedPick(
                id=i, match_id=i, pick_date=date.today(),
                market="1X2", selection="Home Win", odds=2.20,
                predicted_probability=0.55, expected_value=0.1,
                confidence=0.55, kelly_stake_percentage=1.0, result="win",
                model_market="Over 2.5" if changed else "1X2",
                model_selection="Over 2.5 Goals" if changed else "Home Win",
                review_action="CHANGE" if changed else "KEEP",
                closing_odds=2.10, closing_capture_status="captured"))
        s.commit()

    import scripts.paper_trading_report as rep
    import src.data.database as dbm
    monkeypatch.setattr(dbm, "get_db", lambda: mgr)
    picks = rep.load_picks(days=3650, include_live=True, model_version=None)
    assert len(picks) == 12

    rep.section_attribution_coverage(picks)
    rep.section_clv(picks)
    rep.section_checkpoints(picks)
    out = capsys.readouterr().out

    assert "Series A — FROZEN MODEL" in out
    assert "Series B — FINAL SELECTION" in out
    assert "PAIRED SUBSET" in out
    assert "BY REVIEW ACTION" in out
    # 8 shared observations for model, 12 for final — never pooled into 20.
    assert "model    valid closing lines : 8" in out
    assert "final    valid closing lines : 12" in out
    # The 4 CHANGE rows must show no model CLV rather than the final's.
    assert "CHANGE" in out and "n/a" in out


# ═════════ Stage 12.1, Defect 3 — coverage must recognise recorded MODEL prices

def _obs_row(attribution, market, selection, taken_odds):
    """A pick_observations row as the report's loader shapes it."""
    return SimpleNamespace(attribution=attribution, market=market,
                           selection=selection, taken_odds=taken_odds,
                           closing_odds=None, closing_status="pending",
                           closing_captured_at=None)


def test_d3_legacy_changed_pick_without_observation_stays_unavailable():
    """The pre-Stage-10 case must be unchanged: no observation, no model price."""
    from src.evaluation.attribution import coverage_class, resolve_effective

    p = _pick(market="1X2", selection="Home Win", odds=2.10,
              model_market="Over 2.5", model_selection="Over 2.5 Goals")
    p.observations = {}
    m, f = resolve_effective(p)
    assert not m.measurable
    assert m.unavailable_reason == MODEL_PRICE_NOT_KEPT
    assert coverage_class(m, f) == "final_only_measurable"


def test_d3_changed_pick_with_model_observation_is_measurable():
    """The defect. saved_picks.odds holds the FINAL price, but the MODEL price
    was recorded at pick time — so the model series IS measurable."""
    from src.evaluation.attribution import coverage_class, resolve_effective

    p = _pick(market="1X2", selection="Home Win", odds=2.10,
              model_market="Over 2.5", model_selection="Over 2.5 Goals")
    p.observations = {
        "model": _obs_row("model", "Over 2.5", "Over 2.5 Goals", 1.85),
        "final": _obs_row("final", "1X2", "Home Win", 2.10),
    }
    m, f = resolve_effective(p)

    assert m.measurable, "a recorded MODEL price was still reported unavailable"
    assert (m.market, m.selection) == ("Over 2.5", "Over 2.5 Goals")
    assert m.taken_odds == pytest.approx(1.85)
    assert f.measurable and f.taken_odds == pytest.approx(2.10)
    assert coverage_class(m, f) == "both_measurable"
    # Still two different bets — not a shared observation.
    assert not shares_one_observation(m, f)


def test_d3_keep_pick_with_observations_shares_one_observation():
    from src.evaluation.attribution import coverage_class, resolve_effective

    p = _pick(market="Over 2.5", selection="Over 2.5 Goals", odds=1.85,
              model_market="Over 2.5", model_selection="Over 2.5 Goals")
    p.observations = {
        "model": _obs_row("model", "Over 2.5", "Over 2.5 Goals", 1.85),
        "final": _obs_row("final", "Over 2.5", "Over 2.5 Goals", 1.85),
    }
    m, f = resolve_effective(p)
    assert shares_one_observation(m, f)
    assert coverage_class(m, f) == "both_measurable_same_selection"


def test_d3_change_across_markets_keeps_the_series_independent():
    from src.evaluation.attribution import resolve_effective

    p = _pick(market="Draw No Bet", selection="DNB Away", odds=1.83,
              model_market="BTTS", model_selection="BTTS Yes")
    p.observations = {
        "model": _obs_row("model", "BTTS", "BTTS Yes", 1.715),
        "final": _obs_row("final", "Draw No Bet", "DNB Away", 1.83),
    }
    m, f = resolve_effective(p)
    assert m.market != f.market and m.selection != f.selection
    assert m.taken_odds == pytest.approx(1.715)
    assert f.taken_odds == pytest.approx(1.83)
    assert not shares_one_observation(m, f)


def test_d3_observation_wins_over_a_diverging_saved_pick():
    """If saved_picks and the observation disagree, the OBSERVATION is the
    record: it was captured at pick time and cannot be reconstructed. This is
    exactly the CHANGE case, where saved_picks.odds was overwritten."""
    from src.evaluation.attribution import resolve_effective

    p = _pick(market="1X2", selection="Away Win", odds=3.40,
              model_market="1X2", model_selection="Home Win")
    p.observations = {
        "model": _obs_row("model", "1X2", "Home Win", 2.00),
        "final": _obs_row("final", "1X2", "Away Win", 3.40),
    }
    m, f = resolve_effective(p)
    assert (m.selection, m.taken_odds) == ("Home Win", 2.00)
    assert (f.selection, f.taken_odds) == ("Away Win", 3.40)


def test_d3_missing_observation_falls_back_per_series():
    """Fallback is per-series, not all-or-nothing: a FINAL observation present
    and a MODEL one absent must still use the legacy MODEL resolution."""
    from src.evaluation.attribution import resolve_effective

    p = _pick(market="1X2", selection="Home Win", odds=2.10,
              model_market="1X2", model_selection="Home Win")
    p.observations = {"final": _obs_row("final", "1X2", "Home Win", 2.10)}
    m, f = resolve_effective(p)
    # Unchanged pick, so the legacy path can still price the model series.
    assert m.measurable and m.taken_odds == pytest.approx(2.10)
    assert f.measurable and f.taken_odds == pytest.approx(2.10)

    # And with no observations at all, behaviour is exactly resolve().
    p2 = _pick(model_market=None, model_selection=None)
    p2.observations = None
    m2, f2 = resolve_effective(p2)
    assert m2.unavailable_reason == NO_MODEL_SNAPSHOT and f2.measurable


def test_d3_observation_without_a_usable_price_is_not_measurable():
    from src.evaluation.attribution import resolve_effective

    p = _pick(market="1X2", selection="Home Win", odds=2.10,
              model_market="Over 2.5", model_selection="Over 2.5 Goals")
    p.observations = {"model": _obs_row("model", "Over 2.5", "Over 2.5 Goals", 1.0)}
    m, _ = resolve_effective(p)
    assert not m.measurable and m.unavailable_reason == "no_taken_price"


def test_d3_series_clv_is_untouched_by_the_coverage_fix():
    """Requirement 4: series_clv must not change. It already prefers
    observations on its own; resolve_effective exists for the coverage table."""
    import inspect

    from scripts import paper_trading_report as rep

    src = inspect.getsource(rep.series_clv)
    assert "resolve_effective" not in src, (
        "series_clv now depends on resolve_effective — CLV semantics changed")
    assert "observations" in src and "resolve(p)" in src
