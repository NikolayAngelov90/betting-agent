"""Tests for the loss-driven ensemble weight learner (Stage 3, Phase 2 + 3).

Covers the defects the 2026-08-07 audit found in the old Beta/accuracy learner:
market contamination, double updating, and a weight mapping that could not
distinguish model quality. Plus the required cold-start / small-sample /
league-isolation behaviour.
"""

import json
import math
from pathlib import Path

import pytest

import src.models.bayesian_weights as bw
from src.models.bayesian_weights import (
    MODELS,
    EnsembleWeightLearner,
    observation_key,
)


@pytest.fixture
def learner(tmp_path, monkeypatch):
    """A learner persisting to a temp file, with a fixed config prior."""
    monkeypatch.setattr(bw, "WEIGHTS_PATH", tmp_path / "weights.json")

    class _Cfg:
        def get(self, key, default=None):
            return {
                "models.bayesian_weight_half_life_days": 90,
                "models.ensemble_weights": {
                    "poisson": 0.25, "elo": 0.20,
                    "xgboost": 0.35, "random_forest": 0.20,
                },
            }.get(key, default)

    return EnsembleWeightLearner(config=_Cfg())


def _feed(learner, market, league, losses, n=40, start=0):
    """Apply n observations per model with the given per-model loss."""
    for i in range(start, start + n):
        for model, loss in losses.items():
            learner.update(league, model, loss, days_ago=0, market=market,
                           obs_key=f"obs{i}")


# --------------------------------------------------------------- cold start

def test_cold_start_returns_config_prior(learner):
    w = learner.get_weights("england/premier-league", "1X2")
    assert w == pytest.approx(learner._prior, abs=1e-9)
    assert sum(w.values()) == pytest.approx(1.0)


def test_unknown_market_falls_back_to_prior(learner):
    _feed(learner, "1X2", "england/premier-league", {"poisson": 0.5, "elo": 1.5, "ml": 1.5})
    # "btts" has no observations at all — must not inherit the 1X2 estimate.
    assert learner.get_weights("england/premier-league", "btts") == pytest.approx(
        learner._prior, abs=1e-9)


# ------------------------------------------------------------ discrimination

def test_scope_weights_follow_loss_ordering_exactly(learner):
    """The raw (pre-shrinkage) Hedge weights must rank strictly by mean loss.

    The old learner normalised near-identical accuracies and collapsed to
    ~0.33 each; this is the property that replaces it.
    """
    _feed(learner, "1X2", None, {"poisson": 1.05, "elo": 0.95, "ml": 1.10}, n=60)
    raw = learner._weights_for_scope("1X2::__global__")
    assert raw["elo"] > raw["poisson"] > raw["ml"]
    # Material separation, not the 0.33/0.34/0.33 of the old scheme.
    assert raw["elo"] - raw["ml"] > 0.10


def test_shrinkage_blends_scope_weights_toward_the_prior(learner):
    """With moderate evidence the config prior still counts — ML starts at 0.55
    and a 0.05-nat loss deficit over 60 observations should not erase that."""
    _feed(learner, "1X2", None, {"poisson": 1.05, "elo": 0.95, "ml": 1.10}, n=60)
    w = learner.get_weights(None, "1X2")
    raw = learner._weights_for_scope("1X2::__global__")
    # Best model still leads after shrinkage...
    assert w["elo"] == max(w.values())
    # ...but ML sits between its raw score and its prior, not at either extreme.
    assert raw["ml"] < w["ml"] < learner._prior["ml"]


def test_enough_evidence_overturns_the_prior(learner):
    """The prior must be evidence-dominated eventually, or the learner is inert."""
    _feed(learner, "1X2", None, {"poisson": 0.90, "elo": 0.90, "ml": 1.30}, n=600)
    w = learner.get_weights(None, "1X2")
    assert w["ml"] < learner._prior["ml"]
    assert w["poisson"] > w["ml"] and w["elo"] > w["ml"]


def test_separation_grows_with_evidence(learner, tmp_path, monkeypatch):
    """Same loss gap, more data -> more confident weights. This is the property
    the fixed-normalisation Beta scheme could not express."""
    _feed(learner, "1X2", None, {"poisson": 1.05, "elo": 0.95, "ml": 1.05}, n=20)
    small_gap = learner.get_weights(None, "1X2")["elo"] - learner.get_weights(None, "1X2")["poisson"]
    _feed(learner, "1X2", None, {"poisson": 1.05, "elo": 0.95, "ml": 1.05},
          n=200, start=1000)
    big_gap = learner.get_weights(None, "1X2")["elo"] - learner.get_weights(None, "1X2")["poisson"]
    assert big_gap > small_gap


def test_weights_are_clamped(learner):
    """A crushing loss gap must not zero a model out — it needs residual weight
    to demonstrate a recovery."""
    _feed(learner, "1X2", None, {"poisson": 4.0, "elo": 0.05, "ml": 4.0}, n=300)
    w = learner.get_weights(None, "1X2")
    assert all(v >= bw.MIN_WEIGHT - 1e-9 for v in w.values()), w
    assert all(v <= bw.MAX_WEIGHT + 1e-9 for v in w.values()), w
    assert sum(w.values()) == pytest.approx(1.0)


# ---------------------------------------------------------- market isolation

def test_market_isolation(learner):
    """BUG 2 REGRESSION. A goals observation must not move the 1X2 weights.

    The old update() wrote _league_params and _global_params regardless of
    `market`, so this assertion failed by construction.
    """
    before = learner.get_weights("england/premier-league", "1X2")
    for i in range(80):
        # Poisson looks superb on goals and terrible on nothing else.
        learner.update("england/premier-league", "poisson", 0.10,
                       market="goals", obs_key=f"g{i}")
        learner.update("england/premier-league", "elo", 3.0,
                       market="goals", obs_key=f"g{i}")
        learner.update("england/premier-league", "ml", 3.0,
                       market="goals", obs_key=f"g{i}")
    after = learner.get_weights("england/premier-league", "1X2")
    assert after == pytest.approx(before, abs=1e-9)

    # ...while the goals weights DID move.
    goals_w = learner.get_weights("england/premier-league", "goals")
    assert goals_w["poisson"] > goals_w["elo"]


def test_no_market_agnostic_bucket_exists(learner):
    """Every scope key is market-qualified, so contamination is structural."""
    _feed(learner, "1X2", "spain/laliga", {"poisson": 1.0, "elo": 1.0, "ml": 1.0}, n=5)
    for scope in learner._scopes:
        assert "::" in scope
        market, _, _league = scope.partition("::")
        assert market == "1X2"


# -------------------------------------------------------------- idempotency

def test_duplicate_update_is_a_noop(learner):
    assert learner.update("l", "poisson", 1.0, market="1X2", obs_key="x1") is True
    assert learner.update("l", "poisson", 1.0, market="1X2", obs_key="x1") is False
    bucket = learner._scopes["1X2::l"]["poisson"]
    assert bucket["n"] == 1


def test_replaying_the_whole_history_changes_nothing(learner):
    losses = {"poisson": 1.05, "elo": 0.95, "ml": 1.20}
    _feed(learner, "1X2", "italy/serie-a", losses, n=50)
    once = learner.get_weights("italy/serie-a", "1X2")
    _feed(learner, "1X2", "italy/serie-a", losses, n=50)   # identical obs keys
    twice = learner.get_weights("italy/serie-a", "1X2")
    assert twice == pytest.approx(once, abs=1e-12)


def test_same_observation_different_markets_both_apply(learner):
    """One match yields a genuinely different observation per market."""
    assert learner.update("l", "poisson", 1.0, market="1X2", obs_key="m7") is True
    assert learner.update("l", "poisson", 1.0, market="goals", obs_key="m7") is True


def test_dedup_survives_a_save_load_round_trip(learner, tmp_path, monkeypatch):
    learner.update("l", "poisson", 1.0, market="1X2", obs_key="keep-me")
    learner.save()

    class _Cfg:
        def get(self, key, default=None):
            return {"models.bayesian_weight_half_life_days": 90,
                    "models.ensemble_weights": {}}.get(key, default)

    reloaded = EnsembleWeightLearner(config=_Cfg())
    assert reloaded.update("l", "poisson", 1.0, market="1X2", obs_key="keep-me") is False


# ------------------------------------------------------------ league scoping

def test_league_isolation(learner):
    """Two leagues with opposite evidence must get opposite weights, even though
    both contribute to the same market-global scope."""
    _feed(learner, "1X2", "greece/super-league",
          {"poisson": 0.6, "elo": 1.4, "ml": 1.4}, n=100)
    _feed(learner, "1X2", "norway/eliteserien",
          {"poisson": 1.4, "elo": 0.6, "ml": 1.4}, n=100, start=500)
    greece = learner.get_weights("greece/super-league", "1X2")
    norway = learner.get_weights("norway/eliteserien", "1X2")
    assert greece["poisson"] > norway["poisson"]
    assert norway["elo"] > greece["elo"]
    # An unseen league gets the market-global blend, which sits between them.
    unseen = learner.get_weights("brand/new", "1X2")
    assert min(greece["poisson"], norway["poisson"]) <= unseen["poisson"] \
        <= max(greece["poisson"], norway["poisson"])


def test_small_league_sample_is_shrunk_toward_global(learner):
    """A league with 3 observations must not override the global picture."""
    _feed(learner, "1X2", None, {"poisson": 1.30, "elo": 0.80, "ml": 1.30}, n=300,
          start=5000)
    global_w = learner.get_weights(None, "1X2")
    # Three contradictory observations in one league.
    _feed(learner, "1X2", "tiny/league", {"poisson": 0.1, "elo": 3.0, "ml": 3.0}, n=3)
    tiny = learner.get_weights("tiny/league", "1X2")
    # It moved a little, but Elo still leads because global evidence dominates.
    assert tiny["elo"] > tiny["poisson"]
    assert abs(tiny["poisson"] - global_w["poisson"]) < 0.15


def test_scope_needs_every_model_before_it_is_used(learner):
    """Absence of evidence must not read as evidence of quality."""
    for i in range(50):
        learner.update("x/y", "poisson", 0.1, market="1X2", obs_key=f"z{i}")
    # elo/ml have no data in this scope -> fall back to the prior, not to
    # "poisson is perfect".
    assert learner.get_weights("x/y", "1X2") == pytest.approx(learner._prior, abs=1e-9)


# ------------------------------------------------------------- update maths

def test_decay_downweights_old_observations(learner):
    learner.update("l", "poisson", 2.0, days_ago=0, market="1X2", obs_key="new")
    learner.update("l", "elo", 2.0, days_ago=90, market="1X2", obs_key="old")
    # 90 days == one half-life at the configured setting.
    assert learner._scopes["1X2::l"]["elo"]["w"] == pytest.approx(0.5, abs=1e-6)
    assert learner._scopes["1X2::l"]["poisson"]["w"] == pytest.approx(1.0, abs=1e-6)


def test_mean_loss_is_the_decay_weighted_mean(learner):
    learner.update("l", "poisson", 1.0, days_ago=0, market="1X2", obs_key="a")
    learner.update("l", "poisson", 3.0, days_ago=90, market="1X2", obs_key="b")
    b = learner._scopes["1X2::l"]["poisson"]
    assert b["loss"] / b["w"] == pytest.approx((1.0 + 0.5 * 3.0) / 1.5)


def test_invalid_losses_are_rejected(learner):
    assert learner.update("l", "poisson", float("nan"), market="1X2") is False
    assert learner.update("l", "poisson", -1.0, market="1X2") is False
    assert learner.update("l", "not_a_model", 1.0, market="1X2") is False
    assert "1X2::l" not in learner._scopes


def test_extreme_loss_is_clipped(learner):
    learner.update("l", "poisson", 1e9, market="1X2", obs_key="huge")
    assert learner._scopes["1X2::l"]["poisson"]["loss"] == pytest.approx(bw.MAX_LOSS)


# --------------------------------------------------------------- persistence

def test_legacy_schema_file_is_ignored(tmp_path, monkeypatch):
    """The old Beta file's counts came from the double-update bug — importing
    them would carry the contamination forward."""
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({
        "global": {"poisson": {"alpha": 425.0, "beta": 373.0, "n": 872}},
        "leagues": {"england/premier-league": {"poisson": {"alpha": 1, "beta": 1, "n": 5}}},
    }))
    monkeypatch.setattr(bw, "WEIGHTS_PATH", path)

    class _Cfg:
        def get(self, key, default=None):
            return {"models.bayesian_weight_half_life_days": 90,
                    "models.ensemble_weights": {}}.get(key, default)

    learner = EnsembleWeightLearner(config=_Cfg())
    assert learner._scopes == {}


def test_observation_key_prefers_pick_id():
    assert observation_key(pick_id=7, match_id=99, market="1X2") == "p7"
    assert observation_key(match_id=99, market="1X2", selection="Home Win") == \
        "m99|1X2|Home Win"


def test_get_weights_always_sums_to_one(learner):
    _feed(learner, "1X2", "a/b", {"poisson": 0.9, "elo": 1.1, "ml": 1.0}, n=17)
    for league in (None, "a/b", "unseen/league"):
        for market in ("1X2", "goals", ""):
            w = learner.get_weights(league, market)
            assert set(w) == set(MODELS)
            assert sum(w.values()) == pytest.approx(1.0)
