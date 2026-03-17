"""Tests for prediction models."""

import numpy as np
import pytest


class TestPoissonModel:
    """Tests for the Poisson prediction model."""

    def test_score_matrix_sums_to_one(self):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        matrix = model._score_matrix(1.5, 1.2)
        assert abs(np.sum(matrix) - 1.0) < 0.01

    def test_probabilities_sum_to_one(self):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        pred = model.predict(1, 2)
        total = pred["home_win"] + pred["draw"] + pred["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_over_under_complement(self):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        pred = model.predict(1, 2)
        assert abs(pred["over_2.5"] + pred["under_2.5"] - 1.0) < 0.01

    def test_btts_complement(self):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        pred = model.predict(1, 2)
        assert abs(pred["btts_yes"] + pred["btts_no"] - 1.0) < 0.01

    def test_most_likely_score_format(self):
        from src.models.poisson_model import PoissonModel
        model = PoissonModel()
        pred = model.predict(1, 2)
        score = pred["most_likely_score"]
        parts = score.split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)


class TestEloRatingSystem:
    """Tests for the Elo rating system."""

    def test_default_rating(self):
        from src.models.elo_system import EloRatingSystem, DEFAULT_ELO
        elo = EloRatingSystem()
        assert elo.get_rating(999) == DEFAULT_ELO

    def test_expected_score_equal_ratings(self):
        from src.models.elo_system import EloRatingSystem
        elo = EloRatingSystem()
        expected = elo._expected_score(1500, 1500)
        assert abs(expected - 0.5) < 0.01

    def test_expected_score_higher_rating_favored(self):
        from src.models.elo_system import EloRatingSystem
        elo = EloRatingSystem()
        expected = elo._expected_score(1700, 1500)
        assert expected > 0.5

    def test_probabilities_sum_to_one(self):
        from src.models.elo_system import EloRatingSystem
        elo = EloRatingSystem()
        h, d, a = elo._calculate_probabilities(1600, 1500)
        assert abs(h + d + a - 1.0) < 0.01

    def test_process_match_updates_ratings(self):
        from src.models.elo_system import EloRatingSystem, DEFAULT_ELO
        elo = EloRatingSystem()
        elo._process_match(1, 2, 3, 0)  # Team 1 wins big
        assert elo.get_rating(1) > DEFAULT_ELO
        assert elo.get_rating(2) < DEFAULT_ELO


class TestMLModels:
    """Tests for ML model wrapper."""

    def test_default_prediction_when_not_fitted(self):
        from src.models.ml_models import MLModels
        ml = MLModels()
        pred = ml.predict(np.zeros(10))
        assert "ml_average" in pred
        avg = pred["ml_average"]
        assert abs(avg["home_win"] + avg["draw"] + avg["away_win"] - 1.0) < 0.01

    def test_fit_and_predict(self):
        from src.models.ml_models import MLModels
        ml = MLModels()

        # Synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f"f{i}" for i in range(5)]

        ml.fit(X, y, feature_names=feature_names)
        assert ml.is_fitted

        pred = ml.predict(np.random.randn(5), feature_names=feature_names)
        avg = pred["ml_average"]
        total = avg["home_win"] + avg["draw"] + avg["away_win"]
        assert abs(total - 1.0) < 0.01

    def test_trained_at_saved(self, tmp_path):
        """ML models persist trained_at timestamp after save/load."""
        from src.models.ml_models import MLModels
        ml = MLModels()
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        ml.fit(X, y, feature_names=[f"f{i}" for i in range(5)])
        ml.save(str(tmp_path))
        ml2 = MLModels()
        ml2.load(str(tmp_path))
        assert ml2.trained_at is not None
        # Should be a valid ISO datetime
        from datetime import datetime
        dt = datetime.fromisoformat(ml2.trained_at)
        assert dt.year >= 2026


class TestEVCalibrationPersistence:
    """Tests for EV threshold persistence."""

    def test_ev_threshold_persisted(self, tmp_path):
        """EV calibration writes and reads threshold from JSON."""
        import json
        ev_path = tmp_path / "ev_threshold.json"
        data = {"min_ev": 0.045, "hit_rate": 0.62, "n_picks": 40, "updated_at": "2026-03-09"}
        ev_path.write_text(json.dumps(data))
        loaded = json.loads(ev_path.read_text())
        assert loaded["min_ev"] == 0.045
        assert 0.01 <= loaded["min_ev"] <= 0.08

    def test_ml_stale_detection(self):
        """_ml_models_stale returns True when trained_at is None."""
        from unittest.mock import MagicMock
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.predictor = MagicMock()
        agent.predictor.ml_models.trained_at = None
        agent.predictor.goals_model.trained_at = None
        agent.config = MagicMock()
        assert agent._ml_models_stale(max_age_days=3) is True

    def test_ml_fresh_detection(self):
        """_ml_models_stale returns False when both models trained recently."""
        from unittest.mock import MagicMock
        from src.utils.logger import utcnow
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.predictor = MagicMock()
        now_str = utcnow().isoformat()
        agent.predictor.ml_models.trained_at = now_str
        agent.predictor.goals_model.trained_at = now_str
        agent.config = MagicMock()
        assert agent._ml_models_stale(max_age_days=3) is False

    def test_ml_stale_when_goals_model_missing(self):
        """_ml_models_stale returns True when goals model has no trained_at."""
        from unittest.mock import MagicMock
        from src.utils.logger import utcnow
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.predictor = MagicMock()
        agent.predictor.ml_models.trained_at = utcnow().isoformat()
        agent.predictor.goals_model.trained_at = None
        agent.config = MagicMock()
        assert agent._ml_models_stale(max_age_days=3) is True


class TestLearnFromSettled:
    """Tests for the learn_from_settled() orchestrator."""

    def test_learn_from_settled_skips_tuning_on_poisson_failure(self):
        """If Poisson refit fails, ensemble tuning is skipped."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.predictor = MagicMock()
        agent.predictor.fit.side_effect = RuntimeError("DB error")
        agent.predictor.ml_models.trained_at = None
        agent.predictor.goals_model.trained_at = None
        agent.feature_engineer = MagicMock()
        agent.config = MagicMock()
        agent.config.get.return_value = 3
        agent.db = MagicMock()
        agent.db.is_postgres = False
        agent.value_calculator = MagicMock()

        # tune_ensemble_weights should NOT be called when Poisson fails
        agent.tune_ensemble_weights = AsyncMock()
        agent._auto_calibrate_ev_threshold = MagicMock()
        agent.train_ml_models = AsyncMock()

        asyncio.run(agent.learn_from_settled())

        agent.tune_ensemble_weights.assert_not_called()
        # But EV calibration and ML retrain should still run
        agent._auto_calibrate_ev_threshold.assert_called_once()

    def test_learn_from_settled_runs_all_steps(self):
        """learn_from_settled runs all 4 steps when Poisson succeeds."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.predictor = MagicMock()
        agent.predictor.ml_models.trained_at = None
        agent.predictor.goals_model.trained_at = None
        agent.feature_engineer = MagicMock()
        agent.config = MagicMock()
        agent.config.get.return_value = 3
        agent.db = MagicMock()
        agent.db.is_postgres = False
        agent.value_calculator = MagicMock()
        agent.tune_ensemble_weights = AsyncMock(return_value={"weights": {}, "accuracies": {}})
        agent._auto_calibrate_ev_threshold = MagicMock()
        agent.train_ml_models = AsyncMock()

        asyncio.run(agent.learn_from_settled())

        agent.predictor.fit.assert_called_once()
        agent.tune_ensemble_weights.assert_called_once()
        agent._auto_calibrate_ev_threshold.assert_called_once()
        # ML retrain is deferred to --update (too slow for --settle CI timeout)
        agent.train_ml_models.assert_not_called()
