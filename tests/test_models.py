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

        ml.fit(X, y, feature_names=[f"f{i}" for i in range(5)])
        assert ml.is_fitted

        pred = ml.predict(np.random.randn(5))
        avg = pred["ml_average"]
        total = avg["home_win"] + avg["draw"] + avg["away_win"]
        assert abs(total - 1.0) < 0.01
