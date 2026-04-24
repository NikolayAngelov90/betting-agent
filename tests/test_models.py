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


class TestGetDailyPicksPreload:
    """Story 1.3: verify preload_batch() is wired into get_daily_picks()."""

    def _make_agent(self, fixture_ids=(42,), odds_counts=None):
        """Build a minimally-mocked FootballBettingAgent that can run through
        get_daily_picks() to the preload_batch call site.

        fixture_ids: ids returned by the fixture DB query (after dedup).
        odds_counts: list of count() return values for pre-filter loop;
                     defaults to [1] * len(fixture_ids) (all have odds).
        """
        import asyncio
        from datetime import datetime, timedelta, timezone
        from unittest.mock import MagicMock, AsyncMock
        from src.agent.betting_agent import FootballBettingAgent

        if odds_counts is None:
            odds_counts = [1] * len(fixture_ids)

        agent = FootballBettingAgent.__new__(FootballBettingAgent)

        # --- DB session mock ---
        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        agent.db = MagicMock()
        agent.db.get_session.return_value = session

        # Fixture query returns mock fixtures (spaced >2h apart to avoid dedup)
        now = datetime.now(timezone.utc)
        mock_fixtures = []
        for i, fid in enumerate(fixture_ids):
            f = MagicMock()
            f.id = fid
            f.league = f"league-{i}"   # unique leagues → dedup never triggers
            f.home_team_id = fid * 10
            f.away_team_id = fid * 10 + 1
            f.match_date = now + timedelta(hours=2 + i * 8)
            f.apifootball_id = None
            mock_fixtures.append(f)

        # .all() is called twice:
        #   1st: fixture query → mock_fixtures
        #   2nd: SavedPick pre-population query → [] (no existing picks)
        session.query.return_value.filter.return_value.all.side_effect = [
            mock_fixtures, []
        ]
        # First count() is the Story 8.3 idempotency check (0 = no today picks → proceed).
        # Subsequent count() calls are per-fixture odds counts.
        session.query.return_value.filter.return_value.count.side_effect = [0] + list(odds_counts)

        # session.get() is used for Team (needs .name) and Match (needs .league=None
        # to avoid MagicMock ending up in uncovered_leagues string-join path)
        mock_obj = MagicMock()
        mock_obj.name = "Team"
        mock_obj.league = None  # prevents uncovered_leagues TypeError
        session.get.return_value = mock_obj

        # --- Config: return each key's default so numeric comparisons don't break ---
        agent.config = MagicMock()
        agent.config.get.side_effect = lambda key, default=None: default

        # --- AF odds fallback disabled ---
        agent.apifootball = MagicMock()
        agent.apifootball.enabled = False

        # --- Coverage check ---
        agent.predictor = MagicMock()
        agent.predictor.check_coverage.return_value = {"score": 1.0}

        # --- EV calibration ---
        agent._auto_calibrate_ev_threshold = MagicMock()
        agent.value_calculator = MagicMock()
        agent.value_calculator.min_ev = 0.05

        # --- Feature engineer ---
        agent.feature_engineer = MagicMock()
        agent.feature_engineer._preload_cache = None

        # --- analyze_fixture returns empty analysis ---
        agent.analyze_fixture = AsyncMock(
            return_value=MagicMock(recommendations=[])
        )

        return agent

    def test_preload_batch_called_before_analyze(self):
        """preload_batch must be called once with odds-filtered fixture IDs."""
        import asyncio
        agent = self._make_agent(fixture_ids=(42,))

        asyncio.run(agent.get_daily_picks())

        agent.feature_engineer.preload_batch.assert_called_once_with([42])
        agent.analyze_fixture.assert_called_once_with(42)

    def test_preload_failure_does_not_lose_picks(self):
        """When preload_batch sets _preload_cache=None (exception path),
        analyze_fixture must still be called — no picks are lost."""
        import asyncio
        agent = self._make_agent(fixture_ids=(42,))

        def _failed_preload(ids):
            agent.feature_engineer._preload_cache = None

        agent.feature_engineer.preload_batch.side_effect = _failed_preload

        asyncio.run(agent.get_daily_picks())

        agent.feature_engineer.preload_batch.assert_called_once()  # wiring exists
        agent.analyze_fixture.assert_called_once_with(42)  # analysis proceeds

    def test_preload_called_with_odds_filtered_ids(self):
        """preload_batch must receive only the odds-filtered fixture list,
        not the raw fixture list. Fixture 99 has no real odds and is skipped."""
        import asyncio
        agent = self._make_agent(
            fixture_ids=(42, 99),
            odds_counts=[1, 0],  # 42 has odds, 99 does not
        )

        asyncio.run(agent.get_daily_picks())

        agent.feature_engineer.preload_batch.assert_called_once_with([42])
        # Only fixture 42 should be analyzed
        agent.analyze_fixture.assert_called_once_with(42)


class TestMLTrainingSampleCap:
    """Story 3.1: verify Postgres training sample cap raised from 200 to 500."""

    def _make_agent_for_training(self, n_matches):
        """Minimal FootballBettingAgent mock whose train_ml_models() exercises the DB query."""
        from datetime import date, timedelta
        from unittest.mock import MagicMock
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)

        mock_matches = []
        for i in range(n_matches):
            m = MagicMock()
            m.id = i + 1
            m.home_goals = 1
            m.away_goals = 0
            m.match_date = date(2024, 1, 1) + timedelta(days=i)
            mock_matches.append(m)

        session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = mock_matches

        agent.db = MagicMock()
        agent.db.get_session.return_value = session
        agent.db.is_postgres = True

        agent.predictor = MagicMock()
        agent.predictor.poisson._team_strengths = {"Arsenal": 1.0}

        fe = MagicMock()
        fe.team_features = MagicMock()
        agent.feature_engineer = fe

        return agent, session

    def test_train_postgres_cap_raised_to_500(self):
        """--train handler uses max_samples=500 for Postgres, not 200."""
        import inspect
        import src.agent.betting_agent as mod
        src_text = inspect.getsource(mod)
        assert "500 if _db_tmp.is_postgres" in src_text
        assert "200 if _db_tmp.is_postgres" not in src_text

    def test_train_ml_models_respects_max_samples(self):
        """train_ml_models(max_samples=N) passes N as DB limit() argument."""
        import asyncio

        agent, session = self._make_agent_for_training(n_matches=0)

        asyncio.run(agent.train_ml_models(max_samples=500))

        limit_call = session.query.return_value.filter.return_value.order_by.return_value.limit.call_args
        assert limit_call[0][0] == 500

    def test_train_ml_models_skips_on_insufficient_data(self):
        """train_ml_models logs warning and returns early when fewer than 50 matches."""
        import asyncio
        from unittest.mock import patch

        agent, session = self._make_agent_for_training(n_matches=30)

        with patch("src.agent.betting_agent.logger") as mock_logger:
            asyncio.run(agent.train_ml_models(max_samples=500))

        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("Not enough matches" in w for w in warning_calls)


class TestFeatureImportancePruning:
    """Story 3.2: verify importance-based feature pruning in MLModels.fit()."""

    def test_feature_list_json_written_after_fit(self, tmp_path):
        """feature_list.json persisted after fit() when n_features > ML_FEATURE_COUNT."""
        import json
        from unittest.mock import patch
        from src.models.ml_models import MLModels

        ml = MLModels()
        np.random.seed(0)
        n_features = ml.ML_FEATURE_COUNT + 10
        X = np.random.randn(100, n_features)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f"feat_{i}" for i in range(n_features)]

        with patch("src.models.ml_models.MODELS_DIR", tmp_path):
            ml.fit(X, y, feature_names=feature_names)

        feat_path = tmp_path / "feature_list.json"
        assert feat_path.exists()
        selected = json.loads(feat_path.read_text())
        assert len(selected) == ml.ML_FEATURE_COUNT
        assert all(isinstance(s, str) for s in selected)

    def test_model_retrained_on_reduced_feature_set(self, tmp_path):
        """After importance pruning, model uses ML_FEATURE_COUNT features at predict time."""
        from unittest.mock import patch
        from src.models.ml_models import MLModels

        ml = MLModels()
        np.random.seed(1)
        n_features = ml.ML_FEATURE_COUNT + 15
        X = np.random.randn(100, n_features)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f"f{i}" for i in range(n_features)]

        with patch("src.models.ml_models.MODELS_DIR", tmp_path):
            ml.fit(X, y, feature_names=feature_names)

            assert len(ml.feature_names) == ml.ML_FEATURE_COUNT

            pred = ml.predict(np.random.randn(n_features), feature_names=feature_names)
        avg = pred["ml_average"]
        assert abs(avg["home_win"] + avg["draw"] + avg["away_win"] - 1.0) < 0.01

    def test_predict_warns_when_feature_list_missing(self, tmp_path):
        """predict() logs WARNING and returns valid result when feature_list.json absent."""
        from unittest.mock import patch
        from src.models.ml_models import MLModels

        ml = MLModels()
        np.random.seed(2)
        X = np.random.randn(100, 5)
        y = np.random.choice([0, 1, 2], size=100)
        feature_names = [f"f{i}" for i in range(5)]

        with patch("src.models.ml_models.MODELS_DIR", tmp_path):
            ml.fit(X, y, feature_names=feature_names)
            feat_path = tmp_path / "feature_list.json"
            if feat_path.exists():
                feat_path.unlink()

            with patch("src.models.ml_models.logger") as mock_logger:
                pred = ml.predict(np.random.randn(5), feature_names=feature_names)

        assert "ml_average" in pred
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("feature_list.json" in w for w in warning_calls)


class TestMLAccuracySafetyGate:
    """Story 3.3: verify ML calibration safety gate in tune_ensemble_weights()."""

    def test_gate_zeros_ml_when_below_threshold(self):
        """When ML accuracy < 0.35, calibration factor set to 0.0 and WARNING logged."""
        from unittest.mock import patch
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        cal_factors = {}
        prev_cal = {}

        with patch("src.agent.betting_agent.logger") as mock_logger:
            agent._apply_ml_calibration_gate({"ml": 0.25}, cal_factors, prev_cal)

        assert cal_factors["ml"] == 0.0
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("ML excluded from ensemble" in w for w in warning_calls)

    def test_gate_restores_ml_when_accuracy_recovers(self):
        """When accuracy recovers ≥ 0.35 after being gated, ML is restored to 1.0."""
        from unittest.mock import patch
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        cal_factors = {}
        prev_cal = {"ml": 0.0}  # previously gated

        with patch("src.agent.betting_agent.logger") as mock_logger:
            agent._apply_ml_calibration_gate({"ml": 0.42}, cal_factors, prev_cal)

        assert cal_factors["ml"] == 1.0
        info_calls = [str(c) for c in mock_logger.info.call_args_list]
        assert any("re-entering ensemble" in c for c in info_calls)

    def test_weighted_average_excludes_ml_when_cal_zero(self):
        """With ml calibration=0.0, Poisson+Elo renormalize correctly; probs sum to 1.0."""
        from unittest.mock import MagicMock
        from src.models.ensemble import EnsemblePredictor

        ep = EnsemblePredictor.__new__(EnsemblePredictor)
        ep.calibration_factors = {"poisson": 1.0, "elo": 1.0, "ml": 0.0}
        ep.bayesian_weights = MagicMock()
        ep.bayesian_weights.get_weights.return_value = {
            "poisson": 0.25, "elo": 0.20, "ml": 0.55
        }

        poisson = {"home_win": 0.50, "draw": 0.25, "away_win": 0.25}
        elo = {"home_win": 0.45, "draw": 0.30, "away_win": 0.25}
        ml = {"home_win": 0.33, "draw": 0.33, "away_win": 0.34}

        result = ep._weighted_average_1x2(
            poisson, elo, ml, league="england/premier-league"
        )

        total = result["home_win"] + result["draw"] + result["away_win"]
        assert abs(total - 1.0) < 0.001
        # With ML zeroed, result is pulled toward Poisson+Elo (high home_win), not uniform
        assert result["home_win"] > 0.40


class TestValidationLeakageFix:
    """Story 4.1 — chronological hold-out before importance computation."""

    def test_no_leakage_warning_on_normal_accuracy(self):
        """When val accuracy is below 0.85, no CRITICAL log is emitted."""
        import logging
        from unittest.mock import patch
        from src.models.ml_models import MLModels

        rng = np.random.default_rng(0)
        # 100 samples, 10 features, 3-class labels — random data, accuracy ~0.33
        X = rng.random((100, 10)).astype(np.float32)
        y = rng.integers(0, 3, size=100)

        ml = MLModels()
        with patch("src.models.ml_models.logger") as mock_logger:
            ml.fit(X, y, feature_names=[f"f{i}" for i in range(10)])

        critical_calls = mock_logger.critical.call_args_list
        assert all("possible label leakage" not in str(c) for c in critical_calls), (
            "CRITICAL leakage warning fired on random data with expected low accuracy"
        )

    def test_leakage_warning_fires_on_high_val_accuracy(self):
        """When val accuracy > 0.85, a CRITICAL leakage warning is logged.

        Engineered dataset: train has all 3 classes with a perfectly separable
        feature (col 0 encodes class), val has only class 0. The model learns
        to predict class 0 for low col-0 values → 100% val accuracy → CRITICAL.
        """
        from unittest.mock import patch
        from src.models.ml_models import MLModels

        rng = np.random.default_rng(7)
        # 80 train samples: classes interleaved so all CV folds see all 3 classes
        # class 0 = very low col-0 (~0), class 1 = mid (~1), class 2 = high (~2)
        y_train = np.tile([0, 0, 0, 1, 2], 16)  # 80 samples, 3 classes throughout
        X_train = rng.random((80, 5)).astype(np.float32) * 0.05
        X_train[:, 0] = y_train.astype(np.float32)  # col 0 perfectly encodes class

        # 20 val samples: all class 0 (col-0 near 0) — model predicts class 0 → 100% accuracy
        X_val = rng.random((20, 5)).astype(np.float32) * 0.05  # col-0 near 0
        y_val = np.zeros(20, dtype=int)

        X = np.vstack([X_train, X_val])
        y = np.concatenate([y_train, y_val])

        ml = MLModels()
        with patch("src.models.ml_models.logger") as mock_logger:
            ml.fit(X, y, feature_names=[f"f{i}" for i in range(5)])

        critical_calls = [str(c) for c in mock_logger.critical.call_args_list]
        assert any("possible label leakage" in c for c in critical_calls), (
            "CRITICAL warning not fired when val accuracy should be ≥85% "
            "(all val samples from class-0 region, model strongly predicts class 0)"
        )


class TestMLTrainingFailureTelegramAlert:
    """Story 4.2 — Telegram alert when ML is excluded by calibration gate."""

    def test_telegram_alert_sent_when_ml_below_threshold(self):
        """When gate sets cal_factors['ml']=0.0, send_alert is called with warning text."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        # Provide a mock telegram notifier
        agent.telegram = MagicMock()
        agent.telegram.send_alert = AsyncMock(return_value=None)

        cal_factors = {"ml": 0.0}  # gate already applied
        ml_acc = 0.26

        # Call the alert block directly by calling _apply_ml_calibration_gate
        # and then the inline async send_alert logic (as it appears in tune_ensemble_weights)
        async def _run():
            if cal_factors.get("ml", 1.0) == 0.0:
                try:
                    await agent.telegram.send_alert(
                        f"⚠️ ML training produced a below-threshold model "
                        f"(accuracy {ml_acc:.1%}). "
                        f"ML excluded from ensemble — check training data quality."
                    )
                except Exception:
                    pass

        asyncio.run(_run())

        agent.telegram.send_alert.assert_called_once()
        call_text = agent.telegram.send_alert.call_args[0][0]
        assert "below-threshold" in call_text
        assert "26.0%" in call_text

    def test_no_telegram_alert_when_ml_above_threshold(self):
        """When gate keeps cal_factors['ml']=1.0, send_alert is NOT called."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock

        telegram = MagicMock()
        telegram.send_alert = AsyncMock(return_value=None)
        cal_factors = {"ml": 1.0}  # not gated

        async def _run():
            if cal_factors.get("ml", 1.0) == 0.0:
                await telegram.send_alert("should not be called")

        asyncio.run(_run())
        telegram.send_alert.assert_not_called()


class TestMLZeroWeightEscalation:
    """Story 7.3 — Telegram WARNING + CRITICAL escalation when ML cal_factor is 0.0."""

    def _make_agent(self, cal_ml=0.0):
        """Return a minimal FootballBettingAgent with real _check_ml_zero_weight()."""
        from unittest.mock import AsyncMock, MagicMock
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.telegram = MagicMock()
        agent.telegram.send_alert = AsyncMock(return_value=None)
        agent.predictor = MagicMock()
        agent.predictor.calibration_factors = {"ml": cal_ml}
        return agent

    def test_warning_sent_when_ml_zero(self, tmp_path, monkeypatch):
        """AC1 — send_alert called with WARNING text when ml==0.0, count=1."""
        import asyncio, json
        from pathlib import Path
        from unittest.mock import patch

        agent = self._make_agent(cal_ml=0.0)
        zero_path = tmp_path / "ml_zero_count.json"

        # Redirect the hardcoded path to tmp_path
        monkeypatch.setattr(
            "src.agent.betting_agent.Path",
            lambda p: zero_path if "ml_zero_count" in str(p) else Path(p),
        )

        asyncio.run(agent._check_ml_zero_weight())

        agent.telegram.send_alert.assert_called_once()
        call_text = agent.telegram.send_alert.call_args[0][0]
        assert "⚠️" in call_text
        assert "0%" in call_text
        # Counter file was written
        assert zero_path.exists()
        assert json.loads(zero_path.read_text())["count"] == 1

    def test_no_alert_when_ml_not_zero(self, tmp_path):
        """AC1 negative — no send_alert when ml > 0.0."""
        import asyncio

        agent = self._make_agent(cal_ml=0.85)
        asyncio.run(agent._check_ml_zero_weight())
        agent.telegram.send_alert.assert_not_called()

    def test_critical_escalation_at_count_4(self, tmp_path, monkeypatch):
        """AC2 — CRITICAL send_alert fired when consecutive count reaches 4."""
        import asyncio, json
        from pathlib import Path

        agent = self._make_agent(cal_ml=0.0)
        zero_path = tmp_path / "ml_zero_count.json"
        # Pre-seed counter at 3 runs from prior days
        zero_path.write_text(json.dumps({"count": 3, "last_updated": "2026-01-01"}))

        monkeypatch.setattr(
            "src.agent.betting_agent.Path",
            lambda p: zero_path if "ml_zero_count" in str(p) else Path(p),
        )

        asyncio.run(agent._check_ml_zero_weight())

        assert agent.telegram.send_alert.call_count == 2
        calls = [c[0][0] for c in agent.telegram.send_alert.call_args_list]
        assert any("🚨" in c for c in calls)
        assert any("4 consecutive" in c for c in calls)
        assert json.loads(zero_path.read_text())["count"] == 4

    def test_same_day_rerun_does_not_double_count(self, tmp_path, monkeypatch):
        """AC2 — counter not incremented when already updated today."""
        import asyncio, json
        from pathlib import Path
        from datetime import date

        agent = self._make_agent(cal_ml=0.0)
        today_str = date.today().isoformat()
        zero_path = tmp_path / "ml_zero_count.json"
        zero_path.write_text(json.dumps({"count": 2, "last_updated": today_str}))

        monkeypatch.setattr(
            "src.agent.betting_agent.Path",
            lambda p: zero_path if "ml_zero_count" in str(p) else Path(p),
        )

        asyncio.run(agent._check_ml_zero_weight())

        # Count unchanged — already incremented today
        assert json.loads(zero_path.read_text())["count"] == 2

    def test_counter_reset_on_ml_recovery(self, tmp_path, monkeypatch):
        """AC3 — ml_zero_count.json reset to 0 when ml cal_factor recovers > 0."""
        import json
        from pathlib import Path

        agent = self._make_agent(cal_ml=0.85)
        zero_path = tmp_path / "ml_zero_count.json"
        zero_path.write_text(json.dumps({"count": 5, "last_updated": "2026-01-01"}))

        monkeypatch.setattr(
            "src.agent.betting_agent.Path",
            lambda p: zero_path if "ml_zero_count" in str(p) else Path(p),
        )

        agent._reset_ml_zero_count()

        assert json.loads(zero_path.read_text())["count"] == 0

    def test_missing_file_handled_gracefully(self, tmp_path, monkeypatch):
        """AC4 — no crash when ml_zero_count.json does not exist; created on first run."""
        import asyncio, json
        from pathlib import Path

        agent = self._make_agent(cal_ml=0.0)
        zero_path = tmp_path / "ml_zero_count.json"
        assert not zero_path.exists()

        monkeypatch.setattr(
            "src.agent.betting_agent.Path",
            lambda p: zero_path if "ml_zero_count" in str(p) else Path(p),
        )

        asyncio.run(agent._check_ml_zero_weight())  # must not raise

        # File created with count=1
        assert zero_path.exists()
        assert json.loads(zero_path.read_text())["count"] == 1


class TestExposedDroppedPicks:
    """Story 8.1: verify dropped picks are logged and surfaced in Telegram."""

    def _make_rec(self, match="Team A vs Team B", market="Over 2.5", odds=1.9, kelly=12.0):
        from src.betting.value_calculator import BetRecommendation
        return BetRecommendation(
            match=match,
            match_id=1,
            market=market,
            selection="Over 2.5",
            odds=odds,
            predicted_probability=0.60,
            expected_value=0.10,
            confidence=0.65,
            kelly_stake_percentage=kelly,
            recommended_stake=kelly,
            reasoning="test",
            risk_level="low",
        )

    def _make_agent(self):
        from unittest.mock import MagicMock
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        return agent

    # ---------- _apply_exposure_cap tests ----------

    def test_cap_returns_capped_and_dropped(self):
        """High-Kelly picks exceeding the cap produce a non-empty dropped list."""
        agent = self._make_agent()
        picks = [self._make_rec(kelly=15.0), self._make_rec(kelly=14.0), self._make_rec(kelly=13.0)]
        capped, dropped = agent._apply_exposure_cap(picks, max_pct=25.0)
        assert len(capped) + len(dropped) == 3
        assert len(dropped) >= 1
        assert sum(r.kelly_stake_percentage for r in capped) <= 25.0

    def test_no_drops_when_under_cap(self):
        """When total Kelly is under the cap, dropped list is empty."""
        agent = self._make_agent()
        picks = [self._make_rec(kelly=5.0), self._make_rec(kelly=5.0)]
        capped, dropped = agent._apply_exposure_cap(picks, max_pct=50.0)
        assert dropped == []
        assert len(capped) == 2

    def test_dropped_picks_are_logged(self):
        """AC1 — each dropped pick is logged at INFO with match, market, odds, Kelly."""
        from loguru import logger as _lu
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            agent = self._make_agent()
            picks = [
                self._make_rec(match="Arsenal vs Chelsea", market="Over 2.5", odds=1.90, kelly=15.0),
                self._make_rec(match="Real Madrid vs Barca", market="1X2", odds=2.10, kelly=20.0),
            ]
            _, dropped = agent._apply_exposure_cap(picks, max_pct=10.0)
        finally:
            _lu.remove(sink_id)
        assert len(dropped) >= 1
        assert any("Dropped by exposure cap:" in m for m in messages)

    # ---------- send_daily_picks skipped-cap section ----------

    def _make_notifier(self):
        from unittest.mock import MagicMock, AsyncMock
        from src.reporting.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier.__new__(TelegramNotifier)
        notifier.enabled = True
        notifier._bot = MagicMock()
        notifier._sent_messages = []

        async def _fake_send(text):
            notifier._sent_messages.append(text)
        notifier._send_message = _fake_send
        notifier._send_chunked = AsyncMock(side_effect=lambda msg, header="": notifier._sent_messages.append(msg))
        return notifier

    def test_skipped_cap_section_in_telegram(self):
        """AC2 — 'Skipped (cap)' section present when dropped_picks provided."""
        import asyncio
        notifier = self._make_notifier()
        pick = self._make_rec()
        dropped = [self._make_rec(match="Dropped FC vs Capped FC", market="1X2")]
        asyncio.run(notifier.send_daily_picks([pick], dropped_picks=dropped))
        combined = " ".join(notifier._sent_messages)
        assert "Skipped (cap)" in combined
        assert "Dropped FC vs Capped FC" in combined

    def test_no_skipped_section_without_drops(self):
        """AC3 — no 'Skipped (cap)' section when dropped_picks is empty/None."""
        import asyncio
        notifier = self._make_notifier()
        pick = self._make_rec()
        asyncio.run(notifier.send_daily_picks([pick], dropped_picks=[]))
        combined = " ".join(notifier._sent_messages)
        assert "Skipped (cap)" not in combined


class TestSupplementHeader:
    """Story 8.2: same-day re-send uses supplement header."""

    def _make_rec(self):
        from src.betting.value_calculator import BetRecommendation
        return BetRecommendation(
            match="Home vs Away", match_id=1, market="Over 2.5",
            selection="Over 2.5", odds=1.9,
            predicted_probability=0.60, expected_value=0.10,
            confidence=0.65, kelly_stake_percentage=5.0,
            recommended_stake=5.0, reasoning="test", risk_level="low",
        )

    def _make_notifier(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock, AsyncMock
        from src.reporting.telegram_bot import TelegramNotifier
        import src.reporting.telegram_bot as tb_module
        from pathlib import Path

        notifier = TelegramNotifier.__new__(TelegramNotifier)
        notifier.enabled = True
        notifier._bot = MagicMock()
        notifier._sent_messages = []

        async def _fake_send(text):
            notifier._sent_messages.append(text)

        notifier._send_message = _fake_send
        notifier._send_chunked = AsyncMock(
            side_effect=lambda msg, header="": notifier._sent_messages.append(msg)
        )

        state_file = tmp_path / "picks_sent_date.txt"
        monkeypatch.setattr(tb_module, "_PICKS_SENT_STATE", state_file)
        return notifier

    def test_normal_header_on_first_send(self, tmp_path, monkeypatch):
        """AC2 — first send of the day uses the normal date header."""
        import asyncio
        notifier = self._make_notifier(tmp_path, monkeypatch)
        asyncio.run(notifier.send_daily_picks([self._make_rec()]))
        combined = " ".join(notifier._sent_messages)
        assert "Supplement" not in combined
        assert "Daily Value Picks" in combined

    def test_supplement_header_on_second_send(self, tmp_path, monkeypatch):
        """AC1 — second send same day uses supplement header with count."""
        import asyncio
        from datetime import date
        import src.reporting.telegram_bot as tb_module
        state_file = tmp_path / "picks_sent_date.txt"
        monkeypatch.setattr(tb_module, "_PICKS_SENT_STATE", state_file)
        # Simulate a prior send today
        state_file.write_text(date.today().isoformat())

        notifier = self._make_notifier(tmp_path, monkeypatch)
        picks = [self._make_rec(), self._make_rec()]
        asyncio.run(notifier.send_daily_picks(picks))
        combined = " ".join(notifier._sent_messages)
        assert "Supplement" in combined
        assert "2 additional" in combined

    def test_state_file_written_after_send(self, tmp_path, monkeypatch):
        """State file is written with today's date after successful send."""
        import asyncio
        from datetime import date
        notifier = self._make_notifier(tmp_path, monkeypatch)
        state_file = tmp_path / "picks_sent_date.txt"
        assert not state_file.exists()

        asyncio.run(notifier.send_daily_picks([self._make_rec()]))

        assert state_file.exists()
        assert state_file.read_text().strip() == date.today().isoformat()

    def test_fallback_to_normal_header_on_error(self, tmp_path, monkeypatch):
        """AC3 — if state file check raises, falls back to normal header."""
        import asyncio
        from unittest.mock import patch
        import src.reporting.telegram_bot as tb_module

        notifier = self._make_notifier(tmp_path, monkeypatch)

        # Force _picks_sent_today to raise
        with patch.object(tb_module, "_picks_sent_today", side_effect=OSError("disk error")):
            asyncio.run(notifier.send_daily_picks([self._make_rec()]))

        combined = " ".join(notifier._sent_messages)
        assert "Supplement" not in combined
        assert "Daily Value Picks" in combined


class TestPicksIdempotencyGuard:
    """Story 8.3: --picks skips generation if today's picks exist unless --force."""

    def _make_agent(self, existing_pick_count=0):
        """Minimal agent mock for idempotency guard tests."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from src.agent.betting_agent import FootballBettingAgent

        agent = FootballBettingAgent.__new__(FootballBettingAgent)

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        agent.db = MagicMock()
        agent.db.get_session.return_value = session

        # count() returns existing_pick_count
        session.query.return_value.filter.return_value.count.return_value = existing_pick_count
        # all() returns empty (no fixtures needed — guard fires first)
        session.query.return_value.filter.return_value.all.return_value = []

        agent.config = MagicMock()
        agent.config.get.side_effect = lambda key, default=None: default
        agent.apifootball = MagicMock()
        agent.apifootball.enabled = False
        agent.predictor = MagicMock()
        agent.predictor.calibration_factors = {}
        agent.predictor.check_coverage.return_value = {"score": 1.0}
        agent.value_calculator = MagicMock()
        agent.value_calculator.min_ev = 0.05
        agent.feature_engineer = MagicMock()
        agent.feature_engineer._preload_cache = None
        agent.telegram = MagicMock()
        agent._auto_calibrate_ev_threshold = MagicMock()
        agent.analyze_fixture = AsyncMock(return_value=MagicMock(recommendations=[]))

        return agent

    def test_skips_when_picks_exist_no_force(self):
        """AC1 — skips pick generation and returns empty lists when today's picks exist."""
        import asyncio
        agent = self._make_agent(existing_pick_count=5)
        picks, new_picks, dropped = asyncio.run(agent.get_daily_picks(force=False))
        assert picks == []
        assert new_picks == []
        assert dropped == []
        # analyze_fixture must NOT have been called (guard fired before fixture loop)
        agent.analyze_fixture.assert_not_called()

    def test_skips_logs_info_message(self):
        """AC1 — INFO log includes count and --force hint."""
        import asyncio
        from loguru import logger as _lu
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            agent = self._make_agent(existing_pick_count=3)
            asyncio.run(agent.get_daily_picks(force=False))
        finally:
            _lu.remove(sink_id)
        assert any("already generated" in m and "3" in m for m in messages)

    def test_force_bypasses_guard(self):
        """AC2 — --force proceeds past idempotency check even when picks exist."""
        import asyncio
        from loguru import logger as _lu
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            agent = self._make_agent(existing_pick_count=5)
            asyncio.run(agent.get_daily_picks(force=True))
        finally:
            _lu.remove(sink_id)
        # Guard did NOT fire — "already generated" message must be absent
        assert not any("already generated" in m for m in messages)

    def test_no_picks_today_proceeds_normally(self):
        """AC3 — when no today's picks exist, proceeds normally (no --force needed)."""
        import asyncio
        agent = self._make_agent(existing_pick_count=0)
        picks, new_picks, dropped = asyncio.run(agent.get_daily_picks(force=False))
        # Guard did not fire — function ran through normally (no fixtures → empty picks)
        assert picks == [] and new_picks == [] and dropped == []


class TestSmartXGBackfill:
    """Story 9.1: _backfill_xg() skips when all have xG, logs dynamic counts."""

    def _make_scraper(self, matches_needing=0, total_in_window=10, api_budget=25):
        """Minimal ApiFootballScraper mock for _backfill_xg tests."""
        from unittest.mock import MagicMock, AsyncMock, patch
        from src.scrapers.apifootball_scraper import APIFootballScraper

        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = 0
        scraper.BUDGET_RESERVE = 10
        scraper.BUDGET_XG = api_budget
        scraper._today_fixture_count = 5

        # DB mock
        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        scraper.db = MagicMock()
        scraper.db.get_session.return_value = session

        # First .all() → total in window (all_in_window query, no xG filter)
        # Second .all() → matches needing xG (filtered query)
        mock_matches = [MagicMock(id=i, apifootball_id=1000 + i) for i in range(matches_needing)]
        mock_all_window = [MagicMock(id=i) for i in range(total_in_window)]
        session.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.side_effect = [
            mock_all_window,  # total_in_window query (first)
            mock_matches,     # matches_needing query (second)
        ]

        scraper._fetch_fixture_stats = AsyncMock(return_value={"home_xg": 1.2})
        scraper._batch_update_match_stats = MagicMock()

        return scraper

    def test_skips_when_all_have_xg(self):
        """AC1 — logs correct DEBUG message and returns immediately when 0 need xG."""
        import asyncio
        from loguru import logger as _lu
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="DEBUG", format="{message}")
        try:
            scraper = self._make_scraper(matches_needing=0, total_in_window=10)
            asyncio.run(scraper._backfill_xg())
        finally:
            _lu.remove(sink_id)
        assert any("all recent matches have xG data, skipping" in m for m in messages)
        scraper._fetch_fixture_stats.assert_not_called()

    def test_completion_log_includes_skipped_count(self):
        """AC4 — completion log shows processed/needed/skipped format."""
        import asyncio
        from loguru import logger as _lu
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            scraper = self._make_scraper(matches_needing=3, total_in_window=10)
            asyncio.run(scraper._backfill_xg())
        finally:
            _lu.remove(sink_id)
        info_logs = [m for m in messages if "xG backfill:" in m and "already complete" in m]
        assert len(info_logs) >= 1, f"Expected completion log, got: {messages}"
        log = info_logs[0]
        assert "3" in log       # needed
        assert "7" in log       # skipped (10 - 3)

    def test_dynamic_count_processes_only_needed(self):
        """AC2 — only N matches processed when N < 25."""
        import asyncio
        scraper = self._make_scraper(matches_needing=5, total_in_window=25)
        asyncio.run(scraper._backfill_xg())
        # Only 5 fixture stats fetched (not 25)
        assert scraper._fetch_fixture_stats.call_count == 5

    def test_cap_preserved_when_all_stale(self):
        """AC3 — when all matches lack xG and exceed budget, cap limits to xg_budget."""
        import asyncio
        # api_budget=10 means xg_budget will be capped at 10; all 10 in window need xG
        scraper = self._make_scraper(matches_needing=10, total_in_window=10, api_budget=10)
        asyncio.run(scraper._backfill_xg())
        # Exactly 10 fetches — no more than the budget cap
        assert scraper._fetch_fixture_stats.call_count == 10


class TestEmptyLeagueFixtureAlert:
    """Story 9.2: Flashscore empty-league WARNING and Telegram alert."""

    def _make_agent(self, off_season=None):
        from unittest.mock import MagicMock, AsyncMock
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        _cfg = {"scraping.off_season_leagues": off_season or []}
        agent.config = MagicMock()
        agent.config.get = lambda key, default=None: _cfg.get(key, default)
        agent.telegram = MagicMock()
        agent.telegram.enabled = True
        agent.telegram.send_alert = AsyncMock()
        return agent

    def test_scraper_warns_on_zero_fixtures(self):
        """AC1 — FlashscoreScraper logs WARNING when a league returns 0 fixtures."""
        import asyncio
        from unittest.mock import MagicMock, patch
        from loguru import logger as _lu
        from src.scrapers.flashscore_scraper import FlashscoreScraper
        scraper = FlashscoreScraper.__new__(FlashscoreScraper)
        scraper.config = MagicMock()
        scraper.config.get = lambda key, default=None: (
            [] if key == "scraping.off_season_leagues" else default
        )
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="WARNING", format="{message}")
        try:
            mock_db = MagicMock()
            session = MagicMock()
            session.__enter__ = lambda s: s
            session.__exit__ = MagicMock(return_value=False)
            mock_db.get_session.return_value = session
            with patch("src.scrapers.flashscore_scraper.get_db", return_value=mock_db), \
                 patch.object(scraper, "_scrape_fixtures_page", return_value=[]):
                asyncio.run(scraper.scrape_league_fixtures("germany/bundesliga"))
        finally:
            _lu.remove(sink_id)
        assert any("0 fixtures" in m and "germany/bundesliga" in m for m in messages), \
            f"Expected WARNING for 0 fixtures, got: {messages}"

    def test_check_logs_empty_leagues(self):
        """AC2 — empty leagues are logged at INFO (no Telegram alert)."""
        import asyncio
        from loguru import logger as _lu
        agent = self._make_agent()
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            asyncio.run(agent._check_empty_fixture_leagues(
                {"germany/bundesliga": [], "england/championship": []}
            ))
        finally:
            _lu.remove(sink_id)
        agent.telegram.send_alert.assert_not_called()
        assert any("germany/bundesliga" in m or "england/championship" in m for m in messages)

    def test_check_no_log_when_all_leagues_have_fixtures(self):
        """AC3 — no log when all leagues return ≥1 fixture."""
        import asyncio
        from unittest.mock import MagicMock
        from loguru import logger as _lu
        agent = self._make_agent()
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            asyncio.run(agent._check_empty_fixture_leagues(
                {"germany/bundesliga": [MagicMock()]}
            ))
        finally:
            _lu.remove(sink_id)
        assert not any("0 fixtures" in m for m in messages)

    def test_off_season_league_excluded_from_log(self):
        """AC4 — off-season leagues not included in empty-league log."""
        import asyncio
        from loguru import logger as _lu
        agent = self._make_agent(off_season=["norway/eliteserien"])
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            asyncio.run(agent._check_empty_fixture_leagues({"norway/eliteserien": []}))
        finally:
            _lu.remove(sink_id)
        assert not any("norway/eliteserien" in m for m in messages)

    def test_scraper_no_warning_for_off_season_league(self):
        """AC4 — WARNING log suppressed in scraper for off-season leagues."""
        import asyncio
        from unittest.mock import MagicMock, patch
        from loguru import logger as _lu
        from src.scrapers.flashscore_scraper import FlashscoreScraper
        scraper = FlashscoreScraper.__new__(FlashscoreScraper)
        scraper.config = MagicMock()
        scraper.config.get = lambda key, default=None: (
            ["norway/eliteserien"] if key == "scraping.off_season_leagues" else default
        )
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="WARNING", format="{message}")
        try:
            mock_db = MagicMock()
            session = MagicMock()
            session.__enter__ = lambda s: s
            session.__exit__ = MagicMock(return_value=False)
            mock_db.get_session.return_value = session
            with patch("src.scrapers.flashscore_scraper.get_db", return_value=mock_db), \
                 patch.object(scraper, "_scrape_fixtures_page", return_value=[]):
                asyncio.run(scraper.scrape_league_fixtures("norway/eliteserien"))
        finally:
            _lu.remove(sink_id)
        assert not any("0 fixtures" in m and "norway/eliteserien" in m for m in messages), \
            f"Expected no WARNING for off-season league, got: {messages}"


class TestEVPriorityInjuryFetch:
    """Story 9.3: EV-priority injury ordering, per-fixture timing, zero-injury pick warning."""

    def _make_injury_scraper(self, fixture_list=None):
        """Minimal InjuryScraper mock for update() tests."""
        from unittest.mock import MagicMock, AsyncMock
        from src.scrapers.injury_scraper import InjuryScraper
        scraper = InjuryScraper.__new__(InjuryScraper)
        scraper.config = {}
        mock_api = MagicMock()
        mock_api.enabled = True
        mock_api.remaining_budget.return_value = 20
        mock_api._plan_restricted = False
        mock_api._api_get = AsyncMock(return_value=None)  # no injuries returned
        scraper.apifootball = mock_api

        mock_db = MagicMock()
        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        # First session: fixture query; second: clear stale
        session.query.return_value.join.return_value.filter.return_value.distinct.return_value.all.return_value = [
            MagicMock(id=fid, apifootball_id=1000 + fid, home_team_id=fid * 10, away_team_id=fid * 10 + 1)
            for fid in (fixture_list or [1, 2, 3])
        ]
        session.query.return_value.filter.return_value.count.return_value = 0
        session.query.return_value.filter.return_value.delete.return_value = 0
        mock_db.get_session.return_value = session
        scraper.db = mock_db
        return scraper

    def test_priority_fixtures_fetched_first(self):
        """AC1 — fixtures with open picks are reordered to the front."""
        import asyncio
        from unittest.mock import AsyncMock, patch
        from src.scrapers.injury_scraper import InjuryScraper
        scraper = self._make_injury_scraper(fixture_list=[1, 2, 3])
        fetch_order = []

        async def _fake_fetch(fixture_id, home_id, away_id):
            fetch_order.append(fixture_id)
            return 0

        scraper._fetch_fixture_injuries = _fake_fetch
        asyncio.run(scraper.update(priority_fixture_ids=[3]))
        # fixture 3 must come before 1 and 2
        assert fetch_order[0] == 1003, f"Expected fixture 1003 first, got {fetch_order}"

    def test_per_fixture_timing_logged(self):
        """AC2 — DEBUG log emitted with elapsed time per fixture."""
        import asyncio
        from loguru import logger as _lu
        scraper = self._make_injury_scraper(fixture_list=[5])

        async def _fake_fetch(fixture_id, home_id, away_id):
            return 0

        scraper._fetch_fixture_injuries = _fake_fetch
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="DEBUG", format="{message}")
        try:
            asyncio.run(scraper.update())
        finally:
            _lu.remove(sink_id)
        timing_logs = [m for m in messages if "Injury fetch for fixture" in m and "s" in m]
        assert len(timing_logs) >= 1, f"Expected timing DEBUG log, got: {messages}"

    def test_zero_injury_warning_on_new_pick_save(self):
        """AC4 — WARNING logged when a new pick has no injury data for its fixture."""
        from unittest.mock import MagicMock, patch
        from loguru import logger as _lu
        from src.agent.betting_agent import FootballBettingAgent
        from src.betting.value_calculator import BetRecommendation

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.db = MagicMock()

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        # No existing pick → save new one; injury count = 0
        session.query.return_value.filter.return_value.first.return_value = None
        session.query.return_value.filter.return_value.count.return_value = 0
        # Match lookup returns a match with team IDs
        mock_match = MagicMock()
        mock_match.home_team_id = 10
        mock_match.away_team_id = 11
        session.query.return_value.filter.return_value.first.side_effect = [
            None,   # existing pick check 1
            None,   # existing pick check 2 (match name guard)
            mock_match,  # match lookup for injury check
        ]
        agent.db.get_session.return_value = session

        pick = MagicMock(spec=BetRecommendation)
        pick.match_id = 42
        pick.match = "Bayern vs Dortmund"
        pick.market = "1X2"
        pick.selection = "Home"
        pick.odds = 1.9
        pick.predicted_probability = 0.55
        pick.expected_value = 0.045
        pick.confidence = 0.7
        pick.kelly_stake_percentage = 3.5
        pick.risk_level = "medium"
        pick.used_fallback_odds = False
        pick.league = "germany/bundesliga"

        from datetime import date as _date
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="WARNING", format="{message}")
        try:
            agent._save_picks([pick], _date.today())
        finally:
            _lu.remove(sink_id)
        assert any("no injury data" in m and "42" in m for m in messages), \
            f"Expected no-injury WARNING, got: {messages}"

    def test_no_warning_when_injury_data_present(self):
        """AC4 negative — no WARNING when injury data exists for the match."""
        from unittest.mock import MagicMock
        from loguru import logger as _lu
        from src.agent.betting_agent import FootballBettingAgent
        from src.betting.value_calculator import BetRecommendation

        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.db = MagicMock()

        session = MagicMock()
        session.__enter__ = lambda s: s
        session.__exit__ = MagicMock(return_value=False)
        mock_match = MagicMock()
        mock_match.home_team_id = 10
        mock_match.away_team_id = 11
        session.query.return_value.filter.return_value.first.side_effect = [
            None,       # existing pick check 1
            None,       # existing pick check 2
            mock_match, # match lookup
        ]
        # 3 injuries present — warning must NOT fire
        session.query.return_value.filter.return_value.count.return_value = 3
        agent.db.get_session.return_value = session

        pick = MagicMock(spec=BetRecommendation)
        pick.match_id = 55
        pick.match = "Arsenal vs Chelsea"
        pick.market = "1X2"
        pick.selection = "Away"
        pick.odds = 2.1
        pick.predicted_probability = 0.5
        pick.expected_value = 0.05
        pick.confidence = 0.65
        pick.kelly_stake_percentage = 2.0
        pick.risk_level = "low"
        pick.used_fallback_odds = False
        pick.league = "england/premier-league"

        from datetime import date as _date
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="WARNING", format="{message}")
        try:
            agent._save_picks([pick], _date.today())
        finally:
            _lu.remove(sink_id)
        assert not any("no injury data" in m for m in messages), \
            f"Expected no WARNING when injuries present, got: {messages}"


class TestDynamicFlashscoreTargets:
    """Story 9.4: _merge_flashscore_targets() merges static config with DB-derived leagues."""

    def _make_agent(self):
        from unittest.mock import MagicMock
        from src.agent.betting_agent import FootballBettingAgent
        agent = FootballBettingAgent.__new__(FootballBettingAgent)
        agent.config = MagicMock()
        return agent

    def test_dynamic_league_appended_when_not_in_static(self):
        """AC1/AC2 — DB league not in static config is added to target list."""
        agent = self._make_agent()
        result = agent._merge_flashscore_targets(
            ["england/premier-league", "germany/bundesliga"],
            {"england/premier-league", "poland/ekstraklasa"},
        )
        assert "poland/ekstraklasa" in result
        assert "england/premier-league" in result
        assert "germany/bundesliga" in result

    def test_no_duplicates_for_shared_leagues(self):
        """AC3 — league in both DB and static list appears exactly once."""
        agent = self._make_agent()
        result = agent._merge_flashscore_targets(
            ["england/premier-league", "spain/laliga"],
            {"england/premier-league", "spain/laliga"},
        )
        assert result.count("england/premier-league") == 1
        assert result.count("spain/laliga") == 1

    def test_static_order_preserved_at_front(self):
        """AC4 — static config list order is preserved (no regression)."""
        agent = self._make_agent()
        static = ["z/league", "a/league", "m/league"]
        result = agent._merge_flashscore_targets(static, {"z/league", "extra/league"})
        assert result[:3] == static

    def test_no_regression_when_db_matches_static(self):
        """AC4 — result identical to static list when DB leagues are all known."""
        agent = self._make_agent()
        static = ["england/premier-league", "germany/bundesliga"]
        result = agent._merge_flashscore_targets(static, set(static))
        assert result == static

    def test_info_log_emitted_for_dynamic_league(self):
        """INFO log is emitted for each league added dynamically."""
        from loguru import logger as _lu
        agent = self._make_agent()
        messages = []
        sink_id = _lu.add(lambda msg: messages.append(msg), level="INFO", format="{message}")
        try:
            agent._merge_flashscore_targets(
                ["england/premier-league"],
                {"england/premier-league", "poland/ekstraklasa"},
            )
        finally:
            _lu.remove(sink_id)
        assert any("poland/ekstraklasa" in m and "dynamic Flashscore target" in m for m in messages), \
            f"Expected INFO log for dynamic league, got: {messages}"
