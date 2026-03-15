"""Machine learning prediction models."""

import hashlib
import hmac
import numpy as np
import os
import pandas as pd
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger()

MODELS_DIR = Path("data/models")

# HMAC key for model file integrity. Uses env var if set, otherwise a
# deterministic fallback so existing pickle files can still be loaded
# (with a warning) until they are re-saved with a proper key.
_MODEL_HMAC_KEY = os.environ.get("MODEL_HMAC_KEY", "betting-agent-default-key").encode()


def _compute_hmac(data: bytes) -> str:
    """Compute HMAC-SHA256 hex digest for model data."""
    return hmac.new(_MODEL_HMAC_KEY, data, hashlib.sha256).hexdigest()


def _safe_save(state: dict, filepath: Path):
    """Pickle state and write alongside an HMAC signature file."""
    data = pickle.dumps(state)
    sig = _compute_hmac(data)
    with open(filepath, "wb") as f:
        f.write(data)
    filepath.with_suffix(".sig").write_text(sig)


def _safe_load(filepath: Path) -> dict:
    """Load pickle after verifying HMAC signature. Raises on tampering."""
    data = filepath.read_bytes()
    sig_path = filepath.with_suffix(".sig")
    if sig_path.exists():
        expected = sig_path.read_text().strip()
        actual = _compute_hmac(data)
        if not hmac.compare_digest(expected, actual):
            raise RuntimeError(
                f"Model file {filepath} failed integrity check — "
                f"file may have been tampered with. Re-train models to fix."
            )
    else:
        logger.warning(
            f"No signature file for {filepath} — loading without verification. "
            f"Re-save models to create a signature."
        )
    return pickle.loads(data)  # noqa: S301


class MLModels:
    """Manages ML models for match outcome prediction.

    Includes XGBoost, Random Forest, and Logistic Regression classifiers.
    Targets: 0 = away_win, 1 = draw, 2 = home_win
    """

    # Models that need isotonic calibration (RF, XGBoost, LightGBM are overconfident;
    # Logistic Regression is already calibrated by its loss function)
    _CALIBRATE_MODELS = {"random_forest", "xgboost", "lightgbm"}

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.calibrated_models = {}   # post-calibration wrappers (used at predict time)
        self.feature_names = []
        self.is_fitted = False
        self._init_models()

    def _init_models(self):
        """Initialize all model instances."""
        self.models = {
            "logistic_regression": LogisticRegression(
                max_iter=1000, solver="lbfgs",
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                criterion="log_loss",
                random_state=42, n_jobs=-1,
            ),
        }

        # XGBoost is optional
        try:
            from xgboost import XGBClassifier
            self.models["xgboost"] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42, n_jobs=-1,
            )
        except ImportError:
            logger.warning("XGBoost not available, skipping")

        # LightGBM is optional
        try:
            from lightgbm import LGBMClassifier
            self.models["lightgbm"] = LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                objective="multiclass", num_class=3,
                random_state=42, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            logger.debug("LightGBM not available, skipping")

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train all models on the provided data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=away, 1=draw, 2=home)
            feature_names: Optional list of feature names
        """
        # Always reinitialize fresh model instances before training to avoid
        # version-incompatibility issues when loading older pickled models.
        self._init_models()
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Prune sparse features (>80% zeros) to reduce noise
        n_samples = X.shape[0]
        zero_fractions = np.mean(X == 0, axis=0)
        sparse_mask = zero_fractions > 0.80
        n_sparse = int(np.sum(sparse_mask))
        if n_sparse > 0:
            kept_mask = ~sparse_mask
            dropped = [self.feature_names[i] for i in range(len(self.feature_names)) if sparse_mask[i]]
            logger.info(f"Pruning {n_sparse} sparse features (>80% zeros): {dropped[:10]}{'...' if len(dropped) > 10 else ''}")
            X = X[:, kept_mask]
            self.feature_names = [f for f, keep in zip(self.feature_names, kept_mask) if keep]
            self._kept_feature_mask = kept_mask
        else:
            self._kept_feature_mask = None

        # Zero-variance pruning — constant features cause RuntimeWarning/NaN in np.corrcoef
        _var_mask = np.var(X, axis=0) > 0
        n_zero_var = int(np.sum(~_var_mask))
        if n_zero_var > 0:
            _zero_names = [self.feature_names[i] for i in range(len(self.feature_names)) if not _var_mask[i]]
            logger.info(f"Dropping {n_zero_var} zero-variance features: {_zero_names[:10]}{'...' if len(_zero_names) > 10 else ''}")
            X = X[:, _var_mask]
            self.feature_names = [f for f, keep in zip(self.feature_names, _var_mask) if keep]

        # Correlation pruning — drop one of any pair with |corr| > 0.8
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            upper = np.abs(np.triu(corr_matrix, k=1))
            corr_drop = np.zeros(X.shape[1], dtype=bool)
            for i in range(upper.shape[0]):
                if corr_drop[i]:
                    continue
                for j in range(i + 1, upper.shape[1]):
                    if upper[i, j] > 0.8:
                        corr_drop[j] = True
            n_corr = int(np.sum(corr_drop))
            if n_corr > 0:
                dropped_corr = [self.feature_names[i] for i in range(len(self.feature_names)) if corr_drop[i]]
                logger.info(
                    f"Pruning {n_corr} highly-correlated features (|corr|>0.8): "
                    f"{dropped_corr[:10]}{'...' if len(dropped_corr) > 10 else ''}"
                )
                keep_corr = ~corr_drop
                X = X[:, keep_corr]
                self.feature_names = [f for f, keep in zip(self.feature_names, keep_corr) if keep]
                self._corr_drop_mask = keep_corr
            else:
                self._corr_drop_mask = None
        else:
            self._corr_drop_mask = None

        logger.info(f"Training with {X.shape[1]} features, {n_samples} samples")

        # Scale features — use DataFrame with column names so LightGBM/XGBoost
        # store correct feature names (avoids sklearn UserWarning at predict time)
        X_scaled = self.scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_df):
                X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, val_pred))

            avg_cv = np.mean(cv_scores)
            logger.info(f"{name} CV accuracy: {avg_cv:.4f}")

        # ── Calibrated final fit ─────────────────────────────────────────────
        # Use 5-fold cross-calibration (sklearn >= 1.4 removed cv='prefit').
        # CalibratedClassifierCV with cv=5 fits the base estimator on 4 folds
        # and the isotonic calibrator on the held-out fold, then averages all
        # five calibrated models at prediction time — more robust than a single
        # holdout split and compatible with all sklearn versions.
        # LR is already probability-calibrated by its log-loss objective.
        self.calibrated_models = {}
        for name, model in self.models.items():
            if name in self._CALIBRATE_MODELS:
                cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
                cal.fit(X_df, y)
                self.calibrated_models[name] = cal
                logger.info(
                    f"{name}: isotonic calibration with 5-fold CV on {len(X_df)} samples"
                )
            else:
                # LR: train on all data (already probability-calibrated)
                model.fit(X_df, y)
                self.calibrated_models[name] = model

        self.is_fitted = True
        logger.info("All ML models trained successfully")

        # Log top 10 most important features
        importance = self.get_feature_importance()
        for model_name, imp_list in importance.items():
            top_10 = imp_list[:10]
            top_str = ", ".join(f"{name}({score:.3f})" for name, score in top_10)
            logger.info(f"{model_name} top features: {top_str}")

    def predict(self, X: np.ndarray, feature_names: List[str] = None) -> Dict:
        """Get predictions from all models.

        Args:
            X: Feature vector (1, n_features) or (n_features,)
            feature_names: Feature names matching X columns. When provided and
                the count differs from training, features are aligned by name.

        Returns:
            Dictionary with per-model and averaged probabilities
        """
        if not self.is_fitted:
            logger.warning("Models not fitted yet, returning uniform probabilities")
            return self._default_prediction()

        X = X.reshape(1, -1) if X.ndim == 1 else X

        # If feature count differs from training, align by name
        expected_count = len(self.feature_names)
        if feature_names and X.shape[1] != expected_count:
            X = self._align_features(X, feature_names)

        # Apply same feature pruning mask used during training
        if getattr(self, "_kept_feature_mask", None) is not None and X.shape[1] > expected_count:
            X = X[:, self._kept_feature_mask]

        try:
            X_scaled = self.scaler.transform(X)
        except ValueError:
            logger.warning(
                f"ML feature mismatch ({X.shape[1]} vs {expected_count} expected), "
                f"returning default prediction"
            )
            return self._default_prediction()

        # Wrap in DataFrame so LightGBM gets feature names (suppresses warning)
        X_named = pd.DataFrame(X_scaled, columns=self.feature_names)

        predictions = {}
        all_probs = []

        for name, model in self.models.items():
            # Use calibrated model if available (RF/XGBoost); otherwise raw (LR)
            cal_model = self.calibrated_models.get(name, model)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                probs = cal_model.predict_proba(X_named)[0]

            # Ensure we have 3 classes (away=0, draw=1, home=2)
            if len(probs) == 3:
                predictions[name] = {
                    "home_win": float(probs[2]),
                    "draw": float(probs[1]),
                    "away_win": float(probs[0]),
                }
                all_probs.append(probs)

        # Average across models
        if all_probs:
            avg_probs = np.mean(all_probs, axis=0)
            predictions["ml_average"] = {
                "home_win": round(float(avg_probs[2]), 4),
                "draw": round(float(avg_probs[1]), 4),
                "away_win": round(float(avg_probs[0]), 4),
                "model": "ml_average",
            }

        return predictions

    def get_feature_importance(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get feature importance from tree-based models."""
        importance = {}

        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = list(zip(self.feature_names, model.feature_importances_))
                imp.sort(key=lambda x: x[1], reverse=True)
                importance[name] = imp

        return importance

    def save(self, path: str = None):
        """Save all models to disk with HMAC integrity signature."""
        save_dir = Path(path) if path else MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        from src.utils.logger import utcnow as _utcnow
        state = {
            "models": self.models,
            "calibrated_models": getattr(self, "calibrated_models", {}),
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "_kept_feature_mask": getattr(self, "_kept_feature_mask", None),
            "_corr_drop_mask": getattr(self, "_corr_drop_mask", None),
            "trained_at": _utcnow().isoformat(),
        }

        filepath = save_dir / "ml_models.pkl"
        _safe_save(state, filepath)
        logger.info(f"Models saved to {filepath}")

    def load(self, path: str = None):
        """Load models from disk with HMAC integrity verification."""
        load_dir = Path(path) if path else MODELS_DIR
        filepath = load_dir / "ml_models.pkl"

        if not filepath.exists():
            logger.debug(f"No saved models found at {filepath} — using Poisson+Elo only")
            return

        try:
            state = _safe_load(filepath)
        except RuntimeError as e:
            logger.error(str(e))
            return

        self.models = state["models"]
        self.calibrated_models = state.get("calibrated_models", {})
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        self._kept_feature_mask = state.get("_kept_feature_mask")
        self._corr_drop_mask = state.get("_corr_drop_mask")
        self.trained_at = state.get("trained_at")

        logger.info(f"Models loaded from {filepath}")

    def _align_features(self, X: np.ndarray, feature_names: List[str]) -> np.ndarray:
        """Align X columns to match self.feature_names by name.

        Missing features get 0.0; extra features are dropped.
        """
        name_to_idx = {n: i for i, n in enumerate(feature_names)}
        aligned = np.zeros((X.shape[0], len(self.feature_names)))
        matched = 0
        for j, name in enumerate(self.feature_names):
            src_idx = name_to_idx.get(name)
            if src_idx is not None:
                aligned[:, j] = X[:, src_idx]
                matched += 1
        logger.debug(
            f"Feature alignment: {matched}/{len(self.feature_names)} matched "
            f"(input had {X.shape[1]} features)"
        )
        return aligned

    def _default_prediction(self) -> Dict:
        return {
            "ml_average": {
                "home_win": 0.40,
                "draw": 0.25,
                "away_win": 0.35,
                "model": "ml_average",
            }
        }


class GoalsMLModel:
    """Binary ML classifier for the 'over 2.5 goals' outcome.

    Uses the same feature set as the 1X2 classifier but targets a binary
    outcome: 1 = over 2.5 total goals, 0 = under/equal 2.5.
    Trained and saved separately from the main 1X2 models.
    """

    _CALIBRATE = {"random_forest", "xgboost", "lightgbm"}

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.calibrated_models = {}
        self.feature_names: List[str] = []
        self.is_fitted = False
        self._kept_feature_mask = None
        self._corr_drop_mask = None
        self._init_models()

    def _init_models(self):
        self.models = {
            "logistic_regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "random_forest": RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=5,
                criterion="log_loss",
                random_state=43, n_jobs=-1,
            ),
        }
        try:
            from xgboost import XGBClassifier
            self.models["xgboost"] = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                objective="binary:logistic", eval_metric="logloss",
                random_state=43, n_jobs=-1,
            )
        except ImportError:
            pass
        try:
            from lightgbm import LGBMClassifier
            self.models["lightgbm"] = LGBMClassifier(
                n_estimators=200, max_depth=8, learning_rate=0.1,
                objective="binary",
                random_state=43, n_jobs=-1, verbose=-1,
            )
        except ImportError:
            pass

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train the goals model.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Binary labels — 1 if total goals > 2.5, else 0
            feature_names: Optional list of feature names
        """
        self._init_models()
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]

        # Prune sparse features (>80% zeros)
        zero_fractions = np.mean(X == 0, axis=0)
        sparse_mask = zero_fractions > 0.80
        if np.any(sparse_mask):
            kept_mask = ~sparse_mask
            X = X[:, kept_mask]
            self.feature_names = [f for f, k in zip(self.feature_names, kept_mask) if k]
            self._kept_feature_mask = kept_mask

        # Zero-variance pruning — constant features cause RuntimeWarning/NaN in np.corrcoef
        _var_mask = np.var(X, axis=0) > 0
        n_zero_var = int(np.sum(~_var_mask))
        if n_zero_var > 0:
            _zero_names = [self.feature_names[i] for i in range(len(self.feature_names)) if not _var_mask[i]]
            logger.info(f"GoalsMLModel: dropping {n_zero_var} zero-variance features: {_zero_names[:10]}{'...' if len(_zero_names) > 10 else ''}")
            X = X[:, _var_mask]
            self.feature_names = [f for f, keep in zip(self.feature_names, _var_mask) if keep]

        # Correlation pruning — drop one of any pair with |corr| > 0.8
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X.T)
            upper = np.abs(np.triu(corr_matrix, k=1))
            corr_drop = np.zeros(X.shape[1], dtype=bool)
            for i in range(upper.shape[0]):
                if corr_drop[i]:
                    continue
                for j in range(i + 1, upper.shape[1]):
                    if upper[i, j] > 0.8:
                        corr_drop[j] = True
            if np.any(corr_drop):
                keep_corr = ~corr_drop
                X = X[:, keep_corr]
                self.feature_names = [f for f, k in zip(self.feature_names, keep_corr) if k]
                self._corr_drop_mask = keep_corr

        logger.info(f"GoalsMLModel: training with {X.shape[1]} features, {X.shape[0]} samples")
        X_scaled = self.scaler.fit_transform(X)
        X_df = pd.DataFrame(X_scaled, columns=self.feature_names)
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_df):
                X_tr, X_vl = X_df.iloc[train_idx], X_df.iloc[val_idx]
                y_tr, y_vl = y[train_idx], y[val_idx]
                model.fit(X_tr, y_tr)
                cv_scores.append(accuracy_score(y_vl, model.predict(X_vl)))
            logger.info(f"GoalsMLModel {name} CV accuracy: {np.mean(cv_scores):.4f}")

        self.calibrated_models = {}
        for name, model in self.models.items():
            if name in self._CALIBRATE:
                cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
                cal.fit(X_df, y)
                self.calibrated_models[name] = cal
            else:
                model.fit(X_df, y)
                self.calibrated_models[name] = model

        self.is_fitted = True
        logger.info("GoalsMLModel training complete")

    def predict_proba_over25(self, X: np.ndarray,
                             feature_names: List[str] = None) -> float:
        """Return P(over 2.5 total goals). Feature alignment handled automatically.

        Returns 0.5 (uninformative prior) when model is not fitted or on error.
        """
        if not self.is_fitted:
            return 0.5

        X = X.reshape(1, -1) if X.ndim == 1 else X

        # Align features by name when provided
        if feature_names and X.shape[1] != len(self.feature_names):
            name_to_idx = {n: i for i, n in enumerate(feature_names)}
            aligned = np.zeros((X.shape[0], len(self.feature_names)))
            for j, name in enumerate(self.feature_names):
                src = name_to_idx.get(name)
                if src is not None:
                    aligned[:, j] = X[:, src]
            X = aligned

        if self._kept_feature_mask is not None and X.shape[1] > len(self.feature_names):
            X = X[:, self._kept_feature_mask]

        try:
            X_scaled = self.scaler.transform(X)
        except ValueError:
            return 0.5

        # Wrap in DataFrame so LightGBM gets feature names (suppresses warning)
        X_named = pd.DataFrame(X_scaled, columns=self.feature_names)

        probs = []
        for name in self.models:
            cal_model = self.calibrated_models.get(name, self.models[name])
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message=".*feature names.*", category=UserWarning)
                    p = cal_model.predict_proba(X_named)[0]
                if len(p) >= 2:
                    probs.append(float(p[1]))  # p[1] = P(class=1) = P(over 2.5)
            except Exception:
                pass

        return float(np.mean(probs)) if probs else 0.5

    def save(self, path: str = None):
        """Save model to disk with HMAC integrity signature."""
        save_dir = Path(path) if path else MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        from src.utils.logger import utcnow as _utcnow
        state = {
            "models": self.models,
            "calibrated_models": self.calibrated_models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "_kept_feature_mask": self._kept_feature_mask,
            "_corr_drop_mask": self._corr_drop_mask,
            "trained_at": _utcnow().isoformat(),
        }
        filepath = save_dir / "goals_model.pkl"
        _safe_save(state, filepath)
        logger.info(f"GoalsMLModel saved to {filepath}")

    def load(self, path: str = None):
        """Load model from disk with HMAC integrity verification."""
        load_dir = Path(path) if path else MODELS_DIR
        filepath = load_dir / "goals_model.pkl"
        if not filepath.exists():
            logger.debug(f"No saved goals model at {filepath}")
            return
        try:
            state = _safe_load(filepath)
        except RuntimeError as e:
            logger.error(str(e))
            return
        self.models = state["models"]
        self.calibrated_models = state.get("calibrated_models", {})
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        self._kept_feature_mask = state.get("_kept_feature_mask")
        self._corr_drop_mask = state.get("_corr_drop_mask")
        self.trained_at = state.get("trained_at")
        logger.info(f"GoalsMLModel loaded from {filepath}")
