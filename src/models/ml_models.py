"""Machine learning prediction models."""

import hashlib
import hmac
import json
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

# HMAC key for model file integrity.
# In production MODEL_HMAC_KEY must be set so the integrity check provides
# a real guarantee — the previous deterministic fallback was source-visible
# and therefore offered no protection.  For local dev / tests, opt into the
# legacy default explicitly via BETTING_AGENT_ALLOW_DEFAULT_HMAC=1.
_MODEL_HMAC_KEY_ENV = os.environ.get("MODEL_HMAC_KEY")
if _MODEL_HMAC_KEY_ENV:
    _MODEL_HMAC_KEY = _MODEL_HMAC_KEY_ENV.encode()
    _MODEL_HMAC_USING_DEFAULT = False
elif os.environ.get("BETTING_AGENT_ALLOW_DEFAULT_HMAC") == "1":
    _MODEL_HMAC_KEY = b"betting-agent-default-key"
    _MODEL_HMAC_USING_DEFAULT = True
    logger.warning(
        "MODEL_HMAC_KEY not set; using insecure default key "
        "(BETTING_AGENT_ALLOW_DEFAULT_HMAC=1). "
        "Do NOT run this configuration in production — set MODEL_HMAC_KEY."
    )
else:
    raise RuntimeError(
        "MODEL_HMAC_KEY environment variable is not set. "
        "Set it to a secret value, or set BETTING_AGENT_ALLOW_DEFAULT_HMAC=1 "
        "to opt into the insecure default key for local development only."
    )


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


class IntegrityError(RuntimeError):
    """Raised when a model file's HMAC signature does not match its contents.

    Distinguished from RuntimeError so callers can react specifically to
    tampering (vs. missing files or unrelated load errors). The current
    callers log the message at ERROR and refuse to load — the agent then
    falls back to Poisson + Elo. They do NOT delete the suspect file: a
    human operator should investigate before re-saving.
    """


def _safe_load(filepath: Path) -> dict:
    """Load pickle after verifying HMAC signature.

    Raises IntegrityError when the signature exists but doesn't match —
    callers must surface this loudly rather than fall back silently to a
    fresh model, since a forged pickle would otherwise be loaded.
    """
    data = filepath.read_bytes()
    sig_path = filepath.with_suffix(".sig")
    if sig_path.exists():
        expected = sig_path.read_text().strip()
        actual = _compute_hmac(data)
        if not hmac.compare_digest(expected, actual):
            raise IntegrityError(
                f"Model file {filepath} failed integrity check — "
                f"signature mismatch.  File may have been tampered with, or "
                f"MODEL_HMAC_KEY changed since it was last saved.  Investigate "
                f"and re-train rather than auto-deleting."
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

    ML_FEATURE_COUNT = 50  # top features retained after importance pruning

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
            X: Feature matrix (n_samples, n_features) — MUST be in chronological order
                (oldest row first). The last 20% are held out as the validation set, so
                non-chronological ordering silently produces a contaminated split.
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

        # ── Chronological hold-out: last 20% reserved before ANY fitting (AC1) ─
        split_idx = int(len(X) * 0.80)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            f"Training with {X.shape[1]} features, "
            f"{len(X_train_raw)} train / {len(X_val_raw)} val samples "
            f"(total {n_samples})"
        )

        # Scale on training split only; apply same transform to validation (AC1)
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val_scaled, columns=self.feature_names)

        # Time-series cross-validation on training split only
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_df):
                X_cv_tr, X_cv_val = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
                y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]

                model.fit(X_cv_tr, y_cv_tr)
                val_pred = model.predict(X_cv_val)
                cv_scores.append(accuracy_score(y_cv_val, val_pred))

            avg_cv = np.mean(cv_scores)
            logger.info(f"{name} CV accuracy: {avg_cv:.4f}")

        # ── First-pass calibrated fit on training split only (AC2) ───────────
        self.calibrated_models = {}
        for name, model in self.models.items():
            if name in self._CALIBRATE_MODELS:
                cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
                cal.fit(X_train_df, y_train)
                self.calibrated_models[name] = cal
                logger.info(
                    f"{name}: isotonic calibration with 5-fold CV on {len(X_train_df)} samples"
                )
            else:
                model.fit(X_train_df, y_train)
                self.calibrated_models[name] = model

        self.is_fitted = True
        logger.info("First-pass ML models trained on training split")

        # Log top 10 most important features (from training-split models only, AC2)
        importance = self.get_feature_importance()
        for model_name, imp_list in importance.items():
            top_10 = imp_list[:10]
            top_str = ", ".join(f"{name}({score:.3f})" for name, score in top_10)
            logger.info(f"{model_name} top features: {top_str}")

        # ── Feature importance pruning ────────────────────────────────────────
        # Importances from training-split models only (AC2).
        # Second-pass re-fit also on training split only (AC3).
        if importance and len(self.feature_names) > self.ML_FEATURE_COUNT:
            avg_scores = {fn: 0.0 for fn in self.feature_names}
            n_imp_models = 0
            for imp_list in importance.values():
                for feat_name, score in imp_list:
                    if feat_name in avg_scores:
                        avg_scores[feat_name] += score
                n_imp_models += 1
            if n_imp_models:
                for k in avg_scores:
                    avg_scores[k] /= n_imp_models

            top_names = sorted(avg_scores, key=lambda k: avg_scores[k], reverse=True)[
                : self.ML_FEATURE_COUNT
            ]
            top_set = set(top_names)
            selected_idx = [i for i, fn in enumerate(self.feature_names) if fn in top_set]

            X_train_pruned = X_train_raw[:, selected_idx]
            X_val_pruned = X_val_raw[:, selected_idx]
            self.feature_names = [self.feature_names[i] for i in selected_idx]

            logger.info(
                f"Feature importance pruning: retained top {len(self.feature_names)} "
                f"of {len(avg_scores)} features"
            )

            _feat_path = MODELS_DIR / "feature_list.json"
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            _feat_path.write_text(json.dumps(self.feature_names))

            # Re-fit scaler + re-train on reduced training split (AC3)
            self._init_models()
            X_train_scaled_top = self.scaler.fit_transform(X_train_pruned)
            X_val_scaled_top = self.scaler.transform(X_val_pruned)
            X_train_df_top = pd.DataFrame(X_train_scaled_top, columns=self.feature_names)
            X_val_df_top = pd.DataFrame(X_val_scaled_top, columns=self.feature_names)

            self.calibrated_models = {}
            for name, model in self.models.items():
                if name in self._CALIBRATE_MODELS:
                    cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
                    cal.fit(X_train_df_top, y_train)
                    self.calibrated_models[name] = cal
                else:
                    model.fit(X_train_df_top, y_train)
                    self.calibrated_models[name] = model

            X_val_df = X_val_df_top  # point to pruned val df for validation

        logger.info("All ML models trained successfully")

        # ── Validation on TRUE chronological hold-out 20% (AC3) ──────────────
        val_preds = []
        for name in self.models:
            cal_model = self.calibrated_models.get(name, self.models[name])
            try:
                val_preds.append(cal_model.predict(X_val_df))
            except Exception:
                pass
        if val_preds:
            votes = np.array(val_preds)
            majority = np.apply_along_axis(
                lambda col: np.bincount(col.astype(int)).argmax(), 0, votes
            )
            val_acc = accuracy_score(y_val, majority)
            logger.info(f"ML validation accuracy: {val_acc:.1%}")
            if val_acc > 0.85:
                logger.critical(
                    f"CRITICAL: val accuracy {val_acc:.1%} on 3-class target — "
                    "possible label leakage, inspect training data"
                )

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

        # AC3: warn if feature_list.json absent (old model or pre-importance-pruning run)
        _feat_path = MODELS_DIR / "feature_list.json"
        if not _feat_path.exists():
            logger.warning(
                "feature_list.json not found — using all available features. "
                "Run --train to generate feature importance list."
            )

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
        except IntegrityError as e:
            # Surface tampering at ERROR; do NOT auto-fall-back silently —
            # log the file path so operators can investigate.
            logger.error(f"INTEGRITY FAILURE: {e}")
            return
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Could not load {filepath}: {e}")
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
            X: Feature matrix (n_samples, n_features) — MUST be in chronological order
                (oldest row first). The last 20% are held out as the validation set.
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

        # ── Chronological hold-out: last 20% before any fitting ──────────────
        split_idx = int(len(X) * 0.80)
        X_train_raw, X_val_raw = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        logger.info(
            f"GoalsMLModel: training with {X.shape[1]} features, "
            f"{len(X_train_raw)} train / {len(X_val_raw)} val samples"
        )
        X_train_scaled = self.scaler.fit_transform(X_train_raw)
        X_val_scaled = self.scaler.transform(X_val_raw)
        X_train_df = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_val_df = pd.DataFrame(X_val_scaled, columns=self.feature_names)
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            cv_scores = []
            for train_idx, val_idx in tscv.split(X_train_df):
                X_tr, X_vl = X_train_df.iloc[train_idx], X_train_df.iloc[val_idx]
                y_tr, y_vl = y_train[train_idx], y_train[val_idx]
                model.fit(X_tr, y_tr)
                cv_scores.append(accuracy_score(y_vl, model.predict(X_vl)))
            logger.info(f"GoalsMLModel {name} CV accuracy: {np.mean(cv_scores):.4f}")

        self.calibrated_models = {}
        for name, model in self.models.items():
            if name in self._CALIBRATE:
                cal = CalibratedClassifierCV(model, cv=5, method="isotonic")
                cal.fit(X_train_df, y_train)
                self.calibrated_models[name] = cal
            else:
                model.fit(X_train_df, y_train)
                self.calibrated_models[name] = model

        self.is_fitted = True

        # ── Validate on held-out chronological 20% ────────────────────────────
        val_preds = []
        for name in self.models:
            cal_model = self.calibrated_models.get(name, self.models[name])
            try:
                val_preds.append(cal_model.predict(X_val_df))
            except Exception:
                pass
        if val_preds:
            votes = np.array(val_preds)
            majority = np.apply_along_axis(
                lambda col: np.bincount(col.astype(int)).argmax(), 0, votes
            )
            val_acc = accuracy_score(y_val, majority)
            logger.info(f"GoalsMLModel validation accuracy: {val_acc:.1%}")
            if val_acc > 0.90:
                logger.critical(
                    f"CRITICAL: GoalsMLModel val accuracy {val_acc:.1%} on binary target — "
                    "possible label leakage, inspect training data"
                )

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
        except IntegrityError as e:
            logger.error(f"INTEGRITY FAILURE: {e}")
            return
        except (RuntimeError, EOFError, pickle.UnpicklingError) as e:
            logger.warning(f"Could not load {filepath}: {e}")
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
