"""Machine learning prediction models."""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger()

MODELS_DIR = Path("data/models")


class MLModels:
    """Manages ML models for match outcome prediction.

    Includes XGBoost, Random Forest, and Logistic Regression classifiers.
    Targets: 0 = away_win, 1 = draw, 2 = home_win
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
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

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str] = None):
        """Train all models on the provided data.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (0=away, 1=draw, 2=home)
            feature_names: Optional list of feature names
        """
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

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            cv_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model.fit(X_train, y_train)
                val_pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, val_pred))

            # Final fit on all data
            model.fit(X_scaled, y)

            avg_cv = np.mean(cv_scores)
            logger.info(f"{name} CV accuracy: {avg_cv:.4f}")

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

        predictions = {}
        all_probs = []

        for name, model in self.models.items():
            probs = model.predict_proba(X_scaled)[0]

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
        """Save all models to disk."""
        save_dir = Path(path) if path else MODELS_DIR
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "models": self.models,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted,
            "_kept_feature_mask": getattr(self, "_kept_feature_mask", None),
            "_corr_drop_mask": getattr(self, "_corr_drop_mask", None),
        }

        filepath = save_dir / "ml_models.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

        logger.info(f"Models saved to {filepath}")

    def load(self, path: str = None):
        """Load models from disk."""
        load_dir = Path(path) if path else MODELS_DIR
        filepath = load_dir / "ml_models.pkl"

        if not filepath.exists():
            logger.warning(f"No saved models found at {filepath}")
            return

        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.models = state["models"]
        self.scaler = state["scaler"]
        self.feature_names = state["feature_names"]
        self.is_fitted = state["is_fitted"]
        self._kept_feature_mask = state.get("_kept_feature_mask")
        self._corr_drop_mask = state.get("_corr_drop_mask")

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
