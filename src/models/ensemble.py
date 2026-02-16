"""Ensemble predictor that combines all prediction models."""

import json
from pathlib import Path
from typing import Dict, Optional

from src.models.poisson_model import PoissonModel
from src.models.elo_system import EloRatingSystem
from src.models.ml_models import MLModels
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


class EnsemblePredictor:
    """Combines predictions from Poisson, Elo, and ML models using weighted averaging.

    Weights are configured in config.yaml under models.ensemble_weights.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self.poisson = PoissonModel()
        self.elo = EloRatingSystem()
        self.ml_models = MLModels()

        weights = self.config.get("models.ensemble_weights", {})
        self.weights = {
            "poisson": weights.get("poisson", 0.25),
            "elo": weights.get("elo", 0.20),
            "xgboost": weights.get("xgboost", 0.35),
            "random_forest": weights.get("random_forest", 0.20),
        }

        # Load tuned weights if available (overrides config)
        tuned_path = Path("data/models/ensemble_weights.json")
        if tuned_path.exists():
            try:
                tuned = json.loads(tuned_path.read_text())
                self.weights.update(tuned)
                logger.info(f"Loaded tuned ensemble weights: {self.weights}")
            except Exception as e:
                logger.warning(f"Failed to load tuned weights: {e}")

        # Try to load previously trained ML models from disk
        self.ml_models.load()

    def fit(self, league: str = None):
        """Fit all sub-models."""
        logger.info("Fitting ensemble models...")
        self.poisson.fit(league)
        self.elo.fit(league)
        # ML models are fitted separately via train() with feature data
        logger.info("Ensemble models fitted")

    def predict(self, home_team_id: int, away_team_id: int,
                features_vector=None) -> Dict:
        """Generate ensemble prediction for a match.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID
            features_vector: Optional numpy array of features for ML models

        Returns:
            Dictionary with ensemble and per-model predictions
        """
        results = {}

        # Poisson predictions
        poisson_pred = self.poisson.predict(home_team_id, away_team_id)
        results["poisson"] = poisson_pred

        # Elo predictions
        elo_pred = self.elo.predict(home_team_id, away_team_id)
        results["elo"] = elo_pred

        # ML predictions (if features provided and models fitted)
        ml_pred = None
        if features_vector is not None and self.ml_models.is_fitted:
            ml_predictions = self.ml_models.predict(features_vector)
            ml_pred = ml_predictions.get("ml_average")
            results["ml"] = ml_predictions

        # Weighted ensemble for 1X2
        ensemble_1x2 = self._weighted_average_1x2(poisson_pred, elo_pred, ml_pred)
        results["ensemble"] = ensemble_1x2

        # Blend goals/BTTS predictions using both Poisson and ensemble 1X2
        # Poisson provides the base; then we adjust using ensemble 1X2 confidence
        # to differentiate unknown teams whose Poisson predictions are generic.
        poisson_goals = {
            "over_1.5": poisson_pred.get("over_1.5", 0),
            "over_2.5": poisson_pred.get("over_2.5", 0),
            "over_3.5": poisson_pred.get("over_3.5", 0),
            "under_2.5": poisson_pred.get("under_2.5", 0),
            "btts_yes": poisson_pred.get("btts_yes", 0),
            "btts_no": poisson_pred.get("btts_no", 0),
        }

        # Derive goal market adjustments from ensemble 1X2 confidence:
        # - Strong favourite (high home/away win prob) -> more goals expected
        # - High draw prob -> fewer goals, but BTTS more likely
        ens_1x2 = ensemble_1x2
        max_win_prob = max(ens_1x2.get("home_win", 0.33), ens_1x2.get("away_win", 0.33))
        draw_prob = ens_1x2.get("draw", 0.33)
        decisiveness = max_win_prob - 0.40  # positive when there's a clear favourite

        # Adjust over/under: decisive matches tend to have more goals
        goal_boost = decisiveness * 0.15  # up to ~6% boost for 80% favourite
        draw_penalty = (draw_prob - 0.25) * 0.20  # draws reduce goal expectation

        adjusted_goals = {}
        for key in ["over_1.5", "over_2.5", "over_3.5"]:
            base = poisson_goals[key]
            adjusted = base + goal_boost - draw_penalty
            adjusted_goals[key] = max(0.05, min(0.98, adjusted))

        adjusted_goals["under_2.5"] = 1.0 - adjusted_goals["over_2.5"]

        # BTTS: boosted when both teams are competitive (neither dominates)
        competitiveness = 1.0 - abs(ens_1x2.get("home_win", 0.33) - ens_1x2.get("away_win", 0.33))
        btts_boost = (competitiveness - 0.50) * 0.10
        adjusted_goals["btts_yes"] = max(0.05, min(0.95,
            poisson_goals["btts_yes"] + btts_boost))
        adjusted_goals["btts_no"] = 1.0 - adjusted_goals["btts_yes"]

        results["ensemble"]["home_xg"] = poisson_pred.get("home_xg", 0)
        results["ensemble"]["away_xg"] = poisson_pred.get("away_xg", 0)
        results["ensemble"]["over_1.5"] = round(adjusted_goals["over_1.5"], 4)
        results["ensemble"]["over_2.5"] = round(adjusted_goals["over_2.5"], 4)
        results["ensemble"]["over_3.5"] = round(adjusted_goals["over_3.5"], 4)
        results["ensemble"]["under_2.5"] = round(adjusted_goals["under_2.5"], 4)
        results["ensemble"]["btts_yes"] = round(adjusted_goals["btts_yes"], 4)
        results["ensemble"]["btts_no"] = round(adjusted_goals["btts_no"], 4)
        results["ensemble"]["most_likely_score"] = poisson_pred.get("most_likely_score", "")
        results["ensemble"]["model"] = "ensemble"

        return results

    def _weighted_average_1x2(self, poisson: Dict, elo: Dict,
                               ml: Optional[Dict]) -> Dict:
        """Compute weighted average of 1X2 probabilities across models."""
        total_weight = 0.0
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        # Poisson
        w = self.weights.get("poisson", 0.25)
        home_win += w * poisson.get("home_win", 0.33)
        draw += w * poisson.get("draw", 0.33)
        away_win += w * poisson.get("away_win", 0.33)
        total_weight += w

        # Elo
        w = self.weights.get("elo", 0.20)
        home_win += w * elo.get("home_win", 0.33)
        draw += w * elo.get("draw", 0.33)
        away_win += w * elo.get("away_win", 0.33)
        total_weight += w

        # ML (use combined xgboost + random_forest weight)
        if ml:
            w = self.weights.get("xgboost", 0.35) + self.weights.get("random_forest", 0.20)
            home_win += w * ml.get("home_win", 0.33)
            draw += w * ml.get("draw", 0.33)
            away_win += w * ml.get("away_win", 0.33)
            total_weight += w

        # Normalize
        if total_weight > 0:
            home_win /= total_weight
            draw /= total_weight
            away_win /= total_weight

        return {
            "home_win": round(home_win, 4),
            "draw": round(draw, 4),
            "away_win": round(away_win, 4),
        }
