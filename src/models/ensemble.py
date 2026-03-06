"""Ensemble predictor that combines all prediction models."""

import json
from pathlib import Path
from typing import Dict, Optional

from src.models.poisson_model import PoissonModel
from src.models.elo_system import EloRatingSystem
from src.models.ml_models import MLModels, GoalsMLModel
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
        self.goals_model = GoalsMLModel()

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
        self.goals_model.load()

    def fit(self, league: str = None):
        """Fit all sub-models."""
        logger.info("Fitting ensemble models...")
        self.poisson.fit(league)
        self.elo.fit(league)
        # ML models are fitted separately via train() with feature data
        logger.info("Ensemble models fitted")

    def check_coverage(self, home_team_id: int, away_team_id: int) -> Dict:
        """Check data coverage for a team pair.

        Returns dict with per-model coverage flags and an overall score (0-1).
        """
        home_poisson = home_team_id in self.poisson._team_strengths
        away_poisson = away_team_id in self.poisson._team_strengths
        home_elo = home_team_id in self.elo.ratings
        away_elo = away_team_id in self.elo.ratings

        checks = [home_poisson, away_poisson, home_elo, away_elo]
        score = sum(checks) / len(checks)

        return {
            "home_poisson": home_poisson,
            "away_poisson": away_poisson,
            "home_elo": home_elo,
            "away_elo": away_elo,
            "ml_fitted": self.ml_models.is_fitted,
            "score": score,
        }

    # Leagues where teams come from different domestic leagues,
    # so Poisson league-calibrated strengths are less reliable.
    INTERNATIONAL_LEAGUES = {
        "europe/champions-league", "europe/europa-league", "europe/europa-conference-league",
    }

    def predict(self, home_team_id: int, away_team_id: int,
                features_vector=None, feature_names=None,
                league: str = "") -> Dict:
        """Generate ensemble prediction for a match.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID
            features_vector: Optional numpy array of features for ML models
            feature_names: Optional list of feature names matching features_vector

        Returns:
            Dictionary with ensemble and per-model predictions
        """
        results = {}

        # Data coverage check — penalise confidence when teams lack history
        coverage = self.check_coverage(home_team_id, away_team_id)
        results["coverage"] = coverage

        # Poisson predictions
        poisson_pred = self.poisson.predict(home_team_id, away_team_id, league=league)
        results["poisson"] = poisson_pred

        # Elo predictions
        elo_pred = self.elo.predict(home_team_id, away_team_id)
        results["elo"] = elo_pred

        # ML predictions (if features provided and models fitted)
        ml_pred = None
        if features_vector is not None and self.ml_models.is_fitted:
            ml_predictions = self.ml_models.predict(features_vector, feature_names=feature_names)
            ml_pred = ml_predictions.get("ml_average")
            results["ml"] = ml_predictions

        # Weighted ensemble for 1X2
        is_intl = league in self.INTERNATIONAL_LEAGUES
        ensemble_1x2 = self._weighted_average_1x2(poisson_pred, elo_pred, ml_pred,
                                                    international=is_intl)
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

        # International matches: Poisson goal predictions use league-calibrated
        # strengths that are unreliable when teams come from different leagues.
        # Dampen toward neutral priors (same logic as 1X2 weight adjustment).
        if is_intl:
            intl_dampen = self.config.get("models.intl_goals_dampen", 0.30)
            _priors = {"over_1.5": 0.75, "over_2.5": 0.50, "over_3.5": 0.25}
            for key, prior in _priors.items():
                poisson_goals[key] = poisson_goals[key] * (1 - intl_dampen) + prior * intl_dampen
            poisson_goals["under_2.5"] = 1.0 - poisson_goals["over_2.5"]
            poisson_goals["btts_yes"] = poisson_goals["btts_yes"] * (1 - intl_dampen) + 0.50 * intl_dampen
            poisson_goals["btts_no"] = 1.0 - poisson_goals["btts_yes"]

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

        # Bookmaker blend for goals markets — when real bookmaker over/under / BTTS
        # implied probabilities are available as features, blend them with the
        # Poisson-derived predictions.  The bookmaker encodes information (team news,
        # weather, sharp money) that our model doesn't see.  A 40 % weight is used
        # so Poisson still dominates but the market's signal is respected.
        bk_blend = self.config.get("models.bookmaker_blend_weight", 0.40)
        home_over15 = poisson_pred.get("home_over_1.5", 0)
        away_over15 = poisson_pred.get("away_over_1.5", 0)
        if features_vector is not None and feature_names is not None:
            _fd = dict(zip(feature_names, map(float, features_vector)))

            if _fd.get("goals_bookmaker_available", 0):
                bk_o25 = _fd.get("over25_implied_prob", 0.0)
                if bk_o25 > 0:
                    adjusted_goals["over_2.5"] = round(
                        adjusted_goals["over_2.5"] * (1 - bk_blend) + bk_o25 * bk_blend, 4
                    )
                    adjusted_goals["under_2.5"] = round(1.0 - adjusted_goals["over_2.5"], 4)
                bk_o15 = _fd.get("over15_implied_prob", 0.0)
                if bk_o15 > 0:
                    adjusted_goals["over_1.5"] = round(
                        adjusted_goals["over_1.5"] * (1 - bk_blend) + bk_o15 * bk_blend, 4
                    )

            if _fd.get("btts_bookmaker_available", 0):
                bk_btts = _fd.get("btts_yes_implied_prob", 0.0)
                if bk_btts > 0:
                    adjusted_goals["btts_yes"] = round(
                        adjusted_goals["btts_yes"] * (1 - bk_blend) + bk_btts * bk_blend, 4
                    )
                    adjusted_goals["btts_no"] = round(1.0 - adjusted_goals["btts_yes"], 4)

            if _fd.get("team_goals_bookmaker_available", 0):
                bk_h15 = _fd.get("home_over15_implied_prob", 0.0)
                if bk_h15 > 0:
                    home_over15 = round(
                        home_over15 * (1 - bk_blend) + bk_h15 * bk_blend, 4
                    )
                bk_a15 = _fd.get("away_over15_implied_prob", 0.0)
                if bk_a15 > 0:
                    away_over15 = round(
                        away_over15 * (1 - bk_blend) + bk_a15 * bk_blend, 4
                    )

        # Goals ML model blend — when the dedicated over/under classifier is
        # fitted, blend its P(over 2.5) with the current Poisson+bookmaker estimate.
        # A 25% weight keeps Poisson dominant while respecting the ML signal.
        if features_vector is not None and self.goals_model.is_fitted:
            try:
                ml_o25 = self.goals_model.predict_proba_over25(
                    features_vector, feature_names=feature_names
                )
                results["goals_ml_over25"] = ml_o25  # store for model agreement check
                goals_ml_w = self.config.get("models.goals_ml_blend_weight", 0.25)
                adjusted_goals["over_2.5"] = round(
                    adjusted_goals["over_2.5"] * (1 - goals_ml_w) + ml_o25 * goals_ml_w, 4
                )
                adjusted_goals["under_2.5"] = round(1.0 - adjusted_goals["over_2.5"], 4)
            except Exception as _e:
                logger.debug(f"GoalsMLModel blend skipped: {_e}")

        # Bookmaker 1X2 blend — same blend weight as goals markets.
        # The 1X2 ensemble above only uses Poisson + Elo + ML; blending the
        # bookmaker's implied 1X2 probabilities here keeps the output consistent.
        if features_vector is not None and feature_names is not None:
            _fd = dict(zip(feature_names, map(float, features_vector)))
            if _fd.get("bookmaker_available", 0):
                bk_1x2_w = self.config.get("models.bookmaker_blend_weight", 0.40)
                bk_h = _fd.get("home_implied_prob", 0.0)
                bk_d = _fd.get("draw_implied_prob", 0.0)
                bk_a = _fd.get("away_implied_prob", 0.0)
                if bk_h > 0 and bk_d > 0 and bk_a > 0:
                    ensemble_1x2["home_win"] = round(
                        ensemble_1x2["home_win"] * (1 - bk_1x2_w) + bk_h * bk_1x2_w, 4
                    )
                    ensemble_1x2["draw"] = round(
                        ensemble_1x2["draw"] * (1 - bk_1x2_w) + bk_d * bk_1x2_w, 4
                    )
                    ensemble_1x2["away_win"] = round(
                        ensemble_1x2["away_win"] * (1 - bk_1x2_w) + bk_a * bk_1x2_w, 4
                    )
                    # Renormalize to sum to 1.0
                    _tot = ensemble_1x2["home_win"] + ensemble_1x2["draw"] + ensemble_1x2["away_win"]
                    if _tot > 0:
                        ensemble_1x2["home_win"] = round(ensemble_1x2["home_win"] / _tot, 4)
                        ensemble_1x2["draw"] = round(ensemble_1x2["draw"] / _tot, 4)
                        ensemble_1x2["away_win"] = round(ensemble_1x2["away_win"] / _tot, 4)

        # Extreme-confidence dampening — shrink any prediction above the ceiling
        # toward the ceiling to prevent the model from over-committing on sparse data.
        # Excess above ceiling is retained at 30% to preserve signal direction.
        _ceiling = self.config.get("models.extreme_confidence_ceiling", 0.90)
        if _ceiling < 1.0:
            for _key, _comp in [
                ("over_2.5", "under_2.5"), ("over_1.5", None),
                ("over_3.5", None), ("btts_yes", "btts_no"),
            ]:
                _v = adjusted_goals.get(_key, 0)
                if _v > _ceiling:
                    adjusted_goals[_key] = round(_ceiling + (_v - _ceiling) * 0.30, 4)
                    if _comp:
                        adjusted_goals[_comp] = round(1.0 - adjusted_goals[_key], 4)
            # Dampen 1X2 probabilities, renormalize, then cap again.
            # Two passes prevent renormalization from pushing a dampened value
            # back above the ceiling.
            for _pass in range(2):
                for _key in ("home_win", "draw", "away_win"):
                    _v = ensemble_1x2.get(_key, 0)
                    if _v > _ceiling:
                        ensemble_1x2[_key] = round(_ceiling + (_v - _ceiling) * 0.30, 4)
                _tot1x2 = sum(ensemble_1x2.get(k, 0) for k in ("home_win", "draw", "away_win"))
                if _tot1x2 > 0:
                    ensemble_1x2["home_win"] = round(ensemble_1x2["home_win"] / _tot1x2, 4)
                    ensemble_1x2["draw"] = round(ensemble_1x2["draw"] / _tot1x2, 4)
                    ensemble_1x2["away_win"] = round(ensemble_1x2["away_win"] / _tot1x2, 4)

        results["ensemble"]["home_xg"] = poisson_pred.get("home_xg", 0)
        results["ensemble"]["away_xg"] = poisson_pred.get("away_xg", 0)
        results["ensemble"]["over_1.5"] = round(adjusted_goals["over_1.5"], 4)
        results["ensemble"]["over_2.5"] = round(adjusted_goals["over_2.5"], 4)
        results["ensemble"]["over_3.5"] = round(adjusted_goals["over_3.5"], 4)
        results["ensemble"]["under_1.5"] = round(1.0 - adjusted_goals["over_1.5"], 4)
        results["ensemble"]["under_2.5"] = round(adjusted_goals["under_2.5"], 4)
        results["ensemble"]["under_3.5"] = round(1.0 - adjusted_goals["over_3.5"], 4)
        results["ensemble"]["btts_yes"] = round(adjusted_goals["btts_yes"], 4)
        results["ensemble"]["btts_no"] = round(adjusted_goals["btts_no"], 4)
        # Team goal lines: blended with bookmaker when available
        results["ensemble"]["home_over_1.5"] = home_over15
        results["ensemble"]["away_over_1.5"] = away_over15
        results["ensemble"]["most_likely_score"] = poisson_pred.get("most_likely_score", "")
        results["ensemble"]["model"] = "ensemble"

        # Apply confidence penalty for low-coverage matches:
        # Pull probabilities toward their priors proportional to missing data.
        penalty = (1.0 - coverage["score"]) * 0.25  # max 25% regression
        if penalty > 0:
            # 1X2: regress toward uniform 0.33
            prior_1x2 = 1.0 / 3.0
            for key in ("home_win", "draw", "away_win"):
                results["ensemble"][key] = round(
                    results["ensemble"][key] * (1 - penalty) + prior_1x2 * penalty, 4
                )
            # Goals markets: regress toward neutral priors
            _goals_priors = {
                "over_1.5": 0.75, "under_1.5": 0.25,
                "over_2.5": 0.50, "under_2.5": 0.50,
                "over_3.5": 0.25, "under_3.5": 0.75,
                "btts_yes": 0.50, "btts_no": 0.50,
                "home_over_1.5": 0.40, "away_over_1.5": 0.30,
            }
            for key, prior in _goals_priors.items():
                if key in results["ensemble"]:
                    results["ensemble"][key] = round(
                        results["ensemble"][key] * (1 - penalty) + prior * penalty, 4
                    )

        return results

    def _weighted_average_1x2(self, poisson: Dict, elo: Dict,
                               ml: Optional[Dict],
                               international: bool = False) -> Dict:
        """Compute weighted average of 1X2 probabilities across models.

        For international matches (CL/EL/ECL), Poisson weight is halved and
        redistributed to Elo because Poisson's league-calibrated attack/defence
        strengths are misleading when teams come from different domestic leagues.
        """
        total_weight = 0.0
        home_win = 0.0
        draw = 0.0
        away_win = 0.0

        # Poisson — downweight for international matches
        w = self.weights.get("poisson", 0.25)
        if international:
            w *= 0.5  # halve Poisson influence
        home_win += w * poisson.get("home_win", 0.33)
        draw += w * poisson.get("draw", 0.33)
        away_win += w * poisson.get("away_win", 0.33)
        total_weight += w

        # Elo — upweight for international matches (captures cross-league quality)
        w = self.weights.get("elo", 0.20)
        if international:
            w *= 1.5  # boost Elo influence
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
