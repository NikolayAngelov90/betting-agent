"""Hierarchical Bayesian ensemble weight learner.

Tracks per-model prediction accuracy per league and computes adaptive
ensemble weights using Bayesian updating with conjugate Beta priors.

Architecture:
- Global weights serve as the prior (from config ensemble_weights).
- Per-league weights are posteriors updated after each settled pick.
- New/low-data leagues inherit the global prior until enough evidence
  accumulates to deviate.
- Temporal decay ensures recent performance matters more than old results.
"""

import json
import math
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()

# Models tracked in the Bayesian weight system
MODELS = ("poisson", "elo", "ml")

# Minimum observations before league-specific weights deviate from global prior
MIN_LEAGUE_OBS = 15

# Path to persist learned weights between sessions
WEIGHTS_PATH = Path("data/models/bayesian_weights.json")


class BayesianWeightLearner:
    """Learns ensemble weights per league using Bayesian updating.

    Each model's "skill" per league is tracked as a Beta distribution:
        Beta(alpha, beta) where alpha = successes + prior_alpha,
                                beta = failures + prior_beta.

    The expected accuracy is alpha / (alpha + beta), which is used to
    derive relative ensemble weights.

    Temporal decay: older observations are down-weighted by decaying
    alpha/beta toward the prior over time.
    """

    def __init__(self, config=None):
        self.config = config or get_config()
        self._half_life_days = self.config.get(
            "models.bayesian_weight_half_life_days", 90
        )

        # Global prior weights from config (used as the baseline)
        ew = self.config.get("models.ensemble_weights", {})
        self._global_prior = {
            "poisson": ew.get("poisson", 0.25),
            "elo": ew.get("elo", 0.20),
            "ml": ew.get("xgboost", 0.35) + ew.get("random_forest", 0.20),
        }

        # Prior strength: how many pseudo-observations the prior represents.
        # Higher = more data needed before league weights diverge from global.
        self._prior_strength = self.config.get(
            "models.bayesian_prior_strength", 10
        )

        # Per-league Beta parameters: {league: {model: {"alpha": float, "beta": float, "n": int}}}
        self._league_params: Dict[str, Dict[str, Dict]] = {}

        # Global Beta parameters (aggregated across all leagues)
        self._global_params: Dict[str, Dict] = {}

        self._load()

    def _default_params(self, model: str) -> Dict:
        """Create default Beta parameters from the global prior."""
        # Convert prior weight to Beta parameters:
        # If prior weight for a model is 0.35, that means we expect it to be
        # "right" 35% of the time relative to other models.
        # Use prior_strength as the effective sample size.
        w = self._global_prior.get(model, 0.25)
        alpha = w * self._prior_strength
        beta = (1 - w) * self._prior_strength
        return {"alpha": alpha, "beta": beta, "n": 0}

    def update(self, league: str, model: str, correct: bool, days_ago: int = 0,
               market: str = ""):
        """Record a prediction outcome for a model in a league.

        Args:
            league: League identifier (e.g., "england/premier-league")
            model: Model name ("poisson", "elo", or "ml")
            correct: Whether the model's top prediction was correct
            days_ago: How many days ago this observation occurred (for decay)
            market: Optional market type ("1X2", "goals", "btts") for per-market weights
        """
        if model not in MODELS:
            return

        # Temporal decay weight for this observation
        if days_ago > 0 and self._half_life_days > 0:
            decay = math.exp(-math.log(2) * days_ago / self._half_life_days)
        else:
            decay = 1.0

        # Update league-specific parameters
        if league not in self._league_params:
            self._league_params[league] = {
                m: self._default_params(m) for m in MODELS
            }

        params = self._league_params[league][model]
        if correct:
            params["alpha"] += decay
        else:
            params["beta"] += decay
        params["n"] += 1

        # Update global parameters
        if model not in self._global_params:
            self._global_params[model] = self._default_params(model)
        gp = self._global_params[model]
        if correct:
            gp["alpha"] += decay
        else:
            gp["beta"] += decay
        gp["n"] += 1

        # Per-market tracking (league+market combo)
        if market:
            mkt_key = f"{league}::{market}"
            if mkt_key not in self._league_params:
                self._league_params[mkt_key] = {
                    m: self._default_params(m) for m in MODELS
                }
            mp = self._league_params[mkt_key][model]
            if correct:
                mp["alpha"] += decay
            else:
                mp["beta"] += decay
            mp["n"] += 1

    def get_weights(self, league: str = None, market: str = "") -> Dict[str, float]:
        """Get ensemble weights, optionally specialized for a league and market.

        Returns normalized weights that sum to 1.0 across models.
        For leagues with insufficient data (< MIN_LEAGUE_OBS), returns
        a blend of league-specific and global weights proportional to
        the amount of league data available.

        Args:
            league: Optional league to get specialized weights for.
                    None returns global weights.
            market: Optional market type ("1X2", "goals", "btts") for
                    per-market specialization.

        Returns:
            Dict mapping model names to weights (sum = 1.0)
        """
        # Try per-market weights first (most specific)
        if league and market:
            mkt_key = f"{league}::{market}"
            if mkt_key in self._league_params:
                mkt_data = self._league_params[mkt_key]
                total_obs = sum(p["n"] for p in mkt_data.values())
                if total_obs >= MIN_LEAGUE_OBS:
                    # Blend market-specific with league-level weights
                    blend = min(1.0, (total_obs - MIN_LEAGUE_OBS) / (2 * MIN_LEAGUE_OBS))
                    mkt_weights = self._params_to_weights(mkt_data)
                    league_weights = self.get_weights(league)  # recurse for league-level
                    weights = {}
                    for m in MODELS:
                        weights[m] = mkt_weights[m] * blend + league_weights[m] * (1 - blend)
                    return self._normalize(weights)

        if league and league in self._league_params:
            league_data = self._league_params[league]
            total_obs = sum(p["n"] for p in league_data.values())

            if total_obs >= MIN_LEAGUE_OBS:
                # Enough data: use league-specific weights
                # Blend factor: how much to trust league-specific vs global
                # Sigmoid transition: at MIN_LEAGUE_OBS we start using league weights,
                # at 3x MIN_LEAGUE_OBS we fully trust them.
                blend = min(1.0, (total_obs - MIN_LEAGUE_OBS) / (2 * MIN_LEAGUE_OBS))
                league_weights = self._params_to_weights(league_data)
                global_weights = self._get_global_weights()
                weights = {}
                for m in MODELS:
                    weights[m] = (
                        league_weights[m] * blend
                        + global_weights[m] * (1 - blend)
                    )
                return self._normalize(weights)

        # Fall back to global weights
        return self._get_global_weights()

    def _get_global_weights(self) -> Dict[str, float]:
        """Get global weights from aggregated observations."""
        if not self._global_params:
            return dict(self._global_prior)
        return self._params_to_weights(
            {m: self._global_params.get(m, self._default_params(m)) for m in MODELS}
        )

    def _params_to_weights(self, params: Dict[str, Dict]) -> Dict[str, float]:
        """Convert Beta parameters to normalized weights."""
        weights = {}
        for model in MODELS:
            p = params.get(model, self._default_params(model))
            # Expected value of Beta distribution = alpha / (alpha + beta)
            expected = p["alpha"] / (p["alpha"] + p["beta"]) if (p["alpha"] + p["beta"]) > 0 else 0.25
            weights[model] = expected
        return self._normalize(weights)

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total <= 0:
            n = len(weights)
            return {m: 1.0 / n for m in weights}
        return {m: w / total for m, w in weights.items()}

    def get_league_summary(self) -> Dict:
        """Get a summary of learned weights per league for diagnostics."""
        summary = {"global": self._get_global_weights()}
        for league in sorted(self._league_params.keys()):
            if "::" in league:
                continue  # skip per-market keys in summary
            total_obs = sum(p["n"] for p in self._league_params[league].values())
            summary[league] = {
                "weights": self.get_weights(league),
                "observations": total_obs,
            }
        return summary

    def save(self):
        """Persist learned parameters to disk."""
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "global": {m: dict(p) for m, p in self._global_params.items()},
            "leagues": {
                lg: {m: dict(p) for m, p in models.items()}
                for lg, models in self._league_params.items()
            },
        }
        WEIGHTS_PATH.write_text(json.dumps(data, indent=2))
        logger.debug(f"Saved Bayesian weights to {WEIGHTS_PATH}")

    def _load(self):
        """Load previously saved parameters from disk."""
        if not WEIGHTS_PATH.exists():
            return
        try:
            data = json.loads(WEIGHTS_PATH.read_text())
            self._global_params = {
                m: dict(p) for m, p in data.get("global", {}).items()
                if m in MODELS
            }
            self._league_params = {}
            for lg, models in data.get("leagues", {}).items():
                self._league_params[lg] = {
                    m: dict(p) for m, p in models.items()
                    if m in MODELS
                }
            total_obs = sum(
                p["n"] for params in self._league_params.values()
                for p in params.values()
            )
            logger.info(
                f"Loaded Bayesian weights: {len(self._league_params)} leagues, "
                f"{total_obs} total observations"
            )
        except Exception as e:
            logger.warning(f"Failed to load Bayesian weights: {e}")
