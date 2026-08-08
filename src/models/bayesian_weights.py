"""Adaptive ensemble weight learner driven by out-of-sample predictive loss.

Design (rewritten 2026-08-07 — see docs/predictive-audit-2026-08-07.md §5 and
docs/stage3-reconstruction-2026-08-07.md).

The previous implementation tracked each model's *classification accuracy* per
league as a Beta(alpha, beta) and set weights to the normalised posterior means.
The 2026-08-07 audit found three defects that made it unable to do its job:

1. ``update()`` wrote to the league and global buckets regardless of ``market``,
   so an over/under observation moved the weights used for 1X2. Poisson was also
   updated twice per settled pick (once for 1X2, once for goals) while Elo was
   updated once, giving Poisson 872 recorded observations to Elo's 436.
2. Weights were ``E[Beta] / sum(E[Beta])``. On a 3-way task every model's accuracy
   sits in a narrow band (measured live: 0.533 / 0.549 / 0.550), so normalising
   compressed every difference away — the live file resolved to 0.33/0.34/0.34.
   The learner was mathematically incapable of preferring Elo over Poisson even
   though Elo wins on log-loss, Brier and accuracy.
3. Nothing prevented the same settled pick being applied twice across runs.

This version fixes all three:

* **Scopes are market-qualified by construction.** State lives under
  ``(market, league)`` and ``(market, __global__)``. There is no market-agnostic
  bucket, so cross-market contamination is impossible rather than merely avoided.
* **The objective is log-loss, not accuracy.** A proper scoring rule is what
  ensemble weights should optimise; argmax-accuracy is blind to calibration and,
  on 1X2, blind to the draw class almost entirely.
* **Weights come from the Hedge / multiplicative-weights rule**
  ``w_m ∝ exp(-eta * decayed_cumulative_loss_m)`` with the textbook learning rate
  ``eta = sqrt(8 ln K / n_eff)``. Because the exponent is
  ``sqrt(8 ln K * n_eff) * mean_loss``, the separation between models grows with
  evidence: uniform at cold start, meaningfully separated once the data supports
  it. That is the property the Beta scheme lacked, and it is a standard algorithm
  with known regret bounds rather than a tuned constant.
* **Updates are idempotent.** Each observation carries a key; re-applying it is a
  no-op. Keys are pruned once they have decayed to irrelevance.

Fallback order for ``get_weights(league, market)``:
    (market, league)  ->  (market, global)  ->  config prior
with shrinkage toward the next level up proportional to effective sample size, so
a league with 3 observations cannot override the global picture.
"""

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger, utcnow

logger = get_logger()

# Models tracked in the weight system.
MODELS = ("poisson", "elo", "ml")

# Scope key for the market-wide (all leagues) bucket. Contains "::" so it can
# never collide with a real league slug.
GLOBAL_SCOPE = "__global__"

# Effective observations at which a scope is trusted 50% against its parent.
# With shrinkage lambda = n_eff / (n_eff + SHRINK_PRIOR), a league needs
# SHRINK_PRIOR observations to carry half its own weight.
SHRINK_PRIOR = 30.0

# Hard bounds on any single model's share. Stops one lucky stretch from
# collapsing the ensemble onto a single model, and guarantees a demoted model
# retains enough weight to demonstrate a recovery.
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.70

# Log-loss is clipped before it is accumulated: a single p~0 forecast would
# otherwise contribute an unbounded loss and permanently sink a model.
MAX_LOSS = 5.0

WEIGHTS_PATH = Path("data/models/ensemble_loss_weights.json")

# Schema marker. The old accuracy/Beta file (bayesian_weights.json) is not
# migrated: its counts were produced by the double-update bug, so carrying them
# forward would import the contamination this rewrite exists to remove.
SCHEMA = 2


def observation_key(pick_id=None, match_id=None, market: str = "",
                    selection: str = "") -> str:
    """Stable identity for one settled observation.

    Prefer the saved-pick primary key; fall back to (match, market, selection)
    so callers replaying history without pick ids still get idempotency.
    """
    if pick_id is not None:
        return f"p{pick_id}"
    return f"m{match_id}|{market}|{selection}"


class EnsembleWeightLearner:
    """Learns per-(market, league) ensemble weights from decayed log-loss."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self._half_life_days = float(self.config.get(
            "models.bayesian_weight_half_life_days", 90))

        ew = self.config.get("models.ensemble_weights", {}) or {}
        prior = {
            "poisson": float(ew.get("poisson", 0.25)),
            "elo": float(ew.get("elo", 0.20)),
            "ml": float(ew.get("xgboost", 0.35)) + float(ew.get("random_forest", 0.20)),
        }
        total = sum(prior.values()) or 1.0
        self._prior = {m: w / total for m, w in prior.items()}

        # scope key -> {model: {"loss": decayed loss sum, "w": decayed weight sum,
        #                       "n": raw count}}
        self._scopes: Dict[str, Dict[str, Dict]] = {}
        # observation key -> ISO date applied (for idempotency + pruning)
        self._seen: Dict[str, str] = {}

        self._load()

    # ------------------------------------------------------------------ scopes

    @staticmethod
    def _scope_key(market: str, league: Optional[str]) -> str:
        """Every scope is market-qualified — cross-market contamination is
        structurally impossible, not merely avoided by a conditional."""
        return f"{market or 'default'}::{league or GLOBAL_SCOPE}"

    def _bucket(self, scope: str, model: str) -> Dict:
        return self._scopes.setdefault(scope, {}).setdefault(
            model, {"loss": 0.0, "w": 0.0, "n": 0})

    # ------------------------------------------------------------------ update

    def update(self, league: str, model: str, loss: float,
               days_ago: int = 0, market: str = "",
               obs_key: Optional[str] = None) -> bool:
        """Record one out-of-sample predictive loss for ``model``.

        Args:
            league: League slug. Falsy values update only the market-global scope.
            model: One of MODELS.
            loss: Negative log-likelihood of the ACTUAL outcome under this model's
                forecast, i.e. ``-ln p(actual)``. Lower is better. Clipped to
                [0, MAX_LOSS].
            days_ago: Age of the observation, for exponential decay.
            market: Market family ("1X2", "goals", "btts", ...). Required for the
                weights to be usable; an empty market lands in the "default" scope.
            obs_key: Identity of the observation. When supplied, applying the same
                (obs_key, market, model) triple twice is a no-op and returns False.

        Returns:
            True when the observation was applied, False when it was a duplicate
            or was rejected as invalid.
        """
        if model not in MODELS:
            return False
        if loss is None or not math.isfinite(loss) or loss < 0:
            return False

        if obs_key is not None:
            dedup = f"{obs_key}|{market}|{model}"
            if dedup in self._seen:
                return False
            self._seen[dedup] = date.today().isoformat()

        loss = min(float(loss), MAX_LOSS)
        if days_ago > 0 and self._half_life_days > 0:
            decay = math.exp(-math.log(2) * days_ago / self._half_life_days)
        else:
            decay = 1.0

        # The observation lands in its market's league scope AND its market's
        # global scope — and nowhere else.
        scopes = [self._scope_key(market, GLOBAL_SCOPE)]
        if league:
            scopes.append(self._scope_key(market, league))
        for scope in scopes:
            b = self._bucket(scope, model)
            b["loss"] += decay * loss
            b["w"] += decay
            b["n"] += 1
        return True

    # ----------------------------------------------------------------- weights

    def _weights_for_scope(self, scope: str) -> Optional[Dict[str, float]]:
        """Hedge weights from decayed mean loss, or None when the scope is empty
        or does not cover every model.

        Requiring all models to be present is deliberate: comparing a model that
        has 50 observations against one that has none would let absence of
        evidence read as evidence of quality.
        """
        data = self._scopes.get(scope)
        if not data:
            return None
        means, n_effs = {}, {}
        for m in MODELS:
            b = data.get(m)
            if not b or b["w"] <= 0:
                return None
            means[m] = b["loss"] / b["w"]
            n_effs[m] = b["w"]

        n_eff = min(n_effs.values())
        if n_eff <= 0:
            return None

        # Hedge / multiplicative weights: w ∝ exp(-eta * cumulative_loss), with
        # the standard eta = sqrt(8 ln K / n). Substituting cumulative = n * mean
        # gives an exponent of sqrt(8 ln K * n) * mean — separation that grows
        # with evidence instead of being fixed by a magic temperature.
        eta_scale = math.sqrt(8.0 * math.log(len(MODELS)) * n_eff)
        best = min(means.values())          # subtract the min for numerical safety
        raw = {m: math.exp(-eta_scale * (means[m] - best)) for m in MODELS}
        return self._normalize(raw)

    def _effective_n(self, scope: str) -> float:
        data = self._scopes.get(scope)
        if not data:
            return 0.0
        return min((b["w"] for b in data.values()), default=0.0)

    def get_weights(self, league: str = None, market: str = "") -> Dict[str, float]:
        """Weights for a (league, market), shrunk toward the market-global weights
        and ultimately toward the config prior.

        Never returns weights derived from a different market.
        """
        global_scope = self._scope_key(market, GLOBAL_SCOPE)
        global_w = self._weights_for_scope(global_scope)
        if global_w is None:
            base = dict(self._prior)
        else:
            # Shrink the market-global estimate toward the config prior too, so a
            # market with a handful of observations does not swing the ensemble.
            lam = self._shrink(self._effective_n(global_scope))
            base = {m: lam * global_w[m] + (1 - lam) * self._prior[m] for m in MODELS}

        if league:
            league_scope = self._scope_key(market, league)
            league_w = self._weights_for_scope(league_scope)
            if league_w is not None:
                lam = self._shrink(self._effective_n(league_scope))
                base = {m: lam * league_w[m] + (1 - lam) * base[m] for m in MODELS}

        return self._clamp(base)

    @staticmethod
    def _shrink(n_eff: float) -> float:
        """Trust in a scope's own estimate: 0 at cold start, ->1 with evidence."""
        return n_eff / (n_eff + SHRINK_PRIOR) if n_eff > 0 else 0.0

    @staticmethod
    def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {m: 1.0 / len(weights) for m in weights}
        return {m: w / total for m, w in weights.items()}

    @classmethod
    def _clamp(cls, weights: Dict[str, float]) -> Dict[str, float]:
        """Project onto {w: sum=1, MIN_WEIGHT <= w_m <= MAX_WEIGHT}.

        Clip-then-renormalise is NOT enough: renormalising after a clip scales the
        clipped entry straight back over the bound (a 0.875 share clipped to 0.70
        renormalises to 0.875 again when the others are at the floor). Instead the
        residual is water-filled across the entries that are still free to move,
        which converges in a couple of passes for three models.
        """
        w = cls._normalize(weights)
        for _ in range(64):
            clipped = {m: min(max(v, MIN_WEIGHT), MAX_WEIGHT) for m, v in w.items()}
            residual = 1.0 - sum(clipped.values())
            if abs(residual) < 1e-12:
                return clipped
            free = [
                m for m, v in clipped.items()
                if (residual < 0 and v > MIN_WEIGHT) or (residual > 0 and v < MAX_WEIGHT)
            ]
            if not free:
                return cls._normalize(clipped)
            share = residual / len(free)
            w = {m: v + (share if m in free else 0.0) for m, v in clipped.items()}
        return cls._normalize(w)

    # ------------------------------------------------------------- diagnostics

    def get_league_summary(self) -> Dict:
        """Per-scope weights, effective sample size and mean loss, for CLI output."""
        summary: Dict[str, Dict] = {}
        for scope in sorted(self._scopes):
            market, _, league = scope.partition("::")
            data = self._scopes[scope]
            summary[scope] = {
                "weights": self.get_weights(
                    None if league == GLOBAL_SCOPE else league, market),
                "n_eff": round(self._effective_n(scope), 1),
                "mean_loss": {
                    m: round(b["loss"] / b["w"], 4)
                    for m, b in data.items() if b["w"] > 0
                },
            }
        return summary

    # ----------------------------------------------------------------- persist

    def _prune_seen(self) -> None:
        """Drop dedup keys older than six half-lives — their observations now
        carry <2% weight, so replaying them could not move any estimate."""
        horizon = self._half_life_days * 6
        if horizon <= 0:
            return
        cutoff = (date.today() - timedelta(days=int(horizon))).isoformat()
        self._seen = {k: v for k, v in self._seen.items() if v >= cutoff}

    def save(self) -> None:
        self._prune_seen()
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "saved_at": utcnow().isoformat(),
            "half_life_days": self._half_life_days,
            "scopes": self._scopes,
            "seen": self._seen,
        }
        tmp = WEIGHTS_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        tmp.replace(WEIGHTS_PATH)
        logger.debug(
            f"Saved ensemble loss weights: {len(self._scopes)} scopes, "
            f"{len(self._seen)} dedup keys"
        )

    def _load(self) -> None:
        if not WEIGHTS_PATH.exists():
            return
        try:
            payload = json.loads(WEIGHTS_PATH.read_text())
            if payload.get("schema") != SCHEMA:
                logger.warning(
                    f"Ignoring {WEIGHTS_PATH}: schema {payload.get('schema')} "
                    f"!= {SCHEMA} — starting from the config prior"
                )
                return
            self._scopes = {
                scope: {
                    m: {"loss": float(b.get("loss", 0.0)),
                        "w": float(b.get("w", 0.0)),
                        "n": int(b.get("n", 0))}
                    for m, b in models.items() if m in MODELS
                }
                for scope, models in payload.get("scopes", {}).items()
            }
            self._seen = dict(payload.get("seen", {}))
            logger.info(
                f"Loaded ensemble loss weights: {len(self._scopes)} scopes, "
                f"{len(self._seen)} dedup keys"
            )
        except Exception as e:
            logger.warning(f"Failed to load {WEIGHTS_PATH}: {e} — using config prior")
            self._scopes = {}
            self._seen = {}


# Backwards-compatible alias. `EnsemblePredictor` and the tuning pipeline import
# this name; the class is no longer Bayesian (it is a decayed-loss Hedge learner),
# but renaming every call site is a separate change.
BayesianWeightLearner = EnsembleWeightLearner
