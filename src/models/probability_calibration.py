"""Empirical probability calibration from settled picks.

The ensemble is systematically overconfident (871 settled picks, 2026-07-16:
every predicted-probability bucket lands 6-15pp below its prediction — predicted
47% won 32%, predicted 75% won 61%). This module fits a monotonic (isotonic)
mapping p_cal = f(p_raw) on (predicted_probability, won) pairs from saved_picks
and applies it to ensemble probabilities, so EV / confidence gates and Kelly
stakes operate on numbers that mean what they say.

Design (docs/calibration-plan.md):
- isotonic regression per market FAMILY (1x2 / goals / btts / team_goals) when
  the family has enough settled data, else a single global mapping, else
  identity (unfitted = no-op — the pipeline degrades gracefully).
- fitted in learn_from_settled() after every settlement, persisted to
  data/models/probability_calibration.json (same lifecycle as the Bayesian
  weights / per-model calibration factors).
- honest caveat: the training pairs are the CHOSEN selections only (selection
  bias) — accepted in the plan; the mapping corrects the miscalibration of the
  probabilities we actually bet on, which is the quantity that matters.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()

DEFAULT_PATH = Path("data/models/probability_calibration.json")

# Selection string (as stored on SavedPick) → market family.
SELECTION_FAMILY = {
    "Home Win": "1x2", "Draw": "1x2", "Away Win": "1x2",
    "Over 1.5 Goals": "goals", "Over 2.5 Goals": "goals", "Over 3.5 Goals": "goals",
    "Under 1.5 Goals": "goals", "Under 2.5 Goals": "goals", "Under 3.5 Goals": "goals",
    "BTTS Yes": "btts", "BTTS No": "btts",
    "Home Over 1.5": "team_goals", "Away Over 1.5": "team_goals",
}

# Ensemble dict key → market family (for application time).
def key_family(market_key: str) -> Optional[str]:
    if market_key in ("home_win", "draw", "away_win"):
        return "1x2"
    if market_key.startswith("over_") or market_key.startswith("under_"):
        return "goals"
    if market_key.startswith("btts_"):
        return "btts"
    if market_key in ("home_over_1.5", "away_over_1.5"):
        return "team_goals"
    return None


class ProbabilityCalibrator:
    """Piecewise-linear monotonic calibration maps, per family + global."""

    MIN_FAMILY = 120   # settled picks needed for a family-specific map
    MIN_GLOBAL = 300   # settled picks needed for the global fallback map
    CLIP_LO, CLIP_HI = 0.02, 0.98

    def __init__(self):
        # family -> (x_thresholds, y_thresholds) as lists; applied via np.interp
        self.maps: Dict[str, Tuple[list, list]] = {}
        self.fitted_at: Optional[str] = None
        self.n_samples: Dict[str, int] = {}

    # ------------------------------------------------------------------ fit
    def fit(self, pairs: List[Tuple[str, float, int]]) -> bool:
        """Fit from (family, predicted_probability, won 0/1) tuples.

        Returns True when at least the global map could be fitted.
        """
        try:
            from sklearn.isotonic import IsotonicRegression
        except ImportError:
            logger.warning("Probability calibration skipped: sklearn missing")
            return False

        pairs = [(f, float(p), int(w)) for f, p, w in pairs
                 if f and p is not None and 0.0 < float(p) < 1.0]
        if len(pairs) < self.MIN_GLOBAL:
            logger.info(
                f"Probability calibration not fitted: {len(pairs)} settled "
                f"picks < {self.MIN_GLOBAL} minimum"
            )
            return False

        def _fit_map(subset):
            xs = np.array([p for _, p, _ in subset])
            ys = np.array([w for _, _, w in subset])
            iso = IsotonicRegression(
                y_min=self.CLIP_LO, y_max=self.CLIP_HI, out_of_bounds="clip"
            )
            iso.fit(xs, ys)
            # Persist the learned step function's breakpoints.
            return list(map(float, iso.X_thresholds_)), list(map(float, iso.y_thresholds_))

        maps: Dict[str, Tuple[list, list]] = {}
        n: Dict[str, int] = {}
        maps["global"] = _fit_map(pairs)
        n["global"] = len(pairs)
        for family in sorted({f for f, _, _ in pairs}):
            subset = [t for t in pairs if t[0] == family]
            if len(subset) >= self.MIN_FAMILY:
                maps[family] = _fit_map(subset)
                n[family] = len(subset)

        self.maps = maps
        self.n_samples = n
        self.fitted_at = datetime.utcnow().isoformat()
        logger.info(
            "Probability calibration fitted: "
            + ", ".join(f"{k}(n={n[k]})" for k in maps)
        )
        return True

    def fit_from_db(self, db) -> bool:
        """Fit from all settled picks in the database."""
        from src.data.models import SavedPick
        with db.get_session() as session:
            rows = session.query(
                SavedPick.selection, SavedPick.predicted_probability, SavedPick.result
            ).filter(SavedPick.result.in_(["win", "loss"])).all()
        pairs = [
            (SELECTION_FAMILY.get(sel), prob, 1 if res == "win" else 0)
            for sel, prob, res in rows
            if SELECTION_FAMILY.get(sel) and prob
        ]
        return self.fit(pairs)

    # ---------------------------------------------------------------- apply
    @property
    def is_fitted(self) -> bool:
        return bool(self.maps)

    def apply(self, market_key: str, p: float) -> float:
        """Calibrate one probability; identity when unfitted/unknown key."""
        if not self.maps or p is None:
            return p
        family = key_family(market_key)
        m = self.maps.get(family) or self.maps.get("global")
        if not m:
            return p
        xs, ys = m
        cal = float(np.interp(p, xs, ys))
        return min(max(cal, self.CLIP_LO), self.CLIP_HI)

    # -------------------------------------------------------------- persist
    def save(self, path: Path = DEFAULT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "fitted_at": self.fitted_at,
            "n_samples": self.n_samples,
            "maps": {k: {"x": xs, "y": ys} for k, (xs, ys) in self.maps.items()},
        }
        path.write_text(json.dumps(payload))
        logger.info(f"Probability calibration saved to {path}")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "ProbabilityCalibrator":
        """Load a persisted calibrator; returns an UNFITTED (identity) instance
        when the file is missing or unreadable."""
        cal = cls()
        try:
            path = Path(path)
            if not path.exists():
                return cal
            payload = json.loads(path.read_text())
            cal.maps = {
                k: (list(map(float, v["x"])), list(map(float, v["y"])))
                for k, v in payload.get("maps", {}).items()
                if v.get("x") and v.get("y")
            }
            cal.fitted_at = payload.get("fitted_at")
            cal.n_samples = payload.get("n_samples", {})
        except Exception as e:
            logger.warning(f"Probability calibration load failed ({e}) — identity")
            cal.maps = {}
        return cal
