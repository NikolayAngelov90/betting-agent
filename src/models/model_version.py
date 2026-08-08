"""The model version identifier stamped on every prediction.

Stage 5, Phase 1. The model is now an experimental subject, frozen and observed
prospectively. That only works if every prediction records which configuration
produced it.

Why this is computed rather than hard-coded
-------------------------------------------
A literal string in a config file drifts. Across Stages 1-4 the blend weight
moved 0.40 -> 0.60 -> 0.80, the Poisson half-life 180 -> 540, rho -0.13 -> 0,
the de-vigging rule changed from single-book to gated cross-book consensus, and
six betting gates were switched off. Every one of those silently changed what
`predicted_probability` means, and nothing in the saved row recorded it — so a
pick from March and one from August were pooled in the same statistics as if
they came from the same system. That is the single biggest reason the Stage 1-4
analyses had to keep re-deriving their own cohorts.

So the version is a **label plus a fingerprint of the values that actually
change predictions**. Change any of them and the fingerprint changes on the next
prediction, without anyone remembering to bump a string.

Format
------
``stage5_baseline_20260807.a3f19c``
  │                │        └── 6-char BLAKE2s digest of the tracked settings
  │                └─────────── the date the baseline was frozen
  └──────────────────────────── the experiment label

The label and freeze date come from config so a future experiment can declare
itself; the fingerprint cannot be faked from config.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()

#: Default experiment label + freeze date. Overridable via
#: models.experiment_label / models.experiment_frozen_at.
DEFAULT_LABEL = "stage5_baseline"
DEFAULT_FROZEN_AT = "20260807"

#: Config keys whose values change what a prediction MEANS. Deliberately a
#: closed list rather than "hash the whole config": scraping league lists,
#: Telegram tokens and logging levels churn constantly and would make every
#: prediction look like a new model.
TRACKED_KEYS: List[str] = [
    "models.bookmaker_blend_weight",
    "models.goals_ml_blend_weight",
    "models.extreme_confidence_ceiling",
    "models.dixon_coles_rho",
    "models.dc_rho_per_league",
    "models.strength_half_life_days",
    "models.shrinkage_sample_cap",
    "models.intl_goals_dampen",
    "models.poisson_use_xg",
    "models.poisson_xg_min_coverage",
    "models.probability_calibration_enabled",
    "models.bayesian_weight_half_life_days",
    "models.bayesian_prior_strength",
    "models.ensemble_weights",
    "betting.min_odds",
    "betting.max_odds",
    "betting.min_expected_value",
    "betting.min_confidence",
    "betting.min_ev_confidence_score",
    "betting.kelly_fraction",
    "betting.max_stake_percentage",
    "betting.excluded_markets",
    "betting.gates",
]

#: Bumped by hand only when the CODE path changes in a way config cannot express
#: — e.g. the Stage 4 switch from single-book de-vigging to gated cross-book
#: consensus. Without this, a pure-code change would leave the fingerprint
#: unmoved and two genuinely different models would share a version.
CODE_REVISION = "s5.1"


def _stable(value: Any) -> Any:
    """Normalise a config value so equal settings always hash equally."""
    if isinstance(value, dict):
        return {str(k): _stable(value[k]) for k in sorted(value)}
    if isinstance(value, (list, tuple, set)):
        items = [_stable(v) for v in value]
        # excluded_markets / gates are order-insensitive sets in meaning.
        try:
            return sorted(items, key=lambda x: json.dumps(x, sort_keys=True))
        except TypeError:
            return items
    if isinstance(value, float) and value.is_integer():
        # 0.8 and 0.80 must hash the same; so must 1 and 1.0.
        return float(value)
    return value


def fingerprint_inputs(config) -> Dict[str, Any]:
    """The exact settings that feed the fingerprint, for logging and debugging."""
    out: Dict[str, Any] = {"__code__": CODE_REVISION}
    for key in TRACKED_KEYS:
        try:
            out[key] = _stable(config.get(key, None))
        except Exception:
            out[key] = None
    return out


def fingerprint(config) -> str:
    """6-char digest of the prediction-affecting configuration."""
    payload = json.dumps(fingerprint_inputs(config), sort_keys=True,
                         separators=(",", ":"), default=str)
    return hashlib.blake2s(payload.encode(), digest_size=3).hexdigest()


def model_version(config) -> str:
    """The identifier to stamp on a prediction.

    Never raises: a prediction must not fail because versioning failed. On error
    it returns a clearly-marked unknown value rather than a plausible-looking
    wrong one, because a wrong version silently pools incomparable cohorts.
    """
    try:
        label = config.get("models.experiment_label", DEFAULT_LABEL) or DEFAULT_LABEL
        frozen = config.get("models.experiment_frozen_at", DEFAULT_FROZEN_AT) or DEFAULT_FROZEN_AT
        return f"{label}_{frozen}.{fingerprint(config)}"
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"model_version could not be computed ({e}) — stamping 'unknown'")
        return "unknown"


def describe(config) -> str:
    """Human-readable breakdown for --stats and the Stage 5 report."""
    lines = [f"model_version = {model_version(config)}", "tracked settings:"]
    for key, value in fingerprint_inputs(config).items():
        lines.append(f"    {key:<45} {value}")
    return "\n".join(lines)
