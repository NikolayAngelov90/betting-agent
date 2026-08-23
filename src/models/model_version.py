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
    # Stage 13 Part C: one pick per match. Changing this changes
    # which picks exist, so it must split the cohort.
    "betting.max_picks_per_match",
    "betting.gates",
]

#: Bumped by hand only when the CODE path changes in a way config cannot express
#: — e.g. the Stage 4 switch from single-book de-vigging to gated cross-book
#: consensus. Without this, a pure-code change would leave the fingerprint
#: unmoved and two genuinely different models would share a version.
#:
#: History
#: -------
#: s5.1  Stage 5 freeze.
#: s5.2  Stage 8 (2026-08-10). SELECTION-affecting, not prediction-affecting:
#:       the model's probabilities are untouched, but the set of picks it
#:       persists changed, and a changed population of predictions is a
#:       different experiment. Three edits:
#:         1. `_CORRELATED_PAIRS` gained the six Over X.5 / Under Y.5 cross
#:            pairs — the table had every same-direction pair and no opposite
#:            ones, so that whole class passed the filter.
#:         2. The Claude KEEP/CHANGE review now re-checks correlation before
#:            switching a selection. It ran after `_filter_correlated_picks`
#:            and only guarded exact-selection duplicates, so a switch could
#:            land on a selection correlated with one already held — which is
#:            how all three correlated pairs in production were created.
#:         3. The in-memory duplicate key moved from (match_name, selection)
#:            to (match_id, market, selection), matching the DB unique index.
#: s5.3  Stage 13 (2026-08-23). SELECTION-affecting AND a training-data
#:       correction — the second of which this fingerprint does NOT cover, so
#:       read this entry before comparing anything across the boundary.
#:
#:       Config change (covered by the fingerprint):
#:         · `betting.max_picks_per_match: 1` — at most one pick per fixture,
#:           and it must be the best one. Three different orderings existed:
#:           picks were sorted by `_rank_key` (EV x confidence x agreement x
#:           contrarian), the per-match survivor was then chosen by CONFIDENCE
#:           ALONE, and the final order dropped the contrarian term. At a cap
#:           of 2 that was survivable; at a cap of 1 it decides which single
#:           pick represents the match. All three now use `_rank_key`.
#:
#:       Selection-affecting, not in the fingerprint's inputs:
#:         · Team-identity gate at API-Football team resolution. A row matched
#:           by AF id is now verified against the payload in hand — country
#:           first (unconditional: a club plays in exactly one domestic
#:           league), then a lexical-anchor name check. Fails closed: the
#:           fixture is skipped, the suspect row is neither renamed nor
#:           re-keyed. Correct by construction and UNVERIFIED IN PRODUCTION
#:           while the API-Football account is suspended (ledger OPS-1).
#:         · The KEEP/CHANGE decision prompt no longer contains statistics
#:           computed from paper picks (ledger EXP-1). Changing the prompt
#:           changes Claude's decisions, which changes which picks persist.
#:
#:       TRAINING-DATA CORRECTION — the dimension `model_version` cannot see:
#:         29 matches carry a participant whose row belongs to a different club
#:         and are marked `training_exclusion_reason = corrupt_team_identity`:
#:           Telstar/Maccabi Tel Aviv 2, SK Rapid/Rapid Bucuresti 10,
#:           St. Pauli/Pau FC 14, Levski Sofia 3.
#:         Picks 1148, 309 and 314 are marked
#:         `evidence_status = void_corrupt_features` — excluded from every
#:         learner and measurer, retained in the ROI record, because the wagers
#:         were real.
#:
#:         A future REPAIR that lifts an exclusion re-includes those matches in
#:         the fitting set. That is prediction-affecting and needs its own
#:         CODE_REVISION bump. It is not bookkeeping.
#:
#:       HOW THE REFIT ACTUALLY WORKS — an earlier claim in this stage was
#:       overturned and the corrected version is what follows.
#:         OVERTURNED: "Poisson and Elo need no artifact surgery because fit()
#:         replays from `self.ratings = {}` against the DATABASE."
#:         Half right. Both DO replay from an empty state — no rating or
#:         strength table was edited and none needed to be — but they replay
#:         from whatever `get_completed_matches` returns, and that is the
#:         Parquet mirror whenever one is warm. The database is the fallback.
#:         A stale mirror would have fed the excluded matches straight back
#:         into a fit that believed it had excluded them.
#:         CORRECTED: exclusion is sufficient only because BOTH caches of the
#:         excluded data are stamped with the filter's generation and refuse
#:         themselves on mismatch — the Parquet mirror and the ML pickles. Two
#:         mechanisms, not one. An exclusion is only real where every cached
#:         derivative of the excluded data is invalidated.
#:
#:         The first retrain is not forced by a flag, a deletion or a dispatch.
#:         The stamp is a property of the DEPLOYED CODE, so the run that has
#:         the filter is the run that refuses the artifact: the restored pickle
#:         has no stamp, `is_fitted` goes false, `trained_at` is CLEARED so the
#:         age check cannot call it fresh, and `--train` retrains. There is no
#:         ordering for anyone to get wrong later.
#:
#:       THIS COHORT OPENS INSIDE AN OUTAGE. API-Football has been suspended
#:       since 2026-08-19 10:10:28 UTC, so the first picks under s5.3 are made
#:       with no fixtures, odds, xG or injuries from that provider. If the
#:       account is restored mid-cohort, this fingerprint spans two materially
#:       different input regimes — ledger OPS-1 records the boundary so any
#:       analysis can split rather than pool. Low pick counts in the first days
#:       are the expected consequence of a one-pick cap on a card discovered
#:       without API-Football, not a defect.
#:
#:       Verification prompt written BEFORE deployment:
#:       docs/stage-13-s53-verification-prompt.md
CODE_REVISION = "s5.3"


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
