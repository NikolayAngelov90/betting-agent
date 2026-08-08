"""The immutable baseline experiment (Stage 3, Phase 1).

Every claim about whether a change improves the system is measured against this
harness. Its rules:

* **Chronological only.** Train on everything before a cutoff, evaluate on the
  next ``window_days``. Never a random shuffle.
* **Identical match set for every model.** A model is only scored on fixtures
  where *all* candidates can produce a forecast, so differences reflect the
  models and not their coverage.
* **Proper scoring rules first.** Log-loss and Brier are the primary metrics;
  accuracy is reported but never optimised. Calibration error is reported
  separately because a model can be accurate and badly calibrated.
* **Metrics that cannot be computed are reported as unavailable, not omitted
  and not approximated.** CLV in particular: the database holds no closing line
  (0 of 124,158 odds rows on picked matches were written after the pick day), so
  ``clv`` comes back ``None`` with a reason until closing-line capture exists.

The output is a plain dict, snapshotted to JSON by ``scripts/run_baseline.py``.
Once a snapshot is committed it is not edited — a new run writes a new file, and
before/after comparisons diff the two.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from src.utils.logger import get_logger

logger = get_logger()

# Outcome index convention used throughout: 0 = home win, 1 = draw, 2 = away win.
HOME, DRAW, AWAY = 0, 1, 2

#: Bookmakers whose 1X2 market is trusted for the market baseline. Excludes the
#: Flashscore display prices (composite, not bettable).
DEFAULT_BOOKMAKERS = ("Pinnacle", "Bet365", "1xBet")

#: Overround bands outside which a book's market is treated as broken and
#: dropped. See feature_engineer._get_bookmaker_features for the incident that
#: made this necessary (Bet365 1X2 at a median overround of 1.3524).
OVERROUND_3WAY = (1.005, 1.25)


@dataclass
class MatchRow:
    """One completed match with everything the harness needs."""

    id: int
    match_date: date
    home_team_id: int
    away_team_id: int
    home_goals: int
    away_goals: int
    league: str
    #: {bookmaker: {selection: decimal odds}} for the 1X2 market
    odds_1x2: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def outcome(self) -> int:
        if self.home_goals > self.away_goals:
            return HOME
        return DRAW if self.home_goals == self.away_goals else AWAY

    @property
    def total_goals(self) -> int:
        return self.home_goals + self.away_goals


# --------------------------------------------------------------------- metrics

def log_loss(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    total = 0.0
    for p, y in zip(probs, outcomes):
        arr = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
        arr = arr / arr.sum()
        total += -math.log(arr[y])
    return total / len(outcomes)


def brier(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    total = 0.0
    for p, y in zip(probs, outcomes):
        arr = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
        arr = arr / arr.sum()
        target = np.zeros(len(arr))
        target[y] = 1.0
        total += float(((arr - target) ** 2).sum())
    return total / len(outcomes)


def accuracy(probs: Sequence[Sequence[float]], outcomes: Sequence[int]) -> float:
    hits = sum(int(int(np.argmax(p)) == y) for p, y in zip(probs, outcomes))
    return hits / len(outcomes)


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """ECE over the flattened (probability, hit) pairs across all three outcomes.

    Reported separately from Brier because Brier conflates calibration with
    discrimination: a model can improve Brier while getting worse at meaning what
    it says.
    """
    pairs = []
    for p, y in zip(probs, outcomes):
        arr = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
        arr = arr / arr.sum()
        for k, pk in enumerate(arr):
            pairs.append((float(pk), 1.0 if k == y else 0.0))
    if not pairs:
        return float("nan")
    pairs.sort()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total, n = 0.0, len(pairs)
    for lo, hi in zip(edges[:-1], edges[1:]):
        chunk = [(p, h) for p, h in pairs if lo <= p < hi]
        if not chunk:
            continue
        mean_p = sum(p for p, _ in chunk) / len(chunk)
        mean_h = sum(h for _, h in chunk) / len(chunk)
        total += (len(chunk) / n) * abs(mean_p - mean_h)
    return total


def paired_bootstrap(loss_a: Sequence[float], loss_b: Sequence[float],
                     iters: int = 2000, seed: int = 0) -> Dict:
    """95% CI on the mean per-match loss difference (a - b).

    Phase 19: "for every claimed improvement calculate uncertainty". A positive
    interval that excludes zero means b genuinely beats a.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(d, len(d), replace=True).mean()
                      for _ in range(iters)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean_improvement": float(d.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": bool(lo > 0 or hi < 0),
    }


# ----------------------------------------------------------------- market model

def devig_1x2(book_odds: Dict[str, float]) -> Optional[List[float]]:
    """De-vig one book's 1X2 market, or None when it is missing a leg or its
    overround is implausible."""
    legs = [("Home", "Home Win", "1"), ("Draw", "X"), ("Away", "Away Win", "2")]
    prices = []
    for alts in legs:
        v = next((book_odds[k] for k in alts if book_odds.get(k)), None)
        if not v or v <= 1.0:
            return None
        prices.append(v)
    inv = [1.0 / p for p in prices]
    overround = sum(inv)
    lo, hi = OVERROUND_3WAY
    if not (lo <= overround <= hi):
        return None
    return [i / overround for i in inv]


def market_consensus(row: MatchRow) -> Optional[List[float]]:
    """Median de-vigged probability per outcome across every usable book."""
    per_book = [d for d in (devig_1x2(o) for o in row.odds_1x2.values()) if d]
    if not per_book:
        return None
    med = []
    for col in zip(*per_book):
        s = sorted(col)
        n = len(s)
        med.append(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
    total = sum(med)
    return [v / total for v in med] if total > 0 else None


def market_raw(row: MatchRow) -> Optional[List[float]]:
    """Raw 1/odds, normalised but with the margin left in — the 'do nothing at
    all' reference."""
    per_book = []
    for od in row.odds_1x2.values():
        legs = [("Home", "Home Win", "1"), ("Draw", "X"), ("Away", "Away Win", "2")]
        prices = []
        for alts in legs:
            v = next((od[k] for k in alts if od.get(k)), None)
            if not v or v <= 1.0:
                prices = []
                break
            prices.append(v)
        if prices:
            per_book.append([1.0 / p for p in prices])
    if not per_book:
        return None
    mean = np.mean(per_book, axis=0)
    return list(mean / mean.sum())


# ------------------------------------------------------------------ the harness

@dataclass
class Candidate:
    """A named forecaster.

    ``fit`` is called once per window with the training rows; ``predict`` is then
    called per test row and returns a 3-vector or None (no forecast possible).
    """

    name: str
    fit: Callable[[List[MatchRow], date], object]
    predict: Callable[[object, MatchRow], Optional[List[float]]]


def walk_forward(rows: List[MatchRow], candidates: List[Candidate],
                 cutoffs: Sequence[date], window_days: int = 60,
                 min_train: int = 500) -> Dict:
    """Run every candidate over the same chronological windows.

    Returns a dict of per-candidate metrics plus the shared evaluation set size.
    Only matches where EVERY candidate produced a forecast are scored, so the
    comparison is like-for-like.
    """
    rows = sorted(rows, key=lambda r: r.match_date)
    per_candidate_probs: Dict[str, List] = defaultdict(list)
    per_candidate_keys: Dict[str, List] = defaultdict(list)

    for cutoff in cutoffs:
        train = [r for r in rows if r.match_date < cutoff]
        test = [r for r in rows
                if cutoff <= r.match_date < cutoff + timedelta(days=window_days)]
        if len(train) < min_train or not test:
            logger.debug(f"baseline: skipping cutoff {cutoff} "
                         f"(train={len(train)}, test={len(test)})")
            continue
        for cand in candidates:
            state = cand.fit(train, cutoff)
            for r in test:
                p = cand.predict(state, r)
                if p is not None:
                    per_candidate_probs[cand.name].append(list(p))
                    per_candidate_keys[cand.name].append(r.id)

    # Intersect on match ids so every candidate is scored on the same fixtures.
    id_sets = [set(v) for v in per_candidate_keys.values()]
    if not id_sets:
        return {"n": 0, "candidates": {}}
    shared = set.intersection(*id_sets)
    outcome_by_id = {r.id: r.outcome for r in rows}

    results: Dict[str, Dict] = {}
    per_match_loss: Dict[str, List[float]] = {}
    for name in per_candidate_probs:
        probs, outs, losses = [], [], []
        for mid, p in zip(per_candidate_keys[name], per_candidate_probs[name]):
            if mid not in shared:
                continue
            y = outcome_by_id[mid]
            probs.append(p)
            outs.append(y)
            arr = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
            arr = arr / arr.sum()
            losses.append(-math.log(arr[y]))
        if not probs:
            continue
        per_match_loss[name] = losses
        results[name] = {
            "log_loss": round(log_loss(probs, outs), 5),
            "brier": round(brier(probs, outs), 5),
            "calibration_error": round(expected_calibration_error(probs, outs), 5),
            "accuracy": round(accuracy(probs, outs), 5),
            "n": len(probs),
            # Reported, not omitted: see the module docstring.
            "roi": None,
            "roi_note": "not computed here — the harness scores forecasts, not a "
                        "staking plan; ROI lives in the settled-pick analysis",
            "clv": None,
            "clv_note": "UNAVAILABLE — no closing line is stored (0 of 124,158 "
                        "odds rows on picked matches were written after the pick "
                        "day). Requires closing-line capture.",
        }

    return {
        "n_shared_matches": len(shared),
        "window_days": window_days,
        "cutoffs": [c.isoformat() for c in cutoffs],
        "candidates": results,
        "_per_match_loss": per_match_loss,
    }


def compare(results: Dict, baseline_name: str, challenger_name: str) -> Dict:
    """Paired bootstrap of challenger against baseline on per-match log-loss."""
    losses = results.get("_per_match_loss", {})
    if baseline_name not in losses or challenger_name not in losses:
        return {"error": f"missing {baseline_name!r} or {challenger_name!r}"}
    return paired_bootstrap(losses[baseline_name], losses[challenger_name])


def format_table(results: Dict) -> str:
    """The Phase 1 table, rendered for the console and the report."""
    header = (f"{'Model':<34}{'LogLoss':>9}{'Brier':>9}{'CalErr':>9}"
              f"{'Acc':>8}{'ROI':>8}{'CLV':>8}{'n':>7}")
    lines = [header, "-" * len(header)]
    for name, m in sorted(results["candidates"].items(),
                          key=lambda kv: kv[1]["log_loss"]):
        lines.append(
            f"{name:<34}{m['log_loss']:>9.4f}{m['brier']:>9.4f}"
            f"{m['calibration_error']:>9.4f}{m['accuracy']:>8.1%}"
            f"{'n/a':>8}{'n/a':>8}{m['n']:>7}"
        )
    lines.append("")
    lines.append("ROI: not measured by this harness (it scores forecasts, not stakes).")
    lines.append("CLV: UNAVAILABLE — no closing line is stored. See docs/"
                 "predictive-audit-2026-08-07.md section 11.")
    return "\n".join(lines)
