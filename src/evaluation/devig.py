"""De-vigging estimators, and the margin models each one inverts.

WHY THIS MODULE EXISTS. `docs/flb-preregistration.md` fixed PROPORTIONAL
de-vigging and the band test read its artefact as a finding: a perfectly
calibrated book that adds margin in equal shares shows a calibration gap of
`M*(p - 1/n)/(1+M)` under the proportional estimator. The null was not zero.

AN ESTIMATOR IS EXACT ONLY FOR THE MARGIN MODEL IT INVERTS. That is the whole
content of the synthetic control in `scripts/sharp_anchor.py`, and it is why
four estimators live here beside their four generating models:

    estimator       inverts the book that quotes
    proportional    q_i = p_i * (1 + M)                multiplicative
    additive        q_i = p_i + M/n                    equal shares
    power           q_i = p_i ** (1/k)                 power
    shin            Shin (1992, 1993) insider model    z = insider share

For a TWO-WAY market Shin and additive are the SAME estimator (a known
identity; driven in the tests). They differ for three or more legs.

`q` is always the vector of QUOTED IMPLIED probabilities, `1/odds`, for one
complete leg set of one book at one instant. Nothing here pools across leg
sets, and nothing here reads a database. Pure stdlib, on purpose.

Measurement only. Nothing in the prediction path imports this.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence


def _valid(q: Sequence[float]) -> bool:
    return len(q) >= 2 and all(0.0 < x < 1.0 for x in q) and sum(q) > 1.0


def proportional(q: Sequence[float]) -> Optional[List[float]]:
    """`p_i = q_i / sum(q)`. Exact for a MULTIPLICATIVE margin only."""
    if not _valid(q):
        return None
    s = sum(q)
    return [x / s for x in q]


def additive(q: Sequence[float]) -> Optional[List[float]]:
    """`p_i = q_i - (sum(q) - 1)/n`. Exact for an EQUAL-SHARE margin.

    REFUSES rather than clips when a leg would go non-positive: an extreme
    longshot quoted below its own margin share has no additive fair price,
    and clipping it would be a silent third state.
    """
    if not _valid(q):
        return None
    share = (sum(q) - 1.0) / len(q)
    out = [x - share for x in q]
    if any(x <= 0.0 for x in out):
        return None
    return out


def power(q: Sequence[float], tol: float = 1e-12) -> Optional[List[float]]:
    """`p_i = q_i ** k` with `k` solving `sum(q_i ** k) = 1`. Bisection."""
    if not _valid(q):
        return None
    lo, hi = 1.0, 1.0
    # sum(q^k) is decreasing in k; grow hi until it drops below 1
    for _ in range(200):
        hi *= 2.0
        if sum(x ** hi for x in q) < 1.0:
            break
    else:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(x ** mid for x in q) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    k = (lo + hi) / 2.0
    return [x ** k for x in q]


def _shin_probs(q: Sequence[float], z: float) -> List[float]:
    s = sum(q)
    return [(math.sqrt(z * z + 4.0 * (1.0 - z) * x * x / s) - z)
            / (2.0 * (1.0 - z)) for x in q]


def shin(q: Sequence[float], tol: float = 1e-12) -> Optional[List[float]]:
    """Shin's estimator: `z` in [0, 1) solving `sum(p_i(z)) = 1`. Bisection.

    `z` is the model's share of insider money. `sum(p_i(z))` is DECREASING in
    `z`, equals `sqrt(sum(q))`-scaled above 1 at `z = 0`, and falls through 1.
    """
    if not _valid(q):
        return None
    lo, hi = 0.0, 0.999
    if sum(_shin_probs(q, hi)) > 1.0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(_shin_probs(q, mid)) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return _shin_probs(q, (lo + hi) / 2.0)


ESTIMATORS: Dict[str, Callable[[Sequence[float]], Optional[List[float]]]] = {
    "proportional": proportional,
    "additive": additive,
    "power": power,
    "shin": shin,
}


# ── THE GENERATING MODELS: a perfectly calibrated book, four ways ────────

def quote_multiplicative(p: Sequence[float], margin: float) -> List[float]:
    return [x * (1.0 + margin) for x in p]


def quote_additive(p: Sequence[float], margin: float) -> List[float]:
    return [x + margin / len(p) for x in p]


def quote_power(p: Sequence[float], margin: float) -> List[float]:
    """`q_i = p_i ** (1/k)` with `k > 1` chosen so `sum(q) = 1 + margin`."""
    lo, hi = 1.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(x ** (1.0 / mid) for x in p) < 1.0 + margin:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2.0
    return [x ** (1.0 / k) for x in p]


def quote_shin(p: Sequence[float], margin: float) -> List[float]:
    """Shin's equilibrium quotes, `z` chosen so `sum(q) = 1 + margin`.

    `q_i = sqrt(z*p_i + (1-z)*p_i**2) * sum_j sqrt(z*p_j + (1-z)*p_j**2)`.
    """
    def quotes(z: float) -> List[float]:
        roots = [math.sqrt(z * x + (1.0 - z) * x * x) for x in p]
        s = sum(roots)
        return [r * s for r in roots]
    lo, hi = 0.0, 0.999
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if sum(quotes(mid)) < 1.0 + margin:
            lo = mid
        else:
            hi = mid
    return quotes((lo + hi) / 2.0)


GENERATORS: Dict[str, Callable[[Sequence[float], float], List[float]]] = {
    "multiplicative": quote_multiplicative,
    "additive": quote_additive,
    "power": quote_power,
    "shin": quote_shin,
}

#: Which estimator is EXACT for which generating model. The control asserts
#: this diagonal is zero and reports everything off it.
EXACT_FOR = {
    "multiplicative": "proportional",
    "additive": "additive",
    "power": "power",
    "shin": "shin",
}
