"""Closing line value — the measurement, its validity rules, and its coverage.

Stage 4, Phases 10 and 11.

CLV asks one question: **did the price we took beat the price the market settled
on?** It is the strongest diagnostic available to a betting model because it
answers in weeks what ROI needs years to answer — a strategy with persistent
positive CLV and a bad month is working, and one with negative CLV is not,
however its ROI happens to look.

What CLV is NOT
---------------
``predicted_probability - 1/odds`` is the model's *own claimed edge*. It was
reported as ``avg_clv`` in ``--stats``, ``--report`` and Telegram until
2026-08-07, reading **+6.3%** while realised flat ROI was **-3.6%**. A metric
that says you are winning while you are losing is worse than no metric. That
quantity is now called ``model_market_divergence`` and never CLV.

The formula
-----------
Both prices are decimal. Three views are stored, because each answers a
different question and none can be recovered from the others once the raw prices
are gone:

* ``price_clv``     = taken / closing - 1
  The headline. +5% means the price taken paid 5% more than the closing price.
* ``prob_clv``      = 1/taken - 1/closing   (negative when the taken price is better)
  Movement in raw implied-probability terms, which is what compounds across bets.
* ``fair_clv``      = closing_fair / taken_fair - 1, using de-vigged prices
  The margin-free view. Only computable when the full market is known at both
  timestamps, so it is optional — but it is the only one immune to a change in
  the bookmaker's margin between capture and close.

``price_clv`` is the primary metric because it needs only the two prices, which
is what a real bettor observes. The raw prices are always retained so any of
these can be recomputed later under a different definition.

Market type matters
-------------------
For a stake-refunding market (draw no bet) the realised value of a price is
conditional on a decisive result. The price comparison itself is still valid —
both prices carry the same refund clause — so ``price_clv`` needs no adjustment.
What must NOT happen is comparing a price from one market against a price from
another; ``validate_pair`` enforces that explicitly, because mixing markets is
exactly the bug that corrupted 2,548 matches of 1X2 data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger()

#: A closing capture taken earlier than this before kickoff is not a closing
#: line. Recorded as a validity failure rather than silently averaged in.
DEFAULT_MAX_CAPTURE_LEAD = timedelta(minutes=180)

#: A capture at or after kickoff may contain in-play information.
DEFAULT_MAX_CAPTURE_LAG = timedelta(minutes=0)


class CLVInvalid(Exception):
    """A (pick, closing) pair cannot yield a trustworthy CLV."""


@dataclass(frozen=True)
class CLVResult:
    price_clv: float
    prob_clv: float
    fair_clv: Optional[float]
    taken_odds: float
    closing_odds: float
    lead_minutes: Optional[float]

    @property
    def beat_close(self) -> bool:
        return self.price_clv > 0


def validate_pair(*, taken_odds: Optional[float], closing_odds: Optional[float],
                  pick_market: Optional[str], closing_market: Optional[str],
                  pick_selection: Optional[str],
                  closing_selection: Optional[str],
                  kickoff: Optional[datetime],
                  captured_at: Optional[datetime],
                  max_lead: timedelta = DEFAULT_MAX_CAPTURE_LEAD,
                  max_lag: timedelta = DEFAULT_MAX_CAPTURE_LAG) -> None:
    """Raise CLVInvalid unless the pair can produce a trustworthy CLV.

    Phase 11's checklist, in one place, so no caller can compute CLV from a pair
    that fails one of them.
    """
    if not taken_odds or taken_odds <= 1.0:
        raise CLVInvalid(f"taken odds {taken_odds!r} is not a valid decimal price")
    if not closing_odds or closing_odds <= 1.0:
        raise CLVInvalid(f"closing odds {closing_odds!r} is not a valid decimal price")
    if pick_market != closing_market:
        raise CLVInvalid(
            f"market mismatch: pick is {pick_market!r} but closing price is "
            f"{closing_market!r} — comparing across markets is meaningless")
    if pick_selection != closing_selection:
        raise CLVInvalid(
            f"selection mismatch: {pick_selection!r} vs {closing_selection!r}")
    if captured_at is None:
        raise CLVInvalid("closing price has no capture timestamp, so its "
                         "closeness to kickoff cannot be established")
    if kickoff is not None:
        if captured_at > kickoff + max_lag:
            raise CLVInvalid(
                f"closing price captured {captured_at - kickoff} AFTER kickoff — "
                f"it may contain in-play information")
        lead = kickoff - captured_at
        if lead > max_lead:
            raise CLVInvalid(
                f"closing price captured {lead} before kickoff, beyond the "
                f"{max_lead} limit — this is a pre-match snapshot, not a close")


def compute(*, taken_odds: float, closing_odds: float,
            kickoff: Optional[datetime] = None,
            captured_at: Optional[datetime] = None,
            taken_fair: Optional[float] = None,
            closing_fair: Optional[float] = None,
            **validate_kwargs) -> CLVResult:
    """Compute CLV for one validated pair.

    Callers that have the market/selection metadata should call ``validate_pair``
    first; this function re-runs the price checks it can do on its own.
    """
    if not taken_odds or taken_odds <= 1.0 or not closing_odds or closing_odds <= 1.0:
        raise CLVInvalid(f"invalid prices: taken={taken_odds!r} close={closing_odds!r}")

    price_clv = taken_odds / closing_odds - 1.0
    prob_clv = (1.0 / taken_odds) - (1.0 / closing_odds)
    fair_clv = None
    if taken_fair and closing_fair and 0 < taken_fair < 1 and 0 < closing_fair < 1:
        # Margin-free view: how much the FAIR probability moved toward us.
        fair_clv = closing_fair / taken_fair - 1.0

    lead_minutes = None
    if kickoff is not None and captured_at is not None:
        lead_minutes = (kickoff - captured_at).total_seconds() / 60.0

    return CLVResult(price_clv=price_clv, prob_clv=prob_clv, fair_clv=fair_clv,
                     taken_odds=taken_odds, closing_odds=closing_odds,
                     lead_minutes=lead_minutes)


@dataclass
class CLVCoverage:
    """Phase 11's coverage report."""

    total_picks: int = 0
    valid: int = 0
    invalid_reasons: Dict[str, int] = None
    results: List[CLVResult] = None

    def __post_init__(self):
        if self.invalid_reasons is None:
            self.invalid_reasons = {}
        if self.results is None:
            self.results = []

    @property
    def coverage_rate(self) -> float:
        return self.valid / self.total_picks if self.total_picks else 0.0

    @property
    def avg_price_clv(self) -> Optional[float]:
        if not self.results:
            return None
        return sum(r.price_clv for r in self.results) / len(self.results)

    @property
    def median_price_clv(self) -> Optional[float]:
        if not self.results:
            return None
        s = sorted(r.price_clv for r in self.results)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def beat_close_rate(self) -> Optional[float]:
        if not self.results:
            return None
        return sum(1 for r in self.results if r.beat_close) / len(self.results)

    def render(self) -> str:
        lines = [
            "CLV coverage",
            f"  picks considered   : {self.total_picks}",
            f"  valid CLV pairs    : {self.valid}",
            f"  clv_coverage_rate  : {self.coverage_rate:.1%}",
        ]
        if self.results:
            lines += [
                f"  avg price CLV      : {self.avg_price_clv:+.2%}",
                f"  median price CLV   : {self.median_price_clv:+.2%}",
                f"  beat the close     : {self.beat_close_rate:.1%}",
            ]
        else:
            lines.append("  avg price CLV      : n/a — no valid pairs yet")
        if self.invalid_reasons:
            lines.append("  rejected because:")
            for reason, n in sorted(self.invalid_reasons.items(),
                                    key=lambda kv: -kv[1]):
                lines.append(f"      {reason:<52} {n}")
        return "\n".join(lines)


def _reason_bucket(msg: str) -> str:
    """Collapse a message to a stable category for the coverage table."""
    for key, label in (
        ("no capture timestamp", "missing capture timestamp"),
        ("AFTER kickoff", "captured after kickoff"),
        ("before kickoff, beyond", "captured too early (not a close)"),
        ("market mismatch", "market mismatch"),
        ("selection mismatch", "selection mismatch"),
        ("closing odds", "missing/invalid closing price"),
        ("taken odds", "missing/invalid taken price"),
    ):
        if key in msg:
            return label
    return "other"


def coverage_report(picks: Sequence, *, max_lead: timedelta = DEFAULT_MAX_CAPTURE_LEAD
                    ) -> CLVCoverage:
    """Build the coverage report over pick-like objects.

    Each pick needs: odds, closing_odds, closing_odds_captured_at, market,
    selection, and (optionally) a `kickoff` attribute.
    """
    cov = CLVCoverage(total_picks=len(picks))
    for p in picks:
        try:
            validate_pair(
                taken_odds=getattr(p, "odds", None),
                closing_odds=getattr(p, "closing_odds", None),
                pick_market=getattr(p, "market", None),
                closing_market=getattr(p, "market", None),
                pick_selection=getattr(p, "selection", None),
                closing_selection=getattr(p, "selection", None),
                kickoff=getattr(p, "kickoff", None),
                captured_at=getattr(p, "closing_odds_captured_at", None),
                max_lead=max_lead,
            )
            res = compute(
                taken_odds=p.odds, closing_odds=p.closing_odds,
                kickoff=getattr(p, "kickoff", None),
                captured_at=getattr(p, "closing_odds_captured_at", None),
            )
        except CLVInvalid as e:
            bucket = _reason_bucket(str(e))
            cov.invalid_reasons[bucket] = cov.invalid_reasons.get(bucket, 0) + 1
            continue
        cov.valid += 1
        cov.results.append(res)
    return cov
