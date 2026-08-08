"""The single source of truth for what a stored betting market must look like.

Stage 4, Phase 1. This module exists because of one incident.

API-Football's ``"Home/Away"`` bet is the TWO-WAY, draw-excluded market. It was
mapped to ``market_type: "1X2"`` with the same ``Home``/``Away`` selection labels
as ``"Match Winner"``, so for any bookmaker offering both bets the two-way prices
overwrote the genuine three-way ones through the odds table's
``ON CONFLICT (match_id, bookmaker, market_type, selection) DO UPDATE``.

Measured blast radius (2026-08-07): **13,274 rows across 2,548 matches and seven
bookmakers** — Bet365 92% corrupt, William Hill 94%, Unibet 96%, Betfair 86%,
10Bet 98%, Betano 96%, 888Sport 97%, and even Pinnacle 26%. 92% of all saved
picks were built on the resulting implied probabilities.

Nothing in the codebase could have caught it, because "what shape is a 1X2
market" was implicit knowledge spread across a scraper, a feature builder and a
value calculator. It is now declared once, here, and every writer and reader
checks against it.

Two independent invariants are enforced:

1. **Arity and legs.** A three-way market has exactly three named outcomes; a
   two-way market has exactly two. A writer may not create a 1X2 record from a
   bet that cannot supply all three.
2. **Overround plausibility.** A real book's margin is bounded. A three-way book
   summing to 1.35 is not a bookmaker's opinion, it is a bug — and the bound
   catches the *next* mapping error too, not just this one.

Markets whose outcomes OVERLAP (double chance: 1X, 12, X2 each contain two of
the three base outcomes) are declared with ``overlapping=True`` and are NOT
subject to a sum-to-one check. Getting this wrong produces a false alarm: the
first pass of this audit flagged double chance as corrupt because
P(1X)+P(X2) = 1 + P(D) ≈ 1.25, which is correct behaviour, not corruption.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class MarketSpec:
    """The declared shape of one stored market type."""

    market_type: str
    #: Each tuple is one outcome, listed as its accepted selection aliases.
    #: Order matters: it is the canonical probability order.
    legs: Tuple[Tuple[str, ...], ...]
    #: Plausible range for sum(1/odds) across the legs of ONE bookmaker.
    overround_min: float
    overround_max: float
    #: True when legs overlap (double chance), so they do not partition the
    #: outcome space and no sum-to-one interpretation applies.
    overlapping: bool = False
    description: str = ""

    @property
    def arity(self) -> int:
        return len(self.legs)


#: Three-way books run ~1.02-1.12 in practice; 1.25 is a deliberately generous
#: ceiling so only genuinely broken markets are rejected. The corruption sat at a
#: median of 1.3524, comfortably outside.
_OR3 = (1.005, 1.25)
_OR2 = (1.005, 1.20)

MARKET_SPECS: Dict[str, MarketSpec] = {
    "1X2": MarketSpec(
        market_type="1X2",
        legs=(("Home", "Home Win", "1"), ("Draw", "X"), ("Away", "Away Win", "2")),
        overround_min=_OR3[0], overround_max=_OR3[1],
        description="Three-way match result. NEVER populated from a two-way bet.",
    ),
    "draw_no_bet": MarketSpec(
        market_type="draw_no_bet",
        legs=(("DNB Home", "Home"), ("DNB Away", "Away")),
        overround_min=_OR2[0], overround_max=_OR2[1],
        description="Two-way, stake refunded on a draw. This is what API-Football "
                    "calls both 'Draw No Bet' and 'Home/Away'.",
    ),
    "over_under": MarketSpec(
        market_type="over_under",
        legs=(("Over",), ("Under",)),   # line-qualified at validation time
        overround_min=_OR2[0], overround_max=_OR2[1],
        description="Match total goals, one line at a time (Over 2.5 / Under 2.5).",
    ),
    "btts": MarketSpec(
        market_type="btts",
        legs=(("Yes", "BTTS Yes"), ("No", "BTTS No")),
        overround_min=_OR2[0], overround_max=_OR2[1],
        description="Both teams to score.",
    ),
    "team_goals": MarketSpec(
        market_type="team_goals",
        legs=(("Over",), ("Under",)),   # line- and side-qualified at validation
        overround_min=_OR2[0], overround_max=_OR2[1],
        description="Per-team goal line (Home Over 1.5 / Home Under 1.5).",
    ),
    "double_chance": MarketSpec(
        market_type="double_chance",
        legs=(("Double Chance 1X", "1X"), ("Double Chance 12", "12"),
              ("Double Chance X2", "X2")),
        # Legs overlap: sum(1/odds) over all three is ~2 + margin, not ~1.
        overround_min=1.90, overround_max=2.40,
        overlapping=True,
        description="Double chance. Outcomes OVERLAP — each covers two of the "
                    "three base results, so the three inverse prices sum to ~2, "
                    "not ~1. Do not apply a sum-to-one check.",
    ),
}

#: Bet names that are authoritative for a market_type. A source bet not listed
#: here may not write that market, which is the structural fix for the incident:
#: "Home/Away" is simply not an authority for 1X2, whatever its labels look like.
AUTHORITATIVE_BETS: Dict[str, frozenset] = {
    "1X2": frozenset({"Match Winner"}),
    "draw_no_bet": frozenset({"Draw No Bet", "Home/Away"}),
    "over_under": frozenset({"Goals Over/Under"}),
    "btts": frozenset({"Both Teams Score"}),
    "team_goals": frozenset({"Total - Home", "Total - Away"}),
    "double_chance": frozenset({"Double Chance"}),
}


class MarketValidationError(ValueError):
    """A market record violates its declared shape."""


def get_spec(market_type: str) -> Optional[MarketSpec]:
    return MARKET_SPECS.get(market_type)


def is_authoritative(market_type: str, bet_name: str) -> bool:
    """Whether ``bet_name`` is allowed to write ``market_type``.

    Unknown market types return True — this guard is about protecting the markets
    we have declared, not about blocking new ones.
    """
    allowed = AUTHORITATIVE_BETS.get(market_type)
    return True if allowed is None else bet_name in allowed


def overround(prices: Sequence[float]) -> float:
    """sum(1/odds). Raises on a non-positive or sub-evens-impossible price."""
    if not prices or any(p is None or p <= 1.0 for p in prices):
        raise MarketValidationError(f"invalid decimal odds in {prices!r}")
    return sum(1.0 / p for p in prices)


def check_overround(market_type: str, prices: Sequence[float]) -> Tuple[bool, str]:
    """Whether a book's prices for one market are internally plausible."""
    spec = get_spec(market_type)
    if spec is None:
        return True, "no spec declared"
    try:
        orr = overround(prices)
    except MarketValidationError as e:
        return False, str(e)
    if len(prices) != spec.arity:
        return False, (f"{market_type} needs {spec.arity} legs, got {len(prices)}")
    if not (spec.overround_min <= orr <= spec.overround_max):
        return False, (
            f"{market_type} overround {orr:.4f} outside "
            f"[{spec.overround_min}, {spec.overround_max}] — the legs are almost "
            f"certainly not all from the same market"
        )
    return True, f"overround {orr:.4f} ok"


def devig(market_type: str, prices: Sequence[float]) -> Optional[List[float]]:
    """Margin-free probabilities, or None when the book fails validation.

    Overlapping markets are refused: normalising 1X/12/X2 to sum to 1 would be
    meaningless, and silently doing so is how a plausible-looking wrong number
    gets into a model.
    """
    spec = get_spec(market_type)
    if spec is not None and spec.overlapping:
        return None
    ok, _ = check_overround(market_type, prices)
    if not ok:
        return None
    orr = overround(prices)
    return [(1.0 / p) / orr for p in prices]


def extract_legs(market_type: str, odds_by_selection: Dict[str, float],
                 line: Optional[str] = None,
                 side: Optional[str] = None) -> Optional[List[float]]:
    """Pull one book's prices for a market in canonical leg order.

    ``line`` qualifies over/under markets ("2.5"); ``side`` additionally
    qualifies team_goals ("Home"/"Away"). Returns None when any leg is missing —
    a partial market is never silently completed.
    """
    spec = get_spec(market_type)
    if spec is None:
        return None
    prices: List[float] = []
    for aliases in spec.legs:
        found = None
        for alias in aliases:
            if market_type == "over_under" and line:
                key = f"{alias} {line}"
            elif market_type == "team_goals" and line and side:
                key = f"{side} {alias} {line}"
            else:
                key = alias
            v = odds_by_selection.get(key)
            if v and v > 1.0:
                found = v
                break
        if found is None:
            return None
        prices.append(found)
    return prices


def validate_write(market_type: str, bet_name: str,
                   selections: Iterable[str]) -> Tuple[bool, str]:
    """Gate for an odds WRITER, applied before persisting a bookmaker's market.

    Rejects the exact class of bug that caused the incident: a non-authoritative
    source bet writing into a declared market type.
    """
    if not is_authoritative(market_type, bet_name):
        return False, (
            f"bet {bet_name!r} is not authoritative for market_type "
            f"{market_type!r} (allowed: "
            f"{sorted(AUTHORITATIVE_BETS.get(market_type, []))}). Writing it "
            f"would overwrite the genuine market under the same unique key."
        )
    spec = get_spec(market_type)
    if spec is None:
        return True, "no spec declared"
    sels = list(selections)
    if len(set(sels)) != len(sels):
        return False, f"duplicate selections in one write: {sels}"
    return True, "ok"
