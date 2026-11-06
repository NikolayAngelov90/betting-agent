"""MARKET MIX — Part A: the cost of each market, on the FULL population.

Pre-registered in `docs/market-mix-preregistration.md`, committed 2026-11-05
before any restricted ROI was read. This script computes Part A only: the
per-market cost table. It adopts nothing.

WHY IT IS RECOMPUTED. Part A was first computed on **683** of 1,583 settled
paper picks, because the join needed `odds.selection = saved_picks.selection`
and the two name the same bet differently -- so 1X2 and over_under, the two
largest markets, were ABSENT from it. The vocabulary extraction of 2026-11-05
closed that: `src.data.market_vocabulary.canonical` reconciles both sides and
the join now matches **1,603 of 1,603**.

    A figure that changes when the population triples is a figure the
    smaller population could not support.

So every figure is reported with its 683-pick predecessor beside it.

THE COST IS A PROPERTY OF THE PRICE, NOT OF THE OUTCOME. `cost` is the
bookmaker's margin, computed from quoted odds over a market's COMPLETE leg
set. It is knowable before any bet settles and would be identical if every
result had gone the other way. That is what makes selecting on it different
from the selectivity test this project already ran and refuted.

DOUBLE CHANCE IS DERIVED DIFFERENTLY, AND THAT IS NOT A CHOICE. Its three
legs OVERLAP and are not exhaustive, so `sum(1/o)` over them is ~2 by
construction and is not a margin. `market_spec.devig` refuses them outright.
The fair price of `1X` is `1/(p_home + p_draw)` from the DE-VIGGED 1X2 prices,
and the margin follows from that -- the derivation already in the ledger.

    python -m scripts.market_mix            # the table
    python -m scripts.market_mix --json     # machine-readable
"""
from __future__ import annotations

import argparse
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

#: The registered 1.5% cut. Fixed in the pre-registration BEFORE the numbers
#: were read, and not re-chosen on where our markets fall.
CHEAP_CUT = 0.015

#: Bootstrap draws, resampling FIXTURES (not picks). Registered.
BOOTSTRAP_DRAWS = 4000

#: Part A as first computed, on the 683 picks the selection-only join reached.
#: Kept so every recomputed figure is reported against its predecessor.
PRIOR_683 = {
    "population": 683,
    "note": ("selection-only join; 1X2 and over_under ABSENT because the two "
             "sides name the same bet differently"),
    "cost": {"1X2": 0.0713, "btts": 0.0619, "double_chance": 0.0580},
    "record": {"double_chance": {"n": 265, "win": 0.600, "roi": -0.0551,
                                 "avg_odds": 1.58}},
}


#: Part A as computed on the FULL population, 2026-11-06. Recorded so the
#: next recomputation has a predecessor, the way PRIOR_683 served this one.
#: TAKEN cost is the quantity the hypothesis turns on; `required` is
#: cost/(1+cost), the ROI a zero-edge bettor loses in that market.
MEASURED_FULL = {
    "population": 1639, "joined": 1639, "measured": "2026-11-06",
    "taken_cost": {"double_chance": 0.0574, "over_under": 0.0609,
                   "1X2": 0.0632, "team_goals": 0.0667,
                   "draw_no_bet": 0.0718, "btts": 0.0767},
    "n": {"over_under": 583, "team_goals": 317, "double_chance": 271,
          "1X2": 192, "btts": 153, "draw_no_bet": 123},
    "restricted_at_cut": 0,
    "weighted_mean_cost": 0.0640,
    "max_saving_available": 0.0066,
    "h_mix_predicted": 0.0500,
    "verdict": ("REFUTED AT PART A: no market is within 3.8x of the 1.5% cut, "
                "so the restricted population is empty and the decision "
                "rule's ROI comparison was never computable"),
}


def _load_env() -> None:
    """INSIDE a function, never at import: a module-level load undoes
    conftest's DATABASE_URL strip and has reached production before."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in (".env", ".env.local"):
        try:
            with open(os.path.join(here, name), encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                    if m and not os.environ.get(m.group(1)):
                        os.environ[m.group(1)] = \
                            m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


def _cost_from_legs(market: str, prices: List[float]) -> Optional[float]:
    """The book's margin for one complete leg set, or None if implausible."""
    from src.data.market_spec import check_overround, overround
    ok, _why = check_overround(market, prices)
    if not ok:
        return None
    return overround(prices) - 1.0


def _double_chance_cost(dc_price: float, p_home: float, p_draw: float,
                        p_away: float, leg: str) -> Optional[float]:
    """The margin on one double-chance leg, from DE-VIGGED 1X2 probabilities.

    `sum(1/o)` over 1X/12/X2 is ~2 by construction and is NOT a margin;
    `market_spec.devig` refuses the market for that reason. The fair price of
    `1X` is `1/(p_home + p_draw)`, so the margin is the quoted inverse price
    over that fair probability, minus one.
    """
    fair = {"Double Chance 1X": p_home + p_draw,
            "Double Chance 12": p_home + p_away,
            "Double Chance X2": p_draw + p_away}.get(leg)
    if not fair or fair <= 0 or dc_price <= 1.0:
        return None
    return (1.0 / dc_price) / fair - 1.0


def measure(verbose: bool = True) -> dict:
    _load_env()
    from sqlalchemy import text

    from scripts.capture_closing_lines import SELECTION_SPEC
    from src.data.database import get_db
    from src.data.market_spec import devig, extract_legs, get_spec
    from src.data.market_vocabulary import canonical

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"

    with db.get_session() as s:
        picks = s.execute(text("""
            SELECT id, match_id, market, selection, odds, result
            FROM saved_picks
            WHERE pick_date >= '2026-08-11'
              AND disposition IS NULL
              AND result IN ('win','loss')
              AND market IS NOT NULL AND selection IS NOT NULL
              AND odds IS NOT NULL AND odds > 1.0
            ORDER BY id
        """)).fetchall()
        match_ids = sorted({p[1] for p in picks})
        odds_rows = s.execute(text("""
            SELECT match_id, bookmaker, market_type, selection, odds_value
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
        """), {"ids": match_ids}).fetchall()

    # ── the book-level price book, per (match, bookmaker) ──────────────────
    book: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    for mid, bm, mt, sel, val in odds_rows:
        book[(mid, bm)][mt][sel] = float(val)

    # ── Part A: the cost of each market, from the PRICES ──────────────────
    #
    # TWO COST QUANTITIES, AND ONLY ONE OF THEM IS WHAT WE PAY.
    #
    #   per-book   each bookmaker's own overround over its complete leg set.
    #              A property of that book's pricing.
    #   TAKEN      the overround of the price set WE ACTUALLY TAKE. The pricer
    #              returns the cross-book MEDIAN per leg, so the taken set is
    #              assembled leg by leg across books and its overround is not
    #              any single book's.
    #
    # The hypothesis is about what restricting the market mix would SAVE US,
    # so the taken figure is the one it turns on. The first version of this
    # script reported only the per-book median and would have compared it
    # against the prior's TAKEN 7.13% -- two different quantities read as a
    # movement.
    costs: Dict[str, List[float]] = defaultdict(list)
    taken_costs: Dict[str, List[float]] = defaultdict(list)
    dc_costs: List[float] = []
    dc_taken: List[float] = []
    # the cross-book median price per (match, market, selection)
    med: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
    _acc: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    for mid, bm, mt, sel, val in odds_rows:
        _acc[(mid, mt, sel)].append(float(val))
    for (mid, mt, sel), vals in _acc.items():
        med[(mid, mt)][sel] = statistics.median(vals)
    for (mid, _bm), by_market in book.items():
        # the ordinary markets: a complete leg set, validated
        for market, by_sel in by_market.items():
            spec = get_spec(market)
            if spec is None or spec.overlapping:
                continue
            lines = {None}
            if market in ("over_under", "team_goals"):
                lines = {m.group(1) for k in by_sel
                         for m in [re.search(r"(\d+\.\d+)", k)] if m}
            sides = {None} if market != "team_goals" else {"Home", "Away"}
            for line in lines:
                for side in sides:
                    prices = extract_legs(market, by_sel, line=line, side=side)
                    if not prices:
                        continue
                    c = _cost_from_legs(market, prices)
                    if c is not None:
                        costs[market].append(c)
        # double chance, from the de-vigged 1X2 of the SAME book
        dc = by_market.get("double_chance") or {}
        legs1x2 = extract_legs("1X2", by_market.get("1X2") or {})
        if dc and legs1x2:
            fair = devig("1X2", legs1x2)
            if fair:
                ph, pd, pa = fair
                for leg_name, price in dc.items():
                    canon = canonical("double_chance", leg_name)
                    if canon is None:
                        continue
                    c = _double_chance_cost(float(price), ph, pd, pa, canon[1])
                    if c is not None:
                        dc_costs.append(c)
    if dc_costs:
        costs["double_chance"] = dc_costs

    # ── THE TAKEN PRICE SET, assembled leg by leg from cross-book medians ──
    for (mid, market), by_sel in med.items():
        spec = get_spec(market)
        if spec is None or spec.overlapping:
            continue
        lines = {None}
        if market in ("over_under", "team_goals"):
            lines = {m.group(1) for k in by_sel
                     for m in [re.search(r"(\d+\.\d+)", k)] if m}
        sides = {None} if market != "team_goals" else {"Home", "Away"}
        for line in lines:
            for side in sides:
                prices = extract_legs(market, by_sel, line=line, side=side)
                if not prices:
                    continue
                c = _cost_from_legs(market, prices)
                if c is not None:
                    taken_costs[market].append(c)
    # double chance on the taken set, from the taken 1X2
    for (mid, market), by_sel in med.items():
        if market != "double_chance":
            continue
        legs1x2 = extract_legs("1X2", med.get((mid, "1X2")) or {})
        if not legs1x2:
            continue
        fair = devig("1X2", legs1x2)
        if not fair:
            continue
        ph, pd, pa = fair
        for leg_name, price in by_sel.items():
            canon = canonical("double_chance", leg_name)
            if canon is None:
                continue
            c = _double_chance_cost(float(price), ph, pd, pa, canon[1])
            if c is not None:
                dc_taken.append(c)
    if dc_taken:
        taken_costs["double_chance"] = dc_taken

    # ── the pick record, per market family, canonical-to-canonical ─────────
    joined = 0
    per: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    unnameable = 0
    for _pid, mid, market, selection, odds, result in picks:
        canon = canonical(market, selection)
        if canon is None:
            unnameable += 1
            continue
        fam = canon[0]
        spec_line = SELECTION_SPEC.get(canon[1])
        if spec_line is None:
            unnameable += 1
            continue
        joined += 1
        per[fam].append((mid, float(odds), 1 if result == "win" else 0))

    out = {"population": len(picks), "joined": joined,
           "unnameable": unnameable, "markets": {}}

    for fam, rows in sorted(per.items(), key=lambda kv: -len(kv[1])):
        n = len(rows)
        wins = sum(w for _m, _o, w in rows)
        roi = statistics.mean(o * w - 1.0 for _m, o, w in rows)
        avg_odds = statistics.mean(o for _m, o, _w in rows)
        cs = costs.get(fam) or []
        ts = taken_costs.get(fam) or []
        cost = statistics.median(cs) if cs else None
        taken = statistics.median(ts) if ts else None
        out["markets"][fam] = {
            "n": n, "fixtures": len({m for m, _o, _w in rows}),
            "win_rate": wins / n, "roi": roi, "avg_odds": avg_odds,
            "cost": cost, "cost_obs": len(cs),
            "taken_cost": taken, "taken_obs": len(ts),
            "taken_required_edge": (taken / (1.0 + taken)) if taken else None,
            # THE REQUIRED EDGE. Derived, not chosen: a bettor with NO edge
            # pays the margin, so their ROI is 1/(1+cost) - 1. The edge
            # required to break even is therefore cost/(1+cost) in ROI
            # points -- directly comparable to the +3% ROI target.
            "required_edge": (cost / (1.0 + cost)) if cost is not None else None,
            "ci95": _bootstrap_roi(rows),
        }
    return out


def _bootstrap_roi(rows: List[Tuple[int, float, int]]) -> Optional[list]:
    """95% CI on ROI, RESAMPLING FIXTURES. Registered method.

    Picks on one fixture are not independent -- the per-match cap allows two,
    and they share the same match state -- so the resampling unit is the
    fixture, not the pick.
    """
    by_fix: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
    for mid, o, w in rows:
        by_fix[mid].append((o, w))
    keys = list(by_fix)
    if len(keys) < 5:
        return None
    rng = random.Random(20261106)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        tot = cnt = 0.0
        for _ in range(len(keys)):
            for o, w in by_fix[keys[rng.randrange(len(keys))]]:
                tot += o * w - 1.0
                cnt += 1
        draws.append(tot / cnt)
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[int(0.975 * len(draws)) - 1]
    return [lo, hi]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        import json
        print(json.dumps(r, indent=1, default=str))
        return 0

    print("MARKET MIX — PART A, the cost of each market")
    print(f"  population : {r['population']} settled paper picks "
          f"(pick_date >= 2026-08-11, disposition IS NULL, result in "
          f"win/loss)")
    print(f"  joined     : {r['joined']} canonical-to-canonical "
          f"({100.0 * r['joined'] / max(1, r['population']):.1f}%)")
    print(f"  unnameable : {r['unnameable']}")
    print(f"  PRIOR      : {PRIOR_683['population']} picks — "
          f"{PRIOR_683['note']}")
    print()
    hdr = (f"  {'market':15s} {'n':>5s} {'win%':>7s} {'ROI':>8s} "
           f"{'odds':>6s} | {'TAKEN cost':>11s} {'req.edge':>9s} | "
           f"{'per-book':>9s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for fam, m in r["markets"].items():
        tk = (f"{100 * m['taken_cost']:.2f}%"
              if m["taken_cost"] is not None else "  —")
        req = (f"{100 * m['taken_required_edge']:.2f}%"
               if m["taken_required_edge"] is not None else "  —")
        pb = f"{100 * m['cost']:.2f}%" if m["cost"] is not None else "  —"
        print(f"  {fam:15s} {m['n']:5d} {100 * m['win_rate']:6.1f}% "
              f"{100 * m['roi']:+7.2f}% {m['avg_odds']:6.2f} | "
              f"{tk:>11s} {req:>9s} | {pb:>9s}")
    print()
    print(f"  the registered cut is {100 * CHEAP_CUT:.1f}% — fixed in the "
          f"pre-registration before these numbers were read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
