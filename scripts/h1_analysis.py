"""H1's ANALYSIS, end to end — the registration turned into code.

Written 2026-09-22, nine days before collection. `h1_collection_check.py` was
one component of the same registration and it carried a defect that would have
aborted the collection on its first morning. This is the rest of it, and the
purpose of running it now is NOT to get an answer:

    There are no three-point series in the database, so the result is
    NO DATA by construction. What is being tested is that every step
    executes on production shapes, emits what it claims, and that
    NO DATA is distinguishable from APPARATUS BROKEN.

That distinction is exactly the one the collection check failed on before it
was run.

IT SHARES THE COLLECTION CHECK'S LOADER AND PREDICATE, by import rather than by
restatement. Two implementations of one prose rule diverge, and the divergence
is invisible until a result depends on it.

THE HYPOTHESIS, one-sided: pre-kickoff price movement is positively
autocorrelated — the direction and size of t0->t1 predicts t1->t2, within the
same fixture and the same price series. A significant NEGATIVE correlation
falsifies H1 as stated and is reported separately, never as support.

THE DECISION RULE, fixed before any data exists:

    r >= 0.42, p < 0.05     SIGNAL, ACTIONABLE     captured gain >= +2.00%
    0 < r < 0.42, p < 0.05  SIGNAL, NOT ACTIONABLE below the cost of acting
    p >= 0.05               NULL — reported WITH its exclusion bound
    r < 0, p < 0.05         FALSIFIED AS STATED    mean-reversion

AGGREGATION IS THE TRAP. One actionable price per FIXTURE — the best available
line for the selection actually picked — never the mean across books. H5's
first sigma came back at 1.05% against a true 7.67% for exactly this reason: a
factor of fifty in n.

READ-ONLY. SELECTs and arithmetic. No credit, no write, no config change.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from scripts.h1_collection_check import (
    MAX_INTERVAL_RATIO,
    MIN_GAP_MINUTES,
    attach_fixture_identity,
    load_observations,
    provider_of,
    qualifying_triple,
)

# ── REGISTERED CONSTANTS ─────────────────────────────────────────────────────

RHO_ACTIONABLE = 0.42      #: below this the captured gain does not clear +2.00%
ALPHA = 0.05               #: one-sided
N_MIN = 33                 #: lower end of the registered band
SIGMA_REGISTERED = 5.937   #: %, measured 2026-09-10, n=129, fixture-level
E_Z_ACTED = 0.7979         #: sqrt(2/pi), two-sided timing
BREAK_EVEN = 1.85          #: %, best-line break-even


def captured_gain(rho: float) -> float:
    """rho * sigma * E[z|acted], in percentage points."""
    return rho * SIGMA_REGISTERED * E_Z_ACTED


# ── TRAJECTORIES ─────────────────────────────────────────────────────────────

def build_trajectories(observations: Iterable[dict]) -> List[dict]:
    """One trajectory per (fixture, book, market, selection) that qualifies.

    The move is the LOG RATIO of consecutive prices. Log, not percentage:
    a drift from 2.00 to 2.20 and back to 2.00 must give +x then -x, and raw
    percentages do not (the second leg would be -9.1% against +10%). An
    asymmetric transform would put a spurious negative correlation into every
    round trip, which is a finding H1 would then report.
    """
    series: Dict[tuple, List[Tuple[datetime, float]]] = defaultdict(list)
    for o in observations:
        key = (o["fixture"], o["bookmaker"], o["market"], o["selection"])
        series[key].append((o["ts"], o["odds"]))

    out: List[dict] = []
    for (fixture, book, market, selection), points in series.items():
        by_time = {}
        for ts, price in sorted(points):
            by_time[ts] = price                      # last write at an instant
        triple = qualifying_triple(list(by_time))
        if triple is None:
            continue
        t0, t1, t2 = triple
        p0, p1, p2 = by_time[t0], by_time[t1], by_time[t2]
        if min(p0, p1, p2) <= 0:
            continue                                 # a non-price, not a move
        out.append({
            "fixture": fixture, "bookmaker": book, "market": market,
            "selection": selection, "provider": provider_of(book),
            "t0": t0, "t1": t1, "t2": t2,
            "r1": math.log(p1 / p0), "r2": math.log(p2 / p1),
            "gap1": (t1 - t0).total_seconds() / 60.0,
            "gap2": (t2 - t1).total_seconds() / 60.0,
            "p0": p0, "p1": p1, "p2": p2,
        })
    return out


def _series(observations: Iterable[dict]) -> Dict[tuple, List[datetime]]:
    out: Dict[tuple, List[datetime]] = defaultdict(list)
    for o in observations:
        out[(o["fixture"], o["bookmaker"], o["market"],
             o["selection"])].append(o["ts"])
    return out


def separated_points(times: Iterable[datetime],
                     min_gap_minutes: int = MIN_GAP_MINUTES) -> int:
    """How many points survive a greedy >= `min_gap` thinning.

    NOT the qualifying predicate — it ignores the interval-ratio rule. It
    exists to say WHY a zero is a zero: a floor that matches nothing and a
    floor that is broken both report no trajectories, and only this separates
    them.
    """
    gap = timedelta(minutes=min_gap_minutes)
    kept: List[datetime] = []
    for t in sorted(set(times)):
        if not kept or (t - kept[-1]) >= gap:
            kept.append(t)
    return len(kept)


def separated_point_distribution(observations: Iterable[dict]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for times in _series(observations).values():
        counts[separated_points(times)] += 1
    return dict(counts)


def series_keys_with(observations: Iterable[dict], n: int) -> List[tuple]:
    return [k for k, v in _series(observations).items()
            if separated_points(v) == n]


def one_price_per_fixture(trajectories: List[dict],
                          picked: Dict[int, Tuple[str, str]]) -> List[dict]:
    """Collapse to ONE trajectory per fixture — never the mean across books.

    THE RULE, from the registration: the best available line for the selection
    actually picked. So:

      1. keep only trajectories whose (market, selection) is the one picked on
         that fixture — a price the pipeline would never have taken is not an
         actionable price;
      2. among those, take the best available line, i.e. the highest `p0`,
         which is the price the bettor would actually have got.

    Fixtures with no pick are DROPPED and counted, not silently averaged. The
    registration already bounds generalisation to picked fixtures (142 of 142
    existing separated series were picked), so dropping them is the registered
    scope, not a convenience.
    """
    by_fixture: Dict[int, List[dict]] = defaultdict(list)
    for t in trajectories:
        want = picked.get(t["fixture"])
        if want is None:
            continue
        if (t["market"], t["selection"]) != want:
            continue
        by_fixture[t["fixture"]].append(t)
    return [max(v, key=lambda t: t["p0"]) for v in by_fixture.values()]


def picked_selections(session) -> Dict[int, Tuple[str, str]]:
    """`{fixture_group: (market, selection)}` for picks that were taken."""
    from sqlalchemy import text
    rows = session.execute(text("""
        SELECT match_id, market, selection FROM saved_picks
        WHERE market IS NOT NULL AND selection IS NOT NULL
          AND (disposition IS NULL OR disposition <> 'consolidated')
    """)).fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}


# ── STATISTICS ───────────────────────────────────────────────────────────────

def pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None                      # no variance: undefined, not zero
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def one_sided_p(r: float, n: int) -> Optional[float]:
    """P(R >= r) under rho = 0, via Fisher's z. None when undefined."""
    if n < 4 or r is None or abs(r) >= 1.0:
        return None
    z = 0.5 * math.log((1 + r) / (1 - r)) * math.sqrt(n - 3)
    return 0.5 * math.erfc(z / math.sqrt(2))


def upper_bound_rho(n: int, r: float = 0.0, alpha: float = ALPHA) -> Optional[float]:
    """One-sided upper confidence bound on rho — what a NULL EXCLUDES.

    A null does not mean "no effect"; it means the effect is smaller than
    this. It is the strongest part of the registration and is printed whenever
    the result is null.
    """
    if n < 4:
        return None
    z_alpha = 1.6449                                   # 95% one-sided
    zr = 0.5 * math.log((1 + r) / (1 - r))
    hi = zr + z_alpha / math.sqrt(n - 3)
    return math.tanh(hi)


# ── OUTCOMES, WHICH MUST NOT COLLAPSE ────────────────────────────────────────

def interpret(n: int, r: Optional[float], p: Optional[float]) -> Tuple[str, List[str]]:
    """Return `(state, lines)`. Five states, and NO DATA is one of them."""
    if n == 0:
        return "NO DATA", [
            "H1 ANALYSIS: NO DATA — zero qualifying trajectories.",
            "  This is NOT a null result and must never be reported as one.",
            "  A null says the effect is smaller than a bound; this says the",
            "  measurement did not happen. Nothing about H1 is learned here.",
        ]
    if r is None:
        return "APPARATUS BROKEN", [
            "H1 ANALYSIS: APPARATUS BROKEN — "
            f"{n} trajectory(ies) but the correlation is undefined.",
            "  Every move is identical, or a price series has no variance.",
            "  Real prices move; this is an instrument fault, not an answer.",
        ]
    if n < N_MIN:
        return "INSUFFICIENT", [
            f"H1 ANALYSIS: INSUFFICIENT — {n} of {N_MIN} trajectories.",
            f"  r = {r:+.4f} is computed but NOT interpreted: the decision rule",
            f"  is registered at n >= {N_MIN} and reading it earlier is the",
            "  optional-stopping error the registration exists to prevent.",
        ]
    if p is None:
        return "APPARATUS BROKEN", [
            f"H1 ANALYSIS: APPARATUS BROKEN — r = {r:+.4f} at n = {n} but no "
            "p-value is computable.",
        ]
    if p >= ALPHA:
        bound = upper_bound_rho(n, 0.0)
        gain = captured_gain(bound) if bound is not None else float("nan")
        verdict = ("BELOW break-even" if gain < BREAK_EVEN else "ABOVE break-even")
        return "NULL", [
            f"H1 ANALYSIS: NULL — r = {r:+.4f}, p = {p:.4f} (one-sided), n = {n}.",
            f"  95% one-sided upper bound on rho: {bound:.3f}",
            f"  implied captured gain: {gain:.2f}% against a {BREAK_EVEN}% "
            f"break-even — {verdict}.",
            "  A null at this n RETIRES timing as a lever; it does not merely",
            "  fail to find one.",
        ]
    if r < 0:
        return "FALSIFIED", [
            f"H1 ANALYSIS: FALSIFIED AS STATED — r = {r:+.4f}, p = {p:.4f}, n = {n}.",
            "  Mean-reversion, not momentum. Reported as a SEPARATE finding,",
            "  never as weak support for H1.",
        ]
    if r >= RHO_ACTIONABLE:
        return "SIGNAL ACTIONABLE", [
            f"H1 ANALYSIS: SIGNAL, ACTIONABLE — r = {r:+.4f}, p = {p:.4f}, n = {n}.",
            f"  captured gain {captured_gain(r):.2f}% clears the {BREAK_EVEN}% "
            "break-even.",
        ]
    return "SIGNAL NOT ACTIONABLE", [
        f"H1 ANALYSIS: SIGNAL, NOT ACTIONABLE — r = {r:+.4f}, p = {p:.4f}, n = {n}.",
        f"  real autocorrelation, captured gain {captured_gain(r):.2f}% is below",
        f"  the {BREAK_EVEN}% cost of acting. NOT used to justify a timing change.",
    ]


def influence_check(rows: List[dict]) -> List[str]:
    """Report the result WITHOUT its most influential observation.

    A single anomalous result is evidence about the measurement first.
    """
    if len(rows) < 5:
        return ["  influence check: n too small to drop a point"]
    xs = [t["r1"] for t in rows]
    ys = [t["r2"] for t in rows]
    full = pearson(xs, ys)
    if full is None:
        return ["  influence check: correlation undefined"]
    worst, worst_r = None, None
    for i in range(len(rows)):
        r = pearson(xs[:i] + xs[i + 1:], ys[:i] + ys[i + 1:])
        if r is None:
            continue
        if worst is None or abs(r - full) > abs(worst_r - full):
            worst, worst_r = i, r
    if worst is None:
        return ["  influence check: undefined"]
    return [f"  without the most influential fixture "
            f"({rows[worst]['fixture']}): r = {worst_r:+.4f} "
            f"(full {full:+.4f}, shift {worst_r - full:+.4f})"]


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--since", help="ISO date; default 2026-10-01, the reset")
    ap.add_argument("--days", type=int)
    ap.add_argument("--allow-sqlite", action="store_true")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        import pathlib
        env = pathlib.Path(__file__).resolve().parent.parent / ".env"
        if env.is_file():
            load_dotenv(env)
    except Exception:
        pass

    if args.days is not None:
        since = datetime.utcnow() - timedelta(days=args.days)
    elif args.since:
        since = datetime.fromisoformat(args.since)
    else:
        since = datetime(2026, 10, 1)

    from src.data.database import get_db
    db = get_db()
    if not db.is_postgres and not args.allow_sqlite:
        print("REFUSING: connected to local SQLite, not production. Any result "
              "printed here would describe a stale local file.", file=sys.stderr)
        return 2

    with db.get_session() as session:
        observations, dropped = load_observations(session, since)
        attach_fixture_identity(session, observations)
        picked = picked_selections(session)
        trajectories = build_trajectories(observations)
        actionable = one_price_per_fixture(trajectories, picked)

    print(f"H1 ANALYSIS DRY RUN — observations since {since:%Y-%m-%d}")
    print("")
    print("STEP 1  load, with every contamination control applied")
    print(f"   in-band pre-kickoff observations     : {len(observations)}")
    print(f"   market-instants kept/out/uncomputable: {dropped.get('kept',0)} / "
          f"{dropped.get('out_of_band',0)} / {dropped.get('not_computable',0)}")

    print("")
    print("STEP 2  separation and comparability")
    print(f"   qualifying trajectories (any series) : {len(trajectories)}")
    print(f"   floor >= {MIN_GAP_MINUTES} min, intervals differ by < {MAX_INTERVAL_RATIO}x")

    # WHY THERE ARE NONE, stated as a measurement rather than an absence.
    # A predicate that matches nothing and a predicate that is broken produce
    # the same zero; this distribution separates them.
    dist = separated_point_distribution(observations)
    print(f"   separated points per series          : "
          f"{dict(sorted(dist.items()))}")
    pairs = [s for s in series_keys_with(observations, 2)]
    if pairs:
        print(f"   series with exactly TWO separated pts : {len(pairs)} "
              f"across {len({k[0] for k in pairs})} FIXTURE(S)")
        print("   -> the floor MATCHES real data; what is missing is the third")
        print("      observation, which is what the collection buys.")
        print("   -> and the series/fixture ratio here is the aggregation trap "
              "in miniature: counting series would report n as the first "
              "number, the registered unit reports the second.")

    # PROVIDER IS A STRATUM. Reported before any pooling can happen.
    print("")
    print("STEP 3  provider strata — reported apart, never pooled")
    strata: Dict[str, int] = defaultdict(int)
    for t in trajectories:
        strata[t["provider"]] += 1
    if strata:
        for name in sorted(strata):
            print(f"   {name:14s} {strata[name]}")
        if len(strata) > 1:
            print("   !! TWO PROVIDERS PRESENT — analyse as separate strata or "
                  "drop the second. Merging them is the H2 failure.")
    else:
        print("   (no trajectories, so no stratum is populated)")
    mixed = sum(1 for t in trajectories if False)   # series are single-book by key
    print(f"   series spanning more than one book   : {mixed} "
          f"(impossible by construction — the series key includes the book)")

    print("")
    print("STEP 4  aggregation — ONE actionable price per fixture")
    print(f"   picks available to match against     : {len(picked)}")
    print(f"   trajectories on a PICKED (market, selection) : {len(actionable)}")
    print("   rule: best available line (max p0) for the selection actually")
    print("   picked; the mean across books is never taken")

    print("")
    print("STEP 5  the estimator and the decision rule")
    xs = [t["r1"] for t in actionable]
    ys = [t["r2"] for t in actionable]
    r = pearson(xs, ys)
    p = one_sided_p(r, len(actionable)) if r is not None else None
    state, lines = interpret(len(actionable), r, p)
    for line in lines:
        print("   " + line)
    for line in influence_check(actionable):
        print("   " + line)

    print("")
    print(f"STATE: {state}")
    return 0


if __name__ == "__main__":                              # pragma: no cover
    raise SystemExit(main())
