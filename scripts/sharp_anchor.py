"""SHARP ANCHOR - a correct fair price, and selection by price dispersion.

Stage 30. Measurement only: adopts nothing, moves no threshold, touches no
registered statistic and no cohort.

    python -m scripts.sharp_anchor --control        # Part A: synthetic control
    python -m scripts.sharp_anchor --select         # Part A: outcome-FREE
                                                    #   estimator evidence
    python -m scripts.sharp_anchor --coverage       # Part A: anchor coverage
    python -m scripts.sharp_anchor --dev            # Part B: development
    python -m scripts.sharp_anchor --open-holdout   # Part B: THE SEAL. Once.

THE ORDER IS THE DESIGN. `--control`, `--select` and `--coverage` read NO
match result. `docs/sharp-anchor-preregistration.md` is committed after them
and before `--dev`; `--open-holdout` is run once, after `--dev`, and its
output is the registered test.

SAME-SNAPSHOT, ALWAYS. A fair price and the price it is compared against
come from the same feed, written within `SNAPSHOT_TOLERANCE_S` of each
other, strictly before kickoff. A price that did not exist at the instant is
not a price: the cross-time best line was arb-shaped 21.28% of the time.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.evaluation.devig import (ESTIMATORS, EXACT_FOR, GENERATORS)

#: Bands for the calibration control: the external study's, as before.
BANDS = ((0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
         (0.60, 0.70), (0.70, 0.80), (0.80, 0.90))

#: Margins the control is run at: Pinnacle-like and the taken-median level.
CONTROL_MARGINS = (0.035, 0.066)

#: A leg set and the price compared against it are ONE snapshot when every
#: row was written within this many seconds. Measured 2026-11-12: within a
#: feed 97.8-99.8% of pairs sit inside 120s (median 0.0s); ACROSS feeds the
#: median is 255s and p90 is 8.1 hours.
SNAPSHOT_TOLERANCE_S = 120.0

#: The anchor, per feed. Never compared across feeds: that is two instants.
ANCHORS = {"apifootball": "Pinnacle", "theodds": "TheOddsAPI-pinnacle"}

#: Near-margin-free venues, used ONLY as an outcome-free truth proxy in
#: `--select`. Their quoted overround is ~1%, so every estimator agrees on
#: their fair price to within a few hundredths of a point.
EXCHANGES = ("TheOddsAPI-matchbook", "TheOddsAPI-betfair_ex_eu")

#: PRE-REGISTERED in docs/sharp-anchor-preregistration.md. Not tunable here.
EDGE_THRESHOLDS = (0.02, 0.03, 0.05)
PRIMARY_THRESHOLD = 0.03
DEV_FROM = _dt.date(2026, 2, 28)
HOLDOUT_FROM = _dt.date(2026, 9, 1)       # first HOLDOUT kickoff date
HOLDOUT_UNTIL = _dt.date(2026, 11, 10)    # exclusive
PRIMARY_ESTIMATOR = "shin"
TARGET_ODDS_RANGE = (1.50, 2.20)
BOOTSTRAP_DRAWS = 4000

EXCLUDE_REASONS = ("corrupt_1x2_assignment", "corrupt_team_identity",
                   "phantom_kickoff_now_stamp")


def _band_of(p: float) -> Optional[str]:
    for lo, hi in BANDS:
        if lo <= p < hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


# ══ PART A.1 — THE SYNTHETIC CONTROL ═════════════════════════════════════

def _true_sets(arity: int) -> List[List[float]]:
    """A grid of TRUE probability vectors. Deterministic, no sampling."""
    out = []
    if arity == 2:
        for i in range(5, 96):
            p = i / 100.0
            out.append([p, 1.0 - p])
    else:
        # 1X2-shaped: a draw between 0.18 and 0.32, the rest split
        for d in (0.18, 0.22, 0.26, 0.30, 0.32):
            for i in range(5, 96):
                h = (1.0 - d) * i / 100.0
                a = 1.0 - d - h
                if a > 0.02 and h > 0.02:
                    out.append([h, d, a])
    return out


def control() -> dict:
    """Calibration gap per band, for every (generating model, estimator).

    The gap is `mean(p_true - p_estimated)` over legs whose ESTIMATED
    probability falls in the band -- the same statistic the band test read --
    computed exactly on a perfectly calibrated book. No outcome is simulated:
    with a calibrated book the expected observed rate IS `p_true`.
    """
    out: Dict[str, dict] = {}
    for arity in (2, 3):
        for margin in CONTROL_MARGINS:
            for gname, gen in GENERATORS.items():
                for ename, est in ESTIMATORS.items():
                    per_band: Dict[str, List[float]] = defaultdict(list)
                    refused = 0
                    worst = 0.0
                    for p in _true_sets(arity):
                        q = gen(p, margin)
                        e = est(q)
                        if e is None:
                            refused += 1
                            continue
                        for pt, pe in zip(p, e):
                            b = _band_of(pe)
                            if b:
                                per_band[b].append(100.0 * (pt - pe))
                                worst = max(worst, abs(100.0 * (pt - pe)))
                    key = f"{arity}way|M={margin}|{gname}|{ename}"
                    out[key] = {
                        "arity": arity, "margin": margin,
                        "generator": gname, "estimator": ename,
                        "gap_pp": {b: statistics.mean(v)
                                   for b, v in sorted(per_band.items())},
                        "max_abs_gap_pp": worst,
                        "refused": refused,
                        "exact_expected": EXACT_FOR[gname] == ename,
                    }
    return out


def _print_control(c: dict) -> None:
    print("=" * 78)
    print("FAIR PRICE  -  Part A.1: THE SYNTHETIC CONTROL")
    print("=" * 78)
    print("  A PERFECTLY CALIBRATED book, quoted under four margin models,")
    print("  de-vigged by four estimators. The figure is the worst absolute")
    print("  calibration gap in any band, in probability points. ZERO means")
    print("  the estimator recovers the truth under that margin model.")
    for arity in (2, 3):
        for margin in CONTROL_MARGINS:
            print()
            print(f"  {arity}-way leg sets, margin {100 * margin:.1f}%   "
                  f"(rows: how the book quotes; columns: the estimator)")
            hdr = f"    {'book quotes':16s}" + "".join(
                f"{e:>14s}" for e in ESTIMATORS)
            print(hdr)
            print("    " + "-" * (len(hdr) - 4))
            for g in GENERATORS:
                cells = []
                for e in ESTIMATORS:
                    v = c[f"{arity}way|M={margin}|{g}|{e}"]
                    mark = "*" if v["exact_expected"] else " "
                    cells.append(f"{v['max_abs_gap_pp']:12.4f}{mark} ")
                print(f"    {g:16s}" + "".join(cells))
    print()
    print("  * = the estimator that inverts that margin model; it must be 0.")
    print()
    print("  THE ADDITIVE-BOOK ROW, per band (the control the band test")
    print("  lacked), 3-way at 6.6% -- signed gap in pp:")
    hdr = f"    {'estimator':14s}" + "".join(
        f"{f'{lo:.2f}-{hi:.2f}':>11s}" for lo, hi in BANDS)
    print(hdr)
    for e in ESTIMATORS:
        v = c[f"3way|M=0.066|additive|{e}"]["gap_pp"]
        print(f"    {e:14s}" + "".join(
            f"{v.get(f'{lo:.2f}-{hi:.2f}', float('nan')):+11.3f}"
            for lo, hi in BANDS))
    print("  and 2-way at 6.6%:")
    for e in ESTIMATORS:
        v = c[f"2way|M=0.066|additive|{e}"]["gap_pp"]
        print(f"    {e:14s}" + "".join(
            f"{v.get(f'{lo:.2f}-{hi:.2f}', float('nan')):+11.3f}"
            for lo, hi in BANDS))


# ══ THE LOADER — and the seal lives here ═════════════════════════════════

def _load_env() -> None:
    """INSIDE a function. Reads `.env` and `.env.local` from the repo root."""
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


def _feed_of(bookmaker: str) -> str:
    return "theodds" if bookmaker.startswith("TheOddsAPI-") else "apifootball"


def load(with_results: bool, date_from=None, date_until=None) -> dict:
    """Leg sets per (match, book, market, line, side), PRE-KICKOFF only.

    `with_results=False` DOES NOT SELECT THE SCORE COLUMNS AT ALL. The
    outcome-free steps cannot read a result by accident, because the query
    they run never asks for one.

    `date_from` / `date_until` bound the KICKOFF date. The development and
    holdout periods are disjoint ranges of it, and the holdout range is only
    ever passed by `--open-holdout`.
    """
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
    from src.data.market_spec import get_spec

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"
    score = ", m.home_goals, m.away_goals" if with_results else ""
    where = ["m.league NOT LIKE 'world/%'"]
    params = {}
    if date_from is not None:
        where.append("m.match_date >= :lo")
        params["lo"] = _dt.datetime.combine(date_from, _dt.time.min)
    if date_until is not None:
        where.append("m.match_date < :hi")
        params["hi"] = _dt.datetime.combine(date_until, _dt.time.min)
    if with_results:
        where.append("m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL")
    with db.get_session() as s:
        ms = s.execute(text(f"""
            SELECT m.id, m.match_date, m.league, m.training_exclusion_reason
                   {score}
            FROM matches m WHERE {' AND '.join(where)}
              AND EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id)
        """), params).fetchall()
        matches = {}
        excluded = defaultdict(int)
        for row in ms:
            if row[3] in EXCLUDE_REASONS:
                excluded[row[3]] += 1
                continue
            if row[1] is None or row[1].microsecond != 0 or row[1].hour < 6:
                excluded["placeholder_or_phantom_kickoff"] += 1
                continue
            matches[row[0]] = {
                "kickoff": row[1], "league": row[2],
                "hg": int(row[4]) if with_results else None,
                "ag": int(row[5]) if with_results else None,
            }
        mids = sorted(matches)
        rows = s.execute(text("""
            SELECT match_id, bookmaker, market_type, selection, odds_value,
                   timestamp
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
              AND bookmaker IS NOT NULL AND timestamp IS NOT NULL
        """), {"ids": mids}).fetchall()

    raw: Dict[Tuple[int, str, str], Dict[str, Tuple[float, object]]] = \
        defaultdict(dict)
    after_kickoff = 0
    for mid, bm, mt, sel, val, ts in rows:
        # A PRICE WRITTEN AT OR AFTER KICKOFF IS AN IN-PLAY PRICE. It knows
        # the score. 19,707 rows of the record are in that state.
        if ts >= matches[mid]["kickoff"]:
            after_kickoff += 1
            continue
        raw[(mid, bm, mt)][sel] = (float(val), ts)

    sets: Dict[Tuple[int, str, str, Optional[str], Optional[str]], dict] = {}
    refused = defaultdict(int)
    for (mid, bm, market), by_sel in raw.items():
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
                legs = []
                for aliases in spec.legs:
                    found = None
                    for alias in aliases:
                        if market == "over_under" and line:
                            key = f"{alias} {line}"
                        elif market == "team_goals" and line and side:
                            key = f"{side} {alias} {line}"
                        else:
                            key = alias
                        if key in by_sel:
                            found = (key,) + by_sel[key]
                            break
                    if found is None:
                        legs = None
                        break
                    legs.append(found)
                if not legs:
                    continue
                prices = [pr for _k, pr, _t in legs]
                orr = sum(1.0 / x for x in prices)
                if not (spec.overround_min <= orr <= spec.overround_max):
                    refused["overround_out_of_band"] += 1
                    continue
                tss = [t for _k, _p, t in legs]
                if (max(tss) - min(tss)).total_seconds() \
                        > SNAPSHOT_TOLERANCE_S:
                    refused["legs_not_one_snapshot"] += 1
                    continue
                sets[(mid, bm, market, line, side)] = {
                    "keys": [k for k, _p, _t in legs],
                    "prices": prices,
                    "q": [1.0 / x for x in prices],
                    "ts": max(tss), "ts_min": min(tss),
                    "overround": orr,
                }
    return {"matches": matches, "sets": sets,
            "excluded": dict(excluded), "refused": dict(refused),
            "rows_after_kickoff": after_kickoff}


def _same_snapshot(a: dict, b: dict) -> bool:
    lo = min(a["ts_min"], b["ts_min"])
    hi = max(a["ts"], b["ts"])
    return (hi - lo).total_seconds() <= SNAPSHOT_TOLERANCE_S


def _boot_mean(by_fix: Dict[int, List[float]], draws: int = 1000):
    keys = list(by_fix)
    if len(keys) < 30:
        return None
    rng = random.Random(20261112)
    out = []
    for _ in range(draws):
        vals: List[float] = []
        for _k in keys:
            vals.extend(by_fix[keys[rng.randrange(len(keys))]])
        out.append(statistics.mean(vals))
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


# ══ PART A.2 — WHICH MARGIN MODEL DO REAL BOOKS FOLLOW? (outcome-free) ══

def select() -> dict:
    """Each estimator on the ANCHOR's quotes, against the exchange price.

    NO RESULT IS READ. The truth proxy is the exchange: its quoted overround
    is ~1%, so its fair price is nearly estimator-independent, and the
    exchange set must be the SAME SNAPSHOT as the anchor set. The deviation
    `exchange_fair - estimator(anchor)` per band is then a property of how
    the anchor loads its margin, measured without an outcome.
    """
    d = load(with_results=False)
    sets = d["sets"]
    out = {}
    for feed, anchor in ANCHORS.items():
        dev: Dict[str, Dict[str, Dict[int, List[float]]]] = {
            e: defaultdict(lambda: defaultdict(list)) for e in ESTIMATORS}
        n_pairs = 0
        ex_margin: List[float] = []
        an_margin: List[float] = []
        by_arity = defaultdict(int)
        for (mid, bm, market, line, side), a in sets.items():
            if bm != anchor:
                continue
            for ex in EXCHANGES:
                x = sets.get((mid, ex, market, line, side))
                if x is None or not _same_snapshot(a, x):
                    continue
                n_pairs += 1
                by_arity[len(a["q"])] += 1
                ex_margin.append(x["overround"] - 1.0)
                an_margin.append(a["overround"] - 1.0)
                # the exchange's fair price: the MEAN of its proportional
                # and additive de-vigs, which bracket every other estimator
                xs = sum(x["q"])
                share = (xs - 1.0) / len(x["q"])
                proxy = [0.5 * (q / xs) + 0.5 * (q - share) for q in x["q"]]
                for ename, est in ESTIMATORS.items():
                    e = est(a["q"])
                    if e is None:
                        continue
                    for pe, px in zip(e, proxy):
                        b = _band_of(pe)
                        if b:
                            dev[ename][b][mid].append(100.0 * (px - pe))
                break
        res = {}
        for ename in ESTIMATORS:
            bands = {}
            tot_abs, tot_n = 0.0, 0
            for b in [f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS]:
                by_fix = dev[ename].get(b) or {}
                vals = [v for vs in by_fix.values() for v in vs]
                if not vals:
                    continue
                m = statistics.mean(vals)
                bands[b] = {"n": len(vals), "fixtures": len(by_fix),
                            "mean_dev_pp": m, "ci95": _boot_mean(by_fix)}
                tot_abs += abs(m) * len(vals)
                tot_n += len(vals)
            res[ename] = {"bands": bands,
                          "weighted_abs_dev_pp": (tot_abs / tot_n)
                          if tot_n else None}
        out[anchor] = {
            "feed": feed, "pairs": n_pairs, "by_arity": dict(by_arity),
            "exchange_margin_median": statistics.median(ex_margin)
            if ex_margin else None,
            "anchor_margin_median": statistics.median(an_margin)
            if an_margin else None,
            "estimators": res,
        }
    return out


def _print_select(r: dict) -> None:
    print("=" * 78)
    print("FAIR PRICE  -  Part A.2: WHICH MARGIN MODEL? (NO RESULT READ)")
    print("=" * 78)
    print("  Each estimator applied to the ANCHOR's quotes, compared with the")
    print("  same-snapshot EXCHANGE price (~1% overround, so its fair price")
    print("  is nearly estimator-free). deviation = exchange - estimator, pp.")
    for anchor, v in r.items():
        print()
        print(f"  ANCHOR {anchor}  ({v['feed']})   same-snapshot pairs: "
              f"{v['pairs']}   by arity {v['by_arity']}")
        if not v["pairs"]:
            print("      UNEVALUATED: no same-snapshot exchange pair exists "
                  "for this feed")
            continue
        print(f"      anchor margin median "
              f"{100 * v['anchor_margin_median']:.2f}%   exchange margin "
              f"median {100 * v['exchange_margin_median']:.2f}%")
        hdr = f"      {'estimator':13s}" + "".join(
            f"{f'{lo:.2f}-{hi:.2f}':>11s}" for lo, hi in BANDS) \
            + f"{'wtd |dev|':>11s}"
        print(hdr)
        for e, ev in v["estimators"].items():
            cells = "".join(
                f"{ev['bands'][b]['mean_dev_pp']:+11.3f}"
                if b in ev["bands"] else f"{'-':>11s}"
                for b in [f"{lo:.2f}-{hi:.2f}" for lo, hi in BANDS])
            w = ev["weighted_abs_dev_pp"]
            print(f"      {e:13s}{cells}"
                  f"{'-' if w is None else f'{w:11.3f}'}")
        any_e = next(iter(v["estimators"].values()))
        print("      n per band: " + "  ".join(
            f"{b}:{bv['n']}" for b, bv in any_e["bands"].items()))


# ══ PART A.3 — ANCHOR COVERAGE ═══════════════════════════════════════════

def coverage() -> dict:
    d = load(with_results=False)
    matches, sets = d["matches"], d["sets"]
    have: Dict[str, set] = defaultdict(set)
    for (mid, bm, market, line, side) in sets:
        if bm in ANCHORS.values():
            have[market].add(mid)
            have["ANY"].add(mid)
            have[f"feed:{_feed_of(bm)}"].add(mid)
    by_league_tot = defaultdict(int)
    by_league_cov = defaultdict(int)
    for mid, m in matches.items():
        by_league_tot[m["league"]] += 1
        if mid in have["ANY"]:
            by_league_cov[m["league"]] += 1
    return {
        "fixtures_with_prekickoff_odds": len(matches),
        "per_market": {k: len(v) for k, v in sorted(have.items())},
        "per_league": {lg: (by_league_cov[lg], by_league_tot[lg])
                       for lg in sorted(by_league_tot,
                                        key=lambda x: -by_league_tot[x])},
        "excluded": d["excluded"], "refused": d["refused"],
        "rows_after_kickoff": d["rows_after_kickoff"],
    }


def _print_coverage(c: dict) -> None:
    n = c["fixtures_with_prekickoff_odds"]
    print("=" * 78)
    print("FAIR PRICE  -  Part A.3: ANCHOR COVERAGE (no result read)")
    print("=" * 78)
    print(f"  fixtures with any odds row, kickoff readable: {n}")
    print(f"  excluded: {c['excluded']}")
    print(f"  leg sets refused: {c['refused']};  rows written at/after "
          f"kickoff and dropped: {c['rows_after_kickoff']}")
    print()
    for k, v in c["per_market"].items():
        print(f"    {k:22s} {v:6d}  ({100 * v / n:5.1f}% of fixtures)")
    print()
    print("  per league (anchor present / fixtures):")
    for lg, (cov, tot) in c["per_league"].items():
        print(f"    {str(lg)[:34]:34s} {cov:5d} / {tot:5d}  "
              f"({100 * cov / tot:5.1f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    if a.control:
        c = control()
        if a.json:
            print(json.dumps(c, indent=1))
        else:
            _print_control(c)
        return 0
    if a.select:
        r = select()
        if a.json:
            print(json.dumps(r, indent=1, default=str))
        else:
            _print_select(r)
        return 0
    if a.coverage:
        c = coverage()
        if a.json:
            print(json.dumps(c, indent=1, default=str))
        else:
            _print_coverage(c)
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
