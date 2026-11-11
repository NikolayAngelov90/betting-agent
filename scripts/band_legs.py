"""BAND TESTED ON LEGS - the favourite-longshot test on prices, not on picks.

THE HYPOTHESIS IS ABOUT MARKET CALIBRATION. It does not need a pick, and it
did not need `min_odds` opened: its unit is a LEG -- one (fixture, market,
selection) with a de-vigged implied probability and a known result -- and
those exist in the odds record at ANY floor.

    H-BAND: within a de-vigged probability band, the observed outcome rate
            EXCEEDS the de-vigged implied probability.

The pick-level version is a SELECTED SUBSET of this, selected by a model
established to add nothing over the price. It is not a substitute.

WHAT THIS SCRIPT IS CAREFUL ABOUT, because each one would bias a band:

  ONE LEG PER (fixture, market, selection). Using every bookmaker's quote
    would multiply the same leg by the number of books and inflate n. The
    price is the TAKEN one -- the cross-book median per leg, which is what
    `_find_best_odds` actually returns and therefore what we would face.

  DE-VIGGED PER LEG SET, NEVER POOLED. `p_devig = p_implied / sum(p_implied)`
    over that set's own legs. A band assignment using a median margin
    misassigns the legs whose own set is wide, and the margin spread across
    books is 0.6pp to 3.9pp.

  BANDS FIXED AT THE EXTERNAL STUDY'S BOUNDARIES, read before our data.

  BOOTSTRAP RESAMPLES FIXTURES. One fixture contributes up to three
    correlated legs (a 1X2 set is three legs on one result), so resampling
    legs would treat them as independent and inflate n threefold.

    python -m scripts.band_legs
    python -m scripts.band_legs --json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

#: THE BANDS, fixed at the external study's boundaries. Read before our data.
#: Half-open on the right, as `docs/flb-preregistration.md` fixed them.
BANDS = ((0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
         (0.60, 0.70), (0.70, 0.80), (0.80, 0.90))

#: THE EXTERNAL TABLE, kept labelled and NOT re-derived.
#: `external, Footiqo, 2025/26, ONE bookmaker (1xBet), five leagues,
#: fetched 2026-11-04`. Their own limitations section says one season, five
#: leagues, one book.
EXTERNAL = {
    (0.40, 0.50): {"gap_pp": -1.34, "n": 424},
    (0.50, 0.60): {"gap_pp": +1.29, "n": 591},
    (0.60, 0.70): {"gap_pp": +3.21, "n": 266},
    (0.70, 0.80): {"gap_pp": +3.74, "n": 134},
    (0.80, 0.90): {"gap_pp": +6.32, "n": 32},
}
EXTERNAL_LABEL = ("external, Footiqo, 2025/26, ONE bookmaker (1xBet), five "
                  "leagues, fetched 2026-11-04")

#: `docs/flb-preregistration.md`'s own floor to TEST a band at all.
MIN_N_TO_TEST = 100
MIN_FIXTURES_TO_TEST = 50

#: Match-level exclusions, by `matches.training_exclusion_reason`. Each is a
#: registered corruption, not a judgement call:
#:   corrupt_1x2_assignment     the away price written into the Home slot
#:                              (28 matches, closed and marked at s5.16)
#:   corrupt_team_identity      a participant row belongs to another club
#:   phantom_kickoff_now_stamp  an ingestion stamp, not a kickoff
EXCLUDE_REASONS = ("corrupt_1x2_assignment", "corrupt_team_identity",
                   "phantom_kickoff_now_stamp")

BOOTSTRAP_DRAWS = 4000

#: WHAT THIS MEASURED. `measured` 2026-11-11, grain=LEG, unit = PROBABILITY
#: POINTS, bootstrap over FIXTURES, 4,000 draws.
MEASURED_LEGS = {
    "measured": "2026-11-11",
    "unit": "probability points",
    "grain": "LEG; bootstrap over FIXTURES",
    "n_legs": 52737,
    "matches_completed": 37691,
    "matches_kept": 37079,
    # band -> (n, mean_devig_pct, observed_pct, gap_pp, ci_lo, ci_hi, verdict)
    "bands": {
        "0.30-0.40": (8830, 35.15, 33.52, -1.63, -2.60, -0.71, "NEGATIVE"),
        "0.40-0.50": (11833, 45.12, 44.76, -0.35, -1.25, +0.56, "NOT PRESENT"),
        "0.50-0.60": (11379, 54.62, 55.76, +1.14, +0.19, +2.11, "PRESENT"),
        "0.60-0.70": (7606, 64.76, 66.39, +1.64, +0.55, +2.76, "PRESENT"),
        "0.70-0.80": (8927, 74.82, 76.25, +1.43, +0.49, +2.34, "PRESENT"),
        "0.80-0.90": (4162, 83.99, 86.38, +2.38, +1.21, +3.53, "PRESENT"),
    },
    #: THE IN-BAND COST, per venue, at the band's own price. pp.
    "in_band_cost_pp": {
        "0.50-0.60": {"TheOddsAPI-matchbook": 0.54, "Pinnacle": 2.28,
                      "Bet365": 4.14, "1xBet": 3.69, "TAKEN_policy": 3.71},
        "0.60-0.70": {"TheOddsAPI-matchbook": 0.63, "Pinnacle": 2.93,
                      "Bet365": 4.72, "1xBet": 5.23, "TAKEN_policy": 4.56},
        "0.70-0.80": {"TheOddsAPI-matchbook": 0.64, "Pinnacle": 3.87,
                      "Bet365": 4.90, "1xBet": 6.48, "TAKEN_policy": 5.01},
        "0.80-0.90": {"TheOddsAPI-matchbook": 0.74, "Pinnacle": 3.68,
                      "Bet365": 4.68, "1xBet": 7.58, "TAKEN_policy": 5.73},
    },
    "exchange_fixture_coverage": 0.061,
    "surviving_cell": {
        "band": "0.80-0.90",
        "venue": "exchange (matchbook or betfair_ex_eu)",
        "ci_lo_minus_cost_pp": 0.48,
        "breakeven_commission": 0.036,
        "net_at_2pct": 0.21,
        "net_at_6pct": -0.33,
        "why_not_a_plan": (
            "depth is UNAVAILABLE in the feed so the price is UNESTABLISHED; "
            "fixture coverage is 6.1%; Matchbook's 2% was read from their own "
            "domain but NOT directly fetched; and the band is raw odds ~1.19, "
            "BELOW the restored 1.50 floor -- a selection the deployed config "
            "refuses, by the operator's own decision of 2026-11-11"),
    },
    "verdict": (
        "H-BAND IS SUPPORTED AND NOT COLLECTABLE. Four consecutive bands "
        "PRESENT with monotone gaps and the lowest band NEGATIVE -- the "
        "favourite-longshot shape -- agreeing in DIRECTION with the external "
        "study in all five comparable bands at 20-130x its n, at roughly a "
        "third to a half of its magnitudes. But the net is NEGATIVE at every "
        "bookmaker in every band on BOTH the point estimate and the CI lower "
        "bound. One cell survives and it is at an exchange whose depth is "
        "unmeasurable, in a price band the deployed floor excludes"),
}

#: THE s5.20 PICK DISTRIBUTION. `measured` 2026-11-11, grain=PICK.
MEASURED_S520 = {
    "measured": "2026-11-11",
    "cohort": "s5.20",
    "fingerprint": "12f2c7",
    "n": 86,
    "span": ("2026-11-08", "2026-11-09"),
    "days": 2,
    "odds_min": 1.14, "odds_p25": 1.25, "odds_median": 1.440,
    "odds_mean": 1.464, "odds_max": 1.99,
    "below_1_15": 4, "below_1_18": 14, "below_1_20": 15, "below_1_50": 46,
    "band_counts": {"0.40-0.50": 1, "0.50-0.60": 33, "0.60-0.70": 23,
                    "0.70-0.80": 17, "0.80-0.90": 10},
    "UNEVALUATED": 2,
    "breakeven_win_rate_at_mean": 0.683,
    "settled_n": 85, "settled_win_rate": 0.659, "settled_roi": -0.0749,
    "pending": 1,
    "floor_was_an_attractor": False,
    "attractor_verdict": (
        "NO. Only 4 of 86 (4.7%) sat at the floor band below 1.15 and the "
        "median landed at 1.440, mid-way through the newly admitted range -- "
        "so the selection did not systematically take the shortest "
        "admissible price. But only 27 of 86 (31%) reached the 70-79 or "
        "80-90 bands where the anomaly is largest, and 33 went to 50-60 "
        "where it is smallest: the floor change moved the distribution "
        "DOWNWARD without aiming it at the band"),
    "caveat": (
        "the model is established to add nothing over the price, so its edge "
        "estimate inside the band is uninformative, and n=85 settled over "
        "two days supports no conclusion about ROI"),
}


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


def _band_of(p: float) -> Optional[tuple]:
    for lo, hi in BANDS:
        if lo <= p < hi:
            return (lo, hi)
    return None


def _bootstrap_gap(by_fixture: Dict[int, List[Tuple[float, int]]],
                   draws: int = BOOTSTRAP_DRAWS) -> Optional[list]:
    """95% CI on the gap, resampling FIXTURES.

    A fixture contributes up to three correlated legs -- a 1X2 set is three
    legs settled by one scoreline -- so a leg-level resample would treat them
    as independent and understate the interval.
    """
    keys = list(by_fixture)
    if len(keys) < 2:
        return None
    rng = random.Random(20261111)
    out = []
    for _ in range(draws):
        legs: List[Tuple[float, int]] = []
        for _ in keys:
            legs.extend(by_fixture[keys[rng.randrange(len(keys))]])
        if not legs:
            continue
        rate = sum(w for _p, w in legs) / len(legs)
        pred = sum(p for p, _w in legs) / len(legs)
        out.append(100.0 * (rate - pred))
    if not out:
        return None
    out.sort()
    return [out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1]]


def measure() -> dict:
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
    from src.data.market_spec import extract_legs, get_spec, overround
    from src.data.market_vocabulary import canonical
    # THE DEPLOYED GRADER, reused rather than reimplemented: a second
    # settlement table is how the Under 1.5/3.5 branches came to be
    # missing once and mis-settled 7 picks. `_grade_selection` is a
    # staticmethod, so this constructs no agent -- the constructor
    # issues DDL including a DROP and must not run here.
    from src.agent.betting_agent import FootballBettingAgent

    grade = FootballBettingAgent._grade_selection

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"

    with db.get_session() as s:
        # COMPLETED matches only: a leg needs a known result.
        matches = s.execute(text("""
            SELECT m.id, m.home_goals, m.away_goals, m.league,
                   m.training_exclusion_reason
            FROM matches m
            WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
              AND m.match_date >= '2023-01-01'
              AND m.league NOT LIKE 'world/%'
        """)).fetchall()
        kept = {m[0]: (int(m[1]), int(m[2]), m[3]) for m in matches
                if m[4] not in EXCLUDE_REASONS}
        excluded = {r: sum(1 for m in matches if m[4] == r)
                    for r in EXCLUDE_REASONS}
        mids = sorted(kept)
        odds_rows = s.execute(text("""
            SELECT match_id, bookmaker, market_type, selection, odds_value
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
              AND bookmaker IS NOT NULL
        """), {"ids": mids}).fetchall()

    # ── THE TAKEN PRICE: cross-book median per leg ───────────────────────
    quotes: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    per_book: Dict[Tuple[int, str, str], Dict[str, float]] = defaultdict(dict)
    for mid, bm, mt, sel, val in odds_rows:
        quotes[(mid, mt, sel)].append(float(val))
        per_book[(mid, bm, mt)][sel] = float(val)
    taken: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
    for (mid, mt, sel), vals in quotes.items():
        taken[(mid, mt)][sel] = statistics.median(vals)

    # ── EVERY LEG, DE-VIGGED AGAINST ITS OWN SET, GRADED ─────────────────
    legs: List[dict] = []
    refused = defaultdict(int)
    for (mid, market), by_sel in taken.items():
        spec = get_spec(market)
        if spec is None:
            refused["no_market_spec"] += 1
            continue
        if spec.overlapping:
            # double chance legs overlap and sum to ~2: `sum(p)` is not a
            # margin, so proportional de-vigging is undefined for them.
            refused["overlapping_market"] += 1
            continue
        lines = {None}
        if market in ("over_under", "team_goals"):
            lines = {m.group(1) for k in by_sel
                     for m in [re.search(r"(\d+\.\d+)", k)] if m}
        sides = {None} if market != "team_goals" else {"Home", "Away"}
        for line in lines:
            for side in sides:
                prices = extract_legs(market, by_sel, line=line, side=side)
                if not prices or len(prices) != spec.arity:
                    refused["incomplete_leg_set"] += 1
                    continue
                orr = overround(prices)
                # THE SPEC'S OWN BAND. A two-way price written into a
                # three-way slot lands far outside it -- the API-Football
                # "Home/Away" trap, 2,486 matches -- so this is the gate
                # that excludes it, and it is the existing one.
                if not (spec.overround_min <= orr <= spec.overround_max):
                    refused["overround_out_of_band"] += 1
                    continue
                # the SELECTION KEYS in the same order as `prices`
                keys = _keys_for(spec, market, by_sel, line, side)
                if len(keys) != len(prices):
                    refused["key_recovery_failed"] += 1
                    continue
                hg, ag, league = kept[mid]
                for key, price in zip(keys, prices):
                    canon = canonical(market, key)
                    if canon is None:
                        refused["not_canonicalisable"] += 1
                        continue
                    _fam, canon_sel = canon
                    res = grade(canon_sel, hg, ag)
                    if res not in ("win", "loss"):
                        refused["ungradeable_or_void"] += 1
                        continue
                    p_raw = 1.0 / price
                    legs.append({
                        "mid": mid, "market": market, "sel": canon_sel,
                        "league": league, "odds": price,
                        "p_raw": p_raw, "p_devig": p_raw / orr,
                        "margin": orr - 1.0,
                        "won": 1 if res == "win" else 0,
                    })

    # ── PER BAND ──────────────────────────────────────────────────────────
    per_band: Dict[tuple, List[dict]] = defaultdict(list)
    for L in legs:
        b = _band_of(L["p_devig"])
        if b:
            per_band[b].append(L)
    out_of_band = sum(1 for L in legs if _band_of(L["p_devig"]) is None)

    results = {}
    for band in BANDS:
        rows = per_band.get(band) or []
        fixtures = {L["mid"] for L in rows}
        if not rows:
            results[f"{band[0]:.2f}-{band[1]:.2f}"] = {
                "n": 0, "fixtures": 0, "verdict": "INSUFFICIENT"}
            continue
        pred = statistics.mean(L["p_devig"] for L in rows)
        rate = sum(L["won"] for L in rows) / len(rows)
        by_fix: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        for L in rows:
            by_fix[L["mid"]].append((L["p_devig"], L["won"]))
        testable = (len(rows) >= MIN_N_TO_TEST
                    and len(fixtures) >= MIN_FIXTURES_TO_TEST)
        ci = _bootstrap_gap(by_fix) if testable else None
        gap = 100.0 * (rate - pred)
        verdict = "INSUFFICIENT"
        if testable and ci:
            verdict = "PRESENT" if ci[0] > 0 else (
                "NEGATIVE" if ci[1] < 0 else "NOT PRESENT")
        results[f"{band[0]:.2f}-{band[1]:.2f}"] = {
            "n": len(rows), "fixtures": len(fixtures),
            "mean_devig": pred, "observed_rate": rate, "gap_pp": gap,
            "ci95": ci, "verdict": verdict,
            "mean_odds": statistics.mean(L["odds"] for L in rows),
            "mean_margin_rel": statistics.mean(L["margin"] for L in rows),
            "external": EXTERNAL.get(band),
        }

    # ══ PART B — IS IT COLLECTABLE? IN-BAND COST, PER VENUE ══════════════
    #
    # EXISTENCE AND COLLECTABILITY ARE TWO QUESTIONS. A gap against a
    # de-vigged price is not money: the margin has to be paid out of it.
    #
    # AND IT MUST BE THE IN-BAND COST, NOT THE OVERALL ONE. Bookmaker
    # margins RISE in the favourites band (+0.77 to +1.18pp measured
    # 2026-11-06) while both exchanges' are flat, so the cheapest venue
    # overall is not necessarily the cheapest venue in the band. Converting
    # at the population's average odds instead of the band's own flipped
    # Pinnacle's sign once already.
    cost: Dict[str, Dict[str, dict]] = defaultdict(dict)
    acc: Dict[Tuple[str, tuple], List[Tuple[float, float]]] = defaultdict(list)
    books_seen: Dict[str, set] = defaultdict(set)
    for (mid, bm, market), by_sel in per_book.items():
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
                if not prices or len(prices) != spec.arity:
                    continue
                orr = overround(prices)
                if not (spec.overround_min <= orr <= spec.overround_max):
                    continue
                for price in prices:
                    pv = (1.0 / price) / orr
                    band = _band_of(pv)
                    if band is None:
                        continue
                    # pp at THIS leg's own price, which is the band's price
                    pp = (1.0 / price) * ((orr - 1.0) / orr) * 100.0
                    acc[(bm, band)].append((pp, price))
                    books_seen[bm].add(mid)

    pop_fixtures = len(kept)
    for (bm, band), rows in acc.items():
        if len(rows) < 30:
            continue
        key = f"{band[0]:.2f}-{band[1]:.2f}"
        cost[key][bm] = {
            "cost_pp": statistics.median(pp for pp, _o in rows),
            "n_legs": len(rows),
            "median_odds": statistics.median(o for _pp, o in rows),
            "fixture_coverage": len(books_seen[bm]) / pop_fixtures,
        }

    # the TAKEN policy's own in-band cost, the same way
    taken_acc: Dict[tuple, List[float]] = defaultdict(list)
    for L in legs:
        band = _band_of(L["p_devig"])
        if band is None:
            continue
        orr = 1.0 + L["margin"]
        taken_acc[band].append(L["p_raw"] * ((orr - 1.0) / orr) * 100.0)
    for band, vals in taken_acc.items():
        if vals:
            cost[f"{band[0]:.2f}-{band[1]:.2f}"]["TAKEN_policy"] = {
                "cost_pp": statistics.median(vals), "n_legs": len(vals),
                "median_odds": None, "fixture_coverage": 1.0,
            }

    return {
        "measured": "2026-11-11",
        "unit": "LEG: one (fixture, market, selection) at the TAKEN price",
        "grain": "LEG, bootstrap over FIXTURES",
        "devig": "proportional, per leg set: p/sum(p) over that set's own legs",
        "bands": "fixed at the external study's boundaries, before our data",
        "n_matches_completed": len(matches),
        "n_matches_kept": len(kept),
        "excluded_matches": excluded,
        "n_legs": len(legs),
        "legs_out_of_every_band": out_of_band,
        "refused": dict(sorted(refused.items(), key=lambda kv: -kv[1])),
        "results": results,
        "external_label": EXTERNAL_LABEL,
        "in_band_cost": {k: v for k, v in sorted(cost.items())},
    }


def _keys_for(spec, market_type: str, by_sel: dict,
              line, side) -> list:
    """The selection keys `extract_legs` would have read, in its own order.

    RECONSTRUCTED THE SAME WAY, NOT MATCHED BY VALUE. `extract_legs` returns
    prices only, so the key each price came from has to be recovered -- and
    the first version of this matched the price back against the dict. That
    mis-grades a leg set whose prices TIE: `Over 2.5` and `Under 2.5` both at
    1.90 are one value, the two selections grade OPPOSITELY, and the first
    unused match would take the wrong one. The de-vigged probability is
    identical either way, so the band would be right and the RESULT wrong --
    the worst shape of error available here.

    So this mirrors `extract_legs` exactly: walk `spec.legs` in order, build
    the qualified key the same way, take the first alias present.
    """
    keys = []
    for aliases in spec.legs:
        found = None
        for alias in aliases:
            if market_type == "over_under" and line:
                key = f"{alias} {line}"
            elif market_type == "team_goals" and line and side:
                key = f"{side} {alias} {line}"
            else:
                key = alias
            v = by_sel.get(key)
            if v and v > 1.0:
                found = key
                break
        if found is None:
            return []
        keys.append(found)
    return keys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        print(json.dumps(r, indent=1, default=str))
        return 0

    print("=" * 78)
    print("BAND TESTED ON LEGS")
    print("=" * 78)
    print(f"  unit    : {r['unit']}")
    print(f"  grain   : {r['grain']}")
    print(f"  de-vig  : {r['devig']}")
    print(f"  bands   : {r['bands']}")
    print(f"  measured: {r['measured']}")
    print()
    print(f"  completed matches {r['n_matches_completed']}, kept "
          f"{r['n_matches_kept']} after registered exclusions:")
    for k, v in r["excluded_matches"].items():
        print(f"      -{v:5d}  {k}")
    print(f"  legs graded: {r['n_legs']}  "
          f"({r['legs_out_of_every_band']} fall outside every band)")
    print("  leg sets refused:")
    for k, v in r["refused"].items():
        print(f"      {v:6d}  {k}")
    print()
    hdr = (f"  {'band':12s} {'n legs':>7s} {'fixt':>6s} {'mean devig':>11s} "
           f"{'observed':>9s} {'GAP pp':>8s} {'95% CI':>18s} "
           f"{'verdict':>13s} {'external':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for band in BANDS:
        key = f"{band[0]:.2f}-{band[1]:.2f}"
        v = r["results"][key]
        ext = EXTERNAL.get(band)
        exts = f"{ext['gap_pp']:+.2f}@{ext['n']}" if ext else "-"
        if not v["n"]:
            print(f"  {key:12s} {0:>7d} {0:>6d} {'-':>11s} {'-':>9s} "
                  f"{'-':>8s} {'-':>18s} {'INSUFFICIENT':>13s} {exts:>10s}")
            continue
        ci = v["ci95"]
        cis = f"[{ci[0]:+.2f},{ci[1]:+.2f}]" if ci else "not tested"
        print(f"  {key:12s} {v['n']:>7d} {v['fixtures']:>6d} "
              f"{100 * v['mean_devig']:10.2f}% {100 * v['observed_rate']:8.2f}% "
              f"{v['gap_pp']:+8.2f} {cis:>18s} {v['verdict']:>13s} "
              f"{exts:>10s}")
    print()
    print(f"  external comparator: {r['external_label']}")
    print(f"  a band is TESTED only at n >= {MIN_N_TO_TEST} over "
          f">= {MIN_FIXTURES_TO_TEST} fixtures; below that it is "
          f"INSUFFICIENT and no verdict is given")

    print()
    print("=" * 78)
    print("COLLECTABLE?  -  the IN-BAND cost, per venue, at the band's price")
    print("=" * 78)
    print("  EXISTENCE AND COLLECTABILITY ARE TWO QUESTIONS. A gap against a")
    print("  de-vigged price is not money: the margin is paid out of it.")
    print("  Bookmaker margins RISE in the favourites band while both")
    print("  exchanges' are flat, so this is the IN-BAND cost, never the")
    print("  overall one.")
    VENUES = ("TheOddsAPI-matchbook", "TheOddsAPI-betfair_ex_eu",
              "TheOddsAPI-pinnacle", "Pinnacle", "Bet365", "1xBet",
              "TAKEN_policy")
    for band in BANDS:
        key = f"{band[0]:.2f}-{band[1]:.2f}"
        v = r["results"][key]
        if v.get("verdict") != "PRESENT":
            continue
        gap = v["gap_pp"]
        lo = v["ci95"][0] if v["ci95"] else None
        print()
        print(f"  BAND {key}   gap {gap:+.2f}pp   CI lower bound "
              f"{lo:+.2f}pp   (n={v['n']} legs)")
        hdr = (f"    {'venue':26s} {'in-band cost':>13s} {'cover':>7s} "
               f"{'gap - cost':>11s} {'CIlo - cost':>12s}")
        print(hdr)
        print("    " + "-" * (len(hdr) - 4))
        by_venue = r["in_band_cost"].get(key) or {}
        for bm in VENUES:
            c = by_venue.get(bm)
            if not c:
                continue
            net = gap - c["cost_pp"]
            netlo = (lo - c["cost_pp"]) if lo is not None else None
            ex = "  [EX]" if "matchbook" in bm or "betfair" in bm else ""
            print(f"    {bm[:26]:26s} {c['cost_pp']:12.2f}pp "
                  f"{100 * c['fixture_coverage']:6.1f}% {net:+10.2f}pp "
                  f"{'' if netlo is None else f'{netlo:+11.2f}pp'}{ex}")
    print()
    print("  `gap - cost` uses the POINT estimate; `CIlo - cost` uses the")
    print("  interval's lower bound, which is the figure a decision needs.")
    print("  [EX] = EXCHANGE: commission is charged on NET WINNINGS, is NOT")
    print("  in this price, and its DEPTH is UNAVAILABLE in the feed -- so")
    print("  its cost is UNESTABLISHED, not low.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
