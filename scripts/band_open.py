"""BAND OPEN - derive the `min_odds` that admits the FLB band, and size it.

The anomaly's band is unreachable: `betting.min_odds: 1.50` admits only
selections at or below 66.7% implied, and the band is 70-79%. This script
derives the floor that admits it and measures what opening it would add.

THE DERIVATION IS NOT `1/0.79`, AND THAT MATTERS.
--------------------------------------------------
`docs/flb-preregistration.md` fixes the bands on **DE-VIGGED** probability:
`p_devig = p_implied / sum(p_implied)` over the market's own legs. But
`min_odds` gates the **RAW** price. The two differ by the market's margin, so

    raw implied  =  p_devig * sum(p_implied)  =  p_devig * (1 + margin)

and a leg whose DE-VIGGED probability is 0.79 has a RAW implied probability
near 0.84 and therefore odds near 1.19 -- not 1.2658. Deriving the floor from
`1/0.79` would open the band only partway and the measurement could not see
the top of its own band.

So the floor is derived from the band's upper de-vigged bound and the margin
MEASURED on our own prices, not assumed:

    min_odds  <=  1 / (0.80 * (1 + margin_max))

and it is cross-checked against the raw odds of the legs that actually fall
in the de-vigged band in our record.

MEASUREMENT ONLY. This script changes no config. It reports the value that
would, and what it would cost.

UNITS. `pp` is probability points; `rel` is relative margin. They differ by a
factor of the odds.

    python -m scripts.band_open
    python -m scripts.band_open --json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

#: The band, from `docs/flb-preregistration.md`: `[0.70, 0.80)` on DE-VIGGED
#: probability. Half-open, as the pre-registration fixed it.
BAND_DEVIG = (0.70, 0.80)

#: The external comparator, kept labelled. NOT ours and NOT re-derived.
EXTERNAL_GAP_PP = 3.74
EXTERNAL_LABEL = ("external, Footiqo, 2025/26, ONE bookmaker (1xBet), five "
                  "leagues, n=134 at 70-79%")

#: The floor this stage MOVED, kept as the labelled baseline the candidate
#: counts are measured against. It is NOT read as "current": the deployed
#: value is read live by `_live_floor()` below, because a literal here went
#: one cohort out of date the moment the config changed and that is the
#: stale-literal class this register keeps finding.
PREVIOUS_MIN_ODDS = 1.50


def _live_floor() -> float:
    """The DEPLOYED floor, from `config.example.yaml` -- the file CI copies
    over `config/config.yaml`, so it is the one that runs."""
    import yaml
    cfg = yaml.safe_load(
        pathlib.Path("config/config.example.yaml").read_text(encoding="utf-8"))
    return float(cfg["betting"]["min_odds"])

#: Power target for the n calculation. One-sided, because the
#: pre-registration's alternative is directional (a POSITIVE gap).
ALPHA = 0.05
POWER = 0.80

BOOTSTRAP_DRAWS = 4000


def _load_env() -> None:
    """INSIDE a function, never at import. Reads `.env` and `.env.local`."""
    import os
    here = __import__("os").path.dirname(
        __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    for name in (".env", ".env.local"):
        try:
            with open(__import__("os").path.join(here, name),
                      encoding="utf-8") as fh:
                for line in fh:
                    m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                    if m and not os.environ.get(m.group(1)):
                        os.environ[m.group(1)] = \
                            m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


def n_for_gap(gap_pp: float, p: float, alpha: float = ALPHA,
              power: float = POWER) -> int:
    """n to distinguish `gap_pp` from zero, one-sided, at `power`.

    A band's outcome is binary at rate ~`p`, so SE = sqrt(p(1-p)/n) and the
    detectable effect is `(z_alpha + z_power) * SE`. Inverting:

        n = p(1-p) * ((z_a + z_b) / effect)^2

    One pick per fixture (`max_picks_per_match: 1`), so picks in the band are
    one per fixture and the independence assumption holds at this grain --
    which is why this is an analytic n and the bootstrap below is a check on
    it rather than a replacement.
    """
    z_a = 1.6448536269514722          # one-sided 0.05
    z_b = 0.8416212335729143          # 0.80
    effect = gap_pp / 100.0
    return int(math.ceil(p * (1.0 - p) * ((z_a + z_b) / effect) ** 2))


def _bootstrap_n_check(p: float, gap: float, n: int,
                       draws: int = BOOTSTRAP_DRAWS) -> float:
    """What fraction of simulated bands of size `n` would detect the gap.

    Resamples FIXTURES (= picks, at one per fixture) from a Bernoulli at the
    true rate `p + gap`, and asks how often the one-sided 95% interval on the
    observed rate excludes `p`. This is the empirical power at that n.
    """
    rng = random.Random(20261106)
    hits = 0
    true_rate = p + gap
    se_crit = 1.6448536269514722
    for _ in range(draws):
        wins = sum(1 for _ in range(n) if rng.random() < true_rate)
        phat = wins / n
        se = math.sqrt(max(phat * (1 - phat), 1e-12) / n)
        if phat - se_crit * se > p:
            hits += 1
    return hits / draws


def measure() -> dict:
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
    from src.data.market_spec import extract_legs, get_spec, overround
    from src.data.market_vocabulary import canonical

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"

    with db.get_session() as s:
        picks = s.execute(text("""
            SELECT sp.id, sp.match_id, sp.market, sp.selection, sp.odds,
                   sp.result, sp.pick_date
            FROM saved_picks sp
            WHERE sp.pick_date >= '2026-08-11'
              AND sp.disposition IS NULL
              AND sp.market IS NOT NULL AND sp.selection IS NOT NULL
              AND sp.odds IS NOT NULL AND sp.odds > 1.0
            ORDER BY sp.id
        """)).fetchall()
        # every fixture we have priced in the experiment window, whether or
        # not it produced a pick -- the band's candidate population is drawn
        # from these, not from the picks.
        fixtures = s.execute(text("""
            SELECT DISTINCT o.match_id, m.match_date::date
            FROM odds o JOIN matches m ON m.id = o.match_id
            WHERE m.match_date >= '2026-08-11'
        """)).fetchall()
        mids = [f[0] for f in fixtures]
        date_of = {f[0]: f[1] for f in fixtures}
        odds_rows = s.execute(text("""
            SELECT match_id, bookmaker, market_type, selection, odds_value
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
              AND bookmaker IS NOT NULL
        """), {"ids": mids}).fetchall()

    # -- the TAKEN price: cross-book median per leg, as `_find_best_odds` ---
    by_leg: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    for mid, _bm, mt, sel, val in odds_rows:
        by_leg[(mid, mt, sel)].append(float(val))
    taken: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
    for (mid, mt, sel), vals in by_leg.items():
        taken[(mid, mt)][sel] = statistics.median(vals)

    # -- every leg, with its RAW implied and its DE-VIGGED probability ------
    #
    # De-vigging is PROPORTIONAL NORMALISATION over the market's own legs,
    # exactly as `docs/flb-preregistration.md` fixed it before measuring:
    # `p_devig = p_implied / sum(p_implied)`.
    legs: List[dict] = []
    margins: List[float] = []
    for (mid, market), by_sel in taken.items():
        spec = get_spec(market)
        if spec is None or spec.overlapping:
            continue            # double chance legs overlap; not a margin
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
                margins.append(orr - 1.0)
                for o in prices:
                    legs.append({
                        "mid": mid, "market": market, "odds": o,
                        "p_raw": 1.0 / o, "p_devig": (1.0 / o) / orr,
                        "margin": orr - 1.0, "date": date_of.get(mid),
                    })

    live_floor = _live_floor()
    lo, hi = BAND_DEVIG
    in_band = [L for L in legs if lo <= L["p_devig"] < hi]

    # -- THE DERIVED FLOOR -------------------------------------------------
    margin_max = max(margins) if margins else None
    margin_p95 = (statistics.quantiles(margins, n=20)[18]
                  if len(margins) > 19 else None)
    margin_med = statistics.median(margins) if margins else None
    derived = {
        "band_devig": BAND_DEVIG,
        "naive_1_over_0_79": 1.0 / 0.79,
        "margin_median_rel": margin_med,
        "margin_p95_rel": margin_p95,
        "margin_max_rel": margin_max,
        "n_legsets": len(margins),
        # the floor that admits the band at a given margin
        "floor_at_median_margin": 1.0 / (hi * (1.0 + margin_med))
        if margin_med else None,
        "floor_at_p95_margin": 1.0 / (hi * (1.0 + margin_p95))
        if margin_p95 else None,
        "floor_at_max_margin": 1.0 / (hi * (1.0 + margin_max))
        if margin_max else None,
        # cross-check: the raw odds actually seen on in-band legs
        "observed_min_odds_in_band": min(
            (L["odds"] for L in in_band), default=None),
        "observed_max_odds_in_band": max(
            (L["odds"] for L in in_band), default=None),
        "observed_p01_odds_in_band": (
            statistics.quantiles([L["odds"] for L in in_band], n=100)[0]
            if len(in_band) > 99 else None),
        "provenance": "measured 2026-11-06, grain=LEG, taken (median) prices",
    }

    # -- THE CANDIDATE POPULATION the floor would admit --------------------
    #
    # A price the floor admits is a CANDIDATE, not a pick: the EV, value and
    # per-match gates still apply and are untouched. So this is an UPPER
    # BOUND on what opening the band adds, and it is labelled as one.
    rejected_now = [L for L in in_band if L["odds"] < PREVIOUS_MIN_ODDS]
    fixtures_with_band = {L["mid"] for L in rejected_now}
    days = len({d for d in date_of.values() if d})
    picked_mids = {p[1] for p in picks}
    band_fixtures_unpicked = fixtures_with_band - picked_mids

    per_market: Dict[str, int] = defaultdict(int)
    for L in rejected_now:
        per_market[L["market"]] += 1

    candidates = {
        "legs_in_band": len(in_band),
        "legs_in_band_below_baseline_floor": len(rejected_now),
        "fixtures_with_an_admissible_band_leg": len(fixtures_with_band),
        "of_those_we_never_picked_at_all": len(band_fixtures_unpicked),
        "window_days": days,
        "fixtures_per_day": len(fixtures_with_band) / days if days else None,
        "per_market": dict(sorted(per_market.items(),
                                  key=lambda kv: -kv[1])),
        "note": ("UPPER BOUND. These are prices the floor would ADMIT. The "
                 "EV, value, correlation and max_picks_per_match gates are "
                 "unchanged and would reject most of them, so the realised "
                 "pick rate is at most this and probably far below it."),
        "provenance": "measured 2026-11-06, grain=LEG and FIXTURE",
        "baseline_floor": PREVIOUS_MIN_ODDS,
        "live_floor": live_floor,
        "live_floor_admits_whole_band": live_floor <= min(
            (L["odds"] for L in in_band), default=float("inf")),
    }

    # -- THE SPILL: a RAW floor cannot express a DE-VIGGED band ------------
    #
    # `min_odds` gates raw odds; the band is de-vigged. So any floor low
    # enough to admit all of [0.70, 0.80) also admits some of [0.80, 0.90) --
    # the NEXT band in the pre-registration -- because a high-margin leg set
    # at p_devig 0.82 can price lower than a low-margin one at 0.79.
    #
    # This is reported, not engineered away. The pre-registration tests each
    # band separately, so the spill POPULATES a second band rather than
    # confounding the first, and hard rule 1 is satisfied at the grain it is
    # written for: one config key, one cohort, effects attributable per band.
    CANDIDATE_FLOOR = 1.14
    spill = {"floor": CANDIDATE_FLOOR}
    adm = [L for L in legs if L["odds"] >= CANDIDATE_FLOOR]
    newly = [L for L in adm if L["odds"] < PREVIOUS_MIN_ODDS]
    spill["newly_admitted_legs"] = len(newly)
    buckets: Dict[str, int] = defaultdict(int)
    for L in newly:
        pv = L["p_devig"]
        if pv < 0.70:
            buckets["below_0.70"] += 1
        elif pv < 0.80:
            buckets["THE_BAND_0.70_0.80"] += 1
        elif pv < 0.90:
            buckets["spill_0.80_0.90"] += 1
        else:
            buckets["spill_0.90_plus"] += 1
    spill["by_devig_band"] = dict(buckets)
    spill["band_share_of_newly_admitted"] = (
        buckets["THE_BAND_0.70_0.80"] / len(newly) if newly else None)
    spill["in_band_legs_admitted"] = sum(
        1 for L in in_band if L["odds"] >= CANDIDATE_FLOOR)
    spill["in_band_legs_total"] = len(in_band)
    spill["in_band_coverage"] = (
        spill["in_band_legs_admitted"] / len(in_band) if in_band else None)
    spill["provenance"] = "measured 2026-11-06, grain=LEG"

    # -- THE n, AND THE DATE IT ARRIVES ------------------------------------
    p_mid = sum(BAND_DEVIG) / 2.0
    n_req = n_for_gap(EXTERNAL_GAP_PP, p_mid)
    emp = _bootstrap_n_check(p_mid, EXTERNAL_GAP_PP / 100.0, n_req)
    # at one pick per fixture, the in-band pick rate cannot exceed the
    # in-band FIXTURE rate.
    rate_ceiling = candidates["fixtures_per_day"]
    power_block = {
        "gap_pp": EXTERNAL_GAP_PP,
        "gap_label": EXTERNAL_LABEL,
        "band_midpoint_rate": p_mid,
        "alpha_one_sided": ALPHA,
        "target_power": POWER,
        "n_required": n_req,
        "empirical_power_at_n": emp,
        "prereg_minimum_n": 100,
        "max_in_band_picks_per_day": rate_ceiling,
        "days_at_the_ceiling": (n_req / rate_ceiling
                                if rate_ceiling else None),
        "provenance": "derived; the rate is measured, the n is analytic",
    }
    return {"derived": derived, "candidates": candidates, "spill": spill,
            "power": power_block,
            "picks_now": len(picks), "fixtures_priced": len(mids)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        print(json.dumps(r, indent=1, default=str))
        return 0

    d, c, pw = r["derived"], r["candidates"], r["power"]
    print("=" * 78)
    print("BAND OPENED  -  the derived floor")
    print("=" * 78)
    print(f"  the band, from docs/flb-preregistration.md: DE-VIGGED "
          f"[{d['band_devig'][0]:.2f}, {d['band_devig'][1]:.2f})")
    print(f"  measured over {d['n_legsets']} leg sets at taken (median) prices")
    print()
    print(f"  THE NAIVE DERIVATION IS WRONG: 1/0.79 = "
          f"{d['naive_1_over_0_79']:.4f} is the floor for the RAW-implied")
    print(f"  band. The pre-registration's bands are DE-VIGGED, and")
    print(f"  raw = p_devig * (1 + margin), so the floor is lower.")
    print()
    print(f"  measured margin (relative): median "
          f"{100 * d['margin_median_rel']:.2f}%, p95 "
          f"{100 * d['margin_p95_rel']:.2f}%, max "
          f"{100 * d['margin_max_rel']:.2f}%")
    print()
    print(f"  floor = 1 / (0.80 * (1 + margin)):")
    print(f"    at the MEDIAN margin : {d['floor_at_median_margin']:.4f}")
    print(f"    at the p95 margin    : {d['floor_at_p95_margin']:.4f}")
    print(f"    at the MAX margin    : {d['floor_at_max_margin']:.4f}")
    print()
    print(f"  CROSS-CHECK on the legs that actually fall in the de-vigged")
    print(f"  band in our record (n={c['legs_in_band']}):")
    print(f"    raw odds range: {d['observed_min_odds_in_band']:.4f} .. "
          f"{d['observed_max_odds_in_band']:.4f}")
    if d["observed_p01_odds_in_band"]:
        print(f"    1st percentile: {d['observed_p01_odds_in_band']:.4f}")
    print()
    print("=" * 78)
    print("THE CANDIDATE POPULATION  -  an upper bound, not a forecast")
    print("=" * 78)
    print(f"  legs in the de-vigged band            : {c['legs_in_band']}")
    print(f"  of those, below the BASELINE 1.50 floor: "
          f"{c['legs_in_band_below_current_floor']}")
    print(f"  fixtures carrying such a leg          : "
          f"{c['fixtures_with_an_admissible_band_leg']}")
    print(f"  of those, fixtures we never picked    : "
          f"{c['of_those_we_never_picked_at_all']}")
    print(f"  over {c['window_days']} days = "
          f"{c['fixtures_per_day']:.2f} fixtures/day at most")
    print()
    print(f"  baseline floor (the one this stage moved): "
          f"{c['baseline_floor']}")
    print(f"  LIVE floor in config.example.yaml        : {c['live_floor']}")
    print(f"  does the live floor admit the whole band : "
          f"{'YES' if c['live_floor_admits_whole_band'] else 'NO'}")
    print()
    print("  by market:")
    for m, n in c["per_market"].items():
        print(f"    {m:18s} {n:6d}")
    print()
    print(f"  {c['note']}")
    print()
    sp = r["spill"]
    print("=" * 78)
    print(f"  A RAW FLOOR CANNOT EXPRESS A DE-VIGGED BAND. At the candidate")
    print(f"  floor {sp['floor']}, the legs newly admitted "
          f"({sp['newly_admitted_legs']}) fall in these de-vigged bands:")
    for k, v in sorted(sp["by_devig_band"].items(),
                       key=lambda kv: -kv[1]):
        print(f"    {k:22s} {v:6d}  "
              f"({100 * v / sp['newly_admitted_legs']:5.1f}%)")
    print(f"  the target band is "
          f"{100 * sp['band_share_of_newly_admitted']:.1f}% of them, and the")
    print(f"  floor admits {sp['in_band_legs_admitted']} of "
          f"{sp['in_band_legs_total']} in-band legs "
          f"({100 * sp['in_band_coverage']:.2f}%)")
    print()
    print("=" * 78)
    print("THE n, AND THE DATE IT ARRIVES")
    print("=" * 78)
    print(f"  to distinguish {pw['gap_pp']:.2f}pp from zero at "
          f"{100 * pw['target_power']:.0f}% power, one-sided alpha="
          f"{pw['alpha_one_sided']}")
    print(f"  at the band's midpoint rate "
          f"{pw['band_midpoint_rate']:.3f}:")
    print(f"      n = {pw['n_required']} in-band picks")
    print(f"      empirical power at that n (bootstrap, "
          f"{BOOTSTRAP_DRAWS} draws): "
          f"{100 * pw['empirical_power_at_n']:.1f}%")
    print(f"  the pre-registration's own minimum to test a band at all: "
          f"n >= {pw['prereg_minimum_n']}")
    print()
    if pw["days_at_the_ceiling"]:
        print(f"  at the CEILING rate of "
              f"{pw['max_in_band_picks_per_day']:.2f}/day (one pick per "
              f"fixture):")
        print(f"      n={pw['n_required']} arrives in "
              f"{pw['days_at_the_ceiling']:.0f} days")
        print(f"      n={pw['prereg_minimum_n']} arrives in "
              f"{pw['prereg_minimum_n'] / pw['max_in_band_picks_per_day']:.0f}"
              f" days")
        print()
        print("  AND THAT IS A CEILING. The EV and value gates are unchanged,")
        print("  so the realised rate is lower and the date is later.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
