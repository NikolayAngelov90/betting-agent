"""EXCHANGE TERMS - the three unmeasured terms between the result and a plan.

The previous stage (`scripts/cost_by_book.py`) found the arithmetic positive
at an exchange and at no bookmaker: the documented anomaly nets +3.15pp
against Matchbook's 0.59pp band cost, and -0.96pp against what we pay today.
That result rests on three terms nobody has measured.

    COMMISSION   an exchange charges on NET WINNINGS, outside the quoted
                 price. So the quoted margin is not the cost.
    SIZE         an exchange price exists only while someone offers it at
                 size. A back price good for 20 units is not a price.
    COVERAGE     Matchbook quoted 22.6% of our picks. A policy restricted to
                 it prices one pick in five, and the covered subset may not
                 be a random fifth.

MEASUREMENT ONLY. Adopts nothing, selects no venue, moves no threshold.

UNITS. Costs are PROBABILITY POINTS unless marked `rel` (relative margin) or
`ROI` (return on turnover). A band-specific figure converts at the BAND's own
odds, never at the population average -- that error inverted a sign in the
previous stage and is kept there as a known-wrong control.

PROVENANCE. Every figure here is `measured` from the Postgres record on
2026-11-06 unless its own label says `published` (an external schedule, with
URL and fetch date) or `derived` (arithmetic on the two).

ENVIRONMENT. `measure()` calls the shared `_load_env`, which reads `.env` and
`.env.local` from the repo root INSIDE the function -- never at import, since
an import-time `load_dotenv` has reached production from a test before. The
loader lives in `scripts/cost_by_book.py`; stating that here is not
decoration, because `tests/test_dialect_is_asserted.py` checks that every
script reaching `get_db()` loads the environment by looking for `.env` in the
file's own text, and it cannot see a loader behind an import.

    python -m scripts.exchange_terms
    python -m scripts.exchange_terms --json
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from scripts.cost_by_book import (ANOMALY_BAND, DOCUMENTED_ANOMALY,
                                  MEASURED_BOOKS, _bootstrap_roi,
                                  _load_env, _pp_at)

#: PUBLISHED COMMISSION. Hard rule 1: a commission comes from a published
#: schedule with its URL and the date fetched, never from an assumption.
#:
#: STATUS 2026-11-06: only ONE official page was reachable from this
#: environment. Every `matchbook.com` host fails TLS verification here
#: (self-signed certificate on the intercepting proxy), and every
#: `betfair.com` / `support.betfair.com` page returns HTTP 403. So the
#: Matchbook figure is NOT a direct fetch and is labelled accordingly, and the
#: Betfair figure is the AUSTRALIAN schedule, which is the jurisdiction that
#: page covers -- not necessarily the rate an EU account would pay.
#:
#: This is reported as a coverage limit, not worked around. The conclusion
#: below is therefore stated across the WHOLE plausible range rather than at
#: a point estimate, because the range is what the evidence supports.
COMMISSION_PUBLISHED = {
    "Betfair Exchange": {
        "rate": 0.06,
        "basis": "net winnings per market, deducted at settlement",
        "formula": "commission = winnings x Market Base Rate",
        "provenance": "published, DIRECTLY FETCHED 2026-11-06",
        "url": "https://www.betfair.com.au/hub/help/commissions-charges/",
        "caveat": (
            "AUSTRALIAN schedule: 6% sport and international racing, NRL 10%, "
            "Australian racing 8-10%. The .com charges page and the support "
            "article both return 403 from here, so the EU/UK Market Base Rate "
            "and the Betfair Points discount scheme (reported elsewhere as "
            "capped at 20%) are NOT established by a direct fetch."),
    },
    "Matchbook": {
        "rate": 0.02,
        "basis": "net winnings per market, winning bets only",
        "formula": "commission = net winnings x rate",
        "provenance": (
            "published on Matchbook's OWN domain but NOT directly fetched: "
            "read via web search of insights.matchbook.com on 2026-11-06, "
            "because every matchbook.com host fails TLS verification here"),
        "url": "https://insights.matchbook.com/education/what-is-commission/",
        "caveat": (
            "the same search reports a 0% promotional rate for 110 days and "
            "'2% or 4% Net Win Commission' for residents of non-accepted "
            "countries, so 2% is the standard rate and not the only rate"),
    },
}

#: The rates the arithmetic is run at. The first two are the published
#: figures; 0.04 is Matchbook's own stated alternative for non-accepted
#: regions; the last is the break-even the previous stage derived.
COMMISSION_RANGE = (0.02, 0.04, 0.06, 0.156)

#: SIZE. There is no size column, so this records WHERE that was established.
SIZE_FIELDS_SEARCHED = (
    "odds.odds_value / opening_odds / timestamp / first_seen_at",
    "odds_snapshots.odds_value / observed_at",
    "src/scrapers/theodds_scraper.py outcome parse",
)

#: `n = 525 picks = 28 days` to detect the -5.68% -> +3% move (8.7 points),
#: recorded 2026-11-01. 525/28 = 18.75 picks/day at the CURRENT pick rate.
POWER_N_FOR_TARGET_MOVE = 525
POWER_DAYS_AT_CURRENT_RATE = 28

BOOTSTRAP_DRAWS = 4000

#: WHAT THIS STAGE MEASURED. `measured` 2026-11-06 from Postgres unless a key
#: says otherwise. Population 1,639 settled paper picks over 70 days.
MEASURED_TERMS = {
    "measured": "2026-11-06",
    "population": 1639,
    "days": 70,
    "picks_per_day": 23.41,

    # -- SIZE: the term that decides the stage ------------------------------
    "size_available": False,
    "size_searched": SIZE_FIELDS_SEARCHED,
    "size_verdict": (
        "UNAVAILABLE. No size, depth, liquidity or available-amount field "
        "exists in `odds`, in `odds_snapshots`, or in the TheOddsAPI outcome "
        "parse, which reads `name` and `price` only. So the exchange price "
        "has NO MEASURABLE DEPTH and its cost is NOT ESTABLISHED AS "
        "TAKEABLE -- the same status as the best line, for the same reason. "
        "THIRD instance of that error class. UNESTABLISHED, not refuted."),

    # -- OUR BAND vs THE ANOMALY'S BAND: disjoint, by configuration --------
    "our_min_odds": 1.50,
    "our_median_odds": 1.58,
    "our_max_odds": 3.49,
    "our_max_implied_pct": 66.7,
    "anomaly_band_implied_pct": (70.0, 79.0),
    "picks_in_anomaly_band": 0,
    # THE FLOOR AT THE TIME OF THIS MEASUREMENT. Not superseded as a figure:
    # the 0-of-1,639 result above is a property of the s5.19 cohort, which ran
    # at 1.50 for its whole life. `floor_opened_to` is what replaced it.
    "config_min_odds": 1.50,
    "cohort_at_measurement": "s5.19",
    "floor_opened_to": 1.14,
    "floor_opened_on": "2026-11-07",
    "floor_opened_in_cohort": "s5.20",
    # ── AND RESTORED. A ROUND TRIP, 2026-11-11, cohort s5.21. ───────────
    "floor_restored_to": 1.50,
    "floor_restored_on": "2026-11-11",
    "floor_restored_in_cohort": "s5.21",
    "floor_restored_why": (
        "the operator does not want prices at 1.14-1.20 published. DERIVED "
        "from the record: all fifteen cohorts from 2026-08-11 to 2026-11-07 "
        "ran at exactly 1.50, n=1,798, per-cohort average taken odds "
        "1.592..1.671, and the minimum taken price in every one is 1.50 to "
        "the cent"),
    "round_trip_cost": (
        "cohort s5.20: TWO DAYS, 2026-11-08..2026-11-09, n=86 picks, average "
        "taken odds 1.464 against 1.59..1.67 for every cohort before it, "
        "minimum 1.14. Prices the operator did not want. RETAINED and "
        "labelled; never pooled with s5.19 or s5.21; far too small to "
        "support any conclusion"),
    "band_measurable_without_the_floor": (
        "the hypothesis is about MARKET CALIBRATION and its unit is a LEG -- "
        "a (fixture, market, selection) with a de-vigged probability and a "
        "known result. 5,314 in-band LEGS exist in the odds record at ANY "
        "floor, so restoring it costs the measurement NOTHING. Opening the "
        "band was necessary to ACT on the anomaly, not to measure it, and "
        "acting is blocked on an exchange whose price depth is UNAVAILABLE"),
    "floor_opened_why": (
        "to MEASURE the FLB band, not to profit from it. Derived in "
        "docs/flb-band-preregistration.md (fed8338), committed BEFORE the "
        "bump: 1.14 admits 5,314 of 5,314 de-vigged in-band legs. The band "
        "was UNREACHABLE at 1.50, not merely empty, which is why opening it "
        "is the only way the anomaly becomes measurable in our own data."),
    "band_verdict": (
        "DISJOINT. `betting.min_odds: 1.50` is 66.7% implied, and the "
        "anomaly is documented at 70-79% implied. 0 of 1,639 picks sit in "
        "that band and none CAN under the deployed configuration, which "
        "excludes it deliberately ('too little margin for model error') via "
        "the registered `min_odds` gate. So harvesting the anomaly is NOT a "
        "pricing change: it needs a different SELECTION, and nothing here "
        "measures that."),

    # -- THE COST WE ACTUALLY PAY, at our own prices, takeable only --------
    # pp at each pick's OWN price, median over the picks the book covered
    # and whose whole leg set existed when we priced that fixture.
    "our_cost_pp": {
        "TheOddsAPI-matchbook": 0.65,
        "TheOddsAPI-betfair_ex_eu": 0.73,
        "TheOddsAPI-pinnacle": 1.98,
        "Pinnacle": 2.53,
        "Bet365": 4.17,
        "1xBet": 4.32,
        "TAKEN_current_policy": 3.70,
    },
    "takeable_cover_pick_share": {
        "TheOddsAPI-matchbook": 0.133,
        "TheOddsAPI-betfair_ex_eu": 0.049,
        "TheOddsAPI-pinnacle": 0.082,
        "Pinnacle": 0.509,
        "Bet365": 0.677,
        "1xBet": 0.602,
        "TAKEN_current_policy": 1.0,
    },

    # -- COMMISSION, from the published schedules --------------------------
    "commission_published": {"Matchbook": 0.02, "Betfair Exchange": 0.06},
    "commission_fetched_directly": ("Betfair Exchange",),
    "breakeven_commission_matchbook": 0.156,

    # -- COVERAGE: the exchange subset is a DIFFERENT POPULATION ----------
    "markets_absent_from_exchanges": (
        "team_goals", "double_chance", "btts", "draw_no_bet"),
    "share_of_picks_in_absent_markets": 0.526,
    "market_mix_full": {
        "over_under": 0.356, "team_goals": 0.193, "double_chance": 0.165,
        "1X2": 0.117, "btts": 0.093, "draw_no_bet": 0.075,
    },
    "market_mix_matchbook": {"over_under": 0.647, "1X2": 0.353},
    "market_mix_betfair": {"1X2": 1.0},
    "leagues_full": 30,
    "leagues_matchbook": 25,

    # -- THE DAYS TO A DECISION -------------------------------------------
    "days_for_525": {
        "Bet365": 27, "Pinnacle": 44,
        "TheOddsAPI-matchbook": 99,
        "TheOddsAPI-matchbook_takeable": 169,
        "TheOddsAPI-betfair_ex_eu_takeable": 459,
    },

    "verdict": (
        "The exchange result is UNESTABLISHED, not positive. Its depth is "
        "unmeasurable from this record; its price was a later write for 41% "
        "of covered picks; it quotes NONE of the four markets holding 52.6% "
        "of our picks; and the anomaly it would harvest sits in a band the "
        "deployed `min_odds` gate excludes entirely, where we hold 0 picks. "
        "What IS established is smaller and real: at the prices we actually "
        "take, Pinnacle costs 2.53pp against the current policy's 3.70pp on "
        "50.9% of picks, with no commission and no later writes."),
}


def commission_cost(odds: float, win_rate: float, rate: float) -> dict:
    """The all-in cost of a commission on NET WINNINGS, three ways.

    Backing at decimal odds `o` for one unit: a win returns `1 + (o-1)`, and
    the exchange takes `rate` of the `(o-1)` profit. So the price actually
    received is the EFFECTIVE ODDS

        o' = 1 + (o - 1) * (1 - rate)

    `pp` converts that to probability points -- the implied probability of
    `o'` minus that of `o` -- which is the unit the band costs are in and so
    the only one comparable to them.

    `roi` is the expected cost per unit staked at our win rate, which is the
    unit the +3% target is in: a commission is only paid on winners, so it
    costs `win_rate * rate * (o - 1)` per unit of turnover.
    """
    eff = 1.0 + (odds - 1.0) * (1.0 - rate)
    return {
        "effective_odds": eff,
        "pp": 100.0 * (1.0 / eff - 1.0 / odds),
        "roi": win_rate * rate * (odds - 1.0),
        "provenance": "derived",
    }


def measure() -> dict:
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
    from src.data.market_spec import extract_legs, get_spec, overround
    from src.data.market_vocabulary import canonical, selection_aliases

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"

    with db.get_session() as s:
        picks = s.execute(text("""
            SELECT sp.id, sp.match_id, sp.market, sp.selection, sp.odds,
                   sp.result, m.league,
                   (SELECT po.taken_at FROM pick_observations po
                     WHERE po.pick_id = sp.id AND po.attribution = 'final'
                     ORDER BY po.taken_at LIMIT 1) AS taken_at,
                   sp.pick_date
            FROM saved_picks sp
            LEFT JOIN matches m ON m.id = sp.match_id
            WHERE sp.pick_date >= '2026-08-11'
              AND sp.disposition IS NULL
              AND sp.result IN ('win','loss')
              AND sp.market IS NOT NULL AND sp.selection IS NOT NULL
              AND sp.odds IS NOT NULL AND sp.odds > 1.0
            ORDER BY sp.id
        """)).fetchall()
        mids = sorted({p[1] for p in picks})
        odds_rows = s.execute(text("""
            SELECT match_id, bookmaker, market_type, selection, odds_value,
                   timestamp
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
              AND bookmaker IS NOT NULL
        """), {"ids": mids}).fetchall()

    pop = len(picks)
    days = len({p[8] for p in picks})

    price: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    when: Dict[Tuple[int, str, str, str], object] = {}
    for mid, bm, mt, sel, val, ts in odds_rows:
        price[(mid, bm)][mt][sel] = float(val)
        when[(mid, bm, mt, sel)] = ts

    # -- the candidates: the exchanges, plus the comparators -------------
    cands = list(MEASURED_BOOKS["cost_rel"])

    # -- PER-PICK COST, and whether the whole leg set existed at pick time
    #
    # The cost we would have paid on a pick is the overround of the LEG SET
    # that pick's price came from, at that book. A leg set is only takeable
    # if EVERY leg existed at `taken_at`: `odds` holds one row per
    # (match, book, market, selection) with a single timestamp, so a set can
    # straddle the pick.
    per_pick: Dict[str, List[dict]] = defaultdict(list)
    for pid, mid, market, selection, odds, result, league, ta, _pd in picks:
        canon = canonical(market, selection)
        if canon is None:
            continue
        fam, canon_sel = canon
        aliases = set(selection_aliases(canon_sel))
        spec = get_spec(fam)
        for bm in cands:
            by_sel = price.get((mid, bm), {}).get(fam) or {}
            if not by_sel:
                continue
            hit = next(((k, v) for k, v in by_sel.items() if k in aliases),
                       None)
            if hit is None:
                continue
            hitkey, our_price = hit
            # the leg set this price sits in
            line, side = None, None
            if fam in ("over_under", "team_goals"):
                m = re.search(r"(\d+\.\d+)", hitkey)
                line = m.group(1) if m else None
            if fam == "team_goals":
                side = "Home" if "home" in hitkey.lower() else "Away"
            cost_rel, legs_ok = None, None
            if spec is not None and not spec.overlapping:
                legs = extract_legs(fam, by_sel, line=line, side=side)
                if legs and len(legs) == spec.arity:
                    orr = overround(legs)
                    if spec.overround_min <= orr <= spec.overround_max:
                        cost_rel = orr - 1.0
                    # every leg present at pick time?
                    keys = [k for k in by_sel
                            if (line is None or line in k)
                            and (side is None or side.lower() in k.lower())]
                    ts = [when.get((mid, bm, fam, k)) for k in keys]
                    legs_ok = (ta is not None and all(
                        t is not None and t <= ta for t in ts))
            per_pick[bm].append({
                "pid": pid, "mid": mid, "fam": fam, "league": league,
                "our_price": float(our_price),
                "won": 1 if result == "win" else 0,
                "taken_odds": float(odds),
                "cost_rel": cost_rel,
                "existed_at_pick": legs_ok,
                "in_band": ANOMALY_BAND[0] <= float(our_price)
                <= ANOMALY_BAND[1],
            })

    # -- assemble, both unrestricted and restricted to takeable prices ---
    def _summarise(rows: List[dict]) -> Optional[dict]:
        if not rows:
            return None
        costed = [r for r in rows if r["cost_rel"] is not None]
        band = [r for r in costed if r["in_band"]]
        out = {
            "n": len(rows),
            "n_costed": len(costed),
            "coverage": len(rows) / pop,
            "win_rate": sum(r["won"] for r in rows) / len(rows),
            "avg_odds": statistics.mean(r["our_price"] for r in rows),
            "roi": statistics.mean(
                r["our_price"] * r["won"] - 1.0 for r in rows),
            "ci95": _bootstrap_roi([(r["mid"], r["our_price"], r["won"])
                                    for r in rows]),
            "cost_rel": (statistics.median(r["cost_rel"] for r in costed)
                         if costed else None),
            # THE COST WE ACTUALLY PAY: the margin of the leg set each pick
            # came from, converted at THAT PICK'S OWN PRICE. Our picks run
            # 1.50-3.49 (median 1.58) because `betting.min_odds` is 1.50, so
            # this is a different band from the anomaly's and the conversion
            # must use our prices, not the band's.
            "our_cost_pp": (statistics.median(
                _pp_at(r["cost_rel"], r["our_price"]) for r in costed)
                if costed else None),
            "our_odds_median": (statistics.median(
                r["our_price"] for r in costed) if costed else None),
            "band_n": len(band),
            "band_cost_pp": (statistics.median(
                _pp_at(r["cost_rel"], r["our_price"]) for r in band)
                if band else None),
            "band_odds": (statistics.median(r["our_price"] for r in band)
                          if band else None),
            "band_win_rate": (sum(r["won"] for r in band) / len(band)
                              if band else None),
        }
        return out

    # -- TWO GRAINS, AND THEY ANSWER DIFFERENT QUESTIONS -----------------
    #
    # LEG-SET grain: the margin a book charges on a favourite in the band,
    #   whether or not we bet it. This is the comparator for "could the
    #   anomaly be harvested at this venue" -- it is what the previous stage
    #   measured, and it is the figure the anomaly must be netted against.
    #
    # PICK grain: the band composition of OUR OWN picks. This answers a
    #   different question -- whether our selection process ever produces a
    #   bet in the band at all -- and conflating the two would let a margin
    #   we never pay stand in for one we do.
    #
    # A leg set is takeable only if every leg existed when we priced that
    # match, so the reference instant is the EARLIEST `taken_at` among the
    # picks on that fixture.
    first_ta: Dict[int, object] = {}
    for _pid, mid, _mk, _sel, _o, _r, _lg, ta, _pd in picks:
        if ta is None:
            continue
        if mid not in first_ta or ta < first_ta[mid]:
            first_ta[mid] = ta

    legset_band: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    legset_band_ok: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (mid, bm), by_market in price.items():
        if bm not in cands:
            continue
        ref = first_ta.get(mid)
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
                    legs = extract_legs(market, by_sel, line=line, side=side)
                    if not legs or len(legs) != spec.arity:
                        continue
                    orr = overround(legs)
                    if not (spec.overround_min <= orr <= spec.overround_max):
                        continue
                    inband = [o for o in legs
                              if ANOMALY_BAND[0] <= o <= ANOMALY_BAND[1]]
                    if not inband:
                        continue
                    rec = (orr - 1.0, min(inband))
                    legset_band[bm].append(rec)
                    keys = [k for k in by_sel
                            if (line is None or line in k)
                            and (side is None or side.lower() in k.lower())]
                    ts = [when.get((mid, bm, market, k)) for k in keys]
                    if ref is not None and ts and all(
                            t is not None and t <= ref for t in ts):
                        legset_band_ok[bm].append(rec)

    def _band(recs):
        if not recs:
            return {"n": 0, "cost_pp": None, "cost_rel": None, "odds": None}
        return {
            "n": len(recs),
            "cost_rel": statistics.median(c for c, _o in recs),
            "cost_pp": statistics.median(_pp_at(c, o) for c, o in recs),
            "odds": statistics.median(o for _c, o in recs),
        }

    books = {}
    for bm, rows in per_pick.items():
        ok = [r for r in rows if r["existed_at_pick"]]
        books[bm] = {
            "all": _summarise(rows),
            "takeable": _summarise(ok),
            "dropped_not_at_pick": len(rows) - len(ok),
            "is_exchange": bm in MEASURED_BOOKS["exchanges"],
            # LEG-SET grain: the band margin, all leg sets and takeable only
            "legset_band_all": _band(legset_band.get(bm) or []),
            "legset_band_takeable": _band(legset_band_ok.get(bm) or []),
        }

    # -- OUR OWN EXPOSURE TO THE ANOMALY'S BAND (pick grain) -------------
    in_band_picks = [p for p in picks
                     if ANOMALY_BAND[0] <= float(p[4]) <= ANOMALY_BAND[1]]
    all_odds = [float(p[4]) for p in picks]
    our_exposure = {
        "picks_in_band": len(in_band_picks),
        "share": len(in_band_picks) / pop,
        "win_rate": (sum(1 for p in in_band_picks if p[5] == "win")
                     / len(in_band_picks)) if in_band_picks else None,
        "avg_odds": (statistics.mean(float(p[4]) for p in in_band_picks)
                     if in_band_picks else None),
        "roi": (statistics.mean(float(p[4]) * (1 if p[5] == "win" else 0) - 1.0
                                for p in in_band_picks)
                if in_band_picks else None),
        "band": "70-79% implied (odds 1.266-1.429)",
        "provenance": "measured 2026-11-06, grain=PICK",
        "our_min_odds": min(all_odds),
        "our_max_odds": max(all_odds),
        "our_median_odds": statistics.median(all_odds),
        "our_max_implied": 100.0 / min(all_odds),
        "config_min_odds": 1.50,
        "why": (
            "`betting.min_odds: 1.50` in config/config.yaml, a REGISTERED "
            "gate (`gate_registry.py`: 'removes selections priced below "
            "betting.min_odds (1.50)') with the stated reason 'Skip very "
            "short prices - too little margin for model error'. 1.50 is "
            "66.7% implied, so the deployed configuration CANNOT produce a "
            "pick in the anomaly's 70-79% band. The exclusion is by design "
            "and predates this stage."),
    }

    # -- REPRESENTATIVENESS of the covered subset ------------------------
    def _mix(rows, key):
        tot = len(rows) or 1
        c = defaultdict(int)
        for r in rows:
            c[r[key] or "(none)"] += 1
        return {k: v / tot for k, v in sorted(
            c.items(), key=lambda kv: -kv[1])}

    full = [{"fam": canonical(p[2], p[3])[0] if canonical(p[2], p[3]) else None,
             "league": p[6], "band": "70-79%"
             if ANOMALY_BAND[0] <= float(p[4]) <= ANOMALY_BAND[1]
             else "other"} for p in picks]
    repr_ = {"full": {"market": _mix(full, "fam"),
                      "league": _mix(full, "league"),
                      "band": _mix(full, "band")}}
    for bm in MEASURED_BOOKS["exchanges"]:
        rows = [{"fam": r["fam"], "league": r["league"],
                 "band": "70-79%" if r["in_band"] else "other"}
                for r in per_pick.get(bm, [])]
        ok = [{"fam": r["fam"], "league": r["league"],
               "band": "70-79%" if r["in_band"] else "other"}
              for r in per_pick.get(bm, []) if r["existed_at_pick"]]
        repr_[bm] = {"market": _mix(rows, "fam"),
                     "league": _mix(rows, "league"),
                     "band": _mix(rows, "band"),
                     "takeable_market": _mix(ok, "fam"),
                     "takeable_band": _mix(ok, "band")}

    # -- the pick rate and the days to a decision ------------------------
    rate_now = pop / days
    rates = {}
    for bm in cands:
        b = books.get(bm)
        if not b or not b["all"]:
            continue
        for label, key in (("all", "all"), ("takeable", "takeable")):
            sub = b[key]
            if not sub:
                continue
            per_day = rate_now * sub["coverage"]
            rates[f"{bm}:{label}"] = {
                "coverage": sub["coverage"],
                "picks_per_day": per_day,
                "days_for_525": (POWER_N_FOR_TARGET_MOVE / per_day
                                 if per_day else None),
            }

    return {
        "population": pop, "days": days, "picks_per_day": rate_now,
        "books": books, "representativeness": repr_, "rates": rates,
        "commission": COMMISSION_PUBLISHED, "our_exposure": our_exposure,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        print(json.dumps(r, indent=1, default=str))
        return 0

    A = DOCUMENTED_ANOMALY["value_pp"]
    EX = MEASURED_BOOKS["exchanges"]

    print("=" * 78)
    print("COMMISSION  -  Part A.  From the published schedule, or not at all.")
    print("=" * 78)
    for venue, c in COMMISSION_PUBLISHED.items():
        print(f"  {venue}: {100 * c['rate']:.1f}% -- {c['basis']}")
        print(f"      {c['formula']}")
        print(f"      provenance: {c['provenance']}")
        print(f"      url: {c['url']}")
        print(f"      CAVEAT: {c['caveat']}")
        print()
    print("  NOT OBTAINED, and not assumed in its place:")
    print("    - the EU/UK Market Base Rate (betfair.com returns 403 here)")
    print("    - Betfair's Points discount scheme, which only REDUCES the")
    print("      rate, so omitting it is conservative in the safe direction")
    print("    - any Matchbook page by direct fetch (TLS interception)")
    print("    - any volume tier or rebate for either venue")
    print()
    print("  So the arithmetic is run ACROSS the range, not at a point.")
    print()
    hdr = (f"  {'rate':>7s} {'eff odds':>9s} {'cost pp':>8s} "
           f"{'cost ROI':>9s} {'band cost+comm':>15s} {'ANOMALY NET':>12s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    mb = r["books"].get("TheOddsAPI-matchbook", {})
    sub = (mb.get("takeable") or mb.get("all")) if mb else None
    lb = mb.get("legset_band_takeable") or {}
    band_odds = lb.get("odds") or 1.355
    band_cost = lb.get("cost_pp")
    if band_cost is None:
        band_cost = MEASURED_BOOKS["band_cost_pp"]["TheOddsAPI-matchbook"]
    # the win rate used for the ROI form of the commission. OUR OWN in-band
    # win rate where we have one; otherwise the band's implied probability,
    # which is the only honest stand-in and is labelled as such.
    ex = r["our_exposure"]
    band_wr = ex["win_rate"] if ex["picks_in_band"] >= 30 else 1.0 / band_odds
    band_wr_src = ("measured on our own in-band picks, n="
                   f"{ex['picks_in_band']}" if ex["picks_in_band"] >= 30
                   else f"ASSUMED = the band's implied probability "
                        f"{100.0 / band_odds:.1f}%, because we hold only "
                        f"{ex['picks_in_band']} in-band picks")
    for rate in COMMISSION_RANGE:
        cc = commission_cost(band_odds, band_wr, rate)
        allin = band_cost + cc["pp"]
        print(f"  {100 * rate:6.1f}% {cc['effective_odds']:9.3f} "
              f"{cc['pp']:8.2f} {100 * cc['roi']:+8.2f}% {allin:15.2f} "
              f"{A - allin:+12.2f}")
    print()
    print(f"  at the in-band price {band_odds:.3f} (measured, leg-set grain)")
    print(f"  win rate {100 * band_wr:.1f}% -- {band_wr_src}")
    print()
    print("  OUR OWN EXPOSURE TO THAT BAND, which is a different question:")
    print(f"    {ex['picks_in_band']} of {r['population']} picks "
          f"({100 * ex['share']:.1f}%) sit at {ex['band']}")
    if ex["picks_in_band"]:
        print(f"    win rate {100 * ex['win_rate']:.1f}%, avg odds "
              f"{ex['avg_odds']:.3f}, ROI {100 * ex['roi']:+.2f}%")
    print("    > The anomaly is documented in a band our selection process")
    print("    > barely bets. Harvesting it is not a price change; it is a")
    print("    > DIFFERENT SELECTION, and nothing here measures that.")

    print()
    print("=" * 78)
    print("SIZE  -  Part B.  Measured, not read from documentation.")
    print("=" * 78)
    print("  UNAVAILABLE. There is no size, depth, liquidity or available-")
    print("  amount field anywhere in the record. Measured by inspection of:")
    for f in SIZE_FIELDS_SEARCHED:
        print(f"    - {f}")
    print("  `odds` carries exactly: match_id, bookmaker, market_type,")
    print("  selection, odds_value, opening_odds, timestamp, first_seen_at.")
    print("  `odds_snapshots` carries odds_value and observed_at. The")
    print("  TheOddsAPI parse reads `outcome['name']` and `outcome['price']`")
    print("  and nothing else -- so no size is dropped on the way in; none")
    print("  arrives.")
    print()
    print("  CONSEQUENCE, stated as the directive requires:")
    print("  > The exchange price has NO MEASURABLE DEPTH. So its cost is")
    print("  > NOT ESTABLISHED AS TAKEABLE -- the same status as the best")
    print("  > line, for the same reason, and this is the THIRD instance of")
    print("  > that error class. The exchange result is UNESTABLISHED, which")
    print("  > is NOT the same as refuted, and only one of those ends the")
    print("  > question.")

    print()
    print("  PRICES THAT EXISTED AT `taken_at`, AND THE COST ON THAT SET ONLY:")
    hdr = (f"    {'book':26s} {'n':>5s} {'dropped':>8s} {'cover':>7s} "
           f"{'band pp':>8s} {'band n':>7s} {'ROI':>8s} {'ANOM NET':>9s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for bm in sorted(r["books"], key=lambda b: MEASURED_BOOKS["cost_rel"][b]):
        b = r["books"][bm]
        t = b["takeable"]
        if not t:
            print(f"    {bm[:26]:26s}  NO takeable rows")
            continue
        lb = b["legset_band_takeable"]
        bc = lb["cost_pp"]
        net = f"{A - bc:+9.2f}" if bc is not None else "        -"
        print(f"    {bm[:26]:26s} {t['n']:5d} {b['dropped_not_at_pick']:8d} "
              f"{100 * t['coverage']:6.1f}% "
              f"{'-' if bc is None else f'{bc:8.2f}'} {lb['n']:7d} "
              f"{100 * t['roi']:+7.2f}% {net}"
              f"{'  [EX]' if b['is_exchange'] else ''}")

    print()
    print("=" * 78)
    print("COVERAGE  -  Part C.  And what the covered subset is ABOUT.")
    print("=" * 78)
    print(f"  current pick rate: {r['picks_per_day']:.2f} picks/day "
          f"({r['population']} picks over {r['days']} days, measured)")
    print(f"  to detect the -5.68% -> +3% move: n="
          f"{POWER_N_FOR_TARGET_MOVE} picks = {POWER_DAYS_AT_CURRENT_RATE} "
          f"days at that rate")
    print()
    hdr = (f"    {'policy':40s} {'cover':>7s} {'picks/day':>10s} "
           f"{'days for n=525':>15s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    for key in sorted(r["rates"], key=lambda k: -r["rates"][k]["coverage"]):
        v = r["rates"][key]
        bm = key.split(":")[0]
        if bm not in EX and not key.endswith(":all"):
            continue
        d = v["days_for_525"]
        print(f"    {key[:40]:40s} {100 * v['coverage']:6.1f}% "
              f"{v['picks_per_day']:10.2f} "
              f"{'-' if d is None else f'{d:15.0f}'}")
    print()
    print("  IS THE COVERED SUBSET REPRESENTATIVE? Market mix, exchange")
    print("  against the full record:")
    fullm = r["representativeness"]["full"]["market"]
    for bm in EX:
        m = r["representativeness"][bm]["market"]
        print(f"    {bm}")
        for fam in sorted(set(fullm) | set(m), key=lambda k: -fullm.get(k, 0)):
            fv, mv = 100 * fullm.get(fam, 0.0), 100 * m.get(fam, 0.0)
            flag = "  <-- ABSENT" if mv == 0 and fv > 2 else ""
            print(f"        {str(fam)[:18]:18s} full {fv:5.1f}%   "
                  f"exchange {mv:5.1f}%{flag}")
    print()
    print("  Band composition: the anomaly's 70-79% band, as a share of")
    print("  OUR picks -- which is zero everywhere, by configuration:")
    fb = r["representativeness"]["full"]["band"]
    print(f"    full record            {100 * fb.get('70-79%', 0.0):5.1f}%")
    for bm in EX:
        bb = r["representativeness"][bm]
        print(f"    {bm[:22]:22s} {100 * bb['band'].get('70-79%', 0.0):5.1f}%"
              f"   takeable only "
              f"{100 * bb['takeable_band'].get('70-79%', 0.0):5.1f}%")
    print()
    print("  Top leagues in the exchange subset vs the full record:")
    fl = r["representativeness"]["full"]["league"]
    for bm in EX[:1]:
        ml = r["representativeness"][bm]["league"]
        for lg in list(fl)[:8]:
            print(f"        {str(lg)[:28]:28s} full {100 * fl[lg]:5.1f}%   "
                  f"exchange {100 * ml.get(lg, 0.0):5.1f}%")

    print()
    print("=" * 78)
    print("ARITHMETIC  -  Part D.  Cost, coverage, commission, net.")
    print("=" * 78)
    print(f"  the anomaly: {A:.2f}pp  ({DOCUMENTED_ANOMALY['label']})")
    print()
    print(f"  1. CURRENT POLICY")
    print(f"       cost {MEASURED_BOOKS['band_cost_pp']['TAKEN_current_policy']:.2f}pp"
          f"   coverage 100%   commission none   "
          f"NET {A - MEASURED_BOOKS['band_cost_pp']['TAKEN_current_policy']:+.2f}pp")
    pin = r["books"].get("Pinnacle", {}).get("takeable") or {}
    pbc = pin.get("band_cost_pp") or MEASURED_BOOKS["band_cost_pp"]["Pinnacle"]
    print(f"  2. CHEAPEST TAKEABLE BOOKMAKER - Pinnacle")
    print(f"       cost {pbc:.2f}pp   coverage "
          f"{100 * (pin.get('coverage') or 0.509):.1f}%   commission none   "
          f"NET {A - pbc:+.2f}pp")
    print(f"  3. EXCHANGE - Matchbook, net of published commission,")
    print(f"     on prices that existed at pick time only")
    for rate in (0.02, 0.06):
        cc = commission_cost(band_odds, band_wr, rate)
        allin = band_cost + cc["pp"]
        print(f"       cost {band_cost:.2f}pp + commission {cc['pp']:.2f}pp "
              f"at {100 * rate:.0f}% = {allin:.2f}pp")
        print(f"       coverage {100 * (sub or {}).get('coverage', 0):.1f}% "
              f"of picks   NET {A - allin:+.2f}pp")
    print()
    print("  ALL THREE LINES ARE IN A BAND WE HOLD ZERO PICKS IN.")
    print("  The anomaly is a 70-79% figure; `betting.min_odds: 1.50` caps us")
    print("  at 66.7%. So here is the fourth line, which is the only one")
    print("  about prices we actually take -- and it has NO documented")
    print("  inefficiency to net against, because none exists for our band:")
    print()
    oc = MEASURED_TERMS["our_cost_pp"]
    cov = MEASURED_TERMS["takeable_cover_pick_share"]
    hdr = (f"    {'venue':26s} {'our cost pp':>12s} {'+comm @2%':>10s} "
           f"{'= all-in':>9s} {'coverage':>9s} {'vs policy':>10s}")
    print(hdr)
    print("    " + "-" * (len(hdr) - 4))
    pol = oc["TAKEN_current_policy"]
    for bm in sorted(oc, key=lambda b: oc[b]):
        if bm == "TAKEN_current_policy":
            print(f"    {'CURRENT POLICY':26s} {pol:12.2f} {'n/a':>10s} "
                  f"{pol:9.2f} {100 * cov[bm]:8.1f}% {'-':>10s}")
            continue
        isx = bm in MEASURED_BOOKS["exchanges"]
        c = commission_cost(
            r["books"][bm]["takeable"]["our_odds_median"],
            r["books"][bm]["takeable"]["win_rate"], 0.02)["pp"] if isx else 0.0
        allin = oc[bm] + c
        print(f"    {bm[:26]:26s} {oc[bm]:12.2f} "
              f"{('%.2f' % c) if isx else 'n/a':>10s} {allin:9.2f} "
              f"{100 * cov[bm]:8.1f}% {pol - allin:+9.2f}pp"
              f"{'  [EX]' if isx else ''}")
    print()
    print("  > A cost saving is not an edge. These are reductions in what we")
    print("  > pay, not inefficiencies we collect, and the largest one that")
    print("  > is ESTABLISHED is Pinnacle's 1.17pp on 50.9% of picks.")
    print()
    print("  BUT LINE 3 IS NOT ESTABLISHED. Its depth is unmeasured, so the")
    print("  price may not exist at any stake that matters. Part B's verdict")
    print("  governs: UNESTABLISHED, not positive.")
    print()
    print("  THE QUESTION THAT IS NIKI'S, NOT A MEASUREMENT:")
    print("  Which of these venues can he actually use from Bulgaria? Listed,")
    print("  with what each would require, and stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
