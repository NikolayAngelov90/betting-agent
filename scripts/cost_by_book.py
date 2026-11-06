"""COST BY BOOK — the last unmeasured term in the price problem.

H-MIX is refuted on cost: every reachable market costs 5.74–7.67% relative
and the cheapest needs 5.43% edge to break even. That is a price problem, and
one term in it has never been measured: **the overround of the individual
books we price from.** The pricer takes a cross-book MEDIAN, so no single
book's cost has ever been the planning figure.

MEASUREMENT ONLY. Adopts nothing, moves no threshold, touches no registered
statistic.

UNITS, STATED ONCE AND ON EVERY NUMBER. Two quantities are both written as a
percentage and differ by a factor of the odds:

    relative margin      sum(1/o) - 1          e.g. 6.40%
    probability points   quoted_prob - fair    e.g. 3.70pp

They are the same fact at a given price. Conflating them already cost one
reading, so every figure here carries its unit and the conversion is printed.

THE BAND MATTERS AS MUCH AS THE BOOK. The documented anomaly is stated for
a 70-79% implied band, so the comparator is each book's margin IN THAT BAND,
converted at the in-band leg's own price. A book's average margin converted
at the population's average odds is the wrong number twice over, and it
inverts the sign of the netting. See ANOMALY_BAND and the known-wrong control
in `tests/test_cost_by_book.py`.

    python -m scripts.cost_by_book            # Parts A, B, C, D
    python -m scripts.cost_by_book --json
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

#: The external anomaly, kept labelled. It is NOT ours and NOT re-derived.
DOCUMENTED_ANOMALY = {
    "value_pp": 3.74,
    "label": ("external, Footiqo, 2025/26, ONE bookmaker (1xBet), five "
              "leagues, n=134 at 70-79%"),
}

#: The registered ROI target.
ROI_TARGET = 0.03

#: WHAT WE PAY TODAY. Measured 2026-11-06 by `scripts/market_mix.py` on the
#: whole settled paper population (n=1,639, grain=PICK): the n-weighted mean
#: overround of the TAKEN price set -- the cross-book median assembled leg by
#: leg, which is what `_find_best_odds` actually returns.
#:
#: BOTH UNITS, because they are the same measurement and have been read as a
#: contradiction once already: 6.40% RELATIVE MARGIN converts to
#: `cost/(1+cost) * odds * 100` = 3.70 PROBABILITY POINTS at avg odds 1.628.
MEDIAN_POLICY = {
    "cost_rel": 0.0640,
    "cost_pp": 3.70,
    "coverage": 1.0,          # by construction: it IS the policy
    "n": 1639,
    "measured": "2026-11-06",
    "provenance": "MEASURED, scripts/market_mix.py, grain=PICK",
}

#: Bootstrap draws, resampling FIXTURES. Same method as the market mix.
BOOTSTRAP_DRAWS = 4000

#: `odds_snapshots` is append-only and starts here; freshness before this date
#: is not measurable from it. Stated as a coverage limit, not worked around.
SNAPSHOTS_BEGIN = "2026-08-25"

#: The anomaly's own probability band, as its source states it: 70-79%
#: implied, which is decimal odds 1/0.79 .. 1/0.70.
ANOMALY_BAND = (1.0 / 0.79, 1.0 / 0.70)


#: WHAT THIS STAGE MEASURED. Pinned so the figures cannot drift silently and
#: so a reader sees the units and the coverage beside every one of them.
#: provenance: MEASURED 2026-11-06, `scripts/cost_by_book.py`, grain stated
#: per figure. Population: 1,639 settled paper picks over 1,629 fixtures.
MEASURED_BOOKS = {
    "measured": "2026-11-06",
    "population": 1639,
    "fixtures": 1629,
    "books_priced": 29,
    "avg_taken_odds": 1.627,
    # cost in RELATIVE MARGIN, each book's own overround, grain=LEG SET
    "cost_rel": {
        "TheOddsAPI-matchbook": 0.0100,
        "TheOddsAPI-betfair_ex_eu": 0.0103,
        "TheOddsAPI-pinnacle": 0.0336,
        "Pinnacle": 0.0457,
        "Bet365": 0.0657,
        "1xBet": 0.0807,
    },
    # coverage as a SHARE OF PICKS -- never reported without this
    "cover_pick_share": {
        "TheOddsAPI-matchbook": 0.226,
        "TheOddsAPI-betfair_ex_eu": 0.084,
        "TheOddsAPI-pinnacle": 0.138,
        "Pinnacle": 0.509,
        "Bet365": 0.843,
        "1xBet": 0.768,
    },
    # cost in PROBABILITY POINTS **in the anomaly's own 70-79% band**,
    # converted at the in-band leg's own price (median odds 1.33-1.36).
    # This is the comparator the anomaly must be netted against, and
    # converting it at the population's average odds of 1.627 instead is a
    # unit error that moves every figure by about 3pp.
    "band_cost_pp": {
        "TheOddsAPI-matchbook": 0.59,
        "TheOddsAPI-betfair_ex_eu": 0.67,
        "TheOddsAPI-pinnacle": 2.77,
        "Pinnacle": 3.69,
        "Bet365": 4.72,
        "1xBet": 5.77,
        "TAKEN_current_policy": 4.70,
    },
    # share of matched prices whose `odds` row was written AFTER the pick
    "later_write_share": {
        "TheOddsAPI-matchbook": 0.412,
        "TheOddsAPI-betfair_ex_eu": 0.416,
        "TheOddsAPI-pinnacle": 0.405,
        "Pinnacle": 0.0,
        "Bet365": 0.0,
        "1xBet": 0.0,
    },
    "later_write_median_minutes": 400.6,
    # 1X2 leg sets implying under 100% -- the stale-quote signature that
    # disqualified the best line
    "sub100_1x2_share": {
        "TheOddsAPI-matchbook": 0.0054,
        "TheOddsAPI-betfair_ex_eu": 0.0,
        "TheOddsAPI-pinnacle": 0.0,
        "Pinnacle": 0.0,
        "Bet365": 0.0,
        "1xBet": 0.0,
        "BEST_LINE_for_contrast": 0.2128,
    },
    # the PAIRED price delta: the SAME picks, book price vs taken price
    "paired_delta_pp": {
        "TheOddsAPI-matchbook": 4.25,
        "TheOddsAPI-betfair_ex_eu": 2.87,
        "TheOddsAPI-pinnacle": 1.60,
        "Pinnacle": 0.98,
        "Bet365": -0.14,
        "1xBet": -0.04,
    },
    "paired_delta_pp_takeable_only": {
        "TheOddsAPI-matchbook": 4.32,
        "TheOddsAPI-betfair_ex_eu": 2.99,
        "TheOddsAPI-pinnacle": 1.86,
        "Pinnacle": 0.98,
    },
    "exchanges": ("TheOddsAPI-matchbook", "TheOddsAPI-betfair_ex_eu"),
    "breakeven_commission": {
        "TheOddsAPI-matchbook": 0.156,
        "TheOddsAPI-betfair_ex_eu": 0.155,
        "TheOddsAPI-pinnacle": 0.049,
        "Pinnacle": 0.003,
    },
    "verdict": (
        "At every book whose price is ESTABLISHED as having existed at pick "
        "time, the documented anomaly net of that book's band cost is at "
        "best +0.05pp (Pinnacle, 50.9% coverage) and is NEGATIVE at the book "
        "where the anomaly was documented (1xBet, -2.03pp). The only "
        "positive nets are at two EXCHANGES whose commission is NOT in the "
        "quoted price and is UNMEASURED here -- so 0.59pp is not their cost "
        "and must not become a planning figure."),
}


def _load_env() -> None:
    """INSIDE a function, never at import."""
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


def _pp_at(cost_rel: float, odds: float) -> float:
    """A relative margin converted to PROBABILITY POINTS at a given price."""
    quoted = 1.0 / odds
    fair = quoted / (1.0 + cost_rel)
    return (quoted - fair) * 100.0


def _bootstrap_roi(rows: List[Tuple[int, float, int]]) -> Optional[list]:
    """95% CI on ROI, resampling FIXTURES. Picks on one fixture are not
    independent: the per-match cap allows two and they share match state."""
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
    return [draws[int(0.025 * len(draws))],
            draws[int(0.975 * len(draws)) - 1]]


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
                   sp.result, sp.created_at,
                   (SELECT po.taken_at FROM pick_observations po
                     WHERE po.pick_id = sp.id AND po.attribution = 'final'
                     ORDER BY po.taken_at LIMIT 1) AS taken_at
            FROM saved_picks sp
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
                   timestamp, first_seen_at
            FROM odds
            WHERE match_id = ANY(:ids)
              AND odds_value IS NOT NULL AND odds_value > 1.0
              AND bookmaker IS NOT NULL
        """), {"ids": mids}).fetchall()

    pop = len(picks)
    fixtures = len(mids)

    # ── the per-book price book ────────────────────────────────────────────
    book: Dict[Tuple[int, str], Dict[str, Dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict))
    seen_at: Dict[str, List] = defaultdict(list)
    # `instant[(mid, bm, mt, sel)] = (timestamp, first_seen_at)`. `odds` is
    # updated IN PLACE, so `timestamp` is when the row was LAST written -- it
    # can therefore be AFTER the pick, which is precisely the Part C question.
    instant: Dict[Tuple[int, str, str, str], Tuple] = {}
    for mid, bm, mt, sel, val, ts, fsa in odds_rows:
        book[(mid, bm)][mt][sel] = float(val)
        instant[(mid, bm, mt, sel)] = (ts, fsa)
        if ts:
            seen_at[bm].append(ts)

    # ── PART A: each book's own overround, per market ──────────────────────
    per_book: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list))
    sub100: Dict[str, List[int]] = defaultdict(list)
    band_cost: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for (mid, bm), by_market in book.items():
        for market, by_sel in by_market.items():
            spec = get_spec(market)
            if spec is None or spec.overlapping:
                continue          # double chance is derived, not summed
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
                    # a leg set implying under 100% is the stale-quote
                    # signature that disqualified the best line
                    if market == "1X2":
                        sub100[bm].append(1 if orr < 1.0 else 0)
                    if not (spec.overround_min <= orr <= spec.overround_max):
                        continue
                    per_book[bm][market].append(orr - 1.0)
                    # does this leg set contain a favourite in the anomaly's
                    # own 70-79% band? If so its margin is the comparator the
                    # anomaly must be netted against.
                    inband = [o for o in prices
                              if ANOMALY_BAND[0] <= o <= ANOMALY_BAND[1]]
                    if inband:
                        # the margin is converted to probability points AT
                        # THE IN-BAND LEG'S OWN PRICE. Converting a band
                        # margin at the population's average odds (1.627,
                        # against a band of 1.27-1.43) is a UNIT ERROR and it
                        # inverts the sign of the netting.
                        band_cost[bm].append((orr - 1.0, min(inband)))

    # ── coverage: did this book quote the SELECTION WE PICKED? ─────────────
    cover_picks: Dict[str, int] = defaultdict(int)
    cover_fix: Dict[str, set] = defaultdict(set)
    book_price: Dict[str, List[Tuple[int, float, int]]] = defaultdict(list)
    book_pids: Dict[str, List[int]] = defaultdict(list)
    matched_key: Dict[Tuple[str, int], Tuple] = {}
    for _pid, mid, market, selection, _odds, result, _cr, _ta in picks:
        canon = canonical(market, selection)
        if canon is None:
            continue
        fam, canon_sel = canon
        aliases = set(selection_aliases(canon_sel))
        for (m2, bm), by_market in book.items():
            if m2 != mid:
                continue
            by_sel = by_market.get(fam) or {}
            hitkv = next(((k, v) for k, v in by_sel.items()
                          if k in aliases), None)
            if hitkv:
                hitkey, hit = hitkv
                cover_picks[bm] += 1
                cover_fix[bm].add(mid)
                book_price[bm].append(
                    (mid, float(hit), 1 if result == "win" else 0))
                book_pids[bm].append(_pid)
                matched_key[(bm, _pid)] = (mid, fam, hitkey)

    # ── assemble Part A ───────────────────────────────────────────────────
    avg_odds = statistics.mean(float(p[4]) for p in picks)
    books = {}
    for bm, by_market in per_book.items():
        allc = [c for v in by_market.values() for c in v]
        if not allc:
            continue
        med = statistics.median(allc)
        s1 = sub100.get(bm) or []
        books[bm] = {
            "cost_rel": med,
            "cost_pp_at_our_odds": _pp_at(med, avg_odds),
            "obs": len(allc),
            "p25": statistics.quantiles(allc, n=4)[0] if len(allc) > 3 else None,
            "p75": statistics.quantiles(allc, n=4)[2] if len(allc) > 3 else None,
            "by_market": {k: {"median": statistics.median(v), "n": len(v)}
                          for k, v in sorted(by_market.items())},
            "cover_picks": cover_picks.get(bm, 0),
            "cover_pick_share": cover_picks.get(bm, 0) / pop,
            "cover_fixtures": len(cover_fix.get(bm, ())),
            "cover_fixture_share": len(cover_fix.get(bm, ())) / fixtures,
            "first_seen": min(seen_at[bm]) if seen_at.get(bm) else None,
            "last_seen": max(seen_at[bm]) if seen_at.get(bm) else None,
            "sub100_1x2_n": len(s1),
            "sub100_1x2_share": (sum(s1) / len(s1)) if s1 else None,
            "required_edge": med / (1.0 + med),
            "band_cost_rel": (statistics.median(
                [c for c, _o in band_cost[bm]])
                if band_cost.get(bm) else None),
            "band_cost_pp": (statistics.median(
                [_pp_at(c, o) for c, o in band_cost[bm]])
                if band_cost.get(bm) else None),
            "band_odds_median": (statistics.median(
                [o for _c, o in band_cost[bm]])
                if band_cost.get(bm) else None),
            "band_n": len(band_cost.get(bm) or []),
        }

    # ── PART B: the single-book counterfactual ────────────────────────────
    #
    # PAIRED ON THE SAME PICKS. A book's ROI on ITS OWN covered subset is not
    # comparable to the full-population ROI -- they are different picks. So
    # each candidate is reported beside what THOSE SAME PICKS returned at the
    # TAKEN (median) price, and the delta is then attributable to the price
    # rather than to which selections the book happened to quote.
    taken_by_pick: Dict[int, Tuple[int, float, int]] = {}
    taken_at_by_pick: Dict[int, object] = {}
    for pid, mid, _mk, _sel, odds, result, _cr, _ta in picks:
        taken_by_pick[pid] = (mid, float(odds), 1 if result == "win" else 0)
        taken_at_by_pick[pid] = _ta
    ranked = sorted(books.items(), key=lambda kv: kv[1]["cost_rel"])
    cheapest = [b for b, _v in ranked[:3]]
    # plus the HIGH-COVERAGE comparators: a cheap book on few picks is not a
    # cheap policy, so the candidates include the best-covered books.
    by_cover = sorted(books.items(), key=lambda kv: -kv[1]["cover_pick_share"])
    candidates = list(dict.fromkeys(cheapest + [b for b, _ in by_cover[:3]]))
    cf = {}
    for bm in candidates:
        rows = book_price.get(bm) or []
        if not rows:
            cf[bm] = {"n": 0}
            continue
        n = len(rows)
        roi = statistics.mean(o * w - 1.0 for _m, o, w in rows)
        cost = books[bm]["cost_rel"]
        # the SAME picks at the taken price
        pids = book_pids.get(bm) or []
        paired = [taken_by_pick[pid] for pid in pids if pid in taken_by_pick]
        roi_taken = (statistics.mean(o * w - 1.0 for _m, o, w in paired)
                     if paired else None)
        cf[bm] = {
            "n": n, "uncovered": pop - n, "coverage": n / pop,
            "win_rate": sum(w for _m, _o, w in rows) / n,
            "avg_odds": statistics.mean(o for _m, o, _w in rows),
            "roi": roi, "ci95": _bootstrap_roi(rows),
            "roi_same_picks_at_taken": roi_taken,
            "delta_vs_taken": (roi - roi_taken) if roi_taken is not None
            else None,
            "cost_rel": cost,
            "cost_pp": _pp_at(cost, statistics.mean(o for _m, o, _w in rows)),
            "required_for_target": cost / (1.0 + cost) + ROI_TARGET,
            "anomaly_net_pp": DOCUMENTED_ANOMALY["value_pp"]
            - _pp_at(cost, statistics.mean(o for _m, o, _w in rows)),
            "is_exchange": bm.lower().find("betfair") >= 0
            or bm.lower().find("matchbook") >= 0,
        }
    # == PART C: IS THE CHEAP PRICE TAKEABLE? ==============================
    #
    # Three separate questions, and they have different answers:
    #
    #   1. Was the book PRESENT at pick time, or does it appear only in later
    #      snapshots? `odds` is written in place, so a row whose `timestamp`
    #      is AFTER the pick is a price that did not exist when the pick was
    #      made. `first_seen_at` (34% populated) says when the book first
    #      quoted the selection at all.
    #   2. How OLD was the quote at `taken_at`? Measured ONLY for picks whose
    #      matched row has `timestamp <= taken_at`; for the rest the age is
    #      not a small number, it is UNDEFINED, and it is reported as such.
    #   3. Are the book's own three prices internally coherent -- the
    #      sub-100% test that disqualified the best line. Already in Part A.
    #
    # UNEVALUATED is a reported outcome here, never a value: a pick with no
    # `taken_at` is not a fresh quote and not a stale one.
    takeable = {}
    for bm in candidates:
        after, before, no_taken, no_ts = 0, 0, 0, 0
        ages, fsa_before, fsa_unknown = [], 0, 0
        lateness = []
        for pid in (book_pids.get(bm) or []):
            ta = taken_at_by_pick.get(pid)
            key = matched_key.get((bm, pid))
            if key is None:
                continue
            mid, fam, hitkey = key
            ts, fsa = instant.get((mid, bm, fam, hitkey), (None, None))
            if ta is None:
                no_taken += 1
                continue
            if ts is None:
                no_ts += 1
                continue
            if ts > ta:
                after += 1
                lateness.append((ts - ta).total_seconds() / 3600.0)
            else:
                before += 1
                ages.append((ta - ts).total_seconds() / 3600.0)
            if fsa is None:
                fsa_unknown += 1
            elif fsa <= ta:
                fsa_before += 1
        n = len(book_pids.get(bm) or [])
        ev = before + after
        takeable[bm] = {
            "covered": n,
            "with_taken_at": ev,
            "UNEVALUATED_no_taken_at": no_taken,
            "UNEVALUATED_no_timestamp": no_ts,
            "quote_existed_at_pick": before,
            "quote_is_a_LATER_snapshot": after,
            "later_share": (after / ev) if ev else None,
            "age_h_median": statistics.median(ages) if ages else None,
            "age_h_p90": (statistics.quantiles(ages, n=10)[8]
                          if len(ages) > 9 else None),
            "age_h_max": max(ages) if ages else None,
            "later_by_h_median": (statistics.median(lateness)
                                  if lateness else None),
            "later_by_h_max": max(lateness) if lateness else None,
            "book_present_by_first_seen": fsa_before,
            "first_seen_unknown": fsa_unknown,
            "sub100_1x2_share": books[bm]["sub100_1x2_share"],
            "sub100_1x2_n": books[bm]["sub100_1x2_n"],
            "first_seen": books[bm]["first_seen"],
            "last_seen": books[bm]["last_seen"],
        }

    # == THE TAKEN PRICE'S COST IN THE ANOMALY'S OWN BAND =================
    #
    # Line 1 of the arithmetic must be like-for-like with lines 2 and 3: the
    # anomaly is BAND-SPECIFIC, so the comparator is what the CURRENT POLICY
    # pays in that band, not its population average. The taken price is the
    # cross-book MEDIAN per leg (`_find_best_odds`), assembled leg by leg.
    legs_by_sel: Dict[Tuple[int, str, str], List[float]] = defaultdict(list)
    for (mid, _bm), by_market in book.items():
        for market, by_sel in by_market.items():
            for sel, val in by_sel.items():
                legs_by_sel[(mid, market, sel)].append(val)
    taken_sel: Dict[Tuple[int, str], Dict[str, float]] = defaultdict(dict)
    for (mid, market, sel), vals in legs_by_sel.items():
        taken_sel[(mid, market)][sel] = statistics.median(vals)
    taken_band: List[Tuple[float, float]] = []
    taken_all: List[float] = []
    for (mid, market), by_sel in taken_sel.items():
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
                taken_all.append(orr - 1.0)
                ib = [o for o in prices
                      if ANOMALY_BAND[0] <= o <= ANOMALY_BAND[1]]
                if ib:
                    taken_band.append((orr - 1.0, min(ib)))
    taken_policy = {
        "all_cost_rel": statistics.median(taken_all) if taken_all else None,
        "all_n": len(taken_all),
        "band_cost_rel": (statistics.median([c for c, _o in taken_band])
                          if taken_band else None),
        "band_cost_pp": (statistics.median([_pp_at(c, o)
                                            for c, o in taken_band])
                         if taken_band else None),
        "band_odds_median": (statistics.median([o for _c, o in taken_band])
                             if taken_band else None),
        "band_n": len(taken_band),
        "provenance": "MEASURED 2026-11-06, grain=LEG SET, cross-book median",
    }

    # == THE BREAK-EVEN EXCHANGE COMMISSION ===============================
    #
    # An exchange's commission is charged on NET WINNINGS and is NOT in the
    # quoted price, so its quoted margin is NOT its cost and must not become
    # a planning figure. Rather than assume a rate, solve for the rate at
    # which the anomaly is exactly erased: `c` such that the all-in cost in
    # the band equals the documented anomaly. Niki compares that to the rate
    # he can actually get.
    def _breakeven_commission(cost_rel: float, odds: float,
                              target_pp: float) -> Optional[float]:
        fair = (1.0 / odds) / (1.0 + cost_rel)
        want = fair + target_pp / 100.0        # implied prob we must not pass
        if want <= 0 or want >= 1:
            return None
        o2 = 1.0 / want                        # effective odds after rake
        if odds <= 1.0 or o2 <= 1.0:
            return None
        c = 1.0 - (o2 - 1.0) / (odds - 1.0)
        return c

    for bm, v in books.items():
        v["breakeven_commission"] = None
        v["later_write_ok"] = (
            takeable[bm]["later_share"] in (None, 0.0)
            if bm in takeable else None)
        if v["band_cost_rel"] is not None and v["band_odds_median"]:
            v["breakeven_commission"] = _breakeven_commission(
                v["band_cost_rel"], v["band_odds_median"],
                DOCUMENTED_ANOMALY["value_pp"])

    # == PART B, RESTRICTED TO PRICES THAT EXISTED AT PICK TIME ===========
    #
    # `odds` is written in place. A matched row whose `timestamp` is AFTER the
    # pick is a price from a LATER scrape -- it is not a price we could have
    # taken, and including it in a counterfactual ROI makes the same mistake
    # as the best line: a price assembled from a moment that was not ours.
    # So each candidate is recomputed on its OWN takeable subset, and both
    # figures are reported. The restriction is not applied to pick selection:
    # the uncovered picks are still counted, never dropped.
    cf_takeable = {}
    for bm in candidates:
        rows, pids_ok = [], []
        for pid in (book_pids.get(bm) or []):
            key = matched_key.get((bm, pid))
            ta = taken_at_by_pick.get(pid)
            if key is None or ta is None:
                continue
            mid, fam, hitkey = key
            ts, _fsa = instant.get((mid, bm, fam, hitkey), (None, None))
            if ts is None or ts > ta:
                continue                      # not takeable: a later write
            price = book[(mid, bm)][fam][hitkey]
            _m, _o, w = taken_by_pick[pid]
            rows.append((mid, float(price), w))
            pids_ok.append(pid)
        if not rows:
            cf_takeable[bm] = {"n": 0}
            continue
        n = len(rows)
        paired = [taken_by_pick[pid] for pid in pids_ok]
        cf_takeable[bm] = {
            "n": n,
            "dropped_as_later_write": len(book_pids.get(bm) or []) - n,
            "uncovered": pop - n,
            "coverage": n / pop,
            "win_rate": sum(w for _m, _o, w in rows) / n,
            "avg_odds": statistics.mean(o for _m, o, _w in rows),
            "roi": statistics.mean(o * w - 1.0 for _m, o, w in rows),
            "ci95": _bootstrap_roi(rows),
            "roi_same_picks_at_taken": statistics.mean(
                o * w - 1.0 for _m, o, w in paired),
        }
        cf_takeable[bm]["delta_vs_taken"] = (
            cf_takeable[bm]["roi"]
            - cf_takeable[bm]["roi_same_picks_at_taken"])

    # == PART D: THREE LINES OF ARITHMETIC ================================
    # Nothing here is adjusted to make it come out. Each line is
    # `documented anomaly - cost`, in PROBABILITY POINTS at our average
    # taken odds, with the cost's coverage beside it.
    median_policy_pp = MEDIAN_POLICY["cost_pp"]
    arithmetic = {
        "unit": "probability points at our n-weighted avg taken odds "
                f"{avg_odds:.3f}",
        "anomaly_pp": DOCUMENTED_ANOMALY["value_pp"],
        "anomaly_label": DOCUMENTED_ANOMALY["label"],
        "line1_current_policy": {
            "cost_pp_avg_odds": MEDIAN_POLICY["cost_pp"],
            "cost_pp_in_band": taken_policy["band_cost_pp"],
            "band_n": taken_policy["band_n"],
            "coverage": 1.0,
            "anomaly_net_pp_in_band": (
                DOCUMENTED_ANOMALY["value_pp"] - taken_policy["band_cost_pp"]
                if taken_policy["band_cost_pp"] is not None else None),
        },
        "taken_policy": taken_policy,
    }
    return {"population": pop, "fixtures": fixtures, "avg_odds": avg_odds,
            "books": books, "ranked": [b for b, _ in ranked],
            "cheapest3": cheapest, "counterfactual": cf,
            "takeable": takeable, "arithmetic": arithmetic,
            "cf_takeable": cf_takeable, "median_policy": MEDIAN_POLICY}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    r = measure()
    if a.json:
        import json
        print(json.dumps(r, indent=1, default=str))
        return 0

    A = r["arithmetic"]["anomaly_pp"]
    EX = MEASURED_BOOKS["exchanges"]

    print("=" * 78)
    print("COST BY BOOK  -  Part A")
    print("=" * 78)
    print(f"  {r['population']} settled paper picks over {r['fixtures']} "
          f"fixtures; {len(r['books'])} books priced; our n-weighted avg "
          f"taken odds {r['avg_odds']:.3f}")
    print("  UNITS. `cost rel` is RELATIVE MARGIN (sum of 1/odds, minus 1).")
    print("  `avg pp` and `band pp` are PROBABILITY POINTS: the first")
    print("  converted at our average odds, the second at the in-band leg's")
    print("  own price. They are NOT interchangeable and differ by ~3pp.")
    print()
    hdr = (f"  {'book':26s} {'cost rel':>9s} {'avg pp':>7s} {'p25':>7s} "
           f"{'p75':>7s} {'obs':>6s} {'picks':>6s} {'=share':>7s} "
           f"{'fixtures':>8s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for bm in r["ranked"]:
        v = r["books"][bm]
        p25 = f"{100 * v['p25']:.2f}%" if v["p25"] is not None else "-"
        p75 = f"{100 * v['p75']:.2f}%" if v["p75"] is not None else "-"
        print(f"  {bm[:26]:26s} {100 * v['cost_rel']:8.2f}% "
              f"{v['cost_pp_at_our_odds']:6.2f} {p25:>7s} {p75:>7s} "
              f"{v['obs']:6d} {v['cover_picks']:6d} "
              f"{100 * v['cover_pick_share']:6.1f}% {v['cover_fixtures']:8d}")
    print()
    print("  THE THREE CHEAPEST, WITH COVERAGE BESIDE THEM:")
    for bm in r["cheapest3"]:
        v = r["books"][bm]
        note = ("  <-- EXCHANGE: commission is charged on NET WINNINGS, is "
                "NOT in this price, and is NOT measured here")
        print(f"    {bm:26s} {100 * v['cost_rel']:5.2f}% rel  on "
              f"{100 * v['cover_pick_share']:4.1f}% of picks"
              f"{note if bm in EX else ''}")

    print()
    print("=" * 78)
    print("SINGLE-BOOK COUNTERFACTUAL  -  Part B")
    print("=" * 78)
    print("  LIMITATION, STATED. This is a COUNTERFACTUAL: the same picks at")
    print("  a different price. For it to transfer, three things must hold.")
    print("  (1) the book quoted that price AT PICK TIME -- MEASURED below;")
    print("  (2) it was available in the size wanted -- NOT MEASURED;")
    print("  (3) the selection process is unchanged by reading one book's")
    print("      prices instead of a cross-book median -- NOT MEASURED, and")
    print("      known false in direction: the pricer takes the median, so")
    print("      with one book there is no median to take.")
    print("  Uncovered picks are COUNTED, never dropped.")
    print()
    hdr = (f"  {'book':26s} {'n':>5s} {'uncov':>6s} {'cover':>6s} "
           f"{'ROI@book':>9s} {'ROI@taken':>10s} {'DELTA':>7s} "
           f"{'CI95 on ROI':>19s} {'need':>7s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for bm, c in sorted(r["counterfactual"].items(),
                        key=lambda kv: kv[1].get("cost_rel", 9)):
        ci = c.get("ci95")
        cis = f"[{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}]" if ci else "-"
        print(f"  {bm[:26]:26s} {c['n']:5d} {c['uncovered']:6d} "
              f"{100 * c['coverage']:5.1f}% {100 * c['roi']:+8.2f}% "
              f"{100 * c['roi_same_picks_at_taken']:+9.2f}% "
              f"{100 * c['delta_vs_taken']:+6.2f}% {cis:>19s} "
              f"{100 * c['required_for_target']:6.2f}%")
    print()
    print("  `DELTA` is the PRICE effect isolated: the same picks, this")
    print("  book's price minus the taken price. `ROI@book` is not")
    print("  comparable across rows -- each is a different pick subset.")
    print()
    print("  THE SAME, RESTRICTED TO PRICES THAT EXISTED AT PICK TIME:")
    for bm, c in sorted(r["cf_takeable"].items(),
                        key=lambda kv: r["books"][kv[0]]["cost_rel"]):
        # `n` is always PRESENT -- set to 0 when the book has no takeable
        # rows -- so this is a count of zero, not a missing key. The
        # explicit comparison keeps those two apart.
        if c["n"] == 0:
            print(f"    {bm:26s}  NO takeable rows")
            continue
        ci = c.get("ci95")
        cis = f"[{100 * ci[0]:+.2f},{100 * ci[1]:+.2f}]" if ci else "-"
        print(f"    {bm[:26]:26s} n={c['n']:5d} "
              f"(-{c['dropped_as_later_write']:3d} later writes)  cover "
              f"{100 * c['coverage']:4.1f}%  ROI {100 * c['roi']:+7.2f}%  "
              f"DELTA {100 * c['delta_vs_taken']:+6.2f}%  CI {cis}")

    print()
    print("=" * 78)
    print("TAKEABLE  -  Part C")
    print("=" * 78)
    print("  Quote age is measured at `pick_observations.taken_at`, which is")
    print("  100% populated on this population -- so freshness IS per-pick")
    print("  measurable, which corrects a recorded limitation. `odds` is")
    print("  written IN PLACE, so a row whose `timestamp` is after the pick")
    print(f"  is a LATER price. (`odds_snapshots` begins {SNAPSHOTS_BEGIN};")
    print("  this measurement does not use it and is not limited by it.)")
    print()
    for bm in r["takeable"]:
        t = r["takeable"][bm]
        print(f"  {bm}{' [EXCHANGE]' if bm in EX else ''}")
        print(f"      covered {t['covered']} picks; with taken_at "
              f"{t['with_taken_at']}; UNEVALUATED "
              f"{t['UNEVALUATED_no_taken_at']} (no taken_at) + "
              f"{t['UNEVALUATED_no_timestamp']} (no timestamp)")
        ls, lm = t["later_share"], t["later_by_h_median"]
        tail = f", median {60 * lm:.0f} min late)" if lm else ")"
        print(f"      price existed at pick: {t['quote_existed_at_pick']}"
              f"   LATER write: {t['quote_is_a_LATER_snapshot']} "
              f"({'0.0%' if ls is None else f'{100 * ls:.1f}%'}{tail}")
        if t["age_h_median"] is not None:
            print(f"      quote AGE at pick: median "
                  f"{60 * t['age_h_median']:.0f} min, p90 "
                  f"{60 * t['age_h_p90']:.0f} min, max "
                  f"{60 * t['age_h_max']:.0f} min")
        s1 = t["sub100_1x2_share"]
        print(f"      1X2 leg sets implying UNDER 100%: "
              f"{'n/a' if s1 is None else f'{100 * s1:.2f}%'} "
              f"(n={t['sub100_1x2_n']})  [best line, for contrast: 21.28%]")
        print(f"      present in record {str(t['first_seen'])[:10]} .. "
              f"{str(t['last_seen'])[:10]}")
    print()
    print("  THE REACHABLE SET IS NOT A MEASUREMENT. These are the books")
    print("  whose prices this record can price a counterfactual at. Which")
    print("  of them Niki can actually hold an account with, at what stake,")
    print("  is HIS answer, not this script's. Listed, and stopped.")

    print()
    print("=" * 78)
    print("ARITHMETIC  -  Part D. Three lines, in PROBABILITY POINTS, in the")
    print("anomaly's OWN 70-79% band, each netted against the anomaly.")
    print("=" * 78)
    tp = r["arithmetic"]["taken_policy"]
    print(f"  the anomaly: {A:.2f}pp")
    print(f"      ({r['arithmetic']['anomaly_label']})")
    print()
    print(f"  1. CURRENT MEDIAN POLICY       cost {tp['band_cost_pp']:5.2f}pp"
          f"   coverage 100% by construction")
    print(f"     (n={tp['band_n']} in-band leg sets)      "
          f"ANOMALY NET {A - tp['band_cost_pp']:+5.2f}pp")
    print()
    best, bestv = None, None
    for bm, v in r["books"].items():
        if bm in EX or v["band_cost_pp"] is None:
            continue
        if v["cover_pick_share"] < 0.20 or not v["later_write_ok"]:
            continue
        if bestv is None or v["band_cost_pp"] < bestv["band_cost_pp"]:
            best, bestv = bm, v
    print(f"  2. CHEAPEST BOOK whose price EXISTED at pick time, on >=20% of")
    print(f"     picks, not an exchange:")
    print(f"     {best:26s}    cost {bestv['band_cost_pp']:5.2f}pp"
          f"   coverage {100 * bestv['cover_pick_share']:.1f}% of picks")
    print(f"                                   ANOMALY NET "
          f"{A - bestv['band_cost_pp']:+5.2f}pp")
    print()
    ch = r["cheapest3"][0]
    cv = r["books"][ch]
    print(f"  3. CHEAPEST BOOK OF ANY KIND:")
    print(f"     {ch:26s}    cost {cv['band_cost_pp']:5.2f}pp"
          f"   coverage {100 * cv['cover_pick_share']:.1f}% of picks")
    print(f"                                   ANOMALY NET "
          f"{A - cv['band_cost_pp']:+5.2f}pp")
    print(f"     BUT IT IS AN EXCHANGE. Commission is charged on net")
    print(f"     winnings, is not in that price, and is NOT measured here,")
    print(f"     so {cv['band_cost_pp']:.2f}pp is NOT its cost. Instead of")
    print(f"     assuming a rate: the commission that would exactly erase")
    print(f"     the anomaly is "
          f"{100 * cv['breakeven_commission']:.1f}%.")
    print()
    print(f"  AND AT THE ANOMALY'S OWN SOURCE: 1xBet charges "
          f"{r['books']['1xBet']['band_cost_pp']:.2f}pp in that band, so the")
    print(f"  anomaly nets {A - r['books']['1xBet']['band_cost_pp']:+.2f}pp "
          f"at the one book where it was documented.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
