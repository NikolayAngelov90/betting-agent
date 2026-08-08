"""Run the immutable baseline experiment and snapshot the result.

    python -m scripts.run_baseline [--window 60] [--out data/baselines/<name>.json]

Produces the Phase 1 table: market-only, raw market, Poisson-only, Elo-only,
Poisson+Elo, and market+model blends at several weights, all scored on the same
chronological out-of-sample fixtures.

Read-only against the database. Snapshots are written once and not edited; a
later run writes a new file and the two are diffed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

import numpy as np
from scipy.stats import poisson as _pois

from src.data.database import get_db
from src.data.models import Match, Odds
from src.evaluation.baseline import (
    Candidate,
    MatchRow,
    compare,
    format_table,
    market_consensus,
    market_raw,
    walk_forward,
)
from src.utils.logger import get_logger

logger = get_logger()

MAX_GOALS = 10
SHRINK_CAP = 100


# --------------------------------------------------------------------- loading

def load_rows(since: date) -> list:
    db = get_db()
    rows: dict = {}
    with db.get_session() as session:
        q = session.query(
            Match.id, Match.match_date, Match.home_team_id, Match.away_team_id,
            Match.home_goals, Match.away_goals, Match.league,
        ).filter(
            Match.is_fixture == False,  # noqa: E712
            Match.home_goals.isnot(None),
            Match.away_goals.isnot(None),
            Match.match_date >= since,
        ).order_by(Match.match_date)
        for m in q.all():
            d = m.match_date.date() if hasattr(m.match_date, "date") else m.match_date
            rows[m.id] = MatchRow(
                id=m.id, match_date=d, home_team_id=m.home_team_id,
                away_team_id=m.away_team_id, home_goals=m.home_goals,
                away_goals=m.away_goals, league=m.league or "unknown",
            )

    with db.get_session() as session:
        q = session.query(
            Odds.match_id, Odds.bookmaker, Odds.selection, Odds.odds_value,
        ).filter(
            Odds.market_type == "1X2",
            Odds.bookmaker != "Flashscore",
            Odds.odds_value > 1.0,
        )
        for o in q.all():
            row = rows.get(o.match_id)
            if row is not None:
                row.odds_1x2.setdefault(o.bookmaker, {})[o.selection] = o.odds_value

    logger.info(f"baseline: loaded {len(rows)} completed matches, "
                f"{sum(1 for r in rows.values() if r.odds_1x2)} with 1X2 odds")
    return list(rows.values())


# ------------------------------------------------------------- model factories
# Reimplemented here rather than imported so the harness never depends on live
# model state (loaded pickles, cached fits) that would leak across windows.

def fit_poisson(train, ref_date, half_life=180):
    if half_life:
        dr = math.log(2) / half_life
        w = np.array([math.exp(-dr * max(0, (ref_date - m.match_date).days))
                      for m in train])
    else:
        w = np.ones(len(train))
    hg = np.array([m.home_goals for m in train], float)
    ag = np.array([m.away_goals for m in train], float)
    ws = w.sum() or 1.0
    gh, ga = float(w @ hg / ws), float(w @ ag / ws)

    lg = defaultdict(lambda: [[], [], []])
    for m, wt in zip(train, w):
        L = lg[m.league]
        L[0].append(m.home_goals); L[1].append(m.away_goals); L[2].append(wt)
    lg_avg = {}
    for name, (hs, as_, ww) in lg.items():
        if len(hs) >= 30:
            ww = np.array(ww); s = ww.sum()
            if s > 0:
                lg_avg[name] = (float(ww @ np.array(hs, float) / s),
                                float(ww @ np.array(as_, float) / s))

    ts = defaultdict(lambda: {"hs": [], "hc": [], "as": [], "ac": [],
                              "hw": [], "aw": []})
    for m, wt in zip(train, w):
        ts[m.home_team_id]["hs"].append(m.home_goals)
        ts[m.home_team_id]["hc"].append(m.away_goals)
        ts[m.home_team_id]["hw"].append(wt)
        ts[m.away_team_id]["as"].append(m.away_goals)
        ts[m.away_team_id]["ac"].append(m.home_goals)
        ts[m.away_team_id]["aw"].append(wt)

    def wavg(v, ww, fb):
        if not v:
            return fb
        ww = np.array(ww); s = ww.sum()
        return float(ww @ np.array(v, float) / s) if s > 0 else float(np.mean(v))

    st = {}
    for t, s in ts.items():
        atk = ((wavg(s["hs"], s["hw"], gh) / gh)
               + (wavg(s["as"], s["aw"], ga) / ga)) / 2
        dfn = ((wavg(s["hc"], s["hw"], ga) / ga)
               + (wavg(s["ac"], s["aw"], gh) / gh)) / 2
        n = len(s["hs"]) + len(s["as"])
        sh = min(n, SHRINK_CAP) / SHRINK_CAP
        st[t] = (max(atk * sh + (1 - sh), 0.15), max(dfn * sh + (1 - sh), 0.15))
    return {"st": st, "lg_avg": lg_avg, "gh": gh, "ga": ga}


def _dc_tau(x, y, lam, mu, rho):
    if x == 0 and y == 0:
        t = 1 - lam * mu * rho
    elif x == 0 and y == 1:
        t = 1 + lam * rho
    elif x == 1 and y == 0:
        t = 1 + mu * rho
    elif x == 1 and y == 1:
        t = 1 - rho
    else:
        return 1.0
    return max(t, 0.01)


def predict_poisson(state, row, rho=-0.13):
    hs, as_ = state["st"].get(row.home_team_id), state["st"].get(row.away_team_id)
    if hs is None or as_ is None:
        return None
    ah, aa = state["lg_avg"].get(row.league, (state["gh"], state["ga"]))
    lam, mu = ah * hs[0] * as_[1], aa * as_[0] * hs[1]
    hp = [_pois.pmf(i, lam) for i in range(MAX_GOALS)]
    ap = [_pois.pmf(i, mu) for i in range(MAX_GOALS)]
    mtx = np.outer(hp, ap)
    if rho:
        for i in range(2):
            for j in range(2):
                mtx[i, j] *= _dc_tau(i, j, lam, mu, rho)
        mtx /= mtx.sum()
    return [float(np.tril(mtx, -1).sum()), float(np.diag(mtx).sum()),
            float(np.triu(mtx, 1).sum())]


def fit_elo(train, ref_date, k=32, ha=65, reg=0.33):
    R, prev_year = {}, None
    for m in train:
        yr = m.match_date.year
        if prev_year is not None and yr > prev_year:
            for t in R:
                R[t] = R[t] * (1 - reg) + 1500 * reg
        prev_year = yr
        he = R.get(m.home_team_id, 1500) + ha
        ae = R.get(m.away_team_id, 1500)
        exp = 1 / (1 + 10 ** ((ae - he) / 400))
        act = 1.0 if m.home_goals > m.away_goals else (
            0.5 if m.home_goals == m.away_goals else 0.0)
        gd = abs(m.home_goals - m.away_goals)
        g = k * math.log(max(gd, 1) + 1)
        R[m.home_team_id] = R.get(m.home_team_id, 1500) + g * (act - exp)
        R[m.away_team_id] = R.get(m.away_team_id, 1500) + g * ((1 - act) - (1 - exp))
    return R


def predict_elo(R, row, ha=65):
    if row.home_team_id not in R or row.away_team_id not in R:
        return None
    he, ae = R[row.home_team_id] + ha, R[row.away_team_id]
    e = 1 / (1 + 10 ** ((ae - he) / 400))
    dp = max(0.15, 0.28 - abs(he - ae) / 2000)
    return [(1 - dp) * e, dp, (1 - dp) * (1 - e)]


def blend(model_p, market_p, w_market):
    if model_p is None or market_p is None:
        return None
    p = [(1 - w_market) * a + w_market * b for a, b in zip(model_p, market_p)]
    t = sum(p)
    return [x / t for x in p] if t > 0 else None


# ------------------------------------------------------------------------ main

def build_candidates():
    def _pe(state, row):
        p, e = predict_poisson(state["p"], row), predict_elo(state["e"], row)
        if p is None or e is None:
            return None
        return [(a + b) / 2 for a, b in zip(p, e)]

    def fit_both(train, cutoff):
        return {"p": fit_poisson(train, cutoff), "e": fit_elo(train, cutoff)}

    cands = [
        Candidate("market (de-vigged consensus)",
                  lambda t, c: None, lambda s, r: market_consensus(r)),
        Candidate("market (raw 1/odds, vig left in)",
                  lambda t, c: None, lambda s, r: market_raw(r)),
        Candidate("poisson only (180d, rho=-0.13)",
                  lambda t, c: fit_poisson(t, c), predict_poisson),
        Candidate("elo only",
                  fit_elo, predict_elo),
        Candidate("poisson + elo (50/50)", fit_both, _pe),
    ]
    for w in (0.4, 0.6, 0.8):
        cands.append(Candidate(
            f"market {int(w*100)}% + poisson/elo {int((1-w)*100)}%",
            fit_both,
            lambda s, r, _w=w: blend(_pe(s, r), market_consensus(r), _w),
        ))
    return cands


def _load_env() -> None:
    """Load .env — called from main(), never at import.

    Import-time load_dotenv() is a trap: tests/conftest.py deliberately pops
    DATABASE_URL so nothing can reach production, and importing this module put
    it straight back. That turned a SQLite unit test into a live write against
    the production database (caught by an IntegrityError, but only by luck).
    Environment mutation is a side effect and belongs in an entry point.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # optional dependency; env may already be exported
        pass


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=60,
                    help="forward evaluation window in days")
    ap.add_argument("--since", default="2022-01-01")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load_rows(date.fromisoformat(args.since))
    cutoffs = [date(2025, 8, 1), date(2025, 11, 1), date(2026, 2, 1),
               date(2026, 5, 1), date(2026, 7, 1)]

    results = walk_forward(rows, build_candidates(), cutoffs,
                           window_days=args.window)
    print()
    print("=" * 92)
    print("IMMUTABLE BASELINE — chronological walk-forward, identical match set")
    print("=" * 92)
    print(format_table(results))

    base = "market (de-vigged consensus)"
    print("\nPaired bootstrap vs the market baseline (positive = market is better):")
    for name in results["candidates"]:
        if name == base:
            continue
        c = compare(results, name, base)
        if "error" in c:
            continue
        print(f"  {name:<40} {c['mean_improvement']:+.4f} nats "
              f"CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] "
              f"{'SIGNIFICANT' if c['significant'] else 'not significant'}")

    out = Path(args.out) if args.out else Path(
        f"data/baselines/baseline-{datetime.utcnow():%Y%m%d-%H%M%S}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    payload["generated_at"] = datetime.utcnow().isoformat()
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nSnapshot written to {out} (do not edit — new runs write new files)")


if __name__ == "__main__":
    main()
