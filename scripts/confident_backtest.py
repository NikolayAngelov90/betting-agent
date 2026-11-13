"""Stage 31 - confident picks at a good price: the blend, the agreement
filter, the selection grid and the closing edge.

REGISTERED BEFORE ANY OUTCOME WAS READ, in
docs/confident-value-preregistration.md (commit PREREGISTRATION_COMMIT). The
grid, the periods, the price definition and the verdict rules are fixed there.

ONE DEFINITION OF THE RULE. Leg sets, the blended probability and the
one-pick-per-fixture choice come from `src.betting.confident_value` -- the
same module the pipeline flag reads. Nothing about the rule is implemented
here.

THE STEPS, IN THE ORDER THEY MAY BE RUN

    --control        OUTCOME-FREE. The leg sets this script builds are
                     identical to the Stage 30 loader's, and the grid has
                     the registered number of cells.
    --blend          development only: walk-forward log-loss, the registered
                     weights with their CIs, and the agreement test.
    --dev            development only: every cell of the grid.
    --open-holdout   THE HOLDOUT, ONCE: blend delta, agreement, every cell,
                     the paired current rule, the decision table.
    --prices         Part D, no outcome read: the closing edge of generous
                     prices, from the snapshot history.
    --gates          the pre-experiment picks restated with today's gates.

Credentials come from `.env` / `.env.local`, read inside `_load_env()`.
Model probabilities are cached with `--cache DIR` (they are pre-match and
identical between steps).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import pickle
import statistics
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

from scripts.sharp_anchor import (
    DEV_FROM,
    EXCLUDE_REASONS,
    HOLDOUT_FROM,
    HOLDOUT_UNTIL,
    KAMBI_FAMILY,
    _load_env,
)
from src.betting import confident_value as cv

PREREGISTRATION_COMMIT = "4fcdd83"
BOOTSTRAP_DRAWS = 4000
WEIGHT_BOOTSTRAP_DRAWS = 1000
MIN_TRAIN_SETS = 300
GRID_STEP = 0.01
CLIP = 1e-6
SEED = 20261112
REGISTERED_CELLS = 26
#: comparisons declared in section 7 of the registration
DECLARED_COMPARISONS = 36
MIXED_SHARES = ((0.90, 0.10), (0.75, 0.25), (0.50, 0.50))
PRE_EXPERIMENT_UNTIL = _dt.date(2026, 8, 5)
PRICE_PATH_MIN_GAP_S = 3600.0
FIXTURE_DAY_CLASSES = (("<10", 0, 10), ("10-40", 10, 41), (">40", 41, 10**9))


# ═════════════════════════════ loading ══════════════════════════════════════

def load(with_results: bool, date_from, date_until) -> dict:
    """Matches and their leg sets, PRE-KICKOFF rows only.

    The population is the Stage 30 loader's, clause for clause; the sets are
    built by `cv.build_sets`, the function the flag uses. `--control` proves
    the two agree on the real table.

    `with_results=False` does not select the score columns at all.
    """
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
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
            SELECT m.id, m.match_date, m.league, m.training_exclusion_reason,
                   m.home_team_id, m.away_team_id {score}
            FROM matches m WHERE {' AND '.join(where)}
              AND EXISTS (SELECT 1 FROM odds o WHERE o.match_id = m.id)
        """), params).fetchall()
        matches = {}
        excluded: Dict[str, int] = defaultdict(int)
        for row in ms:
            if row[3] in EXCLUDE_REASONS:
                excluded[row[3]] += 1
                continue
            if row[1] is None or row[1].microsecond != 0 or row[1].hour < 6:
                excluded["placeholder_or_phantom_kickoff"] += 1
                continue
            matches[row[0]] = {
                "kickoff": row[1], "league": row[2],
                "home": row[4], "away": row[5],
                "hg": int(row[6]) if with_results else None,
                "ag": int(row[7]) if with_results else None,
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
    per_match: Dict[int, list] = defaultdict(list)
    after_kickoff = 0
    for mid, bm, mt, sel, val, ts in rows:
        if ts >= matches[mid]["kickoff"]:
            after_kickoff += 1
            continue
        per_match[mid].append((bm, mt, sel, val, ts))
    sets_by_match: Dict[int, dict] = {}
    refused: Dict[str, int] = defaultdict(int)
    for mid, mrows in per_match.items():
        sets, ref = cv.build_sets(mrows)
        sets_by_match[mid] = sets
        for k, v in ref.items():
            refused[k] += v
    return {"matches": matches, "sets": sets_by_match,
            "excluded": dict(excluded), "refused": dict(refused),
            "rows_after_kickoff": after_kickoff}


def control() -> dict:
    """OUTCOME-FREE. (1) every leg set this script builds equals the Stage 30
    loader's on the real table, key for key and price for price; (2) the grid
    has the registered number of cells."""
    from scripts import sharp_anchor
    mine = load(False, DEV_FROM, HOLDOUT_UNTIL)
    ref = sharp_anchor.load(False, DEV_FROM, HOLDOUT_UNTIL)
    flat = {}
    for mid, sets in mine["sets"].items():
        for (bm, market, line, side), st in sets.items():
            flat[(mid, bm, market, line, side)] = st
    only_mine = set(flat) - set(ref["sets"])
    only_ref = set(ref["sets"]) - set(flat)
    differing = sum(1 for k in set(flat) & set(ref["sets"])
                    if flat[k]["keys"] != ref["sets"][k]["keys"]
                    or flat[k]["prices"] != ref["sets"][k]["prices"]
                    or flat[k]["ts"] != ref["sets"][k]["ts"])
    grid = cv.registered_grid()
    return {
        "sets_mine": len(flat), "sets_stage30": len(ref["sets"]),
        "only_mine": len(only_mine), "only_stage30": len(only_ref),
        "differing": differing,
        "sets_identical": not only_mine and not only_ref and differing == 0,
        "fixtures": len(mine["matches"]),
        "grid_cells": len(grid),
        "grid_names_unique": len({r.name for r in grid}) == len(grid),
        "by_lane": dict(Counter(r.lane for r in grid)),
    }


# ═══════════════════════ model probabilities, pre-match ═════════════════════

def elo_probabilities(matches: dict) -> Dict[str, Dict[int, dict]]:
    """`{"corrected": {mid: pred}, "deployed": {mid: pred}}`.

    corrected = season-boundary regression + draw curve refit MONTHLY on
    prior matches only. deployed = calendar-year regression + the fixed
    curve. Both from ratings as they stood BEFORE the match.
    """
    from scripts.residual_harness import _fit2, _sigmoid, history_pass
    out: Dict[str, Dict[int, dict]] = {"corrected": {}, "deployed": {}}
    wanted = set(matches)
    for mode, name in (("season", "corrected"), ("calendar", "deployed")):
        hp = history_pass(mode)
        rows = [(mid, f) for mid, f in hp["pre"].items()
                if f["date"] is not None]
        rows.sort(key=lambda kv: kv[1]["date"])
        months = sorted({(matches[m]["kickoff"].year,
                          matches[m]["kickoff"].month) for m in wanted})
        for ym in months:
            start = _dt.datetime(ym[0], ym[1], 1)
            a = b = None
            if name == "corrected":
                train = [f for _m, f in rows if f["date"] < start]
                a, b = _fit2([abs(f["elo_diff"]) / 100.0 for f in train],
                             [1 if f["hg"] == f["ag"] else 0 for f in train])
            for mid, f in rows:
                if mid not in wanted or (f["date"].year,
                                         f["date"].month) != ym:
                    continue
                diff = f["elo_diff"]
                eh = 1.0 / (1.0 + 10 ** (-diff / 400.0))
                pdraw = (_sigmoid(a + b * abs(diff) / 100.0)
                         if name == "corrected"
                         else max(0.15, 0.28 - abs(diff) / 2000.0))
                rem = 1.0 - pdraw
                out[name][mid] = {"home_win": rem * eh, "draw": pdraw,
                                  "away_win": rem * (1.0 - eh)}
    return out


def poisson_probabilities(matches: dict) -> Tuple[Dict[int, dict], dict]:
    """Weekly walk-forward. Each Monday 00:00 the model is fitted on matches
    BEFORE it and predicts that week's fixtures. A fixture with a team the
    fit has never seen gets NO Poisson opinion (None), not a league-average
    one."""
    from src.models.poisson_model import PoissonModel
    model = PoissonModel()
    by_week: Dict[_dt.datetime, List[int]] = defaultdict(list)
    for mid, m in matches.items():
        d = m["kickoff"].date()
        monday = d - _dt.timedelta(days=d.weekday())
        by_week[_dt.datetime.combine(monday, _dt.time.min)].append(mid)
    out: Dict[int, dict] = {}
    unseen = 0
    for monday in sorted(by_week):
        model.fit(as_of_date=monday)
        known = set(model._team_strengths)
        for mid in by_week[monday]:
            m = matches[mid]
            if m["home"] not in known or m["away"] not in known:
                unseen += 1
                continue
            p = model.predict(m["home"], m["away"], league=m["league"])
            out[mid] = {k: v for k, v in p.items()
                        if isinstance(v, (int, float))}
    return out, {"weeks": len(by_week), "fixtures_with_unseen_team": unseen}


def model_probabilities(matches: dict, cache_dir: Optional[str]) -> dict:
    key = f"cvmodels_{min(matches)}_{max(matches)}_{len(matches)}.pkl"
    path = os.path.join(cache_dir, key) if cache_dir else None
    if path and os.path.exists(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    elo = elo_probabilities(matches)
    pois, meta = poisson_probabilities(matches)
    out = {"elo": elo, "poisson": pois, "poisson_meta": meta}
    if path:
        with open(path, "wb") as fh:
            pickle.dump(out, fh)
    return out


# ═════════════════════════ observations and grading ═════════════════════════

def observations(d: dict, models: dict, elo_variant: str = "corrected",
                 **slot_kw) -> Tuple[List[dict], dict]:
    """One observation per priced, modelled leg SET, graded.

    Each carries the sources (market / elo / poisson, in the set's key
    order), the taken prices, and which leg won. A set with no winner or a
    void leg is dropped and counted.
    """
    from src.agent.betting_agent import FootballBettingAgent
    from src.data.market_vocabulary import canonical
    grade = FootballBettingAgent._grade_selection

    obs: List[dict] = []
    counts: Dict[str, int] = defaultdict(int)
    for mid, sets in d["sets"].items():
        m = d["matches"][mid]
        priced = cv.slots(sets, **slot_kw)
        if not priced:
            counts["fixtures_without_an_anchor_slot"] += 1
            continue
        preds = {"elo": models["elo"][elo_variant].get(mid),
                 "poisson": models["poisson"].get(mid)}
        for (market, line, side), sl in priced.items():
            fam = cv.family_of(market, line)
            if fam is None:
                counts["slots_unmodelled_line_or_market"] += 1
                continue
            won = []
            for key in sl["keys"]:
                canon = canonical(market, key)
                res = grade(canon[1], m["hg"], m["ag"]) if canon else None
                won.append(None if res not in ("win", "loss")
                           else (1 if res == "win" else 0))
            # A BINARY set with one gradeable leg is fully graded: the
            # other leg is its complement. The grader does not know a
            # team-goals Under, and without this every team-goals set would
            # be dropped for a leg the rule can never pick anyway.
            if len(won) == 2 and sum(w is None for w in won) == 1:
                known = won[0] if won[0] is not None else 1 - won[1]
                won = [known, 1 - known]
                counts["binary_sets_graded_by_complement"] += 1
            if any(w is None for w in won):
                counts["sets_ungradeable_or_void"] += 1
                continue
            if sum(won) != 1:
                counts["sets_without_exactly_one_winner"] += 1
                continue
            sources = cv.sources_for(market, line, side, sl["keys"],
                                     sl["fair"], preds)
            obs.append({
                "mid": mid, "family": fam, "slot": (market, line, side),
                "keys": sl["keys"], "sources": sources, "taken": sl["taken"],
                "fair": sl["fair"], "won": won, "winner": won.index(1),
                "books": sl["books"], "feed": sl["feed"],
                "date": m["kickoff"].date(), "league": m["league"],
            })
            counts["sets"] += 1
    return obs, dict(counts)


# ═══════════════════════════ the blend: fitting ═════════════════════════════

def _simplex(names: Sequence[str]) -> List[Dict[str, float]]:
    steps = int(round(1.0 / GRID_STEP))
    out = []
    if len(names) == 2:
        for i in range(steps + 1):
            out.append({names[0]: i / steps, names[1]: (steps - i) / steps})
    else:
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                out.append({names[0]: i / steps, names[1]: j / steps,
                            names[2]: (steps - i - j) / steps})
    return out


def _winner_matrix(obs: List[dict], names: Sequence[str]):
    """`(matrix, kept)`: the probability each source gave the WINNER, one row
    per set that every source has an opinion on."""
    import numpy as np
    kept = [o for o in obs if all(o["sources"].get(n) is not None
                                  for n in names)]
    mat = np.array([[min(max(o["sources"][n][o["winner"]], CLIP), 1 - CLIP)
                     for n in names] for o in kept], dtype=np.float64)
    return mat, kept


def fit_weights(obs: List[dict], family: str, boot: int = 0) -> dict:
    """Exhaustive search on the simplex. Each source's vector sums to one,
    so the pooled probability of the winner is the weighted sum of the
    sources' winner probabilities -- no renormalisation is involved."""
    import numpy as np
    names = cv.FAMILY_SOURCES[family]
    mat, kept = _winner_matrix([o for o in obs if o["family"] == family],
                               names)
    if len(kept) == 0:
        return {"family": family, "n_sets": 0}
    grid = _simplex(names)
    W = np.array([[g[n] for n in names] for g in grid])       # (G, K)
    L = -np.log(np.clip(mat @ W.T, CLIP, None))               # (n, G)
    mean = L.mean(axis=0)
    best = int(mean.argmin())
    out = {"family": family, "n_sets": len(kept),
           "fixtures": len({o["mid"] for o in kept}),
           "weights": dict(grid[best]),
           "log_loss": float(mean[best]),
           "market_log_loss": float(-np.log(mat[:, 0]).mean()),
           "dropped_missing_source": sum(
               1 for o in obs if o["family"] == family) - len(kept)}
    if boot:
        fix = sorted({o["mid"] for o in kept})
        pos = {f: i for i, f in enumerate(fix)}
        per_fix = np.zeros((len(fix), L.shape[1]))
        cnt = np.zeros(len(fix))
        for row, o in zip(L, kept):
            per_fix[pos[o["mid"]]] += row
            cnt[pos[o["mid"]]] += 1
        rng = np.random.default_rng(SEED)
        draws = {n: [] for n in names}
        for _ in range(boot):
            # a resample of fixtures, as multiplicities: one dot product
            # instead of copying the (fixtures x grid) matrix per draw
            k = np.bincount(rng.integers(0, len(fix), size=len(fix)),
                            minlength=len(fix)).astype(np.float64)
            tot = (k @ per_fix) / (k @ cnt)
            g = grid[int(tot.argmin())]
            for n in names:
                draws[n].append(g[n])
        out["weights_ci95"] = {
            n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for n, v in draws.items()}
    return out


def _delta_ci(obs: List[dict], weights: Dict[str, float], family: str):
    """Out-of-sample log-loss of the blend minus the market's, with a
    fixture-bootstrap CI. `weights` may differ per observation via `w` key."""
    import numpy as np
    rows = []
    for o in obs:
        if o["family"] != family:
            continue
        w = o.get("w", weights)
        p = cv.pool(o["sources"], w)
        if p is None:
            continue
        lb = -math.log(min(max(p[o["winner"]], CLIP), 1 - CLIP))
        lm = -math.log(min(max(o["sources"]["market"][o["winner"]], CLIP),
                           1 - CLIP))
        rows.append((o["mid"], lb, lm))
    if not rows:
        return {"n_sets": 0}
    fix = sorted({r[0] for r in rows})
    pos = {f: i for i, f in enumerate(fix)}
    sb = np.zeros(len(fix)); sm = np.zeros(len(fix)); c = np.zeros(len(fix))
    for mid, lb, lm in rows:
        sb[pos[mid]] += lb; sm[pos[mid]] += lm; c[pos[mid]] += 1
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(fix), size=(BOOTSTRAP_DRAWS, len(fix)))
    d = (sb[idx].sum(axis=1) - sm[idx].sum(axis=1)) / c[idx].sum(axis=1)
    return {"n_sets": len(rows), "fixtures": len(fix),
            "blend_log_loss": float(sb.sum() / c.sum()),
            "market_log_loss": float(sm.sum() / c.sum()),
            "delta": float((sb.sum() - sm.sum()) / c.sum()),
            "delta_ci95": [float(np.percentile(d, 2.5)),
                           float(np.percentile(d, 97.5))]}


def walk_forward(obs: List[dict], family: str) -> dict:
    """Monthly folds inside the period given. Weights fit on sets BEFORE the
    month, scored on the month."""
    fam = [o for o in obs if o["family"] == family]
    months = sorted({(o["date"].year, o["date"].month) for o in fam})
    scored: List[dict] = []
    folds = []
    for ym in months:
        start = _dt.date(ym[0], ym[1], 1)
        train = [o for o in fam if o["date"] < start]
        test = [o for o in fam if (o["date"].year, o["date"].month) == ym]
        if len(train) < MIN_TRAIN_SETS:
            continue
        w = fit_weights(train, family)
        if w.get("n_sets", 0) == 0:
            continue
        for o in test:
            scored.append(dict(o, w=w["weights"]))
        folds.append({"month": f"{ym[0]}-{ym[1]:02d}",
                      "train_sets": len(train), "test_sets": len(test),
                      "weights": w["weights"]})
    out = _delta_ci(scored, {}, family)
    out["folds"] = folds
    return out


def registered_weights(dev_obs: List[dict]) -> Dict[str, dict]:
    return {fam: fit_weights(dev_obs, fam, boot=WEIGHT_BOOTSTRAP_DRAWS)
            for fam in cv.FAMILY_SOURCES}


# ═══════════════════════════ legs, cells, tables ════════════════════════════

def legs_from(obs: List[dict], weights: Dict[str, Dict[str, float]]):
    """`cv.Leg` for every PICKABLE leg, made by `cv.legs_from_sources` --
    the function the flag uses -- plus the outcome and the day of each."""
    legs: List[cv.Leg] = []
    won: Dict[Tuple, int] = {}
    meta: Dict[int, dict] = {}
    for o in obs:
        market, line, side = o["slot"]
        made = cv.legs_from_sources(
            o["mid"], market, line, side, o["keys"], o["sources"],
            o["taken"], o["fair"], o["books"], o["feed"], weights)
        for leg in made:
            legs.append(leg)
            won[(o["mid"], leg.key)] = o["won"][o["keys"].index(leg.key[3])]
        meta[o["mid"]] = {"date": o["date"], "league": o["league"]}
    return legs, won, meta


def _boot_mean(values: Sequence[float], draws: int = BOOTSTRAP_DRAWS,
               level: float = 95.0):
    import numpy as np
    if len(values) == 0:
        return None
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(v), size=(draws, len(v)))
    m = v[idx].mean(axis=1)
    lo = (100.0 - level) / 2.0
    return [float(np.percentile(m, lo)), float(np.percentile(m, 100.0 - lo))]


def _summ(picks: List[cv.Leg], won: dict) -> dict:
    """n, hit rate + CI, PREDICTED beside it, average odds, ROI + CI."""
    n = len(picks)
    if n == 0:
        return {"n": 0}
    hits = [won[(p.fixture, p.key)] for p in picks]
    ret = [h * p.taken - 1.0 for h, p in zip(hits, picks)]
    bonf = 100.0 * (1.0 - 0.05 / REGISTERED_CELLS)
    return {
        "n": n,
        "hit_rate": sum(hits) / n,
        "hit_ci95": _boot_mean(hits),
        "hit_ci_bonferroni": _boot_mean(hits, level=bonf),
        "predicted_hit_rate": sum(p.p for p in picks) / n,
        "anchor_fair_mean": sum(p.fair for p in picks) / n,
        "avg_odds": sum(p.taken for p in picks) / n,
        "roi": sum(ret) / n,
        "roi_ci95": _boot_mean(ret),
    }


def _mix(picks: List[cv.Leg]) -> dict:
    return dict(Counter(p.selection for p in picks).most_common())


def _per_day(picks: List[cv.Leg], meta: dict, fixtures_by_day: dict) -> dict:
    by_day = Counter(meta[p.fixture]["date"] for p in picks)
    counts = [by_day.get(d, 0) for d in sorted(fixtures_by_day)]
    out = {"days_with_fixtures": len(counts),
           "zero_pick_days": sum(1 for c in counts if c == 0),
           "median": statistics.median(counts) if counts else None,
           "min": min(counts) if counts else None,
           "max": max(counts) if counts else None, "by_day_class": {}}
    for name, lo, hi in FIXTURE_DAY_CLASSES:
        cs = [by_day.get(d, 0) for d, n in fixtures_by_day.items()
              if lo <= n < hi]
        out["by_day_class"][name] = {
            "days": len(cs),
            "zero_pick_days": sum(1 for c in cs if c == 0),
            "median": statistics.median(cs) if cs else None,
            "min": min(cs) if cs else None, "max": max(cs) if cs else None}
    return out


def cells(legs, won, meta, current: Optional[dict] = None) -> Dict[str, dict]:
    fixtures_by_day = Counter(m["date"] for m in meta.values())
    out: Dict[str, dict] = {}
    for rule in cv.registered_grid():
        picks = cv.select(legs, rule)
        row = _summ(picks, won)
        row["lane"] = rule.lane
        row["market_mix"] = _mix(picks)
        row["per_day"] = _per_day(picks, meta, fixtures_by_day)
        row["by_band"] = {}
        for band in cv.BANDS:
            sub = [p for p in picks if cv.band_of(p.taken) == band]
            row["by_band"][f"{band[0]:.2f}-{band[1]:.2f}"] = _summ(sub, won)
        if current is not None:
            both = [(won[(p.fixture, p.key)], current[p.fixture])
                    for p in picks if p.fixture in current]
            row["paired_vs_current"] = (
                {"fixtures": 0} if not both else {
                    "fixtures": len(both),
                    "cell_hit": sum(a for a, _ in both) / len(both),
                    "current_hit": sum(b for _, b in both) / len(both),
                    "diff": sum(a - b for a, b in both) / len(both),
                    "diff_ci95": _boot_mean([a - b for a, b in both])})
        out[rule.name] = row
    return out


def agreement(legs, won) -> dict:
    """Among legs priced >= 1.70: hit rate agree vs disagree, clustered on
    the fixture, pooled and per family."""
    import numpy as np

    def one(sel) -> dict:
        fix = sorted({l.fixture for l in sel})
        if not fix:
            return {"n": 0}
        pos = {f: i for i, f in enumerate(fix)}
        s = np.zeros((2, len(fix))); c = np.zeros((2, len(fix)))
        r = np.zeros((2, len(fix)))                  # returns, for the ROI
        for l in sel:
            g = 0 if l.agree else 1
            w = won[(l.fixture, l.key)]
            s[g, pos[l.fixture]] += w
            r[g, pos[l.fixture]] += w * l.taken - 1.0
            c[g, pos[l.fixture]] += 1
        res = {}
        for g, name in ((0, "agree"), (1, "disagree")):
            grp = [l for l in sel if l.agree == (g == 0)]
            res[name] = {"n": int(c[g].sum()),
                         "hit_rate": (float(s[g].sum() / c[g].sum())
                                      if grp else None),
                         # ADDED AFTER THE HOLDOUT WAS OPENED, and labelled:
                         # hard rule 2 wants predicted, odds and ROI beside
                         # every hit rate and the first run printed only the
                         # hit rate and the odds here. The registered verdict
                         # is the hit-rate difference and is unchanged.
                         "predicted_hit_rate": (statistics.mean(
                             l.p for l in grp) if grp else None),
                         "avg_odds": (statistics.mean(
                             l.taken for l in grp) if grp else None),
                         "roi": (float(r[g].sum() / c[g].sum())
                                 if grp else None)}
        if c[0].sum() == 0 or c[1].sum() == 0:
            res["verdict"] = "UNEVALUATED: one side is empty"
            return res
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(fix), size=(BOOTSTRAP_DRAWS, len(fix)))
        with np.errstate(invalid="ignore", divide="ignore"):
            ra = s[0][idx].sum(axis=1) / c[0][idx].sum(axis=1)
            rd = s[1][idx].sum(axis=1) / c[1][idx].sum(axis=1)
        for name, rr in (("agree", ra), ("disagree", rd)):
            res[name]["hit_ci95"] = [float(np.nanpercentile(rr, 2.5)),
                                     float(np.nanpercentile(rr, 97.5))]
        with np.errstate(invalid="ignore", divide="ignore"):
            for g, name in ((0, "agree"), (1, "disagree")):
                b = r[g][idx].sum(axis=1) / c[g][idx].sum(axis=1)
                res[name]["roi_ci95"] = [float(np.nanpercentile(b, 2.5)),
                                         float(np.nanpercentile(b, 97.5))]
        diff = ra - rd
        lo, hi = (float(np.nanpercentile(diff, 2.5)),
                  float(np.nanpercentile(diff, 97.5)))
        res["diff"] = res["agree"]["hit_rate"] - res["disagree"]["hit_rate"]
        res["diff_ci95"] = [lo, hi]
        res["verdict"] = ("RAISES" if lo > 0 else "LOWERS" if hi < 0
                          else "NO DETECTABLE EFFECT at this n")
        return res

    eligible = [l for l in legs if l.taken >= cv.MIN_PRICE]
    out = {"pooled": one(eligible)}
    for fam in cv.FAMILY_SOURCES:
        out[fam] = one([l for l in eligible if l.family == fam])
    return out


def ceiling(legs) -> dict:
    """The zero-edge win rate at a price: 1 / (o * (1 + m)), with m the
    MEASURED margin of the taken price over the anchor's fair price."""
    el = [l for l in legs if l.taken >= cv.MIN_PRICE and l.fair > 0]
    if not el:
        return {"n": 0}
    m = [1.0 / (l.taken * l.fair) - 1.0 for l in el]
    mean_m = statistics.mean(m)
    return {"n_eligible_legs": len(el), "mean_margin": mean_m,
            "median_margin": statistics.median(m),
            "ceiling_at_min_price_zero_margin": 1.0 / cv.MIN_PRICE,
            "ceiling_at_min_price_measured_margin":
                1.0 / (cv.MIN_PRICE * (1.0 + mean_m))}


def blocker(legs, meta) -> dict:
    """`simulated`: legs per day, per band, that clear every price and EV
    gate of the CURRENT rule and are removed by its confidence gate alone --
    with the BLENDED probability standing in for the ensemble's."""
    days = len({m["date"] for m in meta.values()}) or 1
    out = {}
    for band in cv.BANDS:
        in_band = [l for l in legs if cv.band_of(l.taken) == band]
        ev_ok = [l for l in in_band if l.p * l.taken - 1.0 >= 0.05]
        removed = [l for l in ev_ok
                   if l.p < 0.52
                   or (l.p < 0.55 and (l.p * l.taken - 1.0) * l.p < 0.038)]
        out[f"{band[0]:.2f}-{band[1]:.2f}"] = {
            "legs": len(in_band), "legs_per_day": len(in_band) / days,
            "clear_ev_gate": len(ev_ok),
            "removed_by_confidence_alone": len(removed),
            "removed_per_day": len(removed) / days,
            # NOT in the registration, added after the development run and
            # labelled so: the registered count above is ~0 because under a
            # blend that is nearly the market almost no leg clears the 5% EV
            # gate in the first place. This is the confidence gate applied
            # ALONE to every leg in the band -- price information only, no
            # outcome.
            "below_052_floor_alone": sum(1 for l in in_band if l.p < 0.52),
            "below_052_floor_alone_per_day":
                sum(1 for l in in_band if l.p < 0.52) / days,
            "at_or_above_052": sum(1 for l in in_band if l.p >= 0.52),
            "p_needed_at_band_low": 0.52,
            "edge_that_floor_claims_at_band_low": 0.52 * band[0] - 1.0,
            "edge_that_floor_claims_at_band_high": 0.52 * band[1] - 1.0}
    out["days"] = days
    return out


# ═══════════════════════════ the current rule ═══════════════════════════════

def current_rule(date_from, date_until, mids: set) -> dict:
    """The pipeline's saved picks on the same fixtures. Graded on the MODEL
    pick (as of s5.22 the saved pick is the model's), with the reviewed final
    pick beside it."""
    _load_env()
    from sqlalchemy import text

    from src.data.database import get_db
    db = get_db()
    assert db.is_postgres
    with db.get_session() as s:
        rows = s.execute(text("""
            SELECT sp.match_id, sp.odds, sp.result, sp.review_action,
                   sp.model_result, sp.predicted_probability,
                   sp.model_probability,
                   (SELECT po.taken_odds FROM pick_observations po
                     WHERE po.pick_id = sp.id AND po.attribution = 'model'
                     LIMIT 1)
            FROM saved_picks sp JOIN matches m ON m.id = sp.match_id
            WHERE sp.disposition IS NULL AND sp.evidence_status IS NULL
              AND m.match_date >= :lo AND m.match_date < :hi
        """), {"lo": _dt.datetime.combine(date_from, _dt.time.min),
               "hi": _dt.datetime.combine(date_until, _dt.time.min)}
        ).fetchall()
    final, model = [], []
    by_fix: Dict[int, List[int]] = defaultdict(list)
    counts = Counter()
    for mid, odds, result, action, mres, p, mp, model_odds in rows:
        if mid not in mids:
            counts["pick_on_a_fixture_outside_the_population"] += 1
            continue
        if result in ("win", "loss"):
            final.append({"mid": mid, "odds": odds,
                          "won": 1 if result == "win" else 0, "p": p})
        changed = action == "CHANGE"
        r = mres if (changed and mres) else result
        if r not in ("win", "loss"):
            counts["model_pick_unsettled_or_void"] += 1
            continue
        o = (model_odds if changed else odds)
        if changed and model_odds is None:
            counts["changed_pick_with_no_model_price"] += 1
        model.append({"mid": mid, "odds": o, "won": 1 if r == "win" else 0,
                      "p": mp if (changed and mp is not None) else p})
        by_fix[mid].append(1 if r == "win" else 0)

    def summ(rs):
        if not rs:
            return {"n": 0}
        hits = [r["won"] for r in rs]
        priced = [r for r in rs if r["odds"]]
        ret = [r["won"] * r["odds"] - 1.0 for r in priced]
        ps = [r["p"] for r in rs if r["p"] is not None]
        res = {"n": len(rs), "fixtures": len({r["mid"] for r in rs}),
               "hit_rate": sum(hits) / len(hits),
               "hit_ci95": _boot_mean(hits),
               "predicted_hit_rate": (sum(ps) / len(ps)) if ps else None,
               "avg_odds": (sum(r["odds"] for r in priced) / len(priced)
                            if priced else None),
               "roi": (sum(ret) / len(ret)) if ret else None,
               "roi_ci95": _boot_mean(ret), "n_priced": len(priced)}
        return res

    out = {"model_pick": summ(model), "final_pick": summ(final),
           "counts": dict(counts), "by_band": {}}
    for band in cv.BANDS:
        sub = [r for r in model
               if r["odds"] and cv.band_of(r["odds"]) == band]
        out["by_band"][f"{band[0]:.2f}-{band[1]:.2f}"] = summ(sub)
    out["below_1.70"] = summ([r for r in model
                              if r["odds"] and r["odds"] < 1.70])
    out["above_3.50"] = summ([r for r in model
                              if r["odds"] and r["odds"] > 3.50])
    out["_by_fixture"] = {f: sum(v) / len(v) for f, v in by_fix.items()}
    return out


def pre_experiment() -> dict:
    """Saved picks dated <= 2026-08-05, restated per band with TODAY's gates
    applied to the stored fields."""
    _load_env()
    import yaml
    from sqlalchemy import text

    from src.data.database import get_db
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "config", "config.example.yaml"),
              encoding="utf-8") as fh:
        b = yaml.safe_load(fh)["betting"]
    db = get_db()
    assert db.is_postgres
    with db.get_session() as s:
        rows = s.execute(text("""
            SELECT market, selection, odds, predicted_probability,
                   expected_value, result, disposition, evidence_status,
                   pick_date
            FROM saved_picks WHERE pick_date <= :d
        """), {"d": PRE_EXPERIMENT_UNTIL}).fetchall()
    from src.data.market_vocabulary import canonical
    excluded_keys = set(b.get("excluded_markets") or [])
    name_to_key = {"Under 1.5 Goals": "under_1.5", "BTTS No": "btts_no",
                   "Over 3.5 Goals": "over_3.5"}
    counts = Counter()
    kept_all, kept_gated = [], []
    # THE CONFIDENCE GATE ALONE, on real candidates. These picks were saved
    # under looser gates, so they are the pipeline's own legs: for each band,
    # how many clear every OTHER gate of today and fall to the confidence
    # gate by itself.
    alone: Dict[str, Counter] = defaultdict(Counter)
    days = set()
    for market, sel, odds, p, ev, result, disp, evid, pick_date in rows:
        counts["rows"] += 1
        if disp is not None or evid is not None:
            counts["not_valid_evidence_or_superseded"] += 1
            continue
        if result not in ("win", "loss"):
            counts["unsettled_or_void"] += 1
            continue
        counts["settled_valid"] += 1
        rec = {"odds": odds, "won": 1 if result == "win" else 0, "p": p}
        kept_all.append(rec)
        canon = canonical(market, sel)
        key = name_to_key.get(canon[1]) if canon else None
        gates = (
            odds is not None and b["min_odds"] <= odds <= b["max_odds"]
            and ev is not None and ev >= b["min_expected_value"]
            and p is not None and p >= 0.52
            and (p >= b["min_confidence"]
                 or ev * p >= b["min_ev_confidence_score"])
            and key not in excluded_keys)
        if gates:
            kept_gated.append(rec)
        days.add(pick_date)
        band = cv.band_of(odds) if odds is not None else None
        bname = (f"{band[0]:.2f}-{band[1]:.2f}" if band else
                 "below 1.70" if (odds is not None and odds < 1.70)
                 else "above 3.50")
        others = (odds is not None and b["min_odds"] <= odds <= b["max_odds"]
                  and ev is not None and ev >= b["min_expected_value"]
                  and key not in excluded_keys)
        alone[bname]["settled_valid"] += 1
        if others:
            alone[bname]["clear_every_other_gate"] += 1
            if p is None or p < 0.52 or (
                    p < b["min_confidence"]
                    and ev * p < b["min_ev_confidence_score"]):
                alone[bname]["removed_by_confidence_alone"] += 1

    def summ(rs):
        if not rs:
            return {"n": 0}
        hits = [r["won"] for r in rs]
        ret = [r["won"] * r["odds"] - 1.0 for r in rs]
        ps = [r["p"] for r in rs if r["p"] is not None]
        return {"n": len(rs), "hit_rate": sum(hits) / len(hits),
                "hit_ci95": _boot_mean(hits),
                "predicted_hit_rate": (sum(ps) / len(ps)) if ps else None,
                "avg_odds": sum(r["odds"] for r in rs) / len(rs),
                "roi": sum(ret) / len(ret), "roi_ci95": _boot_mean(ret)}

    def banded(rs):
        out = {"all": summ(rs),
               "below 1.70": summ([r for r in rs if r["odds"] < 1.70])}
        for band in cv.BANDS:
            out[f"{band[0]:.2f}-{band[1]:.2f}"] = summ(
                [r for r in rs if cv.band_of(r["odds"]) == band])
        out["above 3.50"] = summ([r for r in rs if r["odds"] > 3.50])
        return out

    n_days = len(days) or 1
    gate_alone = {}
    for bname, c in alone.items():
        gate_alone[bname] = dict(
            c, removed_per_pick_day=(
                c["removed_by_confidence_alone"] / n_days),
            share_of_those_clearing_the_rest=(
                c["removed_by_confidence_alone"]
                / c["clear_every_other_gate"]
                if c["clear_every_other_gate"] else None))
    return {"until": str(PRE_EXPERIMENT_UNTIL), "named_in_directive": 1018,
            "pick_days": n_days, "confidence_gate_alone": gate_alone,
            "counts": dict(counts), "as_saved": banded(kept_all),
            "with_todays_gates": banded(kept_gated),
            "gates": {k: b[k] for k in (
                "min_odds", "max_odds", "min_expected_value",
                "min_confidence", "min_ev_confidence_score",
                "excluded_markets")} | {"hard_probability_floor": 0.52}}


# ═══════════════════════ Part D: are generous prices real ═══════════════════

def prices() -> dict:
    """NO OUTCOME IS READ. For fixtures where the Odds-API anchor was seen at
    two pre-kickoff instants at least an hour apart: the edge of the taken
    price against the anchor's fair probability at the FIRST instant, and
    against its fair probability at the LAST."""
    _load_env()
    import numpy as np
    from sqlalchemy import inspect, text

    from src.data.database import get_db
    from src.data.price_history import QUOTE_FRESH, quote_freshness
    from src.evaluation.devig import shin
    db = get_db()
    assert db.is_postgres
    anchor = cv.ANCHORS["theodds"]
    has_lu = "book_last_update" in {
        c["name"] for c in inspect(db.engine).get_columns("odds_snapshots")}
    lu = ", os.book_last_update" if has_lu else ", NULL"
    with db.get_session() as s:
        rows = s.execute(text(f"""
            SELECT os.match_id, os.bookmaker, os.market_type, os.selection,
                   os.odds_value, os.observed_at, m.match_date,
                   m.training_exclusion_reason {lu}
            FROM odds_snapshots os JOIN matches m ON m.id = os.match_id
            WHERE os.bookmaker LIKE 'TheOddsAPI-%'
              AND os.observed_at < m.match_date
              AND m.league NOT LIKE 'world/%'
        """)).fetchall()
    by_match: Dict[int, list] = defaultdict(list)
    for r in rows:
        if r[7] in EXCLUDE_REASONS:
            continue
        by_match[r[0]].append(r)
    counts = Counter()
    legs = []
    for mid, mrows in by_match.items():
        a_times = sorted({r[5] for r in mrows if r[1] == anchor})
        if not a_times:
            counts["fixtures_without_the_anchor"] += 1
            continue
        t0, t1 = a_times[0], a_times[-1]
        if (t1 - t0).total_seconds() < PRICE_PATH_MIN_GAP_S:
            counts["fixtures_with_one_anchor_instant"] += 1
            continue
        counts["fixtures_with_a_price_path"] += 1
        near = lambda t, ref: abs((t - ref).total_seconds()) \
            <= cv.SNAPSHOT_TOLERANCE_S                      # noqa: E731
        first = [(r[1], r[2], r[3], r[4], r[5]) for r in mrows
                 if near(r[5], t0)]
        last = [(r[1], r[2], r[3], r[4], r[5]) for r in mrows
                if r[1] == anchor and near(r[5], t1)]
        fresh_at_t0 = {(r[1], r[2], r[3]): quote_freshness(r[5], r[8])
                       for r in mrows if near(r[5], t0)}
        sets0, _ = cv.build_sets(first)
        sets1, _ = cv.build_sets(last)
        priced = cv.slots(sets0)
        for (market, line, side), sl in priced.items():
            if sl["feed"] != "theodds":
                continue
            close = sets1.get((anchor, market, line, side))
            if close is None or close["keys"] != sl["keys"]:
                counts["slots_with_no_closing_anchor_set"] += 1
                continue
            fair1 = shin(close["q"])
            if fair1 is None:
                counts["closing_devig_refused"] += 1
                continue
            for i, key in enumerate(sl["keys"]):
                per_book = {bm: p[i] for bm, p in
                            sl["prices_by_book"].items()}
                legs.append({
                    "mid": mid, "market": market, "key": key,
                    "taken": sl["taken"][i], "fair0": sl["fair"][i],
                    "fair1": fair1[i], "per_book": per_book,
                    "fresh": {bm: fresh_at_t0.get((bm, market, key))
                              for bm in per_book},
                    "gap_h": (t1 - t0).total_seconds() / 3600.0})

    def edge_rows(price_of):
        out = []
        for l in legs:
            for price, tag in price_of(l):
                e0 = price * l["fair0"] - 1.0
                if e0 > 0:
                    out.append((l["mid"], price, e0,
                                price * l["fair1"] - 1.0, tag))
        return out

    def summ(rs):
        if not rs:
            return {"n": 0}
        fix = sorted({r[0] for r in rs})
        pos = {f: i for i, f in enumerate(fix)}
        s0 = np.zeros(len(fix)); s1 = np.zeros(len(fix)); c = np.zeros(len(fix))
        for mid, _p, e0, e1, _t in rs:
            s0[pos[mid]] += e0; s1[pos[mid]] += e1; c[pos[mid]] += 1
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, len(fix), size=(BOOTSTRAP_DRAWS, len(fix)))
        b1 = s1[idx].sum(axis=1) / c[idx].sum(axis=1)
        lo, hi = float(np.percentile(b1, 2.5)), float(np.percentile(b1, 97.5))
        e0m, e1m = float(s0.sum() / c.sum()), float(s1.sum() / c.sum())
        return {"n": len(rs), "fixtures": len(fix),
                "avg_price": float(np.mean([r[1] for r in rs])),
                "instant_edge": e0m, "closing_edge": e1m,
                "closing_ci95": [lo, hi],
                "retained_share": (e1m / e0m) if e0m > 0 else None,
                "verdict": ("REAL" if lo > 0 else "SLOW" if hi < 0
                            else "UNRESOLVED at this n")}

    med = lambda l: [(l["taken"], "median")]                  # noqa: E731
    all_rows = edge_rows(med)
    out = {
        "counts": dict(counts), "legs": len(legs),
        "column_present": has_lu,
        "median_gap_hours": (statistics.median(l["gap_h"] for l in legs)
                             if legs else None),
        "primary_taken_ge_1.70": summ(
            [r for r in all_rows if r[1] >= cv.MIN_PRICE]),
        "secondary_all_prices": summ(all_rows),
        "by_band": {f"{b[0]:.2f}-{b[1]:.2f}": summ(
            [r for r in all_rows if cv.band_of(r[1]) == b])
            for b in cv.BANDS},
        "per_book": {},
    }
    books = sorted({bm for l in legs for bm in l["per_book"]})
    for bm in books:
        if bm == anchor:
            continue
        rs = edge_rows(lambda l, bm=bm: (
            [(l["per_book"][bm], bm)] if bm in l["per_book"] else []))
        out["per_book"][bm] = summ([r for r in rs if r[1] >= cv.MIN_PRICE])
    # the FRESH subset: every book quote at t0 that is tagged FRESH
    fresh_rows = []
    unevaluated = 0
    for l in legs:
        for bm, price in l["per_book"].items():
            if bm == anchor:
                continue
            state = l["fresh"].get(bm)
            if state != QUOTE_FRESH:
                unevaluated += state is None or state == "UNEVALUATED"
                continue
            e0 = price * l["fair0"] - 1.0
            if e0 > 0 and price >= cv.MIN_PRICE:
                fresh_rows.append((l["mid"], price, e0,
                                   price * l["fair1"] - 1.0, bm))
    out["fresh_subset"] = summ(fresh_rows)
    out["fresh_subset"]["book_quotes_unevaluated"] = int(unevaluated)
    return out


# ═══════════════════════════════ running ════════════════════════════════════

def run(period: str, cache_dir: Optional[str],
        weights_in: Optional[dict] = None) -> dict:
    """`period` is "dev" or "holdout". The holdout REQUIRES `weights_in`: it
    never fits anything."""
    lo, hi = ((DEV_FROM, HOLDOUT_FROM) if period == "dev"
              else (HOLDOUT_FROM, HOLDOUT_UNTIL))
    d = load(True, lo, hi)
    models = model_probabilities(d["matches"], cache_dir)
    out = {"period": period, "from": str(lo), "until": str(hi),
           "fixtures_loaded": len(d["matches"]), "excluded": d["excluded"],
           "refused": d["refused"],
           "rows_after_kickoff": d["rows_after_kickoff"],
           "poisson_meta": models["poisson_meta"], "variants": {}}
    for variant in ("corrected", "deployed"):
        obs, counts = observations(d, models, variant)
        v = {"counts": counts}
        if period == "dev":
            v["walk_forward"] = {f: walk_forward(obs, f)
                                 for f in cv.FAMILY_SOURCES}
            fitted = registered_weights(obs)
            v["registered_weights"] = fitted
            weights = {f: w["weights"] for f, w in fitted.items()
                       if w.get("n_sets")}
        else:
            assert weights_in is not None, "the holdout never fits"
            weights = weights_in[variant]
        v["weights_used"] = weights
        v["delta_vs_market"] = {f: _delta_ci(obs, weights.get(f, {}), f)
                                for f in cv.FAMILY_SOURCES}
        legs, won, meta = legs_from(obs, weights)
        v["legs"] = len(legs)
        v["fixtures_with_legs"] = len(meta)
        v["agreement"] = agreement(legs, won)
        v["ceiling"] = ceiling(legs)
        v["blocker"] = blocker(legs, meta)
        cur = current_rule(lo, hi, set(meta))
        by_fix = cur.pop("_by_fixture")
        v["current_rule"] = cur
        v["cells"] = cells(legs, won, meta, by_fix)
        if variant == "corrected":
            # sensitivities, on the registered variant only
            for name, kw in (("without_kambi", {"exclude": KAMBI_FAMILY}),
                             ("cross_feed_median", {"cross_feed": True})):
                o2, _c = observations(d, models, variant, **kw)
                l2, w2, m2 = legs_from(o2, weights)
                v[name] = {k: {kk: r[kk] for kk in (
                    "n", "hit_rate", "hit_ci95", "predicted_hit_rate",
                    "avg_odds", "roi", "roi_ci95") if kk in r}
                    for k, r in cells(l2, w2, m2).items()}
        out["variants"][variant] = v
    return out


def decision_table(holdout: dict, variant: str = "corrected") -> dict:
    """One row per cell, lanes never pooled. NOTHING IS RECOMMENDED: per
    lane, the cell with the highest hit rate whose 95% CI lower bound exceeds
    the current rule's holdout hit rate is MARKED, if one exists."""
    v = holdout["variants"][variant]
    base = v["current_rule"]["model_pick"].get("hit_rate")
    rows, marks, real_marks = [], {}, {}
    for name, c in v["cells"].items():
        if c.get("n", 0) == 0:
            rows.append({"cell": name, "lane": c["lane"], "n": 0})
            continue
        clears = base is not None and c["hit_ci95"][0] > base
        survives = base is not None and c["hit_ci_bonferroni"][0] > base
        # A DEFECT IN THE RULE AS REGISTERED, found on opening: it names no
        # minimum n. When every pick in a cell has the same outcome the
        # percentile bootstrap has ZERO width -- one winning pick "clears"
        # anything. The mark as written is kept and reported; this flag says
        # whether there is an interval behind it at all.
        degenerate = c["hit_rate"] in (0.0, 1.0)
        rows.append({
            "cell": name, "lane": c["lane"], "n": c["n"],
            "hit_rate": c["hit_rate"], "hit_ci95": c["hit_ci95"],
            "predicted_hit_rate": c["predicted_hit_rate"],
            "avg_odds": c["avg_odds"], "roi": c["roi"],
            "roi_ci95": c["roi_ci95"],
            "picks_per_day_median": c["per_day"]["median"],
            "picks_per_day_range": [c["per_day"]["min"], c["per_day"]["max"]],
            "zero_pick_days": c["per_day"]["zero_pick_days"],
            "days": c["per_day"]["days_with_fixtures"],
            "clears_current_rule_95": clears,
            "survives_bonferroni": survives,
            "interval_is_degenerate": degenerate})
        if clears and (c["lane"] not in marks
                       or c["hit_rate"] > marks[c["lane"]]["hit_rate"]):
            marks[c["lane"]] = rows[-1]
        if clears and not degenerate and (
                c["lane"] not in real_marks
                or c["hit_rate"] > real_marks[c["lane"]]["hit_rate"]):
            real_marks[c["lane"]] = rows[-1]
    mixed = []
    l1 = [r for r in rows if r["lane"] == cv.LANE_CONFIDENT and r["n"]]
    l2 = [r for r in rows if r["lane"] == cv.LANE_VALUE and r["n"]]
    for a in l1:
        for b in l2:
            if a["cell"].split()[-1] != b["cell"].split()[-1]:
                continue
            mixed.append({"lane1": a["cell"], "lane2": b["cell"], "shares": {
                f"{int(s1 * 100)}/{int(s2 * 100)}":
                    s1 * a["hit_rate"] + s2 * b["hit_rate"]
                for s1, s2 in MIXED_SHARES}})
    return {"variant": variant, "current_rule_hit_rate": base,
            "current_rule": v["current_rule"]["model_pick"],
            "rows": rows,
            "marked": {lane: r["cell"] for lane, r in marks.items()},
            "marked_n": {lane: r["n"] for lane, r in marks.items()},
            "marked_with_a_real_interval":
                {lane: r["cell"] for lane, r in real_marks.items()},
            "mixed_card_arithmetic": mixed,
            "cells": len(rows), "declared_comparisons": DECLARED_COMPARISONS,
            "expected_false_positives_at_95":
                0.05 * DECLARED_COMPARISONS}


#: THE MEASURED RECORD. Written from the run outputs by a generator, not
#: typed: every figure below is the number the run printed. `measured`
#: 2026-11-12; population, n and method are inside the record.
MEASURED = {'measured_on': '2026-11-12',
 'preregistration': '4fcdd83',
 'grain': 'one pick per fixture per cell; legs for the agreement test',
 'method': 'fixture bootstrap, 4000 draws, percentile; weights by exhaustive '
           'search on the simplex at step 0.01',
 'control': {'leg_sets': 225581,
             'differing_from_stage30_loader': 0,
             'grid_cells': 26},
 'population': {'development': {'fixtures_loaded': 3293,
                                'fixtures_with_legs': 3069,
                                'legs': 31990},
                'holdout': {'fixtures_loaded': 1545,
                            'fixtures_with_legs': 1540,
                            'legs': 17938}},
 'weights': {'corrected': {'1X2': {'n_sets': 2322,
                                   'weights': {'market': 0.93,
                                               'elo': 0.06,
                                               'poisson': 0.01},
                                   'ci95': {'market': [0.81, 1.0],
                                            'elo': [0.0, 0.18],
                                            'poisson': [0.0, 0.12]}},
                           'over_under': {'n_sets': 7403,
                                          'weights': {'market': 1.0,
                                                      'poisson': 0.0},
                                          'ci95': {'market': [0.88, 1.0],
                                                   'poisson': [0.0, 0.12]}},
                           'btts': {'n_sets': 671,
                                    'weights': {'market': 1.0,
                                                'poisson': 0.0},
                                    'ci95': {'market': [0.44, 1.0],
                                             'poisson': [0.0, 0.56]}},
                           'team_goals': {'n_sets': 7461,
                                          'weights': {'market': 0.97,
                                                      'poisson': 0.03},
                                          'ci95': {'market': [0.86, 1.0],
                                                   'poisson': [0.0, 0.14]}}},
             'deployed': {'1X2': {'n_sets': 2322,
                                  'weights': {'market': 0.94,
                                              'elo': 0.04,
                                              'poisson': 0.02},
                                  'ci95': {'market': [0.82, 1.0],
                                           'elo': [0.0, 0.16],
                                           'poisson': [0.0, 0.13]}},
                          'over_under': {'n_sets': 7403,
                                         'weights': {'market': 1.0,
                                                     'poisson': 0.0},
                                         'ci95': {'market': [0.88, 1.0],
                                                  'poisson': [0.0, 0.12]}},
                          'btts': {'n_sets': 671,
                                   'weights': {'market': 1.0, 'poisson': 0.0},
                                   'ci95': {'market': [0.44, 1.0],
                                            'poisson': [0.0, 0.56]}},
                          'team_goals': {'n_sets': 7461,
                                         'weights': {'market': 0.97,
                                                     'poisson': 0.03},
                                         'ci95': {'market': [0.86, 1.0],
                                                  'poisson': [0.0, 0.14]}}}},
 'walk_forward_development': {'corrected': {'1X2': {'n': 1662,
                                                    'delta': 0.00167,
                                                    'ci95': [-0.0013, 0.0046],
                                                    'folds': 3},
                                            'over_under': {'n': 5803,
                                                           'delta': 0.00038,
                                                           'ci95': [-7e-05,
                                                                    0.00082],
                                                           'folds': 4},
                                            'btts': {'n': 1,
                                                     'delta': -0.0,
                                                     'ci95': [-0.0, -0.0],
                                                     'folds': 1},
                                            'team_goals': {'n': 6330,
                                                           'delta': 0.00017,
                                                           'ci95': [-8e-05,
                                                                    0.00043],
                                                           'folds': 4}},
                              'deployed': {'1X2': {'n': 1662,
                                                   'delta': 0.00141,
                                                   'ci95': [-0.00099,
                                                            0.00371],
                                                   'folds': 3},
                                           'over_under': {'n': 5803,
                                                          'delta': 0.00038,
                                                          'ci95': [-7e-05,
                                                                   0.00082],
                                                          'folds': 4},
                                           'btts': {'n': 1,
                                                    'delta': -0.0,
                                                    'ci95': [-0.0, -0.0],
                                                    'folds': 1},
                                           'team_goals': {'n': 6330,
                                                          'delta': 0.00017,
                                                          'ci95': [-8e-05,
                                                                   0.00043],
                                                          'folds': 4}}},
 'delta_vs_market': {'development': {'1X2': {'n': 2463,
                                             'delta': -0.00017,
                                             'ci95': [-0.00109, 0.00074]},
                                     'over_under': {'n': 7727,
                                                    'delta': -0.0,
                                                    'ci95': [-0.0, 0.0]},
                                     'btts': {'n': 672,
                                              'delta': -0.0,
                                              'ci95': [-0.0, -0.0]},
                                     'team_goals': {'n': 7803,
                                                    'delta': -3e-05,
                                                    'ci95': [-0.00018,
                                                             0.00013]}},
                     'holdout': {'1X2': {'n': 1538,
                                         'delta': 0.00046,
                                         'ci95': [-0.00055, 0.0015]},
                                 'over_under': {'n': 4538,
                                                'delta': -0.0,
                                                'ci95': [-0.0, 0.0]},
                                 'btts': {'n': 0},
                                 'team_goals': {'n': 4248,
                                                'delta': 0.00024,
                                                'ci95': [1e-05, 0.00046]}}},
 'ml_share': 'UNEVALUATED',
 'agreement_pooled': {'development': {'agree': {'n': 2504,
                                                'hit_rate': 0.5036,
                                                'predicted_hit_rate': 0.5082,
                                                'avg_odds': 1.8958,
                                                'roi': -0.0545,
                                                'hit_ci95': [0.4847, 0.5226],
                                                'roi_ci95': [-0.0912,
                                                             -0.0182]},
                                      'disagree': {'n': 18791,
                                                   'hit_rate': 0.3353,
                                                   'predicted_hit_rate': 0.3343,
                                                   'avg_odds': 3.1695,
                                                   'roi': -0.0816,
                                                   'hit_ci95': [0.3295,
                                                                0.3407],
                                                   'roi_ci95': [-0.1,
                                                                -0.0637]},
                                      'diff': 0.1683,
                                      'diff_ci95': [0.1467, 0.1904],
                                      'verdict': 'RAISES'},
                      'holdout': {'agree': {'n': 1278,
                                            'hit_rate': 0.507,
                                            'predicted_hit_rate': 0.4993,
                                            'avg_odds': 1.9367,
                                            'roi': -0.0374,
                                            'hit_ci95': [0.4785, 0.5352],
                                            'roi_ci95': [-0.0928, 0.0176]},
                                  'disagree': {'n': 10429,
                                               'hit_rate': 0.3141,
                                               'predicted_hit_rate': 0.3154,
                                               'avg_odds': 3.4128,
                                               'roi': -0.1009,
                                               'hit_ci95': [0.3065, 0.3216],
                                               'roi_ci95': [-0.1268,
                                                            -0.0751]},
                                  'diff': 0.1929,
                                  'diff_ci95': [0.1612, 0.2242],
                                  'verdict': 'RAISES'}},
 'ceiling': {'development': {'n_eligible_legs': 21295,
                             'mean_margin': 0.0943,
                             'median_margin': 0.0799,
                             'ceiling_at_min_price_zero_margin': 0.5882,
                             'ceiling_at_min_price_measured_margin': 0.5375},
             'holdout': {'n_eligible_legs': 11707,
                         'mean_margin': 0.0948,
                         'median_margin': 0.0827,
                         'ceiling_at_min_price_zero_margin': 0.5882,
                         'ceiling_at_min_price_measured_margin': 0.5373}},
 'current_rule_model_pick': {'development': {'n': 1310,
                                             'fixtures': 1136,
                                             'hit_rate': 0.5328,
                                             'hit_ci95': [0.5053, 0.5595],
                                             'predicted_hit_rate': 0.6145,
                                             'avg_odds': 1.8557,
                                             'roi': -0.0599,
                                             'roi_ci95': [-0.1132, -0.0086],
                                             'n_priced': 1216},
                             'holdout': {'n': 1456,
                                         'fixtures': 1456,
                                         'hit_rate': 0.5831,
                                         'hit_ci95': [0.5577, 0.6092],
                                         'predicted_hit_rate': 0.6285,
                                         'avg_odds': 1.63,
                                         'roi': -0.0546,
                                         'roi_ci95': [-0.0957, -0.0119],
                                         'n_priced': 1456}},
 'current_rule_by_band_holdout': {'1.70-2.00': {'n': 326,
                                                'hit': 0.5521,
                                                'hit_ci95': [0.4969, 0.6074],
                                                'predicted': 0.5919,
                                                'avg_odds': 1.79,
                                                'roi': -0.0113,
                                                'roi_ci95': [-0.1082,
                                                             0.0852]},
                                  '2.00-2.50': {'n': 41,
                                                'hit': 0.5854,
                                                'hit_ci95': [0.439, 0.7317],
                                                'predicted': 0.6248,
                                                'avg_odds': 2.12,
                                                'roi': 0.2395,
                                                'roi_ci95': [-0.0776,
                                                             0.5529]},
                                  '2.50-3.50': {'n': 5,
                                                'hit': 0.6,
                                                'hit_ci95': [0.2, 1.0],
                                                'predicted': 0.6134,
                                                'avg_odds': 2.54,
                                                'roi': 0.522,
                                                'roi_ci95': [-0.5, 1.544]},
                                  'below 1.70': {'n': 1084,
                                                 'hit': 0.5923,
                                                 'hit_ci95': [0.5618, 0.6208],
                                                 'predicted': 0.6397,
                                                 'avg_odds': 1.56,
                                                 'roi': -0.0814,
                                                 'roi_ci95': [-0.1284,
                                                              -0.036]}},
 'cells': {'development': {'C p>=0.50 all': {'n': 2269,
                                             'hit': 0.5412,
                                             'hit_ci95': [0.5205, 0.5619],
                                             'predicted': 0.5352,
                                             'avg_odds': 1.782,
                                             'roi': -0.0363,
                                             'roi_ci95': [-0.0728, 0.0011],
                                             'per_day_median': 11.0,
                                             'per_day_range': [0, 69],
                                             'zero_pick_days': 5,
                                             'days': 136},
                           'C p>=0.52 all': {'n': 1660,
                                             'hit': 0.5476,
                                             'hit_ci95': [0.5235, 0.5705],
                                             'predicted': 0.5442,
                                             'avg_odds': 1.755,
                                             'roi': -0.0392,
                                             'roi_ci95': [-0.0817, 0.0016],
                                             'per_day_median': 8.0,
                                             'per_day_range': [0, 52],
                                             'zero_pick_days': 9,
                                             'days': 136},
                           'C p>=0.54 all': {'n': 992,
                                             'hit': 0.5585,
                                             'hit_ci95': [0.5282, 0.5897],
                                             'predicted': 0.5537,
                                             'avg_odds': 1.732,
                                             'roi': -0.033,
                                             'roi_ci95': [-0.0851, 0.0213],
                                             'per_day_median': 4.0,
                                             'per_day_range': [0, 30],
                                             'zero_pick_days': 14,
                                             'days': 136},
                           'C p>=0.56 all': {'n': 244,
                                             'hit': 0.5615,
                                             'hit_ci95': [0.5, 0.623],
                                             'predicted': 0.5659,
                                             'avg_odds': 1.711,
                                             'roi': -0.0397,
                                             'roi_ci95': [-0.1451, 0.0644],
                                             'per_day_median': 1.0,
                                             'per_day_range': [0, 10],
                                             'zero_pick_days': 46,
                                             'days': 136},
                           'C p>=0.58 all': {'n': 4,
                                             'hit': 0.5,
                                             'hit_ci95': [0.0, 1.0],
                                             'predicted': 0.5845,
                                             'avg_odds': 1.712,
                                             'roi': -0.15,
                                             'roi_ci95': [-1.0, 0.7],
                                             'per_day_median': 0.0,
                                             'per_day_range': [0, 1],
                                             'zero_pick_days': 132,
                                             'days': 136},
                           'C p>=0.50 agree': {'n': 1542,
                                               'hit': 0.5331,
                                               'hit_ci95': [0.5084, 0.5577],
                                               'predicted': 0.5354,
                                               'avg_odds': 1.782,
                                               'roi': -0.0507,
                                               'roi_ci95': [-0.0951, -0.0069],
                                               'per_day_median': 7.0,
                                               'per_day_range': [0, 50],
                                               'zero_pick_days': 6,
                                               'days': 136},
                           'C p>=0.52 agree': {'n': 1127,
                                               'hit': 0.5421,
                                               'hit_ci95': [0.512, 0.5714],
                                               'predicted': 0.5445,
                                               'avg_odds': 1.756,
                                               'roi': -0.0484,
                                               'roi_ci95': [-0.1012, 0.0035],
                                               'per_day_median': 5.0,
                                               'per_day_range': [0, 35],
                                               'zero_pick_days': 12,
                                               'days': 136},
                           'C p>=0.54 agree': {'n': 682,
                                               'hit': 0.5587,
                                               'hit_ci95': [0.5205, 0.5938],
                                               'predicted': 0.554,
                                               'avg_odds': 1.732,
                                               'roi': -0.0324,
                                               'roi_ci95': [-0.098, 0.0297],
                                               'per_day_median': 3.0,
                                               'per_day_range': [0, 22],
                                               'zero_pick_days': 22,
                                               'days': 136},
                           'C p>=0.56 agree': {'n': 173,
                                               'hit': 0.5376,
                                               'hit_ci95': [0.4624, 0.6069],
                                               'predicted': 0.5658,
                                               'avg_odds': 1.712,
                                               'roi': -0.0804,
                                               'roi_ci95': [-0.2086, 0.0395],
                                               'per_day_median': 1.0,
                                               'per_day_range': [0, 8],
                                               'zero_pick_days': 58,
                                               'days': 136},
                           'C p>=0.58 agree': {'n': 3,
                                               'hit': 0.3333,
                                               'hit_ci95': [0.0, 1.0],
                                               'predicted': 0.5844,
                                               'avg_odds': 1.717,
                                               'roi': -0.4333,
                                               'roi_ci95': [-1.0, 0.7],
                                               'per_day_median': 0.0,
                                               'per_day_range': [0, 1],
                                               'zero_pick_days': 133,
                                               'days': 136},
                           'L1 p>=0.50 all': {'n': 2269,
                                              'hit': 0.5412,
                                              'hit_ci95': [0.5205, 0.5619],
                                              'predicted': 0.5352,
                                              'avg_odds': 1.782,
                                              'roi': -0.0363,
                                              'roi_ci95': [-0.0728, 0.0011],
                                              'per_day_median': 11.0,
                                              'per_day_range': [0, 69],
                                              'zero_pick_days': 5,
                                              'days': 136},
                           'L1 p>=0.52 all': {'n': 1660,
                                              'hit': 0.5476,
                                              'hit_ci95': [0.5235, 0.5705],
                                              'predicted': 0.5442,
                                              'avg_odds': 1.755,
                                              'roi': -0.0392,
                                              'roi_ci95': [-0.0817, 0.0016],
                                              'per_day_median': 8.0,
                                              'per_day_range': [0, 52],
                                              'zero_pick_days': 9,
                                              'days': 136},
                           'L1 p>=0.54 all': {'n': 992,
                                              'hit': 0.5585,
                                              'hit_ci95': [0.5282, 0.5897],
                                              'predicted': 0.5537,
                                              'avg_odds': 1.732,
                                              'roi': -0.033,
                                              'roi_ci95': [-0.0851, 0.0213],
                                              'per_day_median': 4.0,
                                              'per_day_range': [0, 30],
                                              'zero_pick_days': 14,
                                              'days': 136},
                           'L1 p>=0.56 all': {'n': 244,
                                              'hit': 0.5615,
                                              'hit_ci95': [0.5, 0.623],
                                              'predicted': 0.5659,
                                              'avg_odds': 1.711,
                                              'roi': -0.0397,
                                              'roi_ci95': [-0.1451, 0.0644],
                                              'per_day_median': 1.0,
                                              'per_day_range': [0, 10],
                                              'zero_pick_days': 46,
                                              'days': 136},
                           'L1 p>=0.58 all': {'n': 4,
                                              'hit': 0.5,
                                              'hit_ci95': [0.0, 1.0],
                                              'predicted': 0.5845,
                                              'avg_odds': 1.712,
                                              'roi': -0.15,
                                              'roi_ci95': [-1.0, 0.7],
                                              'per_day_median': 0.0,
                                              'per_day_range': [0, 1],
                                              'zero_pick_days': 132,
                                              'days': 136},
                           'L1 p>=0.50 agree': {'n': 1542,
                                                'hit': 0.5331,
                                                'hit_ci95': [0.5084, 0.5577],
                                                'predicted': 0.5354,
                                                'avg_odds': 1.782,
                                                'roi': -0.0507,
                                                'roi_ci95': [-0.0951,
                                                             -0.0069],
                                                'per_day_median': 7.0,
                                                'per_day_range': [0, 50],
                                                'zero_pick_days': 6,
                                                'days': 136},
                           'L1 p>=0.52 agree': {'n': 1127,
                                                'hit': 0.5421,
                                                'hit_ci95': [0.512, 0.5714],
                                                'predicted': 0.5445,
                                                'avg_odds': 1.756,
                                                'roi': -0.0484,
                                                'roi_ci95': [-0.1012, 0.0035],
                                                'per_day_median': 5.0,
                                                'per_day_range': [0, 35],
                                                'zero_pick_days': 12,
                                                'days': 136},
                           'L1 p>=0.54 agree': {'n': 682,
                                                'hit': 0.5587,
                                                'hit_ci95': [0.5205, 0.5938],
                                                'predicted': 0.554,
                                                'avg_odds': 1.732,
                                                'roi': -0.0324,
                                                'roi_ci95': [-0.098, 0.0297],
                                                'per_day_median': 3.0,
                                                'per_day_range': [0, 22],
                                                'zero_pick_days': 22,
                                                'days': 136},
                           'L1 p>=0.56 agree': {'n': 173,
                                                'hit': 0.5376,
                                                'hit_ci95': [0.4624, 0.6069],
                                                'predicted': 0.5658,
                                                'avg_odds': 1.712,
                                                'roi': -0.0804,
                                                'roi_ci95': [-0.2086, 0.0395],
                                                'per_day_median': 1.0,
                                                'per_day_range': [0, 8],
                                                'zero_pick_days': 58,
                                                'days': 136},
                           'L1 p>=0.58 agree': {'n': 3,
                                                'hit': 0.3333,
                                                'hit_ci95': [0.0, 1.0],
                                                'predicted': 0.5844,
                                                'avg_odds': 1.717,
                                                'roi': -0.4333,
                                                'roi_ci95': [-1.0, 0.7],
                                                'per_day_median': 0.0,
                                                'per_day_range': [0, 1],
                                                'zero_pick_days': 133,
                                                'days': 136},
                           'L2 e>=0.00 all': {'n': 48,
                                              'hit': 0.3542,
                                              'hit_ci95': [0.2292, 0.5],
                                              'predicted': 0.3477,
                                              'avg_odds': 2.982,
                                              'roi': 0.0399,
                                              'roi_ci95': [-0.3419, 0.454],
                                              'per_day_median': 0.0,
                                              'per_day_range': [0, 6],
                                              'zero_pick_days': 108,
                                              'days': 136},
                           'L2 e>=0.02 all': {'n': 17,
                                              'hit': 0.3529,
                                              'hit_ci95': [0.1176, 0.5882],
                                              'predicted': 0.3455,
                                              'avg_odds': 3.05,
                                              'roi': 0.0259,
                                              'roi_ci95': [-0.6365, 0.7418],
                                              'per_day_median': 0.0,
                                              'per_day_range': [0, 3],
                                              'zero_pick_days': 123,
                                              'days': 136},
                           'L2 e>=0.05 all': {'n': 4,
                                              'hit': 0.25,
                                              'hit_ci95': [0.0, 0.75],
                                              'predicted': 0.3529,
                                              'avg_odds': 3.07,
                                              'roi': -0.3575,
                                              'roi_ci95': [-1.0, 0.9275],
                                              'per_day_median': 0.0,
                                              'per_day_range': [0, 3],
                                              'zero_pick_days': 134,
                                              'days': 136},
                           'L2 e>=0.00 agree': {'n': 4,
                                                'hit': 0.25,
                                                'hit_ci95': [0.0, 0.75],
                                                'predicted': 0.4188,
                                                'avg_odds': 2.425,
                                                'roi': -0.425,
                                                'roi_ci95': [-1.0, 0.725],
                                                'per_day_median': 0.0,
                                                'per_day_range': [0, 1],
                                                'zero_pick_days': 132,
                                                'days': 136},
                           'L2 e>=0.02 agree': {'n': 0},
                           'L2 e>=0.05 agree': {'n': 0}},
           'holdout': {'C p>=0.50 all': {'n': 1114,
                                         'hit': 0.5655,
                                         'hit_ci95': [0.5359, 0.5943],
                                         'predicted': 0.5344,
                                         'avg_odds': 1.79,
                                         'roi': 0.0113,
                                         'roi_ci95': [-0.0416, 0.0618],
                                         'per_day_median': 14.5,
                                         'per_day_range': [0, 71],
                                         'zero_pick_days': 2,
                                         'days': 56},
                       'C p>=0.52 all': {'n': 793,
                                         'hit': 0.5649,
                                         'hit_ci95': [0.5309, 0.599],
                                         'predicted': 0.5443,
                                         'avg_odds': 1.762,
                                         'roi': -0.0057,
                                         'roi_ci95': [-0.0669, 0.0539],
                                         'per_day_median': 11.0,
                                         'per_day_range': [0, 43],
                                         'zero_pick_days': 4,
                                         'days': 56},
                       'C p>=0.54 all': {'n': 448,
                                         'hit': 0.5737,
                                         'hit_ci95': [0.5268, 0.6183],
                                         'predicted': 0.5556,
                                         'avg_odds': 1.735,
                                         'roi': -0.0054,
                                         'roi_ci95': [-0.085, 0.0724],
                                         'per_day_median': 6.5,
                                         'per_day_range': [0, 21],
                                         'zero_pick_days': 5,
                                         'days': 56},
                       'C p>=0.56 all': {'n': 147,
                                         'hit': 0.6259,
                                         'hit_ci95': [0.551, 0.7007],
                                         'predicted': 0.5673,
                                         'avg_odds': 1.714,
                                         'roi': 0.0736,
                                         'roi_ci95': [-0.0567, 0.2026],
                                         'per_day_median': 2.0,
                                         'per_day_range': [0, 9],
                                         'zero_pick_days': 15,
                                         'days': 56},
                       'C p>=0.58 all': {'n': 5,
                                         'hit': 0.6,
                                         'hit_ci95': [0.2, 1.0],
                                         'predicted': 0.5942,
                                         'avg_odds': 1.74,
                                         'roi': 0.056,
                                         'roi_ci95': [-0.658, 0.77],
                                         'per_day_median': 0.0,
                                         'per_day_range': [0, 1],
                                         'zero_pick_days': 51,
                                         'days': 56},
                       'C p>=0.50 agree': {'n': 725,
                                           'hit': 0.56,
                                           'hit_ci95': [0.5241, 0.5972],
                                           'predicted': 0.5365,
                                           'avg_odds': 1.785,
                                           'roi': -0.0025,
                                           'roi_ci95': [-0.0654, 0.0626],
                                           'per_day_median': 9.0,
                                           'per_day_range': [0, 43],
                                           'zero_pick_days': 4,
                                           'days': 56},
                       'C p>=0.52 agree': {'n': 532,
                                           'hit': 0.5677,
                                           'hit_ci95': [0.5244, 0.609],
                                           'predicted': 0.5459,
                                           'avg_odds': 1.759,
                                           'roi': -0.003,
                                           'roi_ci95': [-0.0775, 0.0697],
                                           'per_day_median': 7.0,
                                           'per_day_range': [0, 32],
                                           'zero_pick_days': 7,
                                           'days': 56},
                       'C p>=0.54 agree': {'n': 322,
                                           'hit': 0.5652,
                                           'hit_ci95': [0.5124, 0.618],
                                           'predicted': 0.5563,
                                           'avg_odds': 1.734,
                                           'roi': -0.0207,
                                           'roi_ci95': [-0.1135, 0.0719],
                                           'per_day_median': 4.0,
                                           'per_day_range': [0, 17],
                                           'zero_pick_days': 8,
                                           'days': 56},
                       'C p>=0.56 agree': {'n': 118,
                                           'hit': 0.6186,
                                           'hit_ci95': [0.5339, 0.7034],
                                           'predicted': 0.5671,
                                           'avg_odds': 1.714,
                                           'roi': 0.0613,
                                           'roi_ci95': [-0.0854, 0.2095],
                                           'per_day_median': 2.0,
                                           'per_day_range': [0, 7],
                                           'zero_pick_days': 18,
                                           'days': 56},
                       'C p>=0.58 agree': {'n': 4,
                                           'hit': 0.5,
                                           'hit_ci95': [0.0, 1.0],
                                           'predicted': 0.5954,
                                           'avg_odds': 1.748,
                                           'roi': -0.1075,
                                           'roi_ci95': [-1.0, 0.785],
                                           'per_day_median': 0.0,
                                           'per_day_range': [0, 1],
                                           'zero_pick_days': 52,
                                           'days': 56},
                       'L1 p>=0.50 all': {'n': 1114,
                                          'hit': 0.5655,
                                          'hit_ci95': [0.5359, 0.5943],
                                          'predicted': 0.5344,
                                          'avg_odds': 1.79,
                                          'roi': 0.0113,
                                          'roi_ci95': [-0.0416, 0.0618],
                                          'per_day_median': 14.5,
                                          'per_day_range': [0, 71],
                                          'zero_pick_days': 2,
                                          'days': 56},
                       'L1 p>=0.52 all': {'n': 793,
                                          'hit': 0.5649,
                                          'hit_ci95': [0.5309, 0.599],
                                          'predicted': 0.5443,
                                          'avg_odds': 1.762,
                                          'roi': -0.0057,
                                          'roi_ci95': [-0.0669, 0.0539],
                                          'per_day_median': 11.0,
                                          'per_day_range': [0, 43],
                                          'zero_pick_days': 4,
                                          'days': 56},
                       'L1 p>=0.54 all': {'n': 448,
                                          'hit': 0.5737,
                                          'hit_ci95': [0.5268, 0.6183],
                                          'predicted': 0.5556,
                                          'avg_odds': 1.735,
                                          'roi': -0.0054,
                                          'roi_ci95': [-0.085, 0.0724],
                                          'per_day_median': 6.5,
                                          'per_day_range': [0, 21],
                                          'zero_pick_days': 5,
                                          'days': 56},
                       'L1 p>=0.56 all': {'n': 147,
                                          'hit': 0.6259,
                                          'hit_ci95': [0.551, 0.7007],
                                          'predicted': 0.5673,
                                          'avg_odds': 1.714,
                                          'roi': 0.0736,
                                          'roi_ci95': [-0.0567, 0.2026],
                                          'per_day_median': 2.0,
                                          'per_day_range': [0, 9],
                                          'zero_pick_days': 15,
                                          'days': 56},
                       'L1 p>=0.58 all': {'n': 5,
                                          'hit': 0.6,
                                          'hit_ci95': [0.2, 1.0],
                                          'predicted': 0.5942,
                                          'avg_odds': 1.74,
                                          'roi': 0.056,
                                          'roi_ci95': [-0.658, 0.77],
                                          'per_day_median': 0.0,
                                          'per_day_range': [0, 1],
                                          'zero_pick_days': 51,
                                          'days': 56},
                       'L1 p>=0.50 agree': {'n': 725,
                                            'hit': 0.56,
                                            'hit_ci95': [0.5241, 0.5972],
                                            'predicted': 0.5365,
                                            'avg_odds': 1.785,
                                            'roi': -0.0025,
                                            'roi_ci95': [-0.0654, 0.0626],
                                            'per_day_median': 9.0,
                                            'per_day_range': [0, 43],
                                            'zero_pick_days': 4,
                                            'days': 56},
                       'L1 p>=0.52 agree': {'n': 532,
                                            'hit': 0.5677,
                                            'hit_ci95': [0.5244, 0.609],
                                            'predicted': 0.5459,
                                            'avg_odds': 1.759,
                                            'roi': -0.003,
                                            'roi_ci95': [-0.0775, 0.0697],
                                            'per_day_median': 7.0,
                                            'per_day_range': [0, 32],
                                            'zero_pick_days': 7,
                                            'days': 56},
                       'L1 p>=0.54 agree': {'n': 322,
                                            'hit': 0.5652,
                                            'hit_ci95': [0.5124, 0.618],
                                            'predicted': 0.5563,
                                            'avg_odds': 1.734,
                                            'roi': -0.0207,
                                            'roi_ci95': [-0.1135, 0.0719],
                                            'per_day_median': 4.0,
                                            'per_day_range': [0, 17],
                                            'zero_pick_days': 8,
                                            'days': 56},
                       'L1 p>=0.56 agree': {'n': 118,
                                            'hit': 0.6186,
                                            'hit_ci95': [0.5339, 0.7034],
                                            'predicted': 0.5671,
                                            'avg_odds': 1.714,
                                            'roi': 0.0613,
                                            'roi_ci95': [-0.0854, 0.2095],
                                            'per_day_median': 2.0,
                                            'per_day_range': [0, 7],
                                            'zero_pick_days': 18,
                                            'days': 56},
                       'L1 p>=0.58 agree': {'n': 4,
                                            'hit': 0.5,
                                            'hit_ci95': [0.0, 1.0],
                                            'predicted': 0.5954,
                                            'avg_odds': 1.748,
                                            'roi': -0.1075,
                                            'roi_ci95': [-1.0, 0.785],
                                            'per_day_median': 0.0,
                                            'per_day_range': [0, 1],
                                            'zero_pick_days': 52,
                                            'days': 56},
                       'L2 e>=0.00 all': {'n': 9,
                                          'hit': 0.3333,
                                          'hit_ci95': [0.0, 0.6667],
                                          'predicted': 0.3997,
                                          'avg_odds': 2.626,
                                          'roi': -0.0339,
                                          'roi_ci95': [-1.0, 0.9143],
                                          'per_day_median': 0.0,
                                          'per_day_range': [0, 3],
                                          'zero_pick_days': 50,
                                          'days': 56},
                       'L2 e>=0.02 all': {'n': 3,
                                          'hit': 0.3333,
                                          'hit_ci95': [0.0, 1.0],
                                          'predicted': 0.4067,
                                          'avg_odds': 2.658,
                                          'roi': 0.1483,
                                          'roi_ci95': [-1.0, 2.445],
                                          'per_day_median': 0.0,
                                          'per_day_range': [0, 1],
                                          'zero_pick_days': 53,
                                          'days': 56},
                       'L2 e>=0.05 all': {'n': 1,
                                          'hit': 0.0,
                                          'hit_ci95': [0.0, 0.0],
                                          'predicted': 0.4479,
                                          'avg_odds': 2.36,
                                          'roi': -1.0,
                                          'roi_ci95': [-1.0, -1.0],
                                          'per_day_median': 0.0,
                                          'per_day_range': [0, 1],
                                          'zero_pick_days': 55,
                                          'days': 56},
                       'L2 e>=0.00 agree': {'n': 1,
                                            'hit': 1.0,
                                            'hit_ci95': [1.0, 1.0],
                                            'predicted': 0.4028,
                                            'avg_odds': 2.5,
                                            'roi': 1.5,
                                            'roi_ci95': [1.5, 1.5],
                                            'per_day_median': 0.0,
                                            'per_day_range': [0, 1],
                                            'zero_pick_days': 55,
                                            'days': 56},
                       'L2 e>=0.02 agree': {'n': 0},
                       'L2 e>=0.05 agree': {'n': 0}}},
 'sensitivity_holdout': {'without_kambi': {'C p>=0.50 all': {'n': 1117,
                                                             'hit': 0.5649,
                                                             'roi': 0.0105},
                                           'C p>=0.52 all': {'n': 795,
                                                             'hit': 0.5648,
                                                             'roi': -0.0059},
                                           'C p>=0.54 all': {'n': 449,
                                                             'hit': 0.5724,
                                                             'roi': -0.0078},
                                           'C p>=0.56 all': {'n': 146,
                                                             'hit': 0.6233,
                                                             'roi': 0.0686},
                                           'C p>=0.58 all': {'n': 5,
                                                             'hit': 0.6,
                                                             'roi': 0.031},
                                           'C p>=0.50 agree': {'n': 725,
                                                               'hit': 0.5586,
                                                               'roi': -0.0044},
                                           'C p>=0.52 agree': {'n': 532,
                                                               'hit': 0.5658,
                                                               'roi': -0.006},
                                           'C p>=0.54 agree': {'n': 322,
                                                               'hit': 0.5621,
                                                               'roi': -0.0263},
                                           'C p>=0.56 agree': {'n': 117,
                                                               'hit': 0.6154,
                                                               'roi': 0.0549},
                                           'C p>=0.58 agree': {'n': 4,
                                                               'hit': 0.5,
                                                               'roi': -0.1387}},
                         'cross_feed_median': {'C p>=0.50 all': {'n': 1075,
                                                                 'hit': 0.5619,
                                                                 'roi': 0.0062},
                                               'C p>=0.52 all': {'n': 746,
                                                                 'hit': 0.5576,
                                                                 'roi': -0.0187},
                                               'C p>=0.54 all': {'n': 392,
                                                                 'hit': 0.5638,
                                                                 'roi': -0.0232},
                                               'C p>=0.56 all': {'n': 115,
                                                                 'hit': 0.6087,
                                                                 'roi': 0.0432},
                                               'C p>=0.58 all': {'n': 7,
                                                                 'hit': 0.7143,
                                                                 'roi': 0.2236},
                                               'C p>=0.50 agree': {'n': 691,
                                                                   'hit': 0.5543,
                                                                   'roi': -0.0111},
                                               'C p>=0.52 agree': {'n': 492,
                                                                   'hit': 0.5549,
                                                                   'roi': -0.0257},
                                               'C p>=0.54 agree': {'n': 279,
                                                                   'hit': 0.5448,
                                                                   'roi': -0.0572},
                                               'C p>=0.56 agree': {'n': 94,
                                                                   'hit': 0.5957,
                                                                   'roi': 0.0208},
                                               'C p>=0.58 agree': {'n': 6,
                                                                   'hit': 0.6667,
                                                                   'roi': 0.1425}}},
 'decision_table': {'current_rule_hit_rate': 0.5831,
                    'marked_as_written': {'value': 'L2 e>=0.00 agree'},
                    'marked_n': {'value': 1},
                    'marked_with_a_real_interval': {},
                    'declared_comparisons': 36,
                    'expected_false_positives_at_95': 1.8},
 'blocker_blend_proxy_holdout': {'1.70-2.00': {'legs_per_day': 38.11,
                                               'clear_ev_gate': 1,
                                               'removed_by_confidence_alone': 0,
                                               'below_052_floor_alone_per_day': 21.41,
                                               'at_or_above_052': 935},
                                 '2.00-2.50': {'legs_per_day': 43.12,
                                               'clear_ev_gate': 1,
                                               'removed_by_confidence_alone': 1,
                                               'below_052_floor_alone_per_day': 43.12,
                                               'at_or_above_052': 0},
                                 '2.50-3.50': {'legs_per_day': 63.82,
                                               'clear_ev_gate': 0,
                                               'removed_by_confidence_alone': 0,
                                               'below_052_floor_alone_per_day': 63.82,
                                               'at_or_above_052': 0}},
 'prices': {'fixtures_with_a_price_path': 317,
            'legs': 1427,
            'median_gap_hours': 9.91,
            'primary_median_taken_ge_1.70': {'n': 2,
                                             'fixtures': 2,
                                             'avg_price': 2.54,
                                             'instant_edge': 0.0038,
                                             'closing_edge': -0.029,
                                             'closing_ci95': [-0.0324,
                                                              -0.0256],
                                             'verdict': 'SLOW'},
            'per_book_n_ge_30': {'TheOddsAPI-betfair_ex_eu': {'n': 183,
                                                              'fixtures': 163,
                                                              'avg_price': 5.1,
                                                              'instant_edge': 0.018,
                                                              'closing_edge': -0.014,
                                                              'closing_ci95': [-0.0283,
                                                                               -0.0002],
                                                              'verdict': 'SLOW'},
                                 'TheOddsAPI-coolbet': {'n': 72,
                                                        'fixtures': 65,
                                                        'avg_price': 4.77,
                                                        'instant_edge': 0.016,
                                                        'closing_edge': 0.0096,
                                                        'closing_ci95': [-0.007,
                                                                         0.0272],
                                                        'verdict': 'UNRESOLVED '
                                                                   'at this '
                                                                   'n'},
                                 'TheOddsAPI-leovegas_se': {'n': 37,
                                                            'fixtures': 34,
                                                            'avg_price': 3.66,
                                                            'instant_edge': 0.0207,
                                                            'closing_edge': 0.0299,
                                                            'closing_ci95': [0.0132,
                                                                             0.0487],
                                                            'verdict': 'REAL'},
                                 'TheOddsAPI-matchbook': {'n': 187,
                                                          'fixtures': 161,
                                                          'avg_price': 4.83,
                                                          'instant_edge': 0.017,
                                                          'closing_edge': -0.0164,
                                                          'closing_ci95': [-0.0291,
                                                                           -0.0033],
                                                          'verdict': 'SLOW'},
                                 'TheOddsAPI-onexbet': {'n': 64,
                                                        'fixtures': 58,
                                                        'avg_price': 4.46,
                                                        'instant_edge': 0.0194,
                                                        'closing_edge': 0.0331,
                                                        'closing_ci95': [0.0118,
                                                                         0.0612],
                                                        'verdict': 'REAL'},
                                 'TheOddsAPI-unibet_nl': {'n': 93,
                                                          'fixtures': 86,
                                                          'avg_price': 5.33,
                                                          'instant_edge': 0.029,
                                                          'closing_edge': 0.0263,
                                                          'closing_ci95': [0.0057,
                                                                           0.0485],
                                                          'verdict': 'REAL'},
                                 'TheOddsAPI-unibet_se': {'n': 99,
                                                          'fixtures': 89,
                                                          'avg_price': 5.22,
                                                          'instant_edge': 0.0264,
                                                          'closing_edge': 0.0282,
                                                          'closing_ci95': [0.0102,
                                                                           0.047],
                                                          'verdict': 'REAL'}},
            'fresh_subset_n': 0,
            'book_quotes_unevaluated': 20565,
            'column_present': False},
 'pre_experiment': {'until': '2026-08-05',
                    'rows': 1018,
                    'settled_valid': 1012,
                    'pick_days': 132,
                    'as_saved': {'all': {'n': 1012,
                                         'hit': 0.5119,
                                         'hit_ci95': [0.4812, 0.5435],
                                         'predicted': 0.5976,
                                         'avg_odds': 1.96,
                                         'roi': -0.0395,
                                         'roi_ci95': [-0.0995, 0.0215]},
                                 'below 1.70': {'n': 349,
                                                'hit': 0.6189,
                                                'hit_ci95': [0.5702, 0.6676],
                                                'predicted': 0.676,
                                                'avg_odds': 1.54,
                                                'roi': -0.0475,
                                                'roi_ci95': [-0.1253,
                                                             0.0296]},
                                 '1.70-2.00': {'n': 267,
                                               'hit': 0.5169,
                                               'hit_ci95': [0.4532, 0.573],
                                               'predicted': 0.6054,
                                               'avg_odds': 1.84,
                                               'roi': -0.0521,
                                               'roi_ci95': [-0.1667, 0.0529]},
                                 '2.00-2.50': {'n': 278,
                                               'hit': 0.4532,
                                               'hit_ci95': [0.3957, 0.5144],
                                               'predicted': 0.5404,
                                               'avg_odds': 2.23,
                                               'roi': 0.0,
                                               'roi_ci95': [-0.1254, 0.1318]},
                                 '2.50-3.50': {'n': 111,
                                               'hit': 0.3153,
                                               'hit_ci95': [0.2342, 0.4054],
                                               'predicted': 0.4831,
                                               'avg_odds': 2.76,
                                               'roi': -0.122,
                                               'roi_ci95': [-0.3557, 0.1226]},
                                 'above 3.50': {'n': 7,
                                                'hit': 0.4286,
                                                'hit_ci95': [0.1429, 0.8571],
                                                'predicted': 0.477,
                                                'avg_odds': 3.75,
                                                'roi': 0.5814,
                                                'roi_ci95': [-0.48, 2.1415]}},
                    'with_todays_gates': {'all': {'n': 470,
                                                  'hit': 0.5213,
                                                  'hit_ci95': [0.4766, 0.566],
                                                  'predicted': 0.6184,
                                                  'avg_odds': 1.93,
                                                  'roi': -0.0076,
                                                  'roi_ci95': [-0.0944,
                                                               0.0823]},
                                          'below 1.70': {'n': 116,
                                                         'hit': 0.569,
                                                         'hit_ci95': [0.4828,
                                                                      0.6552],
                                                         'predicted': 0.6938,
                                                         'avg_odds': 1.6,
                                                         'roi': -0.0899,
                                                         'roi_ci95': [-0.2291,
                                                                      0.0505]},
                                          '1.70-2.00': {'n': 187,
                                                        'hit': 0.5134,
                                                        'hit_ci95': [0.4439,
                                                                     0.5829],
                                                        'predicted': 0.6199,
                                                        'avg_odds': 1.85,
                                                        'roi': -0.0504,
                                                        'roi_ci95': [-0.1802,
                                                                     0.077]},
                                          '2.00-2.50': {'n': 147,
                                                        'hit': 0.517,
                                                        'hit_ci95': [0.4422,
                                                                     0.5986],
                                                        'predicted': 0.5659,
                                                        'avg_odds': 2.17,
                                                        'roi': 0.1198,
                                                        'roi_ci95': [-0.0527,
                                                                     0.2952]},
                                          '2.50-3.50': {'n': 20,
                                                        'hit': 0.35,
                                                        'hit_ci95': [0.15,
                                                                     0.55],
                                                        'predicted': 0.5519,
                                                        'avg_odds': 2.7,
                                                        'roi': -0.0655,
                                                        'roi_ci95': [-0.6085,
                                                                     0.5075]},
                                          'above 3.50': {'n': 0}},
                    'confidence_gate_alone': {'2.00-2.50': {'clear_every_other_gate': 248,
                                                            'removed': 101,
                                                            'per_pick_day': 0.77},
                                              'below 1.70': {'clear_every_other_gate': 116,
                                                             'removed': 0,
                                                             'per_pick_day': 0.0},
                                              '1.70-2.00': {'clear_every_other_gate': 190,
                                                            'removed': 3,
                                                            'per_pick_day': 0.02},
                                              '2.50-3.50': {'clear_every_other_gate': 107,
                                                            'removed': 87,
                                                            'per_pick_day': 0.66},
                                              'above 3.50': {'clear_every_other_gate': 6,
                                                             'removed': 6,
                                                             'per_pick_day': 0.05}}},
 'verdicts': {'BLEND': 'THE MARKET TAKES 93-100% OF THE WEIGHT in every '
                       'family. The blend is not distinguishable from the '
                       'price on development or on the holdout; it is '
                       'carried forward as the registered probability anyway',
              'AGREEMENT': 'RAISES the hit rate among legs priced >= 1.70 -- '
                           'and it does so by selecting SHORTER prices. '
                           'Inside the selection cells, where price and '
                           'probability are already bounded, the filter adds '
                           'nothing',
              'BACKTEST': 'no cell with a real interval clears the current '
                          "rule's holdout hit rate; every ROI interval "
                          'includes zero',
              'PRICES': 'UNEVALUATED at the median: the median beats the '
                        "anchor's fair price on 2 of 1,427 legs. Per book: "
                        'the two exchanges are SLOW; three Kambi-family '
                        "books and onexbet retain their edge to the anchor's "
                        'closing price. Quote age UNEVALUATED',
              'DECISION_TABLE': 'nothing recommended; no cell marked on a '
                                'real interval'}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--open-holdout", action="store_true")
    ap.add_argument("--prices", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--measured", action="store_true",
                    help="print the recorded results; runs nothing")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--weights", default=None,
                    help="JSON written by --dev; required by --open-holdout")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args()
    if a.control:
        r = control()
    elif a.dev:
        r = run("dev", a.cache)
    elif a.open_holdout:
        if not a.weights:
            ap.error("--open-holdout needs --weights from the --dev run")
        with open(a.weights, encoding="utf-8") as fh:
            dev = json.load(fh)
        w = {variant: dev["variants"][variant]["weights_used"]
             for variant in ("corrected", "deployed")}
        r = run("holdout", a.cache, weights_in=w)
        r["decision_table"] = decision_table(r)
        r["decision_table_deployed_elo"] = decision_table(r, "deployed")
    elif a.prices:
        r = prices()
    elif a.gates:
        r = pre_experiment()
    elif a.measured:
        r = MEASURED
    else:
        ap.error("choose a step")
    text = json.dumps(r, indent=1, default=str)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"written to {a.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
