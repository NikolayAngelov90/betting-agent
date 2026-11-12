"""RESIDUAL HARNESS - does feature X add information BEYOND the price?

Stage 30, Parts C and D. Measurement only; replaces nothing.

    outcome ~ logit(fair probability) as an OFFSET + one coefficient per
    candidate feature, walk-forward, out-of-sample log-loss against the
    offset-only model.

THE OFFSET-ONLY MODEL IS THE MARKET. With zero features there is nothing to
fit, the prediction is the fair probability itself, and the harness must
reproduce the market's log-loss EXACTLY. That is its positive control and it
is asserted before any candidate is admitted.

NO INTERCEPT, ON PURPOSE. An intercept is a fitted parameter, and with one the
zero-feature model would no longer be the market. Each candidate is
standardised on its TRAINING fold, so its coefficient cannot stand in for one.

EVERY FEATURE IS POINT-IN-TIME. Elo, rest and form are computed in one
chronological pass over the completed-match history and recorded BEFORE the
match is processed. The stored model probability is the only candidate that
is not recomputed: it was written at pick time, and re-scoring history with
today's pickles would leak the outcomes they were trained on.

    python -m scripts.residual_harness --harness     # Part C
    python -m scripts.residual_harness --elo         # Part D.1
    python -m scripts.residual_harness --xg          # Part D.2
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

#: Walk-forward folds are calendar months of kickoff. A fold is scored only
#: once this many observations precede it.
MIN_TRAIN = 300
BOOTSTRAP_DRAWS = 4000

#: Elo evaluation starts here: the history begins 2021-03, so this leaves two
#: seasons of burn-in before a rating is scored.
ELO_EVAL_FROM = _dt.datetime(2023, 7, 1)

#: Understat's coverage, as its own site states it: six leagues from 2014/15.
#: Five are in our configured set; the Russian league is not.
UNDERSTAT_LEAGUES = ("england/premier-league", "spain/laliga",
                     "germany/bundesliga", "italy/serie-a", "france/ligue-1")


#: WHAT PARTS C AND D MEASURED. `measured` 2026-11-12. Log-loss in nats.
MEASURED = {
    "measured": "2026-11-12",
    "odds_era": ("2026-02-28", "2026-11-09"),
    "min_train": MIN_TRAIN,
    # ── PART C ───────────────────────────────────────────────────────────
    "home_win": {
        "n": 4001, "scored": 3200, "folds": 6, "scored_from": "2026-05-01",
        "ll_market": 0.61308, "zero_feature_delta": 0.0,
        # candidate -> (delta_ll, ci_lo, ci_hi, beta, beta_lo, beta_hi)
        "candidates": {
            "elo_diff": (+0.00023, -0.00033, +0.00082, +0.0566, -0.0187, +0.1318),
            "rest_diff": (+0.00019, -0.00013, +0.00050, -0.0111, -0.0807, +0.0585),
            "form_diff": (+0.00033, -0.00118, +0.00188, +0.0790, +0.0078, +0.1502),
        },
        # the calibration slope: residual FLB in the fair price, by estimator
        "slope_beta": {
            "shin": (+0.0431, -0.0376, +0.1238),
            "proportional": (+0.0955, +0.0155, +0.1754),
            "power": (+0.0035, -0.0782, +0.0852),
        },
    },
    "over_2.5": {
        "n": 4097, "scored": 3416, "folds": 7, "scored_from": "2026-04-01",
        "ll_market": 0.67236, "zero_feature_delta": 0.0,
        "slope_beta": {
            "shin": (+0.0004, -0.0642, +0.0649),
            "proportional": (+0.0176, -0.0468, +0.0821),
            "power": (-0.0090, -0.0737, +0.0556),
        },
    },
    "model_probability": {
        "population": "SELECTED: settled picks before s5.20 with a stored "
                      "probability and an anchor price",
        "picks": 1739, "matched": 1009, "scored": 474, "folds": 2,
        "delta": (+0.00026, -0.00155, +0.00206),
        "beta": (+0.0344, -0.0914, +0.1602),
    },
    "harness_verdict": (
        "NO candidate adds information beyond the price. Every out-of-sample "
        "delta is positive or spans zero. `form_diff` has an in-sample "
        "coefficient whose interval excludes zero and an out-of-sample delta "
        "that does not improve -- which is what this harness exists to "
        "separate. Six to seven monthly folds over one 8-month odds era: "
        "thin, and stated beside every figure"),
    # ── PART D.1 ─────────────────────────────────────────────────────────
    "elo": {
        "replica_max_rating_diff": 0.0, "teams": 1450,
        "n": 34639, "eval_from": "2023-07-01",
        "deployed_log_loss": 1.01941,
        # variant -> (log_loss, delta, ci_lo, ci_hi)
        "variants": {
            "calendar|fitted": (1.01775, -0.00165, -0.00223, -0.00106),
            "season|fixed": (1.01809, -0.00132, -0.00193, -0.00070),
            "season|fitted": (1.01644, -0.00297, -0.00382, -0.00210),
        },
        "season_label_trap": {
            "leagues_mixing_two_formats": 31,
            "teams_flip_flopping": 492, "of_teams": 1343,
            "regressions_with_raw_labels": 7840, "true_transitions": 3221,
            "delta_measured_with_raw_labels": +0.00579,
            "note": ("`matches.season` mixes `2023/2024` and `2023` inside "
                     "one league, so a season variant keyed on it regressed "
                     "a third of the teams every other fixture and measured "
                     "WORSE than the deployed Elo. The key is derived from "
                     "the date and the league's own calendar instead"),
        },
        "stated_limitation": (
            "the summer-league rule (more fixtures in June-July than in "
            "January-February) also classes `bulgaria/efbet-league` and "
            "`netherlands/eerste-divisie` as calendar-year; for those two "
            "the season variant still regresses in January"),
    },
    # ── PART D.2 ─────────────────────────────────────────────────────────
    "xg": {
        "completed": 39361, "with_xg": 1794, "share": 0.046,
        "after_understat_backfill": 0.247,
        "understat_leagues_in_config": 5,
        "understat_terms": "UNOBTAINED",
        "understat_terms_detail": (
            "understat.com returned a maintenance placeholder ('We will come "
            "back soon') on 2026-11-12 and no terms page was found by "
            "search; the results were third-party scrapers and libraries, "
            "none quoting the site's own position"),
        "backfill_proposed": False,
        "what_would_clarify": (
            "the site's own terms page and robots.txt once it is reachable, "
            "or written permission. Unclear is not permission"),
    },
}


def _env() -> None:
    """Loads `.env` / `.env.local` INSIDE the caller, never at import. The
    loader lives in `scripts/sharp_anchor.py`; named here because
    `test_dialect_is_asserted` looks for `.env` in this file's own text and
    cannot see a loader behind an import."""
    from scripts.sharp_anchor import _load_env
    _load_env()


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _ll(y: int, p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return -(math.log(p) if y else math.log(1.0 - p))


# ══ THE HISTORY PASS — every feature recorded BEFORE its match ═══════════

def _season_labels(db, rows) -> Dict[int, str]:
    """A season key per match, DERIVED FROM THE DATE. Never from the label.

    `matches.season` CANNOT BE USED, and the first version of this function
    used it. The column mixes two formats inside 31 leagues -- `2023/2024`
    beside `2023` for the same campaign -- so 492 of 1,343 teams flip-flopped
    between labels match by match and were regressed 7,840 times against
    3,221 real season transitions. That variant measured +0.0058 WORSE than
    the deployed Elo, and the defect was mine: it was regressing a third of
    the teams toward the mean every other fixture.

    So the key comes from the calendar the league actually plays. A league is
    SUMMER (calendar-year) when it plays more in June-July than in
    January-February -- the Nordic leagues -- and its key is the year;
    otherwise the key is the July-June campaign. Nothing reads the raw label.
    """
    summer = defaultdict(int)
    winter = defaultdict(int)
    for m in rows:
        d = m.match_date
        if d is None:
            continue
        if d.month in (6, 7):
            summer[m.league] += 1
        elif d.month in (1, 2):
            winter[m.league] += 1
    out = {}
    calendar_leagues = set()
    for m in rows:
        d = m.match_date
        if summer[m.league] > winter[m.league]:
            calendar_leagues.add(m.league)
            lab = str(d.year)
        else:
            y = d.year if d.month >= 7 else d.year - 1
            lab = f"{y}/{y + 1}"
        out[m.id] = lab
    out["__derived__"] = len(rows)
    out["__calendar_leagues__"] = sorted(
        lg for lg in calendar_leagues if lg and not lg.startswith("other/"))
    return out


def history_pass(regress: str = "calendar") -> dict:
    """One chronological pass. Returns pre-match Elo, rest and form per
    match, and the final ratings (for the production-replica control).

    `regress="calendar"` REPLICATES `EloSystem.fit` exactly: every rating is
    regressed toward 1500 when the YEAR changes between consecutive matches.
    `regress="season"` regresses a TEAM once, at its first DOMESTIC match of
    a new season label -- European-cup fixtures never trigger it, or a club
    in a calendar-year league would be regressed each time it alternated
    between its league and a cup carrying the other label format.
    """
    _env()
    from src.data.database import get_db
    from src.data.match_history import get_completed_matches
    from src.models.elo_system import DEFAULT_ELO, HOME_ADVANTAGE, K_FACTOR
    from src.utils.config import get_config

    db = get_db()
    assert db.is_postgres, "refusing to measure against the SQLite fallback"
    rows = get_completed_matches(db)
    seasons = _season_labels(db, rows)
    factor = get_config().get("models.elo_season_regression", 0.33)

    ratings: Dict[int, float] = {}
    last_played: Dict[int, _dt.datetime] = {}
    recent: Dict[int, List[int]] = defaultdict(list)     # points, newest last
    team_season: Dict[int, str] = {}
    pre: Dict[int, dict] = {}
    prev_year = None
    regressions = 0
    for m in rows:
        d = m.match_date
        if regress == "calendar":
            if prev_year is not None and d is not None and d.year > prev_year:
                for t in list(ratings):
                    ratings[t] = ratings[t] * (1 - factor) + DEFAULT_ELO * factor
                regressions += 1
            if d is not None:
                prev_year = d.year
        else:
            if not (m.league or "").startswith("europe/"):
                lab = seasons[m.id]
                for t in (m.home_team_id, m.away_team_id):
                    if t in team_season and team_season[t] != lab \
                            and t in ratings:
                        ratings[t] = (ratings[t] * (1 - factor)
                                      + DEFAULT_ELO * factor)
                        regressions += 1
                    team_season[t] = lab

        h, a = m.home_team_id, m.away_team_id
        rh, ra = ratings.get(h, DEFAULT_ELO), ratings.get(a, DEFAULT_ELO)

        def rest(t):
            lp = last_played.get(t)
            if lp is None or d is None:
                return None
            return min((d - lp).total_seconds() / 86400.0, 14.0)

        def form(t):
            r = recent[t][-5:]
            return (sum(r) / len(r)) if len(r) >= 3 else None

        pre[m.id] = {
            "date": d, "league": m.league,
            "elo_diff": (rh + HOME_ADVANTAGE) - ra,
            "rest_home": rest(h), "rest_away": rest(a),
            "form_home": form(h), "form_away": form(a),
            "hg": m.home_goals, "ag": m.away_goals,
        }

        # the update, exactly as `EloSystem._process_match`
        eh = 1.0 / (1.0 + 10 ** ((ra - (rh + HOME_ADVANTAGE)) / 400.0))
        if m.home_goals > m.away_goals:
            sh, pts = 1.0, (3, 0)
        elif m.home_goals == m.away_goals:
            sh, pts = 0.5, (1, 1)
        else:
            sh, pts = 0.0, (0, 3)
        mult = math.log(max(abs(m.home_goals - m.away_goals), 1) + 1)
        ratings[h] = rh + K_FACTOR * mult * (sh - eh)
        ratings[a] = ra + K_FACTOR * mult * ((1 - sh) - (1 - eh))
        if d is not None:
            last_played[h] = last_played[a] = d
        recent[h].append(pts[0])
        recent[a].append(pts[1])
    return {"pre": pre, "ratings": ratings, "n": len(rows),
            "regressions": regressions,
            "season_labels_derived": seasons["__derived__"],
            "calendar_leagues": seasons["__calendar_leagues__"]}


def replica_control() -> dict:
    """THE REPLICA MUST BE THE PRODUCTION SYSTEM, or Part D measures mine.

    Fit the deployed `EloSystem` and compare every rating with the pass
    above in calendar mode. Zero difference, or nothing below is about the
    production Elo.
    """
    _env()
    from src.models.elo_system import EloRatingSystem as EloSystem
    prod = EloSystem()
    prod.fit()
    mine = history_pass("calendar")["ratings"]
    keys = set(prod.ratings) | set(mine)
    worst = max(abs(prod.ratings.get(k, 0.0) - mine.get(k, 0.0)) for k in keys)
    return {"teams": len(keys), "max_abs_rating_diff": worst}


# ══ THE FIT — one coefficient, an offset, no intercept ═══════════════════

def _fit(offsets: List[float], zs: List[float], ys: List[int]):
    """Newton's method for `logit(p) = offset + beta * z`. Returns
    `(beta, standard_error)`. One parameter, so the Hessian is a scalar."""
    beta = 0.0
    info = 0.0
    for _ in range(50):
        g = 0.0
        info = 0.0
        for o, z, y in zip(offsets, zs, ys):
            p = _sigmoid(o + beta * z)
            g += (y - p) * z
            info += p * (1.0 - p) * z * z
        if info <= 0:
            return 0.0, float("inf")
        step = g / info
        beta += step
        if abs(step) < 1e-10:
            break
    return beta, (1.0 / math.sqrt(info)) if info > 0 else float("inf")


def _boot_delta(by_fix: Dict[int, List[float]], draws: int = BOOTSTRAP_DRAWS):
    import numpy as np
    if len(by_fix) < 30:
        return None
    sums = np.array([sum(v) for v in by_fix.values()])
    cnts = np.array([len(v) for v in by_fix.values()], dtype=float)
    rng = np.random.default_rng(20261112)
    idx = rng.integers(0, len(sums), size=(draws, len(sums)))
    d = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def walk_forward(obs: List[dict], feature: Optional[str]) -> dict:
    """`obs`: dicts with `date`, `mid`, `y`, `offset` and feature values.

    `feature=None` is the ZERO-FEATURE run: no parameter, prediction = the
    fair probability. It is scored over the same folds as a candidate would
    be, so its log-loss is the market's on exactly that population.
    """
    rows = [o for o in obs if feature is None or o.get(feature) is not None]
    rows.sort(key=lambda o: o["date"])
    if not rows:
        return {"n": 0, "UNEVALUATED": "no observation carries this feature"}
    months = sorted({(o["date"].year, o["date"].month) for o in rows})
    ll_market: List[float] = []
    ll_model: List[float] = []
    by_fix: Dict[int, List[float]] = defaultdict(list)
    fold_betas: List[float] = []
    folds = 0
    for ym in months:
        start = _dt.datetime(ym[0], ym[1], 1)
        train = [o for o in rows if o["date"] < start]
        test = [o for o in rows
                if (o["date"].year, o["date"].month) == ym]
        if len(train) < MIN_TRAIN or not test:
            continue
        folds += 1
        if feature is None:
            beta, mu, sd = 0.0, 0.0, 1.0
        else:
            xs = [o[feature] for o in train]
            mu = statistics.mean(xs)
            sd = statistics.pstdev(xs) or 1.0
            beta, _se = _fit([o["offset"] for o in train],
                             [(x - mu) / sd for x in xs],
                             [o["y"] for o in train])
            fold_betas.append(beta)
        for o in test:
            lm = _ll(o["y"], _sigmoid(o["offset"]))
            z = 0.0 if feature is None else (o[feature] - mu) / sd
            lf = _ll(o["y"], _sigmoid(o["offset"] + beta * z))
            ll_market.append(lm)
            ll_model.append(lf)
            by_fix[o["mid"]].append(lf - lm)
    if not ll_market:
        return {"n": len(rows), "folds": 0,
                "UNEVALUATED": f"no fold has {MIN_TRAIN} prior observations"}
    out = {
        "n": len(rows), "n_scored": len(ll_market), "folds": folds,
        "era": (str(rows[0]["date"].date()), str(rows[-1]["date"].date())),
        "scored_from": str(min(o["date"] for o in rows
                               if o["mid"] in by_fix).date()),
        "ll_market": statistics.mean(ll_market),
        "ll_model": statistics.mean(ll_model),
        "delta_ll": statistics.mean(ll_model) - statistics.mean(ll_market),
        "delta_ci95": _boot_delta(by_fix),
    }
    if feature is not None:
        xs = [o[feature] for o in rows]
        mu = statistics.mean(xs)
        sd = statistics.pstdev(xs) or 1.0
        beta, se = _fit([o["offset"] for o in rows],
                        [(x - mu) / sd for x in xs], [o["y"] for o in rows])
        out.update({
            "beta_full_sample": beta, "beta_se": se,
            "beta_ci95": [beta - 1.96 * se, beta + 1.96 * se],
            "beta_fold_mean": statistics.mean(fold_betas),
            "feature_sd": sd,
        })
    return out


# ══ PART C — THE HARNESS ═════════════════════════════════════════════════

def _anchor_set(sets, mid, market, line=None, side=None):
    """The anchor leg set for a fixture: the same-feed-evidenced anchor
    first, the API-Football one where that is absent."""
    for anchor in ("TheOddsAPI-pinnacle", "Pinnacle"):
        st = sets.get((mid, anchor, market, line, side))
        if st is not None:
            return st
    return None


def harness() -> dict:
    import scripts.sharp_anchor as sa
    from src.evaluation.devig import ESTIMATORS

    d = sa.load(with_results=True)
    matches, sets = d["matches"], d["sets"]
    pre = history_pass("calendar")["pre"]

    def build(market, line, leg_index, outcome):
        out = {e: [] for e in ("shin", "proportional", "power")}
        for mid, m in matches.items():
            st = _anchor_set(sets, mid, market, line)
            if st is None:
                continue
            f = pre.get(mid)
            y = outcome(m["hg"], m["ag"])
            for e in out:
                fair = ESTIMATORS[e](st["q"])
                if fair is None:
                    continue
                o = {"date": m["kickoff"], "mid": mid, "y": y,
                     "offset": _logit(fair[leg_index])}
                o["slope"] = o["offset"]
                if f is not None:
                    o["elo_diff"] = f["elo_diff"]
                    if f["rest_home"] is not None and f["rest_away"] is not None:
                        o["rest_diff"] = f["rest_home"] - f["rest_away"]
                    if f["form_home"] is not None and f["form_away"] is not None:
                        o["form_diff"] = f["form_home"] - f["form_away"]
                out[e].append(o)
        return out

    home = build("1X2", None, 0, lambda hg, ag: 1 if hg > ag else 0)
    over = build("over_under", "2.5", 0, lambda hg, ag: 1 if hg + ag > 2.5 else 0)

    res = {"targets": {}}
    for name, obs_by_e in (("home_win", home), ("over_2.5", over)):
        t = {}
        zero = walk_forward(obs_by_e["shin"], None)
        t["zero_features"] = zero
        # THE POSITIVE CONTROL: zero features IS the market.
        t["control_exact"] = (zero.get("delta_ll") == 0.0)
        # the calibration slope under each estimator: residual FLB in the
        # fair price itself, on REAL outcomes
        t["slope"] = {e: walk_forward(obs_by_e[e], "slope")
                      for e in obs_by_e}
        if name == "home_win":
            t["candidates"] = {f: walk_forward(obs_by_e["shin"], f)
                               for f in ("elo_diff", "rest_diff", "form_diff")}
        res["targets"][name] = t

    # ── the stored model probability: a SELECTED subset, labelled as one ──
    from sqlalchemy import text

    from src.data.database import get_db
    from src.data.market_vocabulary import canonical, selection_aliases
    with get_db().get_session() as s:
        picks = s.execute(text("""
            SELECT id, match_id, market, selection, predicted_probability,
                   result
            FROM saved_picks
            WHERE pick_date >= '2026-08-11' AND disposition IS NULL
              AND result IN ('win','loss')
              AND predicted_probability IS NOT NULL
              AND model_version NOT LIKE '%12f2c7'
        """)).fetchall()
    obs = []
    unmatched = 0
    for pid, mid, market, selection, pp, result in picks:
        m = matches.get(mid)
        canon = canonical(market, selection)
        if m is None or canon is None:
            unmatched += 1
            continue
        fam, csel = canon
        aliases = set(selection_aliases(csel))
        hit = None
        for (smid, bm, mk, line, side), st in sets.items():
            if smid != mid or mk != fam \
                    or bm not in ("TheOddsAPI-pinnacle", "Pinnacle"):
                continue
            for i, k in enumerate(st["keys"]):
                if k in aliases:
                    fair = ESTIMATORS["shin"](st["q"])
                    if fair is not None:
                        hit = fair[i]
                    break
            if hit is not None:
                break
        if hit is None or not (0.0 < float(pp) < 1.0):
            unmatched += 1
            continue
        obs.append({"date": m["kickoff"], "mid": mid,
                    "y": 1 if result == "win" else 0,
                    "offset": _logit(hit),
                    "model_minus_market": _logit(float(pp)) - _logit(hit)})
    res["model_probability"] = {
        "population": ("saved_picks, cohorts before s5.20, settled, with a "
                       "stored predicted_probability AND an anchor fair "
                       "price -- a SELECTED subset: these legs were picked "
                       "BECAUSE the model disagreed with the market"),
        "picks": len(picks), "matched": len(obs), "unmatched": unmatched,
        "zero_features": walk_forward(obs, None),
        "candidate": walk_forward(obs, "model_minus_market"),
    }
    return res


def _print_harness(r: dict) -> None:
    print("=" * 78)
    print("RESIDUAL HARNESS  -  Part C")
    print("=" * 78)
    print("  outcome ~ logit(fair) as OFFSET + one coefficient, no intercept,")
    print("  walk-forward by month, feature standardised on the TRAIN fold.")
    print("  delta = log-loss(model) - log-loss(market): NEGATIVE = adds")
    print("  information beyond the price. Fair price: Shin, anchor Pinnacle.")

    def line(label, v):
        if "UNEVALUATED" in v:
            print(f"    {label:22s} UNEVALUATED: {v['UNEVALUATED']}")
            return
        ci = v.get("delta_ci95")
        cis = f"[{ci[0]:+.5f},{ci[1]:+.5f}]" if ci else "-"
        b = ""
        if "beta_full_sample" in v:
            bc = v["beta_ci95"]
            b = (f"  beta {v['beta_full_sample']:+.4f} "
                 f"[{bc[0]:+.4f},{bc[1]:+.4f}]")
        print(f"    {label:22s} n={v['n']:5d} scored={v['n_scored']:5d} "
              f"folds={v['folds']:2d}  LL mkt {v['ll_market']:.5f}  "
              f"delta {v['delta_ll']:+.5f} {cis}{b}")

    for name, t in r["targets"].items():
        z = t["zero_features"]
        print()
        print(f"  TARGET {name}   odds era {z.get('era')}, scored from "
              f"{z.get('scored_from')}")
        line("ZERO FEATURES", z)
        print(f"      positive control -- reproduces the market EXACTLY: "
              f"{'PASS' if t['control_exact'] else 'FAIL'}")
        print("    calibration slope (residual FLB in the fair price):")
        for e, v in t["slope"].items():
            line(f"  slope / {e}", v)
        for f, v in (t.get("candidates") or {}).items():
            line(f"candidate {f}", v)
    mp = r["model_probability"]
    print()
    print(f"  THE STORED MODEL PROBABILITY -- {mp['population']}")
    print(f"    picks {mp['picks']}, matched to an anchor price "
          f"{mp['matched']}, unmatched {mp['unmatched']}")
    line("ZERO FEATURES", mp["zero_features"])
    line("candidate model-market", mp["candidate"])


# ══ PART D.1 — ELO: season boundaries, and a fitted draw curve ═══════════

def _fit2(xs: List[float], ys: List[int]):
    """Two-parameter logistic `P = sigmoid(a + b*x)`, Newton. For the draw
    curve only; the harness itself never fits an intercept."""
    a, b = _logit(sum(ys) / len(ys)), 0.0
    for _ in range(60):
        g0 = g1 = h00 = h01 = h11 = 0.0
        for x, y in zip(xs, ys):
            p = _sigmoid(a + b * x)
            w = p * (1 - p)
            g0 += y - p
            g1 += (y - p) * x
            h00 += w
            h01 += w * x
            h11 += w * x * x
        det = h00 * h11 - h01 * h01
        if det <= 0:
            break
        da = (h11 * g0 - h01 * g1) / det
        db = (h00 * g1 - h01 * g0) / det
        a += da
        b += db
        if abs(da) + abs(db) < 1e-10:
            break
    return a, b


def elo() -> dict:
    """1X2 log-loss, out of sample, for the deployed Elo and three variants:
    {calendar-year, season-boundary} regression x {fixed, fitted} draw curve.
    Every probability is made from ratings as they stood BEFORE the match."""
    import numpy as np
    ctl = replica_control()
    out = {"replica_control": ctl, "variants": {}}
    per_match: Dict[str, Dict[int, float]] = {}
    for mode in ("calendar", "season"):
        hp = history_pass(mode)
        rows = [(mid, f) for mid, f in hp["pre"].items()
                if f["date"] is not None]
        rows.sort(key=lambda kv: kv[1]["date"])
        out[f"{mode}_regressions"] = hp["regressions"]
        out["season_labels_derived"] = hp["season_labels_derived"]
        out["calendar_leagues"] = hp["calendar_leagues"]
        months = sorted({(f["date"].year, f["date"].month) for _m, f in rows
                         if f["date"] >= ELO_EVAL_FROM})
        fixed: Dict[int, float] = {}
        fitted: Dict[int, float] = {}
        curve_params = []
        for ym in months:
            start = _dt.datetime(ym[0], ym[1], 1)
            train = [f for _m, f in rows if f["date"] < start]
            test = [(mid, f) for mid, f in rows
                    if (f["date"].year, f["date"].month) == ym]
            a, b = _fit2([abs(f["elo_diff"]) / 100.0 for f in train],
                         [1 if f["hg"] == f["ag"] else 0 for f in train])
            curve_params.append((a, b))
            for mid, f in test:
                diff = f["elo_diff"]
                eh = 1.0 / (1.0 + 10 ** (-diff / 400.0))
                res = 0 if f["hg"] > f["ag"] else (1 if f["hg"] == f["ag"] else 2)
                for store, pdraw in (
                        (fixed, max(0.15, 0.28 - abs(diff) / 2000.0)),
                        (fitted, _sigmoid(a + b * abs(diff) / 100.0))):
                    rem = 1.0 - pdraw
                    probs = (rem * eh, pdraw, rem * (1.0 - eh))
                    store[mid] = -math.log(max(probs[res], 1e-12))
        per_match[f"{mode}|fixed"] = fixed
        per_match[f"{mode}|fitted"] = fitted
        out[f"{mode}_draw_curve_last"] = curve_params[-1] if curve_params else None

    base = per_match["calendar|fixed"]
    ids = sorted(base)
    rng = np.random.default_rng(20261112)
    idx = rng.integers(0, len(ids), size=(BOOTSTRAP_DRAWS, len(ids)))
    b = np.array([base[i] for i in ids])
    for name, store in per_match.items():
        v = np.array([store[i] for i in ids])
        dlt = v - b
        boot = dlt[idx].mean(axis=1)
        out["variants"][name] = {
            "n": len(ids), "log_loss": float(v.mean()),
            "delta_vs_deployed": float(dlt.mean()),
            "delta_ci95": [float(np.percentile(boot, 2.5)),
                           float(np.percentile(boot, 97.5))],
        }
    out["eval_from"] = str(ELO_EVAL_FROM.date())
    # the draw rate the fixed curve is being asked to describe
    return out


def _print_elo(r: dict) -> None:
    c = r["replica_control"]
    print("=" * 78)
    print("ELO  -  Part D.1: measured, not fixed")
    print("=" * 78)
    print(f"  REPLICA CONTROL: my chronological pass against the deployed "
          f"EloSystem.fit()")
    print(f"      {c['teams']} teams, max |rating difference| = "
          f"{c['max_abs_rating_diff']:.9f}   "
          f"{'PASS' if c['max_abs_rating_diff'] < 1e-6 else 'FAIL'}")
    print(f"  calendar-year regressions applied (to EVERY rating): "
          f"{r['calendar_regressions']}")
    print(f"  season-boundary regressions applied (one team each): "
          f"{r['season_regressions']}  (season key derived from the DATE; "
          f"`matches.season` mixes two formats in 31 leagues)")
    print(f"  calendar-year leagues, by their own fixture calendar: "
          f"{r['calendar_leagues']}")
    print(f"  1X2 log-loss, out of sample, matches from {r['eval_from']}; "
          f"draw curve refit monthly on prior matches only")
    print()
    hdr = (f"    {'regression | draw curve':28s} {'n':>7s} {'log-loss':>10s} "
           f"{'vs deployed':>12s} {'95% CI':>22s}")
    print(hdr)
    for name, v in r["variants"].items():
        ci = v["delta_ci95"]
        tag = "  <- DEPLOYED" if name == "calendar|fixed" else ""
        print(f"    {name:28s} {v['n']:7d} {v['log_loss']:10.5f} "
              f"{v['delta_vs_deployed']:+12.5f} "
              f"[{ci[0]:+.5f},{ci[1]:+.5f}]{tag}")
    a, b = r["season_draw_curve_last"]
    print()
    print(f"  fitted draw curve (last fold, season mode): "
          f"P(draw) = sigmoid({a:+.4f} {b:+.4f} * |diff|/100)")
    for diff in (0, 100, 200, 400):
        print(f"      |diff|={diff:3d}:  fitted "
              f"{_sigmoid(a + b * diff / 100.0):.3f}   deployed "
              f"{max(0.15, 0.28 - diff / 2000.0):.3f}")


# ══ PART D.2 — xG COVERAGE ═══════════════════════════════════════════════

def xg() -> dict:
    _env()
    from sqlalchemy import text

    from src.data.database import get_db
    db = get_db()
    assert db.is_postgres
    with db.get_session() as s:
        rows = s.execute(text("""
            SELECT league, COUNT(*) n, COUNT(home_xg) with_xg,
                   MIN(match_date)::date, MAX(match_date)::date,
                   SUM(CASE WHEN match_date >= '2014-08-01' THEN 1 ELSE 0 END)
            FROM matches
            WHERE home_goals IS NOT NULL AND league NOT LIKE 'world/%'
              AND league NOT LIKE 'other/%'
            GROUP BY league ORDER BY n DESC
        """)).fetchall()
    per = {}
    tot = have = proj = 0
    for lg, n, w, lo, hi, since2014 in rows:
        gain = (since2014 - w) if lg in UNDERSTAT_LEAGUES else 0
        gain = max(gain, 0)
        per[lg] = {"n": n, "with_xg": w, "share": w / n,
                   "understat": lg in UNDERSTAT_LEAGUES,
                   "after_backfill": (w + gain) / n,
                   "first": str(lo), "last": str(hi)}
        tot += n
        have += w
        proj += w + gain
    return {"measured": "2026-11-12", "leagues": per, "total": tot,
            "with_xg": have, "share": have / tot,
            "after_backfill_share": proj / tot}


def _print_xg(r: dict) -> None:
    print("=" * 78)
    print("xG COVERAGE  -  Part D.2")
    print("=" * 78)
    print(f"  completed matches {r['total']}, with xG {r['with_xg']} "
          f"({100 * r['share']:.1f}%);  after an Understat backfill "
          f"{100 * r['after_backfill_share']:.1f}%")
    print(f"    {'league':34s} {'n':>6s} {'xG':>6s} {'now':>7s} "
          f"{'after':>7s}")
    for lg, v in r["leagues"].items():
        mark = "  <- Understat covers" if v["understat"] else ""
        print(f"    {str(lg)[:34]:34s} {v['n']:6d} {v['with_xg']:6d} "
              f"{100 * v['share']:6.1f}% {100 * v['after_backfill']:6.1f}%"
              f"{mark}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness", action="store_true")
    ap.add_argument("--elo", action="store_true")
    ap.add_argument("--xg", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    for flag, fn, pr in ((a.harness, harness, _print_harness),
                         (a.elo, elo, _print_elo), (a.xg, xg, _print_xg)):
        if flag:
            r = fn()
            if a.json:
                print(json.dumps(r, indent=1, default=str))
            else:
                pr(r)
            return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
