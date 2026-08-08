"""Stage 4 Phases 4-7 — rebuild every baseline on the CLEAN dataset.

    python -m scripts.run_clean_baseline [--min-books 2] [--window 60]

Nothing here reuses a pre-2026-08-07 result. Every number is recomputed on data
that passed `clean_dataset.build`, because the bookmaker-market corruption means
earlier bookmaker-derived figures are potentially contaminated.

Covers:
  Phase 4  baselines A-I on identical matches
  Phase 5  does consensus quality improve with more bookmakers?
  Phase 6  blend 0.80 / 0.90 / 1.00
  Phase 7  each market evaluated independently

Read-only against the database.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from scipy.stats import poisson as _pois

from src.evaluation.baseline import paired_bootstrap
from src.evaluation.clean_dataset import load_from_db
from src.utils.logger import get_logger

logger = get_logger()

MAX_GOALS = 10
SHRINK_CAP = 100
CUTOFFS = [date(2025, 8, 1), date(2025, 11, 1), date(2026, 2, 1),
           date(2026, 5, 1), date(2026, 7, 1)]


# --------------------------------------------------------------- model fitting

def fit_poisson(train, ref_date, half_life=540):
    dr = math.log(2) / half_life if half_life else 0.0
    w = np.array([math.exp(-dr * max(0, (ref_date - m.match_date).days))
                  for m in train]) if half_life else np.ones(len(train))
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


def poisson_matrix(state, m):
    hs, as_ = state["st"].get(m.home_team_id), state["st"].get(m.away_team_id)
    if hs is None or as_ is None:
        return None
    ah, aa = state["lg_avg"].get(m.league, (state["gh"], state["ga"]))
    lam, mu = ah * hs[0] * as_[1], aa * as_[0] * hs[1]
    hp = [_pois.pmf(i, lam) for i in range(MAX_GOALS)]
    ap = [_pois.pmf(i, mu) for i in range(MAX_GOALS)]
    return np.outer(hp, ap)


def poisson_1x2(state, m):
    mtx = poisson_matrix(state, m)
    if mtx is None:
        return None
    return [float(np.tril(mtx, -1).sum()), float(np.diag(mtx).sum()),
            float(np.triu(mtx, 1).sum())]


def poisson_over(state, m, line):
    mtx = poisson_matrix(state, m)
    if mtx is None:
        return None
    over = sum(mtx[i][j] for i in range(MAX_GOALS) for j in range(MAX_GOALS)
               if i + j > line)
    return [float(over), float(1 - over)]


def poisson_btts(state, m):
    mtx = poisson_matrix(state, m)
    if mtx is None:
        return None
    yes = float(mtx[1:, 1:].sum())
    return [yes, 1 - yes]


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
        g = k * math.log(max(abs(m.home_goals - m.away_goals), 1) + 1)
        R[m.home_team_id] = R.get(m.home_team_id, 1500) + g * (act - exp)
        R[m.away_team_id] = R.get(m.away_team_id, 1500) + g * ((1 - act) - (1 - exp))
    return R


def elo_1x2(R, m, ha=65):
    if m.home_team_id not in R or m.away_team_id not in R:
        return None
    he, ae = R[m.home_team_id] + ha, R[m.away_team_id]
    e = 1 / (1 + 10 ** ((ae - he) / 400))
    dp = max(0.15, 0.28 - abs(he - ae) / 2000)
    return [(1 - dp) * e, dp, (1 - dp) * (1 - e)]


def raw_market(m, label):
    """Baseline A: normalised 1/odds with the margin left in."""
    books = m.raw_prices.get(label)
    if not books:
        return None
    inv = np.mean([[1.0 / p for p in b] for b in books], axis=0)
    return list(inv / inv.sum())


def blend(a, b, w_b):
    if a is None or b is None:
        return None
    p = [(1 - w_b) * x + w_b * y for x, y in zip(a, b)]
    t = sum(p)
    return [x / t for x in p] if t > 0 else None


# ---------------------------------------------------------------------- scoring

def score(pairs):
    """pairs = [(probs, outcome_index)] -> metrics + per-match losses."""
    if not pairs:
        return None
    ll, br, acc, losses = 0.0, 0.0, 0, []
    cal_pairs = []
    for p, y in pairs:
        arr = np.clip(np.asarray(p, float), 1e-9, 1.0)
        arr = arr / arr.sum()
        loss = -math.log(arr[y])
        losses.append(loss)
        ll += loss
        t = np.zeros(len(arr)); t[y] = 1
        br += float(((arr - t) ** 2).sum())
        acc += int(int(np.argmax(arr)) == y)
        for k, pk in enumerate(arr):
            cal_pairs.append((float(pk), 1.0 if k == y else 0.0))
    n = len(pairs)
    # ECE over 10 equal-width bins
    ece, tot = 0.0, len(cal_pairs)
    for lo in np.linspace(0, 1, 11)[:-1]:
        hi = lo + 0.1
        chunk = [(p, h) for p, h in cal_pairs if lo <= p < hi]
        if chunk:
            mp = sum(p for p, _ in chunk) / len(chunk)
            mh = sum(h for _, h in chunk) / len(chunk)
            ece += (len(chunk) / tot) * abs(mp - mh)
    return {"log_loss": ll / n, "brier": br / n, "ece": ece,
            "accuracy": acc / n, "n": n, "_losses": losses}


def run_market(ds, label, outcome_fn, model_fns, window_days=60):
    """Walk-forward every candidate over one market; identical match set."""
    collected = defaultdict(list)   # name -> [(match_id, probs)]
    for cutoff in CUTOFFS:
        train = [m for m in ds if m.match_date < cutoff]
        test = [m for m in ds
                if cutoff <= m.match_date < cutoff + timedelta(days=window_days)
                and label in m.devigged]
        if len(train) < 300 or not test:
            continue
        states = {"poisson": fit_poisson(train, cutoff), "elo": fit_elo(train, cutoff)}
        for name, fn in model_fns.items():
            for m in test:
                p = fn(states, m)
                if p is not None:
                    collected[name].append((m.id, p))

    if not collected:
        return {}
    shared = set.intersection(*[{mid for mid, _ in v} for v in collected.values()])
    out_by_id = {m.id: outcome_fn(m) for m in ds}
    results = {}
    for name, items in collected.items():
        pairs = [(p, out_by_id[mid]) for mid, p in items
                 if mid in shared and out_by_id[mid] is not None]
        s = score(pairs)
        if s:
            results[name] = s
    return results


def show(results, title, baseline_key=None):
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)
    if not results:
        print("  (no data)")
        return
    print(f"{'candidate':<40}{'LogLoss':>10}{'Brier':>9}{'ECE':>8}{'Acc':>8}{'n':>7}")
    print("-" * 96)
    for name, m in sorted(results.items(), key=lambda kv: kv[1]["log_loss"]):
        print(f"{name:<40}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
              f"{m['ece']:>8.4f}{m['accuracy']:>8.1%}{m['n']:>7}")
    if baseline_key and baseline_key in results:
        print(f"\n  paired bootstrap vs {baseline_key!r} "
              f"(positive = baseline is better):")
        for name, m in results.items():
            if name == baseline_key:
                continue
            c = paired_bootstrap(m["_losses"], results[baseline_key]["_losses"])
            verdict = "SIGNIFICANT" if c["significant"] else "not significant"
            print(f"    {name:<38}{-c['mean_improvement']:+.4f} nats "
                  f"CI [{-c['ci_high']:+.4f}, {-c['ci_low']:+.4f}]  {verdict}")


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
    ap.add_argument("--min-books", type=int, default=2)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds, manifest = load_from_db(since=date(2022, 1, 1), min_books=args.min_books)
    print(manifest.render())

    # ---------------------------------------------------------------- Phase 4
    model_fns_1x2 = {
        "A raw bookmaker (vig left in)": lambda s, m: raw_market(m, "1X2"),
        "B de-vigged consensus": lambda s, m: m.consensus("1X2"),
        "C elo only": lambda s, m: elo_1x2(s["elo"], m),
        "D poisson only (540d, rho=0)": lambda s, m: poisson_1x2(s["poisson"], m),
        "F market 80% + elo 20%": lambda s, m: blend(
            elo_1x2(s["elo"], m), m.consensus("1X2"), 0.80),
        "G market 80% + poisson 20%": lambda s, m: blend(
            poisson_1x2(s["poisson"], m), m.consensus("1X2"), 0.80),
        "H market 80% + elo/poisson 20%": lambda s, m: blend(
            _avg(poisson_1x2(s["poisson"], m), elo_1x2(s["elo"], m)),
            m.consensus("1X2"), 0.80),
        "I production ensemble (blend 0.80)": lambda s, m: blend(
            _avg(poisson_1x2(s["poisson"], m), elo_1x2(s["elo"], m)),
            m.consensus("1X2"), 0.80),
    }
    r = run_market(ds, "1X2", lambda m: m.outcome_1x2, model_fns_1x2, args.window)
    show(r, "PHASE 4 — BASELINES A-I ON CLEAN DATA (1X2)",
         baseline_key="B de-vigged consensus")
    print("\n  Baseline E (current ML model): NOT EVALUATED. Running it needs "
          "create_features per match (~45 min for 2,000 matches) and the shipped "
          "pickles are from 2026-03-31, so most of this window is inside their "
          "training set. It is measured prospectively instead, by the log-loss "
          "path added to the weight learner in Stage 3.")

    # ---------------------------------------------------------------- Phase 5
    print("\n" + "=" * 96)
    print("PHASE 5 — DOES CONSENSUS QUALITY IMPROVE WITH MORE BOOKMAKERS?")
    print("=" * 96)
    print(f"{'min books':<12}{'matches':>9}{'LogLoss':>10}{'Brier':>9}{'Acc':>8}"
          f"{'coverage':>10}")
    total_matches = manifest.total_matches
    for mb in (1, 2, 3, 5, 8):
        sub, man2 = load_from_db(since=date(2022, 1, 1), min_books=mb)
        rr = run_market(sub, "1X2", lambda m: m.outcome_1x2,
                        {"B": lambda s, m: m.consensus("1X2")}, args.window)
        if "B" in rr:
            m2 = rr["B"]
            print(f"{mb:<12}{man2.per_market_matches.get('1X2', 0):>9}"
                  f"{m2['log_loss']:>10.4f}{m2['brier']:>9.4f}"
                  f"{m2['accuracy']:>8.1%}"
                  f"{man2.per_market_matches.get('1X2', 0)/total_matches:>10.1%}")

    # ---------------------------------------------------------------- Phase 6
    blend_fns = {f"blend {int(w*100)}% market": (
        lambda s, m, _w=w: blend(
            _avg(poisson_1x2(s["poisson"], m), elo_1x2(s["elo"], m)),
            m.consensus("1X2"), _w))
        for w in (0.80, 0.90, 1.00)}
    r6 = run_market(ds, "1X2", lambda m: m.outcome_1x2, blend_fns, args.window)
    show(r6, "PHASE 6 — BLEND WEIGHT ON CLEAN DATA (1X2)",
         baseline_key="blend 100% market")

    # ---------------------------------------------------------------- Phase 7
    for label, outcome_fn, model_key, poisson_fn in [
        ("over_under_2.5", lambda m: 0 if m.total_goals > 2.5 else 1,
         "over 2.5", lambda s, m: poisson_over(s["poisson"], m, 2.5)),
        ("over_under_1.5", lambda m: 0 if m.total_goals > 1.5 else 1,
         "over 1.5", lambda s, m: poisson_over(s["poisson"], m, 1.5)),
        ("btts", lambda m: 0 if (m.home_goals > 0 and m.away_goals > 0) else 1,
         "btts", lambda s, m: poisson_btts(s["poisson"], m)),
    ]:
        fns = {
            "A raw bookmaker (vig left in)": lambda s, m, _l=label: raw_market(m, _l),
            "B de-vigged consensus": lambda s, m, _l=label: m.consensus(_l),
            "D poisson only": poisson_fn,
            "blend 80% market": lambda s, m, _l=label, _f=poisson_fn: blend(
                _f(s, m), m.consensus(_l), 0.80),
            "blend 100% market": lambda s, m, _l=label: m.consensus(_l),
        }
        rm = run_market(ds, label, outcome_fn, fns, args.window)
        show(rm, f"PHASE 7 — {label} (independent evaluation)",
             baseline_key="B de-vigged consensus")

    if args.out:
        payload = {k: {kk: vv for kk, vv in v.items() if kk != "_losses"}
                   for k, v in r.items()}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"generated_at": datetime.utcnow().isoformat(),
             "manifest": manifest.render(), "phase4_1x2": payload}, indent=2))
        print(f"\nSnapshot written to {args.out}")


def _avg(a, b):
    if a is None or b is None:
        return None
    return [(x + y) / 2 for x, y in zip(a, b)]


if __name__ == "__main__":
    main()
