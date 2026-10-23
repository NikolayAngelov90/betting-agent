"""Every figure in `docs/decision-2026-11-01.md`, computed from production.

Measured 2026-10-23, ahead of the 2026-11-01 decision rather than on it, so
the paper does not need re-running on the day. The directive said "today,
2026-10-24"; today is the 23rd in UTC, in Kyiv local time and in the database
(`select now()`), so the figures carry the date they were actually taken. Figures that will move by then
are flagged MOVES in the output, with the expected direction.

THIS SCRIPT MEASURES. It builds no model, fits no threshold, changes no
config and writes nothing back. Every database access is a read.

TWO ROI CONVENTIONS ARE REPORTED, ALWAYS BOTH:

    kelly   (sum k*(o-1) over wins - sum k over losses) / sum k
    flat    (sum (o-1) over wins - count losses)        / count

`kelly` is the system's own convention (`betting_agent.get_stats`). IT IS A
WEIGHTED RATIO: a Kelly stake rises with the model's claimed edge, so a
Kelly-ROI pooled across price bands weights those bands by the model's own
opinion of them. That is the pooling, and naming it is the requirement.
`flat` is unweighted and is the one to compare across buckets.

VOID IS EXCLUDED FROM BOTH. A Draw-No-Bet push refunds the stake: zero
profit AND zero staked. Counting it as a full-stake loss understated ROI
once DNB shipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime

# THE REPO ROOT, SO THE INVOCATION FORM CANNOT MATTER. `python scripts/x.py`
# puts `scripts/` on sys.path[0]; only `python -m scripts.x` puts the root.
#
# FIFTH INSTANCE, hit while writing this file: `python scripts/
# decision_paper_figures.py` raised `ModuleNotFoundError: No module named src`
# from inside main(). `cohort_status.py:44` records the first four.
#
# AND THE ENTRY-POINT GUARD CANNOT CATCH IT HERE. That test runs each entry
# point with `--help`, which argparse answers before any work — so a script
# whose `src.` imports live inside main(), as these must, passes the guard in
# both forms and still fails in use. The bootstrap is the fix; the guard is
# only belt to its braces.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MEASURED_ON = "2026-10-23"
EXPERIMENT_START = date(2026, 8, 11)

#: `measured` 2026-08-07 by the stage-5 audit: the best-line overround a
#: strategy must beat before it makes money. Provenance: docs/ci-audit-ledger.
BEST_LINE_OVERROUND_PCT = 1.85

#: Price-band boundaries. DESCRIPTIVE — chosen to split the settled
#: population into three comparable parts, NOT fitted to outcome. They are
#: round numbers for that reason; a boundary tuned on ROI would be the
#: sixteenth fitted threshold.
BANDS = ((1.80, "short (<1.80)"), (2.50, "medium (1.80-2.50)"),
         (None, "long (>=2.50)"))


def _load_env():
    """INSIDE a call, never at import: a module-level load_dotenv() undoes
    conftest's DATABASE_URL strip and has let a test suite reach production."""
    for path in (".env", ".env.local"):
        try:
            for line in open(path, encoding="utf-8"):
                m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                if m and not os.environ.get(m.group(1)):
                    os.environ[m.group(1)] = \
                        m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue


# ── the row snapshot ─────────────────────────────────────────────────────

COLS = ("id", "match_id", "pick_date", "league", "market", "selection",
        "odds", "predicted_probability", "expected_value", "confidence",
        "kelly_stake_percentage", "disposition", "evidence_status",
        "review_action", "model_probability", "model_result", "model_version",
        "closing_odds", "closing_fair_probability", "closing_capture_status",
        "market_probability", "pre_claude_ev", "is_paper", "result",
        "settled_at", "created_at",
        # joined from matches, not a saved_picks column
        "kickoff_at")


class Row:
    """A plain snapshot of the columns these figures use.

    DETACHED-INSTANCE RULE: an ORM row read after its session closes returns
    zeros silently rather than raising. Every attribute is pulled inside the
    `with` block, which is what this class is for.
    """
    __slots__ = COLS

    def __init__(self, r):
        for k in COLS:
            setattr(self, k, getattr(r, k, None))


def fetch_all(db):
    """Every pick, with `kickoff_at` joined from matches.

    Both reads happen INSIDE one session, and every attribute is copied out
    before it closes — a detached ORM row returns zeros silently.
    """
    from src.data.models import Match, SavedPick
    with db.get_session() as s:
        kickoff = {m.id: m.match_date for m in s.query(Match.id,
                                                       Match.match_date)}
        rows = []
        for r in s.query(SavedPick).all():
            row = Row(r)
            row.kickoff_at = kickoff.get(r.match_id)
            rows.append(row)
        return rows


# ── the two ratios, the void rule, and win rate ──────────────────────────

def decided(picks):
    return [p for p in picks if p.result in ("win", "loss")]


def roi_pair(picks):
    """`dict(kelly, flat, n, wins, win_rate, voids, avg_odds)`.

    `None` where undefined rather than 0.0 — an undefined ratio is not a
    zero one, and this project has shipped three confident wrong answers
    from that substitution.
    """
    d = decided(picks)
    voids = sum(1 for p in picks if p.result == "void")
    if not d:
        return {"kelly": None, "flat": None, "n": 0, "wins": 0,
                "win_rate": None, "voids": voids, "avg_odds": None,
                "note": "UNEVALUATED: no decided pick in this population"}
    ks = sum(p.kelly_stake_percentage or 0.0 for p in d)
    kp = sum((p.kelly_stake_percentage or 0.0) * (p.odds - 1)
             if p.result == "win" else -(p.kelly_stake_percentage or 0.0)
             for p in d)
    fp = sum((p.odds - 1) if p.result == "win" else -1.0 for p in d)
    wins = sum(1 for p in d if p.result == "win")
    return {"kelly": (kp / ks) if ks > 0 else None,
            "flat": fp / len(d),
            "n": len(d), "wins": wins, "win_rate": wins / len(d),
            "voids": voids,
            "avg_odds": sum(p.odds for p in d) / len(d)}


def break_even_win_rate(odds):
    """The flat-stake break-even win rate at `odds`: 1/odds."""
    return (1.0 / odds) if odds else None


def price_band(odds):
    if odds is None:
        return "UNEVALUATED"
    for bound, label in BANDS:
        if bound is None or odds < bound:
            return label
    return BANDS[-1][1]


# ── bootstrap and effective n, resampling FIXTURES ───────────────────────

def ci_flat(picks, iters=4000, seed=0):
    """95% CI for flat ROI, resampling FIXTURES with replacement.

    Reuses `src.evaluation.clv._boot`, which already resamples clusters and
    whose docstring records the 31.8% co-fixture rate it was built for. Two
    picks on one match are not independent observations.
    """
    from src.evaluation.clv import _boot
    d = decided(picks)
    if len(d) < 5:
        return None, None
    vals = [(p.odds - 1) if p.result == "win" else -1.0 for p in d]
    return _boot(vals, clusters=[p.match_id for p in d],
                 iters=iters, seed=seed)


def ci_clv(picks, iters=4000, seed=0):
    """95% CI for mean CLV, same clustering. `None` where coverage is short."""
    from src.evaluation.clv import _boot
    vals, cl = [], []
    for p in picks:
        v = clv_of(p)
        if v is not None:
            vals.append(v)
            cl.append(p.match_id)
    if len(vals) < 5:
        return None, None
    return _boot(vals, clusters=cl, iters=iters, seed=seed)


def effective_n(picks):
    """`(n_picks, n_fixtures, design_effect, effective_n)`, worst case rho=1."""
    from src.evaluation.clv import _effective_n
    return _effective_n([p.match_id for p in decided(picks)])


def clv_of(p):
    """Closing-line value for one pick, or None.

    `(taken / closing) - 1` in price terms: positive means the price taken
    was better than the close. Requires a REAL captured close — the `late`
    and `missing` statuses are UNEVALUATED, not zero.
    """
    if p.closing_capture_status != "captured":
        return None
    if not p.closing_odds or not p.odds:
        return None
    return (p.odds / p.closing_odds) - 1.0


# ── minimum detectable effect ────────────────────────────────────────────

def mde_flat(picks, power=0.80, alpha=0.05):
    """Smallest true flat-ROI shift detectable at `power`, one-sided.

    Uses the observed per-pick dispersion and the EFFECTIVE n after fixture
    clustering, at the worst-case design effect (rho = 1). One-sided because
    the question is "could a real edge be hiding", not "is this two-sided
    different from zero".
    """
    import numpy as np
    d = decided(picks)
    if len(d) < 5:
        return None
    vals = np.array([(p.odds - 1) if p.result == "win" else -1.0 for p in d],
                    dtype=float)
    _, _, _, eff_n = effective_n(picks)
    if eff_n < 2:
        return None
    z = 1.6449 + 0.8416                    # z(1-alpha) + z(power), one-sided
    return float(z * vals.std(ddof=1) / (eff_n ** 0.5))


def n_for_mde(picks, target, power=0.80, alpha=0.05):
    """Effective n needed to detect `target` (as a fraction), or None."""
    import numpy as np
    d = decided(picks)
    if len(d) < 5 or not target:
        return None
    vals = np.array([(p.odds - 1) if p.result == "win" else -1.0 for p in d],
                    dtype=float)
    z = 1.6449 + 0.8416
    return float((z * vals.std(ddof=1) / target) ** 2)


# ── the splits for PART D ────────────────────────────────────────────────

def quantile_buckets(picks, key, k=4, labels=None):
    """`{label: [picks]}` by quantile of `key`, with Nones kept apart.

    An absent value is its own bucket (`UNEVALUATED`), never folded into the
    lowest quantile — "could not look" is not "found nothing".
    """
    import numpy as np
    vals = [(key(p), p) for p in picks]
    have = [(v, p) for v, p in vals if v is not None]
    out = {}
    missing = [p for v, p in vals if v is None]
    if missing:
        out["UNEVALUATED (no value)"] = missing
    if len(have) < k * 5:
        out["UNEVALUATED (too few to split)"] = [p for _, p in have]
        return out
    xs = np.array([v for v, _ in have], dtype=float)
    cuts = np.quantile(xs, [i / k for i in range(1, k)])
    names = labels or [f"Q{i + 1}" for i in range(k)]
    for i, nm in enumerate(names):
        lo = -np.inf if i == 0 else cuts[i - 1]
        hi = np.inf if i == k - 1 else cuts[i]
        sel = [p for v, p in have if (v >= lo if i == 0 else v > lo) and v <= hi]
        out[f"{nm} ({lo:+.4f}..{hi:+.4f})" if np.isfinite(lo) and np.isfinite(hi)
            else f"{nm}"] = sel
    return out


def by_attr(picks, key):
    out = defaultdict(list)
    for p in picks:
        out[key(p)].append(p)
    return dict(out)


def lead_time_hours(p):
    """Hours from pick creation to REAL KICKOFF (`Match.match_date`).

    THE FIRST VERSION OF THIS WAS WRONG and is worth recording: it measured
    `pick_date` (a DATE, so midnight) minus `created_at`, which came out at
    a flat -8 to -10 hours for every pick. That is not lead time at all, it
    is the clock time of the daily run — `created_at` is ~10:52 and midnight
    is 10.9 hours earlier. A quantile split on it would have split the
    population by nothing and the four buckets' near-identical ROI would
    have read as "lead time does not matter".

    `kickoff_at` is attached by `fetch_all` from the matches table.
    UNEVALUATED where either timestamp is absent.
    """
    if not p.created_at or not getattr(p, "kickoff_at", None):
        return None
    return (p.kickoff_at - p.created_at).total_seconds() / 3600.0


def run_hour(p):
    """The hour of day the pick was written — what the broken lead time was
    actually measuring. Kept as its own split because it is a real question
    (does the run's clock time matter?), just not the one asked for."""
    return p.created_at.hour + p.created_at.minute / 60.0 \
        if p.created_at else None


def summarise(picks, want_clv=True, iters=2000):
    """One bucket's row: both ROIs, win rate, CI, effective n, CLV."""
    r = roi_pair(picks)
    lo, hi = ci_flat(picks, iters=iters)
    n, k, deff, eff = effective_n(picks)
    r.update({"flat_ci": [lo, hi], "n_picks": n, "n_fixtures": k,
              "design_effect": deff, "effective_n": eff})
    if want_clv:
        cv = [clv_of(p) for p in picks]
        cv = [v for v in cv if v is not None]
        r["clv_n"] = len(cv)
        r["clv_mean"] = (sum(cv) / len(cv)) if cv else None
        clo, chi = ci_clv(picks, iters=iters)
        r["clv_ci"] = [clo, chi]
        if not cv:
            r["clv_note"] = ("UNEVALUATED: no captured close in this bucket "
                             "-- not a zero CLV")
    if r.get("avg_odds"):
        r["break_even_win_rate"] = break_even_win_rate(r["avg_odds"])
    return r


def _span(picks):
    ds = [str(p.pick_date)[:10] for p in picks if p.pick_date]
    return [min(ds), max(ds)] if ds else None


def clv_separation(rows):
    """The same overlap question for CLV, and UNEVALUATED where there is no
    captured close — a bucket with `clv_n = 0` has not been measured, which
    is not the same as a bucket measured at zero."""
    usable = [(k, v) for k, v in rows.items()
              if v.get("clv_ci") and v["clv_ci"][0] is not None]
    # `clv_n` ABSENT and `clv_n == 0` ARE DIFFERENT STATES and the falsy-zero
    # detector was right to refuse `if not v.get("clv_n")`. Absent means the
    # bucket was summarised with `want_clv=False` -- CLV was never looked at.
    # Zero means it was looked at and there was no captured close. Both end
    # up in `unmeasured` here, but they are counted apart so a caller cannot
    # read "0 buckets measured" as "CLV is zero".
    not_looked = [k for k, v in rows.items() if "clv_n" not in v]
    looked_found_none = [k for k, v in rows.items() if v.get("clv_n") == 0]
    unmeasured = not_looked + looked_found_none
    if not usable:
        return {"verdict": "UNEVALUATED: no bucket carries enough captured "
                           "closes to form an interval",
                "buckets_unmeasured": len(unmeasured),
                "buckets_clv_never_looked_at": len(not_looked),
                "buckets_clv_looked_found_none": len(looked_found_none)}
    disjoint = []
    for i, (ka, a) in enumerate(usable):
        for kb, b in usable[i + 1:]:
            if a["clv_ci"][1] < b["clv_ci"][0] or \
                    b["clv_ci"][1] < a["clv_ci"][0]:
                disjoint.append((ka, kb))
    return {"buckets_with_intervals": len(usable),
            "buckets_unmeasured": len(unmeasured),
            "buckets_clv_never_looked_at": len(not_looked),
            "buckets_clv_looked_found_none": len(looked_found_none),
            "disjoint_pairs": disjoint,
            "verdict": ("SOME PAIRS SEPARATE" if disjoint
                        else "every CLV interval overlaps every other")}


def spread_exceeds_intervals(rows):
    """Do any two buckets' flat-ROI 95% intervals fail to overlap?

    The only honest summary of a descriptive table: if every interval
    overlaps every other, the spread is inside what the intervals allow and
    the table ranks nothing.
    """
    usable = [(k, v) for k, v in rows.items()
              if v.get("flat_ci") and v["flat_ci"][0] is not None]
    disjoint = []
    for i, (ka, a) in enumerate(usable):
        for kb, b in usable[i + 1:]:
            if a["flat_ci"][1] < b["flat_ci"][0] or \
                    b["flat_ci"][1] < a["flat_ci"][0]:
                disjoint.append((ka, kb))
    return {"buckets_with_intervals": len(usable),
            "disjoint_pairs": disjoint,
            "verdict": ("SOME PAIRS SEPARATE" if disjoint
                        else "every interval overlaps every other -- the "
                             "spread is inside what the intervals allow")}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Figures for the 2026-11-01 decision paper. Read-only.")
    ap.add_argument("--json", help="write every figure as JSON to this path")
    ap.add_argument("--iters", type=int, default=2000,
                    help="bootstrap resamples per bucket (default 2000)")
    args = ap.parse_args(argv)

    _load_env()
    from src.data.database import DatabaseManager
    db = DatabaseManager()
    assert db.is_postgres, (
        "NOT POSTGRES. Every figure in the decision paper requires the "
        "production database; an empty SQLite file answers each query with a "
        "plausible zero, and that substitution has produced three confident "
        "wrong answers in this project.")

    allp = fetch_all(db)
    out = {"measured_on": MEASURED_ON, "is_postgres": True,
           "n_saved_picks": len(allp)}

    # ── PART A ───────────────────────────────────────────────────────────
    settled = [p for p in allp if p.result is not None]
    A = {"saved_picks": len(allp), "settled": len(settled),
         "unsettled": len(allp) - len(settled)}

    A["gates"] = {}
    for label, pred in (
            ("is_paper is true", lambda p: p.is_paper is True),
            ("disposition set (never placed)",
             lambda p: p.disposition is not None),
            ("evidence_status set (model never saw it)",
             lambda p: p.evidence_status is not None)):
        hit = [p for p in settled if pred(p)]
        A["gates"][label] = {"removes": len(hit),
                             "leaves": len(settled) - len(hit)}

    live = [p for p in settled
            if not p.is_paper and p.disposition is None]
    evid = [p for p in live if p.evidence_status is None]
    A["live_only"] = summarise(live)
    A["live_and_valid_evidence"] = summarise(evid)
    A["paper"] = summarise([p for p in settled if p.is_paper])
    A["all_settled"] = summarise(settled)

    # the collinearity claim, CHECKED rather than asserted
    tab = Counter((bool(p.is_paper), p.model_version or "(null)")
                  for p in settled)
    by_mv = defaultdict(set)
    for (paper, mv), _ in tab.items():
        by_mv[mv].add(paper)
    A["paper_x_cohort"] = {
        "table": {f"is_paper={k[0]} / {k[1]}": v for k, v in sorted(tab.items())},
        "cohorts_with_both_values": [mv for mv, s in by_mv.items() if len(s) > 1],
        "perfectly_collinear": all(len(s) == 1 for s in by_mv.values()),
    }

    A["cohorts"] = {}
    for mv, ps in sorted(by_attr(settled, lambda p: p.model_version or "(null)")
                         .items()):
        dates = [p.pick_date for p in ps if p.pick_date]
        s = summarise(ps)
        s["covers"] = [str(min(dates))[:10], str(max(dates))[:10]] if dates \
            else None
        s["is_paper_values"] = sorted({bool(p.is_paper) for p in ps})
        A["cohorts"][mv] = s

    exp = [p for p in settled
           if p.pick_date and _as_date(p.pick_date) >= EXPERIMENT_START]
    A["experiment_window"] = {
        "start": str(EXPERIMENT_START), "n_settled": len(exp),
        "all": summarise(exp),
        "live_only": summarise([p for p in exp
                                if not p.is_paper and p.disposition is None]),
    }

    cap = Counter(p.closing_capture_status or "(null)" for p in allp)
    era = [p for p in allp
           if p.pick_date and _as_date(p.pick_date) >= date(2026, 8, 14)]
    usable = [p for p in settled if clv_of(p) is not None]
    A["clv"] = {
        "capture_status_all_picks": dict(cap),
        "capture_era_start": "2026-08-14 (first captured close)",
        "capture_era_n": len(era),
        "captured_in_era": sum(1 for p in era
                               if p.closing_capture_status == "captured"),
        "coverage_of_era_pct": (100.0 * sum(
            1 for p in era if p.closing_capture_status == "captured")
            / len(era)) if era else None,
        "settled_with_usable_close": len(usable),
        "FINAL_series": summarise(usable),
    }
    # the MODEL series: the pick as the model made it, before the review
    mdl = [p for p in usable if p.model_result in ("win", "loss")]
    A["clv"]["MODEL_series_n"] = len(mdl)
    out["A"] = A

    # ── PART C ───────────────────────────────────────────────────────────
    C = {"overround_best_line_pct": BEST_LINE_OVERROUND_PCT,
         "convention": "flat compared across bands; kelly shown and named "
                       "as weighted by the model's own claimed edge"}
    C["by_price_band"] = {k: summarise(v, iters=args.iters)
                          for k, v in sorted(by_attr(settled, lambda p:
                                                     price_band(p.odds)).items())}
    C["band_spread"] = spread_exceeds_intervals(C["by_price_band"])

    # "always take the shortest price available" -- one pick per fixture, the
    # shortest. DESCRIPTIVE: it is not a strategy the system could have run,
    # because the shortest price on a fixture is only known after the fact.
    shortest = {}
    for p in decided(settled):
        cur = shortest.get(p.match_id)
        if cur is None or p.odds < cur.odds:
            shortest[p.match_id] = p
    C["shortest_price_per_fixture"] = summarise(list(shortest.values()),
                                               iters=args.iters)
    C["shortest_price_note"] = (
        "DESCRIPTIVE ONLY. Picking the shortest price on each fixture is not "
        "a strategy that could have been run -- it selects among picks the "
        "system actually made, and it is shown to demonstrate the win-rate / "
        "ROI relationship, not to propose anything.")
    out["C"] = C

    # ── PART D — descriptive, with intervals, NOT a selection rule ───────
    D = {"NOT_A_SELECTION_RULE": (
        "Descriptive. No threshold is proposed and no number here may "
        "justify a change in this stage -- this is exactly where fifteen "
        "fitted thresholds came from."),
        "bootstrap_unit": "FIXTURE (match_id), 95%, "
                          f"{args.iters} resamples, seed 0"}

    # TWO POPULATIONS, BOTH REPORTED, because they are DISJOINT IN TIME and
    # neither answers the other's question:
    #
    #   live + valid_evidence  the money record. Ends 2026-08-10. Carries NO
    #                          model_version stamp and NO captured close, so
    #                          every CLV cell in it is UNEVALUATED.
    #   paper + valid_evidence the experiment. Begins 2026-08-11. Every
    #                          captured close lives here.
    #
    # `is_paper` is therefore collinear with the ERA, not merely with
    # `model_version`. A split computed on one and read as if it described
    # the other would be the pooling this paper is required to name.
    paper_evid = [p for p in settled
                  if p.is_paper and p.disposition is None
                  and p.evidence_status is None]
    D["populations"] = {
        "live_and_valid_evidence": {"n": len(evid),
                                    "covers": _span(evid),
                                    "captured_closes": sum(
                                        1 for p in evid
                                        if clv_of(p) is not None)},
        "paper_and_valid_evidence": {"n": len(paper_evid),
                                     "covers": _span(paper_evid),
                                     "captured_closes": sum(
                                         1 for p in paper_evid
                                         if clv_of(p) is not None)},
    }
    D["disjoint_in_time"] = (
        "the live record ends 2026-08-10 and the paper record begins "
        "2026-08-11; they share no day, so is_paper is collinear with the "
        "era and the two tables are not two views of one population")

    def split_set(pop):
        return {
            "model_edge_quantiles": quantile_buckets(
                pop, lambda p: p.expected_value, 4),
            "confidence_quantiles": quantile_buckets(
                pop, lambda p: p.confidence, 4),
            "market": by_attr(pop, lambda p: p.market or "(null)"),
            "league": by_attr(pop, lambda p: p.league or "(null)"),
            "price_band": by_attr(pop, lambda p: price_band(p.odds)),
            "lead_time_before_kickoff_quantiles": quantile_buckets(
                pop, lead_time_hours, 4),
            "run_hour_quantiles": quantile_buckets(pop, run_hour, 4),
            "review_changed_the_pick": by_attr(
                pop, lambda p: p.review_action or "(null)"),
        }

    D["splits"] = {}
    for pop_name, pop in (("live_and_valid_evidence", evid),
                          ("paper_and_valid_evidence", paper_evid)):
        blk = {}
        for name, buckets in split_set(pop).items():
            rows = {k: summarise(v, iters=args.iters)
                    for k, v in sorted(buckets.items(),
                                       key=lambda kv: str(kv[0]))}
            blk[name] = {"buckets": rows,
                         "separation": spread_exceeds_intervals(rows),
                         "clv_separation": clv_separation(rows)}
        D["splits"][pop_name] = blk

    # HOW MANY COMPARISONS THIS TABLE MAKES, because "SOME PAIRS SEPARATE" is
    # meaningless without it. Each bucket pair is one 95% interval
    # comparison; across every split and both populations the family is
    # large, so one separating pair is what chance produces.
    pairs = 0
    separating = 0
    for pop_blk in D["splits"].values():
        for blk in pop_blk.values():
            k = blk["separation"]["buckets_with_intervals"]
            pairs += k * (k - 1) // 2
            separating += len(blk["separation"]["disjoint_pairs"])
    D["multiplicity"] = {
        "roi_pair_comparisons": pairs,
        "roi_pairs_separating": separating,
        "expected_by_chance_at_95pct": 0.05 * pairs,
        "note": ("every interval is 95% and marginal. With this many "
                 "comparisons, separating pairs at about 5% of the total are "
                 "what chance produces; no correction is applied and none of "
                 "these is treated as a finding."),
    }
    out["D"] = D

    # ── PART E ───────────────────────────────────────────────────────────
    E = {}
    for label, ps in (("all_settled", settled), ("live_only", live),
                      ("live_and_valid_evidence", evid),
                      ("paper_and_valid_evidence", paper_evid),
                      ("captured_close_subset", usable),
                      ("experiment_window", exp)):
        n, k, deff, eff = effective_n(ps)
        m = mde_flat(ps)
        E[label] = {
            "n_picks": n, "n_fixtures": k, "design_effect": deff,
            "effective_n": eff,
            "mde_flat_roi_80pct_power": m,
            "mde_vs_overround": (
                f"{100 * m:+.2f}% detectable vs {BEST_LINE_OVERROUND_PCT:+.2f}% "
                f"break-even overround" if m else None),
            "n_eff_needed_to_detect_overround":
                n_for_mde(ps, BEST_LINE_OVERROUND_PCT / 100.0),
            "n_eff_needed_to_halve_current_mde":
                n_for_mde(ps, m / 2.0) if m else None,
        }
    # HOW LONG WAITING WOULD TAKE, at the observed pick rate. The rate is
    # measured on the paper era, which is the only era still producing.
    pdays = {str(p.pick_date)[:10] for p in settled
             if p.is_paper and p.pick_date}
    rate = len([p for p in settled if p.is_paper]) / max(len(pdays), 1)
    E["pick_rate"] = {"paper_picks_per_active_day": rate,
                      "active_days": len(pdays),
                      "note": "picks per ACTIVE day, not per calendar day"}
    for label in ("paper_and_valid_evidence", "live_and_valid_evidence"):
        v = E[label]
        cur = v["effective_n"]
        for tgt_name, need in (
                ("to_detect_overround", v["n_eff_needed_to_detect_overround"]),
                ("to_halve_current_mde", v["n_eff_needed_to_halve_current_mde"])):
            if need and rate > 0:
                extra = max(need - cur, 0.0)
                v[f"days_{tgt_name}"] = extra / rate
                v[f"months_{tgt_name}"] = extra / rate / 30.44
    E["clv_mde_note"] = (
        "CLV MDE is computed on the captured subset only; see A.clv for its "
        "n. A CLV figure over the `late` and `missing` rows would be "
        "UNEVALUATED, not zero.")
    out["E"] = E

    # THE JSON IS WRITTEN BEFORE ANYTHING IS PRINTED, and the order is the
    # lesson: a `KeyError` in `_print` once discarded a fifteen-minute
    # computation that had already finished. The expensive artefact is
    # persisted first; presentation cannot cost it.
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
        print(f"wrote {args.json}\n")
    _print(out)
    return 0


def _as_date(v):
    return v.date() if isinstance(v, datetime) else v


def _pct(v, nd=2):
    return "UNEVALUATED" if v is None else f"{100 * v:+.{nd}f}%"


def _print(out):
    """Plain ASCII. A detector that dies on its own output is no detector --
    an emoji here crashed `print` on a cp1251 console once already."""
    A = out["A"]
    print(f"MEASURED {out['measured_on']}   is_postgres=True   "
          f"saved_picks={out['n_saved_picks']}")
    print(f"\n== PART A ==  settled={A['settled']} unsettled={A['unsettled']}")
    for k, v in A["gates"].items():
        print(f"   gate {k:42s} removes {v['removes']:5d} -> {v['leaves']}")
    for label in ("all_settled", "live_only", "live_and_valid_evidence",
                  "paper"):
        r = A[label]
        print(f"   {label:26s} n={r['n']:5d} win={_pct(r['win_rate'],1):>8s} "
              f"flat={_pct(r['flat']):>9s} kelly={_pct(r['kelly']):>9s} "
              f"fixtures={r['n_fixtures']}")
    print(f"   paper x cohort perfectly collinear: "
          f"{A['paper_x_cohort']['perfectly_collinear']}")
    print("   cohorts:")
    for mv, r in A["cohorts"].items():
        print(f"     {mv:38s} n={r['n']:5d} win={_pct(r['win_rate'],1):>8s} "
              f"flat={_pct(r['flat']):>9s} covers={r['covers']}")
    c = A["clv"]
    print(f"   CLV capture: {c['capture_status_all_picks']}")
    print(f"   CLV coverage of era: {c['captured_in_era']}/{c['capture_era_n']}"
          f" = {c['coverage_of_era_pct']:.1f}%   usable settled="
          f"{c['settled_with_usable_close']}")

    print("\n== PART C ==")
    for k, r in out["C"]["by_price_band"].items():
        be = r.get("break_even_win_rate")
        print(f"   {k:22s} n={r['n']:5d} win={_pct(r['win_rate'],1):>8s} "
              f"breakeven={_pct(be,1) if be else 'n/a':>8s} "
              f"flat={_pct(r['flat']):>9s} "
              f"CI=[{_pct(r['flat_ci'][0])},{_pct(r['flat_ci'][1])}]")
    print(f"   band separation: {out['C']['band_spread']['verdict']}")
    s = out["C"]["shortest_price_per_fixture"]
    print(f"   shortest-price-per-fixture: n={s['n']} "
          f"win={_pct(s['win_rate'],1)} flat={_pct(s['flat'])} "
          f"avg_odds={s['avg_odds']:.2f}")

    print("\n== PART D ==")
    for pn, meta in out["D"]["populations"].items():
        print(f"   population {pn:26s} n={meta['n']:5d} covers={meta['covers']}"
              f" captured_closes={meta['captured_closes']}")
    print(f"   {out['D']['disjoint_in_time']}")
    for pop_name, pop_blk in out["D"]["splits"].items():
        print(f"\n   ######## {pop_name}")
        for name, blk in pop_blk.items():
            print(f"   -- {name}")
            for k, r in blk["buckets"].items():
                ci = r["flat_ci"]
                cm = r.get("clv_mean")
                print(f"      {str(k)[:32]:32s} n={r['n']:4d} "
                      f"win={_pct(r['win_rate'],1):>8s} "
                      f"flat={_pct(r['flat']):>9s} "
                      f"CI=[{_pct(ci[0]):>9s},{_pct(ci[1]):>9s}] "
                      f"clv_n={r.get('clv_n', 0):3d} "
                      f"clv={_pct(cm) if cm is not None else 'UNEVAL':>9s}")
            print(f"      => ROI: {blk['separation']['verdict']}")
            print(f"      => CLV: {blk['clv_separation']['verdict']}")

    print(f"\n   multiplicity: {out['D']['multiplicity']['roi_pairs_separating']}"
          f" of {out['D']['multiplicity']['roi_pair_comparisons']} ROI pair "
          f"comparisons separate; "
          f"{out['D']['multiplicity']['expected_by_chance_at_95pct']:.1f} "
          f"expected by chance at 95%")

    print("\n== PART E ==")
    # KEYED ON THE FIELD, NOT ON `isinstance(dict)`. The first version asked
    # "is it a dict", which `pick_rate` also answers yes to, and the printer
    # raised KeyError after every figure had been computed.
    for k, v in out["E"].items():
        if not (isinstance(v, dict) and "n_picks" in v):
            continue
        print(f"   {k:26s} n={v['n_picks']:5d} fixtures={v['n_fixtures']:5d} "
              f"deff={v['design_effect']:.2f} n_eff={v['effective_n']:.0f}")
        print(f"      MDE(80% power, one-sided) = "
              f"{_pct(v['mde_flat_roi_80pct_power'])}   {v['mde_vs_overround']}")
        # `is not None`, not truthiness: a needed-n that came back 0 is a
        # measurement, and skipping it would print nothing while implying
        # the figure was absent.
        if v.get("n_eff_needed_to_detect_overround") is not None:
            print(f"      n_eff to detect the overround: "
                  f"{v['n_eff_needed_to_detect_overround']:.0f}")
        for tgt in ("to_detect_overround", "to_halve_current_mde"):
            if v.get(f"months_{tgt}") is not None:
                print(f"      {tgt:22s} +{v[f'days_{tgt}']:.0f} days "
                      f"= {v[f'months_{tgt}']:.1f} months at the current rate")
    pr = out["E"].get("pick_rate")
    if pr:
        print(f"   pick rate: {pr['paper_picks_per_active_day']:.1f} "
              f"picks/active day over {pr['active_days']} active days")


if __name__ == "__main__":
    sys.exit(main())
