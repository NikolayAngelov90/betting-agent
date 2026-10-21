"""The report's headline figures, each carrying the population it ranges over.

    2026-10-20. "Audit the Weekly Report Against This Project's Own Rules."

WHY THIS EXISTS. `scripts/paper_trading_report.py` is the one output a person
reads, and it was the only consumer of the settled record that applied **none**
of the gates the rest of the project applies. `_live_only()` and
`_valid_evidence()` have eleven call sites in `betting_agent.py` and had zero
here. Measured 2026-10-20 over `--days 120 --include-live`, the invocation CI
actually uses, its `OUTCOMES` block pooled:

* **15 `model_version` cohorts** into one ROI, whose per-cohort ROIs run from
  **-18.3% to +31.0%**;
* `disposition='consolidated'` picks (4) that are excluded from the live record
  everywhere else;
* `evidence_status='void_corrupt_features'` picks (11) whose model inputs
  described a different club;
* and **paper with live** (1,240 + 294).

and printed a model Brier over **n=1,534** beside a market Brier over
**n=479**, with the label "<- the number the model must beat". Those are
different populations. Paired on the 479 they share: model **0.2434** vs market
**0.2433**, delta **+0.0001**, t=**+0.07**. The entire apparent gap was the
unpaired populations, not the model.

THE STRUCTURAL FACT THAT MAKES `--include-live` MISLEADING. `is_paper` and
`model_version` are **perfectly collinear** -- measured 2026-10-20, both
off-diagonal cells are zero:

    is_paper=live  x UNVERSIONED : 1074   2026-02-28 .. 2026-08-10
    is_paper=paper x versioned   : 1316   2026-08-11 .. 2026-10-20

So `--include-live` does not add live trading results. **There has never been a
live wager.** It adds the pre-versioning era, which is exactly the block the
standing request named.
"""

from __future__ import annotations

import math
from collections import Counter
from datetime import date
from typing import Dict, List, NamedTuple, Optional, Sequence

#: WHEN THE EXPERIMENT STARTED, resolved to a literal. Four readings were
#: available and they do not agree, so the choice is stated rather than assumed:
#:
#:   1. the first pick ever                        2026-02-28  n=2314
#:   2. the first SETTLED pick                     2026-02-28  n=2314
#:   3. the first pick carrying a `model_version`  2026-08-11  n=1240
#:   4. the first cohort that still has picks      2026-08-11  n=1240
#:
#: READINGS 3 AND 4 COINCIDE, and that is the one taken. The justification is
#: not a preference: the standing request gave its own reference point as
#: "1,074 picks / 52%", and the block this date excludes is **exactly** that --
#: measured 2026-10-20, the unversioned block is n=1074, win 51.7%. The request
#: identified the population to drop by quoting its figure, so the date that
#: drops precisely that population is what it meant.
#:
#: Readings 1 and 2 are the same date here, which is itself a fact worth
#: keeping: the first pick settled, so "first pick" and "first settled pick"
#: cannot be told apart in this data and neither can carry the distinction.
EXPERIMENT_START = date(2026, 8, 11)

EXPERIMENT_START_WHY = (
    "the first pick carrying a model_version (2026-08-11); readings 3 and 4 of "
    "four coincide, and the block it excludes is exactly the 1,074 picks / "
    "51.7% the standing request named as its reference point"
)

#: THE GATES, each with the reason it exists. A dropped pick is COUNTED and
#: PRINTED, never silently filtered: "nothing was dropped" and "the gate never
#: ran" have to be different observations.
GATES = (
    ("disposition",
     "left the live record (e.g. superseded, consolidated) -- gated by "
     "_live_only() at eleven call sites in betting_agent.py, and at none here"),
    ("evidence_status",
     "not valid evidence about the model (e.g. void_corrupt_features: a real "
     "wager whose model inputs described a different club) -- gated by "
     "_valid_evidence()"),
    ("training_exclusion_reason",
     "the MATCH is excluded from training as corrupt; a pick on it is not "
     "evidence about the model either"),
)


class Row(NamedTuple):
    """One settled pick, flat. Plain tuple so a test can build known input."""

    pick_id: int
    match_id: int
    pick_date: date
    result: str                      # "win" | "loss" | None | "void"
    is_paper: bool
    model_version: Optional[str]
    disposition: Optional[str]
    evidence_status: Optional[str]
    training_exclusion_reason: Optional[str]
    odds: float
    prob: Optional[float]            # the model's probability
    market_prob: Optional[float]     # the de-vigged bookmaker probability


def profit(r: Row) -> float:
    """Flat one-unit stake. Matches `_Pick.profit` in the report."""
    return (r.odds - 1.0) if r.result == "win" else -1.0


def decided(r: Row) -> bool:
    return r.result in ("win", "loss")


def partition(rows: Sequence[Row]) -> Dict:
    """Apply the gates, and report what each one removed.

    Returns `{"kept": [...], "dropped": {gate: n}, "dropped_detail": {...}}`.
    """
    kept: List[Row] = []
    dropped: Counter = Counter()
    detail: Dict[str, Counter] = {g: Counter() for g, _ in GATES}
    for r in rows:
        why = None
        for g, _reason in GATES:
            v = getattr(r, g)
            if v:
                why = g
                detail[g][v] += 1
                break
        if why:
            dropped[why] += 1
        else:
            kept.append(r)
    return {"kept": kept, "dropped": dict(dropped),
            "dropped_detail": {k: dict(v) for k, v in detail.items()}}


def _ci(values: List[float], clusters: List) -> tuple:
    """95% CI, resampling FIXTURES. Returns (lo, hi) or (None, None).

    The clustering is not optional: measured over 180 days of production picks,
    31.8% of picks share a fixture with another pick, and an i.i.d. bootstrap
    understates the standard error in the direction that makes a null look
    significant. See `src.evaluation.clv._boot`, which already does this -- the
    report was right about the method and wrong about the population.
    """
    if len(values) < 5:
        return (None, None)
    from src.evaluation.clv import _boot
    return _boot(values, clusters=clusters)


CI_METHOD = ("cluster bootstrap, 4000 iterations, resampling FIXTURES "
             "(match_id) not picks; seed 0")


def _brier(rs: Sequence[Row], which: str) -> Optional[float]:
    vals = [(getattr(r, which) - (1.0 if r.result == "win" else 0.0)) ** 2
            for r in rs]
    return sum(vals) / len(vals) if vals else None


def paired_brier(rows: Sequence[Row]) -> Dict:
    """The model-vs-market comparison, ON THE PICKS THEY SHARE.

    This is the figure the report got wrong. `prob` is populated for every
    pick; `market_prob` only for picks made after migration 004. Comparing a
    Brier over all of them to a Brier over the subset is not a comparison.
    """
    both = [r for r in rows if r.prob is not None and r.market_prob is not None]
    out = {
        "n_paired": len(both),
        "n_model_only": sum(1 for r in rows if r.prob is not None) - len(both),
        "model": None, "market": None, "delta": None, "se": None, "t": None,
    }
    if not both:
        out["state"] = "NO_PAIRED_PICKS"
        return out
    out["model"] = _brier(both, "prob")
    out["market"] = _brier(both, "market_prob")
    d = [((r.prob - (1.0 if r.result == "win" else 0.0)) ** 2
          - (r.market_prob - (1.0 if r.result == "win" else 0.0)) ** 2)
         for r in both]
    out["delta"] = sum(d) / len(d)
    if len(d) > 1:
        mean = out["delta"]
        var = sum((x - mean) ** 2 for x in d) / (len(d) - 1)
        out["se"] = math.sqrt(var / len(d))
        out["t"] = mean / out["se"] if out["se"] else None
    out["state"] = "PAIRED"
    return out


def _arm(label: str, rs: List[Row]) -> Dict:
    """One population's figures. Never pooled with another arm."""
    if not rs:
        return {"label": label, "n": 0, "state": "EMPTY"}
    profits = [profit(r) for r in rs]
    lo, hi = _ci(profits, [r.match_id for r in rs])
    cohorts = Counter(r.model_version or "UNVERSIONED" for r in rs)
    per_cohort = []
    for k, n in sorted(cohorts.items(), key=lambda kv: -kv[1]):
        sub = [r for r in rs if (r.model_version or "UNVERSIONED") == k]
        per_cohort.append({
            "cohort": k, "n": n,
            "win": sum(1 for r in sub if r.result == "win") / n,
            "roi": sum(profit(r) for r in sub) / n,
        })
    return {
        "label": label, "n": len(rs), "state": "OK",
        "win": sum(1 for r in rs if r.result == "win") / len(rs),
        "roi": sum(profits) / len(profits),
        "ci": (lo, hi), "ci_method": CI_METHOD,
        "fixtures": len({r.match_id for r in rs}),
        "n_cohorts": len(cohorts),
        "per_cohort": per_cohort,
        "brier": paired_brier(rs),
        "first": min(r.pick_date for r in rs),
        "last": max(r.pick_date for r in rs),
    }


def summarise(rows: Sequence[Row], window_from: date, window_to: date,
              measured_on: date) -> Dict:
    """Every figure, with its population. Paper and live are separate arms."""
    part = partition([r for r in rows if decided(r)])
    kept = part["kept"]
    in_scope = [r for r in kept if r.pick_date >= EXPERIMENT_START]
    pre = [r for r in kept if r.pick_date < EXPERIMENT_START]
    return {
        "window": (window_from, window_to),
        "measured_on": measured_on,
        "experiment_start": EXPERIMENT_START,
        "experiment_start_why": EXPERIMENT_START_WHY,
        "gates": part,
        "excluded_pre_experiment": len(pre),
        # RULE 3. Two arms, never added together. `live` here is the
        # pre-versioning era, not a live wager -- see the module docstring.
        "paper": _arm("paper (versioned cohorts)",
                      [r for r in in_scope if r.is_paper]),
        "live": _arm("live (pre-versioning legacy)",
                     [r for r in in_scope if not r.is_paper]),
    }


#: THE QUALIFICATIONS ALREADY ON RECORD. They belong next to the figures they
#: qualify, not in a document the reader of the figure does not open.
ON_RECORD = (
    "NO-INFORMATION CONCLUSION, four independent routes (2026-08-07): "
    "logistic divergence coefficient +0.245 z=1.11 p=0.265 on n=1018; "
    "market-only log-likelihood -685.99 BEATS model-only -690.67; "
    "the blend sweep falls monotonically to w=1.0; and break-even CLV "
    "arithmetic. NO FOLD STRUCTURE in the first of those.",
    "bookmaker_blend_weight -> 1.0 IS NOT RECOMMENDED WITHOUT ITS CAVEATS: "
    "it rests on THREE EFFECTIVE FOLDS OF FIVE, which are ADJACENT, in an "
    "ATYPICAL ODDS ERA.",
    "CLV coverage is low and CLV is reported only from prices that passed "
    "clv.validate_pair; a missing closing line is UNEVALUATED, not clear.",
)


def format_telegram(s: Dict) -> str:
    """The six-line body. Line 1 names the population; a figure whose
    population lives in a linked artifact is a figure read without it."""
    a = s["paper"]
    w0, w1 = s["window"]
    if a["state"] != "OK":
        return (f"PAPER TRADING — no settled picks in scope\n"
                f"population: paper, versioned cohorts, "
                f"{s['experiment_start']}..{w1} — n=0")
    lo, hi = a["ci"]
    ci = (f"[{lo:+.1%}, {hi:+.1%}]" if lo is not None else "CI UNEVALUATED (n<5)")
    b = a["brier"]
    lines = [
        f"\U0001f4c8 PAPER TRADING — {a['n']} settled picks, "
        f"{a['first']}..{a['last']}",
        f"population: PAPER only, from {s['experiment_start']} "
        f"(experiment start), {a['n_cohorts']} cohorts POOLED, "
        f"{a['fixtures']} fixtures",
        f"win {a['win']:.1%}  •  flat ROI {a['roi']:+.2%} 95% CI {ci}",
    ]
    if b["state"] == "PAIRED":
        t = f"t={b['t']:+.2f}" if b["t"] is not None else "t UNEVALUATED"
        lines.append(
            f"Brier PAIRED on n={b['n_paired']}: model {b['model']:.4f} vs "
            f"market {b['market']:.4f}, Δ{b['delta']:+.4f} {t}")
    else:
        lines.append(f"Brier: {b['state']} — no pick carries both "
                     f"probabilities, so there is nothing to pair")
    lines.append(
        f"live arm reported SEPARATELY (n={s['live'].get('n', 0)}, "
        f"pre-versioning legacy — never pooled with paper)")
    lines.append(
        f"gates dropped {sum(s['gates']['dropped'].values())}; "
        f"{s['excluded_pre_experiment']} pre-experiment picks excluded. "
        f"Model adds no information over the price — 4 routes, 2026-08-07. "
        f"Detail in the run artifact.")
    return "\n".join(lines)


def format_block(s: Dict) -> List[str]:
    """The long form for the artifact, with the per-cohort breakdown.

    If this is long, that IS the finding: the pooled single number was short
    because it was answering a question nobody could ask.
    """
    w0, w1 = s["window"]
    out = [
        "=" * 88,
        "EXPERIMENT SUMMARY — every figure with its population",
        "=" * 88,
        f"  query window         : pick_date {w0} .. {w1}",
        f"  measured             : {s['measured_on']} (provenance: this DB read)",
        f"  experiment start     : {s['experiment_start']}",
        f"    chosen because     : {s['experiment_start_why']}",
        f"  excluded as pre-experiment : {s['excluded_pre_experiment']} picks",
        "",
        "  GATES — applied here, and nowhere in this report before 2026-10-20:",
    ]
    for g, reason in GATES:
        n = s["gates"]["dropped"].get(g, 0)
        det = s["gates"]["dropped_detail"].get(g) or {}
        out.append(f"    {g:28s} dropped {n:4d}  {det if det else ''}")
        out.append(f"      {reason}")
    out.append("")
    for key in ("paper", "live"):
        a = s[key]
        out.append("-" * 88)
        if a["state"] != "OK":
            out.append(f"  {a['label']}: NO SETTLED PICKS IN SCOPE (n=0)")
            continue
        lo, hi = a["ci"]
        ci = (f"[{lo:+.2%}, {hi:+.2%}]" if lo is not None
              else "UNEVALUATED (n<5 clusters)")
        out += [
            f"  {a['label'].upper()}",
            f"    n                  : {a['n']} settled picks over "
            f"{a['fixtures']} fixtures, {a['first']} .. {a['last']}",
            f"    win rate           : {a['win']:.1%}   (n={a['n']})",
            f"    flat ROI           : {a['roi']:+.2%}  95% CI {ci}",
            f"      CI method        : {a['ci_method']}",
            f"    cohorts POOLED     : {a['n_cohorts']} — enumerated below, "
            f"because a ratio over a mixed population is named or not printed",
        ]
        b = a["brier"]
        if b["state"] == "PAIRED":
            t = f"{b['t']:+.2f}" if b["t"] is not None else "UNEVALUATED"
            out += [
                f"    Brier PAIRED       : model {b['model']:.4f} vs market "
                f"{b['market']:.4f} on the SAME n={b['n_paired']}",
                f"      delta            : {b['delta']:+.4f}  se "
                f"{b['se']:.4f}  t={t}",
                f"      not compared     : {b['n_model_only']} picks carry the "
                f"model's probability and NOT the market's",
            ]
        else:
            out.append(f"    Brier              : {b['state']}")
        out.append(f"    {'cohort':46s}{'n':>6}{'win':>9}{'ROI':>10}")
        for c in a["per_cohort"]:
            out.append(f"    {c['cohort']:46s}{c['n']:>6}"
                       f"{c['win']:>9.1%}{c['roi']:>10.2%}")
    out += ["-" * 88, "  ON RECORD, carried with the figures:"]
    for q in ON_RECORD:
        out.append(f"    * {q}")
    return out
