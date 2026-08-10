"""Prospective paper-trading report and data-quality health checks.

    python -m scripts.paper_trading_report [--days 90] [--include-live]
                                           [--model-version V] [--health-only]

Stage 5, Phases 11, 15 and 20. Read-only.

This is the instrument for the Stage 5 experiment. It reports what the system
would have done, how the market moved afterwards, and whether the evidence is
yet strong enough to say anything — and it refuses to overstate any of it:

* Nothing is called CLV unless it comes from a genuine closing price that passed
  ``clv.validate_pair``. Coverage is always reported alongside.
* Every rate is shown with its sample size, and the headline figures carry
  bootstrap confidence intervals.
* Checkpoint status (100 / 200 / 500 valid closing lines) is stated explicitly,
  so "not enough data yet" is a visible conclusion rather than an omission.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from src.data.models import Match, SavedPick
from src.evaluation.attribution import (FINAL, MODEL, resolve,
                                        shares_one_observation)
from src.evaluation.clv import coverage_report
from src.utils.logger import get_logger

logger = get_logger()

CHECKPOINTS = [
    (100, "data quality only — NO model decision"),
    (200, "initial CLV signal"),
    (500, "meaningful model-vs-market evaluation"),
]

#: Phase 20 alert threshold.
MIN_CLV_COVERAGE = 0.80


def _load_env() -> None:
    """Load .env — from main() only, never at import (see the note in
    capture_closing_lines: an import-time load re-introduces DATABASE_URL and
    lets tests reach production)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass


class _Pick:
    """Flat view of a saved pick plus its match kickoff."""

    __slots__ = ("id", "match_id", "pick_date", "league", "market", "selection",
                 "odds", "prob", "market_prob", "market_books", "ev", "result",
                 "closing_odds", "closing_odds_captured_at", "closing_status",
                 "closing_fair", "kickoff", "is_paper", "model_version",
                 "review_action", "model_market", "model_selection",
                 "model_result", "observations")

    def __init__(self, r):
        for f in self.__slots__:
            setattr(self, f, None)
        self.id = r.id
        # Stage 8: the clustering key. Picks on one fixture are not independent
        # observations, so every confidence interval has to resample fixtures.
        self.match_id = r.match_id
        self.pick_date = r.pick_date
        self.league = r.league or "unknown"
        self.market = r.market
        self.selection = r.selection
        self.odds = r.odds
        self.prob = r.predicted_probability
        self.market_prob = r.market_probability
        self.market_books = r.market_books
        self.ev = r.expected_value
        self.result = r.result
        self.closing_odds = r.closing_odds
        self.closing_odds_captured_at = r.closing_odds_captured_at
        self.closing_status = r.closing_capture_status or "pending"
        self.closing_fair = r.closing_fair_probability
        self.kickoff = r.match_date
        self.is_paper = bool(r.is_paper)
        self.model_version = r.model_version
        self.review_action = r.review_action
        # Stage 9: the model series resolves its close in the MODEL's market,
        # which may differ from the final one after a Claude CHANGE.
        self.model_market = r.model_market
        self.model_selection = r.model_selection
        self.model_result = r.model_result

    @property
    def decided(self) -> bool:
        return self.result in ("win", "loss")

    @property
    def profit(self) -> Optional[float]:
        if not self.decided or not self.odds:
            return None
        return (self.odds - 1) if self.result == "win" else -1.0


def _boot(values: List[float], clusters: Optional[List] = None,
          iters: int = 4000, seed: int = 0):
    """Bootstrap 95% CI for a mean, resampling CLUSTERS when given.

    Stage 8. Picks on the same fixture are not independent observations: both
    prices respond to the same information flowing into that one match. An
    i.i.d. bootstrap over picks treats them as if they were, which understates
    the standard error and produces a confidence interval that is too narrow —
    the direction that makes a null result look significant.

    Measured on 180 days of production picks: 900 fixtures carried 1,070 picks,
    and 170 fixtures (18.9%) carried two — **31.8% of all picks share a fixture
    with another pick**. That is far too much clustering to ignore.

    The fix is a cluster bootstrap: resample fixtures with replacement and take
    all of each drawn fixture's picks. Every pick keeps contributing its own
    information (nothing is collapsed or averaged away — Phase 5 warns against
    discarding genuinely different markets), but the resampling unit becomes the
    independent one.

    ``clusters`` is a parallel sequence of cluster ids. Passing None keeps the
    old i.i.d. behaviour, which is correct only when the values are already one
    per cluster.
    """
    if len(values) < 5:
        return None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)

    if clusters is None:
        means = np.array([rng.choice(arr, len(arr), replace=True).mean()
                          for _ in range(iters)])
        return tuple(np.percentile(means, [2.5, 97.5]))

    groups: Dict = defaultdict(list)
    for v, c in zip(arr, clusters):
        groups[c].append(v)
    keys = list(groups.keys())
    blocks = [np.asarray(groups[k], dtype=float) for k in keys]
    if len(keys) < 5:
        return None, None

    idx = rng.integers(0, len(blocks), size=(iters, len(blocks)))
    means = np.array([
        np.concatenate([blocks[i] for i in row]).mean() for row in idx
    ])
    return tuple(np.percentile(means, [2.5, 97.5]))


def _effective_n(clusters: List) -> tuple:
    """(n_picks, n_fixtures, design_effect, effective_n) for a clustered sample.

    ``design_effect = 1 + (E[m^2]/E[m] - 1) * rho`` is the factor by which the
    variance of a mean is inflated by clustering. Rho — the intra-fixture
    correlation — is not identifiable from a handful of observations, so this
    reports the WORST CASE, rho = 1: two picks on one fixture carry no more
    information than one. The truth lies between that and the naive count, and
    quoting the pessimistic bound is the right way round for a stopping rule.
    """
    if not clusters:
        return 0, 0, 1.0, 0.0
    sizes = Counter(clusters)
    n = len(clusters)
    k = len(sizes)
    m = np.asarray(list(sizes.values()), dtype=float)
    deff = (m ** 2).sum() / m.sum()          # E[m^2]/E[m] with rho = 1
    return n, k, float(deff), float(n / deff) if deff else 0.0


def _fmt_ci(lo, hi, pct=True):
    if lo is None:
        return "n/a"
    return f"[{lo:+.1%}, {hi:+.1%}]" if pct else f"[{lo:+.4f}, {hi:+.4f}]"


def load_picks(days: int, include_live: bool, model_version: Optional[str]):
    from src.data.database import get_db

    db = get_db()
    cutoff = date.today() - timedelta(days=days)
    with db.get_session() as s:
        q = s.query(
            SavedPick.id, SavedPick.match_id,
            SavedPick.pick_date, SavedPick.league, SavedPick.market,
            SavedPick.selection, SavedPick.odds, SavedPick.predicted_probability,
            SavedPick.market_probability, SavedPick.market_books,
            SavedPick.expected_value, SavedPick.result, SavedPick.closing_odds,
            SavedPick.closing_odds_captured_at, SavedPick.closing_capture_status,
            SavedPick.closing_fair_probability, SavedPick.is_paper,
            SavedPick.model_version, SavedPick.review_action,
            SavedPick.model_market, SavedPick.model_selection,
            SavedPick.model_result,
            Match.match_date,
        ).join(Match, Match.id == SavedPick.match_id).filter(
            SavedPick.pick_date >= cutoff)
        if model_version:
            q = q.filter(SavedPick.model_version == model_version)
        rows = q.all()

    picks = [_Pick(r) for r in rows]
    if not include_live:
        paper = [p for p in picks if p.is_paper]
        if paper:
            picks = paper
    _attach_observations(picks)
    return picks


def _attach_observations(picks: List[_Pick]) -> None:
    """Hang each pick's MODEL/FINAL observation rows off it (Stage 10).

    One extra query for the whole report, column-projected. Picks written before
    migration 006 simply get no observations, and the report falls back to the
    Stage 9 derivation from `saved_picks` — which is correct for them, because
    an unchanged pick's own close IS the model's close.
    """
    from src.data.database import get_db
    from src.data.models import PickObservation

    for p in picks:
        p.observations = {}
    ids = [p.id for p in picks]
    if not ids:
        return

    try:
        with get_db().get_session() as s:
            rows = s.query(
                PickObservation.pick_id, PickObservation.attribution,
                PickObservation.market, PickObservation.selection,
                PickObservation.taken_odds, PickObservation.closing_odds,
                PickObservation.closing_status,
                PickObservation.closing_captured_at,
            ).filter(PickObservation.pick_id.in_(ids)).all()
    except Exception as e:
        logger.debug(f"pick_observations unavailable ({e}) — reporting from "
                     f"saved_picks only. Is migration 006 applied?")
        return

    by_pick: Dict[int, Dict[str, object]] = defaultdict(dict)
    for r in rows:
        by_pick[r.pick_id][r.attribution] = r
    for p in picks:
        p.observations = dict(by_pick.get(p.id, {}))


# ─────────────────────────────────────────────────────────────────── sections

def section_operational(picks: List[_Pick]) -> None:
    """Stage 7 section 16 — the operational picture behind the experiment.

    Answers "is the machine running?" separately from "is the model any good?",
    so an empty CLV series can be attributed to a pipeline gap rather than
    mistaken for a modelling result.
    """
    from src.data.database import get_db
    from src.data.models import Match, Odds
    from src.data.odds_quota import (CREDITS_PER_REQUEST, FREE_TIER_CREDITS,
                                     OddsApiQuota)
    from src.utils.config import get_config

    print()
    print("=" * 88)
    print("OPERATIONAL")
    print("=" * 88)

    db = get_db()
    cfg = get_config()
    with db.get_session() as s:
        from sqlalchemy import func as _f

        fixtures_future = s.query(_f.count(Match.id)).filter(
            Match.is_fixture.is_(True), Match.match_date > _f.now()).scalar() or 0
        fixtures_total = s.query(_f.count(Match.id)).filter(
            Match.is_fixture.is_(True)).scalar() or 0
        latest_fixture = s.query(_f.max(Match.match_date)).filter(
            Match.is_fixture.is_(True)).scalar()
        # Odds written by the Odds API path, which is what closing capture uses.
        newest_odds = s.query(_f.max(Odds.timestamp)).filter(
            Odds.bookmaker.like("TheOddsAPI%")).scalar()

    print(f"  fixtures in DB (total / future) : {fixtures_total} / {fixtures_future}")
    print(f"  latest fixture kickoff          : {latest_fixture}")
    print(f"  newest Odds-API odds row        : {newest_odds}")
    if fixtures_future == 0:
        print("  NOTE: no future fixtures — the refresh job will select 0 leagues")
        print("        and spend 0 credits until ingestion runs.")

    print(f"  paper predictions in range      : {sum(1 for p in picks if p.is_paper)}")

    quota = OddsApiQuota(
        db,
        monthly_budget=int(cfg.get("odds_api.monthly_credit_budget", 400)),
        safety_margin=int(cfg.get("odds_api.safety_margin_credits", 50)),
    )
    used = quota.used() if quota.available() else None
    if used is None:
        print("  API credits (ledger)            : unavailable (api_budget missing)")
    else:
        print(f"  API credits used this month     : {used}/{quota.monthly_budget} "
              f"(free tier {FREE_TIER_CREDITS})")
        print(f"  API credits remaining           : {quota.remaining()} "
              f"= {quota.max_requests()} league request(s) "
              f"at {CREDITS_PER_REQUEST}/request")

    by_status: Dict[str, int] = defaultdict(int)
    for p in picks:
        by_status[p.closing_status or "pending"] += 1
    print("  closing capture status          :")
    for k in ("captured", "missing", "late", "invalid", "pending"):
        if by_status.get(k):
            print(f"      {k:<10}{by_status[k]:>6}")
    resolved = [p for p in picks if p.kickoff and p.kickoff.date() < date.today()]
    if resolved:
        cap = sum(1 for p in resolved if p.closing_status == "captured")
        print(f"  capture coverage (past kickoff) : {cap}/{len(resolved)} "
              f"= {cap/len(resolved):.1%}")


def section_volume(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("VOLUME")
    print("=" * 88)
    print(f"  total picks            : {len(picks)}")
    print(f"  paper                  : {sum(1 for p in picks if p.is_paper)}")
    print(f"  live                   : {sum(1 for p in picks if not p.is_paper)}")
    print(f"  settled                : {sum(1 for p in picks if p.decided)}")

    for label, key in (("market", lambda p: p.market),
                       ("league", lambda p: p.league),
                       ("model_version", lambda p: p.model_version or "unversioned")):
        counts: Dict[str, int] = defaultdict(int)
        for p in picks:
            counts[key(p)] += 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
        print(f"\n  by {label}:")
        for name, n in top:
            print(f"      {str(name)[:44]:<46}{n:>6}")
        if len(counts) > 10:
            print(f"      ... and {len(counts) - 10} more")


def section_pricing(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("PRICING")
    print("=" * 88)
    odds = [p.odds for p in picks if p.odds]
    probs = [p.prob for p in picks if p.prob]
    mprobs = [p.market_prob for p in picks if p.market_prob]
    evs = [p.ev for p in picks if p.ev is not None]
    if not odds:
        print("  (no priced picks)")
        return
    print(f"  average odds                : {np.mean(odds):.3f}")
    print(f"  median odds                 : {np.median(odds):.3f}")
    print(f"  average model probability   : {np.mean(probs):.4f}" if probs else
          "  average model probability   : n/a")
    if mprobs:
        print(f"  average market probability  : {np.mean(mprobs):.4f} "
              f"(n={len(mprobs)} with a stored consensus)")
    else:
        print("  average market probability  : n/a — market_probability is only "
              "populated for picks made after migration 004")
    if evs:
        print(f"  average predicted EV        : {np.mean(evs):+.2%}")


def series_clv(p: _Pick, attribution: str) -> Optional[float]:
    """CLV for one attribution series on one pick, or None if unmeasurable.

    Stage 10. Two sources, in priority order, and never mixed on one pick:

    1. `pick_observations` — the authority once migration 006 is applied. Each
       row carries its OWN taken price, so a CHANGE yields two genuinely
       independent CLVs.
    2. `saved_picks` — the Stage 9 fallback for picks written before the table
       existed. It can only serve the model series when the selection was never
       changed, because that is the only case where `saved_picks.odds` is still
       the model's price. On a changed pick it returns None rather than
       borrowing the final selection's numbers.
    """
    obs = (getattr(p, "observations", None) or {}).get(attribution)
    if obs is not None:
        if obs.closing_status != "captured" or not obs.closing_odds:
            return None
        if not obs.taken_odds or obs.taken_odds <= 1.0:
            return None
        return obs.taken_odds / obs.closing_odds - 1

    if p.closing_status != "captured" or not p.closing_odds:
        return None
    m, f = resolve(p)
    spec = f if attribution == FINAL else m
    if not spec.measurable:
        return None
    if attribution == MODEL and not shares_one_observation(m, f):
        return None
    return spec.taken_odds / p.closing_odds - 1


def _series_stats(label: str, clvs: List[float], fixtures: List) -> None:
    """One attribution series' CLV block, clustered by fixture (Stage 8/9)."""
    if not clvs:
        print(f"  {label:<8} no valid closing lines")
        return
    lo, hi = _boot(clvs, clusters=fixtures)
    n, k, deff, n_eff = _effective_n(fixtures)
    print(f"  {label:<8} valid closing lines : {n}")
    print(f"  {'':<8} independent fixtures: {k}")
    print(f"  {'':<8} effective n         : {n_eff:.0f} "
          f"(design effect {deff:.2f}, worst case)")
    print(f"  {'':<8} CLV mean            : {np.mean(clvs):+.3%}  "
          f"95% CI {_fmt_ci(lo, hi)}")
    print(f"  {'':<8} CLV median          : {np.median(clvs):+.3%}")
    print(f"  {'':<8} positive CLV        : "
          f"{sum(1 for c in clvs if c > 0) / len(clvs):.1%}")


def section_attribution_coverage(picks: List[_Pick]) -> None:
    """Stage 9, section 12 — which series each pick could even contribute to.

    This is about the PICK side only: whether the row records what a series
    needs. Whether a closing price was then found is a separate fact, reported
    in the CLV section. Keeping them apart matters because an unavailable model
    snapshot is not a failed model prediction.
    """
    from src.evaluation.attribution import (MODEL_PRICE_NOT_KEPT,
                                            NO_MODEL_SNAPSHOT, coverage_class,
                                            resolve, selection_changed)

    print("\n" + "=" * 88)
    print("ATTRIBUTION COVERAGE  (model vs final — Stage 9)")
    print("=" * 88)

    classes: Dict[str, int] = defaultdict(int)
    reasons: Dict[str, int] = defaultdict(int)
    changed = same = unknown = 0

    for p in picks:
        m, f = resolve(p)
        classes[coverage_class(m, f)] += 1
        if not m.measurable:
            reasons[m.unavailable_reason] += 1
        ch = selection_changed(p)
        if ch is None:
            unknown += 1
        elif ch:
            changed += 1
        else:
            same += 1

    total = len(picks) or 1
    print(f"  picks considered        : {len(picks)}")
    print(f"  same selection          : {same}")
    print(f"  changed selection       : {changed}")
    print(f"  snapshot unavailable    : {unknown}  "
          f"(cannot say whether the selection changed)")

    print("\n  measurability (from the pick record, before closing prices):")
    for name in ("both_measurable_same_selection", "both_measurable",
                 "model_only_measurable", "final_only_measurable",
                 "neither_measurable"):
        n = classes.get(name, 0)
        print(f"      {name:<32}{n:>6}  {n/total:>6.1%}")

    if reasons:
        print("\n  model series unavailable because:")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            note = ""
            if reason == NO_MODEL_SNAPSHOT:
                note = "  (row predates the snapshot columns)"
            elif reason == MODEL_PRICE_NOT_KEPT:
                note = "  (Claude CHANGE overwrote the taken price)"
            print(f"      {reason:<40}{n:>6}{note}")

    print("\n  NOTE: 'unavailable' means the record cannot say what the model")
    print("        picked. It is NOT a failed prediction and is never counted")
    print("        as zero CLV.")


def section_clv(picks: List[_Pick]) -> None:

    print("\n" + "=" * 88)
    print("CLOSING LINE VALUE  (two attribution series — Stage 9)")
    print("=" * 88)
    cov = coverage_report(picks)
    print(cov.render())

    # A captured close belongs to the FINAL selection: capture_closing_lines
    # resolves SavedPick.selection. Where the review kept the model's pick the
    # two series are one underlying observation and the same close is attributed
    # to both — one fact, two counters, never two fixtures (section 13).
    model_clvs, model_fx = [], []
    final_clvs, final_fx = [], []
    paired = []          # (match_id, model_clv, final_clv) — genuinely both
    shared = 0

    for p in picks:
        mc = series_clv(p, "model")
        fc = series_clv(p, "final")
        if fc is not None:
            final_clvs.append(fc)
            final_fx.append(p.match_id)
        if mc is not None:
            model_clvs.append(mc)
            model_fx.append(p.match_id)
        if mc is not None and fc is not None:
            paired.append((p.match_id, mc, fc))
            m, f = resolve(p)
            if shares_one_observation(m, f):
                shared += 1

    if not final_clvs and not model_clvs:
        print("\n  No genuine CLV yet in either series. Nothing below can be "
              "reported until closing prices exist.")
        print("  NOTE: model probability minus 1/odds is NOT CLV and is not "
              "shown here.")
        return

    print("\n  Series A — FROZEN MODEL (model_market / model_selection)")
    _series_stats("model", model_clvs, model_fx)
    print("\n  Series B — FINAL SELECTION (market / selection, post-review)")
    _series_stats("final", final_clvs, final_fx)

    print(f"\n  shared observations (review kept the model's pick): {shared}")
    print("      counted once as a model CLV and once as a final CLV, but as")
    print("      ONE fixture — never two independent observations.")

    if paired:
        _section_paired(paired)

    _section_by_review_action(picks)

    valid = [p for p in picks
             if p.closing_status == "captured" and p.closing_odds and p.odds]
    clvs = final_clvs
    if not valid or not clvs:
        return
    print("\n  Breakdowns below are the FINAL series.")

    print(f"\n  {'by market':<28}{'n':>5}{'avg CLV':>10}{'positive':>10}")
    by_market: Dict[str, List[float]] = defaultdict(list)
    for p, c in zip(valid, clvs):
        by_market[p.market or "?"].append(c)
    for mkt, vals in sorted(by_market.items(), key=lambda kv: -len(kv[1])):
        print(f"  {mkt[:27]:<28}{len(vals):>5}{np.mean(vals):>10.2%}"
              f"{sum(1 for v in vals if v > 0)/len(vals):>10.0%}")

    print(f"\n  {'by odds bucket':<28}{'n':>5}{'avg CLV':>10}{'positive':>10}")
    buckets = [(1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 100.0)]
    for lo_b, hi_b in buckets:
        vals = [c for p, c in zip(valid, clvs) if lo_b <= p.odds < hi_b]
        if vals:
            print(f"  {f'{lo_b:.1f}-{hi_b:.1f}':<28}{len(vals):>5}"
                  f"{np.mean(vals):>10.2%}"
                  f"{sum(1 for v in vals if v > 0)/len(vals):>10.0%}")


def _section_paired(paired: List[tuple]) -> None:
    """Stage 9, section 10 — the paired subset.

    Only picks where BOTH series produced a valid close. The difference is
    reported as a within-pick delta, clustered by fixture exactly as elsewhere.

    It is not an independent CLV sample and is not presented as one: each delta
    is one pick's two measurements, so the population is the paired picks, not
    2 x that.
    """
    print("\n  PAIRED SUBSET — both series measurable on the same pick")
    fx = [m for m, _, _ in paired]
    mc = [c for _, c, _ in paired]
    fc = [c for _, _, c in paired]
    deltas = [f - m for (_, m, f) in paired]
    n, k, deff, n_eff = _effective_n(fx)
    lo, hi = _boot(deltas, clusters=fx)
    print(f"      paired observations : {n} on {k} fixtures "
          f"(effective n {n_eff:.0f})")
    print(f"      model CLV mean      : {np.mean(mc):+.3%}")
    print(f"      final CLV mean      : {np.mean(fc):+.3%}")
    print(f"      final - model       : {np.mean(deltas):+.3%}  "
          f"95% CI {_fmt_ci(lo, hi)}")
    print("      (observed difference in CLV — not a causal claim about the "
          "review)")


def _section_by_review_action(picks: List[_Pick]) -> None:
    """Stage 9, section 11 — CLV split by what the review did.

    For CHANGE rows the model and final series are different bets, so the delta
    is the diagnostic. Where the model's own close is unobtainable that is
    stated, not filled in.
    """
    print("\n  BY REVIEW ACTION")
    print(f"      {'action':<10}{'n close':>9}{'model CLV':>12}"
          f"{'final CLV':>12}{'delta':>10}")

    buckets: Dict[str, List[_Pick]] = defaultdict(list)
    for p in picks:
        if (series_clv(p, MODEL) is not None
                or series_clv(p, FINAL) is not None):
            buckets[p.review_action or "none"].append(p)

    for action in ("none", "KEEP", "CHANGE"):
        rows = buckets.get(action, [])
        if not rows:
            print(f"      {action:<10}{0:>9}{'—':>12}{'—':>12}{'—':>10}")
            continue
        mc, fc, dl = [], [], []
        for p in rows:
            fv = series_clv(p, FINAL)
            mv = series_clv(p, MODEL)
            if fv is not None:
                fc.append(fv)
            if mv is not None:
                mc.append(mv)
            if mv is not None and fv is not None:
                dl.append(fv - mv)
        print(f"      {action:<10}{len(rows):>9}"
              f"{(f'{np.mean(mc):+.2%}' if mc else 'n/a'):>12}"
              f"{(f'{np.mean(fc):+.2%}' if fc else 'n/a'):>12}"
              f"{(f'{np.mean(dl):+.2%}' if dl else 'n/a'):>10}")

    if buckets.get("CHANGE"):
        print("      NOTE: on CHANGE rows the model and final series are")
        print("            DIFFERENT bets. A model CLV of 'n/a' means the")
        print("            model selection's own close was not obtainable —")
        print("            the final selection's close is never substituted.")


def section_outcomes(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("OUTCOMES")
    print("=" * 88)
    decided = [p for p in picks if p.decided]
    if not decided:
        print("  (nothing settled yet)")
        return
    profits = [p.profit for p in decided]
    lo, hi = _boot(profits, clusters=[p.match_id for p in decided])
    wins = sum(1 for p in decided if p.result == "win")
    print(f"  settled            : {len(decided)}")
    print(f"  win rate           : {wins / len(decided):.1%}")
    print(f"  flat ROI           : {np.mean(profits):+.2%}  95% CI {_fmt_ci(lo, hi)}")

    scored = [p for p in decided if p.prob]
    if scored:
        brier = np.mean([(p.prob - (1.0 if p.result == "win" else 0.0)) ** 2
                         for p in scored])
        ll = np.mean([-math.log(max(min(p.prob if p.result == "win"
                                        else 1 - p.prob, 1 - 1e-9), 1e-9))
                      for p in scored])
        print(f"  Brier (model)      : {brier:.4f}   (n={len(scored)})")
        print(f"  log loss (model)   : {ll:.4f}")

    mscored = [p for p in decided if p.market_prob]
    if mscored:
        mbrier = np.mean([(p.market_prob - (1.0 if p.result == "win" else 0.0)) ** 2
                          for p in mscored])
        print(f"  Brier (market)     : {mbrier:.4f}   (n={len(mscored)}) "
              f"<- the number the model must beat")


def section_calibration(picks: List[_Pick]) -> None:
    """Phase 15 — prospective edge-bucket calibration."""
    print("\n" + "=" * 88)
    print("EDGE CALIBRATION (Phase 15 — measuring, not optimising)")
    print("=" * 88)
    rows = [p for p in picks if p.decided and p.prob and p.market_prob]
    if not rows:
        print("  Requires market_probability, populated only for picks made after")
        print("  migration 004. No qualifying picks yet.")
        return
    print(f"  {'edge bucket':<16}{'n':>5}{'avg pred':>10}{'realised':>10}"
          f"{'Brier':>9}{'ROI':>9}{'avg CLV':>10}")
    edges = [(0.00, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 1.01)]
    for lo_e, hi_e in edges:
        sub = [p for p in rows if lo_e <= (p.prob - p.market_prob) < hi_e]
        if not sub:
            continue
        realised = sum(1 for p in sub if p.result == "win") / len(sub)
        brier = np.mean([(p.prob - (1.0 if p.result == "win" else 0.0)) ** 2
                         for p in sub])
        roi = np.mean([p.profit for p in sub])
        clv_vals = [p.odds / p.closing_odds - 1 for p in sub
                    if p.closing_status == "captured" and p.closing_odds]
        clv = f"{np.mean(clv_vals):+.2%}" if clv_vals else "n/a"
        print(f"  {f'{lo_e:.0%}-{hi_e:.0%}':<16}{len(sub):>5}"
              f"{np.mean([p.prob for p in sub]):>10.3f}{realised:>10.3f}"
              f"{brier:>9.4f}{roi:>9.1%}{clv:>10}")


def section_claude(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("CLAUDE REVIEW (isolated — no conclusion drawn)")
    print("=" * 88)
    reviewed = [p for p in picks if p.review_action]
    if not reviewed:
        print("  (no reviewed picks in range)")
        return
    for action in ("KEEP", "CHANGE"):
        sub = [p for p in reviewed if p.review_action == action and p.decided]
        if not sub:
            continue
        roi = np.mean([p.profit for p in sub])
        lo, hi = _boot([p.profit for p in sub],
                       clusters=[p.match_id for p in sub])
        wr = sum(1 for p in sub if p.result == "win") / len(sub)
        print(f"  {action:<8} n={len(sub):<5} win={wr:.1%} ROI={roi:+.1%} "
              f"95% CI {_fmt_ci(lo, hi)}")
    counter = [p for p in reviewed
               if p.decided and p.model_result in ("win", "loss")]
    if counter:
        final = sum(1 for p in counter if p.result == "win") / len(counter)
        model = sum(1 for p in counter if p.model_result == "win") / len(counter)
        print(f"\n  counterfactual (n={len(counter)}): final {final:.1%} vs "
              f"model's original {model:.1%}")
    else:
        print("\n  counterfactual: no picks yet carry both a final and a "
              "model_result — cannot compare.")


def section_health(picks: List[_Pick]) -> int:
    """Phase 20 — explicit health checks. Returns the number of alerts."""
    print("\n" + "=" * 88)
    print("DATA QUALITY HEALTH CHECKS")
    print("=" * 88)
    alerts: List[str] = []
    ok: List[str] = []

    # Scope to the go-forward cohort. Picks made before Stage 5 carry no
    # model_version and pre-date the closing-capture system entirely — they can
    # never have a closing price, so including them would pin every coverage
    # check at 0% forever. A permanently-red check is one nobody reads.
    legacy = [p for p in picks if not p.model_version]
    scoped = [p for p in picks if p.model_version]
    if legacy:
        print(f"  INFO  {len(legacy)} pre-Stage-5 pick(s) excluded from coverage "
              f"checks (no model_version; they pre-date closing capture)")
    if not scoped:
        print("  INFO  no Stage 5 picks yet — coverage checks will start "
              "reporting once the frozen model writes its first prediction")
        return 0

    resolved = [p for p in scoped
                if p.kickoff and p.kickoff.date() < date.today()]
    captured = [p for p in resolved if p.closing_status == "captured"]
    if resolved:
        rate = len(captured) / len(resolved)
        msg = (f"closing capture coverage {rate:.1%} "
               f"({len(captured)}/{len(resolved)} past-kickoff picks)")
        (alerts if rate < MIN_CLV_COVERAGE else ok).append(
            msg + (f" — below the {MIN_CLV_COVERAGE:.0%} threshold"
                   if rate < MIN_CLV_COVERAGE else ""))

    bad_ts = [p for p in captured
              if p.closing_odds_captured_at and p.kickoff
              and p.closing_odds_captured_at > p.kickoff]
    (alerts if bad_ts else ok).append(
        f"{len(bad_ts)} captured price(s) timestamped after kickoff")

    no_ts = [p for p in captured if not p.closing_odds_captured_at]
    (alerts if no_ts else ok).append(
        f"{len(no_ts)} captured price(s) with no capture timestamp")

    orphan = [p for p in captured if not p.closing_odds]
    (alerts if orphan else ok).append(
        f"{len(orphan)} pick(s) marked captured but carrying no closing price")

    stale = [p for p in scoped
             if p.closing_status == "pending" and p.kickoff
             and p.kickoff.date() < date.today() - timedelta(days=1)]
    (alerts if stale else ok).append(
        f"{len(stale)} Stage 5 pick(s) still 'pending' more than a day after "
        f"kickoff (the capture never ran for them)")

    settle_fail = [p for p in scoped
                   if p.kickoff and p.kickoff.date() < date.today() - timedelta(days=2)
                   and not p.result]
    (alerts if settle_fail else ok).append(
        f"{len(settle_fail)} pick(s) unsettled more than 2 days after kickoff")

    for line in ok:
        print(f"  OK    {line}")
    for line in alerts:
        print(f"  ALERT {line}")
    if not alerts:
        print("\n  All health checks pass.")
    return len(alerts)


def section_checkpoints(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("SAMPLE-SIZE CHECKPOINTS (Phase 16)")
    print("=" * 88)
    # Stage 9: the two series get SEPARATE counters. A single underlying
    # observation counts as one model valid CLV and one final valid CLV, but the
    # fixture behind it is one fixture — so the effective-n figures are computed
    # per series over that series' own fixtures, never pooled.
    model_fx, final_fx = [], []
    for p in picks:
        if series_clv(p, FINAL) is not None:
            final_fx.append(p.match_id)
        if series_clv(p, MODEL) is not None:
            model_fx.append(p.match_id)

    # The checkpoint DEFINITION is unchanged — 100/200/500 valid closing lines,
    # counted in picks. What Stage 8 added is the effective sample size beside
    # it, because a fixture contributing two closing lines advances the counter
    # by two while advancing the evidence by less.
    for label, fx in (("MODEL", model_fx), ("FINAL", final_fx)):
        valid, k, deff, n_eff = _effective_n(fx)
        print(f"\n  {label}:")
        print(f"      valid closing-line picks: {valid}")
        print(f"      independent fixtures    : {k}")
        print(f"      worst-case effective n  : {n_eff:.0f}  "
              f"(design effect {deff:.2f})")
        for target, purpose in CHECKPOINTS:
            status = "REACHED" if valid >= target else f"{target - valid} to go"
            print(f"          {target:>4} — {purpose:<40} {status}")
            if valid >= target > n_eff:
                print(f"               ^ reached on pick count, but only "
                      f"~{n_eff:.0f} effective observations — provisional")

    if len(final_fx) < CHECKPOINTS[0][0] and len(model_fx) < CHECKPOINTS[0][0]:
        print("\n  Below the first checkpoint in both series. No model decision "
              "may be taken from this data.")


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=90)
    ap.add_argument("--include-live", action="store_true",
                    help="include non-paper picks (default: paper only when any exist)")
    ap.add_argument("--model-version", default=None)
    ap.add_argument("--health-only", action="store_true")
    args = ap.parse_args()

    picks = load_picks(args.days, args.include_live, args.model_version)
    print("=" * 88)
    print(f"PAPER TRADING REPORT — last {args.days} days, {len(picks)} picks")
    if args.model_version:
        print(f"filtered to model_version = {args.model_version}")
    print("=" * 88)
    if not picks:
        print("No picks in range.")
        return

    if args.health_only:
        raise SystemExit(1 if section_health(picks) else 0)

    section_operational(picks)
    section_volume(picks)
    section_attribution_coverage(picks)
    section_pricing(picks)
    section_clv(picks)
    section_outcomes(picks)
    section_calibration(picks)
    section_claude(picks)
    section_checkpoints(picks)
    alerts = section_health(picks)
    print()
    raise SystemExit(1 if alerts else 0)


if __name__ == "__main__":
    main()
