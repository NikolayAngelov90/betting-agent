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
from src.evaluation.clv import _boot, _effective_n
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

#: ── CLOSING-CAPTURE COVERAGE: A RATCHET, DERIVED FROM THE MEASUREMENT ──────
#:
#: `MIN_CLV_COVERAGE = 0.80` was Phase 20's alert threshold and it is RETIRED.
#: It required 80% against a measured 12.54%, so it was breached on every run
#: it ever made -- and **nothing ever relied on it**. That is not an opinion:
#: its only consumer was this module's exit code, and the workflow piped the
#: process to `tee`, which discarded it (see `HEALTH_ALERT_EXIT`). A threshold
#: whose breach cannot change any observable has never been a threshold. It
#: could not be kept at 80% either: an alarm that always fires carries no
#: information, and this project has paid for that six times.
#:
#: THE DENOMINATOR, because a coverage figure without one is not a figure.
#: Picks carrying a `model_version` (legacy picks pre-date closing capture and
#: are excluded by `section_health` on purpose), whose MATCH kickoff is
#: strictly before today, inside the report's `--days` window.
#:
#:     measured 2026-10-21, --days 120:  165 / 1316 = 0.1254
#:     late 634 (48.2%) | missing 517 (39.3%) | captured 165 (12.5%)
#:
#: Nearly half are `late` -- captured after kickoff, which `validate_pair`
#: refuses. Raising coverage is NOT this constant's job and is not done here.
CLOSING_COVERAGE_MEASURED = 0.1254
CLOSING_COVERAGE_DENOMINATOR = (
    "model_version IS NOT NULL AND match kickoff < today, within --days; "
    "165/1316 measured 2026-10-21 at --days 120")

#: The floor: the measured value FLOORED to two decimals, so that one bad day
#: cannot trip it. Derived, not chosen -- a day adds roughly 10-45 resolved
#: picks to a 1316 denominator, so a zero-capture day moves the rolling figure
#: about -0.4pp, and 0.0054 of slack absorbs one such day.
CLOSING_COVERAGE_FLOOR = 0.12

#: AND IT RATCHETS BOTH WAYS. A fall below the floor is the alarm. A rise of
#: more than this band ALSO alarms, because the floor must then be re-recorded
#: deliberately -- otherwise the floor drifts upward silently and the next fall
#: is measured against a number nobody wrote down. 3pp is about a week of
#: sustained improvement at the movement rate above, which is worth recording.
CLOSING_COVERAGE_RISE_BAND = 0.03

#: EXIT CODE FOR "THE REPORT RAN AND A HEALTH CHECK ALARMED". Distinct from 1,
#: which now means only "the report could not run".
#:
#: WHY IT NEEDED TO BE DISTINCT. 2026-10-21: the workflow ran this module as
#: `python -m scripts.paper_trading_report | tee logs/...`, and a pipeline
#: returns the LAST command's status. GitHub runs `run:` under `bash -e`
#: WITHOUT `-o pipefail`, so `tee` returned 0 and python's exit code was
#: discarded. Verified: `python -c 'raise SystemExit(1)' | tee /dev/null`
#: exits 0 bare, 1 under pipefail.
#:
#: SO THE 80% COVERAGE THRESHOLD HAD NEVER FIRED. The inventory measured this
#: workflow at 31 runs / 31 ok / 0 fail, and that record was the swallowing,
#: not health -- coverage has been 12.5%. A threshold whose breach cannot
#: change any observable is a check that cannot fail.
#:
#: The workflow now sets `pipefail` and treats this code as a FINDING: a
#: GitHub warning, the alert carried in the delivered summary, and the job
#: still green. A genuine crash exits 1 and fails, and the existing
#: "the instrument is blind today" alert then says something true -- which it
#: would NOT have for a health alert, because the instrument worked and that
#: is how it found the problem.
HEALTH_ALERT_EXIT = 2


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
                 "model_result", "observations", "disposition",
                 "evidence_status", "training_exclusion_reason")

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
        # Stage 13 Step 1b. Non-NULL means the row is not a bet that was taken
        # (currently only 'consolidated' — superseded by a sibling pick on the
        # same match). Kept in the MODEL series, excluded from FINAL.
        self.disposition = getattr(r, "disposition", None)
        # 2026-10-20: the two gates this report applied NOWHERE. See
        # src/reporting/experiment_summary.GATES.
        self.evidence_status = getattr(r, "evidence_status", None)
        self.training_exclusion_reason = getattr(
            r, "training_exclusion_reason", None)
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
            SavedPick.model_result, SavedPick.disposition,
            SavedPick.evidence_status,
            Match.training_exclusion_reason,
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
        logger.warning(f"pick_observations unavailable ({e}) — reporting from "
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
        # ── NAME THE DENOMINATOR. 2026-10-21. ──────────────────────────────
        # This figure and `section_health`'s share the numerator 165 and do
        # NOT share a denominator: 1606 here, 1316 there, because the health
        # check excludes pre-Stage-5 picks that can never have a closing
        # price. Two numbers both called "coverage", 2.2pp apart, and only one
        # said what it ranged over.
        print(f"  capture coverage (past kickoff) : {cap}/{len(resolved)} "
              f"= {cap/len(resolved):.1%}")
        print(f"    denominator: ALL past-kickoff picks in range, legacy "
              f"INCLUDED. The health check's figure is the same numerator "
              f"over model_version-scoped picks only, so the two differ.")


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
    # Stage 13 Step 1b — a superseded pick was never taken, so it cannot
    # contribute a FINAL observation: its sibling on the same match holds the
    # actual bet and counting both would double-weight one wager. The MODEL
    # series keeps it, because the frozen model really did select it at a real
    # price, and that is exactly what the MODEL series measures.
    if attribution == FINAL and getattr(p, "disposition", None):
        return None

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
                                            resolve_effective,
                                            selection_changed)

    print("\n" + "=" * 88)
    print("ATTRIBUTION COVERAGE  (model vs final — Stage 9)")
    print("=" * 88)

    classes: Dict[str, int] = defaultdict(int)
    reasons: Dict[str, int] = defaultdict(int)
    changed = same = unknown = 0

    for p in picks:
        # Stage 12.1: observation-aware. A CHANGE pick whose MODEL price was
        # recorded at pick time IS measurable, even though saved_picks.odds was
        # overwritten by the review.
        m, f = resolve_effective(p)
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


def _to_rows(picks: List[_Pick]):
    """`_Pick` -> `experiment_summary.Row`. One place, so the gated figures and
    the Telegram body can never range over a different population than the
    block printed above them."""
    from src.reporting.experiment_summary import Row
    return [Row(
        pick_id=p.id, match_id=p.match_id, pick_date=p.pick_date,
        result=p.result, is_paper=bool(p.is_paper),
        model_version=p.model_version, disposition=p.disposition,
        evidence_status=p.evidence_status,
        training_exclusion_reason=p.training_exclusion_reason,
        odds=p.odds, prob=p.prob, market_prob=p.market_prob,
    ) for p in picks]


def section_experiment_summary(picks: List[_Pick], days: int) -> dict:
    """Every headline figure with the population it ranges over.

    THE BLOCK ABOVE THIS ONE IS THE UNGATED VIEW, KEPT DELIBERATELY. Both are
    printed so the gates' effect is visible rather than asserted: measured
    2026-10-20, they move the settled count 1534 -> 1519 and the ROI
    -4.67% -> -4.73%, and the paper/live split moves it -5.53% / -1.33%. The
    gates matter for correctness, not for the headline, and saying so requires
    showing both.
    """
    from datetime import date as _date, timedelta as _td
    from src.reporting.experiment_summary import format_block, summarise
    today = _date.today()
    s = summarise(_to_rows(picks), today - _td(days=days), today, today)
    print()
    for line in format_block(s):
        print(line)
    return s


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
    # RULE 2: a ratio without its population is not a figure. This block is
    # the UNGATED view and says so; the gated one follows immediately.
    _paper = sum(1 for p in decided if p.is_paper)
    _cohorts = len({p.model_version or "UNVERSIONED" for p in decided})
    print(f"  population         : UNGATED — {_paper} paper + "
          f"{len(decided) - _paper} live, {_cohorts} cohort(s) POOLED, "
          f"no disposition/evidence_status gate. The gated figures are in "
          f"EXPERIMENT SUMMARY below.")
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
        print(f"  Brier (model, ALL) : {brier:.4f}   (n={len(scored)})  "
              f"NOT comparable to the market figure — see below")
        print(f"  log loss (model)   : {ll:.4f}")

    # ── THE PAIR IS PAIRED. 2026-10-20. ────────────────────────────────────
    #
    # This printed a model Brier over every scored pick beside a market Brier
    # over the subset carrying `market_probability` (populated only after
    # migration 004), labelled "<- the number the model must beat". Measured
    # on the live record that was 0.2466 (n=1534) against 0.2433 (n=479):
    # different populations, presented as a comparison.
    #
    # PAIRED on the 479 they share: model 0.2434 vs market 0.2433, delta
    # +0.0001, t=+0.07. The whole apparent gap was the populations. It does
    # not rescue the model -- t=0.07 is no evidence either way -- but the
    # report was OVERSTATING the deficit, and in the direction nobody checks.
    from src.reporting.experiment_summary import paired_brier
    b = paired_brier(_to_rows(decided))
    if b["state"] == "PAIRED":
        _t = f"t={b['t']:+.2f}" if b["t"] is not None else "t UNEVALUATED"
        print(f"  Brier PAIRED       : model {b['model']:.4f} vs market "
              f"{b['market']:.4f} on the SAME n={b['n_paired']}")
        print(f"    delta            : {b['delta']:+.4f}  "
              f"se {b['se']:.4f}  {_t}  <- the only honest comparison")
        print(f"    not compared     : {b['n_model_only']} picks carry the "
              f"model's probability and NOT the market's")
    else:
        print(f"  Brier PAIRED       : {b['state']} — no pick carries both "
              f"probabilities, so there is nothing to pair")


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


def section_health(picks: List[_Pick]) -> List[str]:
    """Phase 20 — explicit health checks. Returns the ALERT TEXTS.

    It returned a count until 2026-10-21. The count was used only for
    truthiness and for an exit code, and that exit code was discarded by
    `| tee` in the workflow (see `HEALTH_ALERT_EXIT`), so no alert raised here
    had ever reached a person. Returning the texts lets them travel inside the
    delivered summary, which is a channel that works.
    """
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
        return []

    resolved = [p for p in scoped
                if p.kickoff and p.kickoff.date() < date.today()]
    captured = [p for p in resolved if p.closing_status == "captured"]
    if resolved:
        rate = len(captured) / len(resolved)
        msg = (f"closing capture coverage {rate:.2%} "
               f"({len(captured)}/{len(resolved)} past-kickoff picks "
               f"carrying a model_version)")
        _ceiling = CLOSING_COVERAGE_FLOOR + CLOSING_COVERAGE_RISE_BAND
        if rate < CLOSING_COVERAGE_FLOOR:
            alerts.append(
                f"{msg} — FELL below the recorded floor of "
                f"{CLOSING_COVERAGE_FLOOR:.2%} (ratchet, recorded 2026-10-21 "
                f"from {CLOSING_COVERAGE_MEASURED:.2%}). Capture is getting "
                f"worse, which is the one thing this check exists to catch.")
        elif rate >= _ceiling:
            alerts.append(
                f"{msg} — ROSE past {_ceiling:.2%}, more than the "
                f"{CLOSING_COVERAGE_RISE_BAND:.0%} band above the recorded "
                f"floor. RE-RECORD the floor: left alone it drifts upward "
                f"silently and the next fall is measured against a number "
                f"nobody wrote down.")
        else:
            ok.append(f"{msg} — within the ratchet band "
                      f"[{CLOSING_COVERAGE_FLOOR:.2%}, {_ceiling:.2%})")

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
    return alerts


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
    ap.add_argument("--telegram", action="store_true",
                    help="deliver the six-line summary to Telegram on SUCCESS. "
                         "Until 2026-10-20 the only Telegram message this job "
                         "ever sent was its failure alert, so a working report "
                         "was indistinguishable from no report.")
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
        # Same code as the full path, for the same reason: this mode is a GATE
        # somebody may wire up later, and two meanings for exit 1 is how the
        # swallowing went unnoticed in the first place.
        raise SystemExit(HEALTH_ALERT_EXIT if section_health(picks) else 0)

    section_operational(picks)
    section_volume(picks)
    section_attribution_coverage(picks)
    section_pricing(picks)
    section_clv(picks)
    section_outcomes(picks)
    summary = section_experiment_summary(picks, args.days)
    section_calibration(picks)
    section_claude(picks)
    section_checkpoints(picks)
    alerts = section_health(picks)

    if args.telegram:
        from src.reporting.experiment_summary import format_telegram
        body = format_telegram(summary)
        # The health alerts travel WITH the figures. Their exit code was the
        # only channel they had, and it was being swallowed.
        if alerts:
            _heads = "; ".join(a.split(" — ")[0] for a in alerts)
            body += f"{chr(10)}⚠️ {len(alerts)} health alert(s): {_heads[:260]}"
        print()
        print("-" * 88)
        print("TELEGRAM BODY (delivered on success):")
        # The body carries an emoji, as the daily card does. Printing it
        # raised UnicodeEncodeError on a cp1251 console (reproduced
        # 2026-10-21), and a decorative character must not be able to fail
        # the step that produces the report. The DELIVERED text is unaffected:
        # only this echo degrades.
        try:
            print(body)
        except UnicodeEncodeError:
            print(body.encode("ascii", "replace").decode("ascii"))
            print("  (echo transliterated: this console cannot encode the "
                  "body; the delivered message is unchanged)")
        # Delivery failure must not fail the report: the artifact is still
        # written and the figures are still in the log. It is REPORTED, never
        # swallowed -- a silent send is how the broken alert survived months.
        try:
            from scripts.ci_alert import send
            print(f"  delivered: {bool(send(body))}")
        except Exception as e:
            print(f"  NOT DELIVERED: {type(e).__name__}: {e}")

    print()
    if alerts:
        # A GitHub annotation so the finding is visible on the run page even
        # when the job is green, which it now stays for a health alert.
        print(f"::warning::paper-trading-report: {len(alerts)} health "
              f"alert(s) — {'; '.join(alerts)[:400]}")
    raise SystemExit(HEALTH_ALERT_EXIT if alerts else 0)


if __name__ == "__main__":
    main()
