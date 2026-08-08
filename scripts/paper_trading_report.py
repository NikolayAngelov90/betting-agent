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
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np

from src.data.models import Match, SavedPick
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

    __slots__ = ("id", "pick_date", "league", "market", "selection", "odds",
                 "prob", "market_prob", "market_books", "ev", "result",
                 "closing_odds", "closing_odds_captured_at", "closing_status",
                 "closing_fair", "kickoff", "is_paper", "model_version",
                 "review_action", "model_selection", "model_result")

    def __init__(self, r):
        for f in self.__slots__:
            setattr(self, f, None)
        self.id = r.id
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


def _boot(values: List[float], iters: int = 4000, seed: int = 0):
    if len(values) < 5:
        return None, None
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(arr, len(arr), replace=True).mean()
                      for _ in range(iters)])
    return tuple(np.percentile(means, [2.5, 97.5]))


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
            SavedPick.id, SavedPick.pick_date, SavedPick.league, SavedPick.market,
            SavedPick.selection, SavedPick.odds, SavedPick.predicted_probability,
            SavedPick.market_probability, SavedPick.market_books,
            SavedPick.expected_value, SavedPick.result, SavedPick.closing_odds,
            SavedPick.closing_odds_captured_at, SavedPick.closing_capture_status,
            SavedPick.closing_fair_probability, SavedPick.is_paper,
            SavedPick.model_version, SavedPick.review_action,
            SavedPick.model_selection, SavedPick.model_result,
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
    return picks


# ─────────────────────────────────────────────────────────────────── sections

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


def section_clv(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("CLOSING LINE VALUE")
    print("=" * 88)
    cov = coverage_report(picks)
    print(cov.render())

    valid = [p for p in picks
             if p.closing_status == "captured" and p.closing_odds and p.odds]
    if not valid:
        print("\n  No genuine CLV yet. Nothing below can be reported until "
              "closing prices exist.")
        print("  NOTE: model probability minus 1/odds is NOT CLV and is not "
              "shown here.")
        return

    clvs = [p.odds / p.closing_odds - 1 for p in valid]
    lo, hi = _boot(clvs)
    print(f"\n  average CLV : {np.mean(clvs):+.3%}  95% CI {_fmt_ci(lo, hi)}")
    print(f"  median CLV  : {np.median(clvs):+.3%}")
    print(f"  positive CLV: {sum(1 for c in clvs if c > 0) / len(clvs):.1%}")

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


def section_outcomes(picks: List[_Pick]) -> None:
    print("\n" + "=" * 88)
    print("OUTCOMES")
    print("=" * 88)
    decided = [p for p in picks if p.decided]
    if not decided:
        print("  (nothing settled yet)")
        return
    profits = [p.profit for p in decided]
    lo, hi = _boot(profits)
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
        lo, hi = _boot([p.profit for p in sub])
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
    valid = sum(1 for p in picks
                if p.closing_status == "captured" and p.closing_odds and p.odds)
    print(f"  valid closing-line picks: {valid}")
    for target, purpose in CHECKPOINTS:
        status = "REACHED" if valid >= target else f"{target - valid} to go"
        print(f"      {target:>4} — {purpose:<44} {status}")
    if valid < CHECKPOINTS[0][0]:
        print("\n  Below the first checkpoint. No model decision may be taken "
              "from this data.")


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

    section_volume(picks)
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
