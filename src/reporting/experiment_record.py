"""The paper-trading record, for REPORTING only.

WHY THIS EXISTS RATHER THAN A CHANGE TO `live_only()`. That predicate gates the
LEARNERS as well as the report. Widening it to admit paper picks would let the
frozen experiment retrain its own subject — which is the whole reason paper
picks are marked. **Reporting a paper outcome is not the same as learning from
one**, and Stage 14 already drew that line when it left `get_stats` un-gated for
`valid_evidence` because the wagers were real.

So this is a SEPARATE series. `live_only()` is not touched and must not be.

WHAT IT REPLACES. `get_stats()["all_time"]` reads the pre-paper-trading LIVE
record: **1,074 settled picks at 51.676%, last pick 2026-08-10.** It froze when
paper trading began and was reported daily as though it were current. It now
lives in the README with its measurement date, and the daily message counts from
the experiment instead.

THREE THINGS THE OLD BLOCK DID NOT DO, and each is a correction to a specific
way the number misled:

  1. **THE COHORT SPAN IS NAMED.** The paper record crosses six `model_version`
     fingerprints. Pooling them is precisely what `model_version` exists to
     prevent — a single win rate across six configurations measures none of
     them. The message prints `n across k cohorts` so nobody reads it as one
     system's record.
  2. **EVERY RATE CARRIES ITS n**, and the CURRENT cohort's n is printed
     separately, with a caveat DERIVED from that n rather than asserted. The
     first version hardcoded "the current cohort's n is small enough that its
     rate is noise"; four days later n was 262 and the sentence was false. A
     claim that stops following from the data beside it is the same defect as
     the frozen all-time this block replaced.
  3. **CLV LEADS, THE RECORD FOLLOWS.** Stage 16 established that win-rate and
     ROI segments are all p > 0.15 and that CLV is the instrument this
     experiment turns on. A message headlining win rate invites exactly the
     reasoning four audits have corrected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger()


@dataclass
class Series:
    """One CLV attribution series: n, mean, interval, and its cluster count."""

    label: str
    n: int = 0
    fixtures: int = 0
    mean: Optional[float] = None
    lo: Optional[float] = None
    hi: Optional[float] = None

    @property
    def has_interval(self) -> bool:
        return self.lo is not None and self.hi is not None


@dataclass
class ExperimentRecord:
    """The paper record: CLV first, the settled record second, both with n."""

    model: Series = field(default_factory=lambda: Series("MODEL"))
    final: Series = field(default_factory=lambda: Series("FINAL"))
    settled: int = 0
    wins: int = 0
    losses: int = 0
    cohorts: int = 0
    first_date: Optional[object] = None
    last_date: Optional[object] = None
    current_cohort: str = ""
    current_n: int = 0
    current_wins: int = 0
    current_losses: int = 0

    @property
    def win_rate(self) -> Optional[float]:
        d = self.wins + self.losses
        return self.wins / d if d else None

    @property
    def current_win_rate(self) -> Optional[float]:
        d = self.current_wins + self.current_losses
        return self.current_wins / d if d else None


def build(db, current_model_version: str = "") -> Optional[ExperimentRecord]:
    """Compute the paper record. Returns None if it cannot be measured.

    None is NOT an empty record: the caller must be able to tell "the experiment
    has no settled picks yet" from "this could not be computed", and print
    accordingly. The same distinction `coverage_checks` draws between `[]` and
    `None`.
    """
    try:
        from sqlalchemy import text
        rec = ExperimentRecord(current_cohort=current_model_version or "")

        with db.get_session() as s:
            row = s.execute(text("""
                SELECT count(*) FILTER (WHERE result IS NOT NULL) settled,
                       count(*) FILTER (WHERE result = 'win')  w,
                       count(*) FILTER (WHERE result = 'loss') l,
                       count(DISTINCT model_version)           cohorts,
                       min(pick_date), max(pick_date)
                FROM saved_picks
                WHERE is_paper IS TRUE AND disposition IS NULL
            """)).fetchone()
            if not row or not row[0]:
                return None
            (rec.settled, rec.wins, rec.losses, rec.cohorts,
             rec.first_date, rec.last_date) = row

            if current_model_version:
                cur = s.execute(text("""
                    SELECT count(*) FILTER (WHERE result IS NOT NULL),
                           count(*) FILTER (WHERE result = 'win'),
                           count(*) FILTER (WHERE result = 'loss')
                    FROM saved_picks
                    WHERE is_paper IS TRUE AND disposition IS NULL
                      AND model_version = :mv
                """), {"mv": current_model_version}).fetchone()
                if cur:
                    rec.current_n, rec.current_wins, rec.current_losses = cur

            # CLV, per attribution series. Clustered by FIXTURE, because two
            # picks on one match respond to the same information — the design
            # effect Stage 8 measured and Stage 16 confirmed at deff = 1.00.
            obs = s.execute(text("""
                SELECT po.attribution, po.taken_odds, po.closing_odds, sp.match_id
                FROM pick_observations po
                JOIN saved_picks sp ON sp.id = po.pick_id
                WHERE po.closing_odds IS NOT NULL
                  AND po.taken_odds IS NOT NULL
                  AND po.taken_odds > 1.0 AND po.closing_odds > 1.0
                  AND sp.is_paper IS TRUE AND sp.disposition IS NULL
            """)).fetchall()

        by: Dict[str, List] = {}
        for attribution, taken, closing, match_id in obs:
            # Price CLV, the same quantity `clv.compute` returns: how much
            # better the taken price was than the close.
            by.setdefault(attribution, []).append(
                (float(taken) / float(closing) - 1.0, match_id))

        from src.evaluation.clv import _boot, _effective_n
        for key, series in (("model", rec.model), ("final", rec.final)):
            pairs = by.get(key, [])
            if not pairs:
                continue
            vals = [v for v, _ in pairs]
            clusters = [c for _, c in pairs]
            series.n = len(vals)
            series.n, series.fixtures, _, _ = _effective_n(clusters)
            series.mean = sum(vals) / len(vals)
            try:
                series.lo, series.hi = _boot(vals, clusters)
            except Exception:
                series.lo = series.hi = None

        # ANNOUNCES ITSELF UNCONDITIONALLY, including when it finds little.
        # A block that logs only when it has something to say is
        # indistinguishable from a block that stopped running — the defect this
        # ledger has now catalogued five times. `ci_audit` greps this line.
        logger.info(
            f"EXPERIMENT RECORD: n={rec.settled} cohorts={rec.cohorts} "
            f"current={rec.current_cohort or '?'}({rec.current_n}) "
            f"clv_model_n={rec.model.n} clv_final_n={rec.final.n}")
        return rec
    except Exception as exc:
        logger.warning(
            f"EXPERIMENT RECORD could not be built ({exc}) — the report will "
            "say so rather than omitting the block silently.")
        return None


def format_block(rec: Optional[ExperimentRecord], html: bool = True) -> List[str]:
    """Render the block. CLV first, the settled record below it as context."""
    b = (lambda t: f"<b>{t}</b>") if html else (lambda t: t)
    i = (lambda t: f"<i>{t}</i>") if html else (lambda t: t)

    if rec is None:
        return [f"\n{b('─── Experiment ───')}",
                i("record unavailable — not computed this run, which is not the "
                  "same as empty")]

    out = [f"\n{b('─── Experiment: CLV ───')}"]
    for s in (rec.model, rec.final):
        if not s.n:
            out.append(f"{s.label}: no closing prices captured yet")
            continue
        ci = (f" 95% CI [{s.lo:+.2%}, {s.hi:+.2%}]" if s.has_interval
              else "  (CI needs ≥5 fixtures)")
        out.append(f"{b(s.label)}: {s.mean:+.2%} "
                   f"(n={s.n} picks / {s.fixtures} fixtures){ci}")
    out.append(i("CLV is the instrument this experiment turns on — Stage 16: "
                 "win-rate and ROI segments are all p &gt; 0.15."))

    out.append(f"\n{b('─── Experiment: settled record ───')}")
    if rec.win_rate is None:
        out.append("no settled paper picks yet")
        return out

    span = ""
    if rec.first_date and rec.last_date:
        span = f", {rec.first_date} → {rec.last_date}"
    out.append(
        f"{rec.wins}W-{rec.losses}L ({rec.win_rate:.1%}) "
        f"across n={rec.settled} picks in {rec.cohorts} cohorts{span}")
    if rec.current_n:
        cwr = (f"{rec.current_win_rate:.1%}" if rec.current_win_rate is not None
               else "n/a")
        out.append(
            f"current cohort {rec.current_cohort[-6:] or '?'}: "
            f"{rec.current_wins}W-{rec.current_losses}L ({cwr}) n={rec.current_n}")
    # THE CAVEAT IS DERIVED, NOT ASSERTED.
    #
    # It first read "the current cohort's n is small enough that its rate is
    # noise" as a fixed sentence. Four days later n was 262 and the sentence was
    # simply false — a claim that had stopped following from the data it sat
    # beside, which is the same defect as the frozen all-time this block
    # replaced. The threshold is Stage 16's: 17 observations suffice to exclude
    # a decision-relevant effect, so below that a rate genuinely is noise.
    note = (f"{rec.cohorts} cohorts are {rec.cohorts} different configurations, "
            "and a single rate across them measures none of them.")
    if rec.current_n and rec.current_n < 17:
        note += (f" The current cohort holds n={rec.current_n}; that rate is "
                 "noise and should be read as noise.")
    elif rec.current_n:
        note += (f" The current cohort's n={rec.current_n} is the only one of "
                 "these figures that describes the configuration running now.")
    out.append(i(note))
    return out
