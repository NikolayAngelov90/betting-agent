"""What a picks run concluded about its day. Stage 32 Part A, 2026-11-13.

WHY THIS EXISTS. The loss alert counted every day with a card and no saved
pick as a LOSS. Under `confident_value` that is wrong on purpose: the rule
has zero-pick days by design -- 5 of 56 holdout days at P_min 0.54, all of
them days with fewer than ten fixtures. A day the rule evaluated and on which
nothing qualified is a STATE, not a failure, and the only way to tell it from
a pipeline that never got as far as evaluating is for the run to SAY so.

So every picks run writes one row: what it evaluated and what it concluded.

    PICKS_MADE          at least one pick saved
    NONE_QUALIFIED      every fixture on the slate was evaluated, without an
                        error, and none met the rule
    NO_FIXTURES         there was nothing to evaluate
    NO_PRICED_FIXTURES  fixtures existed and none had a bookmaker price. NOT
                        "none qualified": an odds feed that broke looks
                        exactly like this, so it stays a loss for the alert
    INCOMPLETE          analysis raised on at least one fixture and no pick
                        was saved -- "none qualified" cannot be claimed about
                        fixtures that were never evaluated

THE ABSENCE OF A ROW IS NOT A STATE either way. A run that crashed before it
could write is exactly the day the alert exists for, and the loss alert reads
a missing row as PIPELINE_FAILED.

WRITING NEVER RAISES. The record describes the run; it must not be able to
end it.
"""
from __future__ import annotations

from typing import Optional

PICKS_MADE = "PICKS_MADE"
NONE_QUALIFIED = "NONE_QUALIFIED"
NO_FIXTURES = "NO_FIXTURES"
NO_PRICED_FIXTURES = "NO_PRICED_FIXTURES"
INCOMPLETE = "INCOMPLETE"
OUTCOMES = (PICKS_MADE, NONE_QUALIFIED, NO_FIXTURES, NO_PRICED_FIXTURES,
            INCOMPLETE)

#: The outcomes that are a valid answer with no pick: the day was looked at,
#: completely, and the answer was "nothing". Everything else is either a pick
#: or a reason the alert should look.
EVALUATED_EMPTY = (NONE_QUALIFIED, NO_FIXTURES)

#: The stages `get_daily_picks` reports on `agent.last_pick_run`.
STAGE_ALREADY_GENERATED = "ALREADY_GENERATED"
STAGE_NO_FIXTURES = "NO_FIXTURES"
STAGE_NO_PRICED_FIXTURES = "NO_PRICED_FIXTURES"
STAGE_EVALUATED = "EVALUATED"
STAGE_SHARD = "SHARD"


def classify(stage: Optional[dict], picks: int) -> Optional[str]:
    """The outcome to record, or None when this run concluded nothing about
    the day (an idempotent skip, or one shard of several).

    Pure. `stage` is `agent.last_pick_run`: `{"stage", "fixtures",
    "errors"}`. A missing or unrecognised stage cannot support "none
    qualified", so it is INCOMPLETE -- never NONE_QUALIFIED by default.
    """
    if picks > 0:
        return PICKS_MADE
    st = (stage or {}).get("stage")
    if st in (STAGE_ALREADY_GENERATED, STAGE_SHARD):
        return None
    if st == STAGE_NO_FIXTURES:
        return NO_FIXTURES
    if st == STAGE_NO_PRICED_FIXTURES:
        return NO_PRICED_FIXTURES
    if st == STAGE_EVALUATED:
        fixtures = stage.get("fixtures")
        errors = stage.get("errors", 0)
        if fixtures is None:
            return INCOMPLETE
        if fixtures == 0:
            return NO_FIXTURES
        if errors == 0:
            return NONE_QUALIFIED
    return INCOMPLETE


def record(db, *, pick_date, selection_mode: str, outcome: str,
           fixtures_on_slate: Optional[int], fixtures_evaluated: Optional[int],
           analysis_errors: int, picks_saved: int,
           model_version: str = "") -> bool:
    """Append one row. Returns whether it was written. NEVER raises."""
    if outcome not in OUTCOMES:
        return False
    try:
        from src.data.models import PickRun
        with db.get_session() as s:
            s.add(PickRun(
                pick_date=pick_date, selection_mode=str(selection_mode),
                outcome=outcome,
                fixtures_on_slate=(None if fixtures_on_slate is None
                                   else int(fixtures_on_slate)),
                fixtures_evaluated=(None if fixtures_evaluated is None
                                    else int(fixtures_evaluated)),
                analysis_errors=int(analysis_errors),
                picks_saved=int(picks_saved),
                model_version=str(model_version or "")))
            s.commit()
        return True
    except Exception as exc:                          # noqa: BLE001
        try:
            from src.utils.logger import get_logger
            get_logger().warning(f"pick run not recorded: {exc}")
        except Exception:                             # pragma: no cover
            pass
        return False
