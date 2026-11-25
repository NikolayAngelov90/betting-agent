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
#: Stage 34 (2026-11-21): a run that found the day already done and stopped
#: without evaluating or sending anything. A RECORD OF A RUN, NOT AN ANSWER
#: ABOUT THE DAY -- see `NOT_A_DAY_OUTCOME`.
ALREADY_RAN_TODAY = "ALREADY_RAN_TODAY"
OUTCOMES = (PICKS_MADE, NONE_QUALIFIED, NO_FIXTURES, NO_PRICED_FIXTURES,
            INCOMPLETE, ALREADY_RAN_TODAY)

#: The outcomes that are a valid answer with no pick: the day was looked at,
#: completely, and the answer was "nothing". Everything else is either a pick
#: or a reason the alert should look.
EVALUATED_EMPTY = (NONE_QUALIFIED, NO_FIXTURES)

# ── ONE RUN PER DAY. Stage 34, 2026-11-21. ───────────────────────────────────
#
# The card is published by an external dispatch at 10:30 Europe/Sofia, with
# a late fallback cron as a safety net. Either can start a run, and GitHub
# may queue two. The client must never receive a second card, so a run that
# finds the day COMPLETED stops before it fetches, settles, evaluates or
# sends anything, and records `ALREADY_RAN_TODAY`.
#
#: The outcomes that mean the day's client message was PRODUCED: the card, or
#: the one plain zero-pick line. NO_PRICED_FIXTURES and INCOMPLETE are not in
#: it -- a run that failed leaves the day open, so the next run still runs.
COMPLETED = (PICKS_MADE,) + EVALUATED_EMPTY
#: Rows that describe a RUN, not the DAY. Every reader that takes "the latest
#: row per day" as the day's outcome must skip these, or a later no-op run
#: would overwrite the day's real answer.
NOT_A_DAY_OUTCOME = (ALREADY_RAN_TODAY,)

#: The fallback cron's nominal minute, UTC: `0 11 * * *` in `daily-picks.yml`
#: since Stage 34. Pinned to the workflow by `tests/test_once_per_day.py`.
FALLBACK_CRON_UTC_MIN = 11 * 60


def _utcnow():
    """The wall clock, UTC. The ONE clock read of `late_fallback`, here so a
    test sets it instead of inheriting the machine's."""
    import datetime as _dt
    return _dt.datetime.now(_dt.timezone.utc)


def late_fallback(event: Optional[str] = None, now=None) -> bool:
    """Is this run YESTERDAY's fallback slot, landed after midnight?

    Stage 35, 2026-11-25. GitHub never starts a scheduled run early, so a
    `schedule` run that starts before the fallback's nominal minute on its own
    UTC date is the PREVIOUS day's slot. Left alone it reads the NEW date as
    open -- the 10:30 dispatch has not happened yet -- and publishes that day's
    card at one in the morning, on prices eight hours older than the
    dispatch's, after which the dispatch finds the day done and stops. The
    fallback's lateness reached 684 min on 2026-11-23 (22:24 UTC) and two
    sibling workflows' 700 min; midnight is 780 min after 11:00.

    A dispatch is never late in this sense -- someone or something chose its
    time -- so only `schedule` is tested. `event` defaults to the
    `GITHUB_EVENT_NAME` GitHub sets on every step; `now` to `_utcnow()`.
    """
    import os
    event = os.environ.get("GITHUB_EVENT_NAME", "") if event is None else event
    if event != "schedule":
        return False
    now = now or _utcnow()
    return now.hour * 60 + now.minute < FALLBACK_CRON_UTC_MIN


def completed_today(db, pick_date, strict: bool = False) -> Optional[dict]:
    """The row that completed `pick_date`, or None. Never raises.

    STATED LIMIT: the row is written BEFORE the card is sent, so a first run
    that recorded its outcome and then failed to DELIVER leaves the day
    "completed" and a second run will not resend it. That case is what
    `--resend` exists for; a second, different card is what this prevents.
    On a database error this returns None -- FAIL OPEN, because a lost card
    is the larger harm and the in-run guard reads the same table again.
    `strict=True` RAISES instead, for a caller that must SAY it could not
    read: returning None there would print "no completed run" about a
    table nobody read -- an unreadable answer reported as an empty one.
    """
    try:
        from src.data.models import PickRun
        with db.get_session() as s:
            row = (s.query(PickRun)
                   .filter(PickRun.pick_date == pick_date,
                           PickRun.outcome.in_(COMPLETED))
                   .order_by(PickRun.recorded_at)
                   .first())
            # The OUTCOME decides, not the row's existence: anything that is
            # not literally a completed outcome -- including a stand-in
            # object -- is not proof the day is done.
            if row is None or getattr(row, "outcome", None) not in COMPLETED:
                return None
            return {"outcome": row.outcome,
                    "recorded_at": str(row.recorded_at),
                    "picks_saved": row.picks_saved,
                    "model_version": row.model_version}
    except Exception as exc:                          # noqa: BLE001
        if strict:
            raise
        try:
            from src.utils.logger import get_logger
            get_logger().warning(f"completed_today not readable: {exc}")
        except Exception:                             # pragma: no cover
            pass
        return None


def record_already_ran(db, *, pick_date, selection_mode: str, done: dict,
                       model_version: str = "") -> bool:
    """Record that this run found the day done and stopped. NEVER raises."""
    return record(db, pick_date=pick_date, selection_mode=selection_mode,
                  outcome=ALREADY_RAN_TODAY, fixtures_on_slate=None,
                  fixtures_evaluated=None, analysis_errors=0, picks_saved=0,
                  model_version=model_version)

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
