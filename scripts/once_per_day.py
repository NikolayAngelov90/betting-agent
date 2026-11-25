"""One run per day: has today's card already been produced? Stage 34, 2026-11-21.

    python -m scripts.once_per_day            # prints, writes $GITHUB_OUTPUT

WHY. The card is published by an external dispatch at 10:30 Europe/Sofia;
a late fallback cron is the safety net for a missed dispatch. Either can
start `daily-picks`, and GitHub queues a second run behind the first (the
workflow-level concurrency group, never cancelled). Without this the second
run would fetch, settle and send again -- a second card, a second settlement
message. The client must never receive two.

WHAT IT DOES. Reads `pick_runs` for today's date (UTC, the date the run
itself records). If a row there COMPLETED the day -- PICKS_MADE, or the
zero-pick answers NONE_QUALIFIED / NO_FIXTURES -- it records
ALREADY_RAN_TODAY and writes `already_ran=true`; the workflow's main job is
then skipped and nothing is fetched or sent. Otherwise `already_ran=false`.

A RUN THAT FAILED DOES NOT CLOSE THE DAY: INCOMPLETE, NO_PRICED_FIXTURES or
no row at all leave it open, so the next run still runs and sends the card.

FAIL OPEN. Any error writes `already_ran=false`: a lost card is the larger
harm, and `--picks` reads the same table again before it evaluates anything.
"""
from __future__ import annotations

import datetime as _dt
import os
import sys

MARK_ALREADY = "ALREADY RAN TODAY"
MARK_FIRST = "FIRST RUN TODAY"
MARK_LATE = "LATE FALLBACK"


def decide(done) -> bool:
    """Pure: does a completed row close today?"""
    return bool(done)


def _write_output(already: bool) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"already_ran={'true' if already else 'false'}\n")


def main(argv=None, *, db=None, today=None, event=None, now=None) -> int:
    today = today or _dt.date.today()
    try:
        from src.data import pick_runs as pr
        # Stage 35: yesterday's fallback slot, landed after midnight, must not
        # publish today's card ahead of the 10:30 dispatch. Decided before the
        # database is read -- it needs none -- and NOT recorded in
        # `pick_runs`: it is a statement about yesterday's slot, and today
        # has not run.
        if pr.late_fallback(event, now):
            print(f"{MARK_LATE}: this scheduled run started before the "
                  f"fallback's {pr.FALLBACK_CRON_UTC_MIN // 60:02d}:00 UTC "
                  f"slot on {today}, so it is "
                  f"{today - _dt.timedelta(days=1)}'s slot landed after "
                  f"midnight. It does not publish {today}'s card: that is "
                  f"the 10:30 dispatch's. This run fetches, settles, "
                  f"evaluates and sends NOTHING.")
            _write_output(True)
            return 0
        if db is None:
            from src.data.database import get_db
            db = get_db()
        done = pr.completed_today(db, today, strict=True)
        if decide(done):
            mode, version = "standard", ""
            try:
                from src.models.model_version import model_version
                from src.utils.config import Config
                cfg = Config("config/config.example.yaml")   # the deployed file
                mode = str(cfg.get("betting.selection_mode", None)
                           or "standard")
                version = model_version(cfg)
            except Exception:                         # noqa: BLE001
                pass
            pr.record_already_ran(db, pick_date=today, selection_mode=mode,
                                  done=done, model_version=version)
            print(f"{MARK_ALREADY}: {today} was completed by an earlier run "
                  f"({done['outcome']}, recorded {done['recorded_at']}). "
                  f"This run fetches, settles, evaluates and sends NOTHING.")
            _write_output(True)
            return 0
        print(f"{MARK_FIRST}: no completed run for {today}; the run proceeds.")
        _write_output(False)
        return 0
    except Exception as exc:                          # noqa: BLE001
        print(f"::warning::once-per-day check could not read the day "
              f"({type(exc).__name__}); FAILING OPEN -- the run proceeds and "
              f"`--picks` checks again before evaluating.")
        _write_output(False)
        return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
