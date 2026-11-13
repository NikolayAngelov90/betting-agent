"""The alert that carries the LOSS, not the mechanism's state.

    An alert that reports a mechanism's state does not convey the loss.

MEASURED 2026-10-18. `daily-picks` failed on six consecutive days and sent six
messages. Every one of them read:

    "Daily picks workflow FAILED (run #NNN). Logs: <url>"

It fired correctly, every time, and produced no action. It carried:

    no card size          - 137 fixtures were in the cards for 10-13..10-17
    no irreversibility    - a price is not backfillable; the day is simply gone
    no deadline           - nothing said how long was left to dispatch
    no cumulative total   - the sixth message looked exactly like the first

    A repeated identical alarm is background.

THE STREAK IS MEASURED AS THE LOSS, NOT AS THE MECHANISM. "N consecutive days
with a discovered card and zero picks saved" is the thing that matters, and it
is also the check the alarm surface did not carry: a crash, a margin breach and
a `DID_NOT_RUN` were each alarmed, and *a pipeline producing no picks at all*
was not.

ESCALATION, AND N IS CHOSEN BEFORE MEASURING HOW OFTEN IT FIRES:

    N=1   FULL DETAIL. One failed run is often recovered by the next day's
          run, so the first message informs and does not escalate.
    N=2   "NOT SELF-CORRECTING" + the cumulative loss. Two in a row means the
          next run did not fix it, which is a different fact from the first.
    N=3+  "THE ALERT IS NOT BEING SEEN" + the running total in the subject
          position. Three identical messages is the point at which the channel
          itself becomes a candidate, and every message from here carries a
          different number so it cannot flatten into background.

The reason for those thresholds is the measurement above: six identical
messages produced no action, so the escalation has to begin before the message
has been repeated six times - and not at N=1, which would escalate a
transient.
"""

from __future__ import annotations

import datetime as _dt
import os as _os
from typing import List, NamedTuple, Optional, Tuple

#: Time from run start to picks complete on a FULL card.
#: MEASURED 2026-10-01, n=4: 67.4-88.1 min, mean 76.4, sd 10.2. THE UPPER
#: BOUND, never the mean - a dispatch sized on 76 minutes misses a card that
#: takes 88.
PICKS_UPPER_BOUND_MIN = 88.1

#: A choice, stated beside the measurement rather than folded into it.
DISPATCH_SAFETY_MARGIN_MIN = 15

OPERATOR_TZ = "Europe/Sofia"

#: Thresholds, fixed before the firing rate was measured.
ESCALATE_NOT_SELF_CORRECTING = 2
ESCALATE_CHANNEL_SUSPECT = 3

IRREVERSIBLE = ("Prices DO NOT BACKFILL: a fixture that kicks off without a "
                "captured price is gone, and no later run recovers it.")


class DayLoss(NamedTuple):
    day: _dt.date
    card: int
    picks: int
    #: False when the card is 0 AND the update step never ran, so "no
    #: fixtures" and "we never looked" are not the same row.
    card_discovered: bool = True
    #: Stage 32: what the picks run RECORDED about the day
    #: (`src/data/pick_runs.py`), latest row wins. None = no row, which is a
    #: run that never got far enough to say -- read as a failure.
    outcome: Optional[str] = None
    fixtures_evaluated: Optional[int] = None


# ══════════ THREE STATES, PRINTED DIFFERENTLY. Stage 32 Part A, 2026-11-13 ═══
#
# A day with a card and no pick used to be ONE state: a loss. Under
# `confident_value` the rule has zero-pick days by design (5 of 56 holdout
# days at P_min 0.54), so that day is now three:
#
#   PICKS MADE            at least one pick saved
#   NO FIXTURE QUALIFIED  the run evaluated the whole slate without an error
#                         and nothing met the rule -- or there was nothing to
#                         evaluate. A STATE: it ends a streak, it is not one.
#   PIPELINE FAILED       anything else. No recorded outcome, an INCOMPLETE
#                         evaluation, or fixtures with no price at all (a
#                         broken odds feed looks exactly like that).
#
# The burden of proof sits with "nothing qualified": it needs a row saying
# the evaluation completed. Silence stays a loss.
STATE_PICKS_MADE = "PICKS MADE"
STATE_NONE_QUALIFIED = "NO FIXTURE QUALIFIED"
STATE_PIPELINE_FAILED = "PIPELINE FAILED"
STATES = (STATE_PICKS_MADE, STATE_NONE_QUALIFIED, STATE_PIPELINE_FAILED)

#: Recorded outcomes that mean "evaluated completely, and the answer was
#: nothing". Kept equal to `src.data.pick_runs.EVALUATED_EMPTY` by a test;
#: literal here so this script imports nothing from `src` at load.
EVALUATED_EMPTY = ("NONE_QUALIFIED", "NO_FIXTURES")


def day_state(r: DayLoss) -> str:
    """One of `STATES`. Pure."""
    if r.picks > 0:
        return STATE_PICKS_MADE
    if r.outcome in EVALUATED_EMPTY:
        return STATE_NONE_QUALIFIED
    if r.card == 0 and r.card_discovered:
        return STATE_NONE_QUALIFIED      # a real empty card, as before
    return STATE_PIPELINE_FAILED


def format_day_states(rows: List[DayLoss]) -> str:
    """One line per day, each state worded differently, so a reader never has
    to infer which of the three a zero is."""
    out = []
    for r in rows:
        st = day_state(r)
        if st == STATE_PICKS_MADE:
            detail = f"{r.picks} pick(s) saved from a card of {r.card}"
        elif st == STATE_NONE_QUALIFIED:
            if r.outcome == "NO_FIXTURES" or (r.card == 0 and r.outcome is None):
                detail = "no fixtures to evaluate - not a loss"
            else:
                n = r.fixtures_evaluated
                detail = (f"{'all ' + str(n) if n is not None else 'every'} "
                          f"fixture(s) evaluated, none met the rule - "
                          f"a state, not a loss")
        else:
            why = {None: "no completed evaluation was recorded",
                   "INCOMPLETE": "the evaluation raised on at least one "
                                 "fixture, so 'none qualified' cannot be "
                                 "claimed",
                   "NO_PRICED_FIXTURES": "fixtures existed but none had a "
                                         "bookmaker price"}.get(
                r.outcome, f"recorded outcome {r.outcome}")
            detail = f"card {r.card}, no picks: {why} - A LOSS"
        out.append(f"  {r.day}  {st:<20s}  {detail}")
    return chr(10).join(out)


def _in_tz(day: _dt.date, minutes_utc: int, tz_name: str = OPERATOR_TZ):
    """`(HH:MM, label)` in `tz_name`, or `(None, reason)`. Every time names
    its clock."""
    try:
        from zoneinfo import ZoneInfo
        z = ZoneInfo(tz_name)
    except Exception as exc:                        # noqa: BLE001
        return None, f"{tz_name} unavailable ({type(exc).__name__})"
    aware = _dt.datetime.combine(
        day, _dt.time(minutes_utc // 60, minutes_utc % 60),
        tzinfo=_dt.timezone.utc).astimezone(z)
    return aware.strftime("%H:%M"), aware.strftime("%Z%z")


def zero_pick_streak(rows: List[DayLoss]) -> List[DayLoss]:
    """The trailing run of days that had a card and produced no picks.

    A day with no card is NOT a loss and breaks nothing: it ends the streak
    only if it genuinely had no fixtures. A day whose card was never
    discovered is counted, because not looking is the failure.
    """
    out: List[DayLoss] = []
    for r in reversed(rows):
        # Stage 32: a day is in the streak only if its state is a FAILURE.
        # PICKS MADE and NO FIXTURE QUALIFIED both end it -- the second
        # includes the old "real empty card" case.
        if day_state(r) != STATE_PIPELINE_FAILED:
            break
        out.append(r)
    return list(reversed(out))


def format_loss_alert(streak: List[DayLoss],
                      earliest_kickoff_min: Optional[int] = None,
                      deadline_basis: str = "",
                      run_url: str = "",
                      now: Optional[_dt.datetime] = None) -> str:
    """The body. Escalates by the length of `streak`."""
    now = now or _dt.datetime.now(_dt.timezone.utc)
    n = len(streak)
    today = streak[-1].day if streak else now.date()
    fixtures = sum(r.card for r in streak)
    undiscovered = [r for r in streak if not r.card_discovered]

    if n >= ESCALATE_CHANNEL_SUSPECT:
        head = (f"!!! DAY {n} OF ZERO PICKS - THIS ALERT IS NOT BEING SEEN. "
                f"{fixtures} fixtures lost so far.")
    elif n >= ESCALATE_NOT_SELF_CORRECTING:
        # "produced no picks either", NOT "did not fix it". 2026-10-22: this
        # is the one clause in the whole body that spoke about a RUN rather
        # than about the loss, and the distinction is the point of the file --
        # whether the next run succeeded or failed is not what escalates this,
        # a SECOND DAY WITH NO PICKS is.
        head = (f"!! DAY {n} OF ZERO PICKS - NOT SELF-CORRECTING. The next "
                f"scheduled run produced no picks either. {fixtures} "
                f"fixtures lost so far.")
    else:
        head = f"NO PICKS PRODUCED for {today}."

    lines = [head, "", IRREVERSIBLE, ""]
    lines.append("THE LOSS, per day:")
    for r in streak:
        card = (f"{r.card} fixtures" if r.card_discovered
                else f"card NEVER DISCOVERED (the update step did not run)")
        lines.append(f"  {r.day}  {card}  picks saved: {r.picks}  "
                     f"[{day_state(r)}: recorded outcome "
                     f"{r.outcome or 'NONE - the run never said'}]")
    if undiscovered:
        lines.append(f"  {len(undiscovered)} day(s) have NO card size because "
                     f"nothing looked - that is not the same as no fixtures.")

    lines.append("")
    if earliest_kickoff_min is not None:
        hhmm = (f"{earliest_kickoff_min // 60:02d}:"
                f"{earliest_kickoff_min % 60:02d}")
        local, label = _in_tz(today, earliest_kickoff_min)
        last = int(earliest_kickoff_min - PICKS_UPPER_BOUND_MIN
                   - DISPATCH_SAFETY_MARGIN_MIN)
        day_before = last < 0
        last %= 24 * 60
        l_hhmm = f"{last // 60:02d}:{last % 60:02d}"
        l_local, l_label = _in_tz(today, last)
        remaining = (_dt.datetime.combine(
            today, _dt.time(last // 60, last % 60),
            tzinfo=_dt.timezone.utc) - now).total_seconds() / 60.0
        lines += [
            f"EARLIEST KICKOFF   {hhmm} UTC"
            + (f" / {local} {label}" if local else f" / {label}")
            + (f"   ({deadline_basis})" if deadline_basis else ""),
            f"LAST SAFE DISPATCH {l_hhmm} UTC"
            + (f" / {l_local} {l_label}" if l_local else "")
            + (" (THE DAY BEFORE)" if day_before else ""),
            # RULE 3: a figure from a RETIRED REGIME is labelled WHEREVER it
            # appears, this message included. `ci_audit` already says so for
            # the same constant; this said only "n=4", which reads as a small
            # sample rather than as a different world.
            f"  = kickoff - {PICKS_UPPER_BOUND_MIN:.0f}m time-to-picks "
            f"(MEASURED upper bound, n=4, 67.4-88.1, NOT the mean - and from "
            f"the RETIRED SERIAL review regime, so this deadline is EARLY "
            f"rather than late: the parallel review's first observation is "
            f"35.1m) - {DISPATCH_SAFETY_MARGIN_MIN}m safety margin",
            f"TIME REMAINING     {remaining:+.0f} min"
            + ("  <-- ALREADY PAST" if remaining < 0 else ""),
        ]
    else:
        lines.append("EARLIEST KICKOFF   UNKNOWN - the card was not "
                     "discovered, so the deadline cannot be computed. That is "
                     "not a safe deadline; it is no deadline.")

    lines += [
        "",
        "ACTION, and it is the OPERATOR'S - there is no automated "
        "alternative today:",
        "    gh workflow run daily-picks.yml",
        # ── A STALE LITERAL, CORRECTED 2026-10-22. ───────────────────────
        #
        # It read "3h58m-7h38m". `7h38m` is 458 minutes: the MAX under the
        # weekday `0 4` cron, measured 2026-10-18 (n=5, mean 357.5). THAT
        # CRON WAS REVERTED THE SAME DAY and this repository has run
        # `0 0 * * 1-5` ever since, so the only Telegram message a reader
        # acts on was quoting the worst case of a schedule that no longer
        # exists. Nine other sites in the repository say `3h58m-5h04m`;
        # this one said something else and nothing compared them.
        #
        # RE-MEASURED 2026-10-22 over the live weekday `0 0` series, n=10:
        # 128.4 to 304.9 minutes. The range below is that, not the
        # 2026-10-18 figure -- and the nine `3h58m-5h04m` sites are
        # themselves now stale at the LOW end, because they predate the
        # 10-19..10-22 cluster. Named here rather than edited in nine
        # places from inside a Telegram change.
        # ── PER CLUSTER, 2026-10-22. ────────────────────────────────────
        #
        # The pooled range 2h08m-5h05m spans a BIMODAL series (largest gap /
        # median gap = 17.33 on the weekday series), so quoting it as one
        # range tells the operator a number that describes neither half. The
        # two clusters are given, with the slow one named as the figure a
        # dispatch is insuring against.
        "A dispatched run skips the scheduler's lateness. MEASURED "
        "2026-10-22 over the weekday series, which is BIMODAL under one "
        "unchanged cron: SLOW cluster n=6, 4h16m-5h05m (this is what a "
        "dispatch insures against); fast cluster n=4, 2h08m-2h24m.",
    ]
    if run_url:
        lines.append(f"Logs: {run_url}")
    return chr(10).join(lines)


def load_recent(days: int = 10, today: Optional[_dt.date] = None,
                session=None) -> Tuple[List[DayLoss], Optional[int], str]:
    """`(rows, earliest_kickoff_min, basis)` from the database.

    Read-only. Returns an empty list rather than raising when the database is
    unreachable, because an alert must not fail on the way to reporting a
    failure.
    """
    today = today or _dt.date.today()
    # `session`: a caller-owned session (a test's scratch SQLite). When one
    # is given NOTHING is loaded from `.env` and `get_db()` is never called,
    # so a test of this reader cannot reach the production database.
    if session is not None:
        return _read_days(session, days, today)
    # LOADED INSIDE THE FUNCTION, never at import. A module-level
    # `load_dotenv()` undoes conftest's DATABASE_URL strip and has previously
    # let a test suite reach production. Without it `get_db()` silently falls
    # back to SQLite, which holds no card - MEASURED on this module's first
    # run, which reported every day as "card NEVER DISCOVERED" while the
    # database held 101 fixtures for 10-17.
    import re as _re
    for _p in (".env", ".env.local"):
        try:
            for _line in open(_p, encoding="utf-8"):
                _m = _re.match('\\s*([A-Z0-9_]+)\\s*=\\s*(.*)', _line)
                if _m and not _os.environ.get(_m.group(1)):
                    _os.environ[_m.group(1)] = (
                        _m.group(2).strip().strip('"').strip("'"))
        except FileNotFoundError:
            continue
    try:
        from src.data.database import get_db
    except Exception:                               # pragma: no cover
        return [], None, ""
    try:
        with get_db().get_session() as s:
            return _read_days(s, days, today)
    except Exception as exc:                        # noqa: BLE001
        print(f"loss alert: database unreadable ({type(exc).__name__}) - "
              f"reporting without card sizes")
        return [], None, ""


def _read_days(s, days: int, today: _dt.date
               ) -> Tuple[List[DayLoss], Optional[int], str]:
    """The reads, on a session the caller owns."""
    from sqlalchemy import text
    rows: List[DayLoss] = []
    earliest: Optional[int] = None
    try:
        # Stage 32: the latest recorded outcome per day. A database that
        # predates the table reads as NO ROW for every day -- which is
        # the conservative reading (every zero-pick day stays a loss),
        # not an error.
        outcomes = {}
        try:
            for d_, oc, fe in s.execute(text(
                    "SELECT pick_date, outcome, fixtures_evaluated "
                    "FROM pick_runs WHERE pick_date >= :a "
                    "ORDER BY recorded_at"),
                    {"a": (today - _dt.timedelta(days=days)).isoformat()}):
                key = d_ if isinstance(d_, _dt.date) else \
                    _dt.date.fromisoformat(str(d_)[:10])
                outcomes[key] = (oc, fe)
        except Exception:                       # noqa: BLE001
            s.rollback()
            outcomes = {}
        for back in range(days, -1, -1):
            d = today - _dt.timedelta(days=back)
            nxt = d + _dt.timedelta(days=1)
            card = s.execute(text(
                "SELECT count(*) FROM matches "
                "WHERE match_date >= :a AND match_date < :b"),
                {"a": d.isoformat(), "b": nxt.isoformat()}).scalar() or 0
            picks = s.execute(text(
                "SELECT count(*) FROM saved_picks "
                "WHERE created_at >= :a AND created_at < :b"),
                {"a": d.isoformat(), "b": nxt.isoformat()}).scalar() or 0
            oc, fe = outcomes.get(d, (None, None))
            rows.append(DayLoss(d, int(card), int(picks),
                                card_discovered=bool(card),
                                outcome=oc, fixtures_evaluated=fe))
        ko = s.execute(text(
            "SELECT min(match_date) FROM matches "
            "WHERE match_date >= :a AND match_date < :b"),
            {"a": today.isoformat(),
             "b": (today + _dt.timedelta(days=1)).isoformat()}).scalar()
        if ko is not None:
            from scripts.schedule_deadline import as_datetime
            ko = as_datetime(ko)
            if ko is not None and not ko.microsecond:
                earliest = ko.hour * 60 + ko.minute
    except Exception as exc:                        # noqa: BLE001
        print(f"loss alert: database unreadable ({type(exc).__name__}) - "
              f"reporting without card sizes")
        return rows, None, ""
    return rows, earliest, ("discovered from the stored card"
                            if earliest is not None else "")


def main() -> int:
    rows, earliest, basis = load_recent()
    streak = zero_pick_streak(rows)
    # The three states, every run, so a zero is never left to inference.
    print("pipeline_loss days (latest recorded outcome per day):")
    print(format_day_states(rows[-7:]))
    if not streak:
        print("pipeline_loss[0 consecutive PIPELINE FAILED days]")
        return 0
    body = format_loss_alert(
        streak, earliest, basis,
        run_url=_os.environ.get("LOSS_RUN_URL", ""))
    print(body)
    path = _os.environ.get("LOSS_ALERT_BODY")
    if path:
        try:
            import pathlib
            pathlib.Path(path).write_text(body, encoding="utf-8")
        except OSError as exc:                      # pragma: no cover
            print(f"could not write the alert body: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
