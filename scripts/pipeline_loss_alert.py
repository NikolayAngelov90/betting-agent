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
        if r.picks > 0:
            break
        if r.card == 0 and r.card_discovered:
            break                       # a real empty card: nothing was lost
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
        lines.append(f"  {r.day}  {card}  picks saved: {r.picks}")
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


def load_recent(days: int = 10, today: Optional[_dt.date] = None
                ) -> Tuple[List[DayLoss], Optional[int], str]:
    """`(rows, earliest_kickoff_min, basis)` from the database.

    Read-only. Returns an empty list rather than raising when the database is
    unreachable, because an alert must not fail on the way to reporting a
    failure.
    """
    today = today or _dt.date.today()
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
        from sqlalchemy import text
    except Exception:                               # pragma: no cover
        return [], None, ""
    rows: List[DayLoss] = []
    earliest: Optional[int] = None
    try:
        with get_db().get_session() as s:
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
                rows.append(DayLoss(d, int(card), int(picks),
                                    card_discovered=bool(card)))
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
    if not streak:
        print("pipeline_loss[0 consecutive zero-pick days with a card]")
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
