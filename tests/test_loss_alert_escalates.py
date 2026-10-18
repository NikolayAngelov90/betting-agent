"""The alert that carries the loss, and escalates — 2026-10-18.

    An alert that reports a mechanism's state does not convey the loss.

MEASURED. `daily-picks` failed on six consecutive days and sent six messages,
each reading:

    "Daily picks workflow FAILED (run #NNN). Logs: <url>"

Every one fired correctly and produced no action. They carried:

    no card size          137 fixtures were in the cards for 10-13..10-17
    no irreversibility    a price is not backfillable; the day is simply gone
    no deadline           nothing said how long was left to dispatch
    no running total      the sixth looked exactly like the first

    A repeated identical alarm is background.

N WAS CHOSEN BEFORE THE FIRING RATE WAS MEASURED, and the reason is that
measurement: six identical messages produced no action, so escalation has to
begin well before the sixth — and not at the first, which would escalate a
transient that the next scheduled run often fixes.

    N=1    full detail, no escalation
    N=2    "NOT SELF-CORRECTING" + cumulative loss  (the next run did not fix
           it, which is a DIFFERENT FACT from the first failure)
    N=3+   "THIS ALERT IS NOT BEING SEEN" + the running number in the subject
           position, so no two messages are ever identical
"""

import datetime as dt

import pytest

import scripts.pipeline_loss_alert as loss

NOW = dt.datetime(2026, 10, 18, 8, 0, tzinfo=dt.timezone.utc)
KICKOFF = 10 * 60 + 15                      # 10:15 UTC, today's real earliest


def _streak(n: int, card: int = 20):
    """`n` consecutive days with a card and no picks, ending today."""
    return [loss.DayLoss(dt.date(2026, 10, 18) - dt.timedelta(days=n - 1 - i),
                         card, 0, True)
            for i in range(n)]


def _body(n: int, **kw):
    return loss.format_loss_alert(_streak(n), KICKOFF,
                                  "discovered from the stored card",
                                  now=NOW, **kw)


# ── THE ESCALATION ───────────────────────────────────────────────────────────

def test_POSITIVE_CONTROL_a_six_day_streak_escalates_and_N2_N3_DIFFER():
    """Part 7's control, on the real streak length.

    An alert body never rendered in a test is a formatter with no consumer —
    and the six messages that produced no action were never rendered in one.
    """
    bodies = [_body(n) for n in range(1, 7)]
    first = bodies[0].splitlines()[0]
    second = bodies[1].splitlines()[0]
    third = bodies[2].splitlines()[0]

    assert second != first, "N=2 repeats the N=1 message"
    assert third != second, "N=3 repeats the N=2 message"
    assert third != first

    assert "NO PICKS PRODUCED" in first
    assert "NOT SELF-CORRECTING" in second
    assert "NOT BEING SEEN" in third

    # and from N=3 on, the running number keeps every message distinct
    heads = [b.splitlines()[0] for b in bodies[2:]]
    assert len(set(heads)) == len(heads), (
        f"identical messages from N=3 onward, which is the defect: {heads}")


def test_the_thresholds_are_CONSTANTS_stated_in_the_module():
    """Chosen in advance, not tuned to the firing rate afterwards."""
    assert loss.ESCALATE_NOT_SELF_CORRECTING == 2
    assert loss.ESCALATE_CHANNEL_SUSPECT == 3
    doc = loss.__doc__ or ""
    assert "BEFORE MEASURING" in doc.upper(), (
        "the module does not record that N was fixed before the firing rate "
        "was known")


# ── WHAT THE BODY MUST CARRY ─────────────────────────────────────────────────

def test_the_body_carries_THE_LOSS_and_its_IRREVERSIBILITY():
    body = _body(6, run_url="https://example.invalid/r/1")
    assert "6 OF ZERO PICKS" in body or "DAY 6" in body
    assert "120 fixtures lost so far" in body, (
        "the cumulative fixture count is missing")      # 6 days x 20
    assert "DO NOT BACKFILL" in body, (
        "nothing says the loss is permanent, which is the fact that makes it "
        "urgent rather than annoying")
    assert "no later run recovers it" in body
    # the per-day breakdown, so the reader can see where it started
    assert "2026-10-13" in body and "2026-10-18" in body
    assert "https://example.invalid/r/1" in body


def test_the_body_carries_BOTH_NUMBERS_in_BOTH_CLOCKS():
    """The same contract the margin alert already meets."""
    body = _body(2)
    assert "EARLIEST KICKOFF   10:15 UTC" in body, body
    assert "13:15" in body, "the Europe/Sofia kickoff is missing"
    assert "EEST" in body or "+0300" in body, "the local clock is not NAMED"
    # 10:15 - 88.1 - 15 = 08:31
    assert "LAST SAFE DISPATCH 08:31 UTC" in body, body
    assert "11:31" in body, "the local last-safe-dispatch is missing"
    assert "TIME REMAINING     +31 min" in body, (
        "the time remaining is absent or wrong; at 08:00 UTC against an 08:31 "
        "deadline it is +31")
    assert "88m time-to-picks" in body and "NOT the mean" in body


def test_the_action_is_the_OPERATORS_with_the_COMMAND():
    body = _body(3)
    assert "OPERATOR'S" in body
    assert "no automated alternative" in body
    assert "gh workflow run daily-picks.yml" in body


def test_an_UNDISCOVERED_card_is_not_reported_as_an_EMPTY_one():
    """The third state. On 10-18 the card read 0 because `--update` never ran,
    which is not the same as a day with no fixtures — and the first render of
    this module reported eleven days that way while the database held 101
    fixtures for 10-17."""
    rows = [loss.DayLoss(dt.date(2026, 10, 17), 0, 0, card_discovered=False)]
    body = loss.format_loss_alert(rows, None, now=NOW)
    assert "NEVER DISCOVERED" in body
    assert "not the same as no fixtures" in body
    assert "UNKNOWN" in body and "no deadline" in body, (
        "an undiscovered card must not yield a confident deadline")


# ── THE STREAK ITSELF ────────────────────────────────────────────────────────

def test_a_day_with_PICKS_ends_the_streak():
    rows = [loss.DayLoss(dt.date(2026, 10, 16), 20, 0),
            loss.DayLoss(dt.date(2026, 10, 17), 20, 4),
            loss.DayLoss(dt.date(2026, 10, 18), 20, 0)]
    assert [r.day for r in loss.zero_pick_streak(rows)] == [
        dt.date(2026, 10, 18)]


def test_a_GENUINELY_EMPTY_card_ends_the_streak_and_an_UNDISCOVERED_one_does_not():
    """A day with no fixtures lost nothing. A day nobody looked at did."""
    empty = [loss.DayLoss(dt.date(2026, 10, 17), 0, 0, card_discovered=True),
             loss.DayLoss(dt.date(2026, 10, 18), 20, 0)]
    assert len(loss.zero_pick_streak(empty)) == 1

    unlooked = [loss.DayLoss(dt.date(2026, 10, 17), 0, 0,
                             card_discovered=False),
                loss.DayLoss(dt.date(2026, 10, 18), 20, 0)]
    assert len(loss.zero_pick_streak(unlooked)) == 2, (
        "not looking is being treated as nothing to find")


def test_NEGATIVE_CONTROL_no_streak_renders_nothing_alarming():
    assert loss.zero_pick_streak(
        [loss.DayLoss(dt.date(2026, 10, 18), 20, 7)]) == []


# ── THE WIRING ───────────────────────────────────────────────────────────────

def test_the_workflow_RUNS_it_and_passes_DATABASE_URL():
    """It reads the card, so it needs the database — and the dialect refusal
    now makes a missing URL raise instead of reporting a plausible zero."""
    import pathlib

    import yaml

    d = yaml.safe_load(pathlib.Path(".github/workflows/daily-picks.yml")
                       .read_text(encoding="utf-8"))
    steps = list(d["jobs"].values())[0]["steps"]
    loss_steps = [s for s in steps if s.get("id") == "loss"]
    assert loss_steps, "the loss alert is not wired into daily-picks"
    s = loss_steps[0]
    assert "DATABASE_URL" in (s.get("env") or {}), (
        "the loss alert cannot read the card, so it would report a plausible "
        "zero — except that database.py now refuses, so it would raise")
    assert "pipeline_loss_alert" in s["run"]
    assert "ci_alert" in s["run"], "it prints but never sends"
    assert s.get("continue-on-error") is True, (
        "a failure to REPORT a failure must not change the run's verdict")
