"""Stage 15, L4 — the capture schedule, and why it has the shape it has.

The lever was ADD an early window, not MOVE the late ones. The distinction is
the whole point and it is not self-evident from the cron line, so it is pinned:

    Moving costs the same as adding in CREDITS. It does not cost the same in
    SEASONAL ROBUSTNESS. The 21:17 and 23:17 runs claim nothing today because
    August's kickoff distribution has nothing that late. Winter fixture lists
    do. Removing them would trade free insurance for a saving of zero.

A future reader looking at "two runs that never spend anything" will be tempted
to delete them. This test is the note explaining why not.
"""

import pathlib
import re

import yaml

WF = pathlib.Path(".github/workflows/closing-lines.yml")


def _crons():
    d = yaml.safe_load(WF.read_text(encoding="utf-8"))
    on = d[True] if True in d else d["on"]      # YAML 1.1 parses bare `on` as True
    return [e["cron"] for e in on["schedule"]]


def _hours():
    hours = set()
    for c in _crons():
        minute, hour = c.split()[0], c.split()[1]
        for h in hour.split(","):
            hours.add((int(h), int(minute)))
    return hours


def test_the_late_windows_are_still_there():
    """They spend nothing in August. That is not a reason to remove them."""
    hours = {h for h, _ in _hours()}
    for late in (21, 23):
        assert late in hours, (
            f"the {late}:17 capture window is gone. It claims no credits in an "
            "August kickoff distribution, so removing it SAVES NOTHING — and "
            "winter fixture lists carry late kickoffs. This was considered and "
            "rejected in Stage 15; see the comment on the cron block.")


def test_an_early_window_exists_and_runs_after_picks_are_written():
    """MEASURED: picks for early kickoffs land 10:07-10:32 UTC.

    A capture window scheduled before its input exists finds no pending picks,
    claims no credits, and reports as "working". 09:17 would have caught zero
    of 54 early-kickoff observations.
    """
    early = sorted(m for h, m in _hours() if h < 11)
    assert early, (
        "no capture window before 11:17 — every pick with an 11:00 kickoff is "
        "already past kickoff at the first capture attempt of the day")
    earliest_hour = min(h for h, _ in _hours())
    earliest_min = min(m for h, m in _hours() if h == earliest_hour)
    as_minutes = earliest_hour * 60 + earliest_min
    assert as_minutes >= 10 * 60 + 35, (
        f"the earliest capture window is {earliest_hour:02d}:{earliest_min:02d} "
        "UTC, but picks are not written until 10:07-10:32. A window that runs "
        "before daily-picks has saved anything finds nothing to capture and "
        "spends nothing — indistinguishable from a broken lever.")


def test_the_schedule_still_covers_the_evening_peak():
    hours = {h for h, _ in _hours()}
    for peak in (17, 19):
        assert peak in hours, f"lost the {peak}:00 window; evening is the busiest"


def test_daily_picks_still_runs_before_the_first_capture_window():
    """The ordering the early window depends on. If daily-picks moves later,
    10:47 silently becomes the 09:17 case this test was written to prevent."""
    picks = pathlib.Path(".github/workflows/daily-picks.yml").read_text(encoding="utf-8")
    m = re.search(r"cron:\s*'(\d+)\s+(\d+)\s", picks)
    assert m, "could not read the daily-picks cron"
    picks_at = int(m.group(2)) * 60 + int(m.group(1))
    earliest = min(h * 60 + mi for h, mi in _hours())
    assert picks_at < earliest, (
        f"daily-picks now fires at {picks_at // 60:02d}:{picks_at % 60:02d} but "
        f"the first capture window is {earliest // 60:02d}:{earliest % 60:02d} — "
        "capture would run before any pick exists")
