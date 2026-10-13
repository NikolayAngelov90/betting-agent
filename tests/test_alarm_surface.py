"""The alarm surface, ENUMERATED rather than inferred — 2026-10-11.

Several states introduced in the week of 2026-10-05 are legitimately
**not-clean and not-faulty**. Each needs a stated membership, because:

> A third state that alarms is a fourth state.

and because an alarm that always fires carries the same information as one that
never fires — `fixtures_zero_active` four times over.

MEASURED 2026-10-11 by driving each one:

| state | where | alarms? | why |
| --- | --- | --- | --- |
| `BROKEN` | run verdict | **YES** | named in `--fail-on` |
| `DID_NOT_RUN` | run verdict | **YES** | named in `--fail-on` |
| `DEGRADED` | run verdict | no | deliberately omitted from `--fail-on` |
| `IN_PROGRESS` | run verdict | no | provisional: no verdict YET |
| `UNAUDITABLE` | run verdict | no | **by omission** from `--fail-on` |
| workflow `state != active` | workflow | **YES** | appended outside `--fail-on` |
| schedule margin < threshold | margin | **YES** | appended outside `--fail-on` |
| margin `UNEVALUATED` (weekday series) | margin | no | no series under the cron yet |
| prospective `ALARM` | **the probe**, not the audit | **YES** | the probe exits 1 and alerts itself |
| prospective `UNKNOWN - NOT safe` | the probe | no | the card is not discovered, or the clock is unverified |
| `DID_NOT_START` | H1 apparatus | no | not on the audit's alarm path at all |
| exit 2 `CRASH` | process | reds the job, **not** in `alarmed` | machinery, announced as such |
| series census `DEGRADED` (unreadable shas) | margin | no | **machinery**: guarded by a TEST on `fetch-depth`, not by a runtime alarm |
| `MODEL INTEGRITY FAILURE` | model artefact | **YES** | added 2026-10-12; baseline 0 in 1,400 load events, so it has dynamic range |
| `MODEL SIGNATURE MISSING` | model artefact | **YES** | added 2026-10-12; same baseline. `.sig` absent means UNVERIFIABLE, not verified |
| scheduled run NEVER EXISTED | schedule | **YES** | added 2026-10-13. A missing run has no row to verdict, so no run-based check could see it |
| UNPARSEABLE WORKFLOW push on `main` | workflow file | **YES** | added 2026-10-13. GitHub's own signal: a run named by PATH with `conclusion: failure` |
| the same on another branch | workflow file | no | a deleted branch's push cannot be cleared by any action |
| today has no run YET | schedule | no | **PENDING**, not missing — a run 458 min late has been observed |

THE TWO DAILY NON-ALARMS ARE THE POINT. `tomorrow UNKNOWN - NOT safe` and the
weekday series' `UNEVALUATED` both print on **every** run and alarm on
**neither**, because no action follows from either. Had they alarmed, the audit
would have gone red every day from the moment they shipped.

ROWS 14 AND 15, added 2026-10-12, ARE THE FIRST ADDITIONS THAT ALARM, and the
reasoning is recorded because a third state that alarms is a fourth state.

Model integrity sits OUTSIDE the verdict axis: a tampered or corrupt pickle
does not make a run BROKEN — the agent catches it, declines to load and
produces picks from Poisson + Elo — so a run can score CLEAN while its artefact
is unusable. `DEGRADED` is not in `--fail-on`, so routing it through the
findings list would print it and alert nobody, which is what it did for the
whole life of the project: both markers were already logged and read by
**nothing**.

They are safe to alarm because the baseline was measured BEFORE the wiring:
**0 integrity failures and 0 unsigned loads across 1,400 load events in 239
runs** of the local audit log cache, `measured` 2026-10-12. An alarm whose
baseline equals its threshold is the `fixtures_zero_active` shape; this one's
baseline is zero, so it has dynamic range.

ROWS 16-19, added 2026-10-13, close the gap this table could not have shown:
**an absent run and an unparseable workflow had no detector at all.**
`DID_NOT_RUN` is not one despite the name — it verdicts `steps_not_run` on a
run that EXISTS, and a run that never fired has no row to verdict. Two of the
four are deliberate NON-alarms, for the reason this table exists:

* **today has no run yet** is PENDING, not missing. The first weekday run
  under `0 4` arrived **458 minutes** late (measured 2026-10-12), so "it is
  07:00 and nothing has run" is not evidence of absence. Alarming on it would
  have fired on a run that was merely very late.
* **an unparseable push on another branch** cannot be cleared: the branch can
  be deleted, and then no action in the repository ends the alarm. CLR-2.

THE THIRTEENTH ROW, added 2026-10-11, is deliberately a NON-alarm. An
unreadable headSha empties the series, and the empty series printed as
`UNEVALUATED: no runs under the current cron yet` for fourteen runs. The fix is
`fetch-depth: 0` plus a test that pins it, because the condition can only
return if someone edits the workflow — which a test catches before merge and an
alarm would only catch after a day of blind reporting. The census still prints
the exclusion count and its clearing action on every run.
"""

import datetime as dt

import pytest

import scripts.ci_audit as ci
from scripts.schedule_deadline import prospective_margin

#: `MEASURED` 2026-10-10, regime `0 0 * * *`, scheduled only.
DELAYS = [271.0] * 5
DURATIONS = [45.0] * 5

#: What CI passes. Read from the workflow so the test cannot drift from it.
def _fail_on():
    import pathlib
    import re

    txt = pathlib.Path(".github/workflows/ci-audit.yml").read_text(encoding="utf-8")
    m = re.search(r"--fail-on\s+([A-Z_,]+)", txt)
    assert m, "the audit step no longer passes --fail-on"
    return {v.strip() for v in m.group(1).split(",") if v.strip()}


def test_the_alarming_verdicts_are_exactly_BROKEN_and_DID_NOT_RUN():
    assert _fail_on() == {"BROKEN", "DID_NOT_RUN"}


def test_DEGRADED_does_not_alarm_and_that_is_deliberate():
    """Narrowed on purpose: a scraper timing out is routine and recoverable,
    and red there trains people to ignore red."""
    assert "DEGRADED" not in _fail_on()


def test_UNAUDITABLE_is_outside_the_alarm_BY_OMISSION_and_still_intended():
    """Confirmed again, and the cost restated.

    A CANCELLED run with an EMPTY LOG scores `UNAUDITABLE`, not `CLEAN` — the
    short-log guard holds. It does not alarm, because `--fail-on` does not name
    it, so **a cancelled run with no log alarms nothing** and run 37369828831
    (2026-10-05) was found by hand.

    Still intended: an unauditable run is not a known failure, and alarming on
    it would fire on every platform hiccup. The cost is that it must be found
    in the ledger rather than by alert.
    """
    assert "UNAUDITABLE" in ci.PROVISIONAL_VERDICTS
    assert "UNAUDITABLE" not in _fail_on()
    assert "IN_PROGRESS" in ci.PROVISIONAL_VERDICTS
    assert "IN_PROGRESS" not in _fail_on()


def test_a_prospective_UNKNOWN_does_NOT_alarm():
    """It prints every single run. Alarming would be daily noise with no
    action — the card for tomorrow is not discoverable, which is a measured
    property of the horizon, not a fault anyone can clear that morning."""
    m = prospective_margin(None, 4 * 60, DELAYS, DURATIONS,
                           serves=dt.date(2026, 10, 12))
    assert m["verdict"] == "UNKNOWN"
    assert m["verdict"] != "ALARM"
    assert m.get("binding") is None


def test_a_prospective_ALARM_DOES_alarm():
    """The other direction, or the non-alarm above proves nothing."""
    m = prospective_margin(10 * 60 + 15, 4 * 60, DELAYS, DURATIONS,
                           serves=dt.date(2026, 4, 6))
    assert m["verdict"] == "ALARM", m
    assert m["binding"] < m["min_margin"]


def test_an_UNEVALUATED_day_type_does_NOT_alarm_and_cannot_win_WORSE():
    """The weekday series is empty until a run fires under `0 4`.

    A missing series must not win a comparison it did not enter, and it must
    not alarm — it is UNKNOWN, and the report says so in words instead.
    """
    out = ci.schedule_margins_by_daytype(
        {"weekend": [(None, 271.0)] * 3, "weekday": []},
        {"weekend": 0, "weekday": 240})
    assert out["alarm"] is False
    assert out["unevaluated"] == ["weekday"]
    assert out["worse"] == "weekend"
    text = ci.format_schedule_margins(out)
    assert "UNEVALUATED" in text and "not safe" in text


def test_a_breaching_margin_DOES_alarm():
    """The control for the test above."""
    out = ci.schedule_margins_by_daytype(
        {"weekend": [(None, 271.0 + 400)] * 5}, {"weekend": 0})
    assert out["alarm"] is True
    assert out["worse"] == "weekend"


def test_a_CRASH_reds_the_job_but_is_NOT_a_finding():
    """Exit 2 is machinery and says so; it never enters `alarmed`.

    Three reds on 2026-10-10 were a `ModuleNotFoundError` reported as
    "CI audit alarmed - see the audit step for the run ids", with no run ids
    and nothing audited for ~13 hours.
    """
    assert (ci.EXIT_CLEAN, ci.EXIT_FINDING, ci.EXIT_CRASH) == (0, 1, 2)
    import inspect

    src = inspect.getsource(ci._main_guarded)
    assert "alarmed" not in src.lower() or "MACHINERY" in src, (
        "the crash path describes itself as an alarm")


def test_H1s_DID_NOT_START_is_not_on_the_audits_alarm_path():
    """It is an H1 apparatus state reported by `h1_collection_check`, not a run
    verdict. Outside the audit's alarm set entirely, and that is why it does
    not need an entry in `--fail-on`."""
    from scripts.h1_collection_check import APPARATUS_STATES

    assert "DID_NOT_START" in APPARATUS_STATES
    assert "DID_NOT_START" not in _fail_on()


def test_the_FULL_alarm_surface_is_enumerated_here():
    """The list, so the surface is read rather than reconstructed.

    If a new alarm is appended to `alarmed` anywhere in `ci_audit.main`, this
    count moves and the test fails — which is the point: the alarm surface
    should not grow without someone writing down what it now contains.
    """
    import inspect

    src = inspect.getsource(ci)
    appends = [l.strip() for l in src.splitlines()
               if "alarmed.append(" in l]
    assert len(appends) == 7, (
        f"the alarm surface changed: {len(appends)} append sites, expected "
        f"7 (workflow state, run verdict, schedule margin, model integrity "
        f"failure, model signature missing, scheduled run never existed, "
        f"unparseable workflow on main). The prospective "
        f"margin is NOT here: it lives in the probe, which has the database "
        f"this step deliberately does not. Sites: " + repr(appends))


def test_a_DEGRADED_series_census_does_NOT_alarm_and_that_is_deliberate():
    """Row 13. Machinery degradation is announced, not alarmed — the same
    disposition as exit 2, and for the same reason: no pipeline run failed.

    What makes the non-alarm safe is that the condition is pinned by
    `test_the_audit_checks_out_FULL_history_and_the_reason_is_recorded`, so it
    cannot return silently. The runtime line exists for the case where the
    audit is run by hand on a shallow clone, where no test is in the loop.
    """
    degraded = ci.format_series_census(
        {"weekend": [], "weekday": [], ci.UNREADABLE_KEY: ["a" * 40]},
        {"weekend": 0, "weekday": 240})
    assert "EXCLUDED" in degraded and "fetch-depth: 0" in degraded
    # it is reported by the census, which is NOT on the alarm path
    assert "::error::" not in degraded
    out = ci.schedule_margins_by_daytype(
        {"weekend": [], "weekday": [], ci.UNREADABLE_KEY: ["a" * 40]},
        {"weekend": 0, "weekday": 240})
    assert out["alarm"] is False
    assert "UNAUDITABLE" not in _fail_on() and "DEGRADED" not in _fail_on()
