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
    # ── THE WORDING CHANGED 2026-10-22, AND THE REASON IS THE POINT. ─────
    #
    # It said "is UNEVALUATED, not safe" and alarmed on nothing. "Not safe"
    # asserts a safety property, and this line cannot evaluate one -- it is
    # reporting that there is nothing to evaluate. Of the three lines in that
    # shape, this is the one that was REWORDED rather than promoted to an
    # alarm, because it is cleared by the next scheduled run firing and
    # CLR-2 exists because an alarm cleared by waiting teaches waiting.
    assert "margin NOT COMPUTED" in text
    assert "neither clear nor breached" in text, (
        "the line no longer distinguishes 'not computed' from 'clear'")
    assert "not safe" not in text, (
        "the safety claim is back in a line that still alarms on nothing")
    assert "does not alarm" in text, (
        "the line must say WHY it does not alarm, or the mismatch between "
        "its words and the alarm set gets rediscovered")


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
    # ── `_keyed.append(` COUNTS TOO, AS OF 2026-10-23. ───────────────────
    #
    # The schedule margin moved from `alarmed` to `_keyed`, which is a change
    # of ROUTING and not a removal from the surface: it still alarms, but only
    # when `classify_finding` returns NEW, DETERIORATED or UNACKNOWLEDGED.
    # Counting only `alarmed.append(` dropped the total 12 -> 11 and would
    # have read a re-route as a shrinking surface.
    appends = [l.strip() for l in src.splitlines()
               if "alarmed.append(" in l or "_keyed.append((" in l]
    # 9 -> 11 on 2026-10-21, both named rather than absorbed into the number.
    #
    # THE TENTH is a RED THIS AUDIT CANNOT CLASSIFY. `reminders.yml` reddens
    # deliberately when a self-expiring guard reaches its date, so its red is
    # a VERDICT and does not alarm -- but a red whose marker cannot be read
    # (empty log, cancelled run, or a marker with no exit beside it) is
    # UNCLASSIFIABLE, and that alarms, because "could not classify" is not
    # "nothing to classify".
    #
    # THE ELEVENTH is the STRUCTURAL MARGIN DETERIORATING past its
    # acknowledged figure. The structural breach is acknowledged and reports
    # without alarming; it alarms again when it gets WORSE than the recorded
    # figure or when its review date passes. The previous acknowledgement
    # compared only the figure and absorbed a breach for days past its own
    # review date, so both are compared now.
    #
    # The census of all eleven -- firings, distinct findings, and firings that
    # produced a change -- is in `tests/test_silent_surfaces_can_fire.py`.
    # 11 -> 12 on 2026-10-21. THE TWELFTH IS THE PATTERN SELF-TEST: a
    # registered pattern that matches its own example and still yields no
    # value is a detector that CANNOT FIRE, and two of them sat dead for ten
    # days because the only diagnostic was gated on their condition. It
    # alarms at REGISTRATION, before the early return on "no runs to audit".
    # THE TWELVE ARE UNCHANGED IN NUMBER; ONE IS NOW KEYED. The schedule
    # margin is the finding that reddened this workflow every day for two
    # weeks, so it is the first to carry a key and a figure and be classified
    # against its recorded occurrence.
    assert sum(1 for l in appends if "_keyed.append((" in l) == 1, (
        f"{sum(1 for l in appends if '_keyed.append((' in l)} keyed alarm "
        f"sites, expected 1 (the schedule margin). A new keyed finding needs "
        f"a row in KNOWN_FINDINGS and a line here.")
    assert len(appends) == 12, (
        f"the alarm surface changed: {len(appends)} append sites, expected "
        f"12 (pattern self-test, workflow state, red UNCLASSIFIABLE, run "
        f"verdict, model "
        f"integrity failure, model signature missing, scheduled run never "
        f"existed, UNTRIAGED DIAGNOSED FINDING, unparseable workflow on "
        f"main, actionable margin for tomorrow, structural margin "
        f"deterioration, schedule margin). Sites: " + repr(appends))


def test_the_NINTH_surface_is_an_UNTRIAGED_finding_NOT_the_backlog_count():
    """Added 2026-10-20. The distinction is the whole design.

    The weather producer was diagnosed on 2026-08-07 and deleted on
    2026-10-20 — seventy-four days — and nothing counted it in between. The
    register now reports the backlog daily, but **a triaged backlog is a
    decision, not a finding**, so the count must NOT alarm: six of the eight
    still-live entries are `DECLARE` because they touch the predictive core,
    and alarming on them every day would make red mean nothing, which is the
    `fixtures_zero_active` failure this project has paid for.

    What alarms is an entry with **no outcome**, or a `DECLARE` with **no
    re-triage date** — because an unmarked entry is how 74 days happened.
    """
    import inspect

    src = inspect.getsource(ci.main)
    assert "diagnosed finding(s) UNTRIAGED" in src
    assert "if _fr_bad:" in src, "the alarm is not gated on the untriaged set"
    assert "if _fr_live:" not in src, (
        "the audit alarms on the live backlog count, which would fire every "
        "day on a deliberately-declared entry")
    # and an unreadable register is UNEVALUATED, not an empty backlog
    assert "NOT an empty backlog" in src


def test_the_EIGHTH_surface_is_the_PROSPECTIVE_margin_and_why_it_moved_here():
    """It was the probe's, and the probe is off.

    The count above said for eight days: "the prospective margin is NOT here:
    it lives in the probe, which has the database this step deliberately does
    not." Both halves have changed.

      * the PROBE was disabled 2026-10-18 -- 240 page-loads, 8 refusals, zero
        deadlines -- so from that day **nothing** computed tomorrow's margin
      * the DATABASE is no longer the way to get the deadline. Our own
        `matches` never holds tomorrow's early card (max observed fixture lead
        23.62h against a 24-28h requirement), so the database was never going
        to answer it. The external reference does, over ONE stdlib HTTPS GET.

    The objection that removed this line in the first place was `credentials
    AND psycopg2 AND SQLAlchemy on every audit run`. It now costs one secret
    and no packages, which is why it can live here.
    """
    import ast
    import inspect
    import textwrap

    src = inspect.getsource(ci.main)
    # RENAMED 2026-10-21: the single margin became TWO, and the alarm
    # moved to the ACTIONABLE one because that is the only quantity a
    # manual dispatch changes (CLR-2). The surface still exists.
    assert "actionable margin" in src, "the eighth surface is gone"
    assert "structural margin" in src, (
        "the structural half is gone, so the pair was reduced to the "
        "optimistic half")

    # BY AST, NOT BY TEXT -- AND THIS IS THE THIRD TIME TODAY. A text scan
    # for `DATABASE_URL` matched the COMMENT above the new block, which names
    # it to explain why it is absent. Comments are dropped by the parser and
    # docstrings are excluded explicitly, so what remains is use.
    tree = ast.parse(textwrap.dedent(src))
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr)                 and isinstance(body[0].value, ast.Constant)                 and isinstance(body[0].value.value, str):
            docstrings.add(id(body[0].value))
    used = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            used.add(node.id)
        elif isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str)                 and id(node) not in docstrings:
            used.add(node.value)
    for banned in ("get_db", "DATABASE_URL", "psycopg2", "create_engine",
                   "discovered_deadline"):
        assert banned not in used, (
            f"`{banned}` is USED in ci_audit.main; this step installs no "
            f"packages and holds no credentials beyond GH_TOKEN and the "
            f"reference key")


def test_the_prospective_alarm_CARRIES_the_deadline_its_BASIS_and_COVERAGE():
    """An alarm that says only a number cannot be acted on.

    A margin is meaningless without the deadline it was measured against, and
    a reference-derived deadline is meaningless without its coverage: 9 of 30
    leagues means the figure is an UPPER BOUND on the true minimum.
    """
    import inspect

    src = inspect.getsource(ci.main)
    # ANCHOR ON THE ALARM TEXT, NOT THE FIRST MENTION. 2026-10-21: the
    # phrase now also appears in the comment above the block, and a
    # slice from there measured prose instead of the message.
    block = src[src.index(chr(102) + chr(34) + "actionable margin "):][:700]
    # `bound_by` IS GONE, DELIBERATELY. 2026-10-21: the deadline was
    # `min(reference, floor)` and `bound_by` named which side won -- which was
    # always the floor, because the floor IS the observed earliest kickoff.
    # The two are now separate margins, so the actionable one is
    # reference-derived BY DEFINITION and there is no side to name. What the
    # pin was really protecting -- can the reader tell how much of the card
    # was visible -- is carried by `coverage` and `reference_state`, and
    # `residual` is added because that is what says the figure is an UPPER
    # BOUND rather than leaving the reader to infer it from "9 of 30".
    for part in ("deadline", "reference", "reference_state", "coverage",
                 "residual", "UPPER"):
        assert part in block, (
            f"the actionable alarm omits {part!r}, so the reader cannot tell "
            f"how much of the card was visible or that it is an upper bound")
    assert "CLEARING_ACTION" in src, (
        "the alarm does not name the action that clears it")


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
