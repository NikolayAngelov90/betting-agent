"""A run that never fired, and a file GitHub refused — 2026-10-13.

Neither had a detector, and `DID_NOT_RUN` is not one despite the name: it
verdicts `steps_not_run` on a run that **exists** whose core steps were
skipped. A workflow that produced no run at all has no row to verdict, so
every run-based check in `ci_audit` was blind to it.

    A workflow GitHub refuses does not fail, it disappears.

THAT SECOND HALF IS WRONG, AND MEASURING IT IS WHAT MADE THE DETECTOR
POSSIBLE. GitHub **does** create a run for an unparseable workflow. Pushing a
duplicate `env:` to main on 2026-10-12 produced run `38238629301` four seconds
later: `event=push`, `conclusion=failure`, and **named by the file's path**
rather than its `name:` field — because the `name:` could not be parsed.
`gh run list --workflow <file>` lists it.

So the signal existed and nothing read it. The silence was **scope**, not
timing: `fixture-horizon-probe` is in `NEVER_ALARM_WORKFLOWS`.

WHAT GITHUB DOES AND DOES NOT OFFER, measured 2026-10-13 on a scratch branch:

| probe | result |
| --- | --- |
| `gh api .../actions/workflows` | `state: active` for a file containing the refused defect — **state is not a parse verdict** |
| dispatch a file with no `workflow_dispatch` | `422 "does not have 'workflow_dispatch' trigger"` — the trigger check short-circuits before any parse verdict |
| dispatch a file WITH the trigger and a duplicate key | `422 "failed to parse workflow: (Line: 11, Col: 9): 'env' is already defined"` — **GitHub's own parser, with line and column** |
| the same, valid | **creates a run** |

So GitHub's parser is reachable only in the FAILING direction, and only by an
attempt that runs the workflow when it parses. **It cannot be a pre-push gate.**
The local gate is therefore a strict YAML loader, and this file anchors it to
the one input GitHub demonstrably refused rather than claiming equivalence.
"""

import datetime as dt
import json
import pathlib

import pytest
import yaml

import scripts.ci_audit as ci
from tests.test_entrypoints_are_invocable import _strict_loader

# ── THE EMPIRICAL ANCHOR ─────────────────────────────────────────────────────

#: THE EXACT SHAPE GITHUB REFUSED, reduced to its essentials: a step with two
#: `env:` keys. Reproduced on a scratch branch 2026-10-13 and refused by
#: GitHub's parser with `(Line: 11, Col: 9): 'env' is already defined`.
GITHUB_REFUSED_THIS = (
    "name: ZZ Parse Bad\n"
    "on:\n"
    "  workflow_dispatch:\n"
    "jobs:\n"
    "  j:\n"
    "    runs-on: ubuntu-latest\n"
    "    steps:\n"
    "      - run: echo hi\n"
    "        env:\n"
    "          A: 1\n"
    "        env:\n"
    "          B: 2\n"
)


def test_the_LOCAL_gate_refuses_what_GITHUB_refused():
    """The anchor, and the honest limit of it.

    This proves the local checker rejects **one input GitHub is measured to
    reject** — not that the two parsers agree in general. Equivalence is not
    claimable: there is no side-effect-free endpoint to compare against.
    """
    # the PERMISSIVE loader accepts it, which is the whole defect
    assert yaml.safe_load(GITHUB_REFUSED_THIS) is not None
    with pytest.raises(yaml.YAMLError):
        yaml.load(GITHUB_REFUSED_THIS, Loader=_strict_loader())


def test_the_anchor_is_the_SHAPE_that_actually_shipped():
    """If the anchor drifts from what was pushed, it stops anchoring anything.
    """
    assert GITHUB_REFUSED_THIS.count("        env:\n") == 2, (
        "the anchor no longer carries two step-level env keys, which is the "
        "shape GitHub refused")


def test_the_live_workflows_all_pass_the_STRICT_loader():
    """The gate, applied. A duplicate key here means the workflow is not
    scheduled at all."""
    for f in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
        yaml.load(f.read_text(encoding="utf-8"), Loader=_strict_loader())


# ── THE UNPARSEABLE-WORKFLOW DETECTOR ────────────────────────────────────────

def test_the_PATH_as_NAME_is_the_discriminator():
    """GitHub names the run by the file path when `name:` is unparseable.

    Driven against recorded rows rather than the live API, so the test does
    not depend on a run still being inside the query window.
    """
    rows = [
        {"databaseId": 1, "workflowName": ".github/workflows/x.yml",
         "conclusion": "failure", "headBranch": "main"},
        {"databaseId": 2, "workflowName": "Daily Betting Picks",
         "conclusion": "failure", "headBranch": "main"},
        {"databaseId": 3, "workflowName": ".github/workflows/y.yml",
         "conclusion": "success", "headBranch": "main"},
        {"databaseId": 4, "workflowName": ".github/workflows/z.yml",
         "conclusion": "failure", "headBranch": "scratch"},
    ]
    text = ci.format_unparseable(rows)
    # only 1 alarms: 2 is a named workflow, 3 did not fail, 4 is off main
    assert "::error::1 UNPARSEABLE WORKFLOW push(es) on main" in text, text
    assert ".github/workflows/x.yml" in text
    assert "scratch" in text, "the off-main case is not even reported"
    assert "CLEARING ACTION" in text


def test_an_OFF_MAIN_unparseable_push_is_reported_and_NOT_alarmed():
    """CLR-2. A branch can be deleted, after which no action in the repository
    clears the alarm — the `fixtures_zero_active` shape."""
    rows = [{"databaseId": 9, "workflowName": ".github/workflows/q.yml",
             "conclusion": "failure", "headBranch": "gone"}]
    text = ci.format_unparseable(rows)
    assert "::error::" not in text, text
    assert "unparseable_workflows[0 on main]" in text
    assert "NOT alarmed" in text


def test_NEGATIVE_CONTROL_a_clean_history_says_zero_not_nothing():
    """A check that prints nothing when clean cannot be told from one that did
    not run — the third-state family, which this project has met repeatedly."""
    text = ci.format_unparseable([])
    assert "unparseable_workflows[0 on main]" in text


# ── THE ABSENT-RUN DETECTOR ──────────────────────────────────────────────────

def test_a_PAST_day_with_no_scheduled_run_is_MISSING():
    """The alarm case, driven through the formatter."""
    text = ci.format_missing_runs({
        "state": "CHECKED", "window_days": 14, "pending": None,
        "missing": [dt.date(2026, 10, 5), dt.date(2026, 10, 6)],
        "crons": {"weekday": 240, "weekend": 0}})
    assert "::error::2 scheduled run(s) NEVER EXISTED" in text, text
    assert "2026-10-05" in text and "2026-10-06" in text
    assert "not late runs" in text
    assert "must NOT enter any delay series" in text, (
        "the output does not say these are excluded from the delay "
        "population, which is the whole point")
    assert "CLEARING ACTION" in text


def test_TODAY_with_no_run_yet_is_PENDING_and_does_NOT_alarm():
    """The third state, and the reason it exists is a measurement.

    The first weekday run under `0 4` arrived **458 minutes** late. Alarming
    on "it is 07:00 and nothing has run" would have fired on a run that was
    merely very late — and the margin alarm already covers the lateness.
    """
    text = ci.format_missing_runs({
        "state": "CHECKED", "window_days": 14,
        "missing": [], "pending": (dt.date(2026, 10, 13), 147),
        "crons": {"weekday": 240, "weekend": 0}})
    assert "::error::" not in text, text
    assert "PENDING, not missing" in text
    assert "147 min past nominal" in text
    assert str(ci.PENDING_BOUND_MIN) in text, (
        "the bound the PENDING state is measured against is not stated")


def test_the_MEASURED_bound_is_RETIRED_and_the_PENDING_bound_is_STRUCTURAL():
    """458 replaced 304 on one observation (2026-10-12), and 684 overtook 458
    on 2026-11-23: the detector called the fallback ABSENT at 666 min and
    four audits went red before it landed. The measured figure is KEPT as the
    record; the PENDING bound is one day, the next slot, which no lateness
    reading can overtake."""
    assert ci.OBSERVED_LATENESS_MAX_MIN == 458
    assert ci.PENDING_BOUND_MIN == 24 * 60
    assert max(ci.LATENESS_THAT_RETIRED_THE_BOUND.values()) >         ci.OBSERVED_LATENESS_MAX_MIN, "the record no longer shows the overtaking"
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    block = src.split("PENDING_BOUND_MIN = ", 1)[0][-2400:]
    for figure in ("458", "304", "684", "700"):
        assert figure in block, f"the bound's record lost {figure}"
    assert "not a distribution" in block


# ── THE DETECTOR, DRIVEN ON THE 2026-11-23 EVENING. Stage 35. ───────────────

_FALLBACK = {"weekday": 660, "weekend": 660}


def _detector(monkeypatch, created, today, now, days=3):
    monkeypatch.setattr(ci, "current_crons_by_daytype",
                        lambda *_a, **_k: dict(_FALLBACK))
    listing = json.dumps([{"createdAt": c, "event": "schedule"}
                          for c in created])
    monkeypatch.setattr(ci, "_sh", lambda *_a, **_k: listing)
    return ci.missing_scheduled_runs(days=days, today=today, now=now)


def test_a_slot_666_min_late_is_PENDING_not_ABSENT(monkeypatch):
    """The audit of 2026-11-23 22:06 UTC; the run landed at 22:24."""
    m = _detector(monkeypatch, ["2026-11-22T18:11:29Z"],
                  dt.date(2026, 11, 23), dt.datetime(2026, 11, 23, 22, 6),
                  days=1)
    assert m["missing"] == [], m
    assert m["pending"] == (dt.date(2026, 11, 23), 666), m


def test_KNOWN_WRONG_the_retired_bound_called_it_ABSENT():
    assert 666 > ci.OBSERVED_LATENESS_MAX_MIN, (
        "the reproduction does not reproduce: 666 min was within the bound")


def test_a_run_landing_AFTER_MIDNIGHT_belongs_to_YESTERDAY(monkeypatch):
    """Created 00:40 on 11-24, before the 11:00 nominal: the 11-23 slot."""
    m = _detector(monkeypatch, ["2026-11-22T18:11:29Z", "2026-11-24T00:40:00Z"],
                  dt.date(2026, 11, 24), dt.datetime(2026, 11, 24, 2, 0),
                  days=1)
    assert m["missing"] == [] and m["pending"] is None, m
    assert ci._nominal_day(dt.datetime(2026, 11, 24, 0, 40), _FALLBACK) ==         dt.date(2026, 11, 23)
    assert ci._nominal_day(dt.datetime(2026, 11, 24, 20, 58), _FALLBACK) ==         dt.date(2026, 11, 24)


def test_a_slot_SKIPPED_while_a_LATER_one_ran_is_MISSING_at_once(monkeypatch):
    m = _detector(monkeypatch, ["2026-11-22T18:11:29Z", "2026-11-24T20:58:07Z"],
                  dt.date(2026, 11, 24), dt.datetime(2026, 11, 24, 21, 30),
                  days=2)
    assert m["missing"] == [dt.date(2026, 11, 23)], m


def test_a_slot_a_DAY_past_nominal_with_nothing_after_it_is_MISSING(
        monkeypatch):
    """The cron that stopped altogether: no later slot ever runs."""
    m = _detector(monkeypatch, ["2026-11-21T18:00:00Z"],
                  dt.date(2026, 11, 24), dt.datetime(2026, 11, 24, 21, 30))
    assert m["missing"] == [dt.date(2026, 11, 22), dt.date(2026, 11, 23)], m
    assert m["pending"] == (dt.date(2026, 11, 24), 630), m


def test_an_UNEVALUATED_check_says_so_rather_than_passing():
    """No cron resolved, or the run list unreadable, must not read as clean."""
    for m in ({"state": "UNEVALUATED", "reason": "no cron resolved"},
              {"state": "UNEVALUATED", "reason": "run list unreadable"}):
        text = ci.format_missing_runs(m)
        assert "UNEVALUATED" in text
        assert "NOT a clean bill" in text


@pytest.mark.live
def test_the_detector_does_not_call_a_day_with_a_RUN_missing():
    """NO HARD-CODED DATE, and OUT OF THE GATING SUITE.

    THIS TEST COST SIX CARDS. Its first version asserted that **2026-10-12**
    was not in `missing`, against a rolling 14-day window over a live
    `gh run list`. The date aged out of the window on schedule, the assertion
    failed, `Run tests` failed, and every downstream step - `--update`,
    `--settle`, the picks - was SKIPPED for six consecutive days:
    10-13..10-18, 137 fixtures, **zero picks**.

    Two defects in one test, and the workflow already documented both:

      1. it queried LIVE EXTERNAL STATE from inside the gate, so its outcome
         changed with no code change - the `calendar` marker exists because
         Stage 28 paid one card for exactly this on 2026-10-09;
      2. it pinned a DATE rather than a PROPERTY, so it could only ever
         expire.

    The property is now derived from the same run list the detector reads: a
    day that HAS a scheduled run must not be reported missing. That cannot
    expire, because both sides move together.
    """
    m = ci.missing_scheduled_runs()
    if m.get("state") != "CHECKED":
        # REPORTED, NOT SKIPPED. 2026-11-05. This was `pytest.skip`, and a
        # skip is indistinguishable from a pass in a green run -- so a
        # detector that had stopped evaluating looked exactly like one that
        # had nothing to report.
        raise AssertionError(
            f"UNEVALUATED, not clear: the detector did not evaluate, so this "
            f"control verifies nothing about whether a day with a run is "
            f"reported missing. reason: {m.get('reason')}")

    from tests.gh_support import gh_json_or_unevaluated
    runs = gh_json_or_unevaluated(
        ["run", "list", "--workflow", "daily-picks.yml", "--limit", "120",
         "--json", "createdAt,event"],
        what="daily-picks' run list")
    # Stage 35: attributed to the day each run BELONGS to, as the detector
    # does -- a slot that lands after midnight is yesterday's.
    have = {ci._nominal_day(dt.datetime.fromisoformat(
        r["createdAt"].replace("Z", "+00:00")), m["crons"])
        for r in runs if r.get("event") == "schedule"}
    overlap = have & set(m["missing"])
    assert not overlap, (
        f"days with a scheduled run are reported MISSING: {sorted(overlap)}")


# ── WHAT `DID_NOT_RUN` IS, AND IS NOT ────────────────────────────────────────

def test_DID_NOT_RUN_is_about_SKIPPED_STEPS_not_an_absent_run():
    """The naming trap, pinned so nobody relies on it for the wrong thing.

    It is a verdict on a run that EXISTS. The question "did the workflow run
    today?" is answered by `missing_scheduled_runs`, and before 2026-10-13 it
    was answered by nothing.
    """
    assert ci.verdict({"steps_not_run": ["picks"]}, [], None) == "DID_NOT_RUN"
    # ...and with no run at all there is no facts dict to pass: the verdict
    # function is never reached, which is exactly the gap.
    assert ci.verdict({}, [], "a log with content" * 40) in ("CLEAN", "DEGRADED")


# ── MIDNIGHT IS FALSY, AND IT KILLED THE DETECTOR FOR A DAY ─────────────────
#
#     An always-unevaluated detector is indistinguishable from a working one
#     whose answer is always "nothing to report".
#
# `0 0` is stored as 0 minutes past midnight. `any([0, 0])` is False, so the
# guard that meant "did we resolve a cron at all" rejected the one cron
# expression the repo actually uses on both day types. It survived review
# because the weekday cron was `0 4` at the time: `any([0, 240])` is True, and
# the weekend zero was carried by the weekday value. The revert at 566c927
# made both zero and the detector returned UNEVALUATED on every call for its
# entire first day.
#
# Found 2026-10-18 by `current_crons_by_daytype()` reporting `{'weekend': 0,
# 'weekday': 0}` - a successful resolution - while the consumer reported "no
# cron resolved for either day type" one line later.

def test_KNOWN_WRONG_both_crons_at_MIDNIGHT_do_not_make_it_UNEVALUATED(
        monkeypatch):
    """The live configuration, and the one that broke it."""
    monkeypatch.setattr(ci, "current_crons_by_daytype",
                        lambda *_a, **_k: {"weekend": 0, "weekday": 0})
    monkeypatch.setattr(ci, "_sh", lambda *_a, **_k: "[]")
    m = ci.missing_scheduled_runs(today=dt.date(2026, 10, 18))
    assert m["state"] == "CHECKED", (
        f"midnight crons still read as absent: {m.get('reason')!r}. The "
        f"detector cannot fire, and an unevaluated detector reports nothing "
        f"while looking identical to a clean one.")
    assert m["missing"], (
        "with an empty run list every past day in the window is MISSING; an "
        "empty result here means the loop is being skipped too")


def test_an_UNREADABLE_TREE_is_a_DIFFERENT_REASON_than_NO_CRON(monkeypatch):
    """Two causes, one sentence, until now.

    `or {}` turned a None (we could not read the workflow at this sha) into an
    empty dict (the workflow has no cron). The first is a statement about our
    clone, the second about the repo, and only the first is fixed by
    `fetch-depth: 0`.
    """
    monkeypatch.setattr(ci, "current_crons_by_daytype", lambda *_a, **_k: None)
    unreadable = ci.missing_scheduled_runs(today=dt.date(2026, 10, 18))
    assert unreadable["state"] == "UNEVALUATED"
    assert "unreadable" in unreadable["reason"]
    assert "fetch-depth" in unreadable["reason"], (
        "the reason does not name the remedy")

    monkeypatch.setattr(ci, "current_crons_by_daytype",
                        lambda *_a, **_k: {"weekend": None, "weekday": None})
    nocron = ci.missing_scheduled_runs(today=dt.date(2026, 10, 18))
    assert nocron["state"] == "UNEVALUATED"
    assert nocron["reason"] != unreadable["reason"], (
        "an unreadable tree and a cron-less workflow still give the same "
        "reason, so the remedy cannot be chosen from the alarm")


def test_a_RESOLVED_cron_of_ZERO_is_not_confused_with_an_ABSENT_one():
    """The distinction itself, independent of the consumer: a dict holding 0
    and a dict holding None must not compare equal in the places that branch
    on it."""
    assert 0 is not None
    for d in ({"weekday": 0, "weekend": 0}, {"weekday": 0, "weekend": None}):
        assert not all(v is None for v in d.values()), (
            f"{d} reads as 'nothing resolved' and it holds a midnight cron")


def test_KNOWN_WRONG_a_run_from_the_EARLIER_regime_is_NOT_shifted():
    """2026-11-21 07:30 UTC: the old weekend cron `30 0`, not a late `0 11`.
    The live read called 11-21 MISSING until the shift was bounded to the
    current cron's regime."""
    assert ci._nominal_day(dt.datetime(2026, 11, 21, 7, 30), _FALLBACK) == \
        dt.date(2026, 11, 21)
    assert ci._nominal_day(dt.datetime(2026, 11, 22, 7, 30), _FALLBACK) == \
        dt.date(2026, 11, 21), "the bound must not switch the shift off"
