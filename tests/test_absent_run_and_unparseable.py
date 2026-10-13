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
    assert str(ci.OBSERVED_LATENESS_MAX_MIN) in text, (
        "the bound the PENDING state is measured against is not stated")


def test_the_LATENESS_bound_is_the_MEASURED_maximum_and_says_so():
    """458, not 304. A single observation replaced the previous maximum, and
    the constant's comment must say that rather than presenting it as a
    distribution."""
    assert ci.OBSERVED_LATENESS_MAX_MIN == 458
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    block = src.split("OBSERVED_LATENESS_MAX_MIN", 1)[0][-900:]
    assert "458" in block and "304" in block, (
        "the bound does not record the figure it replaced")
    assert "is not a distribution" in block or "not a distribution" in block


def test_an_UNEVALUATED_check_says_so_rather_than_passing():
    """No cron resolved, or the run list unreadable, must not read as clean."""
    for m in ({"state": "UNEVALUATED", "reason": "no cron resolved"},
              {"state": "UNEVALUATED", "reason": "run list unreadable"}):
        text = ci.format_missing_runs(m)
        assert "UNEVALUATED" in text
        assert "NOT a clean bill" in text


def test_the_detector_runs_against_the_LIVE_repo_and_2026_10_12_is_NOT_missing():
    """The correction, pinned.

    2026-10-12 looked absent at 10:55 UTC and was reported as such. It was
    not: the scheduled run fired at 11:37:39 UTC, 458 minutes late, and is a
    legitimate data point in the `0 4` series. A detector that called it
    missing would have been wrong.
    """
    m = ci.missing_scheduled_runs()
    if m.get("state") != "CHECKED":
        pytest.skip(f"detector unevaluated here: {m.get('reason')}")
    assert dt.date(2026, 10, 12) not in m["missing"], (
        "2026-10-12 is reported missing, but its scheduled run exists "
        "(38243861903, fired 11:37:39Z)")


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
