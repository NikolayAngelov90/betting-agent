"""An unreadable run list is UNEVALUATED and REPORTED -- never a skip.

WHY THIS FILE EXISTS. 2026-11-05.

The two live `gh` callers ended an unreadable run list with `pytest.skip`.
A SKIP AND A PASS ARE INDISTINGUISHABLE IN A GREEN RUN, so a reminder whose
API access had died reported exactly what a healthy one reports: nothing.
That is `a matcher with zero hits` in pytest's own vocabulary, and it is why a
deleted PAT could be invisible -- the consumer would be broken and green.

The guard keyed on the token being PRESENT, never on the call SUCCEEDING, so a
token that is present and DEAD read as healthy. These controls pin both
directions: the unreadable case must REPORT, and the readable case must still
reach a verdict.

    known-wrong   a dead token -> UNEVALUATED, reported; not skipped, not passed
    positive      a readable list -> evaluates and returns rows
"""
import pathlib
import subprocess

import pytest  # noqa: F401  -- present so a stray skip here would be visible

# NOT `tests.conftest`. Importing conftest makes a file `capture_critical`
# (it reaches `src.data.odds_quota`), and these controls must not be able
# to stop price capture. See the docstring of `tests/gh_support.py`.
from tests.gh_support import gh_json_or_unevaluated

TESTS = pathlib.Path(__file__).resolve().parent
ROOT = TESTS.parent

# gh's real message when a workflow step has no usable token.
DEAD_TOKEN_STDERR = (
    "gh: To use GitHub CLI in a GitHub Actions workflow, set the GH_TOKEN "
    "environment variable.")


class _Result:
    """Stands in for `subprocess.CompletedProcess`."""

    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def _outcome(monkeypatch, result):
    """What the helper DID: reported, passed, or something else entirely.

    `pytest.raises(AssertionError)` is not enough on its own: a `Skipped`
    would propagate past it and mark THIS control skipped, which is the very
    state under test. So every outcome is captured and named.
    """
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: result)
    try:
        rows = gh_json_or_unevaluated(
            ["run", "list", "--workflow", "daily-picks.yml", "--json",
             "createdAt,event"], what="the daily-picks run list")
    except AssertionError as exc:
        return "REPORTED", str(exc)
    except BaseException as exc:  # noqa: BLE001 -- a Skipped lands here
        return type(exc).__name__, str(exc)
    return "PASSED", rows


# -- THE KNOWN-WRONG CONTROL -------------------------------------------------

def test_KNOWN_WRONG_a_DEAD_TOKEN_is_REPORTED_as_UNEVALUATED(monkeypatch):
    """A dead token must report UNEVALUATED. Not skip. Not pass."""
    how, msg = _outcome(monkeypatch, _Result(rc=1, err=DEAD_TOKEN_STDERR))
    assert how == "REPORTED", (
        f"a dead token produced {how}, not a reported AssertionError. A "
        f"Skipped or a pass here IS the defect this file exists for: "
        f"{str(msg)[:200]}")
    assert msg.startswith("UNEVALUATED"), (
        f"the report must LEAD with UNEVALUATED so a reader can tell 'could "
        f"not look' from 'a reminder is due'. got: {msg[:200]}")


def test_the_DEAD_TOKEN_report_NAMES_THE_TOKEN_STATE(monkeypatch):
    """So that `present but dead` is diagnosable from the message alone.

    The 2026-10-21 fix supplied a token and the guard then read `present` as
    `working`. The message now states what was set, so the next reader does
    not have to re-derive it.
    """
    _, msg = _outcome(monkeypatch, _Result(rc=1, err=DEAD_TOKEN_STDERR))
    for needle in ("GH_TOKEN set:", "GITHUB_TOKEN set:"):
        assert needle in msg, f"{needle!r} absent from the report: {msg[:200]}"


def test_the_DEAD_TOKEN_report_CARRIES_STDERR_rather_than_swallowing_it(
        monkeypatch):
    """`ci_audit._sh` returns stdout-or-empty, which hides 401/403/rate limit."""
    _, msg = _outcome(monkeypatch, _Result(rc=1, err=DEAD_TOKEN_STDERR))
    assert "set the GH_TOKEN" in msg, (
        f"gh's own stderr is not in the report, so the reason is lost: "
        f"{msg[:200]}")


def test_the_report_NAMES_A_CLEARING_ACTION(monkeypatch):
    _, msg = _outcome(monkeypatch, _Result(rc=1, err=DEAD_TOKEN_STDERR))
    assert "CLEARING ACTION" in msg and "actions: read" in msg


# -- THE POSITIVE CONTROL ----------------------------------------------------

def test_POSITIVE_CONTROL_a_READABLE_list_EVALUATES_and_returns_rows(
        monkeypatch):
    """With the call succeeding, the helper gets out of the way."""
    body = ('[{"createdAt": "2026-11-05T07:29:45Z", "event": "schedule"},'
            ' {"createdAt": "2026-11-04T09:41:00Z", "event": '
            '"workflow_dispatch"}]')
    how, rows = _outcome(monkeypatch, _Result(rc=0, out=body))
    assert how == "PASSED", f"a readable list did not evaluate: {how} {rows}"
    assert len(rows) == 2 and rows[0]["event"] == "schedule"


# -- THE OTHER UNREADABLE SHAPES ---------------------------------------------

def test_an_EMPTY_list_is_UNEVALUATED_and_not_a_measurement(monkeypatch):
    """Zero rows does not distinguish `none` from `this query cannot match`."""
    how, msg = _outcome(monkeypatch, _Result(rc=0, out="[]"))
    assert how == "REPORTED" and msg.startswith("UNEVALUATED"), (
        f"an empty list produced {how}: {str(msg)[:200]}")


def test_OUTPUT_THAT_IS_NOT_JSON_is_UNEVALUATED(monkeypatch):
    how, msg = _outcome(monkeypatch, _Result(rc=0, out="<html>429</html>"))
    assert how == "REPORTED" and msg.startswith("UNEVALUATED"), (
        f"non-JSON produced {how}: {str(msg)[:200]}")


def test_the_helper_CHECKS_RETURNCODE_and_not_only_stdout():
    """The defect shape, pinned in the source as well as in behaviour.

    `code_of` is not used here on purpose: it lives in `conftest`, and
    importing conftest would put these controls in the capture gate. So the
    needles are CODE-SHAPED (`p.returncode`, `p.stderr`) rather than bare
    words, which a comment would not satisfy.
    """
    import inspect
    code = inspect.getsource(gh_json_or_unevaluated)
    assert "subprocess.run" in code, (
        "this is not the helper's body, so what follows verifies nothing")
    assert "p.returncode" in code, (
        "the helper does not look at `returncode`, so a 401, a 403 and a "
        "rate limit all become an empty stdout that reads as zero rows")
    assert "p.stderr" in code, "the reason for the failure is discarded"


# -- THE RATCHET -------------------------------------------------------------

LIVE_GH_CALLERS = ("test_absent_run_and_unparseable.py",
                   "test_fixture_horizon_probe.py")


def test_RATCHET_no_live_gh_caller_SKIPS_an_unreadable_run_list():
    """The skip must not come back, in either file."""
    for name in LIVE_GH_CALLERS:
        src = (TESTS / name).read_text(encoding="utf-8")
        # THE ANCHOR FIRST, inline rather than via `conftest.anchored`: a
        # negative claim about a file that no longer routes through the helper
        # would verify nothing.
        assert "gh_json_or_unevaluated" in src, (
            f"{name} no longer routes through the helper, so the absence "
            f"checks below are empty")
        for gone in ("run list unreadable here", "run history unreadable here",
                     "detector unevaluated here"):
            assert gone not in src, (
                f"{name} still skips on an unreadable list ({gone!r}). A skip "
                f"and a pass are the same thing in a green run.")


def test_the_REMINDERS_workflow_DECLARES_actions_read():
    """`gh run` is the Actions API, so the scope is stated, not inherited.

    It is not load-bearing today -- this repository is PUBLIC and run metadata
    is publicly readable, proved by run 42011987381 evaluating the probe
    reminder with no `actions` permission in force. It becomes load-bearing
    the moment the repository goes private, and a declared scope is the
    difference between `ci-audit` surviving that and these two reminders going
    quiet.
    """
    import yaml
    wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "reminders.yml")
        .read_text(encoding="utf-8"))
    perms = wf.get("permissions") or {}
    assert perms.get("actions") == "read", (
        f"reminders.yml must declare `actions: read`; got {perms!r}")
    assert perms.get("contents") == "read", (
        f"reminders.yml must declare `contents: read`; got {perms!r}")


def test_the_AUDIT_workflow_STILL_declares_actions_read_and_contents_READ():
    """The other direction: `contents` stays READ, because nothing commits.

    Measured 2026-11-05: no workflow in this repository contains `git push`,
    `git commit` or `git add`, and `ci_audit.py`'s only git calls are
    `rev-list` and `show`. `contents: write` would widen a public repository
    whose history already carries a leaked credential, for no consumer.
    """
    import yaml
    wf = yaml.safe_load(
        (ROOT / ".github" / "workflows" / "ci-audit.yml")
        .read_text(encoding="utf-8"))
    perms = wf.get("permissions") or {}
    assert perms.get("actions") == "read"
    assert perms.get("contents") == "read", (
        f"ci-audit.yml commits and pushes nothing, so `contents` stays read; "
        f"got {perms!r}")
