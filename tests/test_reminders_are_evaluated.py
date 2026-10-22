"""`reminders` must distinguish a VERDICT from a FAILURE — 2026-10-22.

WHAT HAPPENED. `reminders.yml` had exactly one run, 39714443709, on
2026-10-22 at 03:00 UTC. It announced:

    ##[error]REMINDER DUE - a self-expiring guard reached its date.
    This red is a VERDICT, not a failure.
    📅 REMINDERS — A DECISION IS DUE.

NO REMINDER WAS DUE. Higher in the same log:

    collected 1613 items / 17 errors / 1607 deselected / 1 skipped / 6 SELECTED
    1 skipped, 1607 deselected, 17 errors in 4.16s

Seventeen test modules could not be IMPORTED -- the job installed a "minimal"
dependency set and pytest must import every module under `tests/` to find out
which ones carry the marker. pytest therefore stopped at collection and
**none of the six selected reminders ran**. It exited 2. `continue-on-error`
turned that into the word `failure`, and the classifier read `failure` as "a
reminder is due".

    An exit code of 2 means "I could not look". It is not a test result.

THREE DEFECTS, EACH WITH ITS OWN FIX, because any one of them alone would
have been enough to produce the false verdict:

  1. the classifier read the STEP OUTCOME, which collapses 1 and 2;
  2. the install could not collect the suite;
  3. nothing checked that the reminders had actually RUN -- so a reminder
     that silently loses its marker is still a clean exit 0 today.
"""

import ast
import pathlib
import re
import textwrap

import pytest

YML = pathlib.Path(".github/workflows/reminders.yml")
SRC = YML.read_text(encoding="utf-8")


# ── THE REGISTRY: which tests ARE the reminders ───────────────────────────

#: EVERY test in the `calendar or live` selection, `measured` 2026-10-22 by
#: `pytest -m "calendar or live" --collect-only`. Verified below BY AST, not
#: by grep: three other `@pytest.mark.calendar`/`live` decorators in the
#: repository sit inside STRING LITERALS (planted source in
#: `test_calendar_marker_enforced.py` and `test_calendar_separation.py`), and
#: a text scan counts those.
REMINDERS = {
    "tests/test_absent_run_and_unparseable.py": [
        "test_the_detector_does_not_call_a_day_with_a_RUN_missing"],
    "tests/test_calendar_separation.py": [
        "test_the_H1_RESERVATION_WINDOW_has_EXPIRED",
        "test_the_H1_RESERVATION_ACKNOWLEDGEMENT_EXPIRES"],
    "tests/test_cron_split_by_daytype.py": [
        "test_the_weekday_cron_CLEARS_the_stated_margin_and_LATEST_IS_RETIRED"],
    "tests/test_empty_card_suppression.py": [
        "test_the_suppression_has_EXPIRED"],
    "tests/test_fixture_horizon_probe.py": [
        "test_the_SIX_HOUR_control_reports_the_MEASURED_gap_per_run"],
    "tests/test_reference_deadline.py": [
        "test_LIVE_the_reference_answers_for_TOMORROW_and_our_table_does_not"],
}

REMINDERS_EXPECTED_MIN = sum(len(v) for v in REMINDERS.values())


def _marked_tests(path):
    """`{name: {markers}}` for top-level test functions, by AST."""
    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        marks = set()
        for dec in node.decorator_list:
            # @pytest.mark.<name>  and  @pytest.mark.<name>(...)
            target = dec.func if isinstance(dec, ast.Call) else dec
            if isinstance(target, ast.Attribute) \
                    and isinstance(target.value, ast.Attribute) \
                    and target.value.attr == "mark":
                marks.add(target.attr)
        out[node.name] = marks
    return out


@pytest.mark.parametrize("path", sorted(REMINDERS))
def test_every_registered_reminder_EXISTS_and_CARRIES_ITS_MARKER(path):
    found = _marked_tests(path)
    for name in REMINDERS[path]:
        assert name in found, (
            f"{path}::{name} is gone. If it was deliberately retired, remove "
            f"it from REMINDERS and lower the floor in reminders.yml in the "
            f"same change -- otherwise the floor silently stops being met")
        assert found[name] & {"calendar", "live"}, (
            f"{path}::{name} lost its calendar/live marker, so it has LEFT "
            f"the reminder selection. The job would exit 0 and report "
            f"'nothing due' while this guard sails past its date")


def test_the_FLOOR_in_the_workflow_MATCHES_the_registry():
    """Self-calibrating in the one direction that matters: the floor may not
    quietly drop below the number of reminders that exist."""
    m = re.search(r"REMINDERS_EXPECTED_MIN = (\d+)", SRC)
    assert m, "the floor is gone from the workflow"
    assert int(m.group(1)) == REMINDERS_EXPECTED_MIN == 7, (
        f"the workflow's floor is {m.group(1)} against {REMINDERS_EXPECTED_MIN} "
        f"registered reminders")


# ── driving the classifier ────────────────────────────────────────────────

def _classifier_source():
    """The `Say what it means` step's Python, lifted out of the YAML.

    Driven rather than text-scanned. The whole defect was a BRANCH reading
    the wrong variable, and a grep for `pytest_rc` would have passed on code
    that read it and then ignored it.
    """
    lines = SRC.splitlines()
    start = next(i for i, l in enumerate(lines) if "python - <<'EOF'" in l)
    end = next(i for i, l in enumerate(lines[start:], start)
               if l.strip() == "EOF")
    return textwrap.dedent(chr(10).join(lines[start + 1:end]))


class _FakeSubprocess:
    def __init__(self):
        self.sent = []

    def run(self, argv, **kw):
        self.sent.append(argv)
        return None


def _drive(rc, ran, up="success", event="workflow_run"):
    """`(exit_code, stdout, [telegram bodies])`."""
    import io
    import os
    import sys
    from contextlib import redirect_stdout

    fake = _FakeSubprocess()
    env = {"RC": str(rc), "RAN": str(ran), "UP_CONCL": up, "EVENT": event,
           "RUN_URL": "https://example/runs/1"}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    buf = io.StringIO()
    code = 0
    # ── THE FAKE GOES IN `sys.modules`, NOT IN THE GLOBALS. ───────────────
    #
    # Passing `{"subprocess": fake}` does NOT work and the first version of
    # this helper did exactly that: the extracted source starts with
    # `import subprocess`, which REBINDS the name over the injection. The
    # test then spawned the real `scripts.ci_alert` and got as far as its
    # `print` before a Windows codepage error stopped it -- a test that was
    # one encoding away from sending real Telegram messages on every run.
    real_sp = sys.modules.get("subprocess")
    sys.modules["subprocess"] = fake
    try:
        with redirect_stdout(buf):
            exec(compile(_classifier_source(), "<reminders.yml>", "exec"),
                 {"__name__": "__main__"})
    except SystemExit as e:
        code = e.code if e.code is not None else 0
    finally:
        if real_sp is None:
            sys.modules.pop("subprocess", None)
        else:
            sys.modules["subprocess"] = real_sp
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    bodies = [a[-1] for a in fake.sent]
    return code, buf.getvalue(), bodies


# ── the state that produced the false verdict ────────────────────────────

def test_KNOWN_WRONG_the_EXACT_FAILING_RUN_is_no_longer_a_verdict():
    """rc=2, 0 reminders run: run 39714443709's real state."""
    code, out, bodies = _drive(rc=2, ran=0)
    assert code == 1, "it must still redden; the job did not do its job"
    assert "REMINDER DUE" not in out, (
        "the VERDICT marker is still printed on a collection error, which is "
        "exactly the message that was wrong")
    assert "NOT EVALUATED" in out
    assert "pytest exited 2" in out
    assert "a module failed to IMPORT" in out, (
        "the reader is not told what exit 2 means, so they will read it as a "
        "test result again")
    assert bodies == [], (
        "a job that could not run still pushes to Telegram; that is run "
        "status")


@pytest.mark.parametrize("rc", [2, 3, 4, 5])
def test_every_NON_RESULT_exit_is_NOT_EVALUATED_and_SILENT(rc):
    code, out, bodies = _drive(rc=rc, ran=0)
    assert code == 1 and bodies == []
    assert "NOT EVALUATED" in out and "REMINDER DUE" not in out


def test_a_MISSING_exit_code_is_NOT_EVALUATED_rather_than_assumed():
    """`continue-on-error` plus a step that died before writing its output
    leaves RC empty. Empty is not zero."""
    code, out, bodies = _drive(rc="", ran="")
    assert code == 1 and bodies == []
    assert "did not reach this step" in out


# ── the two real results ─────────────────────────────────────────────────

def test_rc1_IS_the_verdict_and_DOES_send():
    code, out, bodies = _drive(rc=1, ran=7)
    assert code == 1
    assert "REMINDER DUE" in out, "the registered marker is gone"
    assert "VERDICT, not a failure" in out
    assert len(bodies) == 1, "a due reminder stopped reaching Telegram"
    assert "A DECISION IS DUE" in bodies[0]
    assert "NOT A PIPELINE FAILURE" in bodies[0]
    assert "its conclusion on this trigger: success" in bodies[0], (
        "the upstream state is no longer printed, so a reader cannot tell "
        "this from a capture failure")


def test_rc0_with_the_FULL_SET_is_nothing_due_and_GREEN():
    code, out, bodies = _drive(rc=0, ran=7)
    assert code == 0, out
    assert "nothing due" in out
    assert bodies == []
    assert "7 reminder test(s) ran" in out, (
        "the clean path does not say how many ran, so 'nothing due' and "
        "'nothing looked' read identically")


# ── the defect rc alone cannot catch ─────────────────────────────────────

@pytest.mark.parametrize("ran", [0, 1, 6])
def test_a_CLEAN_EXIT_with_TOO_FEW_reminders_is_NOT_EVALUATED(ran):
    """A reminder that loses its `calendar` marker leaves the selection and
    pytest exits 0. Without this, that is indistinguishable from a quiet day
    -- and it is the shape that hides a guard past its date."""
    code, out, bodies = _drive(rc=0, ran=ran)
    assert code == 1, (
        f"pytest exited 0 having run only {ran} of 7 reminders and this was "
        f"reported as a clean result")
    assert "NOT EVALUATED" in out
    assert f"only {ran} reminder test(s) RAN" in out
    assert bodies == []


def test_MORE_reminders_than_the_floor_is_FINE():
    """The floor is a minimum. Adding a reminder must not break the job."""
    code, _out, _b = _drive(rc=0, ran=9)
    assert code == 0


# ── the plumbing the classification depends on ───────────────────────────

def test_the_step_OUTCOME_is_NO_LONGER_READ():
    code_only = chr(10).join(
        l for l in SRC.splitlines() if not l.strip().startswith("#"))
    assert "steps.reminders.outcome" not in code_only, (
        "the step outcome is being read again; it is the single word that "
        "collapses 'a reminder is due' into 'pytest could not run'")
    assert "steps.reminders.outputs.pytest_rc" in code_only


def test_the_EXIT_CODE_SURVIVES_bash_e():
    """`set -o pipefail` alone is not enough. Under `bash -e` a failing
    pytest terminates the step AT the pipeline and the `rc=` line never
    runs -- the same family as the `| tee` that swallowed the coverage
    gate's exit code, in the opposite direction."""
    i = SRC.index("rc=${PIPESTATUS[0]}")
    head = SRC[max(0, i - 700):i]
    assert "set +e" in head, "the pipeline is not bracketed, so rc is unset"
    assert "set -o pipefail" in head


def test_the_INSTALL_can_COLLECT_the_suite():
    """The four modules whose absence stopped collection, each named with the
    import that needs it."""
    for pkg in ("aiohttp", "beautifulsoup4", "httpx", "pandas"):
        assert pkg in SRC, (
            f"{pkg} left the reminders install; 17 test modules import a "
            f"scraper and pytest must import every module to find the marked "
            f"ones")
    assert "17 errors" in SRC, (
        "the measurement that justifies those four is no longer recorded")


def test_the_NOT_EVALUATED_marker_is_NOT_REGISTERED_as_a_deliberate_red():
    """`ci-audit` returns FAILURE for a red that lacks the registered marker,
    which is the correct classification for this one. Registering it would
    re-create the original defect in the audit instead of the workflow."""
    import scripts.ci_audit as ci

    marker, _why = ci.DELIBERATE_RED_MARKERS[".github/workflows/reminders.yml"]
    assert marker == "##[error]REMINDER DUE"
    assert "NOT EVALUATED" not in marker
    state, why = ci.classify_red(
        ".github/workflows/reminders.yml",
        "x" * 5000 + chr(10) + "##[error]REMINDERS NOT EVALUATED",
        {"exit_annotations": 1, "run_conclusion": "failure"})
    assert state == "FAILURE", (state, why)


def test_the_RUN_THAT_LIED_is_recorded_in_the_audit_beside_the_table():
    """So the next reader of `DELIBERATE_RED_MARKERS` knows a present marker
    is not by itself evidence."""
    audit = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    assert "39714443709" in audit
    assert "ONLY AS HONEST AS THE PREDICATE THAT PRINTS IT" in audit
    assert "NO REMINDER WAS DUE" in audit, (
        "the record no longer states the conclusion that matters")


# ── what reddens next, so the date is not rediscovered ───────────────────

def test_the_NEXT_SCHEDULED_RED_is_recorded_and_is_2026_11_01():
    """`measured` 2026-10-22: of the seven reminders, the only one driven by
    a DATE that has not yet passed is `H1_RESERVATION_ACK_UNTIL`. The other
    two calendar guards already fired and had their decisions taken
    (`EMPTY_CARD_UNTIL` 2026-10-09, the H1 window 2026-10-17/acknowledged
    10-18); the remaining four are `live` and depend on data, not the clock.
    """
    import datetime as dt

    from tests.experiment_pins import H1_RESERVATION_ACK_UNTIL

    assert H1_RESERVATION_ACK_UNTIL == dt.date(2026, 11, 1)
    assert H1_RESERVATION_ACK_UNTIL > dt.date(2026, 10, 22), (
        "the H1 reservation acknowledgement has expired; `reminders` is now "
        "red BY VERDICT and the clearing action is the deletion stage")
