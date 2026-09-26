"""A step absorbed by `continue-on-error` must not be invisible to the audit.

FOUND 2026-09-26, by sizing the blind spot rather than by an incident.

A step's `outcome` is NOT exposed by the GitHub REST API. Verified against it:
the step fields are `completed_at, conclusion, name, number, started_at,
status`. `outcome` is a workflow-expression concept and does not survive into the
record. So under `continue-on-error` a FAILING step reports
`conclusion: success`, and post-hoc triage from the API alone cannot tell it from
a passing one.

`daily-picks` carries NINE such steps. They recover three different ways:

    5 of 9   the outcome is echoed into an `env:` block and is in the log
    2 of 9   the outcome is consumed by a later step's `if:`, so it is
             inferable from whether that step ran
    3 of 9   ONLY from the `##[error]` annotation — the camoufox download,
             the Claude Code CLI install, and the weekly report

    `grep -n 'error\\]' scripts/ci_audit.py` returned NOTHING.

The tool could not read the one signal that closes the last slice of a blind
spot it exists to cover. Rule 1, in the tool, on its own gap.

TWO KINDS OF `##[error]`, AND CONFLATING THEM WOULD BREAK IT. A step exiting
non-zero and a script deliberately annotating an error are different facts, and
this pipeline emits the second on purpose — DEL-2's `::error::Pick generation
FAILED` and the audit's own `::error::audit alarm` both render as `##[error]`.
"""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_absorbed", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

PAD = "x\n" * 200                      # clears the minimum-auditable-log floor
EXIT = "##[error]Process completed with exit code 1.\n"


def _run(log):
    f = ci.extract(log)
    hits = ci.assertions(f, [])
    return ci.verdict(f, hits, log), f, hits


def _absorbed(hits):
    return [h for h in hits if "ABSORBED" in h]


# ── THE POSITIVE CONTROL ─────────────────────────────────────────────────────

def test_an_absorbed_failure_CHANGES_THE_VERDICT():
    """THE CONTROL THE FIX WAS REQUIRED TO HAVE.

    The same log with and without the annotation must not audit the same. If
    the verdict is unchanged, the pattern is decoration.
    """
    before, _, _ = _run(PAD)
    after, f, hits = _run(PAD + EXIT)
    assert before == "CLEAN"
    assert after == "DEGRADED", (
        "a step exited non-zero and the verdict did not move — the annotation "
        "is parsed but nothing acts on it, which is PNC-1")
    assert f["steps_nonzero_exit"] == 1
    assert _absorbed(hits), hits


def test_the_finding_NAMES_why_the_api_cannot_see_it():
    """A reader has to know why this line exists and not look for a step."""
    _, _, hits = _run(PAD + EXIT)
    text = _absorbed(hits)[0]
    assert "continue-on-error" in text
    assert "conclusion: success" in text


def test_several_absorbed_failures_are_counted():
    _, f, hits = _run(PAD + EXIT + EXIT + EXIT)
    assert f["steps_nonzero_exit"] == 3
    assert "3 step(s)" in _absorbed(hits)[0]


# ── THE TWO KINDS OF ANNOTATION ──────────────────────────────────────────────

def test_a_DELIBERATE_annotation_is_not_an_absorbed_failure():
    """DEL-2 emits `::error::Pick generation FAILED` on purpose.

    Counting it here would make every red run look like it had an absorbed
    failure — a definition read as an occurrence, one level down.
    """
    _, f, hits = _run(
        PAD + "##[error]Pick generation FAILED — no picks were produced.\n")
    assert f["steps_nonzero_exit"] == 0
    assert f["error_annotations"] == 1
    assert not _absorbed(hits)


def test_the_audits_OWN_alarm_line_is_not_counted():
    """`::error::audit alarm — …` is this tool's output, not a step failure."""
    _, f, hits = _run(PAD + "##[error]audit alarm — 123 closing-lines BROKEN\n")
    assert f["steps_nonzero_exit"] == 0
    assert not _absorbed(hits)


def test_the_two_counts_partition_the_annotations():
    """Every `##[error]` is one kind or the other, never both and never neither."""
    log = PAD + EXIT + "##[error]Pick generation FAILED.\n" + EXIT
    _, f, _ = _run(log)
    total = log.count("##[error]")
    assert f["steps_nonzero_exit"] + f["error_annotations"] == total == 3


# ── THE GUARDS, WHICH THE FIRST VERSION DID NOT HAVE ─────────────────────────

def test_a_run_with_a_TRACEBACK_does_not_also_report_absorbed():
    """THE REAL-LOG CASE THAT CAUGHT IT.

    The 09-24 closing-lines psycopg failures exited non-zero — and
    `closing-lines` carries ZERO continue-on-error steps, so nothing was
    absorbed and the word was wrong. The failure was already BROKEN on its
    tracebacks. The claim is about a failure that is OTHERWISE INVISIBLE.
    """
    verdict, _, hits = _run(
        PAD + EXIT + "Traceback (most recent call last)\nModuleNotFoundError\n")
    assert verdict == "BROKEN"
    assert not _absorbed(hits), (
        "reported an absorbed failure on a run whose traceback already made it "
        "visible — that is a second line for one fact, with the wrong word")


def test_a_DID_NOT_RUN_run_does_not_also_report_absorbed():
    """A halted job exits non-zero; the skip detection already names it."""
    verdict, _, hits = _run(
        PAD + EXIT + "All critical steps OK: {'update': 'skipped', "
        "'picks (incl. review)': 'skipped'}\n")
    assert verdict == "DID_NOT_RUN"
    assert not _absorbed(hits)


def test_a_named_failed_core_step_does_not_also_report_absorbed():
    verdict, _, hits = _run(
        PAD + EXIT + "⚠️ Daily picks: step(s) FAILED — update. Logs: x\n")
    assert verdict == "BROKEN"
    assert not _absorbed(hits)


def test_it_is_NOT_self_calibrating():
    """One absorbed failure is wrong on the first occurrence.

    Passed an empty history, as `--run` does, it must still fire — like a spent
    credit that returned no rows.
    """
    f = ci.extract(PAD + EXIT)
    assert _absorbed(ci.assertions(f, [])), "needed history to fire"


# ── THE REAL LOGS MUST STAY QUIET ────────────────────────────────────────────

def test_a_clean_log_reports_nothing():
    verdict, f, hits = _run(PAD)
    assert verdict == "CLEAN"
    assert f["steps_nonzero_exit"] == 0 and f["error_annotations"] == 0
    assert not _absorbed(hits)


def test_the_api_still_does_not_expose_step_outcome():
    """The premise, pinned as a comment-bearing assertion.

    If GitHub ever adds `outcome` to the steps payload, this whole pattern can
    be replaced by reading it — and this test is where a future reader finds
    that out rather than re-deriving it.
    """
    import inspect
    src = inspect.getsource(ci.extract)
    assert "NOT exposed by the REST API" in src, (
        "the reason this pattern exists is no longer recorded beside it")
