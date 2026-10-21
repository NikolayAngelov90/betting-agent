"""A red means "failure" only for workflows whose reds are failures.

    2026-10-21. Rule 1: the declaration is READ, never assumed.

`reminders.yml` exists to go red when a self-expiring guard reaches its date.
Alarming on that is `fixtures_zero_active` for the eighth time — an alarm that
fires daily with nothing a reader can do about it.

BUT SILENCING THE WORKFLOW IS NOT AVAILABLE. That is exactly what hid eight
`ci-audit` reds and the fixture-horizon probe's `BROKEN`, and it is why
`NEVER_ALARM_WORKFLOWS` is kept as small as it is. A crashed reminder workflow
must still be heard.

So the red is classified, three-valued, and BOTH DIRECTIONS ARE CONTROLLED —
one direction is not a control.
"""

import pathlib

import pytest

import scripts.ci_audit as ci

REM = ".github/workflows/reminders.yml"
MARKER = "##[error]REMINDER DUE"
EXIT = "##[error]Process completed with exit code 1."
PAD = "x" * (ci._MIN_AUDITABLE_LOG_BYTES + 50)


def _facts(**kw):
    f = {"exit_annotations": 1, "run_conclusion": "failure"}
    f.update(kw)
    return f


# ── the registry's own contract ─────────────────────────────────────────────

def test_the_marker_is_the_RENDERED_form_not_the_source_literal():
    """Rule 3. GitHub consumes the workflow command: the log holds
    `##[error]X`, and `::error::X` survives only in the `##[group]Run` echo of
    the step's own source. Pinning the source literal is what made the
    annotation matcher count echoed script text and miss every real
    annotation — measured on three runs, 2026-10-13.
    """
    for wf, (marker, why) in ci.DELIBERATE_RED_MARKERS.items():
        assert marker.startswith("##[error]"), (
            f"{wf}'s marker {marker!r} is the SOURCE form; the audit reads "
            f"the log, which carries the rendered form")
        assert "::error::" not in marker
        assert len(why) > 40, f"{wf}'s marker has no stated reason"


def test_the_marker_is_REGISTERED_as_self_emitted_and_PINNED_to_its_emitter():
    """Rule 4's bookkeeping half: the partition subtracts one exit per
    registered annotation, so the marker must also be in the registry that
    does the subtracting, and its emitter must really contain it."""
    src_form = MARKER.replace("##[error]", "::error::")
    reg = {pat: src for pat, src in ci.SELF_EMITTED_ANNOTATIONS}
    assert src_form in reg, (
        f"{src_form!r} is not in SELF_EMITTED_ANNOTATIONS, so the audit will "
        f"count this workflow's own marker as a FOREIGN annotation")
    emitter = pathlib.Path(reg[src_form])
    assert emitter.exists(), f"{emitter} does not exist"
    assert src_form in emitter.read_text(encoding="utf-8"), (
        f"{emitter} does not contain {src_form!r}")


def test_the_MARKER_PRECEDES_the_exit_it_explains():
    """Rule 4. A registered marker must co-occur with the non-zero exit it
    explains, or the subtraction loses headroom — the defect that had the
    partition reporting a phantom remainder of 1 on every red audit run.
    """
    text = pathlib.Path(REM).read_text(encoding="utf-8")
    i_mark = text.index("::error::REMINDER DUE")
    # the exit that follows it in the same branch
    tail = text[i_mark:]
    assert "sys.exit(1)" in tail, "the marker is not followed by a non-zero exit"
    # and nothing between them can return early
    between = tail[:tail.index("sys.exit(1)")]
    assert "sys.exit(0)" not in between, (
        "a clean exit sits between the marker and the failure it explains")


def test_UNCLASSIFIABLE_goes_somewhere_STATED():
    """The directive asked which side it falls on. It is stated in code, not
    left to the reader, and it is the conservative side."""
    assert ci.UNCLASSIFIABLE_ALARMS is True, (
        "UNCLASSIFIABLE no longer alarms; if that is deliberate, say so here "
        "and record why silence is safe for a red nobody could classify")


# ── KNOWN-WRONG CONTROL, DIRECTION 1: the deliberate red must NOT alarm ─────

def test_KNOWN_WRONG_a_reminders_red_WITH_the_marker_is_a_VERDICT():
    log = PAD + "\n" + MARKER + " a self-expiring guard reached its date\n" + EXIT
    state, why = ci.classify_red(REM, log, _facts())
    assert state == "VERDICT", f"{state}: {why}"
    assert "self-expiring" in why


# ── KNOWN-WRONG CONTROL, DIRECTION 2: a crash must alarm ───────────────────

@pytest.mark.parametrize("crash,label", [
    ("Traceback (most recent call last):\n  File \"x\", line 1\nValueError: x",
     "a traceback"),
    ("ERROR: Could not find a version that satisfies the requirement pytest",
     "a failed install"),
    ("yaml: line 7: did not find expected key", "a syntax error"),
    ("Error: Input required and not supplied: token", "a missing credential"),
])
def test_KNOWN_WRONG_a_reminders_red_WITHOUT_the_marker_is_a_FAILURE(crash,
                                                                    label):
    """One direction is not a control. A crashed reminder workflow carries no
    marker and must still be heard."""
    log = PAD + "\n" + crash + "\n" + EXIT
    state, why = ci.classify_red(REM, log, _facts())
    assert state == "FAILURE", f"{label} classified as {state}: {why}"
    assert "must still be heard" in why


def test_an_UNREGISTERED_workflows_red_is_always_a_FAILURE():
    state, why = ci.classify_red("daily-picks.yml", PAD + EXIT, _facts())
    assert state == "FAILURE"
    assert "no registered marker" in why


# ── the third state, three ways into it ────────────────────────────────────

def test_an_EMPTY_log_is_UNCLASSIFIABLE_not_a_verdict():
    """"The marker could not be read" is not "there was no marker"."""
    for log in (None, "", "   ", "short"):
        state, why = ci.classify_red(REM, log, _facts())
        assert state == "UNCLASSIFIABLE", f"{log!r} -> {state}"
        assert "could not be looked for" in why


def test_a_CANCELLED_run_is_UNCLASSIFIABLE():
    log = PAD + "\n" + EXIT
    state, why = ci.classify_red(REM, log, _facts(run_conclusion="cancelled"))
    assert state == "UNCLASSIFIABLE"
    assert "never reached the point" in why


def test_a_MARKER_WITHOUT_ITS_EXIT_is_UNCLASSIFIABLE_not_a_verdict():
    """Rule 4, enforced at read time as well as at registration time. A
    marker whose step then SUCCEEDED explains nothing: the red came from
    somewhere else and is unaccounted for."""
    log = PAD + "\n" + MARKER + "\n"
    state, why = ci.classify_red(REM, log, _facts(exit_annotations=0))
    assert state == "UNCLASSIFIABLE", f"{state}: {why}"
    assert "explains nothing" in why


# ── the wiring, so the classification actually reaches `alarmed` ───────────

def test_the_AUDIT_consults_the_classification_at_its_alarm_site():
    """AST, not prose: `classify_red` is called in `main`, and the VERDICT
    state is what suppresses the alarm."""
    import ast
    import inspect

    src = inspect.getsource(ci.main)
    tree = ast.parse("def _m():\n" + "\n".join(
        "    " + l for l in src.splitlines()[1:]))
    called = {n.func.id for n in ast.walk(tree)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "classify_red" in called, (
        "the audit never classifies a red, so every reminders run will alarm")
    assert 'VERDICT' in src, "nothing suppresses the alarm on a VERDICT"
    assert "UNCLASSIFIABLE" in src, "UNCLASSIFIABLE is not handled at the site"


def test_the_reminders_workflow_is_NOT_in_NEVER_ALARM_WORKFLOWS():
    """Rule 2. Silencing the workflow loses its own failures, and that cost
    has been paid twice."""
    assert REM not in ci.NEVER_ALARM_WORKFLOWS
    assert "reminders.yml" not in " ".join(ci.NEVER_ALARM_WORKFLOWS), (
        "the reminder workflow was silenced instead of classified, so a "
        "crashed reminders run is now invisible")
    # and it IS audited
    assert "reminders.yml" in ci.WORKFLOWS
