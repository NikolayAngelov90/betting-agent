"""The command echo is source, not output — 2026-10-23.

GitHub prints every step's shell source into the log inside
`##[group]Run …` / `##[endgroup]`. The audit matched its patterns against that
echo, so it read its own comments, its own pattern literals and its own test
IDs as findings. At least three false findings came from this one source, and
comment-blindness had been applied to the CODE and never to the LOG.

> A detector that reads logs containing its own source will match itself.
> Comment-blindness is a property of the input path, not of the scanner.

AND THE FIRST ATTEMPT AT THE FIX PASSED ITS UNIT TEST AND CHANGED NOTHING.
It dropped a line only if it carried the echo's ANSI colour, and `fetch_log`
runs `ANSI.sub("", …)` before `extract` ever sees the log. So every test here
drives the data down the path it really takes.
"""

import pathlib
import re

import pytest

import scripts.ci_audit as ci

#: The log shape `fetch_log` returns: tab-separated `job<TAB>step<TAB>stamp
#: text`, ANSI already stripped. Built from real 2026-10-22/23 lines.
PREFIX = "daily-picks\tDecide the run colour from the critical steps\t"


def _line(text, stamp="2026-10-23T06:37:23.3255273Z"):
    return f"{PREFIX}{stamp} {text}"


# ── KNOWN-WRONG CONTROL 1: the comment about the removed sender ───────────

ECHOED_COMMENT = "\n".join([
    _line("##[group]Run # EXIT 2 IS A CRASH"),
    _line('    parts.append("step(s) FAILED — " + ", ".join(failed))'),
    _line('# It sent "⚠️ Daily picks: step(s) FAILED — update, '
          'picks". A list of'),
    _line("# step names and their outcomes is run status in its purest form."),
    _line("##[endgroup]"),
    _line("daily picks step states: all critical steps OK"),
]) + "\n"


def test_CONTROL_1_the_echoed_COMMENT_is_no_longer_a_finding():
    """On 2026-10-23 the audit reported `core step(s) reported failure:
    update, picks` for a run whose every critical step SUCCEEDED. The matched
    line was a comment written the day before, ABOUT the sender being removed.

    The 2026-10-13 remedy -- require `[A-Za-z]` after the dash -- excludes the
    emitting source line and NOT a comment with a real step name in it.
    """
    before = re.findall(r"step\(s\) FAILED [—-] ([A-Za-z][^\n]{0,120})",
                        ECHOED_COMMENT)
    assert before, "the fixture no longer reproduces the false match"
    assert ci.extract(ECHOED_COMMENT)["steps_failed"] == [], (
        f"the echoed comment is still a finding: {before}")


# ── KNOWN-WRONG CONTROL 2: the pytest ID parametrised on "Traceback" ──────

PYTEST_ID = "\n".join([
    "daily-picks\tRun tests\t2026-10-23T05:37:05.4085322Z "
    "tests/test_red_is_classified_per_workflow.py::"
    "test_KNOWN_WRONG_a_reminders_red_WITHOUT_the_marker_is_a_FAILURE"
    "[Traceback (most recent call last)] PASSED [ 84%]",
    "daily-picks\tRun tests\t2026-10-23T05:37:05.4221003Z "
    "tests/test_red_is_classified_per_workflow.py::test_another PASSED [ 84%]",
]) + "\n"


def test_CONTROL_2_a_pytest_ID_parametrised_on_Traceback_is_not_one():
    """A test PARAMETRISED ON THE STRING "Traceback (most recent call last)"
    was counted as a traceback, which made a green run BROKEN.

    Every character of a pytest `-v` progress line is a test name or a
    parameter; none of it is program output.
    """
    assert "Traceback (most recent call last)" in PYTEST_ID
    assert ci.extract(PYTEST_ID).get("tracebacks", 0) == 0, (
        "a test id is still being counted as a traceback")


# ── KNOWN-WRONG CONTROL 3: `870` inside a run id ─────────────────────────

def test_CONTROL_3_a_digit_run_inside_a_longer_one_is_not_a_literal():
    """The SOURCE-side instance of the same class, from 2026-10-18.

    `870` as a bare substring matched the RUN ID `38708288664` inside a
    `RESOLVED_RUNS` comment and reported a deadline literal that was not
    there. Fixed by word-bounding; this pins the boundary, because the class
    is identical -- a scanner matching inside a longer token.
    """
    hay = "RESOLVED_RUNS = { '38708288664': ('7514a50', 'the outage') }"
    assert re.search(r"870", hay), "the fixture no longer contains the trap"
    assert not re.search(r"(?<![0-9])870(?![0-9])", hay), (
        "the word boundary no longer excludes a digit run inside a run id")
    # and the guard that enforces it is still in the suite
    src = pathlib.Path("tests/test_prospective_deadline.py").read_text(
        encoding="utf-8")
    assert "(?<![0-9])" in src, (
        "the deadline-literal scan lost its word boundary, so a run id can "
        "report a literal again")


# ── POSITIVE CONTROL: the signal outside the echo must survive ───────────

REAL_SIGNAL = "\n".join([
    _line("##[group]Run python -m scripts.ci_audit"),
    _line("# a comment mentioning ::error:: and Traceback, inside the echo"),
    _line("##[endgroup]"),
    _line("Traceback (most recent call last):"),
    _line('  File "scripts/ci_audit.py", line 1, in main'),
    _line("KeyError: 'binding'"),
    _line("##[error]CI AUDIT CRASHED - this is MACHINERY"),
    _line("##[error]Process completed with exit code 2."),
]) + "\n"


def test_POSITIVE_CONTROL_a_real_traceback_and_error_SURVIVE_the_strip():
    """A strip that removes the signal with the noise is the worse failure."""
    out = ci.strip_non_output(REAL_SIGNAL)
    assert "Traceback (most recent call last):" in out
    assert "##[error]CI AUDIT CRASHED" in out
    assert "inside the echo" not in out, "the echo was not stripped at all"
    f = ci.extract(REAL_SIGNAL)
    assert f.get("tracebacks") == 1, f.get("tracebacks")
    # ONE `error_annotation` AND ONE `exit_annotation`, not two of the first:
    # `##[error]Process completed with exit code 2.` is counted as the EXIT,
    # which is the partition working -- an exit and the annotation that
    # explains it are different terms and pooling them would lose the
    # `unexplained_nonzero_exit` residual.
    assert f.get("error_annotations", 0) >= 1, f.get("error_annotations")
    assert f.get("exit_annotations", 0) >= 1, f.get("exit_annotations")


def test_the_strip_leaves_NON_RUN_GROUPS_alone():
    """Only `##[group]Run ` is an echo. `##[group]Operating System` and its
    siblings carry real information about the runner."""
    log = "\n".join([
        _line("##[group]Operating System"),
        _line("Ubuntu 24.04.5"),
        _line("##[endgroup]"),
    ]) + "\n"
    out = ci.strip_non_output(log)
    assert "Ubuntu 24.04.5" in out, out


def test_an_UNTERMINATED_group_does_not_eat_the_rest_of_the_log():
    """A truncated log can have a group that never closes. Stripping to EOF
    would silently remove every later finding."""
    log = "\n".join([
        _line("##[group]Run something"),
        _line("echoed source line"),
        _line("Traceback (most recent call last):"),
    ]) + "\n"
    # it IS inside an unterminated group, so it is dropped -- and the test
    # records that, because the alternative (keeping it) re-opens the class.
    out = ci.strip_non_output(log)
    assert "echoed source line" not in out
    assert out.count("\n") <= 1, (
        "an unterminated group must not leak echo lines; if this changes, "
        "say which is preferred and why")


# ── END TO END, on the path the data takes ──────────────────────────────

def test_the_STRIP_RUNS_INSIDE_extract_not_at_the_call_site():
    """The property that makes it unmissable: every caller of `extract` gets
    the strip, including the pattern self-test."""
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    i = src.index("def extract(log: str)")
    body = src[i:i + 700]
    assert "log = strip_non_output(log)" in body, (
        "the strip left `extract`; a caller that forgets it reopens the class")


def test_fetch_log_STRIPS_ANSI_which_is_why_the_COLOUR_fix_did_nothing():
    """Recorded as a test because it is the lesson, not a detail.

    The first fix dropped a line only if it carried `\\x1b[36;1m`. It passed a
    direct test against a raw `gh run view --log` and changed NOTHING in
    production, because `fetch_log` strips ANSI first.
    """
    # ── COMMENT-BLIND, AND THE IRONY IS THE POINT. ──────────────────────
    #
    # The first version of this assertion read `"_ECHO_COLOUR" not in src`
    # against the raw file and matched THE COMMENT RECORDING THAT
    # `_ECHO_COLOUR` HAD BEEN DELETED -- a comment-blindness failure inside
    # the test about comment-blindness, and the eleventh instance in this
    # programme. `conftest.code_of` tokenises away comments and docstrings.
    from tests.conftest import code_of
    raw = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    src = code_of(raw)
    i = src.index("def fetch_log(")
    assert 'ANSI.sub("", ' in src[i:i + 500], (
        "fetch_log no longer strips ANSI; re-check whether a colour-based "
        "filter would now work, rather than assuming either way")
    assert "_ECHO_COLOUR" not in src, (
        "the colour constant is back in CODE; it had no consumer once the "
        "strip widened to the whole group region")
    # the group boundaries, which DO survive ANSI stripping, are what is used
    assert '"##[group]Run "' in src and '"##[endgroup]"' in src


#: MEASURED 2026-10-23 end to end over the 156 cached real logs in `ci_logs/`,
#: comparing `extract()` with the strip against `extract()` without it. EXACTLY
#: THREE FACTS IN THREE LOGS MOVED, every one of them a false positive:
#:
#:   tracebacks    1 -> 0   in 2 logs  (the pytest-ID parameter)
#:   steps_failed  ['update, picks". A list of'] -> []   in 1 log  (the comment)
#:
#: Nothing else changed, and NO `PATTERNS` entry matched only inside the echo
#: on any of the 156 logs -- so the strip removed the noise and kept the
#: signal. The figures are recorded here because `ci_logs/` is a local cache
#: and is not committed.
STRIP_EFFECT_MEASURED = {
    "logs_examined": 156,
    "facts_changed": {"tracebacks": 2, "steps_failed": 1},
    "patterns_matching_only_inside_the_echo": 0,
}


def test_the_MEASURED_EFFECT_is_recorded_with_its_window():
    m = STRIP_EFFECT_MEASURED
    assert m["logs_examined"] == 156
    assert sum(m["facts_changed"].values()) == 3, (
        "the recorded effect changed; re-measure over ci_logs/ rather than "
        "editing the number")
    assert m["patterns_matching_only_inside_the_echo"] == 0, (
        "a pattern now matches ONLY inside the echo, so the strip is "
        "removing signal; name that pattern and keep its case explicitly")


# ── THE OTHER DIRECTION: does the audit match a file it also writes? ─────

#: ASKED ONCE, ACROSS EVERY CANDIDATE, and measured 2026-10-23. The
#: `KNOWN_INSTANCES` self-suppression in `pnc_scan` was exactly this shape, so
#: the question is whether anything else has it.
SELF_MATCH_CENSUS = {
    "pnc_scan / KNOWN_INSTANCES": (
        "WAS REAL: a blanket string-as-reference visitor made the scanner's "
        "own table of symbol NAMES suppress its own findings. Already "
        "guarded -- `visit_Constant` must stay absent, pinned by "
        "test_pnc_detector."),
    "ledger / ledger_history": (
        "CLEAN, measured: 498 parsed rows, every id 10-12 digits, five real "
        "workflow names, no row from prose or from the markdown tables the "
        "entries contain."),
    "workflow crons / current_crons_by_daytype": (
        "CLEAN BY CONSTRUCTION: `^\\s*- cron:` anchors to the line start, so "
        "a commented-out cron cannot match. Driven with a file carrying a "
        "commented `0 2` -- the rejected option, which is exactly what a "
        "comment there would hold -- and the parse was unchanged."),
    "ci-audit's own run logs": (
        "BY DESIGN: `ci-audit` is in WORKFLOWS, so it audits itself. 76 of "
        "the 156 cached logs are its own, and every fact it extracts from "
        "them is either a partition term that exists for exactly that "
        "purpose (`self_emitted_annotations`, `self_emitted_exits`) or a "
        "REAL traceback -- 8 of 76, all from the two crash episodes of "
        "2026-10-22/23."),
    "RESOLVED_RUNS / the pattern registry": (
        "NO RUNTIME SELF-SOURCE READ. The registry self-tests through "
        "`PATTERN_EXAMPLES`, which is the point. The `870` case was a TEST "
        "scanning `ci_audit.py`, fixed by word-bounding -- see CONTROL 3."),
}


def test_the_SELF_MATCH_CENSUS_covers_every_candidate_with_a_verdict():
    assert len(SELF_MATCH_CENSUS) == 5
    for name, verdict in SELF_MATCH_CENSUS.items():
        assert verdict.strip(), name
        assert any(w in verdict for w in
                   ("CLEAN", "BY DESIGN", "WAS REAL", "NO RUNTIME")), (
            f"{name} has no verdict, only prose")


def test_the_pnc_SELF_SUPPRESSION_guard_is_still_in_place():
    """The one that was real. If it regresses, the detector stops being able
    to measure itself."""
    src = pathlib.Path("scripts/pnc_scan.py").read_text(encoding="utf-8")
    assert "visit_Constant" not in src, (
        "the blanket string-as-reference visitor is back; pnc_scan's own "
        "KNOWN_INSTANCES table would suppress its findings again")


def test_the_CRON_PARSE_is_anchored_so_a_COMMENTED_cron_cannot_match():
    """Driven, not read: the comment in `daily-picks.yml` documents the
    REJECTED `0 2`, which is precisely the value a self-match would pick up."""
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    assert "`0 2`" in wf, (
        "the rejected cron is no longer documented there, so this control "
        "no longer exercises the risk it was written for")
    poisoned = wf.replace("    - cron: '0 1 * * 1-5'",
                          "    # - cron: '0 2 * * 1-5'\n"
                          "    - cron: '0 1 * * 1-5'")
    found = re.findall(r"^\s*- cron:\s*'([^']+)'", poisoned, re.M)
    assert "0 2 * * 1-5" not in found, (
        f"a commented-out cron parsed as live: {found}")
    assert "0 1 * * 1-5" in found and "30 0 * * 6,0" in found, found


@pytest.mark.live
def test_LIVE_the_strip_still_removes_only_NOISE_on_the_real_cache():
    """The end-to-end check re-run against whatever `ci_logs/` holds now.

    `live` because it reads a local cache that CI does not have; the committed
    figures in `STRIP_EFFECT_MEASURED` are the record.
    """
    logs = sorted(pathlib.Path("ci_logs").glob("run_*/full.log"))
    if not logs:
        pytest.skip("UNEVALUATED: no cached logs here, so the end-to-end "
                    "comparison cannot be made. The committed figures in "
                    "STRIP_EFFECT_MEASURED are the record.")
    real = ci.strip_non_output
    moved = {}
    for f in logs:
        raw = ci.ANSI.sub("", f.read_text(encoding="utf-8", errors="replace"))
        after = ci.extract(raw)
        ci.strip_non_output = lambda s: s
        try:
            before = ci.extract(raw)
        finally:
            ci.strip_non_output = real
        for k in set(before) | set(after):
            if before.get(k) != after.get(k):
                moved.setdefault(k, 0)
                moved[k] += 1
    assert set(moved) <= {"tracebacks", "steps_failed", "steps_not_run"}, (
        f"the strip changed a fact it was not expected to: {moved}. Either a "
        f"new false positive was removed (record it) or SIGNAL is being lost.")
