"""A check whose SCOPE comes from the repository must not depend on staging.

FOUND 2026-09-20, and it cost a day of picks.

`test_no_secrets_in_repo` enumerated its inputs with a plain `git ls-files`.
A new script carrying a dummy `api_key` literal was written, the suite was run
locally — **1072 passed** — the file was committed, and CI ran the same suite
at the same commit and reported **1 failed, 1071 passed**. The file was
untracked when the local run enumerated its inputs and tracked when CI did.
`git add` changed the test's population, not the code.

    THE COUNT WAS 1072 BOTH TIMES.

That is what makes this the sharpest of UNI-1's instances. The other four gave
DIFFERENT numbers from different units — a design effect of ~11, a fifty-fold
gap in n, 66-78 credits against 106-168 — and a number that moves invites the
question. This one gave the SAME number, and an unchanged figure is read as
confirmation. It was offered as evidence the commit was clean.

THE FIX IS THE SCOPE, NOT A REMINDER. `--others --exclude-standard` makes the
input set "every file that will reach the remote" rather than "every file
staged so far", so the answer no longer depends on when the suite runs.
Remembering to stage first is the alternative, and this project has measured
what remembering is worth.

This file pins the property for the existing caller and for any future one.
"""

import pathlib
import re
import subprocess

import pytest

TESTS = pathlib.Path("tests")
SRC_DIRS = [pathlib.Path("tests"), pathlib.Path("scripts"), pathlib.Path("src")]

# `git ls-files` with no `--others` answers "what is staged", which is a
# different question from "what will be committed".
LS_FILES = re.compile(r"""["']git["']\s*,\s*["']ls-files["'](?P<rest>[^\]]*)\]""")


def _py_files():
    for d in SRC_DIRS:
        for p in d.rglob("*.py"):
            if "__pycache__" in p.parts:
                continue
            # This file describes the shape and uses the bare form twice ON
            # PURPOSE — once to prove the probe is untracked, once to compute
            # the old scope for the superset check. Both ask "what is staged",
            # which is the correct question there.
            #
            # A DEFINITION IS NOT AN OCCURRENCE — the same exemption
            # `test_no_secrets_in_repo` takes for itself, and the same error
            # `_is_provisional` made when it matched a row whose NOTE
            # mentioned a verdict. Caught by this test failing on itself.
            if p.name == pathlib.Path(__file__).name:
                continue
            yield p


def test_no_check_scopes_itself_on_staged_files_alone():
    """THE GENERAL FORM. Any `git ls-files` caller must include untracked.

    If this fails on a new call site, the fix is `--others
    --exclude-standard`, not a note asking the next person to stage first.
    """
    offenders = []
    for p in _py_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in LS_FILES.finditer(text):
            rest = m.group("rest")
            if "--others" not in rest or "--exclude-standard" not in rest:
                line = text[:m.start()].count("\n") + 1
                offenders.append(f"{p.as_posix()}:{line}")
    assert not offenders, (
        "`git ls-files` without `--others --exclude-standard` — the check's "
        "scope then depends on what has been staged, so it gives one answer "
        "before `git add` and another after, at the same commit:\n  "
        + "\n  ".join(offenders))


def test_the_secrets_scan_sees_an_UNTRACKED_file():
    """THE POSITIVE CONTROL, and the exact case that was missed.

    Written to the real working tree (git-ignored name would defeat the
    point), scanned, then removed. If the scope regressed to staged-only this
    file would not appear and the assertion fails.
    """
    import tests.test_no_secrets_in_repo as mod

    probe = pathlib.Path("_scope_probe_untracked.py")
    assert not probe.exists()
    probe.write_text("# scope probe\n", encoding="utf-8")
    try:
        staged = subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True,
            encoding="utf-8", errors="replace").stdout
        assert probe.name not in staged, (
            "the probe is tracked — the test is no longer testing anything")
        seen = {name for name, _ in mod._tracked_text_files()}
        assert probe.as_posix() in seen, (
            "an untracked, non-ignored file was invisible to the secrets "
            "scan — a credential committed in one step would pass the suite "
            "run in the step before it")
    finally:
        probe.unlink()


def test_gitignored_files_stay_OUT_of_the_scan():
    """The widening must not reach trees that never leave the machine.

    `mcp-servers/` and `.env` are gitignored and hold real credentials; the
    scan must not read them, or it reports offenders nobody can act on and
    the suite fails permanently on a developer's machine.
    """
    import tests.test_no_secrets_in_repo as mod

    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".env"], capture_output=True)
    if ignored.returncode != 0:
        pytest.skip(".env is not gitignored in this checkout")
    seen = {name for name, _ in mod._tracked_text_files()}
    assert ".env" not in seen, (
        "the scan reached a gitignored file that holds real credentials")


def test_the_scan_is_a_SUPERSET_of_what_it_replaced():
    """Widening must not drop anything. Every tracked file is still scanned."""
    import tests.test_no_secrets_in_repo as mod

    tracked = {n for n in subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True,
        encoding="utf-8", errors="replace").stdout.splitlines() if n}
    seen = {name for name, _ in mod._tracked_text_files()}
    # Binary and unreadable files are skipped by the reader, not the lister,
    # so compare only against files the old scope would also have yielded.
    missing = [n for n in tracked
               if pathlib.Path(n).is_file()
               and n not in seen
               and _is_text(pathlib.Path(n))]
    assert not missing, f"tracked files dropped from the scan: {missing[:5]}"


def _is_text(path):
    try:
        path.read_text(encoding="utf-8")
        return True
    except (UnicodeDecodeError, OSError):
        return False
