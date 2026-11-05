"""`gh ... --json ...` read so that an UNREADABLE result is REPORTED.

WHY THIS IS ITS OWN MODULE AND NOT IN `conftest.py`. 2026-11-05.

It was written into `conftest.py` first, and that pulled 25 tests into the
capture gate -- all 11 controls plus every static test in
`test_absent_run_and_unparseable.py`. `tests.conftest` is a HELPER under
`scripts/capture_critical.classify`, and it reaches `src.data.odds_quota`, so
ANY test file importing it becomes `capture_critical`. `_modules_imported`
walks the whole AST, so hiding the import inside a function does not help.

    THE IMPORT-PULLS-A-FILE-INTO-THE-GATE TRAP, FOURTH OCCURRENCE.

A test about whether a reminder reports UNEVALUATED must not be able to stop
price capture -- the same judgement that split the sealed-chat file on
2026-11-04. So this module imports the standard library and nothing else, and
`_on_capture_path` of its imports is empty by construction.
"""
import os

def gh_json_or_unevaluated(args, what="the run list"):
    """`gh ... --json ...` parsed -- or an UNEVALUATED **failure**, never a skip.

    THE THIRD STATE, IN PYTEST'S OWN VOCABULARY. 2026-11-05.

    Both live `gh` callers used to end an unreadable run list with
    `pytest.skip("run list unreadable here")`. A SKIP AND A PASS ARE
    INDISTINGUISHABLE IN A GREEN RUN: the reminder reports nothing, the
    workflow stays green, and the check has stopped checking without saying
    so. That is `a matcher with zero hits`, written in pytest.

    It is not hypothetical. On 2026-10-21 a reminder reported "the weekday
    series shrank to n=0" from a step that had no token, turning the absence
    of a credential into a finding about football. A token was then supplied
    -- but the guard keyed on the token being PRESENT, never on the call
    SUCCEEDING, so a token that is present and DEAD reads exactly like a
    healthy one. A deleted PAT is therefore invisible until something reports
    it, which is why this is reported and not skipped.

    So this RAISES, like `anchored` and `must_match`, and its message leads
    with UNEVALUATED so a reader can tell "could not look" from "a reminder is
    due" -- the same distinction `reminders.yml` already draws on pytest's
    exit code.

    `returncode` AND `stderr`, NOT `stdout or ""`. `ci_audit._sh` returns
    `r.stdout or ""`, which turns a 401, a 403 and a rate limit alike into an
    empty string that `json.loads` then reads as zero rows. Three entries in
    this register have that shape.
    """
    import json
    import subprocess

    def _seen():
        return (f"GH_TOKEN set: {bool(os.environ.get('GH_TOKEN'))}, "
                f"GITHUB_TOKEN set: {bool(os.environ.get('GITHUB_TOKEN'))}")

    cmd = ["gh", *args]
    shown = " ".join(cmd)
    try:
        p = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                           errors="replace", timeout=180)
    except FileNotFoundError:
        raise AssertionError(
            f"UNEVALUATED, not clear: the `gh` CLI is absent, so {what} could "
            f"not be read. This is 'could not look', NOT 'nothing there'. "
            f"CLEARING ACTION: install `gh`, or run this where it exists. "
            f"({_seen()})")
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"UNEVALUATED, not clear: `{shown}` did not return within 180s, "
            f"so {what} could not be read. ({_seen()})")
    if p.returncode != 0:
        raise AssertionError(
            f"UNEVALUATED, not clear: `{shown}` exited {p.returncode}, so "
            f"{what} could not be read. This is 'could not look', NOT "
            f"'nothing there', and it is exactly the state a DELETED OR DEAD "
            f"token produces. CLEARING ACTION: give the job a live token and, "
            f"for `gh run`, the `actions: read` permission. ({_seen()}) "
            f"stderr: {(p.stderr or '').strip()[:400]!r}")
    try:
        rows = json.loads(p.stdout or "")
    except json.JSONDecodeError:
        raise AssertionError(
            f"UNEVALUATED, not clear: `{shown}` exited 0 but its output is "
            f"not JSON, so {what} could not be read. ({_seen()}) first 200 "
            f"bytes: {(p.stdout or '')[:200]!r}")
    if not rows:
        raise AssertionError(
            f"UNEVALUATED, not clear: {what} came back EMPTY. Zero rows does "
            f"not distinguish 'no such runs' from 'this query can no longer "
            f"match', and nothing can be measured from it either way. "
            f"({_seen()})")
    return rows
