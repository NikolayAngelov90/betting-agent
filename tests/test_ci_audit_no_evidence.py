"""CLEAN cannot mean "nothing ran". Third instance of the same collapse.

FOUND 2026-09-13. Run 34745992077 (daily-picks, 09-13 07:44) had a ZERO-BYTE
cached log and `ci_audit --unaudited` reported it CLEAN. No assertion can fire
against an empty file, so `hits` was empty, so the run scored clean.

    A verdict from no evidence.

THE SAME COLLAPSE, THIRD TIME, AND ALL THREE IN THE ODDS/AUDIT PATH:

  * `[]` (this league priced nothing) versus `None` (the call failed)
  * 429-with-credits (rate limited) versus 429-with-zero (exhausted)
  * an empty result versus an unmeasured result

Every one returned the same value for "found nothing" and "could not look", and
every one was fixed by naming the third state rather than by widening a check.
`UNAUDITABLE` is that third state here: named and counted, so it takes its own
ledger row instead of inflating the CLEAN count — which is precisely how nine
DEGRADED runs went unnoticed in the manual pass this tool was built to replace.
"""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_ci_audit", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)


def test_an_empty_log_is_never_clean():
    """THE POSITIVE CONTROL. Feed it nothing; the verdict must not be CLEAN."""
    v = ci.verdict({}, [], log="")
    assert v != "CLEAN", (
        "an empty log scored CLEAN — the tool reported that its checks passed "
        "when no check could run")
    assert v == "UNAUDITABLE"


def test_whitespace_only_is_also_no_evidence():
    assert ci.verdict({}, [], log="   \n\n  \t ") == "UNAUDITABLE"


def test_a_truncated_log_is_no_evidence():
    """Short enough to carry no step output at all.

    A real daily-picks log is ~600k and the smallest real capture log ~10k, so
    the bound is not a quality threshold — it separates "a log" from "nothing".
    """
    assert ci.verdict({}, [], log="x" * 10) == "UNAUDITABLE"


def test_a_real_log_with_no_hits_is_still_CLEAN():
    """UNAUDITABLE must not swallow the ordinary clean run."""
    assert ci.verdict({}, [], log="y" * 5000) == "CLEAN"


def test_a_real_log_with_hits_is_DEGRADED():
    assert ci.verdict({}, ["something"], log="y" * 5000) == "DEGRADED"


def test_a_real_log_with_a_failed_step_is_BROKEN():
    assert ci.verdict({"steps_failed": 1}, [], log="y" * 5000) == "BROKEN"


def test_an_empty_log_outranks_broken():
    """A traceback count of 0 from an empty file is not evidence of no traceback.

    If the log is absent, every fact derived from it is absent too — including
    the ones that would have said BROKEN. UNAUDITABLE is checked FIRST for that
    reason, not as a precedence preference.
    """
    assert ci.verdict({"steps_failed": 0, "tracebacks": 0}, [], log="") == \
        "UNAUDITABLE"


def test_omitting_the_log_preserves_the_old_signature():
    """Callers that do not pass a log keep the previous behaviour."""
    assert ci.verdict({}, []) == "CLEAN"
    assert ci.verdict({}, ["hit"]) == "DEGRADED"


# ── IN_PROGRESS: the state UNAUDITABLE was hiding ─────────────────────────

def test_an_in_flight_run_is_IN_PROGRESS_not_unauditable():
    """Found 2026-09-13 on run 34745992077.

    Its log was empty because `gh run view --log` returns nothing until a run
    completes, NOT because anything failed — it was 20 minutes old and midway
    through step 15 of 31. UNAUDITABLE was reported and read as a possible
    incident; it was a run in flight.

    From the log alone the two are identical. Only the run metadata separates
    them, which is why this is checked BEFORE the log test.
    """
    for status in ("in_progress", "queued", "waiting", "pending", "requested"):
        assert ci.verdict({"run_status": status}, [], log="") == "IN_PROGRESS", (
            f"a run with status={status!r} reported something other than "
            "IN_PROGRESS — an executing run has no verdict to give")


def test_a_completed_run_with_an_empty_log_is_still_UNAUDITABLE():
    """IN_PROGRESS must not swallow the genuine no-evidence case."""
    assert ci.verdict({"run_status": "completed"}, [], log="") == "UNAUDITABLE"


def test_in_progress_outranks_a_failed_step():
    """A step that has not finished cannot have failed.

    Facts extracted from a partial log describe a partial run, so no verdict
    derived from them is final.
    """
    assert ci.verdict({"run_status": "in_progress", "steps_failed": 1},
                      [], log="y" * 5000) == "IN_PROGRESS"
