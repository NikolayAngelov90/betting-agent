"""A run whose core steps were SKIPPED is neither CLEAN nor BROKEN.

FOUND 2026-09-20. Run 35498465743 halted at `Run tests` — the only core step
with no `continue-on-error` — so update, settle and picks never executed. The
workflow's own alert printed `All critical steps OK`, `ci_audit` found no
failure marker and no traceback, and the run was listed:

    35498465743  daily-picks  2026-09-20T08:02  CLEAN

while GitHub reported `conclusion: failure` and the day produced zero picks,
zero odds rows and zero match rows.

FOURTH INSTANCE OF THE COLLAPSE IN THIS TOOL, and this time the three-valued
field is GitHub's own `outcome`:

  * ``[]`` vs ``None``                 measured-and-empty vs never-measured
  * 429-with-credits vs 429-with-zero  rate-limited vs exhausted
  * empty log vs unmeasured log        UNAUDITABLE
  * **success / failure / SKIPPED**    nothing ran is not nothing crashed

`BROKEN` says something exited non-zero. `CLEAN` says the checks ran and found
nothing. Neither describes a run where the steps did not execute, so it gets
its own verdict and its own ledger row rather than inflating either count.

BOTH LOG SHAPES ARE READ. The workflow now emits `step(s) DID NOT RUN — …`,
but every log written before 2026-09-20 carries only the printed outcome dict,
and the ledger has to stay readable across that change.
"""

import importlib.util
import pathlib

_spec = importlib.util.spec_from_file_location(
    "_ci_audit_skipped", pathlib.Path("scripts/ci_audit.py"))
ci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ci)

PAD = "x\n" * 200          # clears the minimum-auditable-log floor

# The exact line run 35498465743 printed.
REAL_0920 = (
    "All critical steps OK: {'update': 'skipped', 'settle (pre-picks)': "
    "'skipped', 'picks (incl. review)': 'skipped', 'update-results': "
    "'success', 'settle (post-results)': 'success'}\n")


def _v(log):
    f = ci.extract(log)
    return ci.verdict(f, ci.assertions(f, []), log), f


def test_the_real_2026_09_20_log_is_no_longer_CLEAN():
    """THE POSITIVE CONTROL, on the actual bytes the run produced."""
    v, f = _v(PAD + REAL_0920)
    assert v != "CLEAN", (
        "the run that produced no picks, no odds and no fixtures scored "
        "CLEAN — this is the verdict ci_audit gave run 35498465743")
    assert v == "DID_NOT_RUN"
    assert f["steps_not_run"], "the skipped steps were not extracted"


def test_the_skipped_steps_are_NAMED_not_counted():
    """Which steps did not run decides what was lost; a count does not."""
    _, f = _v(PAD + REAL_0920)
    named = f["steps_not_run"][-1]
    assert "picks (incl. review)" in named
    assert "update" in named
    assert "settle (post-results)" not in named, (
        "a step that SUCCEEDED was listed as not run")


def test_the_new_workflow_line_is_read_too():
    """The shape the workflow emits after 2026-09-20."""
    v, f = _v(PAD + "⚠️ Daily picks: step(s) DID NOT RUN — picks (incl. "
                    "review), update. Logs: http://example/run\n")
    assert v == "DID_NOT_RUN"
    assert "picks (incl. review)" in f["steps_not_run"][-1]


def test_it_surfaces_as_an_assertion_so_it_reaches_the_ledger_note():
    _, f = _v(PAD + REAL_0920)
    hits = ci.assertions(f, [])
    assert any("DID NOT RUN" in h for h in hits), hits


def test_a_FAILURE_still_outranks_a_skip():
    """A run that both failed and skipped is BROKEN first.

    Ordering matters: the skip is a consequence of the failure, and the
    failure is the finding.
    """
    v, _ = _v(PAD + REAL_0920
              + "⚠️ Daily picks: step(s) FAILED — update. Logs: x\n")
    assert v == "BROKEN"


def test_an_empty_log_still_outranks_a_skip():
    """No evidence is not a skip verdict either — UNAUDITABLE stays first."""
    v, _ = _v(REAL_0920)          # short log, below the auditable floor
    assert v == "UNAUDITABLE"


def test_an_ALL_SUCCESS_run_is_untouched():
    """The regression that would matter: every clean run turning DID_NOT_RUN."""
    v, f = _v(PAD + "All critical steps OK: {'update': 'success', "
                    "'picks (incl. review)': 'success'}\n")
    assert not f["steps_not_run"]
    assert v == "CLEAN"


def test_DID_NOT_RUN_is_not_provisional():
    """It is a settled verdict. A run that did not run will not re-run.

    `UNAUDITABLE` and `IN_PROGRESS` re-list because more evidence may arrive;
    this one is complete, so re-listing it every day would be noise.
    """
    assert not ci._is_provisional("| DID_NOT_RUN | core step(s) DID NOT RUN |")
    assert ci._is_provisional("| UNAUDITABLE — empty log | |")
