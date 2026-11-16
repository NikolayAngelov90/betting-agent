"""Which kind of selected test failed. 2026-11-16. Standard library only.

`reminders.yml` runs `-m "calendar or live"` and used to read every failure
as "REMINDER DUE ... a VERDICT". On 2026-11-16 a LIVE check failed on live
input and was printed as a dated guard reaching its date -- neither true.
pytest is the only place that knows a failing test's markers, so it says
which kind failed, one line per call-phase failure (the same failures the
workflow counts as `FAILED`). Inert unless the workflow sets the variable.

  calendar  a dated guard reached its date      -> REMINDER DUE
  live      a live test failed on live input    -> LIVE CHECK FAILED
  other     neither marker                      -> FAILURE

No test carries both markers (measured 2026-11-16: `-m "calendar and live"`
collects 0); one that did would count as `calendar`, the dated guard.

A MODULE OF ITS OWN, NOT A FUNCTION IN `conftest.py`: conftest imports
`src.data.odds_quota`, so a test importing the hook from there becomes
capture-critical by the import graph and moves its whole file into the gate
-- 44 nodes, measured when it happened. `conftest.py` re-exports the hook,
which is how pytest finds it.
"""
import os

REMINDER_KINDS_ENV = "REMINDER_KINDS_FILE"


def reminder_kind(keywords) -> str:
    if "calendar" in keywords:
        return "calendar"
    if "live" in keywords:
        return "live"
    return "other"


def pytest_runtest_logreport(report):
    path = os.environ.get(REMINDER_KINDS_ENV)
    if not path or report.when != "call" or not report.failed:
        return
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(f"{reminder_kind(report.keywords)} {report.nodeid}\n")
