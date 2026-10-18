"""The calendar/live marker, enforced rather than declared — 2026-10-18.

    A remedy that must be remembered when writing new code is not a remedy.

THE TWO INSTANCES, both measured:

    2026-10-09  two calendar assertions failed on a date rolling over, stopped
                `Run tests`, stopped `--update`, and LOST ONE CARD — 26
                fixtures, 2416 odds rows — recovered by a manual dispatch with
                3h01m to spare, by someone watching. The remedy was the
                `calendar` MARKER: declared membership.

    2026-10-13  a new test asserted a hard-coded date was absent from a
                ROLLING 14-day window over a live `gh run list`. The date aged
                out, `Run tests` failed, and SIX CONSECUTIVE CARDS were lost —
                10-13..10-18, 137 fixtures, ZERO picks. The marker existed and
                was simply not applied.

THE CENSUS, `measured` 2026-10-18 and reported BEFORE the check was wired:

    predicate                                   flagged   in the gate
    "touches date.today()"  (the naive grep)       56         —
    first AST pass, with `get` in LIVE_NAMES       88         —
    FINAL predicate                                 1          1

The first two numbers are not backlogs, they are predicates that had not been
thought through. `d.get("k")` is a dict read; a date literal handed to a pure
function is a FIXTURE. The final predicate flags two shapes and nothing else:

    (1) a CLOCK-DERIVED value compared against anything FIXED
    (2) a DATE or a COUNT compared against LIVE STATE

and its one finding is exempted with a reason, because both sides of that
comparison move together.
"""

import pathlib
import subprocess
import sys

import pytest

import scripts.calendar_marker_scan as scan
import tests.conftest as conftest

#: The 2026-10-13 test, verbatim as it was written. The known-wrong case.
THE_TEST_THAT_COST_SIX_CARDS = '''import datetime as dt
import pytest
import scripts.ci_audit as ci


def test_the_detector_runs_against_the_LIVE_repo_and_2026_10_12_is_NOT_missing():
    m = ci.missing_scheduled_runs()
    if m.get("state") != "CHECKED":
        pytest.skip(f"detector unevaluated here: {m.get('reason')}")
    assert dt.date(2026, 10, 12) not in m["missing"], (
        "2026-10-12 is reported missing, but its scheduled run exists")
'''


def _plant(body: str, name: str):
    """A probe inside `tests/`, where the conftest applies, removed after."""
    import os
    p = pathlib.Path("tests") / f"test_zz_cal_{name}_{os.getpid()}.py"
    p.write_text(body, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(p),
             "-p", "no:cacheprovider"],
            capture_output=True, encoding="utf-8", errors="replace",
            cwd=str(pathlib.Path.cwd()), timeout=300)
        return r, r.stdout + r.stderr
    finally:
        p.unlink(missing_ok=True)


# ── THE KNOWN-WRONG CONTROL ──────────────────────────────────────────────────

def test_KNOWN_WRONG_the_test_that_cost_six_cards_is_REFUSED():
    """A detector that cannot catch the instance that motivated it has no
    measured coverage. This is that instance, verbatim."""
    r, out = _plant(THE_TEST_THAT_COST_SIX_CARDS, "kw")
    assert r.returncode != 0, f"it was ACCEPTED:{chr(10)}{out[-1200:]}"
    assert "DATE LITERAL compared against LIVE STATE" in out, out[-1500:]
    assert "rolling window" in out
    assert "@pytest.mark.calendar" in out, "the marker is not named"
    assert "@pytest.mark.live" in out, "the other marker is not named"
    assert "no tests ran" in out, (
        "the probe body RAN - the refusal is not at collection")


def test_the_scan_finds_ALL_THREE_tests_the_MARKER_EXISTS_FOR():
    """Coverage on the other known instances, in the live tree.

    These are the self-expiring guards. A predicate that missed them would be
    measuring nothing — and an earlier draft did, because it only accepted AST
    literals as the 'fixed' side and these compare against a module constant.
    """
    found = {f.name for f in scan.scan() if f.markers}
    for name in ("test_the_H1_RESERVATION_WINDOW_has_EXPIRED",
                 "test_the_H1_RATE_DECISION_is_DUE",
                 "test_the_suppression_has_EXPIRED"):
        assert name in found, (
            f"{name} is no longer flagged; the predicate has lost the "
            f"instances the marker exists for")


# ── THE POSITIVE CONTROL: MARKED PASSES, UNMARKED FAILS ──────────────────────

_UNMARKED = '''import datetime as dt


def test_it():
    today = dt.date.today()
    assert today < dt.date(2027, 1, 1)
'''

_MARKED = '''import datetime as dt
import pytest


@pytest.mark.calendar
def test_it():
    today = dt.date.today()
    assert today < dt.date(2027, 1, 1)
'''


def test_POSITIVE_CONTROL_unmarked_is_refused_and_MARKED_is_accepted():
    """Both halves. Without the second, the check could be refusing
    everything."""
    r1, out1 = _plant(_UNMARKED, "unmarked")
    assert r1.returncode != 0, out1[-900:]
    assert "CLOCK-DERIVED value compared against" in out1, out1[-900:]

    r2, out2 = _plant(_MARKED, "marked")
    assert r2.returncode == 0, out2[-1200:]
    assert "1 passed" in out2 or "1 deselected" in out2, out2[-600:]


def test_BOTH_SIDES_MOVING_TOGETHER_is_NOT_refused():
    """The discrimination that keeps the backlog at one.

    `assert written == date.today()` passes every day: the expected value
    tracks the actual one. The 2026-10-13 defect was a FROZEN literal against
    a moving window — only one side moved.
    """
    body = ('import datetime as dt' + chr(10) + chr(10) + chr(10) +
            'def test_it():' + chr(10) +
            '    a = dt.date.today()' + chr(10) +
            '    b = dt.date.today()' + chr(10) +
            '    assert a == b' + chr(10))
    r, out = _plant(body, "together")
    assert r.returncode == 0, out[-1200:]


# ── THE BACKLOG, REPORTED AND PINNED ────────────────────────────────────────

def test_the_BACKLOG_is_one_and_it_is_EXEMPTED_with_a_reason():
    """Reported before the check was wired, and pinned so it cannot grow.

    Its single member is in the GATE, which is why it is exempted rather than
    marked: `tests/conftest.py`'s own rule is that a correctness test which
    happens to use a date stays in the gating suite.
    """
    bad = [f for f in scan.unmarked(scan.scan())
           if f.nodeid not in conftest.CALENDAR_SCAN_ALLOWED]
    assert bad == [], (
        f"the backlog grew: {[f.nodeid for f in bad]}. Mark them, fix them, "
        f"or add them to CALENDAR_SCAN_ALLOWED with a reason.")
    assert len(conftest.CALENDAR_SCAN_ALLOWED) == 1, (
        f"the exemption list moved to {len(conftest.CALENDAR_SCAN_ALLOWED)}; "
        f"each entry must carry why both sides move together")
    for nodeid, reason in conftest.CALENDAR_SCAN_ALLOWED.items():
        assert "::" in nodeid
        assert len(reason) > 80, f"{nodeid} has no real reason"
        assert "MOVE TOGETHER" in reason.upper(), (
            f"{nodeid}'s exemption does not explain why it is safe")


def test_the_gate_EXCLUDES_both_markers_and_the_other_step_RUNS_them():
    """The separation the markers exist to express, asserted on the workflow.
    """
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    assert '-m "not calendar and not live"' in wf, (
        "the gate no longer excludes both markers")
    assert '-m "calendar or live"' in wf, (
        "the non-gating step no longer runs them, so a marked test would "
        "never run at all")


def test_the_predicate_is_PURELY_STATIC_and_that_is_why_it_may_gate():
    """The distinction that makes a collection-time refusal safe here.

    A refusal that could fire on a date roll would reproduce the disease. This
    one reads the test's own AST and nothing else, so it fires only on the
    commit that introduces an unmarked time-varying test — locally, in the
    same pytest the author is already running.
    """
    # BY AST, NOT BY TEXT. A text scan trips on this module's own PROSE: its
    # docstring names `date.today()` to explain the predicate, and stripping
    # `#` comments does not remove a docstring. Same shape as the `::error::`
    # comment-blind scan two stages ago.
    import ast

    tree = ast.parse(pathlib.Path("scripts/calendar_marker_scan.py")
                     .read_text(encoding="utf-8"))
    banned = {"today", "now", "utcnow", "urlopen", "run", "check_output",
              "get", "post"}
    called = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        mod = (f.value.id if isinstance(f, ast.Attribute)
               and isinstance(f.value, ast.Name) else "")
        if name in banned and mod in ("", "dt", "date", "datetime", "time",
                                      "requests", "subprocess", "urllib"):
            called.add(f"{mod}.{name}" if mod else name)
    assert not called, (
        f"the scan itself calls {sorted(called)} - its verdict could then "
        f"change with no code change, which is the defect it exists to "
        f"refuse")
