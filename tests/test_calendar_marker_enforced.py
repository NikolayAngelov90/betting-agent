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


def _plant(body: str, name: str, extra=()):
    """A probe inside `tests/`, where the conftest applies, removed after.

    `extra` passes further pytest arguments, so the positive control can run
    the probe under the GATE'S OWN marker selection rather than a paraphrase
    of it.
    """
    import os
    p = pathlib.Path("tests") / f"test_zz_cal_{name}_{os.getpid()}.py"
    p.write_text(body, encoding="utf-8")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(p),
             "-p", "no:cacheprovider", *extra],
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
    # TWO, not three, from 2026-10-18: `test_the_H1_RATE_DECISION_is_DUE` was
    # RETIRED that day because the decision it was holding was taken (H1
    # collected n=4 of 39 against a floor of 33, so no analysis runs). Its
    # successor contract lives in tests/test_calendar_separation.py. The
    # predicate's coverage is unchanged; the population shrank.
    # RENAMED 2026-10-28 WITH THE WINDOW MOVE. The guard now fires on the
    # NEW expiry (2026-11-09) instead of the old one, so the name says
    # something true again. The marker POPULATION is unchanged: it is the
    # same test. Both this assertion and the reminders registry's copy
    # FAILED on the rename before being updated, which is how a
    # name-keyed guard is shown not to be vacuous.
    found = {f.name for f in scan.scan() if f.markers}
    for name in ("test_the_H1_RESERVATION_WINDOW_EXPIRES",
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

#: THE ONLY GROUNDS ON WHICH AN EXEMPTION MAY REST. An exemption says "the
#: scan is right that this LOOKS time-varying and wrong that it IS", so it has
#: to name which way it is wrong. "It is fine" is not a ground.
#:
#: The second ground was added 2026-10-21 with the two `finding_register`
#: entries: `evaluate()` is live only through finding 13, and neither test's
#: assertions reach that row. The dependence exists in the callee and does not
#: reach the verdict — which is a different fact from both sides moving
#: together, and collapsing them into one phrase would have meant accepting a
#: reason that does not explain anything.
EXEMPTION_GROUNDS = (
    "MOVE TOGETHER",            # the expected value tracks the actual one
    "DOES NOT REACH",           # the live/clock input never reaches the assert
    "WAS REMOVED, NOT DECLARED",  # the dependence was engineered out
    "ASSERTS ABOUT FINDING",    # only a committed-source row is asserted
)


def test_the_BACKLOG_is_EMPTY_and_every_EXEMPTION_names_its_GROUND():
    """Reported before the check was wired, and pinned so it cannot grow.

    Every member is in the GATE, which is why each is exempted rather than
    marked: `tests/conftest.py`'s own rule is that a correctness test which
    happens to touch a date or a registry stays in the gating suite.

    THE COUNT MOVED 1 -> 3 ON 2026-10-21, deliberately, when
    `finding_register.evaluate` was added to the live set after one of its
    callers cost a card. Both new entries are hermetic in fact; marking them
    would have moved real coverage out of the gate to silence a scan that was
    over-reaching, which is the wrong direction.
    """
    bad = [f for f in scan.unmarked(scan.scan())
           if f.nodeid not in conftest.CALENDAR_SCAN_ALLOWED]
    assert bad == [], (
        f"the backlog grew: {[f.nodeid for f in bad]}. Mark them, fix them, "
        f"or add them to CALENDAR_SCAN_ALLOWED with a reason.")
    assert len(conftest.CALENDAR_SCAN_ALLOWED) == 3, (
        f"the exemption list moved to {len(conftest.CALENDAR_SCAN_ALLOWED)}; "
        f"each entry must name one of {EXEMPTION_GROUNDS}")
    for nodeid, reason in conftest.CALENDAR_SCAN_ALLOWED.items():
        assert "::" in nodeid
        assert len(reason) > 80, f"{nodeid} has no real reason"
        assert any(g in reason.upper() for g in EXEMPTION_GROUNDS), (
            f"{nodeid}'s exemption does not name a ground from "
            f"{EXEMPTION_GROUNDS}; 'it is fine' is not one")


def test_the_gate_EXCLUDES_both_markers_and_the_other_step_RUNS_them():
    """The separation the markers exist to express, asserted on the workflow.
    """
    # ── THE TWO SELECTIONS ARE NOW IN TWO WORKFLOWS. 2026-10-21. ──────────
    # They were two steps of `daily-picks`, which stopped a reminder gating
    # capture but not reddening the run. The partition is unchanged; only its
    # address is.
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    rem = pathlib.Path(".github/workflows/reminders.yml").read_text(
        encoding="utf-8")
    assert '-m "not calendar and not live"' in wf, (
        "the gate no longer excludes both markers")
    assert '-m "calendar or live"' in rem, (
        "the reminder workflow no longer runs them, so a marked test would "
        "never run at all")
    assert '-m "calendar or live"' not in wf, (
        "the reminder selection is back inside the capture workflow, whose "
        "conclusion then carries two facts again")


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


# ───────────────────── THE 2026-10-21 PAIR, AND THE COLLAPSE ─────────────────
#
# A SECOND GUARD EXISTED FOR ONE DAY AND IS GONE.
# `tests/test_no_live_state_in_the_capture_gate.py` was written on 2026-10-21
# for the live-reader kind; its rule now lives in this scanner as a third
# reason, because two guards for one defect class is the shape that let the
# 10-21 pair through — each guard covered part of the class and both looked
# covered. The class is: A GATING TEST IS A PURE FUNCTION OF THE REPOSITORY AT
# THAT COMMIT.
#
# AND THE SEQUENCE IS THE FINDING, which is the second confirmation of the
# sentence at the top of this file. The remedy for this exact shape shipped on
# 2026-10-18 in `7514a50`. The warning sits FOUR LINES above the gate's own
# pytest command in `daily-picks.yml`. On 2026-10-20 I added two more, and on
# 2026-10-21 they cost a card. A remedy that must be remembered is not a
# remedy; it has to be a check.

#: KNOWN WRONG, verbatim as written on 2026-10-20. Tuple unpacking defeated the
#: taint tracking: `by_type, _crons = collect_lag_series_by_daytype()` bound
#: nothing, so the live result was clean at the assertion.
THE_TUPLE_UNPACK_TEST = '''import statistics
import scripts.ci_audit as ci


def test_planted_weekday_margin():
    by_type, _crons = ci.collect_lag_series_by_daytype()
    lag = [l for _d, l in (by_type.get("weekday") or [])]
    assert len(lag) >= 8, f"the weekday series shrank to n={len(lag)}"
'''

#: KNOWN WRONG, verbatim as written on 2026-10-20. The dependence is a
#: WORKING-TREE read inside a registry predicate: finding 13 is
#: `not Path("data/models/feature_list.json").exists()`, which actions/cache
#: restores in CI and a dev box does not have.
THE_WORKING_TREE_TEST = '''import datetime as dt
import scripts.finding_register as fr


def test_planted_counts():
    rows = fr.evaluate(today=dt.date(2026, 10, 20))
    live = [r for r in rows if r["is_live"] is True]
    assert len(live) == 8, f"{len(live)} findings still live, expected 8"
'''


def test_KNOWN_WRONG_the_TUPLE_UNPACK_caller_is_REFUSED():
    """Control 2 of 3. Named: the `collect_lag_series_by_daytype()` caller."""
    r, out = _plant(THE_TUPLE_UNPACK_TEST, "tup")
    assert r.returncode != 0, f"it was ACCEPTED:{chr(10)}{out[-1200:]}"
    assert ("COUNT compared against LIVE STATE" in out
            or "no injection seam" in out), out[-1500:]
    assert "no tests ran" in out, (
        "the probe body RAN - the refusal is not at collection")


def test_KNOWN_WRONG_the_WORKING_TREE_read_is_REFUSED():
    """Control 3 of 3. Named: the `feature_list.json` predicate."""
    r, out = _plant(THE_WORKING_TREE_TEST, "wt")
    assert r.returncode != 0, f"it was ACCEPTED:{chr(10)}{out[-1200:]}"
    assert "COUNT compared against LIVE STATE" in out, out[-1500:]
    assert "no tests ran" in out


def test_POSITIVE_CONTROL_the_same_test_MARKED_is_accepted():
    """The marker must be what clears the refusal.

    Without this the three controls above prove only that SOMETHING fails, and
    a guard that refuses everything would pass them all.
    """
    marked = THE_TUPLE_UNPACK_TEST.replace(
        "import scripts.ci_audit as ci",
        "import pytest\nimport scripts.ci_audit as ci").replace(
        "def test_planted_weekday_margin():",
        "@pytest.mark.live\ndef test_planted_weekday_margin():")
    assert "@pytest.mark.live" in marked
    r, out = _plant(marked, "pos")
    assert "This test's verdict can change" not in out, (
        f"the marker did NOT clear the refusal, so the marker is not what the "
        f"guard reads:{chr(10)}{out[-1200:]}")
    # it is deselected by the gate's own selection, which is the point
    r2, out2 = _plant(marked, "pos2", extra=["-m", "not calendar and not live"])
    assert "1 deselected" in out2 or "deselected" in out2, out2[-800:]


#: WHAT THIS PREDICATE CANNOT DETECT. Rule 2: the uncoverable list IS the
#: guard's coverage statement, and it is more useful than the hit count.
#:
#: Each line is a way a gating test's verdict could still move without the
#: scan seeing it. They are not hypothetical shapes — every one of them would
#: have passed the scan on 2026-10-20.
UNCOVERABLE = (
    "INDIRECTION THROUGH A HELPER the scan does not know: a test calling a "
    "local function that itself reads the clock, the network or the working "
    "tree. Only the names in LIVE_NAMES, LIVE_QUALIFIED and SEAMLESS_READERS "
    "are known, and a reader nobody registered is invisible.",
    "DEPENDENCE VIA A FIXTURE: a conftest fixture that reads live state hands "
    "it to the test as an argument, and the test body is clean.",
    "A MODULE-LEVEL CONSTANT computed at import from live state - the scan "
    "reads function bodies.",
    "A PURE-CLOCK TEST with no comparison against a fixed side: `assert "
    "f(date.today())` is flagged only when something frozen is on the other "
    "side, and the legitimate form (both sides moving) is deliberately not.",
    "A FILESYSTEM READ ON A COMPUTED PATH: only a literal argument can be "
    "checked against the index, so `open(base / name)` is invisible.",
    "UNSEEDED RANDOMNESS, test ORDER dependence, and shared mutable state - "
    "none of which is time-varying in this sense but all of which can move a "
    "verdict with no code change.",
    "THE DATABASE: conftest strips DATABASE_URL and the SQLite fallback is "
    "refused, so this is covered by a different mechanism, not by this scan.",
)


def test_the_UNCOVERABLE_kinds_are_ENUMERATED_not_implied():
    """Rule 2. A guard narrower than its defect class leaves the rest
    unguarded and looking covered, so the gap is written down."""
    assert len(UNCOVERABLE) >= 7
    for kind in UNCOVERABLE:
        assert len(kind) > 60, f"too vague to be a coverage statement: {kind}"
    joined = " ".join(UNCOVERABLE).upper()
    for must in ("FIXTURE", "INDIRECTION", "COMPUTED PATH", "DATABASE"):
        assert must in joined, f"{must} is not in the uncoverable list"


def test_the_SECOND_GUARD_IS_GONE_and_its_rule_is_HERE():
    """Rule 1: one predicate per defect class. Three guards would mean the
    class was never named correctly."""
    assert not pathlib.Path(
        "tests/test_no_live_state_in_the_capture_gate.py").exists(), (
        "the second guard is back; fold its rule into the scan instead")
    assert scan.SEAMLESS_READERS, "its rule did not survive the collapse"
    assert "collect_lag_series_by_daytype" in scan.SEAMLESS_READERS
    assert scan.TRANSPORT_STUBS, "the stub carve-out did not survive"
