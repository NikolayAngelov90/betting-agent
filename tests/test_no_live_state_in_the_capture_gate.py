"""No test in the CAPTURE-GATING selection may read live external state.

    2026-10-21. Written after the THIRD occurrence of one shape.

THE SHAPE. `daily-picks.yml` gates price capture on
`pytest tests/ -m "not calendar and not live"`. A test in that selection whose
verdict depends on something other than the code can fail with no code change,
and when it does, every downstream step is skipped and the day's card is lost.
Prices do not backfill.

    2026-10-09  two date-dependent assertions        -> card lost
    2026-10-13..18  a live `gh run list` assertion   -> SIX cards lost
    2026-10-21  a GitHub-API read + a working-tree read -> card lost, recovered

The remedy each time was the same marker. The markers exist, they are
documented in `conftest.pytest_configure`, and they keep not being applied --
because nothing checked. This file checks.

IT IS A STRUCTURAL GUARD, NOT A STYLE RULE. It fails on the thing that
actually costs cards: an unmarked test that calls a known live-state reader.
The registry below is explicit, so adding a new live-state helper means adding
it here, which is the point -- a reader nobody registered is the gap this
closes.
"""

import ast
import pathlib

import pytest

#: Callables with NO INJECTION SEAM: every call reaches the GitHub API, so
#: there is no safe way to use one from a gating test. Each entry is
#: `name -> why`, and the `why` is what a failure prints, because "forbidden"
#: without a reason is how a guard gets deleted by the next person.
#:
#: DELIBERATELY NARROW. `reference_earliest(day, fetch=...)` and
#: `current_crons_by_daytype(sha=...)` are NOT here: both have a documented
#: seam, and the tests that call them pass a stub or assert the degradation
#: path on purpose. Flagging them produced 20 false positives on the first
#: draft of this file. A guard that cries wolf is removed, and then the real
#: one goes with it. `audited_run_ids` is not here either: it reads the
#: committed ledger, which is repo state, not live state.
LIVE_STATE_READERS = {
    "collect_lag_series_by_daytype":
        "queries the GitHub API for run timings and takes no fetcher; it "
        "returns an EMPTY series where there are no credentials, which is "
        "exactly how 2026-10-21 lost its card",
    "fetch_log":
        "downloads a run log from the GitHub API; no seam",
    "missing_scheduled_runs":
        "enumerates live runs from the GitHub API; no seam",
}

#: Stubbing the TRANSPORT makes a call hermetic, and that is a legitimate way
#: to test a no-seam reader from inside the gate. `ci._sh` is the only shell
#: runner these three go through, so replacing it replaces the network.
#: Measured 2026-10-21: two tests in test_absent_run_and_unparseable.py call
#: `missing_scheduled_runs(today=...)` with `_sh` stubbed and are hermetic;
#: flagging them would have moved real coverage out of the gate for nothing.
#: `current_crons_by_daytype` is here because stubbing it SHORT-CIRCUITS
#: `missing_scheduled_runs` before the network: that function reads the
#: crons first and returns UNEVALUATED on None or an all-None dict, both
#: BEFORE its `_sh(...)` run-list call. Verified against the source, not
#: assumed -- the early returns precede the shell call.
TRANSPORT_STUBS = {"_sh", "fetch_log", "_reference_fetch", "_sh_json",
                   "current_crons_by_daytype"}


def _stubs_the_transport(node) -> bool:
    """True when this test replaces the transport or the reader itself."""
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        fn = call.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "setattr"):
            continue
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if arg.value in TRANSPORT_STUBS or arg.value in LIVE_STATE_READERS:
                    return True
    return False


#: Modules that are themselves live-state gateways: importing one inside a
#: gating test is a strong signal even when the call is indirect.
LIVE_STATE_MODULES = {"urllib.request", "requests", "httpx"}

GATE_SELECTION = '-m "not calendar and not live"'


def _gating_tests():
    """Every test function NOT excluded by the gate's marker selection.

    Markers are read from the AST, so a decorator is honoured without
    importing the module -- importing every test module here would run their
    collection-time side effects.
    """
    out = []
    for path in sorted(pathlib.Path("tests").glob("test_*.py")):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        # a module-level `pytestmark` excludes the whole file
        module_marks = set()
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", "") == "pytestmark" for t in node.targets):
                module_marks |= set(_mark_names(node.value))
        if module_marks & {"live", "calendar"}:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            marks = set()
            for dec in node.decorator_list:
                marks |= set(_mark_names(dec))
            if marks & {"live", "calendar"}:
                continue
            out.append((path, node, src))
    return out


def _mark_names(node):
    """Marker names reachable from a decorator or a `pytestmark` value."""
    names = []
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Attribute):
            names.append(n.attr)
            stack.append(n.value)
        elif isinstance(n, ast.Call):
            stack.append(n.func)
        elif isinstance(n, (ast.List, ast.Tuple)):
            stack.extend(n.elts)
        elif isinstance(n, ast.Name):
            names.append(n.id)
    return names


def test_the_gate_selection_is_still_what_this_file_guards():
    """If the workflow's selection changes, this guard is measuring nothing."""
    wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")
    assert GATE_SELECTION in wf, (
        f"the capture gate no longer runs {GATE_SELECTION!r}; re-point this "
        f"file at whatever selection now gates --update, or it guards nothing")


def test_no_live_state_reader_is_called_from_a_gating_test():
    """THE GUARD. A card costs more than a marker."""
    offenders = []
    for path, node, src in _gating_tests():
        if _stubs_the_transport(node):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if name in LIVE_STATE_READERS:
                offenders.append(
                    f"{path.as_posix()}::{node.name} calls {name}() — "
                    f"{LIVE_STATE_READERS[name]}")
    assert not offenders, (
        "live external state is read by "
        f"{len(offenders)} test(s) inside the capture-gating selection. Mark "
        "each `@pytest.mark.live`; its verdict can change with no code change, "
        "and when it does the day's card is lost and prices do not "
        "backfill:\n  " + "\n  ".join(offenders))


def test_no_gating_test_imports_a_network_client():
    offenders = []
    for path, node, src in _gating_tests():
        for n in ast.walk(node):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                mods = [n.module]
            for m in mods:
                if m in LIVE_STATE_MODULES:
                    offenders.append(f"{path.as_posix()}::{node.name} "
                                     f"imports {m}")
    assert not offenders, (
        "a network client is imported inside a capture-gating test:\n  "
        + "\n  ".join(offenders))


def test_DISCRIMINATION_the_guard_catches_a_planted_offender(tmp_path,
                                                             monkeypatch):
    """A guard that has never caught anything is not known to work.

    Plant an unmarked gating test that calls a registered reader, point the
    collector at it, and assert the guard fires. Then mark it `live` and assert
    the guard goes quiet — so the marker is proven to be the thing that clears
    it, not the file's absence.
    """
    offending = (
        "import pytest\n"
        "from scripts.ci_audit import collect_lag_series_by_daytype\n"
        "def test_planted():\n"
        "    collect_lag_series_by_daytype()\n")
    marked = "import pytest\n" + offending.replace(
        "def test_planted():", "@pytest.mark.live\ndef test_planted():")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    planted = tests_dir / "test_planted.py"

    def _run():
        monkeypatch.chdir(tmp_path)
        try:
            return [o for o in _collect_offenders()]
        finally:
            monkeypatch.undo()

    planted.write_text(offending, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    hits = _collect_offenders()
    monkeypatch.undo()
    assert len(hits) == 1, f"the guard missed a planted offender: {hits}"
    assert "collect_lag_series_by_daytype" in hits[0]

    planted.write_text(marked, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    hits_marked = _collect_offenders()
    monkeypatch.undo()
    assert hits_marked == [], (
        f"`@pytest.mark.live` did not clear the guard: {hits_marked} — then "
        f"the marker is not what the guard reads, and the remedy it prints is "
        f"not the remedy that works")


def _collect_offenders():
    """The guard's body, reusable so the control exercises the real thing."""
    found = []
    for path, node, src in _gating_tests():
        if _stubs_the_transport(node):
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            fn = call.func
            name = (fn.attr if isinstance(fn, ast.Attribute)
                    else fn.id if isinstance(fn, ast.Name) else None)
            if name in LIVE_STATE_READERS:
                found.append(f"{path.as_posix()}::{node.name} calls {name}()")
    return found
