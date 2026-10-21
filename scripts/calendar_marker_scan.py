"""Which tests' verdicts change with no code change? An AST pass, not a habit.

    A remedy that must be remembered when writing new code is not a remedy.

THE TWO INSTANCES THIS EXISTS FOR, both measured:

    2026-10-09  two calendar assertions failed on a date rolling over, stopped
                `Run tests`, stopped `--update`, and LOST ONE CARD - 26
                fixtures, 2416 odds rows - recovered by a manual dispatch with
                3h01m to spare, by someone watching. The remedy was the
                `calendar` MARKER: declared membership.

    2026-10-13  a new test asserted that a hard-coded date was absent from a
                ROLLING 14-day window over a live `gh run list`. The date aged
                out, `Run tests` failed, and SIX CONSECUTIVE CARDS were lost -
                10-13..10-18, 137 fixtures, ZERO picks. The marker existed and
                was not applied.

Declared membership failed for the reason it always does: it asks the author to
classify the thing at the moment they are not thinking about the
classification. So the classification is now computed.

THE PREDICATE, and the discrimination is the whole of the work. A naive scan
over `date.today()` in tests/ finds 56 candidates and misses the three that
fired - the `calendar` marker's own note records that. A first version of this
pass, with `get` in its live-call set, returned 88. Neither number is a
backlog; both are a predicate that has not been thought through.

    (1) a CLOCK-DERIVED value in a comparison        -> time-varying by
                                                        construction
    (2) a DATE or a COUNT compared against LIVE      -> asserts over a rolling
        STATE                                           window; expires when
                                                        the window moves

and nothing else. A date literal handed to a pure function is a FIXTURE, not a
clock dependency, and flagging those was most of the 88.

    python -m scripts.calendar_marker_scan            # the census
    python -m scripts.calendar_marker_scan --list     # with reasons
"""

from __future__ import annotations

import argparse
import ast
import os
import pathlib
import sys
from typing import List, NamedTuple, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Attribute or function names that read the clock.
CLOCK_NAMES = {"today", "now", "utcnow", "monotonic"}

#: Functions that reach LIVE EXTERNAL STATE. Project-specific on purpose:
#: `get`, `run` and `time` were in this set for exactly one census and
#: produced most of its 88 findings, because `d.get("k")` is a dict read.
LIVE_NAMES = {"missing_scheduled_runs", "unparseable_workflow_runs",
              "collect_lag_series_by_daytype", "workflow_states", "fetch_log",
              "urlopen"}

#: NOT live, and each was measured into this comment. `probe_fixtures` and
#: `verify_offset` are driven with monkeypatched scrapers and a local SQLite
#: session in every test that calls them, and `subprocess.run` of a local
#: command is deterministic - including `subprocess.run([pytest, ...])`. All
#: six findings they produced in the 2026-10-18 census were false positives.

#: Matched on the QUALIFIED form, so `subprocess.run` counts and `x.run` does
#: not.
#:
#: ── `evaluate` IS QUALIFIED, AND THAT IS NOT FASTIDIOUSNESS. 2026-10-21. ──
#:
#: `finding_register.evaluate()` runs each finding's `live` predicate, and
#: finding 13's is `not Path("data/models/feature_list.json").exists()` -- a
#: read of a path git does not track, which `actions/cache` restores in CI and
#: a dev box does not have. Its result therefore depends on something outside
#: the commit, which is this scanner's whole subject; "live" here means that,
#: not "networked". `test_the_COUNTS_and_the_MEDIAN...` asserted a COUNT
#: against it and cost a card on 2026-10-21.
#:
#: BUT PUTTING THE BARE NAME IN `LIVE_NAMES` REFUSED
#: `test_h1_precondition::test_NO_FIXTURE_IN_BAND_is_SEPARATE_from_NO_CARD`,
#: which calls `pre.evaluate(_Sess(68, 68, 9, 0), ...)` -- a DIFFERENT
#: `evaluate`, driven by a fake session, entirely hermetic. Measured: one
#: false positive from one unqualified name. Hence the qualified form, exactly
#: as `requests.get` is qualified so that `x.run` does not match
#: `subprocess.run`.
LIVE_QUALIFIED = {("requests", "get"), ("requests", "post"),
                  ("fr", "evaluate"), ("finding_register", "evaluate")}

#: ── THE THIRD KIND, FOLDED IN FROM A SECOND GUARD. 2026-10-21. ────────────
#:
#: `tests/test_no_live_state_in_the_capture_gate.py` existed for one day and
#: is deleted: two guards for one defect class is the shape rule 1 forbids,
#: and the class is "a test whose outcome depends on anything not in the
#: commit". Its rule is kept here as a third reason.
#:
#: WHY IT IS NOT REDUNDANT WITH `LIVE_NAMES`. That set is consulted only at an
#: ASSERTION, so a test that CALLS a live reader without comparing its result
#: still passes -- yet its outcome can still move, because the call itself can
#: raise, return empty, or drive a `pytest.skip`. These three have NO
#: injection seam, so there is no safe way to call one from a gating test.
SEAMLESS_READERS = {
    "collect_lag_series_by_daytype":
        "queries the GitHub API for run timings and takes no fetcher; it "
        "returns an EMPTY series where there are no credentials, which is "
        "exactly how 2026-10-21 lost its card",
    "fetch_log": "downloads a run log from the GitHub API; no seam",
    "missing_scheduled_runs": "enumerates live runs from the GitHub API",
}

#: Stubbing the TRANSPORT makes such a call hermetic, and that is a legitimate
#: way to exercise a seamless reader from inside the gate. `ci._sh` is the
#: shell runner all three go through. `current_crons_by_daytype` is here
#: because stubbing it SHORT-CIRCUITS `missing_scheduled_runs` before the
#: network: that function reads the crons first and returns UNEVALUATED on
#: None or an all-None dict, both BEFORE its `_sh(...)` call -- verified
#: against the source, not assumed.
#:
#: Measured 2026-10-21: without this, three hermetic tests in
#: test_absent_run_and_unparseable.py were refused, and flagging
#: `reference_earliest(fetch=...)` and `current_crons_by_daytype(sha=...)` as
#: calls produced 20 more. A guard that cries wolf is deleted, and the real
#: one goes with it.
TRANSPORT_STUBS = {"_sh", "_sh_json", "fetch_log", "_reference_fetch",
                   "current_crons_by_daytype"}

#: The markers that declare a test out of the capture-gating suite.
MARKERS = {"calendar", "live"}

REFUSAL = (
    "This test's verdict can change with NO CODE CHANGE, so it must not be "
    "able to stop price capture.{reasons}\n\n"
    "MARK IT, with whichever fits:\n\n"
    "    @pytest.mark.calendar   its outcome depends on today's DATE - it "
    "asserts a deployment fact\n"
    "    @pytest.mark.live       it queries LIVE EXTERNAL STATE (the GitHub "
    "API, the network, a provider)\n\n"
    "Either marker moves it out of `Run tests` and into the non-gating step, "
    "where a failure still reddens the run without skipping `--update`, "
    "`--settle` and the picks.\n\n"
    "WHY THIS IS REFUSED RATHER THAN DOCUMENTED: a date rolling over cost one "
    "card on 2026-10-09, and a rolling-window assertion cost SIX on "
    "2026-10-13 - 137 fixtures, zero picks - with the marker already in the "
    "tree and simply not applied."
)


class Finding(NamedTuple):
    path: str
    lineno: int
    name: str
    reasons: Tuple[str, ...]
    markers: Tuple[str, ...]

    @property
    def nodeid(self) -> str:
        return f"{self.path}::{self.name}"


def _call_name(node) -> str:
    if not isinstance(node, ast.Call):
        return ""
    f = node.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        if (f.value.id, f.attr) in LIVE_QUALIFIED:
            return f"{f.value.id}.{f.attr}"
    return f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")


def _is_clock_read(node) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) in CLOCK_NAMES


def _stubs_transport(fn) -> bool:
    """True when this test replaces the transport, or the reader itself."""
    for call in ast.walk(fn):
        if not (isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "setattr"):
            continue
        for a in call.args:
            if isinstance(a, ast.Constant) and isinstance(a.value, str):
                if a.value in TRANSPORT_STUBS or a.value in SEAMLESS_READERS:
                    return True
    return False


def _is_live_call(node) -> bool:
    n = _call_name(node)
    return bool(n) and (n in LIVE_NAMES
                        or n in {f"{a}.{b}" for a, b in LIVE_QUALIFIED})


def _is_date_literal(node) -> bool:
    """`dt.date(2026, 10, 12)` or a `'YYYY-MM-DD'` string."""
    if isinstance(node, ast.Call):
        if _call_name(node) in ("date", "datetime") and len(node.args) >= 3 \
                and all(isinstance(a, ast.Constant)
                        and isinstance(a.value, int) for a in node.args[:3]):
            return node.args[0].value >= 1900
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        v = node.value
        return (len(v) >= 10 and v[4:5] == "-" and v[7:8] == "-"
                and v[:4].isdigit() and v[5:7].isdigit())
    return False


def _is_count(node) -> bool:
    return (isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool))


def _markers(fn) -> Tuple[str, ...]:
    out = set()
    for d in fn.decorator_list:
        t = d.func if isinstance(d, ast.Call) else d
        if isinstance(t, ast.Attribute):
            out.add(t.attr)
    return tuple(sorted(out & MARKERS))


def scan_function(fn) -> Tuple[str, ...]:
    """The reasons this test's verdict can move on its own, or ()."""
    clock: Set[str] = set()
    live: Set[str] = set()
    assigns = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        # ── TUPLE UNPACKING TAINTS EVERY NAME IT BINDS. 2026-10-21. ────────
        #
        # This required `len(targets) == 1 and isinstance(targets[0], Name)`,
        # so `by_type, _crons = collect_lag_series_by_daytype()` bound NOTHING
        # and the live call's result was clean from there on. That is exactly
        # how `test_the_weekday_cron_CLEARS...` passed this scanner while
        # calling a function already listed in LIVE_NAMES, and then lost a
        # card on 2026-10-21. Measured: `scan_function` returned `()` for it.
        #
        # Every Name bound by the statement is tainted FOR THE LIVE SET,
        # however it is spelled. Over-tainting is the safe direction there:
        # the cost is a marker, and the cost of under-tainting is a day's
        # prices.
        #
        # THE CLOCK SET USES THE ORIGINAL BINDING AND ONLY THAT. Walking the
        # target for Names taints the RECEIVER of an attribute assignment --
        # `agent.predictor.ml_models.trained_at = utcnow()` marked `agent`
        # clock-derived, so `agent._ml_models_stale(...) is True` read as a
        # clock value against a fixed literal. Measured: one false positive,
        # `test_ml_stale_when_goals_model_missing`, which is hermetic because
        # `now` is always now and the staleness comes from the OTHER model
        # being None. A predicate calibrated to 1 finding on 2026-10-18 is not
        # re-opened as a side effect of fixing a different one.
        targets = []
        for t in node.targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    targets.append(sub.id)
        if not targets:
            continue
        simple = (node.targets[0].id
                  if len(node.targets) == 1
                  and isinstance(node.targets[0], ast.Name) else None)
        assigns.append((targets, node.value, simple))

    # ── TAINT IS TRANSITIVE, TO A FIXED POINT. 2026-10-21. ─────────────────
    #
    # One pass marked only names assigned DIRECTLY from a live call, so
    #
    #     by_type, _crons = collect_lag_series_by_daytype()   # tainted
    #     lag = [l for _d, l in (by_type.get("weekday") or [])]   # CLEAN
    #     assert len(lag) >= 8
    #
    # reached the assertion with a clean name, and the 2026-10-21 card went
    # with it. Derivation has to carry the taint, and `ast.walk` does not
    # visit statements in source order, so this iterates rather than assuming
    # the assignment it needs has already been seen.
    # TRANSITIVITY APPLIES TO `live` ONLY, AND THAT IS MEASURED. Propagating
    # the CLOCK set as well took the census from 1 unmarked finding to 14, and
    # 11 of the 13 new ones were false: `trained_at = utcnow(); assert not
    # stale` is hermetic forever, because the test controls the clock-derived
    # input and `now` is always now. The clock rule was calibrated to 1 on
    # 2026-10-18 and is left exactly as it was.
    #
    # All THREE known-wrong cases are the LIVE-STATE rule, not the clock rule:
    # the 10-13 date literal against `missing_scheduled_runs()`, and both of
    # 2026-10-21's counts against live-derived values. So the gap was here and
    # only here.
    for _ in range(len(assigns) + 1):
        before = len(live)
        for targets, value, _simple in assigns:
            nodes = list(ast.walk(value))
            if any(_is_clock_read(x) for x in nodes):
                continue
            names = {x.id for x in nodes if isinstance(x, ast.Name)}
            if any(_is_live_call(x) for x in nodes) or (names & live):
                live.update(targets)
        if len(live) == before:
            break
    # THE CLOCK SET KEEPS ITS 2026-10-18 BINDING RULE EXACTLY: a single Name
    # target, no tuple unpacking, no transitivity. Widening it to every bound
    # Name added four findings and all four were false. A predicate calibrated
    # to 1 is not re-opened as a side effect of fixing a different one.
    for _targets, value, simple in assigns:
        if simple is not None and any(
                _is_clock_read(x) for x in ast.walk(value)):
            clock.add(simple)

    def clockish(node) -> bool:
        return any(_is_clock_read(x)
                   or (isinstance(x, ast.Name) and x.id in clock)
                   for x in ast.walk(node))

    def livish(node) -> bool:
        return any(_is_live_call(x)
                   or (isinstance(x, ast.Name) and x.id in live)
                   for x in ast.walk(node))

    reasons: List[str] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        for cmp_ in [n for n in ast.walk(node.test)
                     if isinstance(n, ast.Compare)]:
            sides = [cmp_.left] + list(cmp_.comparators)
            # BOTH SIDES MOVING TOGETHER IS SAFE. `assert written ==
            # date.today().isoformat()` passes every day: the expected value
            # tracks the actual one. The 2026-10-13 defect was that only ONE
            # side moved - a frozen literal against a rolling window. So a
            # clock-derived value is flagged only against a FIXED literal.
            clocky = [x for x in sides if clockish(x)]
            # FIXED = anything that is neither clock-derived nor live. A
            # module constant like `H1_RESERVATION_UNTIL` is a frozen date in
            # effect, and restricting this to AST literals lost all three
            # tests the `calendar` marker exists for - measured, not reasoned.
            frozen = [x for x in sides
                      if not clockish(x) and not livish(x)]
            if clocky and frozen:
                reasons.append(
                    "a CLOCK-DERIVED value compared against a FIXED literal - "
                    "only one side moves, so it expires")
                continue
            lits = [x for x in sides if _is_date_literal(x) or _is_count(x)]
            lv = [x for x in sides if livish(x)]
            if lits and lv:
                kind = ("a DATE LITERAL"
                        if any(_is_date_literal(x) for x in lits)
                        else "a COUNT")
                reasons.append(
                    f"{kind} compared against LIVE STATE - it asserts over a "
                    f"rolling window and expires when the window moves")

    # THE THIRD KIND: a CALL to a seamless reader, asserted on or not.
    if not _stubs_transport(fn):
        for node in ast.walk(fn):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = (f.attr if isinstance(f, ast.Attribute)
                    else getattr(f, "id", ""))
            if name in SEAMLESS_READERS:
                reasons.append(
                    f"it CALLS {name}(), which has no injection seam - "
                    f"{SEAMLESS_READERS[name]}")

    return tuple(sorted(set(reasons)))


def scan(paths=None) -> List[Finding]:
    paths = paths or sorted(pathlib.Path("tests").glob("*.py"))
    out: List[Finding] = []
    for p in paths:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not node.name.startswith("test_"):
                continue
            reasons = scan_function(node)
            if reasons:
                out.append(Finding(p.as_posix(), node.lineno, node.name,
                                   reasons, _markers(node)))
    return out


def unmarked(findings: List[Finding]) -> List[Finding]:
    return [f for f in findings if not f.markers]


def refusal_text(findings: List[Finding]) -> str:
    rs = ""
    for f in findings:
        rs += f"\n\n  {f.path}:{f.lineno}  {f.name}"
        for r in f.reasons:
            rs += f"\n    - {r}"
    return REFUSAL.format(reasons=rs)


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="calendar_marker_scan",
        description="Tests whose verdict changes with no code change.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    found = scan()
    bad = unmarked(found)
    ok = [f for f in found if f.markers]
    print("CALENDAR/LIVE MARKER SCAN")
    print()
    print(f"  flagged, UNMARKED (would be refused) : {len(bad)}")
    print(f"  flagged, already marked              : {len(ok)}")
    print()
    for f in ok:
        print(f"  OK   {f.path}:{f.lineno} {f.name} "
              f"[{','.join(f.markers)}]")
    if bad:
        # THE SAME TEXT THE HOOK RAISES. One definition of the message, so the
        # CLI cannot drift from the refusal a developer actually sees - and it
        # gives `refusal_text` a production consumer, which the PNC ratchet
        # asked for rather than an exemption.
        print(refusal_text(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
