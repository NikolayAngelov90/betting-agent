"""Two consumers reading one input on one commit are a free control.

WHY THIS FILE EXISTS. 2026-11-06.

`reminders` read the weekday lateness series as **n=0** while `ci-audit` read
**n=11** — same commit, same minute, same API — and it did so for **sixteen
days**, from the workflow's creation on 2026-10-21.

    NEITHER FIGURE LOOKED WRONG ALONE.

`n=0` is a legitimate value for a series, and `n=11` is a legitimate value for
a series. Only the two together said anything. The cause was mundane: the
series' membership rule reads the cron from each run's own `headSha` via
`git show <sha>:...`, and `reminders` checked out at the default
`fetch-depth: 1`, which makes every sha but HEAD unreadable — so the rule
silently OVER-EXCLUDED to nothing.

> **Wherever two components consume the same input, the cross-read is a
> control that costs nothing. This file makes it standing.**

AND IT IMMEDIATELY CAUGHT ONE OF MINE. Measuring which selections reach a
per-run-history reader found a live call I had just put in
`test_contamination_and_margin.py` — a `capture_critical` file with no
marker, i.e. **inside the capture gate**, where `daily-picks` checks out
shallow, the series reads n=0, and the gate FAILS. A live-reading test in the
capture gate loses the day card; there had been three occurrences, and that
would have been the fourth. It was split out before it ever ran.
"""
import ast
import pathlib

import pytest
import yaml

from scripts.capture_critical import classify

ROOT = pathlib.Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
WORKFLOWS = ROOT / ".github" / "workflows"

#: Functions that read PER-RUN history — `gh run list` plus a per-run
#: `git show <sha>:...` to decide which cron regime the run belongs to. A
#: caller of one of these needs the full history to answer correctly.
PER_RUN_READERS = {"collect_lag_series_by_daytype",
                   "parallel_regime_picks_series"}

#: The three selections, and which workflow runs each.
SELECTION_WORKFLOW = {
    "GATE": "daily-picks.yml",
    "reminders": "reminders.yml",
    "non-gating": "non-gating-tests.yml",
}

#: THE CROSS-READ THAT WAS MISSING, recorded with both figures. `measured`
#: 2026-11-06 on commit 23f0510, both runs within the same minute.
CROSS_READ_2026_11_06 = {
    "input": "the weekday lateness series under cron `0 1`",
    "ci-audit": {"n": 11, "fetch_depth": 0},
    "reminders": {"n": 0, "fetch_depth": "1 (default)"},
    "undetected_for_days": 16,
    "since": "2026-10-21",
    "cause": ("the membership rule reads each run's own headSha via "
              "`git show`, which a shallow checkout cannot resolve"),
    "hidden_by": ("a pytest.skip reporting `UNEVALUATED: the weekday lag "
                  "series is empty`, which is a pass in a green run"),
}


def _unstubbed_callers():
    """`{selection: [(file, test)]}` for tests that call a reader FOR REAL.

    A test that `monkeypatch.setattr(ci, "_sh", ...)` never touches live
    history, so it is not exposed to the checkout depth and is not counted.
    Counting it would make every selection look exposed and the check would
    stop discriminating.
    """
    crit = classify()
    out = {k: [] for k in SELECTION_WORKFLOW}
    for p in sorted(TESTS.glob("test_*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for fn in (n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)):
            body = ast.unparse(fn)
            calls = {(x.func.attr if isinstance(x.func, ast.Attribute)
                      else getattr(x.func, "id", ""))
                     for x in ast.walk(fn) if isinstance(x, ast.Call)}
            if not (calls & PER_RUN_READERS):
                continue
            if "setattr" in body and "_sh" in body:
                continue                      # stubbed: not a live reader
            marks = {d.attr for d in fn.decorator_list
                     if isinstance(d, ast.Attribute)}
            if marks & {"live", "calendar"}:
                sel = "reminders"
            elif crit.get(p.name, {}).get("critical"):
                sel = "GATE"
            else:
                sel = "non-gating"
            out[sel].append((p.name, fn.name))
    return out


def _depth(workflow):
    d = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))
    depths = []
    for job in (d.get("jobs") or {}).values():
        for st in (job.get("steps") or []):
            if "checkout" in str(st.get("uses", "")):
                depths.append((st.get("with") or {}).get("fetch-depth", 1))
    assert depths, f"{workflow} has no checkout step, so no depth to read"
    return depths


# ── THE PREDICATE CAN FIND SOMETHING, OR IT VERIFIES NOTHING ──────────────

def test_the_PREDICATE_FINDS_THE_LIVE_READERS_IT_IS_LOOKING_FOR():
    found = _unstubbed_callers()
    total = sum(len(v) for v in found.values())
    assert total >= 3, (
        f"the scan found {total} unstubbed per-run-history caller(s); it "
        f"should find several, so a zero in any one selection below is "
        f"UNEVALUATED rather than clean. {found}")


# ── THE GATE MUST HAVE NONE. THIS ONE IS A CARD. ──────────────────────────

def test_NO_LIVE_PER_RUN_READER_IS_IN_THE_CAPTURE_GATE():
    """`daily-picks` checks out shallow, so a live reader there reads n=0 and
    FAILS the gate -- which stops `--update` and loses the day's card."""
    offenders = _unstubbed_callers()["GATE"]
    assert offenders == [], (
        f"these gate tests read LIVE per-run history: {offenders}. "
        f"`daily-picks` checks out at fetch-depth "
        f"{_depth('daily-picks.yml')}, so the series reads n=0 there and the "
        f"gate fails. Mark them `live`, or stub `_sh`.")


# ── ANY SELECTION WITH A LIVE READER NEEDS THE FULL HISTORY ───────────────

def test_EVERY_SELECTION_WITH_A_LIVE_READER_CHECKS_OUT_FULL_HISTORY():
    found = _unstubbed_callers()
    for sel, callers in found.items():
        if not callers:
            continue
        wf = SELECTION_WORKFLOW[sel]
        depths = _depth(wf)
        assert all(str(d) == "0" for d in depths), (
            f"{wf} runs {len(callers)} live per-run-history reader(s) "
            f"{callers[:2]}... at fetch-depth {depths}. A shallow checkout "
            f"makes every sha but HEAD unreadable, so the series reads n=0 "
            f"and that is reported as a MEASUREMENT. This is the defect that "
            f"went undetected for "
            f"{CROSS_READ_2026_11_06['undetected_for_days']} days.")


def test_THE_TWO_CONSUMERS_NOW_AGREE_ON_THEIR_DEPTH():
    """The cross-read itself: both readers of this input run full history."""
    for wf in ("ci-audit.yml", "reminders.yml"):
        assert all(str(d) == "0" for d in _depth(wf)), (
            f"{wf} no longer checks out full history; it is one of the two "
            f"consumers of the lateness series and they must not diverge "
            f"again")


def test_THE_CROSS_READ_IS_RECORDED_WITH_BOTH_FIGURES():
    """A finding of this shape is unreadable with only one number."""
    c = CROSS_READ_2026_11_06
    assert c["ci-audit"]["n"] == 11 and c["reminders"]["n"] == 0
    assert c["ci-audit"]["n"] != c["reminders"]["n"], (
        "the two figures agree, so the record no longer shows a disagreement")
    assert c["undetected_for_days"] == 16
    for field in ("input", "cause", "hidden_by", "since"):
        assert c.get(field), f"the cross-read record has no {field}"
    assert "skip" in c["hidden_by"], (
        "the record no longer says what hid it, which is the transferable "
        "part")


# ── KNOWN-WRONG CONTROLS ──────────────────────────────────────────────────

def test_KNOWN_WRONG_a_SHALLOW_WORKFLOW_WITH_A_LIVE_READER_IS_REFUSED(tmp_path):
    """The control: the depth check must reject depth 1, or it is vacuous."""
    shallow = tmp_path / "shallow.yml"
    shallow.write_text(
        "name: X\non: {workflow_dispatch: {}}\njobs:\n  j:\n    "
        "runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n",
        encoding="utf-8")
    d = yaml.safe_load(shallow.read_text(encoding="utf-8"))
    depths = [(st.get("with") or {}).get("fetch-depth", 1)
              for job in d["jobs"].values() for st in job["steps"]
              if "checkout" in str(st.get("uses", ""))]
    # THE ANCHOR: the fixture must actually have produced a checkout step,
    # or "it is shallow" is a claim about nothing.
    assert len(depths) >= 1, (
        f"the synthetic workflow produced no checkout step, so this control "
        f"verifies nothing: {depths}")
    assert depths == [1], depths
    assert not all(str(x) == "0" for x in depths), (
        "the depth predicate accepts a default checkout, so it cannot "
        "detect the condition it exists for")


def test_KNOWN_WRONG_a_STUBBED_CALLER_IS_NOT_COUNTED():
    """The other direction: counting stubbed callers would flag everything.

    `test_picks_bound_series.py` calls a reader in four tests and stubs `_sh`
    in every one, so it is NOT exposed to the checkout depth. If the
    predicate counted it, `non-gating` would look exposed and the real
    signal would be lost in it.
    """
    # STATED AS AN EQUALITY, NOT AS A `not in`. The raw-match ratchet flags
    # any function that reads a `.py` file AND contains a negative membership
    # test, because a comment can satisfy one; an equality on an AST-derived
    # list cannot be satisfied that way, and the clearing action the ratchet
    # asks for is exactly this conversion.
    found = _unstubbed_callers()
    assert found["non-gating"] == [], (
        f"a stubbed caller is being counted as a live reader: "
        f"{found['non-gating']}")

    # and the subject really does call a reader AND stub it -- both POSITIVE,
    # so this control cannot pass by absence
    src = (TESTS / "test_picks_bound_series.py").read_text(encoding="utf-8")
    assert any(r in src for r in PER_RUN_READERS), (
        "test_picks_bound_series no longer calls a reader at all, so this "
        "control verifies nothing")
    assert 'monkeypatch.setattr(ci, "_sh"' in src, (
        "it no longer stubs `_sh`, so it may now be a live reader")
