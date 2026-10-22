"""The parallel-regime picks series, driven on its REAL path — 2026-10-22.

WHY THIS FILE EXISTS. `parallel_regime_picks_series()` shipped in `c2eef6d`
calling a function that did not exist:

    started = as_datetime(r.get("createdAt"))
    NameError: name 'as_datetime' is not defined

Every `ci-audit` run on that commit died with it — three runs, and the audit's
own crash banner was correct: NOTHING was audited. The 2,312-test gating suite
passed, because the only test that reached `main()` had stubbed `_sh` to return
`""`, so the function took its UNEVALUATED early return and the loop below it
was never executed by anything.

    A test that drives only the error branch certifies the error branch.

So: the happy path, with runs that MUST be counted and runs that must NOT be,
each for a different reason, and the timestamp helper on its own.
"""

import datetime as dt
import json

import pytest

import scripts.ci_audit as ci


@pytest.fixture(autouse=True)
def _no_cache():
    """The series memoises per `limit`. Without this, the first test's fake
    runs answer every later test."""
    ci._PICKS_SERIES_CACHE.clear()
    yield
    ci._PICKS_SERIES_CACHE.clear()


# ── the helper that did not exist ──────────────────────────────────────────

@pytest.mark.parametrize("raw,expect", [
    ("2026-10-22T03:00:13Z", dt.datetime(2026, 10, 22, 3, 0, 13,
                                         tzinfo=dt.timezone.utc)),
    ("2026-10-22T03:00:13+00:00", dt.datetime(2026, 10, 22, 3, 0, 13,
                                              tzinfo=dt.timezone.utc)),
])
def test_the_timestamp_helper_PARSES_what_github_sends(raw, expect):
    assert ci._as_datetime(raw) == expect


@pytest.mark.parametrize("raw", [None, "", "not-a-time", "2026-13-99T00:00:00Z"])
def test_the_timestamp_helper_returns_NONE_rather_than_RAISING(raw):
    """`completedAt` is ABSENT on a step that never finished. A raise here
    takes the whole audit down, which is exactly what happened."""
    assert ci._as_datetime(raw) is None


# ── the series, on runs that count ─────────────────────────────────────────

def _fake_sh(runs, step_times, descendants=("cccb0e6", "deadbee")):
    """A `_sh` that answers the three commands the series issues."""
    def _sh(*args):
        if args[:3] == ("gh", "run", "list"):
            return json.dumps(runs)
        if args[:2] == ("git", "rev-list"):
            return chr(10).join(descendants)
        if args[:3] == ("gh", "run", "view"):
            run_id = args[3]
            done = step_times.get(run_id)
            if done is None:
                return json.dumps({"jobs": []})
            return json.dumps({"jobs": [{"steps": [
                {"name": ci.PICKS_STEP_NAME, "completedAt": done}]}]})
        return ""
    return _sh


def test_a_SCHEDULED_WEEKDAY_run_ON_THE_NEW_COMMIT_is_counted(monkeypatch):
    """The path the NameError lived on. 2026-10-22 is a Thursday."""
    runs = [{"databaseId": 1, "createdAt": "2026-10-22T03:00:00Z",
             "event": "schedule", "headSha": "deadbeef1234"}]
    monkeypatch.setattr(ci, "_sh", _fake_sh(
        runs, {"1": "2026-10-22T03:35:06Z"}))
    s = ci.parallel_regime_picks_series()
    assert s["n"] == 1, s
    assert s["members"] == [1]
    assert s["max"] == pytest.approx(35.1, abs=0.05)
    assert s["mean"] == pytest.approx(35.1, abs=0.05)
    assert s["sd"] == 0.0, "one point has no spread"
    assert s["mean_plus_3sd"] == pytest.approx(35.1, abs=0.05)
    assert s.get("state") != "UNEVALUATED"


def test_SEVERAL_runs_give_a_MEAN_and_a_SPREAD(monkeypatch):
    runs = [{"databaseId": i, "createdAt": f"2026-10-2{d}T03:00:00Z",
             "event": "schedule", "headSha": "deadbeef"}
            for i, d in ((1, 0), (2, 1), (3, 2))]      # Tue, Wed, Thu
    monkeypatch.setattr(ci, "_sh", _fake_sh(runs, {
        "1": "2026-10-20T03:30:00Z",    # 30
        "2": "2026-10-21T03:40:00Z",    # 40
        "3": "2026-10-22T03:50:00Z",    # 50
    }))
    s = ci.parallel_regime_picks_series()
    assert s["n"] == 3
    assert s["mean"] == pytest.approx(40.0)
    assert s["max"] == pytest.approx(50.0)
    assert s["sd"] > 0, "three distinct values must have a spread"
    assert s["mean_plus_3sd"] > s["mean"]


# ── the series, on runs that must NOT count, each for its own reason ───────

@pytest.mark.parametrize("run,step,why", [
    ({"databaseId": 9, "createdAt": "2026-10-22T03:00:00Z",
      "event": "workflow_dispatch", "headSha": "deadbeef"},
     "2026-10-22T03:35:00Z",
     "a DISPATCHED run skips the scheduler lateness the bound exists to cover"),
    ({"databaseId": 9, "createdAt": "2026-10-24T03:00:00Z",
      "event": "schedule", "headSha": "deadbeef"},
     "2026-10-24T03:35:00Z",
     "2026-10-24 is a Saturday and the bound is a WEEKDAY population"),
    ({"databaseId": 9, "createdAt": "2026-10-22T03:00:00Z",
      "event": "schedule", "headSha": "0ldc0mmit999"},
     "2026-10-22T03:35:00Z",
     "the commit does not contain the parallel review, so it is the OLD "
     "regime and regimes are not pooled"),
    ({"databaseId": 9, "createdAt": "2026-10-22T03:00:00Z",
      "event": "schedule", "headSha": ""},
     "2026-10-22T03:35:00Z",
     "an empty headSha cannot be placed in either regime"),
    ({"databaseId": 9, "createdAt": "2026-10-22T03:00:00Z",
      "event": "schedule", "headSha": "deadbeef"},
     None,
     "the picks step never completed, so the run has no time-to-picks"),
])
def test_the_runs_that_must_NOT_be_counted(monkeypatch, run, step, why):
    monkeypatch.setattr(ci, "_sh", _fake_sh(
        [run], {} if step is None else {"9": step}))
    s = ci.parallel_regime_picks_series()
    assert s["n"] == 0, f"counted a run it must not: {why}"
    assert s["members"] == []
    assert s["mean_plus_3sd"] is None, (
        "an empty series must report None, never 0.0 -- a zero bound would "
        "read as a perfect margin")


def test_an_EMPTY_series_is_n0_and_an_UNREADABLE_gh_is_UNEVALUATED(monkeypatch):
    """The distinction that cost the most to get right: `""` would otherwise
    read as n=0, which is the state this series is legitimately in."""
    monkeypatch.setattr(ci, "_sh", _fake_sh([], {}))
    empty = ci.parallel_regime_picks_series()
    assert empty["n"] == 0 and empty.get("state") != "UNEVALUATED"

    ci._PICKS_SERIES_CACHE.clear()
    monkeypatch.setattr(ci, "_sh", lambda *a: "")
    unread = ci.parallel_regime_picks_series()
    assert unread["n"] is None
    assert unread["state"] == "UNEVALUATED"
    assert "UNKNOWN" in unread["reason"]


# ── KNOWN-WRONG CONTROL ────────────────────────────────────────────────────

def test_KNOWN_WRONG_the_formatter_would_have_CRASHED_on_a_real_series():
    """The regression, stated as the shape rather than the symbol.

    `format_picks_bound_status` is handed the dict the series returns. Before
    the fix the series could not PRODUCE a populated dict at all -- it raised
    first -- so the formatter's n>0 branch had never run against real output.
    Drive it.
    """
    out = ci.format_picks_bound_status({
        "n": 3, "mean": 40.0, "sd": 8.16, "mean_plus_3sd": 64.5, "max": 50.0,
        "regime": ci.PICKS_BOUND_NEW_POPULATION, "needed": 10,
        "to_ten_weeks": 1.4, "to_ten_date": "2026-11-01", "members": [1, 2, 3],
    })
    assert "n=3" in out and "mean=40.0" in out
    assert "mean+3sd=64.5" in out
    assert str(ci.FULL_CARD_PICKS_MAX) in out, (
        "the block must keep naming the constant the margin actually uses")
    assert "!!" not in out, "a status block is not a finding"


def test_KNOWN_WRONG_a_name_that_does_not_exist_is_caught_at_IMPORT():
    """Why the NameError survived the suite: nothing called the function.

    `compile` with the module's own source would not have caught it either --
    a NameError is a RUNTIME error. The defence is the test above this one,
    not a static check, and saying so here stops the next person reaching for
    a linter and believing it covered them.
    """
    import pathlib
    src = pathlib.Path("scripts/ci_audit.py").read_text(encoding="utf-8")
    compile(src, "scripts/ci_audit.py", "exec")      # passes BEFORE the fix
    assert "def _as_datetime(" in src, (
        "the helper the two call sites need is gone again")
    assert "as_datetime(r.get" not in src.replace("_as_datetime(r.get", ""), (
        "a call to the UNPREFIXED name is back")
