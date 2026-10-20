"""API-Football pacing: the gap is a DEADLINE, not an unconditional sleep.

    2026-10-20. Part B of "Workflow Inventory, Then Close the 19.8 Minutes".

THE CHANGE. `_api_get` used to `await asyncio.sleep(6)` after every response.
That paces correctly and also charges the caller for work it does AFTER the
response, so the injury loop -- which writes to Neon for ~7 s per fixture --
really spaced its requests 6 + 7 s apart and the 6 was dead time.

THE GUARANTEE IT MUST NOT WEAKEN: no request starts less than
`_AF_MIN_REQUEST_GAP_S` after the previous one's RESPONSE. The old code
delivered that plus the caller's work; the new code delivers exactly that.
Every test below asserts the guarantee, and `test_3` is the discrimination
control: it fails if the implementation silently reverts to the old semantics,
and it fails if the new one ever paces TIGHTER than the guarantee.

The clock is virtual. A test that really slept would take 48 s and would
measure the host's scheduler instead of this module's arithmetic.
"""
import asyncio

import pytest

import src.scrapers.apifootball_scraper as af


#: Captured BEFORE any monkeypatch. The fake sleep still has to yield to the
#: loop, and `await asyncio.sleep(0)` inside the fake would be the fake again.
_REAL_SLEEP = asyncio.sleep


class _Clock:
    """A monotonic clock that only `sleep` advances."""

    def __init__(self):
        self.t = 1000.0
        self.slept = []

    def monotonic(self):
        return self.t

    async def sleep(self, s):
        assert s >= 0, f"negative sleep {s}"
        self.slept.append(s)
        self.t += s
        await _REAL_SLEEP(0)     # yield, so this stays a real coroutine


def _scraper(clock, monkeypatch, response=None, work_s=0.0, http_s=0.4):
    """A scraper built via `__new__` -- no config, no DB, no network.

    `work_s` is the caller's post-response work, charged to the clock right
    after `_api_get` returns, which is where a real caller's DB write sits.
    """
    s = af.APIFootballScraper.__new__(af.APIFootballScraper)
    # Empty, not a placeholder string: a plausible-looking placeholder is
    # exactly what test_no_secrets_in_repo flags, and it was right to.
    s.api_key = ""
    s._quota_exhausted = False
    s._requests_today = 0
    s._daily_limit = 1000
    s._rate_limit_retries = 0
    s._plan_restricted = False
    s._account_suspended = False
    s._next_request_at = 0.0
    starts = []

    async def _fetch_json(url, params=None, headers=None):
        starts.append(clock.t)
        clock.t += http_s                      # the request itself takes time
        return {"response": []} if response is None else response

    s.fetch_json = _fetch_json
    monkeypatch.setattr(af, "_time", clock)
    monkeypatch.setattr(asyncio, "sleep", clock.sleep)
    return s, starts


async def _drive(s, clock, n, work_s):
    for _ in range(n):
        await s._api_get("/fixtures", {"date": "2026-10-21"})
        clock.t += work_s                      # the caller's own work


def _spacings(starts, http_s=0.4):
    """Gap between one response ENDING and the next request STARTING."""
    return [starts[i + 1] - (starts[i] + http_s) for i in range(len(starts) - 1)]


GAP = af._AF_MIN_REQUEST_GAP_S


def test_1_no_work_to_overlap_paces_exactly_the_gap(monkeypatch):
    """The xG and odds loops batch their writes, so they have nothing to
    overlap. They are the OLD code's tightest case and must be unchanged."""
    clock = _Clock()
    s, starts = _scraper(clock, monkeypatch)
    asyncio.run(_drive(s, clock, 5, work_s=0.0))
    sp = _spacings(starts)
    assert len(sp) == 4
    assert all(abs(x - GAP) < 1e-6 for x in sp), sp
    assert all(x >= GAP - 1e-9 for x in sp), f"paced TIGHTER than {GAP}: {sp}"


def test_2_work_longer_than_the_gap_absorbs_it_entirely(monkeypatch):
    """The injury loop: ~7 s of DB writes per fixture, which already exceeds
    the gap, so the gap costs nothing and nothing is slept."""
    clock = _Clock()
    s, starts = _scraper(clock, monkeypatch)
    asyncio.run(_drive(s, clock, 5, work_s=7.0))
    sp = _spacings(starts)
    assert all(x >= GAP - 1e-9 for x in sp), f"guarantee broken: {sp}"
    assert all(abs(x - 7.0) < 1e-6 for x in sp), sp
    assert clock.slept == [], f"slept despite 7 s of work: {clock.slept}"


def test_3_DISCRIMINATION_the_old_semantics_would_fail_this(monkeypatch):
    """THE CONTROL. With 12 s of work per call the two semantics differ by
    1.5x, so this test tells them apart instead of passing under either.

    Both expectations are DERIVED from the parameters, not written down -- the
    first draft hard-coded 48 and 72 and forgot that the request itself takes
    time, which is the same class of mistake as pacing from the wrong instant.

      new : start-to-start = http + work          (the gap is already covered)
      old : start-to-start = http + GAP + work    (the gap sat on top)
    """
    HTTP, WORK, N = 0.4, 12.0, 5
    clock = _Clock()
    s, starts = _scraper(clock, monkeypatch, work_s=WORK, http_s=HTTP)
    t0 = clock.t
    asyncio.run(_drive(s, clock, N, work_s=WORK))
    span = starts[-1] - t0
    expect_new = (N - 1) * (HTTP + WORK)
    expect_old = (N - 1) * (HTTP + GAP + WORK)
    assert abs(span - expect_new) < 1e-6, (
        f"span {span:.1f}s: {expect_new:.1f} is the deadline semantics, "
        f"{expect_old:.1f} is the old unconditional sleep")
    assert span < expect_old - 1e-6, "this is the old unconditional-sleep behaviour"
    assert all(x >= GAP - 1e-9 for x in _spacings(starts, HTTP))
    assert clock.slept == [], f"slept despite {WORK} s of work: {clock.slept}"


def test_4_an_ERROR_response_also_arms_the_gap(monkeypatch):
    """The error branches used to `return None` with no sleep at all, so an
    errored call was followed IMMEDIATELY by the next one. That is the one
    place this change makes pacing stricter rather than looser."""
    clock = _Clock()
    s, starts = _scraper(
        clock, monkeypatch, response={"errors": {"plan": "not your season"}})
    asyncio.run(_drive(s, clock, 4, work_s=0.0))
    sp = _spacings(starts)
    assert len(sp) == 3
    assert all(x >= GAP - 1e-9 for x in sp), f"an error path went unpaced: {sp}"


def test_5_the_unconditional_sleep_is_gone_from_the_source():
    """Not a prose check: the literal call, found by AST, in `_api_get`."""
    import ast
    import pathlib
    tree = ast.parse(pathlib.Path(af.__file__).read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.AsyncFunctionDef) and n.name == "_api_get")
    bare = []
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sleep"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, (int, float))):
            bare.append(node.args[0].value)
    assert bare == [], (
        f"_api_get sleeps a CONSTANT {bare} -- the gap must come from "
        f"`_next_request_at`, so that the caller's work counts toward it")
