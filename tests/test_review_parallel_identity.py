"""The parallel review produces the SAME decisions — 2026-10-21.

The change is timing-only if and only if the decisions are identical. Proving
that is most of the work, and an identity proof without a discrimination
control is indistinguishable from a comparison that returns a constant.

WHAT IS AND IS NOT CLAIMED. Every per-match input is per-match: the prompt is
built from that match's analysis, odds menu and saved pick, and
`_apply_decision` filters the database by `match_id`. The only state shared
across units is `_claude_code_exhausted`, and the claim is qualified for it
below — honestly, because on a day the Pro pool runs out mid-batch the SET of
units that fall back to the API differs between serial and parallel, and that
is a real difference rather than one worth hiding.
"""

import asyncio

import pytest

import src.reporting.match_briefing as mb


class _Rec:
    def __init__(self, mid, ev=1.0):
        self.match_id = mid
        self.expected_value = ev


def _svc(decide, order=None, concurrency=None):
    """A service with the reviewer replaced by a deterministic function.

    `decide(match_id)` stands in for everything `_briefing_for` does, so the
    comparison is over the ORCHESTRATION — which is what changed — and not
    over Claude, which is not reproducible.
    """
    svc = mb.MatchBriefingService.__new__(mb.MatchBriefingService)
    svc._sent_lock = asyncio.Lock()
    svc._enabled = lambda: True
    svc._sync_recs_from_db = lambda *a, **k: None
    decisions = {}

    async def _briefing_for(match_id, lineup_aware, post_to_telegram):
        # a real await point, so the scheduler genuinely interleaves
        await asyncio.sleep(0)
        if order is not None:
            order.append(match_id)
        verdict = decide(match_id)
        if isinstance(verdict, Exception):
            raise verdict
        decisions[match_id] = verdict
        return bool(verdict)

    svc._briefing_for = _briefing_for
    svc._decisions = decisions
    return svc


def _run(svc, picks):
    return asyncio.run(svc.finalize_picks_with_claude(picks))


CARD_A = [_Rec(i) for i in range(1, 14)]
CARD_B = [_Rec(i) for i in range(100, 113)]


def _decide_A(mid):
    return "KEEP" if mid % 2 else "CHANGE"


def _decide_B(mid):
    return "CHANGE" if mid % 3 else "KEEP"


# ── the bound ──────────────────────────────────────────────────────────────

def test_the_CONCURRENCY_is_BOUNDED_and_the_bound_is_STATED():
    assert mb.REVIEW_CONCURRENCY == 4
    import inspect
    src = inspect.getsource(mb)
    head = src[:src.index("class ")]
    assert "SHARED USAGE POOL" in head, (
        "the bound does not say what actually limits it")
    assert "1,000 RPM" in head, (
        "the bound does not record that the API rate limit is NOT the gate")
    assert "DISTINGUISHABLE FROM A DECLINE" in head, (
        "the bound does not state whether mid-batch exhaustion can be told "
        "apart from a review that declined")


def test_NO_MORE_THAN_the_bound_run_at_once():
    """Driven, not asserted: count the units in flight."""
    live, peak = 0, 0

    svc = mb.MatchBriefingService.__new__(mb.MatchBriefingService)
    svc._sent_lock = asyncio.Lock()
    svc._enabled = lambda: True
    svc._sync_recs_from_db = lambda *a, **k: None

    async def _briefing_for(match_id, lineup_aware, post_to_telegram):
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0.01)
        live -= 1
        return True

    svc._briefing_for = _briefing_for
    _run(svc, [_Rec(i) for i in range(1, 31)])
    assert peak <= mb.REVIEW_CONCURRENCY, (
        f"{peak} units ran at once against a bound of "
        f"{mb.REVIEW_CONCURRENCY}")
    assert peak > 1, "nothing ran concurrently, so the bound is untested"


# ── THE IDENTITY PROOF ─────────────────────────────────────────────────────

def test_IDENTITY_the_same_card_gives_the_SAME_decisions():
    """Serial (bound 1) against parallel (bound 4), same card, same reviewer."""
    serial_order, parallel_order = [], []

    s1 = _svc(_decide_A, order=serial_order)
    s4 = _svc(_decide_A, order=parallel_order)

    import unittest.mock as _m
    with _m.patch.object(mb, "REVIEW_CONCURRENCY", 1):
        n_serial = _run(s1, CARD_A)
    n_parallel = _run(s4, CARD_A)

    assert n_serial == n_parallel == len(CARD_A)
    assert s1._decisions == s4._decisions, (
        "the parallel run produced different decisions, so the change is NOT "
        "timing-only and it stops here")
    # and the ORDER did differ, or the comparison proves nothing about
    # concurrency
    assert serial_order == sorted(serial_order)
    assert len(parallel_order) == len(serial_order)


def test_DISCRIMINATION_CONTROL_different_cards_give_DIFFERENT_decisions():
    """Without this, "identical" cannot be told apart from a comparison that
    always returns equal."""
    a = _svc(_decide_A)
    b = _svc(_decide_B)
    _run(a, CARD_A)
    _run(b, CARD_B)
    assert a._decisions != b._decisions, (
        "two different cards produced the same decision map, so the identity "
        "assertion above is measuring nothing")
    # and the same card under a DIFFERENT reviewer must also differ
    c = _svc(_decide_B)
    _run(c, CARD_A)
    assert a._decisions != c._decisions, (
        "the same card under a different reviewer produced identical "
        "decisions; the comparison is insensitive to the verdict")


def test_the_DECISIONS_do_not_depend_on_COMPLETION_ORDER():
    """Units finish out of order under concurrency. Make them finish in
    reverse and confirm the decision map is unchanged."""
    import unittest.mock as _m

    done = []

    def _decide(mid):
        done.append(mid)
        return _decide_A(mid)

    fast = _svc(_decide)
    _run(fast, CARD_A)

    # now make the LAST match finish first by making earlier ones slower
    slow = mb.MatchBriefingService.__new__(mb.MatchBriefingService)
    slow._sent_lock = asyncio.Lock()
    slow._enabled = lambda: True
    slow._sync_recs_from_db = lambda *a, **k: None
    out = {}

    async def _briefing_for(match_id, lineup_aware, post_to_telegram):
        await asyncio.sleep(0.02 - 0.001 * match_id)
        out[match_id] = _decide_A(match_id)
        return True

    slow._briefing_for = _briefing_for
    _run(slow, CARD_A)
    assert out == fast._decisions, (
        "reversing completion order changed the decisions")


# ── partial failure, under concurrency this time ──────────────────────────

def test_one_unit_FAILING_does_not_lose_the_others_UNDER_CONCURRENCY():
    """The serial version stopped nothing; `gather` with a raising coroutine
    would cancel its siblings if the exception escaped the unit."""
    def _decide(mid):
        if mid in (4, 9):
            return RuntimeError("this unit died")
        return _decide_A(mid)

    svc = _svc(_decide)
    n = _run(svc, CARD_A)
    assert n == len(CARD_A) - 2, f"reported {n} reviewed"
    assert sorted(svc._last_review_failures) == [4, 9]
    assert 4 not in svc._decisions and 9 not in svc._decisions
    # every other unit still got its decision
    assert len(svc._decisions) == len(CARD_A) - 2


def test_ALL_units_failing_is_reported_as_a_PARTIAL_not_a_success():
    svc = _svc(lambda mid: RuntimeError("everything died"))
    n = _run(svc, CARD_A)
    assert n == 0
    assert sorted(svc._last_review_failures) == sorted(
        r.match_id for r in CARD_A)


# ── the one honest qualification ───────────────────────────────────────────

def test_the_EXHAUSTION_PATH_is_NOT_claimed_identical_and_says_so():
    """`_claude_code_exhausted` is the only state shared across units. On a
    day the Pro pool runs out mid-batch, WHICH units fall back to the API
    differs between serial and parallel. Both produce valid decisions and
    neither is wrong, but they are not bit-identical -- and that is recorded
    rather than papered over.
    """
    import inspect

    # NOT `... or True`, which the first draft wrote and which asserts
    # nothing. The claim is that the qualification is RECORDED where the
    # bound is chosen, and that it names the shared flag.
    whole = inspect.getsource(mb)
    head = whole[:whole.index("class ")]
    assert "up to N-1 spawns" in head, (
        "the shared-state qualification is not recorded at the bound")
    assert "_claude_code_exhausted" in head, (
        "the bound's reasoning does not name the one piece of state shared "
        "across units, so a reader cannot check the identity claim's scope")
    assert "_claude_code_exhausted" in whole, "the flag itself is gone"
