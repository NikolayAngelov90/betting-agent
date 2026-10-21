"""The prerequisite for parallelising the Claude review — 2026-10-21.

Rule 1: the prerequisite ships BEFORE the concurrency, not with it.

TWO THINGS HAD TO BE TRUE FIRST.

A1. `_mark_sent` was a read-modify-write of one JSON file: load the whole
    day's key list, append, rewrite. Two concurrent reviewers both read
    `{today: [a]}`, one writes `[a, b]`, the other writes `[a, c]`, and the
    later write DROPS the earlier key. A lost update, and a silent one —
    nothing compares the marker set to the database, so the only symptom is a
    later `--picks` run finding no `final:<match_id>` freeze and regenerating
    picks behind a ruling Claude already made.

A2. A failed unit must be distinguishable from a decision. Serial code stops
    on an error; concurrent code does not, and a partial result that looks
    complete is this project's most frequent defect.
"""

import asyncio
import json
import os
from datetime import date
from pathlib import Path

import pytest

import src.reporting.match_briefing as mb


class _Svc:
    """The marker machinery alone, with no agent, DB or network."""

    _load_sent = mb.MatchBriefingService._load_sent
    _already_sent = mb.MatchBriefingService._already_sent
    _mark_sent = mb.MatchBriefingService._mark_sent
    _mark_sent_locked = mb.MatchBriefingService._mark_sent_locked

    def __init__(self):
        self._sent_lock = asyncio.Lock()


@pytest.fixture
def sent_path(tmp_path, monkeypatch):
    p = tmp_path / "briefings_sent.json"
    monkeypatch.setattr(mb, "_SENT_PATH", p)
    return p


# ── A1: the race ───────────────────────────────────────────────────────────

def test_the_marker_is_IDEMPOTENT(sent_path):
    svc = _Svc()
    for _ in range(3):
        svc._mark_sent("final", 42)
    # Asserted over the file's VALUES, not indexed by today's key: the key
    # is clock-derived and the gating-test scanner is right to refuse that
    # shape. The file has exactly one day in it, so this is the same claim.
    days = json.loads(sent_path.read_text())
    assert list(days.values()) == [["final:42"]]


def test_the_WRITE_IS_ATOMIC_and_leaves_no_torn_file(sent_path):
    """A torn file is read by `_load_sent`'s bare `except` as "nothing sent
    today", which would un-freeze every match on the next run."""
    svc = _Svc()
    svc._mark_sent("final", 1)
    # the temp file must not survive
    assert not list(sent_path.parent.glob("*.tmp")), (
        "a .tmp file was left behind, so the write was not an atomic replace")
    assert json.loads(sent_path.read_text())  # parses


def test_KNOWN_WRONG_concurrent_marks_DO_NOT_LOSE_EACH_OTHER(sent_path):
    """THE RACE, driven. Thirty concurrent markers, interleaved at the await
    point, must all survive."""
    svc = _Svc()
    ids = list(range(1, 31))

    async def drive():
        await asyncio.gather(*(svc._mark_sent_locked("final", i) for i in ids))

    asyncio.run(drive())
    days = json.loads(sent_path.read_text())
    assert len(days) == 1, f"more than one day in the file: {list(days)}"
    got = set(next(iter(days.values())))
    assert got == {f"final:{i}" for i in ids}, (
        f"{len(got)} of {len(ids)} markers survived -- a lost update")


def test_the_CONTROL_the_OLD_implementation_DOES_lose_updates(sent_path):
    """One direction is not a control. The pre-fix shape, reproduced: load
    once, then write the whole list back. It must LOSE keys, or the test above
    is not evidence that anything was fixed.
    """
    DAY = "2026-10-21"      # a FIXED key: this control is about the write
                            # pattern, not about today

    def old_mark(state_snapshot, kind, mid):
        # the defect: `state` was captured BEFORE the other writer's write
        s = dict(state_snapshot)
        s[DAY] = list(s[DAY]) + [f"{kind}:{mid}"]
        sent_path.write_text(json.dumps(s))

    sent_path.write_text(json.dumps({DAY: []}))
    snapshot = json.loads(sent_path.read_text())      # both readers read here
    old_mark(snapshot, "final", 1)
    old_mark(snapshot, "final", 2)                    # overwrites the first
    lost = json.loads(sent_path.read_text())[DAY]
    assert lost == ["final:2"], (
        f"the old shape did not lose a key ({lost}); then the race this stage "
        f"exists to fix was never real and the fix proves nothing")


def test_a_LOST_marker_was_SILENT_which_is_why_it_mattered(sent_path):
    """Recorded as a fact about the old design: nothing compares the marker
    set to the database, so a dropped key has no symptom until a later run
    regenerates picks behind a ruling."""
    svc = _Svc()
    svc._mark_sent("final", 7)
    assert svc._already_sent("final", 7) is True
    # and a marker that was never written simply reads as "not sent"
    assert svc._already_sent("final", 8) is False, (
        "a missing marker is indistinguishable from one that was never "
        "written -- which is the whole reason the lost update was silent")


def test_a_TORN_file_reads_as_NOTHING_SENT_which_is_the_unsafe_direction(
        sent_path):
    """Why the atomic write matters, stated as a test rather than a claim."""
    svc = _Svc()
    sent_path.write_text('{"2026-10-21": ["final:1"')      # truncated JSON
    assert svc._already_sent("final", 1) is False, (
        "a torn file no longer reads as 'nothing sent'; re-state the reason "
        "the write is atomic")


# ── A2: partial failure ───────────────────────────────────────────────────

def test_NULL_and_KEEP_are_DISTINGUISHABLE_in_every_consumer():
    """Audited 2026-10-21 across every `review_action` reader. The default
    that treats NULL as KEEP is the shape that has cost this project most
    often, and it is absent."""
    import pathlib as _p

    agent = _p.Path("src/agent/betting_agent.py").read_text(encoding="utf-8")
    assert 'SavedPick.review_action == "CHANGE"' in agent, (
        "the CHANGE-vs-model comparison no longer filters explicitly")

    rep = _p.Path("scripts/paper_trading_report.py").read_text(encoding="utf-8")
    assert 'p.review_action or "none"' in rep, (
        "the report no longer buckets a NULL review separately, so an "
        "unreviewed pick may be counted as KEEP")
    assert "reviewed = [p for p in picks if p.review_action]" in rep, (
        "the report's `reviewed` set no longer excludes NULL")
    # and nothing anywhere coerces a missing action to KEEP
    for src in (agent, rep,
                _p.Path("src/reporting/match_briefing.py").read_text(
                    encoding="utf-8")):
        assert 'review_action or "KEEP"' not in src
        assert "review_action or 'KEEP'" not in src


def test_KNOWN_WRONG_one_unit_FAILS_mid_batch_and_the_CARD_SURVIVES():
    """THE CONTROL. Force the middle unit to raise and confirm:

      * the batch CONTINUES and the other units are reviewed,
      * the failed match is TALLIED and NAMED, not merely warned about,
      * the failed unit gets NO freeze marker,
      * and the count reported is the number that actually succeeded.
    """
    svc = mb.MatchBriefingService.__new__(mb.MatchBriefingService)
    svc._sent_lock = asyncio.Lock()
    marks: list = []
    done: list = []

    async def _briefing_for(match_id, lineup_aware, post_to_telegram):
        if match_id == 2:
            raise RuntimeError("the reviewer died on this unit")
        done.append(match_id)
        marks.append(("final", match_id))
        return True

    svc._briefing_for = _briefing_for
    svc._sync_recs_from_db = lambda *a, **k: None
    svc._enabled = lambda: True

    class _Rec:
        def __init__(self, mid):
            self.match_id = mid

    picks = [_Rec(1), _Rec(2), _Rec(3)]
    n = asyncio.run(svc.finalize_picks_with_claude(picks))

    assert n == 2, f"reported {n} reviewed; two units succeeded"
    assert done == [1, 3], f"the batch stopped early: {done}"
    assert svc._last_review_failures == [2], (
        f"the failure was not tallied: {svc._last_review_failures}")
    assert ("final", 2) not in marks, (
        "the failed unit got a freeze marker, so a later run would treat it "
        "as decided")


def test_the_FAILURE_is_NAMED_in_the_log_not_just_arithmetic():
    """"reviewed 27/30" leaves the reader to subtract. The three that failed
    are listed, and the message says the picks stand as the model wrote
    them."""
    import inspect
    src = inspect.getsource(mb.MatchBriefingService.finalize_picks_with_claude)
    assert "FAILED" in src and "review_action=NULL" in src
    assert "PARTIAL" in src, (
        "the log does not say the review was partial, so a reader sees a "
        "smaller number and no reason")
    assert "_last_review_failures" in src, (
        "the failures are not retained, so no caller can report them")
