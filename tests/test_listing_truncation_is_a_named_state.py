"""`returned == limit` means UNKNOWN, not complete. The fourth instance.

    A QUERY AT ITS CAP IS INDISTINGUISHABLE FROM A QUERY THAT FOUND
    EVERYTHING — unless it says which.

`list_runs`'s own docstring recorded this defect for `closing-lines` at
``limit=40``: *"which read as 'no closing-lines runs in the window' rather than
as a truncated query."* On 2026-09-16 it recurred at ``limit=250`` for
`daily-picks`, which returned exactly 250 rows and hid 24 unaudited runs.
**Knowledge present in the file, caller not consulting it.**

RAISING THE LIMIT MOVES THE CLIFF. Naming the state removes it — the same move
that closed:

    ``[]`` vs ``None``                  measured-empty vs never-measured
    429-with-credits vs 429-with-zero   rate-limited vs exhausted
    no-hits vs no-log                   clean run vs unreadable one
"""

import json

import scripts.ci_audit as ci


def _fake_gh(per_workflow):
    """Return a `_sh` stand-in yielding N rows for whichever workflow is asked."""
    def _sh(*args):
        wf = args[args.index("--workflow") + 1]
        n = per_workflow.get(wf, 0)
        return json.dumps([
            {"databaseId": 1000 + i, "startedAt": f"2026-09-{(i % 28) + 1:02d}T10:00:00Z",
             "conclusion": "success", "event": "schedule", "status": "completed"}
            for i in range(n)])
    return _sh


def test_a_listing_below_the_cap_is_COMPLETE(monkeypatch):
    monkeypatch.setattr(ci, "_sh", _fake_gh({"daily-picks.yml": 3}))
    got = ci.list_runs(None, None, limit=250)
    assert got.complete is True
    assert got.warning() is None, (
        "a listing that did not reach its cap was flagged as truncated — the "
        "state must be raised only when it is actually unknown")


def test_a_listing_AT_the_cap_is_TRUNCATED_not_complete(monkeypatch):
    """THE REGRESSION. 250 returned from a limit of 250 is not an answer."""
    monkeypatch.setattr(ci, "_sh", _fake_gh({"daily-picks.yml": 250}))
    got = ci.list_runs(None, None, limit=250)
    assert got.complete is False, (
        "a query that returned exactly its limit reported itself complete — "
        "that is the defect this file exists for")
    assert "daily-picks" in got.truncated
    assert "TRUNCATED" in got.warning()
    assert "LOWER BOUND" in got.warning(), (
        "the warning does not say what it means for the COUNT, which is the "
        "number a reader carries away")


def test_truncation_is_reported_PER_WORKFLOW(monkeypatch):
    """One busy workflow must not mark a quiet one unknown, or vice versa."""
    monkeypatch.setattr(ci, "_sh", _fake_gh(
        {"daily-picks.yml": 250, "closing-lines.yml": 11,
         "paper-trading-report.yml": 4}))
    got = ci.list_runs(None, None, limit=250)
    assert set(got.truncated) == {"daily-picks"}, got.truncated


def test_a_DATE_FILTER_cannot_argue_the_cap_away(monkeypatch):
    """The cap is a property of the QUERY, not of the window asked for.

    `--since`/`--until` filter AFTER the provider truncated, so a narrow window
    can return two rows from a query that was cut off at 250. Those two rows
    are still all that is knowable, and the listing must still say so.
    """
    monkeypatch.setattr(ci, "_sh", _fake_gh({"daily-picks.yml": 250}))
    got = ci.list_runs("2026-09-01", "2026-09-02", limit=250)
    assert len(got) < 250
    assert got.complete is False, (
        "filtering to a narrow window cleared the truncation flag — the rows "
        "dropped by the filter are known to exist; the rows dropped by the cap "
        "are not")


def test_an_empty_listing_from_a_capped_query_is_still_unknown(monkeypatch):
    """The reading MOST changed by truncation is 'no runs'."""
    monkeypatch.setattr(ci, "_sh", _fake_gh({"daily-picks.yml": 250}))
    got = ci.list_runs("2027-01-01", "2027-01-02", limit=250)
    assert list(got) == []
    assert got.warning() is not None, (
        "an empty result from a truncated query claimed to be an empty window")


def test_the_listing_still_behaves_as_a_plain_list(monkeypatch):
    """Every existing caller indexes, iterates and len()s it."""
    monkeypatch.setattr(ci, "_sh", _fake_gh({"closing-lines.yml": 5}))
    got = ci.list_runs(None, None, limit=250)
    assert len(got) == 5
    assert got[0]["databaseId"] == 1000
    assert sorted(r["databaseId"] for r in got) == [1000, 1001, 1002, 1003, 1004]
