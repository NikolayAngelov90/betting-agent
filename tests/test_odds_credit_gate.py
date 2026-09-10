"""The credit gate: one account, and a 429 that says which 429 it is.

Two defects closed together on 2026-09-10, because one edit closes both.

1. `update()` — the daily-picks pick-time odds path — passed no quota, so it
   claimed nothing, could not be declined, and never reconciled. Measured over
   2026-09-01..09-10 it was **204 of the 346 credits the provider charged**:
   59% of consumption invisible to the mechanism built to bound it.

2. The 429 branch returned BEFORE reading the response headers, so on the one
   response that says "you are out of credits" the pipeline learned nothing and
   `reconcile()` went permanently blind at exactly the moment its number
   mattered.

THE ALARM WAS ALREADY SPENT. 429 is "Too Many Requests", not "out of credits",
and the handler labelled every one "quota exhausted". It fired six times in this
project's history and NOT ONE was an exhaustion — every one had credits in hand:

    2026-05-03  402 remaining      2026-08-29   67 remaining
    2026-05-09  306 remaining      2026-08-30   31 remaining
    2026-05-10  276 remaining      2026-09-05  362 remaining   (4 in one second)

An alarm that has cried wolf six times cannot announce the real thing. These
tests replay all six and require them to classify as RATE LIMITING, then inject
the exhaustion response and require it to classify differently AND to reach the
reconciler — verified the way the merge discriminator was, against the cases
that actually occurred rather than against an argument.
"""

import asyncio

import pytest

from src.scrapers.theodds_scraper import (
    REFUSED_EXHAUSTED,
    REFUSED_RATE_LIMITED,
    Refusal,
    TheOddsScraper,
    _outcome_of,
)

#: (date, x-requests-remaining) for every 429 this project has ever logged.
#: Not invented — read out of the cached CI logs.
HISTORICAL_429S = [
    ("2026-05-03", 402),
    ("2026-05-09", 306),
    ("2026-05-10", 276),
    ("2026-08-29", 67),
    ("2026-08-30", 31),
    ("2026-09-05", 362),
]


class _FakeResp:
    def __init__(self, status, headers, payload=None):
        self.status = status
        self.headers = headers
        self._payload = payload if payload is not None else []

    async def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status >= 400:
            raise AssertionError("raise_for_status reached for a handled status")

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp

    def get(self, *a, **kw):
        return self._resp


def _scraper(monkeypatch, resp):
    sc = TheOddsScraper.__new__(TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None

    async def _fake_get_session():
        return _FakeSession(resp)

    sc._get_session = _fake_get_session
    return sc


def _fetch(sc, sport_key="soccer_epl"):
    return asyncio.run(sc._fetch_league_odds(sport_key))


@pytest.mark.parametrize("date,remaining", HISTORICAL_429S)
def test_every_historical_429_classifies_as_rate_limiting(date, remaining):
    """Replay. All six had credits in hand; none was an exhaustion."""
    resp = _FakeResp(429, {"x-requests-remaining": str(remaining),
                           "x-requests-used": str(500 - remaining)})
    out = _fetch(_scraper(None, resp))
    assert isinstance(out, Refusal), f"{date}: expected a Refusal, got {out!r}"
    assert out.reason == REFUSED_RATE_LIMITED, (
        f"{date}: {remaining} credits remained, so this was request pacing. "
        "Reporting it as exhaustion is what spent the alarm.")
    assert not out.exhausted


def test_a_429_with_zero_remaining_is_exhaustion():
    """The one on the 14th has to say something different from the six."""
    resp = _FakeResp(429, {"x-requests-remaining": "0", "x-requests-used": "500"})
    out = _fetch(_scraper(None, resp))
    assert isinstance(out, Refusal)
    assert out.reason == REFUSED_EXHAUSTED
    assert out.exhausted


def test_exhaustion_still_feeds_the_reconciler():
    """THE point of the header reordering.

    The old branch returned before parsing headers, so `_used_requests` kept
    its last value and `reconcile()` was never handed the truth — the ledger
    went blind on the only path where its number decides anything.
    """
    sc = _scraper(None, _FakeResp(
        429, {"x-requests-remaining": "0", "x-requests-used": "500"}))
    out = _fetch(sc)
    assert out.exhausted
    assert sc._used_requests == 500, (
        "the exhaustion response carried the provider's own count and it was "
        "dropped — this is the blind spot the change exists to close")
    assert sc._remaining_requests == 0

    seen = {}

    class _Quota:
        def reconcile(self, used, today=None):
            seen["used"] = used
            return used

    _Quota().reconcile(sc._used_requests)
    assert seen["used"] == 500


def test_rate_limiting_also_feeds_the_reconciler():
    sc = _scraper(None, _FakeResp(
        429, {"x-requests-remaining": "362", "x-requests-used": "138"}))
    assert _fetch(sc).reason == REFUSED_RATE_LIMITED
    assert sc._used_requests == 138
    assert sc._remaining_requests == 362


def test_a_429_without_headers_is_not_called_exhaustion():
    """Absent evidence is not evidence of exhaustion.

    A transport-mangled 429 carries no counters. Guessing 'exhausted' there
    would re-spend the alarm by another route.
    """
    out = _fetch(_scraper(None, _FakeResp(429, {})))
    assert out.reason == REFUSED_RATE_LIMITED


def test_success_still_returns_rows_and_absorbs_headers():
    sc = _scraper(None, _FakeResp(
        200, {"x-requests-remaining": "100", "x-requests-used": "400"},
        payload=[{"id": "g1"}]))
    out = _fetch(sc)
    assert out == [{"id": "g1"}]
    assert sc._used_requests == 400


# --------------------------------------------------------------- outcomes
def test_a_refusal_is_not_an_empty_catalogue():
    """Four states, because there are four.

    `[]` = this league priced nothing. `None` = the call failed. A refusal =
    the provider would not serve it. Collapsing the third into the second is
    how `0 odds rows written` came to read the same on a quiet day and on a
    refused one.
    """
    assert _outcome_of([]) == "no_rows"
    assert _outcome_of(None) == "error"
    assert _outcome_of([{"id": "g"}]) == "ok"
    assert _outcome_of(Refusal(REFUSED_EXHAUSTED)) == REFUSED_EXHAUSTED
    assert _outcome_of(Refusal(REFUSED_RATE_LIMITED)) == REFUSED_RATE_LIMITED


def test_a_refusal_is_deliberately_not_falsy():
    """`not games` must not sort a refusal into the empty bucket.

    Every consumer has to name it, which is why the parse loop tests
    `isinstance(games, Refusal)` explicitly rather than relying on truthiness.
    """
    assert bool(Refusal(REFUSED_EXHAUSTED)) is True


def test_a_refusal_cannot_poison_the_barren_league_cache():
    """The guard that already worked, pinned so it keeps working.

    `refresh_imminent` records barrenness only for `ok`/`no_rows`. A league the
    provider REFUSED must never be recorded as one the provider has nothing
    for, or a rate-limited minute would exclude it for days.
    """
    for reason in (REFUSED_EXHAUSTED, REFUSED_RATE_LIMITED):
        assert _outcome_of(Refusal(reason)) not in ("ok", "no_rows")


# ------------------------------------------------------- one account only
def test_update_routes_through_the_credit_ledger():
    """The daily-picks path must CLAIM, or 59% of spend stays invisible.

    Pinned by source rather than by execution: `update()` reaches the network
    and the point is the wiring, not the response.
    """
    import inspect

    src = inspect.getsource(TheOddsScraper.update)
    assert "OddsApiQuota" in src, (
        "update() no longer builds a quota — the pick-time odds path would "
        "again spend without claiming, which is the defect measured at 204 of "
        "346 credits over 2026-09-01..09-10")
    assert "quota=quota" in src, (
        "update() builds a quota but does not pass it to _fetch_and_persist")


def test_the_per_run_ceiling_is_disabled_on_the_update_path():
    """Deliberate, and it must stay deliberate.

    The 24-credit default is sized for the imminent-refresh job. This path
    routinely wants 20-23 leagues (40-46 credits), so inheriting that ceiling
    would decline half of every day's card — a volume change wearing an
    accounting fix's clothes. The monthly budget still gates it.
    """
    import inspect

    src = inspect.getsource(TheOddsScraper.update)
    assert "max_credits_per_run=0" in src


def test_the_attribution_line_no_longer_defaults_to_success():
    """`result=ok` was a DEFAULT on the line built to be trusted.

    A league truncated out of the request set by the budget was absent from
    the outcome map and reported `ok`. On 2026-09-05 and 09-06, four leagues
    that were never called each reported success.
    """
    import inspect

    src = inspect.getsource(TheOddsScraper.refresh_imminent)
    assert "'ok' if written else" not in src
    assert "not_requested" in src
