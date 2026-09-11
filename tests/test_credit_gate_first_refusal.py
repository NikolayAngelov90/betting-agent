"""THE GATE'S FIRST REFUSAL, exercised deliberately before it fires by itself.

The credit gate on the pick-time odds path shipped 2026-09-10 and, measured over
2026-09-01..09-10, would have declined NOTHING: peak ledger usage 342 against a
450 limit. That is a historical measurement, and it means the refusal branch
**has never executed in production**.

It is about to. At the measured 34.6 credits/day the ledger's spendable budget
is reached around 2026-09-12 and the provider's tier around 09-14, so the first
real decline is likely within a day or two of this file being written.

    THE NEVER-EXECUTED INVENTORY'S OWN FINDING WAS THAT THE MECHANISM WHICH
    HAD NEVER BEEN EXERCISED WAS THE ONE THAT WAS WRONG.

`experiment_record`'s per-series disposition filter shipped, passed its tests,
and was defective on its first real exercise the next day — because no
disposition had ever existed on a paper pick carrying a captured MODEL
observation, so the branch had never run. A branch that has never run is not
low-risk; it is unmeasured.

So this exercises the refusal against a temporary low limit in a test-scoped
ledger and pins the four properties that matter when it fires for real:

  1. the refusal is LOUD — it is not a debug line
  2. `not_requested` appears for the declined leagues, not `ok`
  3. the run CONTINUES — a declined league is not a crash
  4. nothing reads the refusal as "no odds available"

Everything runs against temp SQLite; conftest strips DATABASE_URL.
"""

import asyncio
import contextlib
import logging

import pytest
from loguru import logger as _loguru

import src.data.database as db_mod
from src.data.models import Base
from src.data.odds_quota import OddsApiQuota, credits_for
from src.scrapers.theodds_scraper import TheOddsScraper


@contextlib.contextmanager
def capture_logs(level="WARNING"):
    """Capture LOGURU records.

    pytest's `caplog` sees nothing here: this project logs through loguru, not
    stdlib logging, so an assertion built on caplog passes vacuously whatever
    the code does. Caught by this very file — the first draft asserted on
    caplog.records, found them empty, and the messages were sitting in stderr
    all along. A test that cannot observe the thing it asserts is not a test.
    """
    records = []
    sink = _loguru.add(lambda m: records.append(m.record), level=level)
    try:
        yield records
    finally:
        _loguru.remove(sink)


def _mgr(tmp_path, name="gate.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _quota(tmp_path, budget):
    """A ledger with a deliberately low ceiling, so the refusal is reachable."""
    return OddsApiQuota(_mgr(tmp_path), monthly_budget=budget,
                        safety_margin=0, max_credits_per_run=0)


def _scraper(monkeypatch, calls):
    """A scraper whose HTTP layer records what it was ASKED to fetch."""
    sc = TheOddsScraper.__new__(TheOddsScraper)
    sc.api_key = "k"
    sc._remaining_requests = None
    sc._used_requests = None

    async def _fake_fetch(sport_key):
        calls.append(sport_key)
        return []

    sc._fetch_league_odds = _fake_fetch
    return sc


# ═══════════════════════════════════════════════ the total refusal

def test_a_fully_spent_ledger_refuses_every_request(tmp_path):
    """Budget gone: NO requests, loudly, and the run returns rather than raises."""
    q = _quota(tmp_path, budget=4)
    assert q.claim_requests(2) == 2          # spend it all (2 req x 2 credits)
    assert q.remaining() == 0

    calls = []
    sc = _scraper(None, calls)
    with capture_logs("WARNING") as records:
        written = asyncio.run(sc._fetch_and_persist(
            {"england/premier-league": [{"match_id": 1}]}, quota=q))

    assert written == 0
    assert calls == [], (
        "the gate authorised an HTTP call against an exhausted ledger — the "
        "claim-before-spend contract is the whole mechanism")

    loud = [r for r in records
            if r["level"].no >= logging.WARNING
            and "budget exhausted" in r["message"]]
    assert loud, (
        "the refusal was not logged at WARNING or above. When this fires for "
        "real it has to be visible; a quiet refusal is indistinguishable from "
        "a quiet day, which is the failure mode this gate exists to end.")


def test_the_refusal_does_not_raise(tmp_path):
    """Property 3. A declined league degrades the card; it must not kill the run."""
    q = _quota(tmp_path, budget=0)
    sc = _scraper(None, [])
    written = asyncio.run(sc._fetch_and_persist(
        {"spain/laliga": [{"match_id": 2}]}, quota=q))
    assert written == 0


# ═══════════════════════════════════════════════ the partial refusal

def test_a_partial_budget_truncates_and_names_what_it_dropped(tmp_path):
    """The likelier first refusal: enough for some leagues, not all."""
    q = _quota(tmp_path, budget=credits_for(2))       # room for exactly 2
    calls = []
    sc = _scraper(None, calls)
    wanted = {
        "england/premier-league": [{"match_id": 1}],
        "spain/laliga": [{"match_id": 2}],
        "italy/serie-a": [{"match_id": 3}],
        "france/ligue-1": [{"match_id": 4}],
    }
    with capture_logs("WARNING") as records:
        asyncio.run(sc._fetch_and_persist(dict(wanted), quota=q))

    assert len(calls) == 2, (
        f"budget allowed 2 league requests but {len(calls)} were made")
    dropped = [r["message"] for r in records
               if "budget allows" in r["message"]]
    assert dropped, "the truncation was silent — which leagues went unpriced?"
    assert "skipping" in dropped[0]


def test_declined_leagues_report_not_requested_never_ok(tmp_path):
    """Property 2, and the defect it replaces.

    `result=ok` used to be the DEFAULT for any league absent from the outcome
    map — and a league the budget truncated away is exactly such a league. On
    2026-09-05 and 09-06 four leagues that were never called each reported
    success on the line whose purpose is to be trusted without parsing prose.
    """
    import inspect

    src = inspect.getsource(TheOddsScraper.refresh_imminent)
    assert "'ok' if written else" not in src
    assert "not_requested" in src

    q = _quota(tmp_path, budget=credits_for(1))
    calls = []
    sc = _scraper(None, calls)
    asyncio.run(sc._fetch_and_persist(
        {"england/premier-league": [{"match_id": 1}],
         "spain/laliga": [{"match_id": 2}]}, quota=q))

    outcomes = getattr(sc, "_last_league_outcomes", {})
    assert len(calls) == 1
    fetched = calls[0]
    # The league that WAS fetched has a real outcome; the declined one is
    # absent from the map entirely, which is what makes the attribution line's
    # fallback the thing that decides how it reads.
    assert len(outcomes) == 1, (
        f"the outcome map should describe only what was actually fetched, "
        f"got {outcomes}")
    assert "not_requested" not in outcomes.values()
    assert fetched


# ═══════════════════════════════════════════ refusal is not emptiness

def test_a_refusal_is_never_read_as_no_odds_available(tmp_path):
    """Property 4 — the one that matters most when this fires for real.

    A budget refusal and an empty catalogue must not converge. If they do, the
    pipeline prices a card from stale odds and reports a quiet day.
    """
    from src.scrapers.theodds_scraper import (
        REFUSED_EXHAUSTED,
        REFUSED_RATE_LIMITED,
        Refusal,
        _outcome_of,
    )

    assert _outcome_of([]) == "no_rows"
    for reason in (REFUSED_EXHAUSTED, REFUSED_RATE_LIMITED):
        assert _outcome_of(Refusal(reason)) != "no_rows"
        assert _outcome_of(Refusal(reason)) != "ok"

    # And a ledger refusal never even reaches the outcome map: the league is
    # dropped before any call, so it cannot be mistaken for a league that
    # answered with nothing.
    q = _quota(tmp_path, budget=0)
    calls = []
    sc = _scraper(None, calls)
    asyncio.run(sc._fetch_and_persist(
        {"england/premier-league": [{"match_id": 1}]}, quota=q))
    assert getattr(sc, "_last_league_outcomes", {}) in ({}, None) or \
        "no_rows" not in getattr(sc, "_last_league_outcomes", {}).values()
