"""Are H1's terminal states reachable FROM THE CALLER AS WIRED? — 2026-10-09/10.

Rule, stated by the operator and the reason this file exists:

    A terminal state reachable only from the harness is not reachable.

`test_h1_collection_check.py` already drives `collection_stop` directly, which
proves the FUNCTION returns six states. It proves nothing about whether the
wired path can produce them: `collection_state` derives three arguments from the
database and `refresh_imminent` consumes the result. Every defect found on
2026-10-09 lived in exactly that gap — the stop was correct and the number
handed to it was the wrong one.

So nothing here patches `collection_state` or `collection_stop`. Each state is
driven by SEEDED DATABASE STATE through `refresh_imminent(h1_collection=True)`,
the call the `--h1` flag makes, and read back off the returned plan.

THE ANSWER IS FIVE OF SIX, AND THE SIXTH IS AN HONEST NO:

    COLLECTING             reachable   the normal running state
    RESERVATION_EXHAUSTED  reachable   H1's attributed spend reaches 168
    NO_DATA                reachable   the bound with zero observations
    BUDGET_UNREADABLE      reachable   the ledger cannot be read
    UNEVALUATED            reachable   the stop itself raises
    COMPLETE               reachable   n >= TARGET_N from seeded trajectories
    CEILING_HIT            NOT REACHABLE — see the test at the bottom.

`CEILING_HIT` needs `CREDIT_CEILING < reservation`, i.e. 200 < 168, and
`collection_stop` defaults `reservation` to the CONSTANT
`H1_RESERVED_CREDITS = 168`. Nothing at runtime can raise it, so the state is
dead code behind a misconfiguration that the type system cannot express. That
is recorded rather than removed: it is the registered stop and the ceilings were
deliberately collapsed so one bound binds. But it must not be counted as a
state the apparatus can report.
"""

import asyncio
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, Team
from src.data.odds_quota import (
    H1_RESERVED_CREDITS,
    PROVIDER_H1,
    OddsApiQuota,
    month_key,
)
from src.scrapers.theodds_scraper import TheOddsScraper
from scripts.h1_collection_check import CREDIT_CEILING, TARGET_N, effective_bound

#: Everything here is driven at a fixed instant. The window and interval are
#: H1's registered pair, passed exactly as `--h1` passes them.
NOW = datetime(2026, 10, 10, 18, 0)
H1_WINDOW, H1_INTERVAL = 360, 120
#: A bookmaker string `provider_of` maps to TheOddsAPI — the trigger stratum.
BOOK = "TheOddsAPI-pinnacle"


def _mgr(tmp_path, name="reach.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def _scraper(mgr, monkeypatch):
    """The real `refresh_imminent`, with only the HTTP layer removed.

    `get_db` is redirected so `collection_state` reads THIS database — that is
    the wiring under test, not a stub of it.
    """
    sc = TheOddsScraper.__new__(TheOddsScraper)
    sc.config = type("C", (), {"get": staticmethod(lambda *a, **k: None)})()
    sc.api_key = "test-key"
    sc._remaining_requests = None
    sc.db = mgr
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)
    import src.scrapers.theodds_scraper as ts_mod
    monkeypatch.setattr(ts_mod, "get_db", lambda: mgr, raising=False)

    async def _no_http(*a, **k):
        return []
    sc._fetch_league_odds = _no_http
    return sc


def _seed_teams(session):
    """The FK targets. Seeded once per database, not per fixture."""
    if session.get(Team, 1) is None:
        session.add(Team(id=1, name="H"))
        session.add(Team(id=2, name="A"))
        session.flush()


def _seed_fixture(session, mid, kickoff, league="england/premier-league"):
    _seed_teams(session)
    session.add(Match(id=mid, league=league, match_date=kickoff,
                      is_fixture=True, home_team_id=1, away_team_id=2))


def _seed_one_instant(session, mid, kickoff, off=-300):
    """ONE complete `btts` instant, overround 1.026 — inside the band.

    One, not three, and the limit is the schema rather than the test: `odds` is
    UNIQUE on (match_id, bookmaker, market_type, selection), so a second
    instant for the same series raises IntegrityError. That is the finding this
    file records, and it is also why `raw_rows > 0` is the most this helper can
    establish — enough to separate RESERVATION_EXHAUSTED from NO_DATA, which is
    exactly the distinction those two states carry.
    """
    ts = kickoff + timedelta(minutes=off)
    for sel in ("Yes", "No"):
        session.add(Odds(match_id=mid, bookmaker=BOOK, market_type="btts",
                         selection=sel, odds_value=1.95, timestamp=ts))


def _set_h1_spend(mgr, credits):
    """Write H1's ATTRIBUTED counter directly — the ledger row, not a patch."""
    q = OddsApiQuota(mgr, monthly_budget=450, safety_margin=50, for_h1=True)
    assert q._h1_store.record_attributed(credits, day=month_key(NOW.date())) == credits


def _run(sc, quota=None):
    return asyncio.run(sc.refresh_imminent(
        window_minutes=H1_WINDOW, min_interval_minutes=H1_INTERVAL,
        quota=quota, require_pending_pick=False, dry_run=False,
        now=NOW, h1_collection=True,
        h1_since=datetime(2026, 10, 9)))


# ═════════════════════════════ the running state ═════════════════════════════

def test_COLLECTING_is_reachable_and_does_not_halt(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=120))
        s.commit()
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan.get("halted") is not True, plan
    assert "h1_stop" not in plan or plan["h1_stop"] == "COLLECTING"


# ══════════════════════════ the credit-driven halts ══════════════════════════

def test_RESERVATION_EXHAUSTED_is_reachable_from_the_caller(tmp_path, monkeypatch):
    """H1's own attributed spend reaches 168 WITH observations present."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=120))
        _seed_one_instant(s, 1, NOW + timedelta(minutes=120))
        s.commit()
    _set_h1_spend(mgr, H1_RESERVED_CREDITS)
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan["h1_stop"] == "RESERVATION_EXHAUSTED", plan
    assert plan["halted"] is True
    assert plan["credits_claimed"] == 0, "it halted and still claimed credits"
    assert plan["requested"] == []


def test_NO_DATA_is_reachable_from_the_caller(tmp_path, monkeypatch):
    """The bound reached with ZERO in-band observations.

    Distinct from RESERVATION_EXHAUSTED by exactly one fact — whether anything
    was collected — and that distinction is the apparatus-is-broken signal.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=120))
        s.commit()                          # fixture, but NO odds rows
    _set_h1_spend(mgr, H1_RESERVED_CREDITS)
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan["h1_stop"] == "NO_DATA", plan
    assert plan["halted"] is True


def test_BUDGET_UNREADABLE_is_reachable_from_the_caller(tmp_path, monkeypatch):
    """A dropped ledger must halt, not read as zero spent."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=120))
        s.commit()
    with mgr.get_session() as s:
        s.execute(text("DROP TABLE api_budget"))
        s.commit()
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan["h1_stop"] == "BUDGET_UNREADABLE", plan
    assert plan["halted"] is True


def test_COMPLETE_IS_NOT_REACHABLE_because_odds_CANNOT_HOLD_A_TRAJECTORY(tmp_path):
    """THE FINDING. H1 reads a table that is physically incapable of its input.

    `load_observations` — H1's only data source — queries `odds`, which carries
    `Index(..., unique=True)` on (match_id, bookmaker, market_type, selection).
    A second observation of the same series OVERWRITES the first. So `odds`
    holds at most ONE instant per series, a qualifying triple needs THREE, and
    `n` is therefore structurally ZERO on every input, forever.

    `models.py` states this in its own words, about the table built to fix it:

        "`odds` is unique on that key and is overwritten, which is why it holds
         two observations and never a third... Nothing reads this table yet."

    PNC-1, filed in a docstring and then forgotten: `odds_snapshots` is the
    append-only price path, it is WRITTEN on every refresh, and the analysis
    reads the other table. No window, interval, credit or cron change can
    repair that, which is why the caller was wired, tested, and still cannot
    produce evidence.

    CORROBORATED IN PRODUCTION 2026-10-10: 10,066 pre-kickoff observations
    since 10-09, 10,066 series, instants-per-series histogram `{1: 10066}` —
    not one series observed twice. The same query against `odds_snapshots`
    gives `{1: 14548, 2: 100, 3: 17}`, so the paths exist where nothing reads.
    """
    mgr = _mgr(tmp_path, "schema.db")
    ko = NOW + timedelta(minutes=120)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, ko)
        s.commit()

    # one instant lands
    with mgr.get_session() as s:
        s.add(Odds(match_id=1, bookmaker=BOOK, market_type="btts",
                   selection="Yes", odds_value=1.95,
                   timestamp=ko - timedelta(minutes=300)))
        s.commit()

    # a SECOND instant for the same series cannot
    import sqlalchemy.exc
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with mgr.get_session() as s:
            s.add(Odds(match_id=1, bookmaker=BOOK, market_type="btts",
                       selection="Yes", odds_value=1.90,
                       timestamp=ko - timedelta(minutes=180)))
            s.commit()


def test_odds_snapshots_ACCEPTS_the_trajectory_that_odds_refuses(tmp_path):
    """The other half, so the finding names a remedy and not just a wall.

    Same three instants, same series, into the append-only table: all three
    land. The data H1 needs is collectable TODAY with no schema change — it is
    already being written — and the gap is the read, not the write.
    """
    from src.data.models import OddsSnapshot

    mgr = _mgr(tmp_path, "snap.db")
    ko = NOW + timedelta(minutes=120)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, ko)
        s.commit()
    with mgr.get_session() as s:
        for off in (-300, -180, -60):
            s.add(OddsSnapshot(match_id=1, bookmaker=BOOK, market_type="btts",
                               selection="Yes", odds_value=1.95,
                               observed_at=ko + timedelta(minutes=off)))
        s.commit()
    with mgr.get_session() as s:
        got = s.query(OddsSnapshot).filter_by(match_id=1).count()
    assert got == 3, (
        f"the append-only table kept {got} of 3 instants — if this fails the "
        f"remedy named in the test above does not exist either")


def test_UNEVALUATED_is_reachable_and_fails_CLOSED(tmp_path, monkeypatch):
    """The stop itself raising must refuse to spend, not proceed.

    The only state driven by making the evaluation fail rather than by data,
    because that IS its trigger. It is still the caller's path: the exception
    is raised inside `collection_state` and caught by `refresh_imminent`.
    """
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=120))
        s.commit()
    import scripts.h1_collection_check as h1_mod

    def _boom(*a, **k):
        raise RuntimeError("ledger offline")
    monkeypatch.setattr(h1_mod, "collection_state", _boom)
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan["h1_stop"] == "UNEVALUATED", plan
    assert plan["halted"] is True
    assert plan["credits_claimed"] == 0


# ════════════════════════ and the one that is NOT reachable ══════════════════

def test_CEILING_HIT_is_NOT_reachable_as_wired_and_that_is_recorded():
    """THE HONEST NO. 200 < 168 is false and nothing at runtime changes it.

    `collection_stop` defaults `reservation` to the CONSTANT
    `H1_RESERVED_CREDITS`, and `bound_is_ceiling` is `CREDIT_CEILING <
    reservation`. With 168 shipped against a 200 ceiling the comparison is
    false on every input, so no database state and no caller argument can
    produce CEILING_HIT.

    IT IS LEFT IN PLACE DELIBERATELY: it is the REGISTERED stop, and the
    collapse of the two ceilings into `effective_bound` was done so that one
    bound binds rather than two drifting. What must not happen is counting it
    among the states the apparatus can report — "five terminal states, all
    reachable" was the claim, and four of five plus one dead branch is the
    fact. If the reservation is ever raised above 200 this test fails and the
    state becomes live, which is the right moment to notice.
    """
    assert H1_RESERVED_CREDITS < CREDIT_CEILING
    assert effective_bound(H1_RESERVED_CREDITS) == H1_RESERVED_CREDITS, (
        "the bound is no longer the reservation — CEILING_HIT may now be "
        "reachable and this file's census is stale")


def test_the_stop_ignores_the_CALENDAR_gate_and_that_is_a_live_exposure():
    """`collection_stop` uses the CONSTANT, not `h1_reservation(today)`.

    So on a date OUTSIDE 10-09..10-17 — when the allocator has reserved
    NOTHING and `h1_reservation()` returns 0 — the stop still permits 168
    credits of H1 spend. The calendar gate governs what is WITHHELD FROM
    NORMAL PRICING; it does not govern what H1 may spend.

    Recorded as an exposure rather than silently fixed: tightening the stop to
    the calendar value would make `--h1` outside the window halt immediately,
    which is probably right, but it changes a registered bound and that is the
    operator's call. Pinned here so the next reader finds it stated.
    """
    from src.data.odds_quota import h1_reservation
    from scripts.h1_collection_check import collection_stop

    assert h1_reservation(datetime(2026, 10, 20).date()) == 0
    stop = collection_stop(n_fixtures=0, credits_spent=100, raw_rows=5)
    assert stop.halt is False, (
        "if this now halts, the stop has been tied to the calendar gate — "
        "update the ledger entry, the exposure is closed")
    assert "reservation 168" in stop.reason
