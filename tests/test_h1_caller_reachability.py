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

THE ANSWER IS SIX OF SEVEN, AND THE SEVENTH IS AN HONEST NO.
`COMPLETE` BECAME REACHABLE ON 2026-10-10 when `load_observations` was
repointed at the append-only `odds_snapshots`; before that no database
state could produce it, because `odds` rejects a series' second instant:

    COLLECTING             reachable   the normal running state
    RESERVATION_EXHAUSTED  reachable   H1's attributed spend reaches 168
    NO_DATA                reachable   the bound with zero observations
    BUDGET_UNREADABLE      reachable   the ledger cannot be read
    UNEVALUATED            reachable   the stop itself raises
    COMPLETE               reachable   since 2026-10-10 — see the note below
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
from src.data.models import Base, Match, Odds, OddsSnapshot, Team
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
    """ONE complete `btts` instant into the APPEND-ONLY table.

    Written to `odds_snapshots`, because that is what `load_observations` reads
    since 2026-10-10. Enough to separate RESERVATION_EXHAUSTED from NO_DATA,
    which is exactly the distinction those two states carry.
    """
    ts = kickoff + timedelta(minutes=off)
    for sel in ("Yes", "No"):
        session.add(OddsSnapshot(match_id=mid, bookmaker=BOOK,
                                 market_type="btts", selection=sel,
                                 odds_value=1.95, observed_at=ts))


def _seed_trajectory(session, mid, kickoff, offsets=(-300, -180, -60)):
    """A QUALIFYING TRIPLE — three separated instants, overround 1.026.

    THIS HELPER COULD NOT EXIST BEFORE 2026-10-10. `odds` is UNIQUE on
    (match_id, bookmaker, market_type, selection), so the second instant raised
    IntegrityError and `COMPLETE` was unreachable from the caller for that
    reason alone. Pointing `load_observations` at the append-only table is what
    made the experiment's own success state reachable.

    Gaps 120/120: both above MIN_GAP (30) and ratio 1.0, strictly below the 2.0
    ceiling — a qualifying triple by the registration's own predicate.
    """
    for off in offsets:
        ts = kickoff + timedelta(minutes=off)
        for sel in ("Yes", "No"):
            session.add(OddsSnapshot(match_id=mid, bookmaker=BOOK,
                                     market_type="btts", selection=sel,
                                     odds_value=1.95, observed_at=ts))


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


def test_COMPLETE_is_NOW_reachable_from_the_caller(tmp_path, monkeypatch):
    """The state the whole experiment exists to reach — reachable since
    2026-10-10, and unreachable before it for a structural reason.

    Driven by seeding TARGET_N qualifying triples into `odds_snapshots` and
    running the caller's own call. Until the analysis was repointed this test
    could not be written: `odds` forbids the second instant outright, so no
    database state could produce n > 0.
    """
    mgr = _mgr(tmp_path, "complete.db")
    with mgr.get_session() as s:
        for i in range(TARGET_N):
            ko = NOW + timedelta(minutes=120 + i)
            _seed_fixture(s, 100 + i, ko)
            _seed_trajectory(s, 100 + i, ko)
        s.commit()
    plan = _run(_scraper(mgr, monkeypatch))
    assert plan["h1_stop"] == "COMPLETE", plan
    assert plan["halted"] is True
    assert f"n={TARGET_N}" in plan["h1_reason"], plan["h1_reason"]
    assert plan["credits_claimed"] == 0, "it reached the target and still spent"


def test_odds_CANNOT_hold_a_trajectory_and_that_is_why_n_was_zero(tmp_path):
    """THE FINDING, kept as its own test now that the remedy has landed.

    `odds` carries `Index(..., unique=True)` on
    (match_id, bookmaker, market_type, selection), so a second observation of a
    series OVERWRITES the first. H1 read that table for 46 days while
    `odds_snapshots` — built in Stage 18 for exactly this question — went
    unread. Measured 2026-10-10: 10,066 pre-kickoff series in `odds` since
    10-09, instants-per-series histogram {1: 10066}; not one series ever
    observed twice.

    Pinned so that repointing the analysis back at `odds` fails loudly here
    rather than quietly returning n=0 forever.
    """
    mgr = _mgr(tmp_path, "schema.db")
    ko = NOW + timedelta(minutes=120)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, ko)
        s.commit()
    with mgr.get_session() as s:
        s.add(Odds(match_id=1, bookmaker=BOOK, market_type="btts",
                   selection="Yes", odds_value=1.95,
                   timestamp=ko - timedelta(minutes=300)))
        s.commit()

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


def test_the_instant_is_a_DATETIME_whatever_the_driver_returns(tmp_path):
    """Raw `text()` SQL on SQLite hands back STRINGS; psycopg2 hands back
    datetimes. `qualifying_triple` subtracts them.

    Before the coercion this raised
    `TypeError: unsupported operand type(s) for -: 'str' and 'str'` and
    `refresh_imminent` fail-closed to UNEVALUATED — a halt for a type error
    wearing the label "could not evaluate the stop".

    IT WAS LATENT AND THE OTHER DEFECT HID IT: the subtraction only runs on a
    series with three instants, and `odds` could never hold two, so no input
    ever reached it. Asserted on the loader directly so it does not depend on
    `COMPLETE` being reachable to be caught.
    """
    from scripts.h1_collection_check import load_observations, qualifying_triple

    mgr = _mgr(tmp_path, "dtype.db")
    ko = NOW + timedelta(minutes=120)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, ko)
        _seed_trajectory(s, 1, ko)
        s.commit()

    with mgr.get_session() as s:
        obs, _dropped = load_observations(s, datetime(2026, 10, 9))

    assert obs, "the loader returned nothing — the seed or the band changed"
    for o in obs:
        assert isinstance(o["ts"], datetime), (
            f"instant came back as {type(o['ts']).__name__}: {o['ts']!r} — "
            f"qualifying_triple subtracts these and will raise TypeError")
        assert isinstance(o["match_date"], datetime), type(o["match_date"])

    times = [o["ts"] for o in obs]
    assert qualifying_triple(times) is not None, (
        "three instants 120 minutes apart did not qualify — check MIN_GAP and "
        "MAX_INTERVAL_RATIO against the seeded offsets")
