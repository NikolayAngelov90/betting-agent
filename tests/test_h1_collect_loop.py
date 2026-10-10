"""The in-job collection loop — Part B's four conditions, 2026-10-10.

WHY THE LOOP EXISTS. Measured from run history, the nominal 8-slot 120-minute
`closing-lines` cron fires a median of FOUR slots a day, shaped as a 12-24
minute pair plus ~200-minute gaps. The pair is below `MIN_GAP_MINUTES = 30`, so
it is one separated instant and the second slot buys nothing; against those
measured slots every `min_interval` from 120 down to 30 yields ZERO qualifying
trajectories. The scheduler cannot deliver the spacing, so the job produces it
itself.

CONDITION 3 IS WHAT THIS FILE MOSTLY TESTS: **the stop is checked between
waits, not after them.** An exhausted reservation must halt mid-sequence rather
than after spending again, and the remaining waits must be skipped — a halted
job holding a runner for two more hours is a different failure from an
overspend, and both are avoidable.

Driven by writing the ATTRIBUTED credit counter mid-sequence, through the real
`collection_state`, `h1_credits_used` and `collection_stop`. Nothing here
patches the stop.
"""

from datetime import date, datetime, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, OddsSnapshot, Team
from src.data.odds_quota import H1_RESERVED_CREDITS, OddsApiQuota, month_key

import scripts.h1_collect as h1c

NOW = datetime(2026, 10, 10, 14, 0)
BOOK = "TheOddsAPI-pinnacle"


def _mgr(tmp_path, name="loop.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def _seed_fixture(session, mid, kickoff):
    if session.get(Team, 1) is None:
        session.add(Team(id=1, name="H"))
        session.add(Team(id=2, name="A"))
        session.flush()
    session.add(Match(id=mid, league="england/premier-league",
                      match_date=kickoff, is_fixture=True,
                      home_team_id=1, away_team_id=2))


def _spend_h1(mgr, credits):
    """Write H1's ATTRIBUTED counter — the ledger row, not a patch."""
    q = OddsApiQuota(mgr, monthly_budget=450, safety_margin=50, for_h1=True)
    assert q._h1_store.record_attributed(
        credits, day=month_key(date(2026, 10, 10))) is not None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A database the loop reads, with the clock and sleeps observable."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        _seed_fixture(s, 1, NOW + timedelta(minutes=200))
        # one in-band instant so `raw_rows > 0` and NO_DATA is not the verdict
        for sel in ("Yes", "No"):
            s.add(OddsSnapshot(match_id=1, bookmaker=BOOK, market_type="btts",
                               selection=sel, odds_value=1.95,
                               observed_at=NOW - timedelta(minutes=10)))
        s.commit()
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)
    monkeypatch.setattr(h1c, "_load_env", lambda: None)

    slept: list[int] = []
    monkeypatch.setattr(h1c.time, "sleep", lambda s_: slept.append(s_))
    return mgr, slept, monkeypatch


# ══════════════════ CONDITION 3 — the stop halts BETWEEN waits ══════════════

def test_the_stop_halts_MID_SEQUENCE_and_skips_the_remaining_waits(wired, capsys):
    """THE CONDITION 3 PROOF. The counter is driven to the reservation after
    round 1, so round 2's pre-check must halt before spending OR waiting.

    A stop checked only after the waits would hold the runner for the full
    sequence and then report a halt it could have known about two hours
    earlier.
    """
    mgr, slept, monkeypatch = wired
    rounds = []

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        rounds.append(len(rounds) + 1)
        # Round 1 exhausts the reservation, exactly as a real spend would.
        if len(rounds) == 1:
            _spend_h1(mgr, H1_RESERVED_CREDITS)
        return ({"credits_claimed": 2, "odds_written": 1, "requested": ["x"]},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    rc = h1c.main(["--max-credits", "6", "--rounds", "3",
                   "--gap-minutes", "120", "--allow-sqlite"])
    out = capsys.readouterr().out

    assert rc == 0
    assert rounds == [1], (
        f"the loop ran {rounds} rounds after the reservation was exhausted — "
        f"the stop is not being checked before each round")
    assert ("HALTED after round 1" in out
            or "HALTED before round 2" in out), out
    assert "RESERVATION_EXHAUSTED" in out, out
    assert slept == [], (
        f"the loop slept {slept} after halting — a halted job must not hold a "
        f"runner through the remaining waits")
    assert "SKIPPED" in out


def test_an_unexhausted_loop_DOES_wait_between_rounds(wired, capsys):
    """The other direction, or the test above passes for the wrong reason.

    Without this, a loop that never waited at all would satisfy the halt
    assertion trivially — VAC-1 one level up.
    """
    mgr, slept, monkeypatch = wired

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        return ({"credits_claimed": 0, "odds_written": 0, "requested": []},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    h1c.main(["--max-credits", "6", "--rounds", "3", "--gap-minutes", "120", "--allow-sqlite"])
    assert slept == [120 * 60, 120 * 60], (
        f"expected two 120-minute waits between three rounds, got {slept}")


def test_no_wait_AFTER_the_last_round(wired):
    """Three rounds need two waits. A third would hold the runner for nothing."""
    mgr, slept, monkeypatch = wired

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        return ({"credits_claimed": 0, "odds_written": 0, "requested": []},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    h1c.main(["--max-credits", "6", "--rounds", "3", "--gap-minutes", "120", "--allow-sqlite"])
    assert len(slept) == 2


def test_the_PER_JOB_cap_stops_the_loop_without_waiting_through_it(wired, capsys):
    """The cap is a number stated before the run, and reaching it must also
    skip the remaining waits rather than spend-then-idle."""
    mgr, slept, monkeypatch = wired

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        return ({"credits_claimed": 6, "odds_written": 3, "requested": ["x"]},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    h1c.main(["--max-credits", "6", "--rounds", "3", "--gap-minutes", "120", "--allow-sqlite"])
    out = capsys.readouterr().out
    assert "per-job cap reached" in out, out
    assert slept == [], f"waited after the cap was reached: {slept}"


def test_the_stop_FAILS_CLOSED_when_it_cannot_be_evaluated(wired, capsys):
    """An unevaluable stop is not a stop that passed."""
    mgr, slept, monkeypatch = wired
    import scripts.h1_collection_check as chk

    def _boom(*a, **k):
        raise RuntimeError("ledger offline")

    monkeypatch.setattr(chk, "collection_state", _boom)
    rounds = []

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        rounds.append(1)
        return ({"credits_claimed": 0, "odds_written": 0, "requested": []},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    h1c.main(["--max-credits", "6", "--rounds", "3", "--gap-minutes", "120", "--allow-sqlite"])
    out = capsys.readouterr().out
    assert rounds == [], "it spent against a stop it could not evaluate"
    assert "UNEVALUATED" in out, out


# ══════════════════════ the registered values and the gap ═══════════════════

def test_min_interval_is_STRICTLY_below_the_in_job_gap():
    """The amended pre-registration's condition, as code.

    If `min_interval` ever reaches the gap, the per-league dedup rejects the
    very observations this loop creates — which is the defect the amendment
    withdrew the old premise over.
    """
    assert h1c.H1_MIN_INTERVAL_MINUTES < h1c.H1_GAP_MINUTES, (
        f"min_interval {h1c.H1_MIN_INTERVAL_MINUTES} is not below the "
        f"{h1c.H1_GAP_MINUTES}-minute gap — the dedup will drop round 2")
    assert h1c.H1_GAP_MINUTES - h1c.H1_MIN_INTERVAL_MINUTES >= 30, (
        "less than 30 minutes of slack against the measured scheduler jitter "
        "(sd 16.8 min, max +42.2)")


def test_three_rounds_at_the_gap_FIT_inside_the_registered_window():
    """2 x 120 = 240 minutes of span inside a 360-minute window, and the
    resulting triple must satisfy the registration's own predicate."""
    from scripts.h1_collection_check import qualifying_triple

    span = (h1c.H1_ROUNDS - 1) * h1c.H1_GAP_MINUTES
    assert span <= 360, f"{span} minutes of span exceeds the 360 window"
    t0 = datetime(2026, 10, 10, 14, 0)
    ts = [t0 + timedelta(minutes=i * h1c.H1_GAP_MINUTES)
          for i in range(h1c.H1_ROUNDS)]
    assert qualifying_triple(ts) is not None, (
        "three rounds at the configured gap do not form a qualifying triple")


def test_the_observation_window_starts_at_the_RESERVATION_not_the_month(wired, capsys):
    """`n` must not count observations H1 did not collect.

    The month reset is 10-01 and the reservation opens 10-09; counting from the
    reset would credit the experiment with eight days of normal capture.
    """
    mgr, slept, monkeypatch = wired

    async def _fake_round(window, min_interval, dry_run, max_credits,
                          leagues=None):
        return ({"credits_claimed": 0, "odds_written": 0, "requested": []},
                None)

    monkeypatch.setattr(h1c, "_one_round", _fake_round)
    h1c.main(["--max-credits", "6", "--rounds", "1", "--gap-minutes", "120", "--allow-sqlite"])
    out = capsys.readouterr().out
    assert "observations since  : 2026-10-09" in out, out
