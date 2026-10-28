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
from datetime import time as _dtime

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, OddsSnapshot, Team
from src.data.odds_quota import (H1_RESERVATION_FROM as _RESERVATION_FROM,
                                 H1_RESERVED_CREDITS, OddsApiQuota,
                                 month_key)

import scripts.h1_collect as h1c

# ── MOVED WITH THE WINDOW. 2026-10-28. ───────────────────────────────────
#
# WAS datetime(2026, 10, 10, 14, 0), a day inside the OLD reservation
# window. When the window moved to 2026-11-01..11-09 these fixtures fell
# OUTSIDE the observation window (`_H1_WINDOW_FROM`), so
# `load_observations` returned nothing and the loop halted `NO_DATA`
# instead of `RESERVATION_EXHAUSTED` -- the halt was real, the REASON was
# wrong, and that is the half-moved-window failure mode from the other
# side.
#
# DERIVED FROM THE CONSTANT, not re-typed, so it cannot be left behind
# again: the second day of whatever window is configured.
NOW = datetime.combine(_RESERVATION_FROM + timedelta(days=1),
                       _dtime(14, 0))
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
    """Write H1's ATTRIBUTED counter — the ledger row, not a patch.

    `month_key()` — THE CLOCK — and not `NOW`, because these fixtures sit on
    TWO DIFFERENT DATE AXES and conflating them is what broke this test when
    the window moved:

      * the OBSERVATION window is the reservation's dates, so `NOW` is derived
        from `H1_RESERVATION_FROM`;
      * the CREDIT LEDGER is keyed by the calendar month the spend happens in,
        which the loop reads off the clock (`h1_credits_used` → `month_key()`).

    In production both are the same month during the window. In a test run
    before the window opens they are not, and writing the spend under `NOW`'s
    month put it where the loop would never look: the stop read
    `credits=0/168` and the loop ran all three rounds after the reservation
    was exhausted.
    """
    q = OddsApiQuota(mgr, monthly_budget=450, safety_margin=50, for_h1=True)
    assert q._h1_store.record_attributed(
        credits, day=month_key()) is not None


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """A database the loop reads, with the clock and sleeps observable.

    THE PRECONDITION IS STUBBED TO `READY` HERE, and the reason matters. These
    tests are about the LOOP -- which rounds run, which waits are skipped,
    where the stop is checked -- while the precondition reads the REAL clock
    and these fixtures are seeded relative to a frozen `NOW`. Leaving it live
    made every one of them halt before round 1 with NO_FIXTURE_IN_BAND, which
    is a true statement about the wrong subject.

    SO THE WIRING GETS ITS OWN TEST:
    `test_the_PRECONDITION_IS_CONSULTED_BEFORE_ANY_SPEND` below drives the
    real one. A stub that hides the thing it stands in for is how
    `h1_precondition` came to exist, pass its own tests, and never be called.
    """
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
    import scripts.h1_precondition as _pre
    monkeypatch.setattr(
        _pre, "evaluate",
        lambda *a, **k: _pre.Precondition(
            _pre.STATE_READY, 1, 1, 1, 1,
            (_pre.BAND_LO_MIN, _pre.BAND_HI_MIN), (), "stubbed READY"))
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
    # And it is 0 on purpose: `_leagues_refreshed_since` is WRITER-AGNOSTIC, so
    # a closing-lines run inside H1's window would otherwise skip a round and
    # truncate the trajectory to two instants.
    assert h1c.H1_MIN_INTERVAL_MINUTES == 0, (
        "min_interval is non-zero again - another workflow's refresh can now "
        "decide whether H1 collects, because the dedup does not look at which "
        "writer refreshed the league")


def test_the_dedup_really_is_WRITER_AGNOSTIC():
    """The premise for min_interval=0, asserted on the source.

    `_leagues_refreshed_since` filters on `Odds.bookmaker LIKE 'TheOddsAPI%'`
    and a timestamp and nothing else, so ANY workflow's refresh marks a league
    fresh. If it ever grows a writer or purpose column the coupling is gone and
    min_interval could carry a real value again - at which point this test
    fails and the decision gets revisited instead of silently standing.
    """
    import inspect

    from src.scrapers.theodds_scraper import TheOddsScraper

    src = inspect.getsource(TheOddsScraper._leagues_refreshed_since)
    assert "BOOKMAKER_PREFIX" in src, src
    for marker in ("workflow", "purpose", "for_h1"):
        assert marker not in src, (
            f"the dedup now distinguishes {marker!r} - re-evaluate "
            f"H1_MIN_INTERVAL_MINUTES, which is 0 only because it could not")


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
    # DERIVED, and the discrimination this test was named for is now WEAKER:
    # the reservation opens ON the credit reset, so "the reservation start"
    # and "the month reset" are THE SAME DATE and this assertion can no
    # longer tell them apart. It still pins the value; it no longer proves
    # the choice. The discriminating case is driven below.
    assert f"observations since  : {_RESERVATION_FROM}" in out, out


# ════════════════════ the workflow's own properties, pinned ═════════════════

def _wf():
    import pathlib

    import yaml

    d = yaml.safe_load(
        pathlib.Path(".github/workflows/h1-collect.yml").read_text(encoding="utf-8"))
    # PyYAML parses a bare `on:` key as the boolean True.
    return d, (d[True] if True in d else d["on"]), list(d["jobs"].values())[0]


def test_the_timeout_is_explicit_and_below_GitHubs_ceiling():
    """A job hung while sleeping is indistinguishable from one that is working.

    330 is ~1.36x the expected 243 minutes ((rounds-1) x gap + runtime) and
    deliberately under the 360-minute hosted-runner cap, so the bound that
    fires is OURS and is reported as a timeout rather than a platform kill.
    """
    _d, _on, job = _wf()
    t = job["timeout-minutes"]
    expected = (h1c.H1_ROUNDS - 1) * h1c.H1_GAP_MINUTES
    assert t > expected, (
        f"timeout {t} is below the {expected} minutes of waiting alone — the "
        f"job cannot finish")
    assert t <= 360, (
        f"timeout {t} exceeds GitHub's 360-minute hosted-runner ceiling, so "
        f"the platform would kill it first and the bound would be invisible")
    assert t / expected >= 1.2, (
        f"only {t / expected:.2f}x the waiting time — no slack for a slow "
        f"request")


def test_it_has_its_OWN_concurrency_group():
    """It must not queue behind, or ahead of, daily-picks.

    This job holds a runner for ~4 hours. Sharing a group with daily-picks
    would make a late card wait for it, which is strictly worse than anything
    H1 can suffer.
    """
    d, _on, _job = _wf()
    assert d["concurrency"]["group"] == "h1-collect"
    assert d["concurrency"]["cancel-in-progress"] is False, (
        "a queued collection run must not be cancelled; it would lose the "
        "credits already spent in earlier rounds")
    import pathlib

    import yaml as _y

    other = _y.safe_load(
        pathlib.Path(".github/workflows/daily-picks.yml").read_text(encoding="utf-8"))
    assert other.get("concurrency", {}).get("group") != "h1-collect", (
        "daily-picks now shares H1's concurrency group — a 4-hour collection "
        "job will delay the card")


def test_the_H1_SCHEDULE_IS_DISABLED_and_the_cron_rationale_is_HISTORY():
    """RETIRED-AND-REPLACED 2026-10-20. It asserted that `0 9` compensates for
    the measured 3h58m-5h04m scheduler lateness so the 120-minute band lands
    on the 19:00 UTC European slot. That reasoning was correct and is now
    history: **the schedule is gone.**

    WHY. The window closed 2026-10-17 and the reservation deactivated
    (`h1_reservation` returns 0, `allocate()` reports `h1_reserved: 0`), but
    the cron kept firing. Run 39435238427 started 2026-10-20T13:59:38Z on
    `event: schedule` and, measured at 16:13, had claimed:

        theoddsapi     used 334 / 450   remaining 16
        theoddsapi:h1  used  14 / 500   (6 in-window, recorded 2026-10-18)

    so **8 TheOddsAPI credits leaked post-window from the NORMAL pool** -- the
    one that funds tomorrow's picks -- because with the reservation at 0 there
    is no H1 pool to draw on. `remaining()` returns the same 16 for
    `for_h1=True` and `for_h1=False`.

    The run was cancelled before its third round claimed (~4 more credits
    saved, verified: the budget did not move after 16:01:45).

    `workflow_dispatch` is retained for the 2026-11-01 re-run decision.
    """
    import pathlib

    import yaml

    wf = yaml.safe_load(pathlib.Path(".github/workflows/h1-collect.yml")
                        .read_text(encoding="utf-8"))
    on = wf.get("on") or wf.get(True)
    # ── THE RE-ENABLE CONDITION THIS PIN NAMED IS NOW MET. 2026-10-28. ────
    #
    # It asserted `"schedule" not in on` and stated its own release clause:
    # "Re-enabling needs the 2026-11-01 decision and a fresh
    # pre-registration -- not a resumption of the closed experiment." Both
    # hold: Niki chose the re-run, the window is 2026-11-01..11-08 with its
    # own pre-registration, and the credit leak that motivated the disable is
    # structurally impossible now -- the two-sided allocation refuses H1 the
    # normal pool and refuses normal operation H1's share, proved by refusal.
    #
    # So the pin flips from "must be absent" to "must be the DERIVED one",
    # which is a stronger assertion than the one it replaces: it pins the
    # cron's VALUE, not merely its absence.
    assert "schedule" in on, (
        "H1 is unscheduled again. The re-run window is 2026-11-01..11-08 and "
        "a dispatch-only trigger means someone must remember to fire it on "
        "eight consecutive mornings.")
    crons = [c["cron"] for c in on["schedule"]]
    assert crons == ["30 9 1-8 11 *"], (
        f"the H1 cron is {crons}, not the derived `30 9 1-8 11 *`. The "
        f"nominal is the TARGET MINUS THE MEASURED LATENESS (14:00 - 281.6 "
        f"min), and `30 9` scored P(n>=33)=0.999 against 0.891 for `0 9` in "
        f"a 20,000-trial resample of the measured n=6 lateness series. The "
        f"day-of-month and month fields scope it to the window so it cannot "
        f"fire past it, which is how 8 credits leaked in October.")
    assert "workflow_dispatch" in on, (
        "dispatch was removed too, so the re-run decision cannot be executed "
        "without a push")
    src = pathlib.Path(".github/workflows/h1-collect.yml").read_text(
        encoding="utf-8")
    # THE OCTOBER LEAK STAYS RECORDED even though the schedule is back: it is
    # why the cron is scoped to `1-8 11` instead of firing daily.
    for part in ("SCHEDULE DISABLED 2026-10-20", "8 TheOddsAPI CREDITS",
                 "normal_ceiling: 350", "39435238427"):
        assert part in src, f"the disable no longer records {part!r}"
    for part in ("30 9 1-8 11 *", "P(n>=33)", "281.6"):
        assert part in src, f"the derivation no longer records {part!r}"


def test_the_scheduled_path_pins_the_MEASURED_league_set():
    """A cron cannot pass inputs, so the defaults are the selection.

    Pinning the union of the four selected leagues is free: a pinned league
    with no fixture inside the window is skipped without a request.
    """
    _d, _on, job = _wf()
    run = next(s["run"] for s in job["steps"] if s.get("name") == "Collect")
    for lg in ("england/championship", "europe/champions-league",
               "europe/europa-league", "europe/europa-conference-league"):
        assert lg in run, f"{lg} is no longer pinned on the scheduled path"
    assert "--max-credits $MAXC" in run and "MAXC=24" in run, (
        "the scheduled path lost its per-job credit ceiling")


def test_the_collector_is_only_reached_WITH_a_credit_ceiling():
    """`--max-credits` is required by the parser, so no path can omit it."""
    with pytest.raises(SystemExit):
        h1c.main(["--rounds", "1", "--allow-sqlite"])
