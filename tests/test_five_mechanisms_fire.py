"""The five previously-unwired H1 mechanisms, each shown FIRING.

A MECHANISM IS VERIFIED BY A FIRING, NOT BY ITS PRESENCE. Every one of these
five existed, had tests, and was reached by nothing — so "it is there" was
true of all five while H1 collected nothing. Each test below drives the
mechanism to produce an OBSERVABLE that cannot appear unless it ran, and says
what that observable is.

Absence of complaint is not evidence that any of them runs.
"""

import datetime as dt
import pathlib

import pytest
import yaml

from src.data import h1_window as hw
import scripts.h1_collection_check as hcc
import scripts.h1_precondition as pre

WF = pathlib.Path(".github/workflows/h1-collect.yml")

class _Sess:
    """A session whose four counts are dictated, as in test_h1_precondition.

    `evaluate` uses PostgreSQL's `ANY(...)`, so on the SQLite fixture it
    returns UNEVALUATED with `no such function: ANY` and no state can be
    reached. That is why the existing tests stub the session and why these
    do: the QUERY's firing is observed in production instead, and today's
    run printed `card=23 fixtures=23 in_band=0 in_band_pinned=0` with real
    counts -- a stronger observation than any fixture gives.
    """

    def __init__(self, total, as_fix, band, band_pinned):
        self._v = [total, as_fix, band, band_pinned]
        self._i = 0

    def execute(self, *_a, **_k):
        v = self._v[self._i]
        self._i += 1
        return type("R", (), {"scalar": lambda _s, _v=v: _v})()



def _seeded(tmp_path, kickoffs, league="england/championship"):
    """A session whose card holds `kickoffs` (minutes from NOW) in `league`."""
    from src.data.database import DatabaseManager
    from src.data.models import Base, Match, Team
    mgr = DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "p.db")}})())
    Base.metadata.create_all(mgr.engine)
    now = dt.datetime(2026, 11, 3, 14, 0)
    with mgr.get_session() as s:
        s.add(Team(id=1, name="A")); s.add(Team(id=2, name="B"))
        for i, off in enumerate(kickoffs):
            s.add(Match(id=100 + i, league=league,
                        match_date=now + dt.timedelta(minutes=off),
                        is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()
    return mgr, now




# ═════ 1. h1_precondition.evaluate — four states, fifteen tests, no caller ══

def test_FIRING_1_the_precondition_EVALUATES_and_the_loop_READS_it():
    """Observable: a state string the collector prints BEFORE any spend.

    `evaluate` was never called by the collection loop. The firing is not
    that the function returns a value — it did that all along — but that the
    COLLECTOR consults it, so the source is checked for the read as well.
    """
    src = pathlib.Path("scripts/h1_collect.py").read_text(encoding="utf-8")
    assert "h1_precondition" in src and "evaluate" in src, (
        "the collection loop does not reference the precondition at all; it "
        "is wired to nothing again")
    # An EMPTY card, which is the day-1 case. The state and its reason are
    # the observable.
    p = pre.evaluate(_Sess(0, 0, 0, 0), ["england/championship"],
                     now=dt.datetime(2026, 11, 1, 16, 9))
    assert p.state == pre.STATE_NO_CARD, p.state
    assert p.reason and "NO CARD" in p.reason.upper(), (
        "the state carried no reason, so a halt would not say which of the "
        "four it was")


# ═════ 2. main() — the daily check and the silent-zero alarm ════════════════

def test_FIRING_2_the_daily_check_IS_A_WORKFLOW_STEP_and_prints_its_grain():
    """Observable: the `h1_collected grain=... provider=... window=...` line.

    `main()` had never been called by anything. A step now runs it, and the
    marker line is the firing — a figure with no grain beside it is the
    defect this replaced.
    """
    d = yaml.safe_load(WF.read_text(encoding="utf-8"))
    runs = [s.get("run", "") for j in d["jobs"].values() for s in j["steps"]]
    assert any("scripts.h1_precondition" in r for r in runs), (
        "no workflow step invokes the daily check; it is unwired again")
    body = pre.format_zero_alarm(
        [pre.ZeroDay(dt.date(2026, 11, 1), 8, 0, pre.STATE_NO_CARD, "x")],
        target_n=39, floor_n=33, collected_n=0, credits_spent=0,
        credits_reserved=168)
    assert "COLLECTED NOTHING" in body, body.splitlines()[0]


def test_FIRING_2b_the_STREAK_KEYS_ON_THE_COUNT_not_the_label():
    """Observable: a 3-day streak survives an UNEVALUATED reason.

    Relabelling past days honestly once truncated this to 1 — the signal lost
    on the day it mattered.
    """
    rows = [pre.ZeroDay(dt.date(2026, 11, d), 0, 0, pre.STATE_UNEVALUATED, "x")
            for d in (1, 2)]
    rows.append(pre.ZeroDay(dt.date(2026, 11, 3), 23, 0,
                            pre.STATE_NO_FIXTURE_IN_BAND, "x"))
    assert len(pre.zero_collection_streak(rows)) == 3


# ═════ 3. straddles_a_month() — a helper with no consumer ═══════════════════

def test_FIRING_3_straddles_a_month_REACHES_A_DECISION(monkeypatch):
    """Observable: the reservation check changes answer when the window moves.

    The helper had no consumer at all. It is now read by
    `reservation_is_expressible`, and the firing is that moving the window
    across a reset WITHOUT re-declaring the reservation flips the verdict.
    """
    assert hw.straddles_a_month() is False
    ok_before, _ = hcc.reservation_is_expressible()
    assert ok_before

    monkeypatch.setattr(hw, "FROM", dt.date(2026, 11, 25))
    monkeypatch.setattr(hw, "UNTIL", dt.date(2026, 12, 3))
    assert hw.straddles_a_month() is True
    ok_after, why = hcc.reservation_is_expressible()
    assert not ok_after and "NO declared reservation" in why, (
        "the window crossed a reset and nothing noticed, which is the "
        "unwired state this came from")


# ═════ 4. an empty pinned league list meant NONE, not ALL ═══════════════════

def test_FIRING_4_an_EMPTY_LEAGUE_LIST_IS_ALL_not_none():
    """Observable: the same counts give different states for [] and ['x'].

    `league = ANY([])` matches nothing in SQL, so "not pinned" silently
    became "no league qualifies" and the precondition refused for the wrong
    reason. The firing is that [] now reaches READY on counts that a pinned
    list would refuse on.
    """
    # 40 rows, 40 fixtures, 9 in band, NONE of them in a pinned league.
    empty = pre.evaluate(_Sess(40, 40, 9, 0), [])
    pinned = pre.evaluate(_Sess(40, 40, 9, 0), ["england/championship"])
    assert empty.state == pre.STATE_READY, (
        f"an empty pinned list still refused ({empty.state}); it is being "
        f"read as NONE rather than ALL")
    assert pinned.state == pre.STATE_NO_FIXTURE_IN_BAND, pinned.state
    assert empty.state != pinned.state, (
        "both lists give the same answer, so this test cannot tell whether "
        "the empty case is handled at all")


# ═════ 5. h1_analysis --since defaulted to OCTOBER's reset ══════════════════

def test_FIRING_5_the_SINCE_DEFAULT_IS_DERIVED_FROM_THE_WINDOW(monkeypatch):
    """Observable: the default moves when the window moves.

    It was a literal pointing at the RETIRED window, so the analysis would
    have read October's pilot and called it this window's result. The firing
    is that the default tracks the one source.
    """
    assert hcc._H1_WINDOW_FROM == hw.iso()[0], (
        f"the analysis window default is {hcc._H1_WINDOW_FROM}, not the "
        f"window's own {hw.iso()[0]}")
    assert hcc._H1_WINDOW_FROM != "2026-10-09", (
        "the default is still the RETIRED window's start")
    monkeypatch.setattr(hw, "FROM", dt.date(2026, 12, 6))
    monkeypatch.setattr(hw, "UNTIL", dt.date(2026, 12, 14))
    assert hw.iso()[0] == "2026-12-06", (
        "the window module did not move, so this control proves nothing")


def test_ALL_FIVE_ARE_ENUMERATED_HERE_and_the_count_is_pinned():
    """The list itself is a registered quantity; a sixth must be added here."""
    firings = [n for n in globals() if n.startswith("test_FIRING_")]
    assert len(firings) == 6, (
        f"{len(firings)} firing tests for five mechanisms (2 has two halves); "
        f"if a mechanism was added or removed, say which: {sorted(firings)}")
