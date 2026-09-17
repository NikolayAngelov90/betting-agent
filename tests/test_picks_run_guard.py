"""The picks-run guard, REDESIGNED. Per-league, not a per-day marker.

    A refresh cannot contaminate a pick already TAKEN — `taken_odds` is
    persisted at pick time — so the exposure is to OTHER fixtures in the SAME
    league that a pending run might still price.

WHY THE FIRST VERSION WAS REVERTED (e4a104c, 2026-09-04). It asked "have today's
picks run", keyed on `date.today()`, and declined GLOBALLY. That question and
the real one diverge at every hour outside the picks window: at 23:17 and 01:00
the marker still said "not yet", so every overnight slot declined. **Captures
stopped for a full day and H5's sample rate fell from ~1.8 fixtures/day to
zero** — the sample for the only open question in the project.

    Measured cost: daily and certain.
    Exposure prevented: conditional on a picks delay beyond ~7h40m.

THREE CONDITIONS, ALL REQUIRED, and the first two are what make the overnight
case impossible:

    1. `now` inside the bounded exposure window after the 03:00 cron
    2. no pick carries today's `pick_date` — collapses the moment the run
       writes anything, so a normal day is guarded for minutes, not hours
    3. THIS league still holds a future fixture with no pick

WHY IT IS BACK. Measured 2026-09-17: 0 of 18 runs under the current cron overlap
the first refresh, by a margin of **41 minutes**, against a scheduler documented
at 0.5-5.7h and observed once at 11h21m. Under the OLD cron the same question
answered **133 of 223**. It is also on H1's critical path — the collection widens
the refresh window to 360 minutes, which widens the blast radius of an overlap.
"""

from datetime import datetime, timedelta

import pytest

import src.data.database as db_mod
import src.scrapers.theodds_scraper as ts
from src.data.models import Base, Match, SavedPick, Team


def _mgr(tmp_path):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / "g.db")}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


def _scraper(mgr):
    s = ts.TheOddsScraper.__new__(ts.TheOddsScraper)
    s.db = mgr
    return s


DAY = datetime(2026, 9, 17)


def _seed(mgr, *, leagues_with_unpicked=(), leagues_all_picked=(), picks_today=0):
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
        s.commit()
        mid = 100
        for lg in leagues_with_unpicked:
            s.add(Match(id=mid, league=lg, home_team_id=1, away_team_id=2,
                        match_date=DAY + timedelta(hours=20)))
            mid += 1
        for lg in leagues_all_picked:
            s.add(Match(id=mid, league=lg, home_team_id=1, away_team_id=2,
                        match_date=DAY + timedelta(hours=20)))
            s.flush()
            s.add(SavedPick(match_id=mid,
                            pick_date=(DAY - timedelta(days=3)).date()))
            mid += 1
        for i in range(picks_today):
            s.add(Match(id=900 + i, league="other/x", home_team_id=1, away_team_id=2,
                        match_date=DAY + timedelta(hours=20)))
            s.flush()
            s.add(SavedPick(match_id=900 + i, pick_date=DAY.date()))
        s.commit()


# ── THE REGRESSION THAT CAUSED THE REVERT ─────────────────────────────────

@pytest.mark.parametrize("hour", [21, 23, 1, 2])
def test_the_overnight_slots_NEVER_decline(tmp_path, hour):
    """23:17 and 01:00 declined every night and stopped captures for a day.

    The conditions the database reports are IRRELEVANT here — the time bound
    is checked first, precisely so no database state can reproduce this.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",))
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=hour, minute=17))
    assert got == {}, (
        f"declined at {hour:02d}:17 — no picks run can be pending outside the "
        f"window after the 03:00 cron, and this is the failure that stopped "
        f"captures for a full day")


def test_inside_the_window_with_a_pending_run_it_DOES_decline(tmp_path):
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",))
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=10, minute=47))
    assert set(got) == {"england/premier-league"}
    assert "unpicked" in got["england/premier-league"]


# ── CONDITION 2: the run producing anything stands the guard down ─────────

def test_once_the_run_has_written_a_pick_nothing_declines(tmp_path):
    """A normal day is guarded for minutes, not hours.

    This is the condition that makes the guard cheap: picks land ~08:15, and
    from that moment every remaining slot proceeds.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",), picks_today=1)
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=10, minute=47))
    assert got == {}, (
        "declined after today's run had already written picks — the exposure "
        "is to a PENDING run, and this one has produced output")


# ── CONDITION 3: PER-LEAGUE, which is the whole redesign ──────────────────

def test_a_league_whose_fixtures_are_all_PICKED_is_not_declined(tmp_path):
    """The reverted guard declined globally. This one must not.

    A refresh cannot contaminate a pick already taken — `taken_odds` is
    persisted at pick time — so a league with nothing left to price is not
    exposed, and declining it costs a capture for no reason.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",),
          leagues_all_picked=("spain/laliga",))
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league", "spain/laliga"],
        DAY.replace(hour=10, minute=47))
    assert set(got) == {"england/premier-league"}, (
        f"expected only the league with unpicked fixtures to decline, got "
        f"{sorted(got)} — this is the per-league half of the redesign")


def test_a_past_kickoff_fixture_does_not_hold_a_league_hostage(tmp_path):
    """A run cannot price a fixture that has already started."""
    mgr = _mgr(tmp_path)
    with mgr.get_session() as s:
        s.add_all([Team(id=1, name="A"), Team(id=2, name="B")])
        s.add(Match(id=1, league="england/premier-league", home_team_id=1,
                    away_team_id=2, match_date=DAY.replace(hour=6)))
        s.commit()
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=10, minute=47))
    assert got == {}


# ── FAILS OPEN, which is the trade the revert measured ────────────────────

def test_a_guard_that_cannot_answer_PROCEEDS(tmp_path):
    """The exposure is conditional; blocking every capture on a hiccup is not.

    Reverting taught this exactly: measured cost daily and certain, exposure
    prevented conditional on a delay past ~7h40m.
    """
    class _Broken:
        def get_session(self):
            raise RuntimeError("database unavailable")

    s = ts.TheOddsScraper.__new__(ts.TheOddsScraper)
    s.db = _Broken()
    assert s._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=10, minute=47)) == {}


def test_no_leagues_asked_means_nothing_to_decide(tmp_path):
    mgr = _mgr(tmp_path)
    assert _scraper(mgr)._leagues_a_pending_run_could_price(
        [], DAY.replace(hour=10, minute=47)) == {}


# ── THE RULE FROM THE REVERT: guard and audit pattern ship together ───────

def test_the_guard_and_its_audit_pattern_ship_together():
    """A guard whose decline the audit cannot see is invisible by construction.

    `ci_audit` kept `PICKS-RUN GUARD: DECLINING` through the revert precisely so
    the pair could not drift apart. This asserts the emitted line still matches
    the pattern that reads it — the producer/parser pin, applied to a guard.
    """
    import re
    import pathlib
    import scripts.ci_audit as ci

    pattern = ci.PATTERNS["picks_run_guard_declined"]
    src = pathlib.Path("src/scrapers/theodds_scraper.py").read_text(encoding="utf-8")
    assert 'PICKS-RUN GUARD: DECLINING' in src, (
        "the guard no longer emits the line ci_audit greps for — its silence "
        "would be indistinguishable from health")
    emitted = "PICKS-RUN GUARD: DECLINING england/premier-league — a picks run is pending"
    assert re.search(pattern, emitted), (
        f"ci_audit's pattern {pattern!r} does not match the line the guard "
        f"emits; the two must ship together or the guard is unobservable")
    assert ci.extract(emitted).get("picks_run_guard_declined") == 1
