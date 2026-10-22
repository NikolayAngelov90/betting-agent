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
at 0.5-5.7h and observed once at 11h21m — both figures belong to the RETIRED
`37 9 * * *` regime (n=39), not to the cron in force; `0 3 * * *` measured
4h11m-6h35m over n=32 and `0 0 * * *` has n=1. Under the OLD cron the same question
answered **133 of 223**. It is also on H1's critical path — the collection widens
the refresh window to 360 minutes, which widens the blast radius of an overlap.
"""

from datetime import datetime, timedelta

import pathlib

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

#: EVERY REAL `closing-lines` SLOT, `measured` 2026-10-22 from its own crons
#: (`47 10` and `17 11,13,15,17,19,21,23`). Enumerated rather than sampled,
#: because the question "which slots can this guard decline" has an answer and
#: the previous version of this test asked it about 01:17 and 02:17 -- hours at
#: which NO CAPTURE RUNS AT ALL.
CAPTURE_SLOTS = [(10, 47), (11, 17), (13, 17), (15, 17),
                 (17, 17), (19, 17), (21, 17), (23, 17)]

#: The slots INSIDE the window under the new crons (weekend `30 0`, weekday
#: `0 1` -> window [00:30, 13:30]), and those outside it.
SLOTS_EXPOSED = [(10, 47), (11, 17), (13, 17)]
SLOTS_PROTECTED = [(15, 17), (17, 17), (19, 17), (21, 17), (23, 17)]


def test_the_SLOT_ENUMERATION_matches_the_closing_lines_crons():
    """The enumeration is checked against the workflow, so it cannot drift
    into describing a schedule that no longer exists -- which is exactly what
    happened to this file's `[21, 23, 1, 2]` parametrisation."""
    import re
    wf = pathlib.Path(".github/workflows/closing-lines.yml").read_text(
        encoding="utf-8")
    found = sorted(
        (int(h), int(mins))
        for mins, hours in re.findall(r"- cron: '(\d+) ([0-9,]+) \* \* \*'", wf)
        for h in hours.split(","))
    assert found == sorted(CAPTURE_SLOTS), found
    assert sorted(SLOTS_EXPOSED + SLOTS_PROTECTED) == sorted(CAPTURE_SLOTS)


@pytest.mark.parametrize("hour,minute", SLOTS_PROTECTED)
def test_CONTROL_a_slot_OUTSIDE_the_window_NEVER_declines(tmp_path, hour, minute):
    """THE 2026-09 INCIDENT, REPRODUCED AND REFUSED.

    The reverted guard asked "have today's picks run" through a
    `date.today()` marker, which stayed false all evening and all night, so it
    declined EVERY overnight slot and took captures to zero for a full day.
    The database here is seeded in exactly that state -- a league holding an
    unpicked fixture and no pick written for the day -- so the marker
    condition the incident turned on is TRUE at each of these times.

    The time bound is checked FIRST, so no database state can reproduce it.

    RE-POINTED 2026-10-22: the slots were `[21, 23, 1, 2]` at :17, and 01:17
    and 02:17 are hours at which no capture runs. Under the new crons the
    window is [00:30, 13:30], so the protected set is the five slots from
    15:17 -- one MORE than under the stale [03:00, 16:00] window, which left
    15:17 exposed.
    """
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",))
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=hour, minute=minute))
    assert got == {}, (
        f"declined at {hour:02d}:{minute:02d} — that slot is OUTSIDE the "
        f"pending-run window, so no picks run can be waiting on it. This is "
        f"the failure that stopped captures for a full day in 2026-09, and "
        f"the DB is seeded in the state that caused it.")


@pytest.mark.parametrize("hour,minute", SLOTS_EXPOSED)
def test_CONTROL_a_slot_INSIDE_the_window_DOES_decline(tmp_path, hour, minute):
    """The other half. A guard that never declines is not a guard, and these
    three slots are the ones a pending run can still be racing."""
    mgr = _mgr(tmp_path)
    _seed(mgr, leagues_with_unpicked=("england/premier-league",))
    got = _scraper(mgr)._leagues_a_pending_run_could_price(
        ["england/premier-league"], DAY.replace(hour=hour, minute=minute))
    assert set(got) == {"england/premier-league"}, (
        f"{hour:02d}:{minute:02d} is inside the window with a pending run and "
        f"an unpicked fixture, and the guard stood down")


def test_the_MOVE_PROTECTS_MORE_SLOTS_THAN_IT_EXPOSES():
    """The cost of the cron move, measured rather than feared.

    `measured` 2026-10-22: 4 of the 8 slots could decline under the stale
    [03:00, 16:00] window and 3 can under [00:30, 13:30]. The early edge
    gains nothing because the FIRST capture slot is 10:47 -- there is no slot
    between 00:30 and 03:00 to expose -- and the late edge loses 15:17.
    """
    import src.scrapers.theodds_scraper as ts
    width = ts._PICKS_EXPOSURE_WINDOW_MINUTES
    old_start, new_start = 3 * 60, ts._PICKS_CRON_MINUTE_OF_DAY

    def inside(start, hm):
        m = hm[0] * 60 + hm[1]
        return start <= m <= start + width

    old = [s for s in CAPTURE_SLOTS if inside(old_start, s)]
    new = [s for s in CAPTURE_SLOTS if inside(new_start, s)]
    assert len(old) == 4 and len(new) == 3, (old, new)
    newly_exposed = [s for s in new if s not in old]
    newly_protected = [s for s in old if s not in new]
    assert newly_exposed == [], (
        f"the move exposed {newly_exposed}; over-declining is the failure "
        f"that took captures to zero in 2026-09")
    assert newly_protected == [(15, 17)], newly_protected
    assert sorted(new) == sorted(SLOTS_EXPOSED)


def test_the_WINDOW_START_is_TIED_to_the_CRON_and_cannot_diverge():
    """`3 * 60` was stale by three hours: the cron had been `0 0` since
    2026-10-18 and nothing connected the two. A scraper must not parse
    workflow YAML at runtime, so the literal stays and THIS ties it."""
    import src.scrapers.theodds_scraper as ts
    from scripts.ci_audit import current_crons_by_daytype

    crons = [v for v in current_crons_by_daytype().values() if v is not None]
    assert crons, "no cron could be read from daily-picks.yml"
    assert ts._PICKS_CRON_MINUTE_OF_DAY == min(crons), (
        f"_PICKS_CRON_MINUTE_OF_DAY is {ts._PICKS_CRON_MINUTE_OF_DAY} but the "
        f"earliest daily-picks cron is {min(crons)}. Inside this window the "
        f"guard reserves credits for a pending picks run, so a window that "
        f"opens AFTER the cron leaves the run unprotected at the start of "
        f"its own schedule.")
    # and the width must still cover the latest a run can be pending:
    # cron + max observed delay + p90 duration
    assert ts._PICKS_EXPOSURE_WINDOW_MINUTES >= 342.3 + 106, (
        "the window no longer reaches the latest a picks run can be pending "
        "(max observed delay 342.3 + p90 duration 106)")


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
