"""The odds refresh must not race the picks run.

THE PATH THIS CLOSES. `refresh_and_capture` rewrites `odds` near kickoff; the
picks run reads `odds`. The crons order them — picks 03:00, first refresh 10:47
— so under normal delay they never interact. **That is a scheduling
coincidence.** The observed maximum scheduler delay is 11h21m and it has
happened twice (2026-08-27 19:58, 2026-08-28 20:58). Past ~7h40m the picks run
reads refreshed prices and the run is selection-affecting with no cohort bump
and no announcement.

WHY A WRITTEN MARKER RATHER THAN AN INFERENCE, pinned here because the obvious
implementation is the wrong one: "does a `saved_picks` row exist for today" is
satisfied by a quiet day and by a run that never happened alike. That ambiguity
is the defect this project has now found five times. **Completion is not
derivable from output; the run has to record it.**
"""

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.data.models import Base
from src.data.run_marker import (PICKS_MARKER, mark_picks_complete,
                                 picks_completed, refresh_may_run)

TODAY = date(2026, 9, 3)


class _DB:
    def __init__(self, engine):
        self.engine = engine
        self._Session = sessionmaker(bind=engine)

    def get_session(self):
        return self._Session()


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return _DB(engine)


class _DeadDB:
    def get_session(self):
        raise RuntimeError("connection lost")


# ── the marker ──────────────────────────────────────────────────────────────
def test_marker_is_absent_before_the_picks_run(db):
    assert picks_completed(db, TODAY) is False


def test_marker_is_present_after_the_picks_run(db):
    assert mark_picks_complete(db, TODAY) is True
    assert picks_completed(db, TODAY) is True


def test_the_marker_is_per_day(db):
    mark_picks_complete(db, TODAY)
    assert picks_completed(db, TODAY + timedelta(days=1)) is False, (
        "yesterday's completion must not satisfy today's guard — the race is "
        "daily and so is the marker")


def test_writing_twice_is_harmless(db):
    """A re-run of --picks must not fail on the marker."""
    assert mark_picks_complete(db, TODAY) is True
    assert mark_picks_complete(db, TODAY) is True
    assert picks_completed(db, TODAY) is True


# ── the three guard paths, and only one blocks ─────────────────────────────
def test_guard_DECLINES_before_the_picks_run(db):
    may, why = refresh_may_run(db, TODAY)
    assert may is False
    assert "not complete" in why


def test_guard_ALLOWS_after_the_picks_run(db):
    mark_picks_complete(db, TODAY)
    may, why = refresh_may_run(db, TODAY)
    assert may is True
    assert why == "picks complete"


def test_guard_PROCEEDS_when_it_cannot_answer():
    """Unanswerable is not the same as 'not complete'.

    Declining every capture on a database hiccup costs more than the exposure
    it would prevent, and that exposure is the pre-existing status quo rather
    than something this guard introduced. Absence of evidence is not evidence
    — the same distinction `_parse_match_date` and `coverage_checks` draw.
    """
    assert picks_completed(_DeadDB(), TODAY) is None
    may, why = refresh_may_run(_DeadDB(), TODAY)
    assert may is True
    assert why == "guard unavailable"


def test_none_is_not_false(db):
    """The distinction the whole design rests on, asserted explicitly."""
    assert picks_completed(db, TODAY) is False       # measured: not yet run
    assert picks_completed(_DeadDB(), TODAY) is None  # unmeasured
    assert picks_completed(db, TODAY) is not None


# ── the guard must announce itself on every path ───────────────────────────
@pytest.mark.parametrize("setup,expect", [
    ("absent", "DECLINING"),
    ("present", "picks for"),
    ("dead", "UNAVAILABLE"),
])
def test_the_guard_says_it_ran_on_every_path(db, caplog, setup, expect):
    """A guard that can silently do nothing must say it ran.

    `resolve_fixture_groups` logs only when it unions a group, so three clean
    days were indistinguishable from three dead calls. This guard logs on all
    three paths — including the quiet one where it allows the refresh.
    """
    import logging
    target = db
    if setup == "present":
        mark_picks_complete(db, TODAY)
    elif setup == "dead":
        target = _DeadDB()

    messages = []
    from src.data import run_marker
    for level in ("info", "warning"):
        setattr(run_marker.logger, level,
                (lambda m, _l=level: messages.append(str(m))))
    refresh_may_run(target, TODAY)

    assert any(expect in m for m in messages), (
        f"the {setup} path produced no line containing {expect!r}; a silent "
        f"path is indistinguishable from a guard that never ran. Got: {messages}")


# ── the wiring itself ──────────────────────────────────────────────────────
def test_the_refresh_actually_consults_the_guard():
    """Structural: the guard is worthless if `refresh_imminent` skips it."""
    import inspect
    from src.scrapers.theodds_scraper import TheOddsScraper

    src = inspect.getsource(TheOddsScraper.refresh_imminent)
    assert "refresh_may_run" in src, (
        "refresh_imminent no longer consults the picks-run guard — the race "
        "with pick-time prices is open again")
    assert src.index("refresh_may_run") < src.index("_imminent_league_fixtures"), (
        "the guard must run BEFORE candidate selection, or it declines after "
        "the work it exists to prevent has already been decided")


def test_the_picks_path_writes_the_marker():
    """Structural: without the write, the guard blocks every refresh forever."""
    import pathlib
    src = pathlib.Path("src/agent/betting_agent.py").read_text(encoding="utf-8")
    assert "mark_picks_complete" in src, (
        "the picks path no longer records completion — every subsequent odds "
        "refresh will decline and closing-line capture stops entirely")
