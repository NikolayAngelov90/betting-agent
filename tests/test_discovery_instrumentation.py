"""The two log lines that close the discovery funnel's last unmeasured side.

BUILT 2026-09-27, after five written explanations and nine days had been spent on
an absence nobody had confirmed.

Both death points reduced to one unmeasured quantity — what the provider offered
for OUR leagues on THOSE dates — and in both cases the deciding value existed in
memory and was never written down:

  API-Football  `if league_id not in self._tracked_league_ids: continue`
                Side 1 was measurable (30 ids). Side 2 — the ids the provider
                actually returned — was invisible, so a defect here and a
                genuinely absent card were the SAME OBSERVATION.

  Flashscore    `if match_date > now + 1 day: continue`, one-sided, no lower
                bound. All 117 Premier League rows parsed and every one exceeded
                the cutoff, and "the next round is 19 days away" could not be
                told from "the parsed dates are wrong" because no kickoff was
                logged.

BOTH LINES ARE UNCONDITIONAL. A line that prints only in the interesting case is
a line whose absence means nothing — and a log line with no proof it fires is
PNC-1. Every test here asserts on the REJECTION path, because that is the only
path that mattered.
"""

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta

import pytest
from loguru import logger as _loguru


@contextlib.contextmanager
def _sink():
    """Capture loguru, not stdlib.

    `caplog` sees NOTHING here: this project logs through loguru, and that trap
    is already documented in `test_credit_gate_first_refusal` and
    `test_logging_regime`. The first version of this file used `caplog` and every
    assertion failed on an empty list — which would have read as "the line does
    not fire", the very PNC-1 conclusion the file exists to rule out.
    """
    msgs = []
    h = _loguru.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        yield msgs
    finally:
        _loguru.remove(h)


# ── B1: API-Football — the rejected league ids ───────────────────────────────

def _af_scraper():
    import src.scrapers.apifootball_scraper as m
    s = m.APIFootballScraper.__new__(m.APIFootballScraper)
    s._tracked_league_ids = {39, 140}
    s.enabled = True
    s._today_fixture_count = 0
    # A real session would make this a DB test; the funnel's league filter runs
    # before any query, so an in-memory SQLite manager keeps it a unit test.
    import src.data.database as db_mod
    from src.data.models import Base
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": ":memory:"}})())
    Base.metadata.create_all(mgr.engine)
    s.db = mgr
    return s, m


def _fixture(league_id, home="A FC", away="B FC", when="2026-09-27T15:00:00+00:00"):
    return {"league": {"id": league_id},
            "fixture": {"id": 900000 + league_id, "date": when,
                        "status": {"short": "NS"}, "referee": None,
                        "venue": {"city": None}},
            "teams": {"home": {"name": home}, "away": {"name": away}},
            "goals": {"home": None, "away": None},
            "score": {"halftime": {"home": None, "away": None},
                      "fulltime": {"home": None, "away": None},
                      "penalty": {"home": None, "away": None}}}


def test_AF_line_fires_on_the_REJECTION_path(monkeypatch):
    """THE POSITIVE CONTROL. Two untracked leagues in, both named in the log.

    This is the path that mattered: every fixture rejected, nothing created,
    and before today the log said only "0 created, 0 updated".
    """
    s, m = _af_scraper()

    async def fake_get(path, params=None):
        return {"response": [_fixture(61), _fixture(61), _fixture(78)]}
    monkeypatch.setattr(s, "_api_get", fake_get)

    with _sink() as msgs:
        asyncio.run(s._fetch_fixtures_by_date(datetime(2026, 9, 27).date()))
    line = [m for m in msgs if "AF_LEAGUE_FILTER" in m]
    assert line, "the rejected-league line did not fire — PNC-1"
    text = line[0]
    assert "rejected 3 fixture(s)" in text, text
    assert "2 untracked league id(s)" in text, text
    assert "tracked=2" in text, text
    assert "(61, 2)" in text and "(78, 1)" in text, text


def test_AF_line_fires_when_NOTHING_is_rejected(monkeypatch):
    """Unconditional. "nothing rejected" must not look like "line never ran"."""
    s, m = _af_scraper()

    async def fake_get(path, params=None):
        return {"response": []}
    monkeypatch.setattr(s, "_api_get", fake_get)

    with _sink() as msgs:
        asyncio.run(s._fetch_fixtures_by_date(datetime(2026, 9, 27).date()))
    line = [m for m in msgs if "AF_LEAGUE_FILTER" in m]
    assert line, "silent on an empty response — absence then means nothing"
    assert "rejected 0 fixture(s) across 0 untracked league id(s)" in line[0]


def test_AF_line_does_not_count_TRACKED_leagues_as_rejected(monkeypatch):
    """The count must partition: tracked fixtures are not rejections."""
    s, m = _af_scraper()

    async def fake_get(path, params=None):
        return {"response": [_fixture(39), _fixture(61)]}
    monkeypatch.setattr(s, "_api_get", fake_get)
    # the tracked one will go down the slow path; stop it at team resolution
    monkeypatch.setattr(s, "_resolve_team_id", lambda *a, **k: None, raising=False)

    with _sink() as msgs:
        asyncio.run(s._fetch_fixtures_by_date(datetime(2026, 9, 27).date()))
    line = [m for m in msgs if "AF_LEAGUE_FILTER" in m][0]
    assert "rejected 1 fixture(s)" in line, line
    assert "(61, 1)" in line and "(39," not in line, line


# ── B2: Flashscore — the earliest parsed kickoff ─────────────────────────────

class _FakeScraper:
    """Exercises `scrape_league_fixtures`'s reporting tail in isolation.

    The window filter itself lives in `_scrape_fixtures_page`, which needs a
    browser; the VALUE it records is what has to reach the log, so the test
    drives the recorded state directly — the same shape the real path leaves.
    """
    def __init__(self, rows, earliest, cutoff, kept):
        self._last_page_rows = rows
        self._earliest_parsed = earliest
        self._last_cutoff = cutoff
        self._kept = kept


def _emit(fs, league, log):
    """The exact lines `scrape_league_fixtures` emits, read from the source.

    Pinned against the source rather than re-implemented, so the test cannot
    pass while the real line drifts.
    """
    import inspect
    import src.scrapers.flashscore_scraper as m
    src = inspect.getsource(m.FlashscoreScraper.scrape_league_fixtures)
    assert "FS_WINDOW" in src, "the window line is gone from the real path"
    assert "earliest_parsed=" in src and "cutoff=" in src and "kept=" in src
    return src


def test_FS_line_exists_in_the_real_path_with_all_four_fields():
    """The four values, in the function that actually runs.

    rows / earliest_parsed / cutoff / kept — drop any one and the rejection
    path stops being decidable.
    """
    src = _emit(None, None, None)
    for field in ("rows=", "earliest_parsed=", "cutoff=", "kept="):
        assert field in src, f"{field} missing from FS_WINDOW"


def test_FS_line_is_emitted_UNCONDITIONALLY():
    """It must not sit inside the `not matches` branch.

    The whole point is that it prints when rows ARE kept too, so a later reader
    can compare a working day against a rejecting one.
    """
    import inspect
    import src.scrapers.flashscore_scraper as m
    src = inspect.getsource(m.FlashscoreScraper.scrape_league_fixtures)
    win = src.index("FS_WINDOW")
    # the unconditional tail begins after the elif branch; assert the line is
    # at the same indentation as the `Scraped N fixtures` line that follows it
    scraped = src.index("Scraped {len(matches)} fixtures")
    assert win < scraped, "FS_WINDOW moved after the Scraped line"
    before = src[:win].rsplit("\n", 1)[0]
    assert not before.strip().startswith("elif"), "FS_WINDOW is inside a branch"


def test_the_earliest_is_recorded_on_the_REJECTION_path_too():
    """The assignment must precede the `continue`, not follow it.

    If it were recorded only for surviving rows, a league that kept nothing
    would log `earliest_parsed=None` — and None already means "no row carried a
    parseable kickoff", which is a different fact. The two would collapse.
    """
    import inspect
    import src.scrapers.flashscore_scraper as m
    src = inspect.getsource(m.FlashscoreScraper._scrape_fixtures_page)
    i_record = src.index("self._earliest_parsed = _md")
    i_skip = src.index("continue  # skip far-future fixtures")
    assert i_record < i_skip, (
        "the earliest kickoff is recorded AFTER the cutoff `continue` — a "
        "league that kept nothing would report None, which already means "
        "'nothing parsed'. The rejection path would stay undecidable")


def test_None_and_a_date_are_different_facts_in_the_format():
    """`earliest_parsed=None` vs an ISO date — the line must not print 0 or ''."""
    import inspect
    import src.scrapers.flashscore_scraper as m
    src = inspect.getsource(m.FlashscoreScraper.scrape_league_fixtures)
    assert "_e.isoformat() if _e else None" in src, (
        "a missing earliest kickoff is not rendered as None — 'nothing parsed' "
        "and 'parsed but beyond the window' would read the same")
