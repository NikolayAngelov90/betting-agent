"""The fixture-horizon probe persists NOTHING — proved by observation.

> A guard that enumerates a path and protects nothing has already appeared
> three times in this project, and `path=DEFAULT_PATH` binding at definition
> time is why.

So this file does not read the probe's source and conclude it is read-only. It
**counts the rows in every table before and after a probe run** and compares.
And it carries a POSITIVE CONTROL: the same observation, with a write injected,
must fail — otherwise the comparison is vacuous and would pass against a probe
that wrote freely.

It also pins the two things the probe must refuse to do:

* emit a deadline on an **unestablished clock** — Flashscore renders in the
  BROWSER's timezone, which was UTC+3 on the machine that ran the 2026-10-10
  density probe and is UTC on a GitHub runner. An assumed offset is how the H1
  cron was set three hours late;
* turn a refused or empty league into a deadline. `None` is not an empty list.
"""

import datetime as dt

import pytest
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import text

import src.data.database as db_mod
from src.data.models import Base, Match, Team

import scripts.fixture_horizon_probe as probe

NOW = dt.datetime(2026, 10, 10, 12, 0)
TOMORROW = dt.date(2026, 10, 11)


def _mgr(tmp_path, name="probe.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    assert not mgr.is_postgres
    return mgr


def _seed(mgr):
    """One fixture the probe and the DB can BOTH hold, so the offset is
    verifiable, plus one for tomorrow."""
    with mgr.get_session() as s:
        s.add(Team(id=1, name="H"))
        s.add(Team(id=2, name="A"))
        s.flush()
        s.add(Match(id=1, league="france/ligue-1",
                    match_date=dt.datetime(2026, 10, 10, 18, 45),
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.add(Match(id=2, league="france/ligue-1",
                    match_date=dt.datetime(2026, 10, 11, 16, 0),
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()
    return mgr


def _row_census(mgr) -> dict:
    """Rows in EVERY table. The observation the no-persistence claim rests on."""
    out = {}
    with mgr.get_session() as s:
        for name in sa_inspect(mgr.engine).get_table_names():
            out[name] = s.execute(
                text(f'SELECT count(*) FROM "{name}"')).scalar()
    return out


def _fake_pages(payload):
    """Stand in for Selenium: return parsed rows without a browser."""
    def _scrape(self, url, max_days_ahead=1):
        league = url.split("/football/", 1)[1].rsplit("/fixtures", 1)[0]
        return [{"match_date": d} for d in payload.get(league, [])]
    return _scrape


# ═══════════════════ PART B — persistence, DISPROVED BY OBSERVATION ═════════

def test_a_probe_run_writes_NO_ROW_TO_ANY_TABLE(tmp_path, monkeypatch):
    """Counted before and after, across every table the schema defines."""
    mgr = _seed(_mgr(tmp_path))
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)
    from src.scrapers.flashscore_scraper import FlashscoreScraper
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            dt.datetime(2026, 10, 10, 18, 45),
                            dt.datetime(2026, 10, 11, 16, 0)]}))

    before = _row_census(mgr)
    assert sum(before.values()) > 0, "the census observed nothing - it is vacuous"

    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    with mgr.get_session() as s:
        offset = probe.verify_offset(s, probed)
    probe.earliest_for(probed, TOMORROW, offset or 0)

    after = _row_census(mgr)
    assert after == before, (
        "the probe changed the database. Deltas: "
        + repr({k: (before[k], after[k]) for k in before if before[k] != after[k]}))


def test_the_POSITIVE_CONTROL_the_census_catches_an_injected_write(tmp_path, monkeypatch):
    """PROVE THE TEST ABOVE CAN FAIL.

    Without this, `after == before` would pass against a probe that wrote
    freely if the census were counting the wrong thing — the vacuous-filter
    shape (VAC-1) one level up.
    """
    mgr = _seed(_mgr(tmp_path))
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)
    from src.scrapers.flashscore_scraper import FlashscoreScraper

    real = _fake_pages({"france/ligue-1": [dt.datetime(2026, 10, 11, 16, 0)]})

    def _scrape_and_write(self, url, max_days_ahead=1):
        with mgr.get_session() as s:          # the injected write
            s.add(Match(id=999, league="france/ligue-1",
                        match_date=dt.datetime(2026, 10, 12, 16, 0),
                        is_fixture=True, home_team_id=1, away_team_id=2))
            s.commit()
        return real(self, url, max_days_ahead)

    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _scrape_and_write)
    before = _row_census(mgr)
    probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    after = _row_census(mgr)
    assert after != before, (
        "the census did NOT notice an injected write, so the no-persistence "
        "test above proves nothing")
    assert after["matches"] == before["matches"] + 1


def test_the_probe_opens_no_write_session_of_its_own():
    """The complementary source check — necessary, not sufficient.

    The observation above is the proof; this catches the shape early and names
    what to look for.
    """
    import inspect

    src = inspect.getsource(probe)
    for forbidden in ("session.add", "s.add(", ".commit()", "record_price",
                      "raise_used_to", "claim_requests"):
        assert forbidden not in src, (
            f"the probe contains {forbidden!r} — it must not write, and the "
            f"row census is the proof but this is the earlier warning")


# ═════════════════════ the clock is ESTABLISHED, never assumed ══════════════

def test_the_offset_is_MEASURED_against_the_database(tmp_path, monkeypatch):
    """A UTC+3 probe against a UTC table must read +180, not 0."""
    mgr = _seed(_mgr(tmp_path))
    from src.scrapers.flashscore_scraper import FlashscoreScraper
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            dt.datetime(2026, 10, 10, 21, 45)]}))   # 18:45 + 3h
    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    with mgr.get_session() as s:
        assert probe.verify_offset(s, probed) == 180, (
            "the probe did not detect a three-hour offset against a UTC table "
            "— this is the exact error that set the H1 cron three hours late")


def test_a_UTC_probe_reads_zero(tmp_path, monkeypatch):
    """The other direction, or the detector could be returning 180 always."""
    mgr = _seed(_mgr(tmp_path))
    from src.scrapers.flashscore_scraper import FlashscoreScraper
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            dt.datetime(2026, 10, 10, 18, 45)]}))
    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    with mgr.get_session() as s:
        assert probe.verify_offset(s, probed) == 0


def test_an_UNVERIFIABLE_clock_yields_None_not_zero(tmp_path, monkeypatch):
    """No shared fixture means no offset. `None` is not zero, and the caller
    must not emit a deadline on it."""
    mgr = _seed(_mgr(tmp_path))
    from src.scrapers.flashscore_scraper import FlashscoreScraper
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            dt.datetime(2027, 5, 5, 12, 0)]}))   # nothing matches
    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    with mgr.get_session() as s:
        assert probe.verify_offset(s, probed) is None


def test_a_REFUSED_league_is_None_not_an_empty_list(tmp_path, monkeypatch):
    """`[]` means "the page gave us nothing"; `None` means "we never got the
    page". Collapsing them is the third-state error this project has paid for
    repeatedly, and here it would turn a refusal into a late deadline."""
    mgr = _seed(_mgr(tmp_path))
    from src.scrapers.flashscore_scraper import FlashscoreScraper

    def _boom(self, url, max_days_ahead=1):
        raise RuntimeError("page refused")

    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page", _boom)
    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    assert probed["france/ligue-1"] is None
    assert probed["__meta__"]["refused"] == ["france/ligue-1"]
    assert probe.earliest_for(probed, TOMORROW, 0) is None, (
        "a refused league produced a deadline")


# ══════════════════════════ PART A — the window is the requirement ══════════

def test_the_window_is_TWO_days_and_it_buys_CORRECTNESS_not_load():
    """One day ahead of the run is about two days from the probe.

    AND IT REDUCES NO TRAFFIC, which the first version of this test claimed.
    `_click_load_more` clicks to its own 15-second budget regardless of
    `max_days_ahead`, which filters rows AFTER they are fetched — so a 2-day
    and an 8-day probe load the SAME 30 pages. The narrow window bounds what
    the probe CLAIMS TO KNOW; it does not bound the provider load.

    The real cost is a DOUBLING: daily-picks already fetches these 30 pages
    once a day and this adds a second pass. `MEASURED` 2026-10-10: 30/30 pages,
    0 refused, 155.1s wall for the 2-day window.
    """
    import inspect

    from src.scrapers.flashscore_scraper import FlashscoreScraper

    assert probe.PROBE_DAYS_AHEAD == 2
    loader = inspect.getsource(FlashscoreScraper._click_load_more)
    assert "max_days_ahead" not in loader, (
        "`_click_load_more` now consults the window, so a narrower window DOES "
        "reduce page loads — re-measure the traffic claim, which this test "
        "exists to keep honest")


def test_the_probe_asks_for_ALL_leagues_not_a_subset():
    """The deadline is a MINIMUM, so any league can set it.

    Probing a subset would substitute an assumption about which leagues can
    bind for a measurement of which do — and the measured binding leagues were
    `netherlands/eerste-divisie` and `poland/ekstraklasa` on Easter Monday,
    neither of which anyone would have nominated.
    """
    import inspect

    src = inspect.getsource(probe.main)
    assert "scraping.flashscore_leagues" in src
    assert "flashscore_skip_leagues" in src, (
        "the probe ignores the skip list, so it would fetch a league the "
        "pipeline deliberately does not")
    for nominated in ("premier-league", "laliga", "top5", "[:5]", "[:10]"):
        assert nominated not in src, (
            f"the probe narrows to {nominated!r} — the deadline is a minimum "
            f"and a subset cannot establish one")


# ════════════ PART C — the provider load, bounded and non-overlapping ═══════

#: `MEASURED` 2026-10-10 from CI logs, the 5 most recent scheduled daily-picks
#: runs = 150 league-page fetches, BEFORE the probe shipped.
#: And the probe's own MEASURED cost, 2026-10-10, 2-day window, 30 leagues.
PROBE_COST = {"pages_fetched": 30, "refused": 0, "wall_seconds": 155.1,
              "adds_page_loads_per_day": 30, "doubling_of": 30}

FLASHSCORE_BASELINE = {
    "fixtures_page_failed": 0,
    "retry_also_failed": 0,
    "league_timeout": 0,
    "leagues_scraped_per_run": 30,
    "runs": 5,
}

#: STATED BEFORE THE POST-SHIP MEASUREMENT. Any one of these disables the probe
#: and the deadline falls back to the 12:00 weekday constant.
STOP_CONDITION = {
    "terminal_refusals_single_run": 2,      # strictly greater than
    "terminal_refusals_7day_mean": 1.0,     # strictly greater than
    "leagues_scraped_floor": 28,            # below, on two consecutive runs
}


def _wf(name):
    import pathlib

    import yaml

    d = yaml.safe_load(pathlib.Path(
        f".github/workflows/{name}.yml").read_text(encoding="utf-8"))
    return d, (d[True] if True in d else d["on"]), list(d["jobs"].values())[0]


def test_the_probe_does_not_overlap_H1_or_daily_picks():
    """Clearances computed from the MEASURED lateness, not asserted.

    `0 9` was the probe's first choice and starts in the SAME window as H1's
    own `0 9 * * 2,3,4`. Caught before shipping, which is the only reason this
    test reads as arithmetic rather than as a bug report.
    """
    DELAY_MIN, DELAY_MAX = 3 * 60 + 58, 5 * 60 + 4
    _d, on, job = _wf("fixture-horizon-probe")
    probe_cron = int(on["schedule"][0]["cron"].split()[1]) * 60
    probe_start_lo = probe_cron + DELAY_MIN
    probe_end_hi = probe_cron + DELAY_MAX + job["timeout-minutes"]

    # H1: starts cron+delay, runs (rounds-1)*gap, ends there
    import scripts.h1_collect as h1c
    _d2, on2, _j2 = _wf("h1-collect")
    h1_cron = int(on2["schedule"][0]["cron"].split()[1]) * 60
    h1_end_hi = (h1_cron + DELAY_MAX
                 + (h1c.H1_ROUNDS - 1) * h1c.H1_GAP_MINUTES)
    assert probe_start_lo > h1_end_hi, (
        f"the probe can start at {probe_start_lo} while H1 runs until "
        f"{h1_end_hi} — they overlap")
    assert probe_start_lo - h1_end_hi >= 60, (
        f"only {probe_start_lo - h1_end_hi} min of clearance from H1")

    # daily-picks, both day types
    _d3, on3, _j3 = _wf("daily-picks")
    for expr in [c["cron"] for c in on3["schedule"]]:
        dp_cron = int(expr.split()[1]) * 60
        dp_end_hi = dp_cron + DELAY_MAX + 72        # 72 = measured max duration
        assert probe_start_lo > dp_end_hi, (
            f"the probe can start at {probe_start_lo} while daily-picks "
            f"({expr}) runs until {dp_end_hi}")

    # and it must finish before the NEXT daily-picks run it is warning about
    next_run_lo = 24 * 60 + min(
        int(c["cron"].split()[1]) * 60 for c in on3["schedule"]) + DELAY_MIN
    assert probe_end_hi < next_run_lo, (
        "the probe can still be running when the run it warns about starts, so "
        "the warning would arrive too late to act on")


def test_the_probe_has_its_own_group_and_an_explicit_timeout():
    d, _on, job = _wf("fixture-horizon-probe")
    assert d["concurrency"]["group"] == "fixture-horizon-probe"
    assert d["concurrency"]["cancel-in-progress"] is False
    assert isinstance(job["timeout-minutes"], int)
    assert job["timeout-minutes"] <= 30, (
        "a browser hung on a page must die, not drift — this job fetches 30 "
        "pages and the 8-day demonstration finished inside ten minutes")


def test_the_job_PINS_the_timezone_and_the_probe_still_verifies_it():
    """Pinning TZ is necessary and NOT sufficient.

    TZ=UTC makes the browser render UTC; the probe still cross-checks against
    `matches` and refuses on a non-zero offset. Both, because the 2026-10-10
    density probe was trusted on an assumed offset and the H1 cron went out
    three hours late.
    """
    _d, _on, job = _wf("fixture-horizon-probe")
    assert job.get("env", {}).get("TZ") == "UTC", (
        "the job no longer pins the browser's timezone")
    import inspect
    src = inspect.getsource(probe.main)
    assert "OFFSET_TOLERANCE_MIN" in src and "UNKNOWN" in src, (
        "the probe no longer refuses on an unverified clock, so pinning TZ "
        "became the only defence and a runner-image change would break it "
        "silently")


def test_the_stop_condition_is_stated_and_strictly_above_the_BASELINE():
    """Rule 3: a stop threshold is named before it is measured.

    The baseline is 0 terminal refusals over 150 league-page fetches, so every
    threshold must sit strictly above zero — a threshold AT the baseline fires
    on the first transient and the scraper's own one-shot retry exists to
    absorb those.
    """
    assert FLASHSCORE_BASELINE["retry_also_failed"] == 0
    assert FLASHSCORE_BASELINE["runs"] * FLASHSCORE_BASELINE[
        "leagues_scraped_per_run"] == 150
    assert STOP_CONDITION["terminal_refusals_single_run"] > \
        FLASHSCORE_BASELINE["retry_also_failed"]
    assert STOP_CONDITION["terminal_refusals_7day_mean"] > 0
    assert STOP_CONDITION["leagues_scraped_floor"] < \
        FLASHSCORE_BASELINE["leagues_scraped_per_run"], (
        "the league floor is not below the baseline, so it fires on a healthy "
        "run")
    # the fallback the stop condition falls back TO must be the stricter figure
    from scripts.schedule_deadline import DEADLINE_PERCENTILES
    assert DEADLINE_PERCENTILES["weekday"][0] == 12 * 60, (
        "the probe's stop condition falls back to the weekday constant, and "
        "that constant is no longer the 12:00 the cron was sized against")


def test_the_workflow_names_the_stop_condition_and_the_baseline():
    """Both live in the workflow header, not only in the ledger — the file a
    person opens when the probe misbehaves."""
    import pathlib

    txt = pathlib.Path(
        ".github/workflows/fixture-horizon-probe.yml").read_text(encoding="utf-8")
    assert "STOP CONDITION" in txt
    assert "BASELINE" in txt and "150 league-page fetches" in txt
    assert "12:00 weekday constant" in txt, (
        "the workflow does not say what the deadline falls back to when the "
        "probe is disabled")


def test_the_probe_DOUBLES_the_fixtures_page_traffic_and_says_so():
    """The cost stated as what it is, not as what the window suggests."""
    import pathlib

    assert PROBE_COST["adds_page_loads_per_day"] == PROBE_COST["doubling_of"]
    txt = pathlib.Path(
        ".github/workflows/fixture-horizon-probe.yml").read_text(encoding="utf-8")
    assert "DOUBLING" in txt, (
        "the workflow no longer states that this doubles the Flashscore "
        "fixtures-page load")
    assert "155.1s" in txt, "the measured wall time is not recorded"


# ═══════ PART D — B1's five cases, through the WIRED path this time ═════════

#: Same five as `tests/test_prospective_deadline.py`, but driven through
#: probe -> verify_offset -> earliest_for -> prospective_margin instead of
#: through `prospective_margin` alone. The pure function was verified first;
#: this confirms the wiring did not change the answers.
WIRED_CASES = [
    (dt.date(2026, 4, 3), dt.time(11, 30), "ALARM", "Good Friday 2026"),
    (dt.date(2026, 4, 6), dt.time(10, 15), "ALARM", "Easter Monday 2026"),
    (dt.date(2027, 4, 2), dt.time(11, 30), "ALARM", "Good Friday 2027"),
    (dt.date(2027, 4, 5), dt.time(10, 15), "ALARM", "Easter Monday 2027"),
    (dt.date(2026, 10, 13), dt.time(16, 45), "CLEAR", "ordinary weekday"),
    (dt.date(2026, 10, 11), dt.time(10, 0), "CLEAR", "ordinary weekend"),
]


@pytest.mark.parametrize("serves,kickoff,expected,label", WIRED_CASES)
def test_B1_through_the_wired_path(tmp_path, monkeypatch, serves, kickoff,
                                   expected, label):
    """The probe's own output feeds the margin, and the verdicts hold."""
    from scripts.schedule_deadline import day_type, prospective_margin

    mgr = _mgr(tmp_path, f"w{serves.isoformat()}.db")
    anchor = dt.datetime.combine(serves - dt.timedelta(days=1), dt.time(18, 45))
    with mgr.get_session() as s:
        s.add(Team(id=1, name="H"))
        s.add(Team(id=2, name="A"))
        s.flush()
        # one fixture BOTH sides hold, so the offset is verifiable as 0
        s.add(Match(id=1, league="france/ligue-1", match_date=anchor,
                    is_fixture=True, home_team_id=1, away_team_id=2))
        s.commit()

    from src.scrapers.flashscore_scraper import FlashscoreScraper
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            anchor,
                            dt.datetime.combine(serves, kickoff)]}))

    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    with mgr.get_session() as s:
        offset = probe.verify_offset(s, probed)
    assert offset == 0, f"the wired path could not establish its clock: {offset}"

    deadline = probe.earliest_for(probed, serves, offset)
    assert deadline == kickoff.hour * 60 + kickoff.minute, (
        f"{label}: the probe read {deadline} for a {kickoff} kickoff")

    cron = 0 if day_type(serves) == "weekend" else 4 * 60
    m = prospective_margin(deadline, cron, DELAYS_FOR_WIRED, DURATIONS_FOR_WIRED,
                           serves=serves)
    assert m["verdict"] == expected, f"{label}: {m}"


#: `MEASURED` 2026-10-10, regime `0 0 * * *`, scheduled only, n=9/n=10.
DELAYS_FOR_WIRED = [4 * 60 + 31, 4 * 60 + 45, 4 * 60 + 42, 4 * 60 + 31,
                    5 * 60 + 4, 4 * 60 + 16, 4 * 60 + 31, 3 * 60 + 58,
                    4 * 60 + 15]
DURATIONS_FOR_WIRED = [72.3, 53.3, 29.0, 41.8, 47.5, 37.0, 40.8, 45.4, 43.2,
                       42.3]


def test_the_wired_path_reports_UNKNOWN_on_a_REFUSED_probe(tmp_path, monkeypatch):
    """Rule 4 through the wiring: a refusal never becomes a value.

    This behaviour was verified on the pure function; here it is confirmed to
    survive the probe in front of it.
    """
    from scripts.schedule_deadline import prospective_margin

    mgr = _seed(_mgr(tmp_path, "refused.db"))
    from src.scrapers.flashscore_scraper import FlashscoreScraper

    def _boom(self, url, max_days_ahead=1):
        raise RuntimeError("provider refused")

    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page", _boom)
    probed = probe.probe_fixtures(["france/ligue-1"], days_ahead=2)
    deadline = probe.earliest_for(probed, TOMORROW, 0)
    m = prospective_margin(deadline, 4 * 60, DELAYS_FOR_WIRED,
                           DURATIONS_FOR_WIRED, serves=TOMORROW)
    assert m["verdict"] == "UNKNOWN"
    assert m["deadline_min"] is None


def test_an_UNVERIFIED_CLOCK_blocks_the_deadline_in_main(tmp_path, monkeypatch, capsys):
    """The guard that fired for real on 2026-10-10.

    Run on a Europe/Kyiv machine the probe measured +180 and REFUSED to emit a
    deadline, printing UNKNOWN - NOT safe. That is the whole defence against
    the error that set the H1 cron three hours late, so it is driven here.
    """
    mgr = _seed(_mgr(tmp_path, "tzguard.db"))
    monkeypatch.setattr(db_mod, "get_db", lambda: mgr)
    monkeypatch.setattr(probe, "_load_env", lambda: None)
    from src.scrapers.flashscore_scraper import FlashscoreScraper
    # +180: the same shape as the real Kyiv run
    monkeypatch.setattr(FlashscoreScraper, "_scrape_fixtures_page",
                        _fake_pages({"france/ligue-1": [
                            dt.datetime(2026, 10, 10, 21, 45)]}))
    monkeypatch.setattr(
        "src.utils.config.get_config",
        lambda: type("C", (), {"get": staticmethod(
            lambda k, d=None: ["france/ligue-1"] if "flashscore_leagues" in k
            else [])})())

    rc = probe.main(["--report-only", "--days-ahead", "2"])
    out = capsys.readouterr().out
    assert "+180 min from UTC" in out, out
    assert "UNKNOWN" in out and "NOT safe" in out, out
    assert "three hours late" in out, (
        "the refusal does not say WHY an unverified clock is refused")
    assert rc == 0, "--report-only must not exit non-zero"
