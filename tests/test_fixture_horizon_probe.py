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
import pathlib

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

    NOT APPLICABLE SINCE 2026-10-18, AND THAT IS A STATE, NOT A PASS. The
    probe has no `schedule:` any more, so there is no start time to clear --
    and a check whose subject is gone must SAY so. Left as written, the loop
    over an empty cron list would have gone green while asserting nothing:
    exactly the VAC-1 shape this project names, and the reason the empty-
    population refusal was mistaken for a working verifier for six days.

    So the arithmetic is kept ALIVE against the last live cron as a FIXTURE --
    it still computes and still clears -- and the LIVE reading is restored the
    instant a `schedule:` comes back, with no edit to this test.
    """
    DELAY_MIN, DELAY_MAX = 3 * 60 + 58, 5 * 60 + 4
    _d, on, job = _wf("fixture-horizon-probe")

    #: The cron the probe ran under 2026-10-12..10-18, `measured` from its own
    #: run starts. A FIXTURE here, not live state: the live hours are read
    #: below whenever any exist.
    LAST_LIVE_PROBE_HOUR = 18

    crons = [c["cron"] for c in (on.get("schedule") or [])]
    state = ("CLEARANCE_MEASURED" if crons
             else "CLEARANCE_NOT_APPLICABLE_NO_SCHEDULE")
    probe_hours = ([int(c.split()[1]) for c in crons] if crons
                   else [LAST_LIVE_PROBE_HOUR])
    assert probe_hours, "neither a live cron nor the fixture -- nothing to check"

    import scripts.h1_collect as h1c
    _d2, on2, _j2 = _wf("h1-collect")
    _d3, on3, _j3 = _wf("daily-picks")

    for probe_hour in probe_hours:
        probe_cron = probe_hour * 60
        probe_start_lo = probe_cron + DELAY_MIN
        probe_end_hi = probe_cron + DELAY_MAX + job["timeout-minutes"]
        assert probe_end_hi > probe_start_lo, "the window is not a window"

        # H1: starts cron+delay, runs (rounds-1)*gap, ends there
        # ── H1'S SCHEDULE WAS DISABLED 2026-10-20. ────────────────────
        #
        # Its window closed 2026-10-17 and a scheduled run fired three days
        # later and spent 8 TheOddsAPI credits from the NORMAL pool. The cron
        # is gone, so there is no H1 start time to clear -- and a clearance
        # check over a missing cron would pass while asserting nothing, which
        # is the VAC-1 shape this file already handles for the probe's own
        # cron. Same treatment: the last live value as a FIXTURE, with the
        # state named.
        _h1_sched = (on2.get("schedule") or [])
        h1_state = ("H1_CRON_LIVE" if _h1_sched
                    else "H1_CLEARANCE_NOT_APPLICABLE_NO_SCHEDULE")
        #: `measured`: H1 ran under `0 9 * * 2,3,4` until 2026-10-20.
        h1_cron = (int(_h1_sched[0]["cron"].split()[1]) * 60 if _h1_sched
                   else 9 * 60)
        assert h1_state in ("H1_CRON_LIVE",
                            "H1_CLEARANCE_NOT_APPLICABLE_NO_SCHEDULE")
        h1_end_hi = (h1_cron + DELAY_MAX
                     + (h1c.H1_ROUNDS - 1) * h1c.H1_GAP_MINUTES)
        assert probe_start_lo > h1_end_hi, (
            f"[{state}/{h1_state}] the probe can start at {probe_start_lo} "
            f"while H1 runs until {h1_end_hi} -- they overlap")
        assert probe_start_lo - h1_end_hi >= 60, (
            f"[{state}] only {probe_start_lo - h1_end_hi} min of clearance "
            f"from H1")

        # daily-picks, both day types
        for expr in [c["cron"] for c in on3["schedule"]]:
            dp_cron = int(expr.split()[1]) * 60
            dp_end_hi = dp_cron + DELAY_MAX + 72    # 72 = measured max duration
            assert probe_start_lo > dp_end_hi, (
                f"[{state}] the probe can start at {probe_start_lo} while "
                f"daily-picks ({expr}) runs until {dp_end_hi}")


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


# ───────────────── WHY THE VERIFIER HAS NEVER HAD A NON-EMPTY INPUT
#
# MEASURED 2026-10-13, 2 of 2 successful probe runs:
#
#     timezone : UNVERIFIABLE - no fixture is held by both the probe and the
#                database
#
# AND THE VERIFIER IS NOT VACUOUS. The tests above drive it to +180, to 0 and
# to None, so it can see all three. What is empty is its INPUT, in production,
# and the cause is not the one that looks obvious.
#
# It is not the DB horizon alone. `max_days_ahead` is a ROLLING window from the
# run, so a probe firing at 00:21 with `days_ahead=2` covers 10-13 00:21 ->
# 10-15 00:21 - which includes "today" in full. The cause is ORDERING:
#
#     the probe runs BEFORE the producer of its verification data
#
# `daily-picks` is what writes `matches`. The probe's cron is `0 18` and
# `daily-picks`' weekday cron is `0 4`, so for the day D+1 that the probe
# targets, the scrape that would populate D+1 runs on D+1 - hours AFTER the
# probe has already looked. MEASURED the same day: `matches` held 23 rows for
# 10-12 and ZERO for 10-13, 10-14 and 10-15, which is the probe's whole window.
#
#     The verifier's input is drawn from the population it validates, so it
#     cannot run when that population is empty - which is exactly the state
#     its consumer needs an answer for.
#
# THE PROPOSED FIX DOES NOT WORK, and this is why it is deferred rather than
# applied: "verify against today's rows" assumes the probe's pages render
# today. They do not - a `/fixtures/` page shows UPCOMING fixtures, and the
# rows the database holds for today are already past. There is no overlap to
# reach. A fix has to change WHERE the clock comes from, not which rows are
# compared, and that is a different change with a different risk.

def test_the_PROBE_RUNS_BEFORE_the_producer_of_its_verification_data():
    """The structural cause, derived rather than asserted -- and since
    2026-10-18 there are TWO causes, of which the cron was the WEAKER.

    THE KNOWN-WRONG CASE, restated. Originally: if the cron ordering ever
    inverts, the UNVERIFIABLE refusal stops being structural, so re-measure
    before trusting it. That reading stood until the lead time was measured:

        MEASURED 2026-10-18, n=1,131 fixture rows written since 2026-09-01
          max write -> kickoff lead     23.62h
          rows ever known >24h ahead    ZERO
          `matches` rows for 2026-10-19  0   (that day's scrape ended 08:57)

    The probe needs TOMORROW's earliest kickoff, 24-28h out from any run on
    day D. `matches` has never held a fixture that far ahead, because
    `max_days_ahead` is a rolling 24h from the run and D+1's early card is
    written by D+1's OWN scrape. So the population is missing for a reason no
    clock can fix: a `workflow_run` trigger would have ordered the two jobs
    correctly, cost the same 30 page-loads, and refused identically.

    THE TWO CAUSES CLEAR DIFFERENTLY, which is why both stay readable here:

      CRON ORDERING      clears by moving a schedule. NOT APPLICABLE now --
                         there is no schedule. Held below against the last
                         live cron as a fixture, so the derivation cannot rot
                         while it is dormant.
      POPULATION CEILING clears only by changing what `--update` writes
                         (`max_days_ahead`'s horizon) or by taking the clock
                         from the runner -- `verify_offset`'s deferred repair.
                         Asserted in
                         `test_the_probe_still_cannot_emit_a_deadline_FOR_A_STATED_REASON`.
    """
    import re as _re

    probe_wf = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    picks_wf = pathlib.Path(".github/workflows/daily-picks.yml").read_text(
        encoding="utf-8")

    def _mins(text, pick):
        out = []
        for expr in _re.findall(r"cron: '([^']+)'", text):
            parts = expr.split()
            if parts[0].isdigit() and parts[1].isdigit():
                out.append((int(parts[1]) * 60 + int(parts[0]), parts[4]))
        return [m for m, dow in out if pick(dow)]

    weekday_min = _mins(picks_wf, lambda d: d == "1-5")
    assert weekday_min, (
        "the weekday scrape lost its cron, so the ordering is unmeasurable "
        "rather than fine -- the third state, again")

    probe_min = _mins(probe_wf, lambda d: True)
    #: The probe's cron 2026-10-12..10-18. A FIXTURE once the schedule is off.
    LAST_LIVE_PROBE_MIN = 18 * 60
    cause = ("CRON_ORDERING_LIVE" if probe_min
             else "CRON_ORDERING_NOT_APPLICABLE_NO_SCHEDULE")

    # The probe targets D+1. `daily-picks` populates D+1 by running ON D+1.
    # So the probe's absolute moment is (D, probe_min) and the scrape's is
    # (D+1, weekday_min) - the probe is always earlier, by construction.
    probe_at = (probe_min or [LAST_LIVE_PROBE_MIN])[0]
    scrape_at = 24 * 60 + weekday_min[0]    # minutes into day D, i.e. D+1
    assert probe_at < scrape_at, (
        f"[{cause}] the probe ({probe_at}) no longer precedes the scrape that "
        f"populates its target day ({scrape_at}) - the UNVERIFIABLE refusal "
        f"may no longer be structural, so re-measure before trusting it")
    gap_h = (scrape_at - probe_at) / 60.0
    # ── THE GAP NARROWED, AND THIS CONTROL CAUGHT IT. 2026-10-18. ──────────
    #
    # Built when the weekday scrape was `0 4`, giving a 10-hour nominal gap and
    # a threshold of 6. The revert to `0 0` made the nominal gap EXACTLY 6, and
    # the control fired on the boundary - which is what it was for.
    #
    # AND THE NOMINAL GAP IS NOT THE REAL ONE. The probe's own observed starts
    # (cron `0 18`) include 20:20, 21:06, 21:25 and **00:21** - it already
    # crosses midnight. With the weekday scrape now starting 00:00 + up to
    # 5h04m, THE ORDERING CAN ALREADY INVERT on a late probe, so the overlap
    # becomes sometimes-empty rather than always-empty, and the UNVERIFIABLE
    # refusal becomes INTERMITTENT - which is worse, because an intermittent
    # refusal reads as a transient.
    #
    # The fixture reproduces that exact boundary (1080 vs 1440 = 6.0h), so the
    # dormant derivation is held at the value that fired, not at a safe one.
    assert gap_h >= 6, (
        f"[{cause}] the nominal gap is {gap_h:.1f}h, below the 6h floor. The "
        f"probe no longer reliably precedes the scrape that writes its "
        f"verification data, so the refusal becomes intermittent. Move the "
        f"probe's cron earlier or take the clock from the runner.")

    # AND WHEN THE CRON CAUSE IS DORMANT, THE OTHER ONE MUST STILL BE HELD.
    # Otherwise disabling the schedule would have quietly retired the only
    # assertion about why the probe cannot work.
    if cause == "CRON_ORDERING_NOT_APPLICABLE_NO_SCHEDULE":
        own = pathlib.Path("tests/test_fixture_horizon_probe.py").read_text(
            encoding="utf-8")
        assert "def test_the_probe_still_cannot_emit_a_deadline" in own, (
            "the cron cause is dormant and the population cause is not "
            "asserted anywhere -- the probe's defect is now undocumented by "
            "any live check")


def test_the_verifier_is_NOT_VACUOUS_and_the_distinction_is_recorded():
    """It returns +180, 0 and None on driven inputs - the tests above do that.

    This pins the DISTINCTION, because "vacuous filter" was the first reading
    and it was wrong: a filter that cannot fire is a defect in the filter, and
    a correct filter handed an empty population is a defect in the DESIGN. The
    remedies are different, and only one of them is in this file.
    """
    src = pathlib.Path("tests/test_fixture_horizon_probe.py").read_text(
        encoding="utf-8")
    assert "verify_offset(s, probed) == 180" in src
    assert "verify_offset(s, probed) == 0" in src
    assert "verify_offset(s, probed) is None" in src, (
        "the three-outcome drive is gone; without it the claim that the "
        "verifier is not vacuous is unsupported")


# ───────── WHICH REFUSAL IS IT? THE PROVENANCE, FORCED. 2026-10-18.
#
#     An always-empty verifier is honest. An intermittently-populated one is
#     worse, because its silence and its answer become indistinguishable in
#     provenance - unless the output says which.
#
# `verify_offset` returns None for two different worlds and the log said
# `UNVERIFIABLE` for both. While the ordering is always inverted that
# conflation is harmless; it stops being harmless the moment the ordering can
# invert, and the margin to that has thinned.
#
# MEASURED 2026-10-18, all nine probe runs, separation against the TARGET
# DAY's scrape completion:
#
#     ordering INVERTED in 9 of 9      separation +7.33h .. +33.49h
#     runs below the 6.0h floor        ZERO - the intermittency has not started
#
# AND THE NOMINAL GAP WAS NEVER THE REAL ONE. With the probe's observed
# maximum lateness (18:00 -> 00:21) against the weekday scrape's observed
# 256-305 min, the worst case is +3.92h post-revert and was ALREADY +4.37h
# pre-revert. The nominal 10h -> 6h change moved the real worst case by 27
# minutes.

def test_the_ORDERING_is_MEASURED_from_the_rows_not_the_cron(tmp_path):
    """Force both worlds and confirm each is named."""
    import datetime as _d

    mgr = _mgr(tmp_path)
    day = _d.date(2026, 10, 20)

    # (1) no rows for the day -> the scrape has not run
    with mgr.get_session() as s:
        state, detail = probe.observed_ordering(s, day)
    assert state == probe.ORDERING_INVERTED, (state, detail)
    assert "has NOT run yet" in detail
    assert "this is ORDERING, not an empty result" in detail

    # (2) rows exist -> the population exists, so a miss is about the match
    mgr = _seed(mgr)
    with mgr.get_session() as s:
        from src.data.models import Match
        md = s.query(Match).first()
        day2 = md.match_date.date()
        state2, detail2 = probe.observed_ordering(s, day2)
    assert state2 == probe.ORDERING_POPULATION_EXISTS, (state2, detail2)
    assert "The population EXISTS" in detail2
    assert "not the ordering" in detail2
    assert detail != detail2, "both worlds print the same sentence"


def test_an_UNREADABLE_database_is_ORDERING_UNKNOWN_not_inverted():
    """Third state. "Could not measure the ordering" is not "the ordering is
    inverted" - the sixth time this file set has had to separate those."""
    class _Boom:
        def execute(self, *_a, **_k):
            raise RuntimeError("no connection")

    state, detail = probe.observed_ordering(_Boom(), dt.date(2026, 10, 20))
    assert state == probe.ORDERING_UNKNOWN
    assert "absence of evidence" in detail


def test_the_TWO_REFUSALS_print_DIFFERENTLY():
    """The whole point. A reader must never infer which world produced a
    `None`."""
    assert probe.REFUSAL_EMPTY_POPULATION != probe.REFUSAL_ORDERING_INVERTED
    assert "EMPTY_POPULATION" in probe.REFUSAL_EMPTY_POPULATION
    assert "ORDERING_INVERTED" in probe.REFUSAL_ORDERING_INVERTED
    src = pathlib.Path("scripts/fixture_horizon_probe.py").read_text(
        encoding="utf-8")
    # the selection is on the MEASURED ordering, not on a date or a cron
    assert "if ordering == ORDERING_INVERTED" in src
    assert "else REFUSAL_EMPTY_POPULATION" in src
    assert set(probe.ORDERING_STATES) == {
        probe.ORDERING_POPULATION_EXISTS, probe.ORDERING_INVERTED,
        probe.ORDERING_UNKNOWN}


def test_an_INVERTED_ordering_can_NEVER_silently_produce_a_deadline():
    """REACHED ON THE FIRST PRODUCTION RUN AFTER IT SHIPPED.

    I wrote that it could not be reached, reasoning that `offset is None`
    whenever the population is absent. Run 39131837165 (2026-10-18 13:31)
    disproved it within minutes: `verify_offset` returned **+0, verified** for
    the first time ever, because the morning's dispatch had populated TODAY's
    card and the rolling window overlaps it - while the deadline would have
    been computed for TOMORROW, whose population does not exist.

    THE TWO CONDITIONS ARE INDEPENDENT. The clock verifies against whatever
    day overlaps; the deadline is computed for the target day. Without this
    guard the probe would have emitted a deadline from its own pages with
    nothing to check them against, indistinguishable from a real one.
    """
    src = pathlib.Path("scripts/fixture_horizon_probe.py").read_text(
        encoding="utf-8")
    body = src.split("def main(", 1)[1]
    guard = body.index("if ordering != ORDERING_POPULATION_EXISTS")
    emit = body.index("deadline = earliest_for(")
    assert guard < emit, (
        "the ordering guard sits AFTER the deadline is computed, so an "
        "inverted ordering could produce one")
    assert "MUST NEVER SILENTLY PRODUCE A DEADLINE" in src


def test_an_EMITTED_deadline_carries_the_ORDERING_that_produced_it():
    """A deadline and a refusal must be distinguishable by PROVENANCE, not by
    reading the date and guessing."""
    src = pathlib.Path("scripts/fixture_horizon_probe.py").read_text(
        encoding="utf-8")
    assert 'ordering {ordering}' in src, (
        "the alarm body no longer states which ordering produced its deadline")
    body = probe.format_dispatch_alert(
        dt.date(2026, 10, 20), 11 * 60, "x/y",
        basis="deadline DISCOVERED by the probe; ordering POPULATION_EXISTS")
    assert "ordering POPULATION_EXISTS" in body


@pytest.mark.live
def test_the_SIX_HOUR_control_reports_the_MEASURED_gap_per_run():
    """The nominal floor stays in the gating suite; the MEASURED separation is
    reported here, because it needs `gh` and must not gate price capture.

    `measured` 2026-10-18 over 10 probe runs: 9 had a scrape run on their
    target day and ALL NINE are inverted (min +7.33h, max +33.49h), zero
    below the 6.0h floor. The 10th has NO COUNTERPART - today's dispatched
    13:30 probe targets 10-19, whose scrape has not run - and that is a
    separate column, not a tenth inversion. If `below_6h` becomes non-zero
    the intermittency has started, and THAT is the finding.
    """
    import json
    import subprocess

    def _runs(wf):
        o = subprocess.run(
            ["gh", "run", "list", "--workflow", wf, "--limit", "40", "--json",
             "createdAt,updatedAt,event"],
            capture_output=True, encoding="utf-8", errors="replace",
            timeout=180)
        try:
            return json.loads(o.stdout or "[]")
        except json.JSONDecodeError:
            return []

    def _ts(x):
        return dt.datetime.fromisoformat(
            x.replace("Z", "+00:00")).replace(tzinfo=None)

    probes = _runs("fixture-horizon-probe.yml")
    picks = _runs("daily-picks.yml")
    if not probes or not picks:
        pytest.skip("run history unreadable here")

    below, inverted, no_counterpart, seps = 0, 0, 0, []
    for p in probes:
        start = _ts(p["createdAt"])
        target = (start + dt.timedelta(days=1)).date()
        cand = [r for r in picks if _ts(r["createdAt"]).date() == target]
        if not cand:
            # ── THIRD STATE, AND IT WAS COUNTED AS AN INVERSION. 2026-10-18.
            #
            # This branch used to do `inverted += 1`, conflating "no scrape
            # run exists for the target day" with "the scrape finished after
            # the probe started". They are different facts: the first is NOT
            # YET KNOWABLE - today's 13:30 dispatched probe targets 10-19,
            # whose scrape has not run - and the second is a measurement.
            #
            # It was caught by its own report being impossible: `n=9
            # inverted=10`. The conclusion was unchanged (9 of 9 WITH a
            # counterpart are inverted) but the count that supported it was
            # inflated, in the direction that licensed the claim. A number
            # right for the wrong reason is an unaudited number.
            no_counterpart += 1
            continue
        sep = (_ts(cand[0]["updatedAt"]) - start).total_seconds() / 3600.0
        seps.append(sep)
        if sep > 0:
            inverted += 1
        if abs(sep) < 6.0:
            below += 1
    print(f"probe/scrape separation: n={len(seps)} inverted={inverted} "
          f"ordered={len(seps) - inverted} no_counterpart={no_counterpart} "
          f"min={min(seps) if seps else float('nan'):+.2f}h "
          f"below_6h={below}")
    assert inverted <= len(seps), (
        f"inverted={inverted} exceeds the {len(seps)} measured pairs, so a "
        f"state with no counterpart is being counted as an observed "
        f"inversion again -- the exact conflation this guard exists for")
    assert below == 0, (
        f"{below} probe run(s) now sit within 6.0h of the scrape that writes "
        f"their target day. THE INTERMITTENCY HAS STARTED: the UNVERIFIABLE "
        f"refusal is no longer a constant, and its two causes must be read "
        f"from the ORDERING field rather than assumed. Separations: "
        f"{[round(x, 2) for x in seps]}")


def test_the_CLOCK_and_the_POPULATION_are_INDEPENDENT_conditions():
    """The error this cost nothing to find, pinned so it is not re-made.

    `offset is not None` does NOT imply the target day's population exists.
    Measured on run 39131837165: offset +0 (verified) with
    ORDERING_INVERTED for the target day, simultaneously.
    """
    src = pathlib.Path("scripts/fixture_horizon_probe.py").read_text(
        encoding="utf-8")
    assert "REACHED HERE ON THE NEXT" in src, (
        "the correction is no longer recorded at the guard")
    # the guard must not be conditioned on the offset
    body = src.split("def main(", 1)[1]
    guard_line = [l for l in body.splitlines()
                  if "if ordering != ORDERING_POPULATION_EXISTS" in l]
    assert guard_line, "the guard is gone"
    assert "offset" not in guard_line[0], (
        "the ordering guard now depends on the offset, which is the exact "
        "assumption run 39131837165 disproved")


# ───────── THE SCHEDULE IS DISABLED, 2026-10-18. SUCCESSOR CONTRACT.
#
#     A nominal gap cannot order two jobs that are both late. Ordering is
#     guaranteed by dependency or it is not guaranteed.
#
# The right instinct was a `workflow_run` trigger on `daily-picks` completion.
# The measurement says the ordering was never the binding constraint:
#
#   MEASURED 2026-10-18, n=1,131 fixture rows written since 2026-09-01:
#     lead time write -> kickoff   mean 8.98h  median 8.72h  MAX 23.62h
#     rows ever known >24h ahead   ZERO
#
# The probe serves TOMORROW, whose earliest kickoff is 24-28h from any probe
# run on day D. `matches` has never held a fixture that far ahead, because
# `max_days_ahead` is a rolling 24h and D+1's early card is written by D+1's
# OWN scrape - which runs after every probe targeting D+1. So the target day's
# population cannot exist under ANY ordering, and a dependency trigger would
# have fixed the ordering, changed nothing, and kept paying 30 page-loads a
# day.

def test_the_probe_has_NO_SCHEDULE_until_the_verifier_is_REPAIRED():
    """It cannot emit a deadline, so it does not run.

    WHEN THIS GOES RED: either the schedule came back, or
    `test_the_probe_still_cannot_emit_a_deadline` below has flipped. Both want
    the same question asked - can the verifier see the target day yet? - and
    neither is answered by adding a cron.
    """
    import yaml

    wf = yaml.safe_load(
        pathlib.Path(".github/workflows/fixture-horizon-probe.yml")
        .read_text(encoding="utf-8"))
    on = wf.get("on") or wf.get(True)
    assert "schedule" not in on, (
        "the probe is scheduled again. It has never emitted a deadline: the "
        "target day's population cannot exist when it runs (max observed "
        "fixture lead 23.62h against a 24-28h requirement). Re-enabling costs "
        "30 Flashscore page-loads a day for nothing unless verify_offset's "
        "repair has landed.")
    assert "workflow_dispatch" in on, (
        "dispatch was removed too, so the repair cannot be tested without a "
        "push")
    src = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    assert "RE-ENABLE CONDITION" in src, (
        "the disable does not record what would justify undoing it")
    assert "23.62h" in src and "240 page-loads" in src, (
        "the disable does not carry the measurement that justified it")


def test_the_THREE_ORDERING_STATES_SURVIVE_the_disable():
    """A state that becomes rare must not be deleted — and this one is not
    even rare, it is universal. Disproved once already this morning."""
    assert set(probe.ORDERING_STATES) == {
        probe.ORDERING_POPULATION_EXISTS, probe.ORDERING_INVERTED,
        probe.ORDERING_UNKNOWN}
    assert probe.REFUSAL_EMPTY_POPULATION != probe.REFUSAL_ORDERING_INVERTED


def test_the_probe_still_cannot_emit_a_deadline_FOR_A_STATED_REASON():
    """The arithmetic that makes the disable correct, re-derived here so it is
    checkable rather than quoted.

    If `MAX_OBSERVED_FIXTURE_LEAD_H` is ever measured above the requirement,
    the probe CAN see its target day and the disable must be revisited.
    """
    #: `measured` 2026-10-18 over n=1,131 rows written since 2026-09-01.
    MAX_OBSERVED_FIXTURE_LEAD_H = 23.62
    #: Tomorrow's earliest kickoff, from the measured percentiles.
    # THE ANCHOR IS A LOCAL TIME SINCE 2026-10-22, so it is asked for a DAY.
    # The WINTER conversion is used here because it is the LATER of the two
    # (10:30 vs 09:30 UTC) and therefore the most favourable to the probe --
    # if the probe cannot reach even that, it cannot reach either.
    import datetime as _d

    from scripts.schedule_deadline import (DEADLINE_PERCENTILES,
                                           observed_earliest_min)

    # A probe running at the EARLIEST plausible moment on day D - right after
    # the weekend scrape completes, ~06:16 - is the most favourable case.
    probe_at_h = 6 + 16 / 60
    for label, ko_min in (("weekend 5th pct",
                           DEADLINE_PERCENTILES["weekend"][0]),
                          ("weekday observed min (winter, the laxer side)",
                           observed_earliest_min(
                               "weekday", _d.date(2026, 10, 26))),
                          ("weekday observed min (summer)",
                           observed_earliest_min(
                               "weekday", _d.date(2026, 10, 24)))):
        required_h = 24 + ko_min / 60 - probe_at_h
        assert required_h > MAX_OBSERVED_FIXTURE_LEAD_H, (
            f"{label}: the probe would need a fixture known {required_h:.2f}h "
            f"ahead and the measured maximum is "
            f"{MAX_OBSERVED_FIXTURE_LEAD_H}h. If this now passes, `matches` "
            f"holds fixtures further ahead than it did and the probe's "
            f"schedule should be reconsidered.")


# ───────── ITS REMAINING PURPOSE, SIZED 2026-10-19
#
#     A producer kept alive for a reason that no longer exists is the
#     thirteen-instance pattern with a dispatch trigger attached.
#
# The day-ahead deadline now comes from the external reference, so the probe
# had to justify itself again. It has exactly one remaining job and it is
# measurable: the reference covers 9 of 30 configured leagues and this probe
# fetches all 30 (`pages fetched : 30/30` on every one of the 8 fetching
# runs). The gap is worth FIFTEEN MINUTES, measured on Easter Monday 2026 --
# truth 10:15 in the uncovered `poland/ekstraklasa`, earliest covered fixture
# 10:30 -- and 15 minutes is immaterial under the cron in force (+253m on that
# same date replayed under `0 0`).

def test_the_disable_records_its_REMAINING_PURPOSE_and_BOTH_clauses():
    """Disabled is not the same as pointless, and the file must say which."""
    src = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    assert "9 of the 30 configured leagues" in src, (
        "the coverage gap -- the probe's only remaining advantage -- is not "
        "recorded")
    assert "FIFTEEN MINUTES" in src, (
        "the gap is named but not sized; an unsized advantage cannot be "
        "weighed against 60 page-loads a day")
    assert "TWO clauses" in src and "both required" in src, (
        "the re-enable condition still reads as a single clause. The repair "
        "alone is not sufficient: a repaired probe buys 15 minutes, which is "
        "inside the margin under the cron in force.")
    assert "0 4" in src and "+253m" in src, (
        "the condition does not say under which schedule the gap would start "
        "to matter")


def test_the_PROSPECTIVE_margin_is_no_longer_the_probes_to_lose():
    """The claim `nothing is lost by stopping` has to stay true.

    It was true on 2026-10-18 because the probe had never emitted a deadline.
    It is true for a second reason now: the audit computes the prospective
    margin from the reference every run, so stopping the probe does not leave
    the check unevaluated.
    """
    import inspect

    import scripts.ci_audit as ci

    src = inspect.getsource(ci.main)
    assert "actionable margin" in src, (
        "the audit no longer computes the prospective margin, so disabling "
        "the probe DOES lose something and the workflow comment is wrong")
    wf = pathlib.Path(".github/workflows/fixture-horizon-probe.yml").read_text(
        encoding="utf-8")
    assert "no longer UNEVALUATED" in wf
