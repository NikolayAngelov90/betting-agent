"""Stage 6, Phase 12 — quota guard, fixture selection, refresh dedup, safety.

Everything runs against temp SQLite (conftest strips DATABASE_URL); nothing here
can reach production.
"""

import asyncio
from datetime import date, datetime, timedelta

import pytest

import src.data.database as db_mod
from src.data.models import Base, Match, Odds, SavedPick, Team
from src.data.odds_quota import (
    CREDITS_PER_REQUEST,
    FREE_TIER_CREDITS,
    OddsApiQuota,
    credits_for,
    month_key,
)


def _mgr(tmp_path, name="q.db"):
    mgr = db_mod.DatabaseManager(
        config=type("C", (), {"database": {"sqlite_path": str(tmp_path / name)}})())
    Base.metadata.create_all(mgr.engine)
    return mgr


# ════════════════════════════════════════════════════════════ quota

def test_cost_model_matches_the_api_pricing():
    """1 credit x regions(1) x markets(h2h,totals = 2) = 2 per league request."""
    assert CREDITS_PER_REQUEST == 2
    assert credits_for(0) == 0
    assert credits_for(1) == 2
    assert credits_for(27) == 54
    assert FREE_TIER_CREDITS == 500


def test_month_key_is_the_first_of_the_month():
    assert month_key(date(2026, 8, 31)) == date(2026, 8, 1)
    assert month_key(date(2026, 8, 1)) == date(2026, 8, 1)


def test_request_allowed_below_budget(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=50)
    assert q.available()
    assert q.claim_requests(3) == 3
    assert q.used() == 6


def test_request_blocked_when_it_would_exceed_budget(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=10, safety_margin=0)
    assert q.claim_requests(5) == 5          # 10 credits, exactly the budget
    assert q.used() == 10
    assert q.claim_requests(1) == 0, "spent past the monthly budget"


def test_safety_margin_is_respected(tmp_path):
    """Budget 100 with a 40-credit margin leaves 60 spendable = 30 requests.

    max_credits_per_run=0 disables the per-run ceiling so this isolates the
    MONTHLY guard; the ceiling has its own tests below.
    """
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=100, safety_margin=40,
                     max_credits_per_run=0)
    assert q.remaining() == 60
    assert q.max_requests() == 30
    assert q.claim_requests(30) == 30
    assert q.claim_requests(1) == 0
    # The margin is untouched, not consumed.
    assert q.used() == 60


def test_partial_grant_when_budget_is_nearly_gone(tmp_path):
    """A truncated grant must be honest, not all-or-nothing: the caller orders
    leagues by urgency and takes the ones it can afford."""
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=10, safety_margin=0)
    assert q.claim_requests(4) == 4      # 8 of 10
    assert q.claim_requests(3) == 1      # only 2 credits left -> 1 request
    assert q.used() == 10


def test_multiple_claims_accumulate(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0)
    for _ in range(5):
        q.claim_requests(2)
    assert q.used() == 20
    assert q.remaining() == 380


def test_month_boundary_resets_the_ledger(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=20, safety_margin=0,
                     max_credits_per_run=0)
    july = date(2026, 7, 15)
    august = date(2026, 8, 2)
    assert q.claim_requests(10, today=july) == 10
    assert q.claim_requests(1, today=july) == 0, "July budget should be spent"
    assert q.claim_requests(5, today=august) == 5, "August starts fresh"
    assert q.used(today=july) == 20
    assert q.used(today=august) == 10


def test_release_returns_unspent_credits(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0)
    q.claim_requests(5)
    assert q.used() == 10
    q.release_requests(2)
    assert q.used() == 6


def test_claim_never_raises_and_degrades_to_allowing(tmp_path, monkeypatch):
    """A missing api_budget table must not crash a scheduled job. It degrades to
    the pre-existing header-based guard, and says so loudly."""
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400)
    monkeypatch.setattr(q._store, "available", lambda: False)
    assert q.claim_requests(3) == 3
    assert "UNAVAILABLE" in q.describe()


def test_per_run_ceiling_caps_a_single_execution(tmp_path):
    """The monthly guard alone cannot stop one pathological run: 27 leagues
    looking imminent would spend 54 credits and still be 'within budget'."""
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0,
                     max_credits_per_run=24)
    assert q.claim_requests(20) == 12, "per-run ceiling not applied"
    assert q.spent_this_run == 24
    assert q.claim_requests(1) == 0, "spent past the per-run ceiling"
    # The MONTHLY budget is untouched beyond what the run actually claimed.
    assert q.used() == 24


def test_per_run_ceiling_applies_even_without_the_ledger(tmp_path, monkeypatch):
    """A degraded run (missing api_budget table) must still be bounded."""
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0,
                     max_credits_per_run=8)
    monkeypatch.setattr(q._store, "available", lambda: False)
    assert q.claim_requests(10) == 4
    assert q.claim_requests(1) == 0


def test_per_run_ceiling_can_be_disabled(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0,
                     max_credits_per_run=0)
    assert q.claim_requests(30) == 30


def test_release_returns_per_run_headroom(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=0,
                     max_credits_per_run=24)
    assert q.claim_requests(12) == 12
    assert q.claim_requests(1) == 0
    q.release_requests(6)
    assert q.spent_this_run == 12
    assert q.claim_requests(1) == 1


def test_describe_reports_the_free_tier_and_margin(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path), monthly_budget=400, safety_margin=50)
    text = q.describe()
    assert "400" in text and "500" in text and "50" in text


# ═══════════════════════════════════════════════ fixture selection

def _scraper(mgr, monkeypatch):
    from src.scrapers.theodds_scraper import TheOddsScraper
    from src.data.run_marker import mark_picks_complete

    s = TheOddsScraper.__new__(TheOddsScraper)
    s.db = mgr
    s.api_key = "test-key"
    # PRECONDITION, not decoration. `refresh_imminent` now declines unless the
    # day's picks run has recorded completion, because a refresh that beats the
    # picks run rewrites the prices those picks are taken at. Every test below
    # exercises refresh behaviour GIVEN that precondition; the guard's own
    # three paths are tested in tests/test_picks_run_guard.py.
    mark_picks_complete(mgr)
    return s


def _seed(mgr, fixtures):
    """fixtures: list of (match_id, league, kickoff, has_pending_pick, is_fixture)"""
    with mgr.get_session() as ses:
        ses.add(Team(id=1, name="Home FC"))
        ses.add(Team(id=2, name="Away FC"))
        for mid, league, ko, pick, is_fx in fixtures:
            ses.add(Match(id=mid, home_team_id=1, away_team_id=2, league=league,
                          match_date=ko, is_fixture=is_fx))
            if pick:
                ses.add(SavedPick(id=mid, match_id=mid, pick_date=ko.date(),
                                  market="1X2", selection="Home Win", odds=2.0,
                                  closing_capture_status="pending"))
        ses.commit()


def test_fixture_inside_window_is_included(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "f1.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    s = _scraper(mgr, monkeypatch)
    lf, skips = s._imminent_league_fixtures(120, now=now)
    assert "england/premier-league" in lf
    assert len(lf["england/premier-league"]) == 1


def test_fixture_outside_window_is_excluded(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "f2.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=300), True, True)])
    s = _scraper(mgr, monkeypatch)
    lf, _ = s._imminent_league_fixtures(120, now=now)
    assert lf == {}


def test_already_started_fixture_is_excluded(tmp_path, monkeypatch):
    """A kickoff in the past can never yield a valid closing line."""
    mgr = _mgr(tmp_path, "f3.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now - timedelta(minutes=5), True, True)])
    s = _scraper(mgr, monkeypatch)
    lf, _ = s._imminent_league_fixtures(120, now=now)
    assert lf == {}


def test_completed_match_is_excluded(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "f4.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, False)])
    s = _scraper(mgr, monkeypatch)
    lf, _ = s._imminent_league_fixtures(120, now=now)
    assert lf == {}, "a non-fixture (settled/postponed) row was selected"


def test_only_leagues_with_a_pending_pick_are_requested(tmp_path, monkeypatch):
    """The core Stage 6 saving: no pick, no credit."""
    mgr = _mgr(tmp_path, "f5.db")
    now = datetime(2026, 9, 1, 18, 0)
    ko = now + timedelta(minutes=60)
    _seed(mgr, [
        (1, "england/premier-league", ko, True, True),
        (2, "spain/laliga", ko, False, True),
        (3, "italy/serie-a", ko, False, True),
    ])
    s = _scraper(mgr, monkeypatch)
    lf, skips = s._imminent_league_fixtures(120, now=now)
    assert list(lf) == ["england/premier-league"]
    assert "no pending pick" in skips["spain/laliga"]


def test_unmapped_league_is_skipped_with_a_reason(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "f6.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "romania/liga-1", now + timedelta(minutes=60), True, True)])
    s = _scraper(mgr, monkeypatch)
    lf, skips = s._imminent_league_fixtures(120, now=now)
    assert lf == {}
    assert "not mapped" in skips["romania/liga-1"]


def test_already_captured_pick_does_not_trigger_a_refresh(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "f7.db")
    now = datetime(2026, 9, 1, 18, 0)
    ko = now + timedelta(minutes=60)
    _seed(mgr, [(1, "england/premier-league", ko, True, True)])
    with mgr.get_session() as ses:
        p = ses.get(SavedPick, 1)
        p.closing_odds = 2.1
        p.closing_capture_status = "captured"
        ses.commit()
    s = _scraper(mgr, monkeypatch)
    lf, _ = s._imminent_league_fixtures(120, now=now)
    assert lf == {}, "spent a credit on a pick that already has a closing price"


def test_leagues_are_ordered_by_soonest_kickoff(tmp_path, monkeypatch):
    """If the budget truncates the list, the most urgent leagues keep credits."""
    mgr = _mgr(tmp_path, "f8.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [
        (1, "spain/laliga", now + timedelta(minutes=100), True, True),
        (2, "england/premier-league", now + timedelta(minutes=20), True, True),
        (3, "italy/serie-a", now + timedelta(minutes=60), True, True),
    ])
    s = _scraper(mgr, monkeypatch)
    lf, _ = s._imminent_league_fixtures(120, now=now)
    assert list(lf) == ["england/premier-league", "italy/serie-a", "spain/laliga"]


# ═══════════════════════════════════════════════ refresh dedup

def test_recently_refreshed_league_is_detected(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "d1.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    with mgr.get_session() as ses:
        ses.add(Odds(match_id=1, bookmaker="TheOddsAPI-pinnacle",
                     market_type="1X2", selection="Home", odds_value=2.0,
                     timestamp=now - timedelta(minutes=30)))
        ses.commit()
    s = _scraper(mgr, monkeypatch)
    recent = s._leagues_refreshed_since(["england/premier-league"],
                                        now - timedelta(minutes=180))
    assert recent == {"england/premier-league"}


def test_stale_league_is_not_treated_as_fresh(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "d2.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    with mgr.get_session() as ses:
        ses.add(Odds(match_id=1, bookmaker="TheOddsAPI-pinnacle",
                     market_type="1X2", selection="Home", odds_value=2.0,
                     timestamp=now - timedelta(hours=9)))
        ses.commit()
    s = _scraper(mgr, monkeypatch)
    assert s._leagues_refreshed_since(["england/premier-league"],
                                      now - timedelta(minutes=180)) == set()


def test_non_odds_api_rows_do_not_count_as_a_refresh(tmp_path, monkeypatch):
    """API-Football rows are written by a different job; they must not make an
    Odds API league look freshly refreshed."""
    mgr = _mgr(tmp_path, "d3.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    with mgr.get_session() as ses:
        ses.add(Odds(match_id=1, bookmaker="Bet365", market_type="1X2",
                     selection="Home", odds_value=2.0,
                     timestamp=now - timedelta(minutes=5)))
        ses.commit()
    s = _scraper(mgr, monkeypatch)
    assert s._leagues_refreshed_since(["england/premier-league"],
                                      now - timedelta(minutes=180)) == set()


def test_duplicate_run_makes_no_second_request(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "d4.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    with mgr.get_session() as ses:
        ses.add(Odds(match_id=1, bookmaker="TheOddsAPI-pinnacle",
                     market_type="1X2", selection="Home", odds_value=2.0,
                     timestamp=now - timedelta(minutes=10)))
        ses.commit()
    s = _scraper(mgr, monkeypatch)
    called = []
    monkeypatch.setattr(s, "_fetch_and_persist",
                        lambda *a, **k: called.append(1) or 0)
    plan = asyncio.run(s.refresh_imminent(
        window_minutes=120, min_interval_minutes=180, now=now))
    assert plan["requested"] == []
    assert not called
    assert "refreshed within" in plan["skipped"]["england/premier-league"]


# ═══════════════════════════════════════════════ budget enforcement

def test_no_api_call_when_budget_is_exhausted(tmp_path, monkeypatch):
    """The single most important safety property of Stage 6."""
    mgr = _mgr(tmp_path, "b1.db")
    q = OddsApiQuota(mgr, monthly_budget=2, safety_margin=0)
    assert q.claim_requests(1) == 1        # budget now gone
    s = _scraper(mgr, monkeypatch)

    fetched = []

    async def _boom(sport_key):
        fetched.append(sport_key)
        return []

    monkeypatch.setattr(s, "_fetch_league_odds", _boom)
    written = asyncio.run(s._fetch_and_persist(
        {"england/premier-league": [{"match_id": 1, "home_name": "a",
                                     "away_name": "b",
                                     "match_date": datetime(2026, 9, 1, 19, 0)}]},
        quota=q))
    assert written == 0
    assert fetched == [], "an API call was made with no budget left"


def test_dry_run_spends_nothing(tmp_path, monkeypatch):
    mgr = _mgr(tmp_path, "b2.db")
    now = datetime(2026, 9, 1, 18, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60), True, True)])
    s = _scraper(mgr, monkeypatch)
    called = []
    monkeypatch.setattr(s, "_fetch_and_persist",
                        lambda *a, **k: called.append(1) or 0)
    q = OddsApiQuota(mgr, monthly_budget=400, safety_margin=0)
    plan = asyncio.run(s.refresh_imminent(
        window_minutes=1440, min_interval_minutes=0, quota=q, dry_run=True,
        now=now))
    assert plan["dry_run"] is True
    assert plan["credits_estimated"] == 2
    assert not called
    assert q.used() == 0, "a dry run claimed credits"


# ═══════════════════════════════════════════════ safety

def test_scripts_do_not_load_dotenv_at_import():
    """Same guard as Stage 5, extended to the new scripts."""
    import ast
    import pathlib

    for name in ("refresh_and_capture", "capture_closing_lines",
                 "paper_trading_report", "run_baseline", "run_clean_baseline"):
        path = pathlib.Path("scripts") / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call)
                        and isinstance(call.func, ast.Name)
                        and call.func.id == "load_dotenv"):
                    raise AssertionError(
                        f"{path} calls load_dotenv() at import time — tests "
                        f"could reach production")


def test_no_credentials_in_the_quota_module():
    import pathlib
    src = pathlib.Path("src/data/odds_quota.py").read_text(encoding="utf-8")
    for bad in ("apiKey", "api_key=", "ODDS_API_KEY ="):
        assert bad not in src


def test_workflow_never_enables_real_money():
    import pathlib
    wf = pathlib.Path(".github/workflows/closing-lines.yml").read_text(encoding="utf-8")
    assert "paper_trading_mode: false" not in wf
    assert "--picks" not in wf, "the capture workflow must not generate picks"


# ═══════════════════════════════════ provider reconciliation (Stage 7, section 14)

def test_reconcile_adopts_out_of_band_provider_spend(tmp_path):
    """Measured on the real account 2026-08-10: The Odds API reported 95 credits
    used for August while our ledger read 0. The ledger only sees spend that
    goes through it; the provider counts everything on the key. Budgeting
    against the ledger alone would have authorised 400 more on top of 95."""
    q = OddsApiQuota(_mgr(tmp_path, "rec.db"), monthly_budget=400, safety_margin=50)
    assert q.used() == 0

    assert q.reconcile(95) == 95
    assert q.used() == 95, "out-of-band spend was ignored"
    # 400 budget - 95 already gone - 50 margin = 255 spendable.
    assert q.remaining() == 255
    assert q.max_requests() == 127


def test_reconcile_never_lowers_the_ledger(tmp_path):
    """A provider count below ours means a month boundary or a different key.
    Spending more on the strength of that is the wrong direction to guess."""
    q = OddsApiQuota(_mgr(tmp_path, "rec2.db"), monthly_budget=400,
                     safety_margin=0, max_credits_per_run=0)
    q.claim_requests(50)                      # 100 credits
    assert q.used() == 100
    assert q.reconcile(4) == 100
    assert q.used() == 100, "the ledger was lowered on a stale provider count"


def test_reconcile_tolerates_missing_or_junk_values(tmp_path):
    q = OddsApiQuota(_mgr(tmp_path, "rec3.db"), monthly_budget=400, safety_margin=0)
    q.claim_requests(2)
    assert q.reconcile(None) == 4
    assert q.reconcile("not-a-number") == 4
    assert q.used() == 4


def test_reconciled_ledger_blocks_further_spend_when_tier_is_gone(tmp_path):
    """The failure this exists to prevent: the free tier is exhausted by
    out-of-band use and the pipeline keeps spending because its own count is
    still low."""
    q = OddsApiQuota(_mgr(tmp_path, "rec4.db"), monthly_budget=400, safety_margin=50)
    q.reconcile(FREE_TIER_CREDITS)            # provider says the tier is gone
    assert q.remaining() == 0
    assert q.claim_requests(1) == 0, "spent credits the account does not have"


def test_refresh_reconciles_from_response_headers(tmp_path, monkeypatch):
    """The plan must carry both numbers so a CI log shows the divergence."""
    mgr = _mgr(tmp_path, "rec5.db")
    scraper = _scraper(mgr, monkeypatch)
    now = datetime(2026, 9, 1, 12, 0, 0)
    _seed(mgr, [(1, "england/premier-league", now + timedelta(minutes=60),
                 True, True)])

    q = OddsApiQuota(mgr, monthly_budget=400, safety_margin=0)
    # The scraper "learns" the provider's count the way a real response does.
    scraper._used_requests = 95
    scraper._remaining_requests = 405

    async def _fake_fetch(league_fixtures, quota=None):
        if quota is not None:
            quota.claim_requests(len(league_fixtures))
        return 0

    monkeypatch.setattr(scraper, "_fetch_and_persist", _fake_fetch)
    plan = asyncio.run(scraper.refresh_imminent(
        window_minutes=120, min_interval_minutes=0, quota=q, now=now))

    assert plan["credits_used_provider"] == 95
    # 2 credits claimed for the one league, then reconciled up to the provider.
    assert plan["credits_used_ledger"] == 95
    assert q.used() == 95


# ═══════════════════════════ concurrency (Stage 7, section 13)

def test_concurrent_claims_never_oversell_the_budget(tmp_path):
    """Section 13 asks for the existing mechanism to be TESTED, not replaced.

    The protocol is a conditional UPDATE — `SET used = used + n WHERE used + n
    <= ceiling` — which PostgreSQL serialises with a row lock. The property that
    matters is the same on either backend: however many workers race, the total
    granted can never exceed the budget. A read-then-write counter would fail
    this; that is exactly why the ledger is not one.
    """
    import threading

    mgr = _mgr(tmp_path, "conc.db")
    budget = 40                                # 20 requests
    granted = []
    lock = threading.Lock()
    start = threading.Event()

    def worker():
        q = OddsApiQuota(mgr, monthly_budget=budget, safety_margin=0,
                         max_credits_per_run=0)
        start.wait()
        got = q.claim_requests(3)
        with lock:
            granted.append(got)

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    start.set()
    for t in threads:
        t.join()

    total_credits = credits_for(sum(granted))
    assert total_credits <= budget, (
        f"{len(threads)} racing workers spent {total_credits} of a "
        f"{budget}-credit budget")
    # And the ledger agrees with what it handed out — no lost updates.
    q = OddsApiQuota(mgr, monthly_budget=budget, safety_margin=0)
    assert q.used() == total_credits


def test_a_second_run_cannot_respend_the_first_runs_claim(tmp_path):
    """Two overlapping workflow runs are separate processes with separate
    per-run counters. Only the shared ledger stops the second one."""
    mgr = _mgr(tmp_path, "conc2.db")
    run_a = OddsApiQuota(mgr, monthly_budget=20, safety_margin=0,
                         max_credits_per_run=0)
    run_b = OddsApiQuota(mgr, monthly_budget=20, safety_margin=0,
                         max_credits_per_run=0)

    assert run_a.claim_requests(10) == 10      # 20 credits, the whole budget
    assert run_b.spent_this_run == 0, "run B starts with a clean per-run counter"
    assert run_b.claim_requests(1) == 0, (
        "an overlapping run spent budget the first run had already claimed")
