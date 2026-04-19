"""Tests for scraper modules."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch



class TestAPIFootballScraper:
    """Tests for API-Football scraper budget helpers."""

    def test_remaining_budget_default(self):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = 0
        assert scraper.remaining_budget() == 100 - scraper.BUDGET_RESERVE

    def test_remaining_budget_after_requests(self):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = 60
        assert scraper.remaining_budget() == 100 - 60 - scraper.BUDGET_RESERVE

    def test_remaining_budget_never_negative(self):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = 200
        assert scraper.remaining_budget() == 0

    def test_fallback_bookmakers_defined(self):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        assert len(APIFootballScraper._TOP_BOOKMAKERS) >= 3
        assert len(APIFootballScraper._FALLBACK_BOOKMAKERS) >= 3
        assert APIFootballScraper._TOP_BOOKMAKERS.isdisjoint(
            APIFootballScraper._FALLBACK_BOOKMAKERS
        )

    def test_save_fixture_odds_primary_succeeds(self):
        """When primary bookmakers return odds, fallback is not called."""
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._logged_unknown_bets = set()
        calls = []
        def mock_save(match_id, odds_resp, bookmakers):
            calls.append(bookmakers)
            if bookmakers is APIFootballScraper._TOP_BOOKMAKERS:
                return 5  # primary found odds
            return 3
        scraper._save_odds_from_set = mock_save
        result = scraper._save_fixture_odds(1, [{}])
        assert result == 5
        assert len(calls) == 1  # fallback never called

    def test_save_fixture_odds_fallback_triggers(self):
        """When primary returns 0, fallback tier is tried."""
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._logged_unknown_bets = set()
        calls = []
        def mock_save(match_id, odds_resp, bookmakers):
            calls.append(bookmakers)
            if bookmakers is APIFootballScraper._TOP_BOOKMAKERS:
                return 0  # primary has nothing
            return 4  # fallback finds odds
        scraper._save_odds_from_set = mock_save
        result = scraper._save_fixture_odds(1, [{}])
        assert result == 4
        assert len(calls) == 2  # both tiers called


class TestInjuryScraper:
    """Tests for injury scraper."""

    def test_init_without_apifootball(self):
        from src.scrapers.injury_scraper import InjuryScraper
        scraper = InjuryScraper(config={}, apifootball=None)
        assert scraper.apifootball is None

    def test_get_injury_summary_empty(self):
        from src.scrapers.injury_scraper import InjuryScraper
        import asyncio
        scraper = InjuryScraper(config={}, apifootball=None)
        summary = asyncio.run(
            scraper.get_injury_summary(team_id=99999)
        )
        assert summary["total_injured"] == 0
        assert summary["injuries"] == []


class TestOddsQuotaSemaphore:
    """Story 2.1: verify quota-safe semaphore infrastructure."""

    def _make_scraper(self, requests_today=0):
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = requests_today
        scraper.BUDGET_RESERVE = APIFootballScraper.BUDGET_RESERVE
        scraper._quota_exhausted = False
        return scraper

    def test_semaphore_capacity_min_of_budget_and_fixtures(self):
        """Capacity = min(budget, n_fixtures) when budget is the limiting factor."""
        scraper = self._make_scraper()
        sem = scraper._make_odds_semaphore(remaining_budget=10, n_fixtures=49)
        assert sem._value == 10

    def test_semaphore_capacity_limited_by_fixture_count(self):
        """Capacity = min(budget, n_fixtures) when fixture count is smaller."""
        scraper = self._make_scraper()
        sem = scraper._make_odds_semaphore(remaining_budget=49, n_fixtures=3)
        assert sem._value == 3

    def test_semaphore_capacity_floor_when_budget_zero(self):
        """Capacity floors at 1 when remaining_budget=0 — prevents asyncio.Semaphore(0) hang."""
        scraper = self._make_scraper()
        sem = scraper._make_odds_semaphore(remaining_budget=0, n_fixtures=5)
        assert sem._value == 1

    def test_quota_guard_skips_when_exhausted(self):
        """When quota is at the safety limit, no HTTP call is made."""
        import asyncio
        from unittest.mock import AsyncMock
        from src.scrapers.apifootball_scraper import APIFootballScraper
        scraper = self._make_scraper(
            requests_today=100 - APIFootballScraper.BUDGET_RESERVE
        )
        scraper._fetch_fixture_odds = AsyncMock(return_value=[{"bookmakers": []}])
        sem = asyncio.Semaphore(1)

        result = asyncio.run(scraper._fetch_odds_guarded(sem, 42, 999, "england/premier-league"))

        assert result == (42, None, "england/premier-league")
        scraper._fetch_fixture_odds.assert_not_called()

    def test_quota_guard_fetches_when_budget_ok(self):
        """When quota allows, _fetch_fixture_odds is called and result returned."""
        import asyncio
        from unittest.mock import AsyncMock
        scraper = self._make_scraper(requests_today=0)
        mock_odds = [{"bookmakers": [{"name": "Bet365"}]}]
        scraper._fetch_fixture_odds = AsyncMock(return_value=mock_odds)
        sem = asyncio.Semaphore(1)

        result = asyncio.run(scraper._fetch_odds_guarded(sem, 42, 999, "england/premier-league"))

        assert result == (42, mock_odds, "england/premier-league")
        scraper._fetch_fixture_odds.assert_called_once_with(999)


class TestParallelOddsFetch:
    """Story 2.2: verify asyncio.gather replaces sequential loop in fetch_upcoming_odds."""

    def _make_full_scraper(self, fixture_data):
        """fixture_data: list of (match_id, apifootball_id, league) tuples.
        Use 'england/premier-league' — it is in PRIORITY_LEAGUES.
        """
        from unittest.mock import MagicMock
        from src.scrapers.apifootball_scraper import APIFootballScraper

        scraper = APIFootballScraper.__new__(APIFootballScraper)
        scraper._daily_limit = 100
        scraper._requests_today = 0
        scraper._quota_exhausted = False
        scraper.enabled = True
        scraper._today_fixture_count = len(fixture_data)
        scraper._logged_unknown_bets = set()

        # Two sequential get_session() calls inside fetch_upcoming_odds:
        # 1st: load today's fixtures (.all())
        # 2nd: count existing odds per fixture (.count())
        session1 = MagicMock()
        session1.__enter__ = lambda s: s
        session1.__exit__ = MagicMock(return_value=False)

        mock_fixtures = []
        for mid, afid, lg in fixture_data:
            f = MagicMock()
            f.id = mid
            f.apifootball_id = afid
            f.league = lg
            mock_fixtures.append(f)
        session1.query.return_value.filter.return_value.all.return_value = mock_fixtures

        session2 = MagicMock()
        session2.__enter__ = lambda s: s
        session2.__exit__ = MagicMock(return_value=False)
        session2.query.return_value.filter.return_value.count.return_value = 0

        scraper.db = MagicMock()
        scraper.db.get_session.side_effect = [session1, session2]

        scraper._save_fixture_odds = MagicMock(return_value=1)
        return scraper

    def test_exception_in_gather_result_is_warned_and_skipped(self):
        """When one coroutine raises, it is WARNING-logged and skipped; others proceed."""
        import asyncio
        from unittest.mock import AsyncMock, patch

        LEAGUE = "england/premier-league"
        scraper = self._make_full_scraper([
            (1, 101, LEAGUE),
            (2, 102, LEAGUE),
            (3, 103, LEAGUE),
        ])

        async def _guarded(sem, match_id, fixture_id, league):
            if fixture_id == 102:
                raise RuntimeError("network timeout")
            return (match_id, [{"bookmakers": []}], league)

        scraper._fetch_odds_guarded = AsyncMock(side_effect=_guarded)

        with patch("src.scrapers.apifootball_scraper.logger") as mock_logger:
            asyncio.run(scraper.fetch_upcoming_odds())

        assert scraper._save_fixture_odds.call_count == 2
        warning_calls = [str(c) for c in mock_logger.warning.call_args_list]
        assert any("network timeout" in w for w in warning_calls)

    def test_all_successful_results_save_odds(self):
        """When all gather results are valid, _save_fixture_odds called for each."""
        import asyncio
        from unittest.mock import AsyncMock

        LEAGUE = "england/premier-league"
        scraper = self._make_full_scraper([
            (10, 110, LEAGUE),
            (11, 111, LEAGUE),
            (12, 112, LEAGUE),
        ])

        async def _guarded(sem, match_id, fixture_id, league):
            return (match_id, [{"bookmakers": []}], league)

        scraper._fetch_odds_guarded = AsyncMock(side_effect=_guarded)

        asyncio.run(scraper.fetch_upcoming_odds())

        assert scraper._save_fixture_odds.call_count == 3

    def test_dispatch_sliced_to_odds_budget(self):
        """Only odds_budget coroutines are dispatched even if more fixtures exist."""
        import asyncio
        from unittest.mock import AsyncMock

        LEAGUE = "england/premier-league"
        # 5 fixtures but budget forced to 2 by exhausting quota
        scraper = self._make_full_scraper([
            (20, 120, LEAGUE),
            (21, 121, LEAGUE),
            (22, 122, LEAGUE),
            (23, 123, LEAGUE),
            (24, 124, LEAGUE),
        ])
        # Set requests_today so only 2 budget slots remain
        # odds_budget = daily_limit - requests_today - BUDGET_RESERVE - injury_reserve
        # injury_reserve = min(40, 5+10) = 15
        # We want odds_budget = 2: requests_today = 100 - 9 - 15 - 2 = 74
        scraper._requests_today = 74

        calls = []

        async def _guarded(sem, match_id, fixture_id, league):
            calls.append(fixture_id)
            return (match_id, [{"bookmakers": []}], league)

        scraper._fetch_odds_guarded = AsyncMock(side_effect=_guarded)

        asyncio.run(scraper.fetch_upcoming_odds())

        assert len(calls) <= 2
