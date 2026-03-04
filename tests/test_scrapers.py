"""Tests for scraper modules."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch



class TestNewsScraper:
    """Tests for the news scraper."""

    def test_parse_rss_empty(self):
        from src.scrapers.news_scraper import NewsScraper
        scraper = NewsScraper.__new__(NewsScraper)
        scraper._sentiment_analyzer = None
        result = scraper._parse_rss("<rss><channel></channel></rss>", "test")
        assert result == []

    def test_sentiment_positive(self):
        from src.scrapers.news_scraper import NewsScraper
        scraper = NewsScraper.__new__(NewsScraper)
        scraper._sentiment_analyzer = None
        score = scraper._analyze_sentiment("This is absolutely great and wonderful news")
        assert score > 0.0

    def test_sentiment_negative(self):
        from src.scrapers.news_scraper import NewsScraper
        scraper = NewsScraper.__new__(NewsScraper)
        scraper._sentiment_analyzer = None
        score = scraper._analyze_sentiment("This is terrible and awful news, very bad")
        assert score < 0.0


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
        summary = asyncio.get_event_loop().run_until_complete(
            scraper.get_injury_summary(team_id=99999)
        )
        assert summary["total_injured"] == 0
        assert summary["injuries"] == []
