"""Tests for scraper modules."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch


class TestOddsScraper:
    """Tests for the Odds API scraper."""

    def test_league_mapping_coverage(self):
        from src.scrapers.odds_scraper import LEAGUE_TO_SPORT_KEY
        # Verify key leagues are mapped
        assert "england/premier-league" in LEAGUE_TO_SPORT_KEY
        assert "spain/laliga" in LEAGUE_TO_SPORT_KEY
        assert "champions-league" in LEAGUE_TO_SPORT_KEY

    def test_market_mapping(self):
        from src.scrapers.odds_scraper import MARKET_MAP
        assert MARKET_MAP["h2h"] == "1X2"
        assert MARKET_MAP["totals"] == "over_under"


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

    def test_parse_empty_page(self):
        from src.scrapers.injury_scraper import InjuryScraper
        scraper = InjuryScraper.__new__(InjuryScraper)
        result = scraper._parse_injuries_page("<html><body></body></html>", 1)
        assert result == []
