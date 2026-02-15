"""News aggregator and sentiment analyzer."""

from datetime import datetime, timedelta
from typing import List, Optional

from bs4 import BeautifulSoup

from src.scrapers.base_scraper import BaseScraper
from src.data.models import News, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

# RSS feed URLs for football news
NEWS_SOURCES = {
    "bbc_sport": {
        "url": "https://feeds.bbci.co.uk/sport/football/rss.xml",
        "name": "BBC Sport",
    },
    "espn_fc": {
        "url": "https://www.espn.com/espn/rss/soccer/news",
        "name": "ESPN FC",
    },
    "skysports": {
        "url": "https://www.skysports.com/rss/12040",
        "name": "Sky Sports Football",
    },
}


class NewsScraper(BaseScraper):
    """Aggregates football news from multiple sources with sentiment analysis."""

    def __init__(self, config=None):
        super().__init__(config)
        self._sentiment_analyzer = None

    def _get_sentiment_analyzer(self):
        """Lazy-load the VADER sentiment analyzer."""
        if self._sentiment_analyzer is None:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer loaded")
        return self._sentiment_analyzer

    async def update(self):
        """Fetch latest news from all sources."""
        logger.info("Starting news update cycle")

        for source_key, source_info in NEWS_SOURCES.items():
            try:
                await self.fetch_rss_feed(source_info["url"], source_info["name"])
                await self._rate_limit()
            except Exception as e:
                logger.error(f"Error fetching news from {source_info['name']}: {e}")

        logger.info("News update cycle complete")

    async def fetch_rss_feed(self, feed_url: str, source_name: str) -> List[dict]:
        """Fetch and parse an RSS feed.

        Args:
            feed_url: URL of the RSS feed
            source_name: Name of the news source

        Returns:
            List of news article dictionaries
        """
        try:
            xml_text = await self.fetch(feed_url)
        except Exception as e:
            logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []

        articles = self._parse_rss(xml_text, source_name)

        db = get_db()
        saved = 0
        with db.get_session() as session:
            for article in articles:
                if self._save_article(session, article):
                    saved += 1

        logger.info(f"Saved {saved} new articles from {source_name}")
        return articles

    def _parse_rss(self, xml_text: str, source_name: str) -> List[dict]:
        """Parse RSS XML into article dictionaries."""
        soup = BeautifulSoup(xml_text, "xml")
        articles = []

        items = soup.find_all("item")
        for item in items:
            try:
                title = item.find("title")
                description = item.find("description")
                link = item.find("link")
                pub_date = item.find("pubDate")

                headline = title.text.strip() if title else ""
                content = description.text.strip() if description else ""
                url = link.text.strip() if link else ""

                published_at = datetime.now()
                if pub_date:
                    try:
                        from email.utils import parsedate_to_datetime
                        published_at = parsedate_to_datetime(pub_date.text.strip())
                    except Exception:
                        pass

                # Compute sentiment
                sentiment = self._analyze_sentiment(headline + " " + content)

                # Try to match to a team
                team_id = self._match_team(headline + " " + content)

                articles.append({
                    "headline": headline,
                    "content": content,
                    "url": url,
                    "source": source_name,
                    "published_at": published_at,
                    "sentiment_score": sentiment,
                    "team_id": team_id,
                })

            except Exception as e:
                logger.debug(f"Error parsing RSS item: {e}")
                continue

        return articles

    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using VADER. Returns score from -1 (negative) to +1 (positive)."""
        try:
            analyzer = self._get_sentiment_analyzer()
            scores = analyzer.polarity_scores(text)
            return scores["compound"]  # -1 to +1 compound score
        except Exception as e:
            logger.debug(f"Sentiment analysis failed: {e}")
            return 0.0

    def _match_team(self, text: str) -> Optional[int]:
        """Try to match article text to a team in the database.

        Returns team_id if a match is found, None otherwise.
        """
        db = get_db()
        try:
            with db.get_session() as session:
                teams = session.query(Team).all()
                text_lower = text.lower()
                for team in teams:
                    if team.name.lower() in text_lower:
                        return team.id
        except Exception:
            pass
        return None

    def _save_article(self, session, article: dict) -> bool:
        """Save a news article, avoiding duplicates. Returns True if new."""
        existing = session.query(News).filter_by(url=article["url"]).first()
        if existing:
            return False

        news = News(
            team_id=article.get("team_id"),
            headline=article["headline"],
            content=article["content"],
            source=article["source"],
            url=article["url"],
            sentiment_score=article.get("sentiment_score", 0.0),
            published_at=article.get("published_at"),
        )
        session.add(news)
        return True

    async def get_team_sentiment(self, team_id: int, days: int = 7) -> dict:
        """Get aggregated sentiment for a team over recent days."""
        db = get_db()
        cutoff = datetime.utcnow() - timedelta(days=days)

        with db.get_session() as session:
            articles = session.query(News).filter(
                News.team_id == team_id,
                News.published_at >= cutoff,
            ).order_by(News.published_at.desc()).all()

            if not articles:
                return {"avg_sentiment": 0.0, "article_count": 0, "trend": "neutral"}

            scores = [a.sentiment_score for a in articles if a.sentiment_score is not None]

        avg = sum(scores) / len(scores) if scores else 0.0

        trend = "neutral"
        if avg > 0.3:
            trend = "positive"
        elif avg < -0.3:
            trend = "negative"

        return {
            "avg_sentiment": round(avg, 3),
            "article_count": len(scores),
            "trend": trend,
        }
