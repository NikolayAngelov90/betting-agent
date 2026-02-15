"""Main feature engineering pipeline that combines all feature sources."""

import numpy as np
from typing import Optional

from src.features.team_features import TeamFeatures
from src.features.h2h_features import H2HFeatures
from src.features.injury_features import InjuryFeatures
from src.scrapers.news_scraper import NewsScraper
from src.data.models import Match
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()


class FeatureEngineer:
    """Combines all feature sources into a unified feature vector for predictions."""

    def __init__(self):
        self.team_features = TeamFeatures()
        self.h2h_features = H2HFeatures()
        self.injury_features = InjuryFeatures()
        self.news_scraper = NewsScraper()
        self.db = get_db()

    async def create_features(self, match_id: int) -> dict:
        """Build complete feature dictionary for a match.

        Args:
            match_id: Match database ID

        Returns:
            Dictionary containing all features for the match
        """
        with self.db.get_session() as session:
            match = session.get(Match, match_id)
            if not match:
                logger.error(f"Match {match_id} not found")
                return {}

            home_id = match.home_team_id
            away_id = match.away_team_id
            league = match.league or ""

        features = {}

        # 1. Team form features (overall, home, away)
        home_form_all = self.team_features.get_form_features(home_id, 5, "all")
        home_form_home = self.team_features.get_form_features(home_id, 5, "home")
        away_form_all = self.team_features.get_form_features(away_id, 5, "all")
        away_form_away = self.team_features.get_form_features(away_id, 5, "away")

        features.update(self._prefix_dict(home_form_all, "home_overall_"))
        features.update(self._prefix_dict(home_form_home, "home_home_"))
        features.update(self._prefix_dict(away_form_all, "away_overall_"))
        features.update(self._prefix_dict(away_form_away, "away_away_"))

        # 2. H2H features
        h2h = self.h2h_features.get_h2h_features(home_id, away_id)
        features.update(h2h)

        # 3. Injury features
        home_injuries = self.injury_features.get_injury_features(home_id)
        away_injuries = self.injury_features.get_injury_features(away_id)
        features.update(self._prefix_dict(home_injuries, "home_injury_"))
        features.update(self._prefix_dict(away_injuries, "away_injury_"))

        # 4. League position features
        home_pos = self.team_features.get_league_position(home_id, league)
        away_pos = self.team_features.get_league_position(away_id, league)
        features.update(self._prefix_dict(home_pos, "home_league_"))
        features.update(self._prefix_dict(away_pos, "away_league_"))

        # Position difference
        features["position_difference"] = (
            home_pos.get("league_position", 0) - away_pos.get("league_position", 0)
        )

        # 5. News sentiment features
        home_sentiment = await self.news_scraper.get_team_sentiment(home_id)
        away_sentiment = await self.news_scraper.get_team_sentiment(away_id)
        features["home_news_sentiment"] = home_sentiment.get("avg_sentiment", 0)
        features["away_news_sentiment"] = away_sentiment.get("avg_sentiment", 0)
        features["home_news_count"] = home_sentiment.get("article_count", 0)
        features["away_news_count"] = away_sentiment.get("article_count", 0)

        logger.debug(f"Generated {len(features)} features for match {match_id}")
        return features

    def create_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to a numeric numpy array for ML models.

        Non-numeric features (strings, booleans) are converted appropriately.
        """
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, bool):
                numeric_features[key] = float(value)
            elif isinstance(value, (int, float)):
                numeric_features[key] = float(value)
            # Skip string features like form_string

        return np.array(list(numeric_features.values()))

    def get_feature_names(self, features: dict) -> list:
        """Get ordered list of numeric feature names (matches create_feature_vector order)."""
        return [
            key for key, value in features.items()
            if isinstance(value, (int, float, bool))
        ]

    def _prefix_dict(self, d: dict, prefix: str) -> dict:
        """Add a prefix to all dictionary keys."""
        return {f"{prefix}{k}": v for k, v in d.items()}
