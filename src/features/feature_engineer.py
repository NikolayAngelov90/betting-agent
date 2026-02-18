"""Main feature engineering pipeline that combines all feature sources."""

import numpy as np
from typing import Optional

from sqlalchemy import or_

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
            referee = match.referee or ""

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

        # 6. International competition features (CL/EL/ECL form)
        home_intl = self.team_features.get_international_form(home_id)
        away_intl = self.team_features.get_international_form(away_id)
        features.update(self._prefix_dict(home_intl, "home_"))
        features.update(self._prefix_dict(away_intl, "away_"))

        # Flag if current match is an international competition
        is_international = league in self.team_features.INTERNATIONAL_LEAGUES
        features["is_international_match"] = int(is_international)

        # International experience differential
        features["intl_experience_diff"] = home_intl["intl_matches"] - away_intl["intl_matches"]
        features["intl_quality_diff"] = home_intl["intl_points_per_match"] - away_intl["intl_points_per_match"]

        # 7. xG-based features (from API-Football)
        home_xg = self._get_xg_features(home_id, "home")
        away_xg = self._get_xg_features(away_id, "away")
        features.update(self._prefix_dict(home_xg, "home_"))
        features.update(self._prefix_dict(away_xg, "away_"))

        # xG differentials
        features["xg_for_diff"] = home_xg.get("xg_avg", 0) - away_xg.get("xg_avg", 0)
        features["xg_against_diff"] = home_xg.get("xg_against_avg", 0) - away_xg.get("xg_against_avg", 0)

        # 8. Extended statistics features (from Flashscore — rolling averages)
        home_da = home_form_all.get("dangerous_attacks_per_game_avg", 0)
        away_da = away_form_all.get("dangerous_attacks_per_game_avg", 0)
        features["home_dangerous_attacks_avg"] = home_da
        features["away_dangerous_attacks_avg"] = away_da
        features["dangerous_attacks_diff"] = home_da - away_da

        home_sv = home_form_all.get("saves_per_game_avg", 0)
        away_sv = away_form_all.get("saves_per_game_avg", 0)
        features["home_saves_avg"] = home_sv
        features["away_saves_avg"] = away_sv
        features["saves_diff"] = home_sv - away_sv  # positive = home GK faces more shots

        home_off = home_form_all.get("offsides_per_game_avg", 0)
        away_off = away_form_all.get("offsides_per_game_avg", 0)
        features["home_offsides_avg"] = home_off
        features["away_offsides_avg"] = away_off
        features["offsides_diff"] = home_off - away_off  # proxy for pressing line height

        # 9. Referee features (from Flashscore — if referee is known for this fixture)
        ref_features = self._get_referee_features(referee)
        features.update(ref_features)

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

    def _get_xg_features(self, team_id: int, venue: str = "all",
                          num_matches: int = 10) -> dict:
        """Calculate xG-based features for a team from recent matches.

        Returns rolling averages for xG for/against and overperformance.
        """
        empty = {
            "xg_avg": 0.0, "xg_against_avg": 0.0,
            "xg_overperformance": 0.0, "xg_matches": 0,
        }

        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.home_xg.isnot(None),
            )

            if venue == "home":
                query = query.filter(Match.home_team_id == team_id)
            elif venue == "away":
                query = query.filter(Match.away_team_id == team_id)
            else:
                query = query.filter(
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id)
                )

            matches = query.order_by(Match.match_date.desc()).limit(num_matches).all()

            if not matches:
                return empty

            # Extract data within session context to avoid detached instance errors
            xg_for_list = []
            xg_against_list = []
            goals_for_list = []

            for m in matches:
                is_home = m.home_team_id == team_id
                if is_home:
                    xg_for_list.append(m.home_xg or 0)
                    xg_against_list.append(m.away_xg or 0)
                    goals_for_list.append(m.home_goals or 0)
                else:
                    xg_for_list.append(m.away_xg or 0)
                    xg_against_list.append(m.home_xg or 0)
                    goals_for_list.append(m.away_goals or 0)

        xg_avg = sum(xg_for_list) / len(xg_for_list)
        xg_against_avg = sum(xg_against_list) / len(xg_against_list)
        goals_avg = sum(goals_for_list) / len(goals_for_list)

        return {
            "xg_avg": round(xg_avg, 3),
            "xg_against_avg": round(xg_against_avg, 3),
            "xg_overperformance": round(goals_avg - xg_avg, 3),
            "xg_matches": len(xg_for_list),
        }

    def _get_referee_features(self, referee: str) -> dict:
        """Get historical statistics for a referee across their last 30 matches.

        Returns metrics that inform card/goal probability (referee strictness, pace of play).
        Returns zero-defaults when referee is unknown or has no history.
        """
        empty = {
            "referee_cards_per_match_avg": 0.0,
            "referee_fouls_per_match_avg": 0.0,
            "referee_goals_per_match_avg": 0.0,
            "referee_matches": 0,
        }
        if not referee:
            return empty

        with self.db.get_session() as session:
            matches = session.query(Match).filter(
                Match.referee == referee,
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
            ).order_by(Match.match_date.desc()).limit(30).all()

            if not matches:
                return empty

            cards_list = []
            fouls_total = 0
            fouls_matches = 0
            goals_list = []

            for m in matches:
                yc = (m.home_yellow_cards or 0) + (m.away_yellow_cards or 0)
                rc = (m.home_red_cards or 0) + (m.away_red_cards or 0)
                cards_list.append(yc + rc)
                goals_list.append((m.home_goals or 0) + (m.away_goals or 0))
                hf = m.home_fouls or 0
                af = m.away_fouls or 0
                if hf > 0 or af > 0:
                    fouls_total += hf + af
                    fouls_matches += 1

            n = len(matches)

        return {
            "referee_cards_per_match_avg": round(sum(cards_list) / n, 2),
            "referee_fouls_per_match_avg": round(fouls_total / fouls_matches, 2) if fouls_matches else 0.0,
            "referee_goals_per_match_avg": round(sum(goals_list) / n, 2),
            "referee_matches": n,
        }

    def _prefix_dict(self, d: dict, prefix: str) -> dict:
        """Add a prefix to all dictionary keys."""
        return {f"{prefix}{k}": v for k, v in d.items()}
