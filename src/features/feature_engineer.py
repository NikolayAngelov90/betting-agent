"""Main feature engineering pipeline that combines all feature sources."""

import numpy as np
from datetime import timedelta
from typing import Optional

from sqlalchemy import or_

from src.features.team_features import TeamFeatures
from src.features.h2h_features import H2HFeatures
from src.features.injury_features import InjuryFeatures
from src.scrapers.news_scraper import NewsScraper
from src.data.models import Match, Odds
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

        # Stakes differentials (relegation pressure, title race distance)
        features["relegation_gap_diff"] = (
            home_pos.get("relegation_gap", 0) - away_pos.get("relegation_gap", 0)
        )
        features["title_gap_diff"] = (
            home_pos.get("title_gap", 0) - away_pos.get("title_gap", 0)
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

        # 10. RSI + MACD momentum indicators
        home_mom = self.team_features.get_momentum_indicators(home_id)
        away_mom = self.team_features.get_momentum_indicators(away_id)
        features.update(self._prefix_dict(home_mom, "home_"))
        features.update(self._prefix_dict(away_mom, "away_"))
        features["rsi_diff"] = home_mom["rsi"] - away_mom["rsi"]
        features["macd_diff"] = home_mom["macd"] - away_mom["macd"]

        # 11. Bookmaker implied probability (Bet365/Pinnacle 1X2 odds already in DB)
        bk_features = self._get_bookmaker_features(match_id)
        features.update(bk_features)

        # 12. Situational context: rest days + midweek flag
        with self.db.get_session() as session:
            match_obj = session.get(Match, match_id)
            if match_obj:
                _match_date = match_obj.match_date
                _home_id = match_obj.home_team_id
                _away_id = match_obj.away_team_id
            else:
                _match_date = None
                _home_id = home_id
                _away_id = away_id
        if _match_date:
            home_sit = self._get_situational_features(_home_id, _match_date)
            away_sit = self._get_situational_features(_away_id, _match_date)
            features["home_rest_days"] = home_sit["rest_days"]
            features["away_rest_days"] = away_sit["rest_days"]
            features["home_midweek_flag"] = home_sit["midweek_flag"]
            features["away_midweek_flag"] = away_sit["midweek_flag"]
            features["rest_days_diff"] = home_sit["rest_days"] - away_sit["rest_days"]

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

    def _get_bookmaker_features(self, match_id: int) -> dict:
        """Return margin-adjusted implied probabilities from Bet365/Pinnacle 1X2 odds."""
        defaults = {
            "home_implied_prob": 1/3,
            "draw_implied_prob": 1/3,
            "away_implied_prob": 1/3,
            "bookmaker_available": 0,
        }
        try:
            with self.db.get_session() as session:
                rows = session.query(Odds).filter(
                    Odds.match_id == match_id,
                    Odds.market_type == "1X2",
                ).all()

                if not rows:
                    return defaults

                # Priority: Bet365 first, then Pinnacle, then any
                bk_map: dict[str, dict[str, float]] = {}
                for row in rows:
                    bk = row.bookmaker
                    if bk not in bk_map:
                        bk_map[bk] = {}
                    bk_map[bk][row.selection] = row.odds_value

                odds_dict = None
                for preferred in ("Bet365", "Pinnacle"):
                    if preferred in bk_map:
                        odds_dict = bk_map[preferred]
                        break
                if odds_dict is None:
                    odds_dict = next(iter(bk_map.values()))

                # Support both API-Football ("Home"/"Away") and Flashscore ("Home Win"/"Away Win")
                h_odds = odds_dict.get("Home") or odds_dict.get("Home Win")
                d_odds = odds_dict.get("Draw")
                a_odds = odds_dict.get("Away") or odds_dict.get("Away Win")

                if not all([h_odds, d_odds, a_odds]):
                    return defaults

                raw_h = 1 / h_odds
                raw_d = 1 / d_odds
                raw_a = 1 / a_odds
                margin = raw_h + raw_d + raw_a

                return {
                    "home_implied_prob": round(raw_h / margin, 4),
                    "draw_implied_prob": round(raw_d / margin, 4),
                    "away_implied_prob": round(raw_a / margin, 4),
                    "bookmaker_available": 1,
                }
        except Exception as e:
            logger.warning(f"Bookmaker features failed for match {match_id}: {e}")
            return defaults

    def _get_situational_features(self, team_id: int, match_date) -> dict:
        """Return rest days and midweek flag based on team's previous match date."""
        defaults = {"rest_days": 7, "midweek_flag": 0}
        try:
            with self.db.get_session() as session:
                prev = session.query(Match).filter(
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.match_date < match_date,
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                ).order_by(Match.match_date.desc()).limit(1).first()

                if not prev:
                    return defaults

                delta = (match_date - prev.match_date).days
                rest_days = min(delta, 21)
                # Midweek = previous match was Tue(1), Wed(2), or Thu(3)
                midweek_flag = 1 if prev.match_date.weekday() in (1, 2, 3) else 0

                return {"rest_days": rest_days, "midweek_flag": midweek_flag}
        except Exception as e:
            logger.warning(f"Situational features failed for team {team_id}: {e}")
            return defaults

    def _prefix_dict(self, d: dict, prefix: str) -> dict:
        """Add a prefix to all dictionary keys."""
        return {f"{prefix}{k}": v for k, v in d.items()}
