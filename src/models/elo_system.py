"""Elo rating system for football teams."""

import math
from datetime import date
from typing import Dict, Tuple

from src.data.models import Match
from src.data.database import get_db
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()

DEFAULT_ELO = 1500
K_FACTOR = 32
HOME_ADVANTAGE = 65


class EloRatingSystem:
    """Maintains and updates Elo ratings for all teams.

    Uses home advantage factor and calculates win/draw/lose probabilities
    from rating differences.
    """

    def __init__(self, k_factor: int = K_FACTOR, home_advantage: int = HOME_ADVANTAGE):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[int, float] = {}
        self.history: Dict[int, list] = {}

    def fit(self, league: str = None):
        """Build Elo ratings by processing all historical matches chronologically.

        Applies between-season regression toward the mean so that stale ratings
        from years ago don't dominate. When the year changes between consecutive
        matches, all ratings are regressed toward DEFAULT_ELO by a configurable
        factor (default 1/3 — e.g. a team rated 1800 becomes 1700).

        Args:
            league: Optional league filter
        """
        config = get_config()
        regression_factor = config.get("models.elo_season_regression", 0.33)

        db = get_db()
        with db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
            )
            if league:
                query = query.filter(Match.league == league)

            matches = query.order_by(Match.match_date.asc()).all()

            prev_year = None
            for match in matches:
                # Between-season regression toward the mean
                m_date = match.match_date
                m_year = m_date.year if m_date else None
                if prev_year is not None and m_year is not None and m_year > prev_year:
                    for team_id in list(self.ratings.keys()):
                        self.ratings[team_id] = (
                            self.ratings[team_id] * (1 - regression_factor)
                            + DEFAULT_ELO * regression_factor
                        )
                if m_year is not None:
                    prev_year = m_year

                self._process_match(
                    match.home_team_id, match.away_team_id,
                    match.home_goals, match.away_goals,
                )

        logger.info(f"Elo ratings calculated for {len(self.ratings)} teams")

    def get_rating(self, team_id: int) -> float:
        """Get current Elo rating for a team."""
        return self.ratings.get(team_id, DEFAULT_ELO)

    def predict(self, home_team_id: int, away_team_id: int) -> Dict:
        """Predict match outcome probabilities from Elo ratings.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID

        Returns:
            Dictionary with outcome probabilities
        """
        home_elo = self.get_rating(home_team_id) + self.home_advantage
        away_elo = self.get_rating(away_team_id)

        home_win_prob, draw_prob, away_win_prob = self._calculate_probabilities(
            home_elo, away_elo
        )

        return {
            "home_win": round(home_win_prob, 4),
            "draw": round(draw_prob, 4),
            "away_win": round(away_win_prob, 4),
            "home_elo": round(self.get_rating(home_team_id), 1),
            "away_elo": round(self.get_rating(away_team_id), 1),
            "elo_difference": round(home_elo - away_elo, 1),
            "model": "elo",
        }

    def _process_match(self, home_id: int, away_id: int,
                       home_goals: int, away_goals: int):
        """Update Elo ratings based on a match result."""
        home_elo = self.ratings.get(home_id, DEFAULT_ELO) + self.home_advantage
        away_elo = self.ratings.get(away_id, DEFAULT_ELO)

        # Expected scores
        home_expected = self._expected_score(home_elo, away_elo)
        away_expected = 1 - home_expected

        # Actual scores (1 = win, 0.5 = draw, 0 = loss)
        if home_goals > away_goals:
            home_actual, away_actual = 1.0, 0.0
        elif home_goals == away_goals:
            home_actual, away_actual = 0.5, 0.5
        else:
            home_actual, away_actual = 0.0, 1.0

        # Goal difference multiplier
        gd = abs(home_goals - away_goals)
        gd_multiplier = math.log(max(gd, 1) + 1)

        # Update ratings
        home_new = self.ratings.get(home_id, DEFAULT_ELO) + \
                   self.k_factor * gd_multiplier * (home_actual - home_expected)
        away_new = self.ratings.get(away_id, DEFAULT_ELO) + \
                   self.k_factor * gd_multiplier * (away_actual - away_expected)

        self.ratings[home_id] = home_new
        self.ratings[away_id] = away_new

        # Track history
        self.history.setdefault(home_id, []).append(home_new)
        self.history.setdefault(away_id, []).append(away_new)

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using logistic formula."""
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _calculate_probabilities(self, home_elo: float,
                                  away_elo: float) -> Tuple[float, float, float]:
        """Convert Elo ratings to 1X2 probabilities.

        Uses a calibrated approach where draw probability is estimated
        from the rating difference.
        """
        diff = home_elo - away_elo
        home_expected = self._expected_score(home_elo, away_elo)

        # Draw probability decreases as rating difference increases
        draw_prob = max(0.15, 0.28 - abs(diff) / 2000.0)

        # Distribute remaining probability
        remaining = 1.0 - draw_prob
        home_win = remaining * home_expected
        away_win = remaining * (1 - home_expected)

        return home_win, draw_prob, away_win
