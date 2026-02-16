"""Poisson distribution model for goal prediction."""

import numpy as np
from scipy.stats import poisson
from typing import Dict, Tuple

from src.data.models import Match
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

MAX_GOALS = 10  # Maximum goals to consider in probability calculations


class PoissonModel:
    """Poisson-based goal prediction model.

    Calculates attack/defense strength for each team and uses Poisson
    distribution to predict expected goals and match outcome probabilities.
    """

    def __init__(self):
        self.db = get_db()
        self.league_avg_home_goals = 1.5
        self.league_avg_away_goals = 1.2
        self._team_strengths = {}
        self._team_league = {}  # team_id -> league string
        self._league_avgs = {}  # league -> {"home": float, "away": float}

    def fit(self, league: str = None, num_matches: int = 5000):
        """Calculate attack/defense strength ratings from historical data.

        Args:
            league: Optional league filter
            num_matches: Number of recent matches to use
        """
        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
            )
            if league:
                query = query.filter(Match.league == league)

            matches = query.order_by(Match.match_date.desc()).limit(num_matches).all()

            if not matches:
                logger.warning("No match data available for Poisson model fitting")
                return

            # Global averages
            home_goals = [m.home_goals for m in matches]
            away_goals = [m.away_goals for m in matches]
            self.league_avg_home_goals = np.mean(home_goals)
            self.league_avg_away_goals = np.mean(away_goals)

            # Per-league averages for calibration
            league_goals = {}
            for m in matches:
                lg = m.league or "unknown"
                league_goals.setdefault(lg, {"home": [], "away": []})
                league_goals[lg]["home"].append(m.home_goals)
                league_goals[lg]["away"].append(m.away_goals)
            for lg, goals in league_goals.items():
                if len(goals["home"]) >= 30:  # only calibrate with enough data
                    self._league_avgs[lg] = {
                        "home": np.mean(goals["home"]),
                        "away": np.mean(goals["away"]),
                    }

            # Per-team attack and defense strengths
            team_stats = {}
            for match in matches:
                for team_id, scored, conceded, venue in [
                    (match.home_team_id, match.home_goals, match.away_goals, "home"),
                    (match.away_team_id, match.away_goals, match.home_goals, "away"),
                ]:
                    if match.league:
                        self._team_league[team_id] = match.league
                    if team_id not in team_stats:
                        team_stats[team_id] = {
                            "home_scored": [], "home_conceded": [],
                            "away_scored": [], "away_conceded": [],
                        }
                    team_stats[team_id][f"{venue}_scored"].append(scored)
                    team_stats[team_id][f"{venue}_conceded"].append(conceded)

            for team_id, stats in team_stats.items():
                home_scored_avg = np.mean(stats["home_scored"]) if stats["home_scored"] else self.league_avg_home_goals
                away_scored_avg = np.mean(stats["away_scored"]) if stats["away_scored"] else self.league_avg_away_goals
                home_conceded_avg = np.mean(stats["home_conceded"]) if stats["home_conceded"] else self.league_avg_away_goals
                away_conceded_avg = np.mean(stats["away_conceded"]) if stats["away_conceded"] else self.league_avg_home_goals

                attack_strength = (
                    (home_scored_avg / self.league_avg_home_goals) +
                    (away_scored_avg / self.league_avg_away_goals)
                ) / 2

                defense_strength = (
                    (home_conceded_avg / self.league_avg_away_goals) +
                    (away_conceded_avg / self.league_avg_home_goals)
                ) / 2

                # Floor to prevent zero xG (e.g. team conceded 0 in small sample)
                self._team_strengths[team_id] = {
                    "attack": max(attack_strength, 0.15),
                    "defense": max(defense_strength, 0.15),
                }

            logger.info(
                f"Poisson model fitted: {len(self._team_strengths)} teams, "
                f"{len(self._league_avgs)} leagues calibrated, "
                f"avg home goals={self.league_avg_home_goals:.2f}, "
                f"avg away goals={self.league_avg_away_goals:.2f}"
            )

    def predict(self, home_team_id: int, away_team_id: int) -> Dict:
        """Predict match outcomes using Poisson model.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID

        Returns:
            Dictionary with predicted probabilities for all markets
        """
        home_xg, away_xg = self._expected_goals(home_team_id, away_team_id)

        # Score probability matrix
        score_matrix = self._score_matrix(home_xg, away_xg)

        # Market probabilities
        home_win_prob = np.sum(np.tril(score_matrix, -1))
        draw_prob = np.sum(np.diag(score_matrix))
        away_win_prob = np.sum(np.triu(score_matrix, 1))

        # Over/Under
        over_25_prob = self._over_under_prob(score_matrix, 2.5)
        over_15_prob = self._over_under_prob(score_matrix, 1.5)
        over_35_prob = self._over_under_prob(score_matrix, 3.5)

        # BTTS
        btts_prob = self._btts_prob(score_matrix)

        return {
            "home_xg": round(home_xg, 3),
            "away_xg": round(away_xg, 3),
            "home_win": round(home_win_prob, 4),
            "draw": round(draw_prob, 4),
            "away_win": round(away_win_prob, 4),
            "over_1.5": round(over_15_prob, 4),
            "over_2.5": round(over_25_prob, 4),
            "over_3.5": round(over_35_prob, 4),
            "under_2.5": round(1 - over_25_prob, 4),
            "btts_yes": round(btts_prob, 4),
            "btts_no": round(1 - btts_prob, 4),
            "most_likely_score": self._most_likely_score(score_matrix),
            "model": "poisson",
        }

    def _expected_goals(self, home_team_id: int, away_team_id: int) -> Tuple[float, float]:
        """Calculate expected goals for each team.

        For unknown teams, regress toward league average with slight randomisation
        based on the team ID so that different unknown teams still get different
        predictions instead of all converging to the same xG.
        """
        default = {"attack": 1.0, "defense": 1.0}
        home_strength = self._team_strengths.get(home_team_id)
        away_strength = self._team_strengths.get(away_team_id)

        if home_strength is None:
            # Small deterministic offset so different unknown teams differ
            seed = (home_team_id * 2654435761) & 0xFFFFFFFF
            offset = ((seed % 1000) / 1000.0 - 0.5) * 0.3  # -0.15..+0.15
            home_strength = {"attack": 1.0 + offset, "defense": 1.0 - offset * 0.5}

        if away_strength is None:
            seed = (away_team_id * 2654435761) & 0xFFFFFFFF
            offset = ((seed % 1000) / 1000.0 - 0.5) * 0.3
            away_strength = {"attack": 1.0 + offset, "defense": 1.0 - offset * 0.5}

        # Use per-league averages when both teams share a known league
        home_lg = self._team_league.get(home_team_id)
        away_lg = self._team_league.get(away_team_id)
        if home_lg and home_lg == away_lg and home_lg in self._league_avgs:
            avg_home = self._league_avgs[home_lg]["home"]
            avg_away = self._league_avgs[home_lg]["away"]
        else:
            avg_home = self.league_avg_home_goals
            avg_away = self.league_avg_away_goals

        home_xg = avg_home * home_strength["attack"] * away_strength["defense"]
        away_xg = avg_away * away_strength["attack"] * home_strength["defense"]

        return home_xg, away_xg

    def _score_matrix(self, home_xg: float, away_xg: float) -> np.ndarray:
        """Generate a probability matrix for all scoreline combinations."""
        home_probs = [poisson.pmf(i, home_xg) for i in range(MAX_GOALS)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(MAX_GOALS)]
        return np.outer(home_probs, away_probs)

    def _over_under_prob(self, matrix: np.ndarray, threshold: float) -> float:
        """Calculate probability of total goals being over a threshold."""
        prob = 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i + j > threshold:
                    prob += matrix[i][j]
        return prob

    def _btts_prob(self, matrix: np.ndarray) -> float:
        """Calculate probability of both teams scoring."""
        # Exclude rows/cols where either team scores 0
        return float(np.sum(matrix[1:, 1:]))

    def _most_likely_score(self, matrix: np.ndarray) -> str:
        """Find the most likely scoreline."""
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        return f"{idx[0]}-{idx[1]}"
