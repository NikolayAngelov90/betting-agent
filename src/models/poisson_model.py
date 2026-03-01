"""Poisson distribution model for goal prediction."""

import numpy as np
from datetime import date
from scipy.stats import poisson
from typing import Dict, Tuple

from src.data.models import Match
from src.data.database import get_db
from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()

MAX_GOALS = 10  # Maximum goals to consider in probability calculations


class PoissonModel:
    """Poisson-based goal prediction model with Dixon-Coles correction and time decay.

    Calculates attack/defense strength for each team (with exponential time decay
    on older matches) and uses a Dixon-Coles corrected Poisson distribution to
    predict expected goals and match outcome probabilities.

    Key enhancements vs. plain independent Poisson:
    - **Time decay**: recent matches weighted more heavily (configurable half-life).
    - **Dixon-Coles correction**: adjusts joint probability of low-scoring outcomes
      (0-0, 1-0, 0-1, 1-1) using correction factor τ with parameter ρ ≈ -0.13,
      boosting 0-0 and 1-1 while slightly reducing 1-0 and 0-1 probability.
    """

    def __init__(self):
        self.db = get_db()
        self.config = get_config()
        self.league_avg_home_goals = 1.5
        self.league_avg_away_goals = 1.2
        self._team_strengths = {}
        self._team_league = {}  # team_id -> league string
        self._league_avgs = {}  # league -> {"home": float, "away": float}
        self._league_rhos = {}  # league -> optimized Dixon-Coles rho

    def fit(self, league: str = None, num_matches: int = 5000):
        """Calculate time-decayed attack/defense strength ratings from historical data.

        Applies exponential decay so recent matches count more than older ones.
        Half-life is configurable via models.strength_half_life_days (default: 180 days).

        Args:
            league: Optional league filter
            num_matches: Number of recent matches to use
        """
        half_life = self.config.get("models.strength_half_life_days", 180)
        decay_rate = np.log(2) / max(half_life, 1)
        today = date.today()

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

            # Compute time-decay weight per match
            def match_weight(m):
                if m.match_date is None:
                    return 0.1
                m_date = m.match_date.date() if hasattr(m.match_date, "date") else m.match_date
                days = max(0, (today - m_date).days)
                return float(np.exp(-decay_rate * days))

            weights = [match_weight(m) for m in matches]

            # Weighted global averages
            home_goals_arr = np.array([m.home_goals for m in matches], dtype=float)
            away_goals_arr = np.array([m.away_goals for m in matches], dtype=float)
            w = np.array(weights)
            w_sum = w.sum()
            if w_sum > 0:
                self.league_avg_home_goals = float(np.dot(w, home_goals_arr) / w_sum)
                self.league_avg_away_goals = float(np.dot(w, away_goals_arr) / w_sum)
            else:
                self.league_avg_home_goals = float(np.mean(home_goals_arr))
                self.league_avg_away_goals = float(np.mean(away_goals_arr))

            # Weighted per-league averages for calibration
            league_goals: dict = {}
            for m, wt in zip(matches, weights):
                lg = m.league or "unknown"
                league_goals.setdefault(lg, {"home": [], "away": [], "w": []})
                league_goals[lg]["home"].append(m.home_goals)
                league_goals[lg]["away"].append(m.away_goals)
                league_goals[lg]["w"].append(wt)
            for lg, gdata in league_goals.items():
                if len(gdata["home"]) >= 30:
                    lw = np.array(gdata["w"])
                    lw_sum = lw.sum()
                    if lw_sum > 0:
                        self._league_avgs[lg] = {
                            "home": float(np.dot(lw, gdata["home"]) / lw_sum),
                            "away": float(np.dot(lw, gdata["away"]) / lw_sum),
                        }

            # Per-team attack and defense strengths (time-weighted)
            team_stats: dict = {}
            for match, wt in zip(matches, weights):
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
                            "home_w": [], "away_w": [],
                        }
                    team_stats[team_id][f"{venue}_scored"].append(scored)
                    team_stats[team_id][f"{venue}_conceded"].append(conceded)
                    team_stats[team_id][f"{venue}_w"].append(wt)

            def _wavg(vals, wts, fallback):
                if not vals:
                    return fallback
                wts_arr = np.array(wts)
                ws = wts_arr.sum()
                if ws <= 0:
                    return float(np.mean(vals))
                return float(np.dot(wts_arr, vals) / ws)

            for team_id, stats in team_stats.items():
                home_scored_avg = _wavg(stats["home_scored"], stats["home_w"], self.league_avg_home_goals)
                away_scored_avg = _wavg(stats["away_scored"], stats["away_w"], self.league_avg_away_goals)
                home_conceded_avg = _wavg(stats["home_conceded"], stats["home_w"], self.league_avg_away_goals)
                away_conceded_avg = _wavg(stats["away_conceded"], stats["away_w"], self.league_avg_home_goals)

                attack_strength = (
                    (home_scored_avg / self.league_avg_home_goals) +
                    (away_scored_avg / self.league_avg_away_goals)
                ) / 2

                defense_strength = (
                    (home_conceded_avg / self.league_avg_away_goals) +
                    (away_conceded_avg / self.league_avg_home_goals)
                ) / 2

                # Bayesian shrinkage: regress toward league-average (1.0) for
                # teams with few observations.  A team with < shrinkage_cap
                # matches is pulled toward the prior proportionally.
                shrinkage_cap = self.config.get("models.shrinkage_sample_cap", 100)
                total_matches = (len(stats["home_scored"])
                                 + len(stats["away_scored"]))
                shrinkage = min(total_matches, shrinkage_cap) / shrinkage_cap
                attack_strength = attack_strength * shrinkage + 1.0 * (1 - shrinkage)
                defense_strength = defense_strength * shrinkage + 1.0 * (1 - shrinkage)

                # Floor to prevent zero xG (e.g. team conceded 0 in small sample)
                self._team_strengths[team_id] = {
                    "attack": max(attack_strength, 0.15),
                    "defense": max(defense_strength, 0.15),
                }

            # Estimate per-league Dixon-Coles rho via MLE
            self._estimate_league_rhos(matches)

            logger.info(
                f"Poisson model fitted: {len(self._team_strengths)} teams, "
                f"{len(self._league_avgs)} leagues calibrated, "
                f"{len(self._league_rhos)} per-league rhos estimated, "
                f"avg home goals={self.league_avg_home_goals:.2f}, "
                f"avg away goals={self.league_avg_away_goals:.2f} "
                f"(time-decay half-life={half_life}d)"
            )

    def _estimate_league_rhos(self, matches):
        """Estimate per-league Dixon-Coles rho via maximum likelihood.

        For each league with enough matches, grid-searches the rho value in
        [-0.25, 0.05] that maximises the log-likelihood of observed scores
        given each team's Poisson xG.
        """
        from scipy.optimize import minimize_scalar

        min_matches = self.config.get("models.dc_rho_min_matches", 50)
        default_rho = self.config.get("models.dixon_coles_rho", -0.13)

        # Group matches by league
        league_matches: dict = {}
        for m in matches:
            lg = m.league or "unknown"
            league_matches.setdefault(lg, []).append(m)

        for lg, lg_matches in league_matches.items():
            if len(lg_matches) < min_matches:
                continue

            # Pre-compute xG for each match (need both team strengths known)
            match_data = []
            for m in lg_matches:
                hs = self._team_strengths.get(m.home_team_id)
                aws = self._team_strengths.get(m.away_team_id)
                if hs is None or aws is None:
                    continue
                home_xg, away_xg = self._expected_goals(m.home_team_id, m.away_team_id)
                if home_xg <= 0 or away_xg <= 0:
                    continue
                match_data.append((m.home_goals, m.away_goals, home_xg, away_xg))

            if len(match_data) < min_matches:
                continue

            def neg_log_likelihood(rho):
                ll = 0.0
                for hg, ag, lam, mu in match_data:
                    # Independent Poisson probability
                    p = poisson.pmf(hg, lam) * poisson.pmf(ag, mu)
                    # Dixon-Coles correction
                    tau = self._dc_tau(hg, ag, lam, mu, rho)
                    p *= tau
                    if p > 0:
                        ll += np.log(p)
                    else:
                        ll += -50  # penalty for zero probability
                return -ll

            try:
                result = minimize_scalar(
                    neg_log_likelihood,
                    bounds=(-0.25, 0.05),
                    method="bounded",
                )
                if result.success:
                    self._league_rhos[lg] = round(float(result.x), 4)
                else:
                    self._league_rhos[lg] = default_rho
            except Exception:
                self._league_rhos[lg] = default_rho

        if self._league_rhos:
            logger.debug(f"Per-league DC rhos: {self._league_rhos}")

    def predict(self, home_team_id: int, away_team_id: int,
                league: str = None) -> Dict:
        """Predict match outcomes using Poisson model.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID

        Returns:
            Dictionary with predicted probabilities for all markets
        """
        home_xg, away_xg = self._expected_goals(home_team_id, away_team_id)

        # Resolve league from team mapping if not provided
        if league is None:
            league = self._team_league.get(
                home_team_id, self._team_league.get(away_team_id))

        # Score probability matrix (with Dixon-Coles correction applied)
        score_matrix = self._score_matrix(home_xg, away_xg, league=league)

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

        # Team goal line: P(team scores 2+ goals)
        home_over_15 = self._team_over_prob(score_matrix, 1.5, side="home")
        away_over_15 = self._team_over_prob(score_matrix, 1.5, side="away")

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
            "home_over_1.5": round(home_over_15, 4),
            "away_over_1.5": round(away_over_15, 4),
            "most_likely_score": self._most_likely_score(score_matrix),
            "model": "poisson",
        }

    def _expected_goals(self, home_team_id: int, away_team_id: int) -> Tuple[float, float]:
        """Calculate expected goals for each team.

        For unknown teams, regress toward league average with slight randomisation
        based on the team ID so that different unknown teams still get different
        predictions instead of all converging to the same xG.
        """
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

    def _dc_tau(self, x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Dixon-Coles low-score correction factor τ(x, y).

        Adjusts the joint probability of low-scoring outcomes to correct the
        known under-estimation of 0-0 and 1-1 results by independent Poisson.

        Reference: Dixon & Coles (1997), "Modelling Association Football Scores
        and Inefficiencies in the Football Betting Market."

        With ρ < 0 (e.g. -0.13):
          - τ(0,0) > 1  — 0-0 draws are more likely than independent Poisson predicts
          - τ(1,1) > 1  — 1-1 draws are more likely
          - τ(1,0) < 1  — 1-0 home wins are slightly less likely
          - τ(0,1) < 1  — 0-1 away wins are slightly less likely
        """
        if x == 0 and y == 0:
            return 1.0 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1.0 + lam * rho
        elif x == 1 and y == 0:
            return 1.0 + mu * rho
        elif x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _score_matrix(self, home_xg: float, away_xg: float,
                      league: str = None) -> np.ndarray:
        """Generate a probability matrix for all scoreline combinations.

        Applies Dixon-Coles correction to low-scoring outcomes to address the
        known under-estimation of 0-0 and 1-1 by the independent Poisson model.
        Uses per-league rho when available, falls back to global config.
        Set models.dixon_coles_rho=0.0 in config to disable the correction.
        """
        home_probs = [poisson.pmf(i, home_xg) for i in range(MAX_GOALS)]
        away_probs = [poisson.pmf(i, away_xg) for i in range(MAX_GOALS)]
        matrix = np.outer(home_probs, away_probs)

        default_rho = self.config.get("models.dixon_coles_rho", -0.13)
        rho = self._league_rhos.get(league, default_rho) if league else default_rho
        if rho != 0.0:
            # Apply correction to the 4 low-score cells
            for i in range(min(2, MAX_GOALS)):
                for j in range(min(2, MAX_GOALS)):
                    matrix[i, j] *= self._dc_tau(i, j, home_xg, away_xg, rho)
            # Renormalize so probabilities still sum to 1
            total = matrix.sum()
            if total > 0:
                matrix /= total

        return matrix

    def _over_under_prob(self, matrix: np.ndarray, threshold: float) -> float:
        """Calculate probability of total goals being over a threshold."""
        prob = 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if i + j > threshold:
                    prob += matrix[i][j]
        return prob

    def _team_over_prob(self, matrix: np.ndarray, threshold: float,
                        side: str = "home") -> float:
        """Calculate probability that a specific team scores more than threshold goals.

        Args:
            matrix: Score probability matrix (home goals × away goals)
            threshold: Goal threshold (e.g. 1.5 means the team scores 2+)
            side: 'home' or 'away'
        """
        prob = 0.0
        if side == "home":
            for i in range(matrix.shape[0]):
                if i > threshold:
                    prob += float(np.sum(matrix[i, :]))
        else:
            for j in range(matrix.shape[1]):
                if j > threshold:
                    prob += float(np.sum(matrix[:, j]))
        return prob

    def _btts_prob(self, matrix: np.ndarray) -> float:
        """Calculate probability of both teams scoring."""
        # Exclude rows/cols where either team scores 0
        return float(np.sum(matrix[1:, 1:]))

    def _most_likely_score(self, matrix: np.ndarray) -> str:
        """Find the most likely scoreline."""
        idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        return f"{idx[0]}-{idx[1]}"
