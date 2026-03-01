"""Head-to-head feature calculations."""

from typing import List

from sqlalchemy import or_, and_

from src.data.models import Match
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()


class H2HFeatures:
    """Calculates head-to-head statistics between two teams."""

    def __init__(self):
        self.db = get_db()

    def get_h2h_features(self, home_team_id: int, away_team_id: int,
                         limit: int = 10, as_of_date=None) -> dict:
        """Calculate H2H features between two teams.

        Args:
            home_team_id: Home team database ID
            away_team_id: Away team database ID
            limit: Number of past meetings to analyze
            as_of_date: Only use matches before this date (for training).
                        None = no cutoff (live prediction).

        Returns:
            Dictionary of H2H features
        """
        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                or_(
                    and_(Match.home_team_id == home_team_id, Match.away_team_id == away_team_id),
                    and_(Match.home_team_id == away_team_id, Match.away_team_id == home_team_id),
                ),
            )
            if as_of_date is not None:
                query = query.filter(Match.match_date < as_of_date)
            matches = query.order_by(Match.match_date.desc()).limit(limit).all()

            if not matches:
                return self._empty_h2h()

            return self._calculate_h2h(matches, home_team_id, away_team_id)

    def _calculate_h2h(self, matches: List[Match], home_team_id: int,
                       away_team_id: int) -> dict:
        """Calculate H2H statistics from historical matches.

        'home' and 'away' here refer to the teams in the *upcoming* fixture,
        not necessarily who was at home in each historical meeting.
        """
        total = len(matches)
        home_wins = 0  # wins by home_team_id (in upcoming fixture)
        away_wins = 0
        draws = 0
        total_goals = 0
        btts_count = 0
        over_25_count = 0
        home_team_scored_count = 0
        away_team_scored_count = 0

        for match in matches:
            hg = match.home_goals or 0
            ag = match.away_goals or 0
            match_total_goals = hg + ag
            total_goals += match_total_goals

            # Determine which team is "ours" (home_team_id in the upcoming fixture)
            if match.home_team_id == home_team_id:
                our_goals = hg
                their_goals = ag
            else:
                our_goals = ag
                their_goals = hg

            if our_goals > their_goals:
                home_wins += 1
            elif our_goals < their_goals:
                away_wins += 1
            else:
                draws += 1

            if our_goals > 0:
                home_team_scored_count += 1
            if their_goals > 0:
                away_team_scored_count += 1

            if hg > 0 and ag > 0:
                btts_count += 1
            if match_total_goals > 2.5:
                over_25_count += 1

        return {
            "h2h_total_meetings": total,
            "h2h_home_wins": home_wins,
            "h2h_away_wins": away_wins,
            "h2h_draws": draws,
            "h2h_home_win_pct": round(home_wins / total, 3) if total else 0,
            "h2h_away_win_pct": round(away_wins / total, 3) if total else 0,
            "h2h_draw_pct": round(draws / total, 3) if total else 0,
            "h2h_avg_total_goals": round(total_goals / total, 2) if total else 0,
            "h2h_btts_percentage": round(btts_count / total, 3) if total else 0,
            "h2h_over_25_percentage": round(over_25_count / total, 3) if total else 0,
            "h2h_home_team_scored_pct": round(home_team_scored_count / total, 3) if total else 0,
            "h2h_away_team_scored_pct": round(away_team_scored_count / total, 3) if total else 0,
        }

    def _empty_h2h(self) -> dict:
        return {
            "h2h_total_meetings": 0,
            "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0,
            "h2h_home_win_pct": 0, "h2h_away_win_pct": 0, "h2h_draw_pct": 0,
            "h2h_avg_total_goals": 0,
            "h2h_btts_percentage": 0, "h2h_over_25_percentage": 0,
            "h2h_home_team_scored_pct": 0, "h2h_away_team_scored_pct": 0,
        }
