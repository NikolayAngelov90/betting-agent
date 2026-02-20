"""Team form and league position feature calculations."""

from typing import List, Optional

from sqlalchemy import and_, or_

from src.data.models import Match, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()


class TeamFeatures:
    """Calculates team-level features: form, league position, contextual stats."""

    def __init__(self):
        self.db = get_db()

    def get_form_features(self, team_id: int, num_matches: int = 5,
                          venue: str = "all") -> dict:
        """Calculate form-based features for a team.

        Args:
            team_id: Team database ID
            num_matches: Number of recent matches to consider
            venue: 'home', 'away', or 'all'

        Returns:
            Dictionary of form features
        """
        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
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
                return self._empty_form_features()

            return self._calculate_form(matches, team_id)

    def _calculate_form(self, matches: List[Match], team_id: int) -> dict:
        """Calculate form statistics from a list of matches."""
        points = 0
        goals_scored = 0
        goals_conceded = 0
        clean_sheets = 0
        failed_to_score = 0
        wins = 0
        draws = 0
        losses = 0
        shots_total = 0
        shots_on_target_total = 0
        possession_total = 0.0
        corners_total = 0
        shots_count = 0
        possession_count = 0
        corners_count = 0
        # Extended Flashscore stats
        da_total = 0
        sv_total = 0
        off_total = 0
        fk_total = 0
        da_count = 0
        sv_count = 0
        off_count = 0
        fk_count = 0

        form_string = []
        per_match_pts = []  # for decay form score (index 0 = most recent)

        for match in matches:
            is_home = match.home_team_id == team_id

            if is_home:
                scored = match.home_goals or 0
                conceded = match.away_goals or 0
                shots = match.home_shots
                sot = match.home_shots_on_target
                poss = match.home_possession
                corners = match.home_corners
                da = match.home_dangerous_attacks
                sv = match.home_saves
                off = match.home_offsides
                fk = match.home_free_kicks
            else:
                scored = match.away_goals or 0
                conceded = match.home_goals or 0
                shots = match.away_shots
                sot = match.away_shots_on_target
                poss = match.away_possession
                corners = match.away_corners
                da = match.away_dangerous_attacks
                sv = match.away_saves
                off = match.away_offsides
                fk = match.away_free_kicks

            goals_scored += scored
            goals_conceded += conceded

            if conceded == 0:
                clean_sheets += 1
            if scored == 0:
                failed_to_score += 1

            if scored > conceded:
                points += 3
                wins += 1
                form_string.append("W")
                per_match_pts.append(3)
            elif scored == conceded:
                points += 1
                draws += 1
                form_string.append("D")
                per_match_pts.append(1)
            else:
                losses += 1
                form_string.append("L")
                per_match_pts.append(0)

            if shots is not None:
                shots_total += shots
                shots_count += 1
            if sot is not None:
                shots_on_target_total += sot
            if poss is not None:
                possession_total += poss
                possession_count += 1
            if corners is not None:
                corners_total += corners
                corners_count += 1
            if da is not None:
                da_total += da
                da_count += 1
            if sv is not None:
                sv_total += sv
                sv_count += 1
            if off is not None:
                off_total += off
                off_count += 1
            if fk is not None:
                fk_total += fk
                fk_count += 1

        n = len(matches)

        # Exponential decay form score (index 0 = most recent game, weight = 1.0)
        _decay = 0.85
        _decay_weights = [_decay ** i for i in range(len(per_match_pts))]
        _total_weight = sum(_decay_weights)
        decay_form_score = (
            round(sum(w * p for w, p in zip(_decay_weights, per_match_pts)) / _total_weight, 3)
            if _total_weight else 0.0
        )

        # Calculate streaks
        win_streak = self._calculate_streak(form_string, "W")
        losing_streak = self._calculate_streak(form_string, "L")
        unbeaten_run = self._calculate_unbeaten_run(form_string)

        return {
            "matches_played": n,
            "points": points,
            "points_per_match": round(points / n, 2) if n else 0,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_scored": goals_scored,
            "goals_conceded": goals_conceded,
            "goal_difference": goals_scored - goals_conceded,
            "goals_scored_per_match": round(goals_scored / n, 2) if n else 0,
            "goals_conceded_per_match": round(goals_conceded / n, 2) if n else 0,
            "clean_sheets": clean_sheets,
            "failed_to_score": failed_to_score,
            "win_streak": win_streak,
            "losing_streak": losing_streak,
            "unbeaten_run": unbeaten_run,
            "shots_per_game_avg": round(shots_total / shots_count, 2) if shots_count else 0,
            "shots_on_target_per_game_avg": round(shots_on_target_total / shots_count, 2) if shots_count else 0,
            "possession_avg": round(possession_total / possession_count, 2) if possession_count else 0,
            "corners_per_game_avg": round(corners_total / corners_count, 2) if corners_count else 0,
            "dangerous_attacks_per_game_avg": round(da_total / da_count, 2) if da_count else 0,
            "saves_per_game_avg": round(sv_total / sv_count, 2) if sv_count else 0,
            "offsides_per_game_avg": round(off_total / off_count, 2) if off_count else 0,
            "free_kicks_per_game_avg": round(fk_total / fk_count, 2) if fk_count else 0,
            "form_string": "-".join(form_string),
            "decay_form_score": decay_form_score,
        }

    def _calculate_streak(self, form: List[str], result: str) -> int:
        """Calculate current streak of a specific result from most recent."""
        streak = 0
        for r in form:
            if r == result:
                streak += 1
            else:
                break
        return streak

    def _calculate_unbeaten_run(self, form: List[str]) -> int:
        """Calculate current unbeaten run from most recent."""
        run = 0
        for r in form:
            if r in ("W", "D"):
                run += 1
            else:
                break
        return run

    def _empty_form_features(self) -> dict:
        """Return empty form features when no data available."""
        return {
            "matches_played": 0,
            "points": 0,
            "points_per_match": 0,
            "wins": 0, "draws": 0, "losses": 0,
            "goals_scored": 0, "goals_conceded": 0, "goal_difference": 0,
            "goals_scored_per_match": 0, "goals_conceded_per_match": 0,
            "clean_sheets": 0, "failed_to_score": 0,
            "win_streak": 0, "losing_streak": 0, "unbeaten_run": 0,
            "shots_per_game_avg": 0, "shots_on_target_per_game_avg": 0,
            "possession_avg": 0, "corners_per_game_avg": 0,
            "dangerous_attacks_per_game_avg": 0,
            "saves_per_game_avg": 0,
            "offsides_per_game_avg": 0,
            "free_kicks_per_game_avg": 0,
            "form_string": "",
            "decay_form_score": 0.0,
        }

    # Leagues classified as international competitions
    INTERNATIONAL_LEAGUES = {
        "champions-league", "europa-league", "europa-conference-league",
    }

    def get_international_form(self, team_id: int, num_matches: int = 10) -> dict:
        """Calculate form features specifically from international competition matches.

        This captures how a team performs in CL/EL/ECL — different from domestic form
        due to higher quality opposition, travel, and tactical adjustments.
        """
        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.league.in_(self.INTERNATIONAL_LEAGUES),
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            ).order_by(Match.match_date.desc()).limit(num_matches)

            matches = query.all()

            if not matches:
                return {
                    "intl_matches": 0,
                    "intl_points_per_match": 0.0,
                    "intl_goals_per_match": 0.0,
                    "intl_conceded_per_match": 0.0,
                    "intl_win_rate": 0.0,
                    "intl_clean_sheet_rate": 0.0,
                    "intl_active": 0,
                }

            form = self._calculate_form(matches, team_id)
            n = form["matches_played"]
            return {
                "intl_matches": n,
                "intl_points_per_match": form["points_per_match"],
                "intl_goals_per_match": form["goals_scored_per_match"],
                "intl_conceded_per_match": form["goals_conceded_per_match"],
                "intl_win_rate": round(form["wins"] / n, 3) if n else 0.0,
                "intl_clean_sheet_rate": round(form["clean_sheets"] / n, 3) if n else 0.0,
                "intl_active": 1,  # flag: team is in European competition
            }

    def get_momentum_indicators(self, team_id: int, num_matches: int = 14) -> dict:
        """Calculate RSI and MACD momentum indicators from recent match points.

        RSI > 70 → hot streak likely to cool; RSI < 30 → cold streak likely to bounce.
        MACD > 0 → short-term form accelerating vs longer-term baseline.
        """
        empty = {
            "rsi": 50.0,
            "macd": 0.0,
            "macd_hist": 0.0,
            "momentum_matches": 0,
        }

        with self.db.get_session() as session:
            matches = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
            ).order_by(Match.match_date.desc()).limit(num_matches).all()

            if len(matches) < 5:
                return empty

            pts = []
            for m in matches:
                is_home = m.home_team_id == team_id
                scored = (m.home_goals or 0) if is_home else (m.away_goals or 0)
                conceded = (m.away_goals or 0) if is_home else (m.home_goals or 0)
                if scored > conceded:
                    pts.append(3)
                elif scored == conceded:
                    pts.append(1)
                else:
                    pts.append(0)

        # --- RSI (gains = pts earned, losses = max pts - pts earned) ---
        n = len(pts)
        avg_gain = sum(pts) / n
        avg_loss = sum(3 - p for p in pts) / n
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = round(100 - (100 / (1 + rs)), 2)

        # --- MACD: EMA(5) - EMA(14) of points, chronological order ---
        pts_chron = list(reversed(pts))  # oldest first

        def _ema(series, period):
            if len(series) < period:
                return sum(series) / len(series)
            k = 2.0 / (period + 1)
            val = sum(series[:period]) / period  # SMA seed
            for p in series[period:]:
                val = p * k + val * (1 - k)
            return val

        fast_ema = _ema(pts_chron, 5)
        slow_ema = _ema(pts_chron, min(14, len(pts_chron)))
        macd = round(fast_ema - slow_ema, 3)

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_hist": macd,   # simplified: no separate signal line
            "momentum_matches": n,
        }

    def clear_standings_cache(self):
        """Invalidate the standings cache (call between training runs if needed)."""
        self._standings_cache: dict = {}

    def _get_league_standings(self, league: str) -> list:
        """Build (or return cached) sorted standings for a league.

        Standings are cached per-instance so the expensive multi-team form
        computation only runs once per league per TeamFeatures lifetime
        (one training run / prediction session).
        """
        if not hasattr(self, "_standings_cache"):
            self._standings_cache: dict = {}

        if league in self._standings_cache:
            return self._standings_cache[league]

        with self.db.get_session() as session:
            teams = session.query(Team).filter_by(league=league).all()
            if not teams:
                self._standings_cache[league] = []
                return []
            team_ids = [(t.id, t.name) for t in teams]

        standings = []
        for team_id, team_name in team_ids:
            form_all = self.get_form_features(team_id, num_matches=50, venue="all")
            form_home = self.get_form_features(team_id, num_matches=50, venue="home")
            form_away = self.get_form_features(team_id, num_matches=50, venue="away")

            standings.append({
                "team_id": team_id,
                "team_name": team_name,
                "points": form_all["points"],
                "goal_difference": form_all["goal_difference"],
                "goals_scored": form_all["goals_scored"],
                "matches_played": form_all["matches_played"],
                "points_per_game": form_all["points_per_match"],
                "home_points_per_game": form_home["points_per_match"],
                "away_points_per_game": form_away["points_per_match"],
            })

        standings.sort(key=lambda x: (x["points"], x["goal_difference"], x["goals_scored"]), reverse=True)
        self._standings_cache[league] = standings
        return standings

    def get_league_position(self, team_id: int, league: str, season: str = None) -> dict:
        """Calculate current league position features for a team."""
        not_found = {"league_position": 0, "points": 0, "goal_difference": 0,
                     "points_per_game": 0, "home_points_per_game": 0, "away_points_per_game": 0,
                     "total_teams": 0, "title_gap": 0, "relegation_gap": 0, "in_relegation_zone": 0}

        standings = self._get_league_standings(league)
        if not standings:
            return not_found

        total_teams = len(standings)
        relegation_cutoff = total_teams - 2

        for i, entry in enumerate(standings):
            if entry["team_id"] == team_id:
                position = i + 1
                return {
                    "league_position": position,
                    "points": entry["points"],
                    "goal_difference": entry["goal_difference"],
                    "points_per_game": entry["points_per_game"],
                    "home_points_per_game": entry["home_points_per_game"],
                    "away_points_per_game": entry["away_points_per_game"],
                    "total_teams": total_teams,
                    "title_gap": position - 1,
                    "relegation_gap": relegation_cutoff - position,
                    "in_relegation_zone": int(position > relegation_cutoff),
                }

        return not_found
