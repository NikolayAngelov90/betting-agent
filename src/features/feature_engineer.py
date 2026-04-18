"""Main feature engineering pipeline that combines all feature sources."""

import numpy as np
from collections import defaultdict
from datetime import date as _date, timedelta
from typing import Optional

from sqlalchemy import or_

from src.features.team_features import TeamFeatures
from src.features.h2h_features import H2HFeatures
from src.features.injury_features import InjuryFeatures
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
        self.db = get_db()
        self._weather_service = None  # lazy-loaded on first use
        self.elo_ratings = None  # set externally from predictor.elo.ratings
        self._preload_cache: Optional[dict] = None

    def preload_batch(self, match_ids: list) -> None:
        """Bulk-preload all DB data needed for a list of fixtures into memory.

        Replaces O(N) per-fixture DB round-trips with three bulk queries:
          1. Match metadata for all match_ids (1 query)
          2. All Odds rows for all match_ids (1 query)
          3. Completed match history for all involved teams (1 query)

        Once Story 1.2 wires the cache into the _get_*_features helpers,
        create_features() will read from self._preload_cache instead of issuing
        live DB queries for each fixture.  Calling preload_batch is optional —
        if not called (or if it raises), create_features() falls back to
        per-fixture live queries with zero behaviour change.
        """
        if not match_ids:
            return

        try:
            # Init inside try so a mid-query exception never leaves a partial cache
            # that Story 1.2 consumers could incorrectly read.
            self._preload_cache = {"match_meta": {}, "odds": {}, "team_history": {}}

            # ── Query 1: match metadata ───────────────────────────────────────
            with self.db.get_session() as session:
                fixture_rows = session.query(Match).filter(
                    Match.id.in_(match_ids)
                ).all()

                all_team_ids: set = set()
                for m in fixture_rows:
                    self._preload_cache["match_meta"][m.id] = {
                        "home_team_id": m.home_team_id,
                        "away_team_id": m.away_team_id,
                        "league": m.league or "",
                        "referee": m.referee or "",
                        "match_date": m.match_date,
                        "venue": m.venue,
                    }
                    all_team_ids.add(m.home_team_id)
                    all_team_ids.add(m.away_team_id)

            # ── Query 2: odds for all fixtures ────────────────────────────────
            with self.db.get_session() as session:
                odds_rows = session.query(Odds).filter(
                    Odds.match_id.in_(match_ids)
                ).all()

                odds_by_match: dict = defaultdict(list)
                for row in odds_rows:
                    odds_by_match[row.match_id].append({
                        "market_type": row.market_type,
                        "bookmaker": row.bookmaker,
                        "selection": row.selection,
                        "odds_value": row.odds_value,
                        "opening_odds": row.opening_odds,
                    })
                self._preload_cache["odds"] = dict(odds_by_match)

            # ── Query 3: team history (one bulk query for all teams) ──────────
            if all_team_ids:
                # 365-day window; results ordered desc so first N per team are newest.
                # Cap at 60 per team: largest window used by any feature method is 30
                # (referee stats), and venue-specific windows need ~2× the total to
                # find 10 home/away matches each — 60 covers all cases with headroom.
                _TEAM_HISTORY_CAP = 60
                cutoff = _date.today() - timedelta(days=365)
                with self.db.get_session() as session:
                    history_rows = session.query(Match).filter(
                        Match.is_fixture == False,
                        Match.home_goals.isnot(None),
                        Match.match_date >= cutoff,
                        or_(
                            Match.home_team_id.in_(all_team_ids),
                            Match.away_team_id.in_(all_team_ids),
                        ),
                    ).order_by(Match.match_date.desc()).all()

                    team_history: dict = defaultdict(list)
                    team_counts: dict = defaultdict(int)
                    for m in history_rows:
                        row_data = {
                            "id": m.id,
                            "match_date": m.match_date,
                            "home_team_id": m.home_team_id,
                            "away_team_id": m.away_team_id,
                            "league": m.league,
                            "referee": m.referee,
                            "home_goals": m.home_goals,
                            "away_goals": m.away_goals,
                            "home_xg": m.home_xg,
                            "away_xg": m.away_xg,
                            "home_yellow_cards": m.home_yellow_cards,
                            "away_yellow_cards": m.away_yellow_cards,
                            "home_red_cards": m.home_red_cards,
                            "away_red_cards": m.away_red_cards,
                            "home_fouls": m.home_fouls,
                            "away_fouls": m.away_fouls,
                            "regulation_home_goals": m.regulation_home_goals,
                            "regulation_away_goals": m.regulation_away_goals,
                        }
                        if (m.home_team_id in all_team_ids
                                and team_counts[m.home_team_id] < _TEAM_HISTORY_CAP):
                            team_history[m.home_team_id].append(row_data)
                            team_counts[m.home_team_id] += 1
                        if (m.away_team_id in all_team_ids
                                and team_counts[m.away_team_id] < _TEAM_HISTORY_CAP):
                            team_history[m.away_team_id].append(row_data)
                            team_counts[m.away_team_id] += 1

                    self._preload_cache["team_history"] = dict(team_history)

            logger.debug(
                f"preload_batch: {len(self._preload_cache['match_meta'])} fixtures, "
                f"{sum(len(v) for v in self._preload_cache['odds'].values())} odds rows, "
                f"{len(self._preload_cache.get('team_history', {}))} teams of history"
            )

        except Exception as exc:
            logger.warning(f"preload_batch failed — falling back to per-fixture queries: {exc}")
            self._preload_cache = None

    async def create_features(self, match_id: int, as_of_date=None,
                              for_training: bool = False) -> dict:
        """Build complete feature dictionary for a match.

        Args:
            match_id: Match database ID
            as_of_date: Only use data before this date (for training).
                        None = no cutoff (live prediction).
            for_training: If True, skip external API calls (weather, news)
                         and coarsen league standings cache to monthly
                         granularity. This reduces DB queries from ~146
                         to ~25 per match and eliminates HTTP latency.

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

        import asyncio as _asyncio

        features = {}

        # 1. Team form features (overall, home, away) — all windows at 10 games
        _elo = self.elo_ratings
        home_form_all = self.team_features.get_form_features(home_id, 10, "all", as_of_date=as_of_date, elo_ratings=_elo)
        home_form_home = self.team_features.get_form_features(home_id, 10, "home", as_of_date=as_of_date, elo_ratings=_elo)
        away_form_all = self.team_features.get_form_features(away_id, 10, "all", as_of_date=as_of_date, elo_ratings=_elo)
        away_form_away = self.team_features.get_form_features(away_id, 10, "away", as_of_date=as_of_date, elo_ratings=_elo)

        features.update(self._prefix_dict(home_form_all, "home_overall_"))
        features.update(self._prefix_dict(home_form_home, "home_home_"))
        features.update(self._prefix_dict(away_form_all, "away_overall_"))
        features.update(self._prefix_dict(away_form_away, "away_away_"))

        # Yield event loop so other coroutines (other fixtures) can interleave.
        # Without this, asyncio.gather(concurrency=5) is effectively sequential
        # because synchronous DB calls never yield control.
        await _asyncio.sleep(0)

        # 2. H2H features
        h2h = self.h2h_features.get_h2h_features(home_id, away_id, as_of_date=as_of_date)
        features.update(h2h)

        # 3. Injury features (skip during training — no historical injury data)
        if not for_training:
            home_injuries = self.injury_features.get_injury_features(home_id)
            away_injuries = self.injury_features.get_injury_features(away_id)
            features.update(self._prefix_dict(home_injuries, "home_injury_"))
            features.update(self._prefix_dict(away_injuries, "away_injury_"))

        await _asyncio.sleep(0)

        # 4. League position features
        # Coarsen the standings date to the 1st of the current month so that the
        # standings cache hits across all fixtures in the same league on the same day
        # (multiple fixtures → same cache key → one DB query per league, not N²).
        # League standings barely change within a month so this is accurate enough.
        # Applies to both training (as_of_date set) and live prediction (as_of_date None).
        _standings_date = as_of_date
        from datetime import date as _date
        _effective_date = _standings_date if _standings_date is not None else _date.today()
        _standings_date = _effective_date.replace(day=1)
        home_pos = self.team_features.get_league_position(home_id, league, as_of_date=_standings_date)
        away_pos = self.team_features.get_league_position(away_id, league, as_of_date=_standings_date)
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

        # 5. International competition features (CL/EL/ECL form)
        home_intl = self.team_features.get_international_form(home_id, as_of_date=as_of_date)
        away_intl = self.team_features.get_international_form(away_id, as_of_date=as_of_date)
        features.update(self._prefix_dict(home_intl, "home_"))
        features.update(self._prefix_dict(away_intl, "away_"))

        # Flag if current match is an international competition
        is_international = league in self.team_features.INTERNATIONAL_LEAGUES
        features["is_international_match"] = int(is_international)

        # International experience differential
        features["intl_experience_diff"] = home_intl["intl_matches"] - away_intl["intl_matches"]
        features["intl_quality_diff"] = home_intl["intl_points_per_match"] - away_intl["intl_points_per_match"]

        # 7. xG-based features (from API-Football)
        home_xg = self._get_xg_features(home_id, "home", as_of_date=as_of_date)
        away_xg = self._get_xg_features(away_id, "away", as_of_date=as_of_date)
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

        await _asyncio.sleep(0)

        # 9. Referee features (from Flashscore — if referee is known for this fixture)
        ref_features = self._get_referee_features(referee, as_of_date=as_of_date)
        features.update(ref_features)

        # 10. RSI + MACD momentum indicators
        home_mom = self.team_features.get_momentum_indicators(home_id, as_of_date=as_of_date)
        away_mom = self.team_features.get_momentum_indicators(away_id, as_of_date=as_of_date)
        features.update(self._prefix_dict(home_mom, "home_"))
        features.update(self._prefix_dict(away_mom, "away_"))
        features["rsi_diff"] = home_mom["rsi"] - away_mom["rsi"]
        features["macd_diff"] = home_mom["macd"] - away_mom["macd"]

        await _asyncio.sleep(0)

        # 11. Bookmaker implied probability (Bet365/Pinnacle 1X2 odds already in DB)
        bk_features = self._get_bookmaker_features(match_id)
        features.update(bk_features)

        # 11b. Odds movement features (opening vs current odds)
        odds_movement = self._get_odds_movement_features(match_id)
        features.update(odds_movement)

        # 12. Situational context: rest days + midweek flag
        _venue = None
        with self.db.get_session() as session:
            match_obj = session.get(Match, match_id)
            if match_obj:
                _match_date = match_obj.match_date
                _home_id = match_obj.home_team_id
                _away_id = match_obj.away_team_id
                _venue = match_obj.venue
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
            # Fatigue index features
            features["home_matches_14d"] = home_sit["matches_14d"]
            features["away_matches_14d"] = away_sit["matches_14d"]
            features["home_matches_30d"] = home_sit["matches_30d"]
            features["away_matches_30d"] = away_sit["matches_30d"]
            features["home_fatigue_index"] = home_sit["fatigue_index"]
            features["away_fatigue_index"] = away_sit["fatigue_index"]
            features["fatigue_diff"] = home_sit["fatigue_index"] - away_sit["fatigue_index"]
            features["home_short_rest_count"] = home_sit["short_rest_count"]
            features["away_short_rest_count"] = away_sit["short_rest_count"]

        # 13. League-specific baseline rates (home advantage, avg goals, BTTS rate, etc.)
        # Use the coarsened standings date for both training and live prediction so
        # the league-rates cache hits across all fixtures in the same league on the same day.
        league_feat = self._get_league_features(league, as_of_date=_standings_date)
        features.update(league_feat)

        # 14. Match-day weather (Open-Meteo free API — no key required)
        # Skip during training: historical weather is unavailable and
        # constant placeholders just become zero-variance noise.
        if not for_training:
            weather = self._get_weather_features(_venue, _match_date)
            features.update(weather)

        logger.debug(f"Generated {len(features)} features for match {match_id}")
        return features

    def create_feature_vector(self, features: dict) -> np.ndarray:
        """Convert feature dictionary to a numeric numpy array for ML models.

        Non-numeric features (strings, booleans) are converted appropriately.
        Keys are sorted alphabetically for deterministic ordering regardless
        of which feature sections executed or in what order.
        """
        numeric_features = {}
        for key, value in features.items():
            if isinstance(value, bool):
                numeric_features[key] = float(value)
            elif isinstance(value, (int, float)):
                numeric_features[key] = float(value)
            # Skip string features like form_string

        sorted_keys = sorted(numeric_features.keys())
        return np.array([numeric_features[k] for k in sorted_keys])

    def get_feature_names(self, features: dict) -> list:
        """Get ordered list of numeric feature names (matches create_feature_vector order).

        Sorted alphabetically for deterministic ordering.
        """
        return sorted(
            key for key, value in features.items()
            if isinstance(value, (int, float, bool))
        )

    def _get_xg_features(self, team_id: int, venue: str = "all",
                          num_matches: int = 10, as_of_date=None) -> dict:
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

            if as_of_date is not None:
                query = query.filter(Match.match_date < as_of_date)

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

    def _get_referee_features(self, referee: str, as_of_date=None) -> dict:
        """Get historical statistics for a referee across their last 30 matches.

        Returns metrics that inform card/goal probability (referee strictness, pace of play).
        Returns zero-defaults when referee is unknown or has no history.
        """
        empty = {
            "referee_cards_per_match_avg": 0.0,
            "referee_fouls_per_match_avg": 0.0,
            "referee_goals_per_match_avg": 0.0,
            "referee_over25_rate": 0.0,
            "referee_avg_yellow_cards": 0.0,
            "referee_avg_red_cards": 0.0,
            "referee_matches": 0,
        }
        if not referee:
            return empty

        with self.db.get_session() as session:
            query = session.query(Match).filter(
                Match.referee == referee,
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
            )
            if as_of_date is not None:
                query = query.filter(Match.match_date < as_of_date)
            matches = query.order_by(Match.match_date.desc()).limit(30).all()

            if not matches:
                return empty

            cards_list = []
            yellow_list = []
            red_list = []
            fouls_total = 0
            fouls_matches = 0
            goals_list = []
            over25_count = 0

            for m in matches:
                yc = (m.home_yellow_cards or 0) + (m.away_yellow_cards or 0)
                rc = (m.home_red_cards or 0) + (m.away_red_cards or 0)
                cards_list.append(yc + rc)
                yellow_list.append(yc)
                red_list.append(rc)
                total_goals = (m.home_goals or 0) + (m.away_goals or 0)
                goals_list.append(total_goals)
                if total_goals > 2.5:
                    over25_count += 1
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
            "referee_over25_rate": round(over25_count / n, 3),
            "referee_avg_yellow_cards": round(sum(yellow_list) / n, 2),
            "referee_avg_red_cards": round(sum(red_list) / n, 2),
            "referee_matches": n,
        }

    def _get_bookmaker_features(self, match_id: int) -> dict:
        """Return margin-adjusted implied probabilities from bookmaker odds.

        Covers four markets stored in the Odds table:
          • 1X2          → home/draw/away win probabilities
          • over_under   → Over/Under 1.5 and Over/Under 2.5 probabilities
          • btts         → BTTS Yes/No probabilities
          • team_goals   → Home Over 1.5 and Away Over 1.5 probabilities

        For each market the preferred bookmaker order is: Bet365 → Pinnacle → any.
        Bookmaker overround is removed via the standard margin-normalisation formula.
        """
        defaults = {
            # 1X2
            "home_implied_prob": 1/3,
            "draw_implied_prob": 1/3,
            "away_implied_prob": 1/3,
            "bookmaker_available": 0,
            # Over/Under totals
            "over25_implied_prob": 0.0,
            "under25_implied_prob": 0.0,
            "over15_implied_prob": 0.0,
            "under15_implied_prob": 0.0,
            "goals_bookmaker_available": 0,
            # BTTS
            "btts_yes_implied_prob": 0.0,
            "btts_no_implied_prob": 0.0,
            "btts_bookmaker_available": 0,
            # Team goal lines (home / away score ≥ 2)
            "home_over15_implied_prob": 0.0,
            "away_over15_implied_prob": 0.0,
            "team_goals_bookmaker_available": 0,
        }
        try:
            # Group: {(market_type, bookmaker): {selection: odds_value}}
            # Extract all data inside the session to avoid detached-instance errors.
            bk_data: dict = defaultdict(dict)
            with self.db.get_session() as session:
                rows = session.query(Odds).filter(
                    Odds.match_id == match_id,
                    Odds.market_type.in_(["1X2", "over_under", "btts", "team_goals"]),
                ).all()

                if not rows:
                    return defaults

                for row in rows:
                    bk_data[(row.market_type, row.bookmaker)][row.selection] = row.odds_value

            result = dict(defaults)

            # Helper: margin-remove a 3-way market
            def _3way(od):
                h = od.get("Home") or od.get("Home Win")
                d = od.get("Draw")
                a = od.get("Away") or od.get("Away Win")
                if not (h and d and a) or min(h, d, a) <= 0:
                    return None
                rh, rd, ra = 1/h, 1/d, 1/a
                m = rh + rd + ra
                if m <= 0:
                    return None
                return round(rh/m, 4), round(rd/m, 4), round(ra/m, 4)

            # Helper: margin-remove a 2-way market (over/under style)
            def _2way(od, over_key, under_key):
                o = od.get(over_key)
                u = od.get(under_key)
                if not (o and u) or min(o, u) <= 0:
                    return None
                ro, ru = 1/o, 1/u
                m = ro + ru
                if m <= 0:
                    return None
                return round(ro/m, 4), round(ru/m, 4)

            # Bookmaker priority order
            _priority = ("Bet365", "Pinnacle")

            def _best_bk(market_type):
                """Return the odds dict for the preferred available bookmaker."""
                for pref in _priority:
                    if (market_type, pref) in bk_data:
                        return bk_data[(market_type, pref)]
                # Fall back to any bookmaker for this market
                for (mt, _bk), od in bk_data.items():
                    if mt == market_type:
                        return od
                return None

            # ── 1X2 ──────────────────────────────────────────────────────────
            od = _best_bk("1X2")
            if od:
                r = _3way(od)
                if r:
                    result["home_implied_prob"] = r[0]
                    result["draw_implied_prob"] = r[1]
                    result["away_implied_prob"] = r[2]
                    result["bookmaker_available"] = 1

            # ── Over/Under ────────────────────────────────────────────────────
            od = _best_bk("over_under")
            if od:
                r25 = _2way(od, "Over 2.5", "Under 2.5")
                if r25:
                    result["over25_implied_prob"] = r25[0]
                    result["under25_implied_prob"] = r25[1]
                    result["goals_bookmaker_available"] = 1
                r15 = _2way(od, "Over 1.5", "Under 1.5")
                if r15:
                    result["over15_implied_prob"] = r15[0]
                    result["under15_implied_prob"] = r15[1]
                    result["goals_bookmaker_available"] = 1

            # ── BTTS ─────────────────────────────────────────────────────────
            od = _best_bk("btts")
            if od:
                yes_odds = od.get("Yes") or od.get("BTTS Yes")
                no_odds = od.get("No") or od.get("BTTS No")
                if yes_odds and no_odds and min(yes_odds, no_odds) > 0:
                    ry, rn = 1/yes_odds, 1/no_odds
                    m = ry + rn
                    if m > 0:
                        result["btts_yes_implied_prob"] = round(ry/m, 4)
                        result["btts_no_implied_prob"] = round(rn/m, 4)
                    result["btts_bookmaker_available"] = 1

            # ── Team goal lines ───────────────────────────────────────────────
            od = _best_bk("team_goals")
            if od:
                rh = _2way(od, "Home Over 1.5", "Home Under 1.5")
                if rh:
                    result["home_over15_implied_prob"] = rh[0]
                    result["team_goals_bookmaker_available"] = 1
                ra = _2way(od, "Away Over 1.5", "Away Under 1.5")
                if ra:
                    result["away_over15_implied_prob"] = ra[0]
                    result["team_goals_bookmaker_available"] = 1

            return result

        except Exception as e:
            logger.warning(f"Bookmaker features failed for match {match_id}: {e}")
            return defaults

    def _get_odds_movement_features(self, match_id: int) -> dict:
        """Compute odds movement features from opening_odds vs current odds_value.

        Sharp money typically moves lines in predictable ways:
        - Home odds shortening (dropping) → sharp money on home team
        - Over 2.5 shortening → sharp money expects goals

        Features returned (0 when no movement data available):
        - home_odds_movement: % change in home win odds (negative = shortening)
        - away_odds_movement: % change in away win odds
        - over25_odds_movement: % change in over 2.5 odds
        - max_abs_movement: largest absolute odds movement (sharp signal strength)
        - movement_direction: +1 if home shortening, -1 if away shortening, 0 neutral
        """
        defaults = {
            "home_odds_movement": 0.0,
            "away_odds_movement": 0.0,
            "over25_odds_movement": 0.0,
            "max_abs_movement": 0.0,
            "movement_direction": 0.0,
        }
        try:
            with self.db.get_session() as session:
                rows = session.query(Odds).filter(
                    Odds.match_id == match_id,
                    Odds.opening_odds.isnot(None),
                    Odds.bookmaker != "Flashscore",
                ).all()

                if not rows:
                    return defaults

                # Find movements for key selections
                movements = {}
                for row in rows:
                    if row.opening_odds and row.opening_odds > 0 and row.odds_value > 0:
                        pct_change = (row.odds_value - row.opening_odds) / row.opening_odds
                        key = (row.market_type, row.selection)
                        # Keep the one from the preferred bookmaker (first seen wins)
                        if key not in movements:
                            movements[key] = round(pct_change, 4)

                result = dict(defaults)

                home_mv = movements.get(("1X2", "Home"), 0)
                away_mv = movements.get(("1X2", "Away"), 0)
                over25_mv = movements.get(("over_under", "Over 2.5"), 0)

                result["home_odds_movement"] = home_mv
                result["away_odds_movement"] = away_mv
                result["over25_odds_movement"] = over25_mv

                all_mvs = [abs(v) for v in movements.values() if v != 0]
                result["max_abs_movement"] = max(all_mvs) if all_mvs else 0.0

                # Direction: negative home_mv means home odds dropped = sharp on home
                if home_mv < -0.02 and away_mv > 0.02:
                    result["movement_direction"] = 1.0  # sharp on home
                elif away_mv < -0.02 and home_mv > 0.02:
                    result["movement_direction"] = -1.0  # sharp on away
                else:
                    result["movement_direction"] = 0.0

                return result
        except Exception as e:
            logger.warning(f"Odds movement features failed for match {match_id}: {e}")
            return defaults

    def _get_league_features(self, league: str, as_of_date=None) -> dict:
        """Compute league-specific baseline rates from the last 200 completed matches.

        These give the ML model a league-aware prior so it can learn that, e.g.,
        the Bundesliga has more goals per game than Serie A, or that the Championship
        has a higher draw rate than the Premier League.

        Results are cached per-instance so the query only runs once per league
        per prediction/training session.
        """
        defaults = {
            "league_home_win_rate": 0.45,
            "league_draw_rate": 0.25,
            "league_away_win_rate": 0.30,
            "league_avg_goals": 2.60,
            "league_over25_rate": 0.52,
            "league_btts_rate": 0.50,
            "league_matches_count": 0,
        }
        if not league:
            return defaults

        if not hasattr(self, "_league_features_cache"):
            self._league_features_cache: dict = {}
        cache_key = (league, as_of_date)
        if cache_key in self._league_features_cache:
            return self._league_features_cache[cache_key]

        try:
            with self.db.get_session() as session:
                query = session.query(Match).filter(
                    Match.league == league,
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.away_goals.isnot(None),
                )
                if as_of_date is not None:
                    query = query.filter(Match.match_date < as_of_date)
                matches = query.order_by(Match.match_date.desc()).limit(200).all()

                if len(matches) < 10:
                    self._league_features_cache[cache_key] = defaults
                    return defaults

                n = len(matches)
                home_wins = sum(1 for m in matches if (m.home_goals or 0) > (m.away_goals or 0))
                draws = sum(1 for m in matches if (m.home_goals or 0) == (m.away_goals or 0))
                away_wins = n - home_wins - draws
                total_goals = sum((m.home_goals or 0) + (m.away_goals or 0) for m in matches)
                over25 = sum(
                    1 for m in matches if (m.home_goals or 0) + (m.away_goals or 0) > 2
                )
                btts = sum(
                    1 for m in matches if (m.home_goals or 0) > 0 and (m.away_goals or 0) > 0
                )

                result = {
                    "league_home_win_rate": round(home_wins / n, 4),
                    "league_draw_rate": round(draws / n, 4),
                    "league_away_win_rate": round(away_wins / n, 4),
                    "league_avg_goals": round(total_goals / n, 4),
                    "league_over25_rate": round(over25 / n, 4),
                    "league_btts_rate": round(btts / n, 4),
                    "league_matches_count": n,
                }
                self._league_features_cache[cache_key] = result
                return result

        except Exception as e:
            logger.warning(f"League features failed for {league}: {e}")
            return defaults

    def _get_situational_features(self, team_id: int, match_date) -> dict:
        """Return rest days, midweek flag, and cumulative fatigue index.

        Fatigue index considers:
        - Number of matches in the last 14, 21, and 30 days
        - Whether any recent match went to extra time (120 min)
        - Cumulative short-rest matches (< 4 days between games)
        """
        defaults = {
            "rest_days": 7, "midweek_flag": 0,
            "matches_14d": 0, "matches_21d": 0, "matches_30d": 0,
            "fatigue_index": 0.0, "short_rest_count": 0,
        }
        try:
            with self.db.get_session() as session:
                # Fetch last 10 matches (enough to cover 30 days for busy teams)
                recent = session.query(Match).filter(
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.match_date < match_date,
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                ).order_by(Match.match_date.desc()).limit(10).all()

                if not recent:
                    return defaults

                prev = recent[0]
                delta = (match_date - prev.match_date).days
                rest_days = min(delta, 21)
                midweek_flag = 1 if prev.match_date.weekday() in (1, 2, 3) else 0

                # Count matches in recent windows
                matches_14d = 0
                matches_21d = 0
                matches_30d = 0
                short_rest_count = 0
                extra_time_recent = 0

                match_dates = []
                for m in recent:
                    days_before = (match_date - m.match_date).days
                    if days_before <= 14:
                        matches_14d += 1
                    if days_before <= 21:
                        matches_21d += 1
                    if days_before <= 30:
                        matches_30d += 1
                    match_dates.append(m.match_date)
                    # Check for extra time (regulation score exists and differs from final)
                    if (m.regulation_home_goals is not None
                            and m.home_goals is not None
                            and (m.regulation_home_goals != m.home_goals
                                 or m.regulation_away_goals != m.away_goals)):
                        if days_before <= 14:
                            extra_time_recent += 1

                # Count short-rest intervals (< 4 days between consecutive matches)
                for i in range(len(match_dates) - 1):
                    gap = (match_dates[i] - match_dates[i + 1]).days
                    if gap < 4:
                        short_rest_count += 1

                # Composite fatigue index (0-1 scale):
                # - matches_14d contributes most (congestion is immediate)
                # - short_rest_count penalises accumulated fatigue
                # - extra_time adds penalty
                fatigue_index = (
                    min(matches_14d / 5.0, 1.0) * 0.50       # 5+ matches in 14d = max
                    + min(short_rest_count / 3.0, 1.0) * 0.30  # 3+ short rests = max
                    + min(extra_time_recent, 1) * 0.20          # any extra time = penalty
                )

                return {
                    "rest_days": rest_days,
                    "midweek_flag": midweek_flag,
                    "matches_14d": matches_14d,
                    "matches_21d": matches_21d,
                    "matches_30d": matches_30d,
                    "fatigue_index": round(fatigue_index, 3),
                    "short_rest_count": short_rest_count,
                }
        except Exception as e:
            logger.warning(f"Situational features failed for team {team_id}: {e}")
            return defaults

    def _get_weather_features(self, venue, match_date) -> dict:
        """Return weather features for the match venue and date.

        Uses Open-Meteo free API (no key). Returns neutral defaults on failure.
        Can be disabled via models.weather_features_enabled: false in config.
        """
        defaults = {
            "weather_temp_c": 12.0, "weather_wind_kmh": 10.0,
            "weather_precip_mm": 0.0, "weather_is_raining": 0,
            "weather_is_windy": 0, "weather_available": 0,
        }
        try:
            from src.utils.config import get_config as _gc
            if not _gc().get("models.weather_features_enabled", True):
                return defaults
            if self._weather_service is None:
                from src.features.weather_service import WeatherService
                self._weather_service = WeatherService()
            if match_date is None:
                return defaults
            md = match_date.date() if hasattr(match_date, "date") else match_date
            return self._weather_service.get_match_weather(venue, md)
        except Exception as exc:
            logger.debug(f"Weather features failed: {exc}")
            return defaults

    def _prefix_dict(self, d: dict, prefix: str) -> dict:
        """Add a prefix to all dictionary keys."""
        return {f"{prefix}{k}": v for k, v in d.items()}
