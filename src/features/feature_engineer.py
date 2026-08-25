"""Main feature engineering pipeline that combines all feature sources."""

import numpy as np
from collections import defaultdict
from datetime import date as _date, timedelta
from typing import Optional

from sqlalchemy import or_

from src.data.sql_helpers import id_in
from src.features import preload_cache as _pc
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
        # Per-instance league baseline rates cache.  Keyed by (league, as_of_date).
        # Must be invalidated after the agent ingests new historical data within
        # the same process (CI), or stale rates leak into predictions.
        self._league_features_cache: dict = {}

    def clear_league_cache(self):
        """Invalidate the league baseline rates cache.

        Call this after `daily_update` adds new completed matches so the next
        `get_daily_picks` doesn't re-use stats computed before the new data
        landed.  Also resets the standings cache on TeamFeatures.
        """
        self._league_features_cache = {}
        try:
            self.team_features.clear_standings_cache()
        except Exception:
            pass

    def preload_batch(self, match_ids: list,
                      cap_per_team: int = 60,
                      cutoff_days: int = 365,
                      league_cap: int = 2500,
                      referee_cap: int = 30) -> None:
        """Bulk-preload all DB data needed for a list of fixtures into memory.

        Replaces O(N) per-fixture DB round-trips with four bulk queries and two
        scopes derived from data the process already holds:

          1. Match metadata for all match_ids                        (query)
          2. All Odds rows for all match_ids                         (query)
          3. Completed match history for the fixtures' own teams     (query)
          4. Referee history for the fixtures' referees              (query)
          5. League-wide history, for standings                      (derived)
          6. Head-to-head history per pairing                        (derived)

        Scopes 4-6 exist because three feature families need context *wider*
        than the two teams playing: standings rank the whole division, referee
        stats span every match the official took, and H2H reaches back past a
        team's recent-form window. Without them those lookups were answered from
        the two-team scope and silently returned zeroed or truncated features —
        see ``src/features/preload_cache.py`` for the full history.

        5 and 6 are derived rather than queried because the completed-match
        history is already in memory for Elo/Poisson (from the local Parquet
        mirror), and it carries exactly the columns they need. Querying them
        would have cost ~3.2 MB on a busy 32-league matchday for rows the
        process already had.

        Every cached scope records whether it is *complete*. Consumers go
        through the accessors in ``preload_cache``, which return ``None`` when
        the cache cannot prove it holds the same rows the live query would, and
        the caller then runs that live query. Preload and live are therefore
        equivalent by construction, not by coincidence.

        Args:
            cap_per_team: Max history rows per team (60 for live, 200 for training).
            cutoff_days: History window in days (365 for live, 0=no cutoff for training).
            league_cap: Memory safety valve — a league with more completed
                matches than this is dropped from the standings scope and its
                standings fall back to live queries.
            referee_cap: Rows per referee; matches the live query's LIMIT 30.

        Calling preload_batch is optional — if not called (or if it raises),
        create_features() falls back to per-fixture live queries with zero
        behaviour change.
        """
        if not match_ids:
            return

        try:
            # Init inside try so a mid-query exception never leaves a partial cache
            # that Story 1.2 consumers could incorrectly read.
            self._preload_cache = {
                "match_meta": {}, "odds": {}, "team_history": {},
                "team_complete": set(),
                "league_history": {}, "league_complete": set(),
                "referee_history": {}, "referee_complete": set(),
                "h2h_history": {}, "h2h_complete": set(),
            }

            # ── Query 1: match metadata ───────────────────────────────────────
            # Column-projected: the loop below copies eight fields out of the
            # row, so asking for all 45 (SELECT matches.*) shipped ~4x the bytes
            # needed. Same for queries 2 and 3.
            with self.db.get_session() as session:
                fixture_rows = session.query(
                    Match.id, Match.home_team_id, Match.away_team_id,
                    Match.league, Match.referee, Match.match_date,
                    Match.venue, Match.round, Match.season,
                ).filter(
                    id_in(session, Match.id, match_ids)
                ).all()

                all_team_ids: set = set()
                all_leagues: set = set()
                all_referees: set = set()
                all_pairs: set = set()
                for m in fixture_rows:
                    self._preload_cache["match_meta"][m.id] = {
                        "home_team_id": m.home_team_id,
                        "away_team_id": m.away_team_id,
                        "league": m.league or "",
                        "referee": m.referee or "",
                        "match_date": m.match_date,
                        "venue": m.venue,
                        "round": m.round,
                        "season": m.season,
                    }
                    all_team_ids.add(m.home_team_id)
                    all_team_ids.add(m.away_team_id)
                    if m.league:
                        all_leagues.add(m.league)
                    if m.referee:
                        all_referees.add(m.referee)
                    all_pairs.add(_pc.h2h_key(m.home_team_id, m.away_team_id))

            # ── Query 2: odds for all fixtures ────────────────────────────────
            with self.db.get_session() as session:
                odds_rows = session.query(
                    Odds.match_id, Odds.market_type, Odds.bookmaker,
                    Odds.selection, Odds.odds_value, Odds.opening_odds,
                ).filter(
                    id_in(session, Odds.match_id, match_ids)
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
                with self.db.get_session() as session:
                    _hist_filter = [
                        Match.is_fixture == False,
                        Match.home_goals.isnot(None),
                        # Stage 13 (s5.3). This query does NOT route through
                        # match_history — it carries its own copy of the
                        # predicate above. Excluding the 29 from training while
                        # leaving them here would keep the contamination in
                        # team form, H2H and rolling goals, which is exactly
                        # where it does harm.
                        Match.training_exclusion_reason.is_(None),
                        # Array params, not IN lists: this clause binds the
                        # SAME team set twice, so at 100x volume a plain IN
                        # would put ~52,800 bind parameters in one statement
                        # against PostgreSQL's 65,535 hard cap.
                        or_(
                            id_in(session, Match.home_team_id, all_team_ids),
                            id_in(session, Match.away_team_id, all_team_ids),
                        ),
                    ]
                    if cutoff_days > 0:
                        cutoff = _date.today() - timedelta(days=cutoff_days)
                        _hist_filter.append(Match.match_date >= cutoff)
                    history_rows = session.query(
                        Match.id, Match.match_date,
                        Match.home_team_id, Match.away_team_id,
                        Match.league, Match.referee,
                        Match.home_goals, Match.away_goals,
                        Match.home_xg, Match.away_xg,
                        Match.home_yellow_cards, Match.away_yellow_cards,
                        Match.home_red_cards, Match.away_red_cards,
                        Match.home_fouls, Match.away_fouls,
                        Match.regulation_home_goals, Match.regulation_away_goals,
                        Match.home_shots, Match.away_shots,
                        Match.home_shots_on_target, Match.away_shots_on_target,
                        Match.home_possession, Match.away_possession,
                        Match.home_corners, Match.away_corners,
                        Match.home_dangerous_attacks, Match.away_dangerous_attacks,
                        Match.home_saves, Match.away_saves,
                        Match.home_offsides, Match.away_offsides,
                        Match.home_free_kicks, Match.away_free_kicks,
                    ).filter(
                        *_hist_filter
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
                            # Stats needed by get_form_features_from_cache
                            "home_shots": m.home_shots,
                            "away_shots": m.away_shots,
                            "home_shots_on_target": m.home_shots_on_target,
                            "away_shots_on_target": m.away_shots_on_target,
                            "home_possession": m.home_possession,
                            "away_possession": m.away_possession,
                            "home_corners": m.home_corners,
                            "away_corners": m.away_corners,
                            "home_dangerous_attacks": m.home_dangerous_attacks,
                            "away_dangerous_attacks": m.away_dangerous_attacks,
                            "home_saves": m.home_saves,
                            "away_saves": m.away_saves,
                            "home_offsides": m.home_offsides,
                            "away_offsides": m.away_offsides,
                            "home_free_kicks": m.home_free_kicks,
                            "away_free_kicks": m.away_free_kicks,
                        }
                        if (m.home_team_id in all_team_ids
                                and team_counts[m.home_team_id] < cap_per_team):
                            team_history[m.home_team_id].append(row_data)
                            team_counts[m.home_team_id] += 1
                        if (m.away_team_id in all_team_ids
                                and team_counts[m.away_team_id] < cap_per_team):
                            team_history[m.away_team_id].append(row_data)
                            team_counts[m.away_team_id] += 1

                    self._preload_cache["team_history"] = dict(team_history)
                    # A team's cached list is the WHOLE history only when no date
                    # window was applied and the row cap was never reached.
                    # Conservative on purpose: a wrong "complete" would serve a
                    # short answer as if it were the full one.
                    if cutoff_days <= 0:
                        self._preload_cache["team_complete"] = {
                            tid for tid in all_team_ids
                            if team_counts[tid] < cap_per_team
                        }

            # ── Query 4: referee history ──────────────────────────────────────
            if all_referees:
                self._preload_referee_history(all_referees, referee_cap)

            # ── Scopes 5 & 6: derived from the shared completed-match history ──
            # Loaded once here rather than inside each builder — it is the same
            # rows for both, and on a busy matchday that is 33k of them.
            if all_leagues or all_pairs:
                from src.data.match_history import get_completed_matches
                _history = get_completed_matches(self.db)
                if all_leagues:
                    self._preload_league_history(_history, all_leagues, league_cap)
                if all_pairs:
                    self._preload_h2h_history(_history, all_pairs)

            logger.debug(
                f"preload_batch: {len(self._preload_cache['match_meta'])} fixtures, "
                f"{sum(len(v) for v in self._preload_cache['odds'].values())} odds rows, "
                f"{len(self._preload_cache.get('team_history', {}))} teams of history, "
                f"{len(self._preload_cache.get('league_history', {}))} leagues, "
                f"{len(self._preload_cache.get('referee_history', {}))} referees, "
                f"{len(self._preload_cache.get('h2h_history', {}))} pairings"
            )

        except Exception as exc:
            logger.warning(f"preload_batch failed — falling back to per-fixture queries: {exc}")
            self._preload_cache = None

    # ------------------------------------------------------------------ scopes

    @staticmethod
    def _lean_row(m) -> dict:
        """Goal-only history row.

        Carries the same keys the rich rows use for goals/ids/date/league so the
        shared form maths works unchanged; the stat columns are simply absent,
        and ``_calculate_form_from_dicts`` reads those with ``.get()`` and
        ignores missing values exactly as it does for a match with no stats
        recorded.
        """
        return {
            "id": m.id,
            "match_date": m.match_date,
            "home_team_id": m.home_team_id,
            "away_team_id": m.away_team_id,
            "home_goals": m.home_goals,
            "away_goals": m.away_goals,
            "league": m.league,
        }

    def _preload_league_history(self, history: list, leagues: set,
                                league_cap: int) -> None:
        """Per-league, per-team history for ``_get_league_standings``.

        Standings rank *every* club in the division, so this scope is far wider
        than the fixture's two teams — on a busy 32-league matchday, querying it
        would pull ~33,600 rows, essentially the whole table.

        It does not need to. The process already holds the complete
        completed-match history in memory for Elo/Poisson, sourced from the
        local Parquet mirror (or, failing that, one projected read), and those
        rows carry exactly the columns standings consume: ids, date, goals,
        league. So this scope is *derived*, not fetched — zero extra egress, and
        complete by construction rather than by a cap that might bite.

        ``league_cap`` remains as a safety valve on memory: a league with more
        completed matches than the cap is left out of the scope entirely, and
        its standings fall back to live queries.
        """
        by_league: dict = defaultdict(lambda: defaultdict(list))
        counts: dict = defaultdict(int)
        # Newest-first, to match the ordering every consumer slices with.
        for m in reversed(history):
            if m.league not in leagues:
                continue
            counts[m.league] += 1
            if counts[m.league] > league_cap:
                continue
            row = self._lean_row(m)
            by_league[m.league][m.home_team_id].append(row)
            by_league[m.league][m.away_team_id].append(row)

        over_cap = {lg for lg, n in counts.items() if n > league_cap}
        if over_cap:
            logger.warning(
                f"league history cap ({league_cap}) exceeded for {sorted(over_cap)} "
                f"— their standings will use live queries"
            )
        self._preload_cache["league_history"] = {
            lg: dict(teams) for lg, teams in by_league.items() if lg not in over_cap
        }
        self._preload_cache["league_complete"] = set(leagues) - over_cap

    def _preload_referee_history(self, referees: set, referee_cap: int) -> None:
        """Each referee's most recent matches inside the same 365-day window the
        live query uses, capped at the live query's LIMIT."""
        from sqlalchemy import func as _func

        _ref_cutoff = _date.today() - timedelta(days=365)
        with self.db.get_session() as session:
            _rn = _func.row_number().over(
                partition_by=Match.referee,
                order_by=(Match.match_date.desc(), Match.id.desc()),
            ).label("rn")
            sub = session.query(
                Match.id, Match.match_date, Match.referee,
                Match.home_goals, Match.away_goals,
                Match.home_yellow_cards, Match.away_yellow_cards,
                Match.home_red_cards, Match.away_red_cards,
                Match.home_fouls, Match.away_fouls,
                _rn,
            ).filter(
                id_in(session, Match.referee, referees),
                Match.is_fixture == False,  # noqa: E712
                Match.home_goals.isnot(None),
                Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
                Match.match_date >= _ref_cutoff,
            ).subquery()
            rows = session.query(sub).filter(sub.c.rn <= referee_cap).all()

        by_ref: dict = defaultdict(list)
        for m in rows:
            by_ref[m.referee].append({
                "id": m.id,
                "match_date": m.match_date,
                "referee": m.referee,
                "home_goals": m.home_goals,
                "away_goals": m.away_goals,
                "home_yellow_cards": m.home_yellow_cards,
                "away_yellow_cards": m.away_yellow_cards,
                "home_red_cards": m.home_red_cards,
                "away_red_cards": m.away_red_cards,
                "home_fouls": m.home_fouls,
                "away_fouls": m.away_fouls,
            })

        self._preload_cache["referee_history"] = dict(by_ref)
        # Complete when we hold every qualifying match, not just the newest N —
        # which matters once an as_of_date filter shrinks the slice further.
        self._preload_cache["referee_complete"] = {
            r for r in referees if len(by_ref.get(r, [])) < referee_cap
        }

    def _preload_h2h_history(self, history: list, pairs: set) -> None:
        """All past meetings for each fixture's pairing.

        Derived from the same in-memory history as the league scope. H2H needs
        the *complete* record of a pairing — the previous implementation read it
        out of the 60-row / 365-day team window, which silently halved the
        meeting count for long-standing rivalries.
        """
        by_pair: dict = defaultdict(list)
        for m in reversed(history):  # newest first
            key = _pc.h2h_key(m.home_team_id, m.away_team_id)
            if key not in pairs:
                continue
            by_pair[key].append(self._lean_row(m))

        self._preload_cache["h2h_history"] = dict(by_pair)
        # Every pairing is loaded in full, so all of them are exact.
        self._preload_cache["h2h_complete"] = set(pairs)

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
        _meta = (self._preload_cache or {}).get("match_meta", {}).get(match_id)
        if _meta is not None:
            home_id = _meta["home_team_id"]
            away_id = _meta["away_team_id"]
            league = _meta["league"]
            referee = _meta["referee"]
        else:
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

        # 1. Team form features (overall, home, away) — all windows at 10 games.
        # During training, do NOT pass current Elo ratings: those are computed from
        # ALL historical matches (including the targets) so applying them as
        # opponent-quality weights leaks future information into past form.
        # Live prediction is unaffected because as_of_date is None.
        _elo = None if for_training else self.elo_ratings
        _cache = self._preload_cache
        home_form_all = self.team_features.get_form_features(
            home_id, 10, "all", as_of_date=as_of_date, elo_ratings=_elo, preload_cache=_cache)
        home_form_home = self.team_features.get_form_features(
            home_id, 10, "home", as_of_date=as_of_date, elo_ratings=_elo, preload_cache=_cache)
        away_form_all = self.team_features.get_form_features(
            away_id, 10, "all", as_of_date=as_of_date, elo_ratings=_elo, preload_cache=_cache)
        away_form_away = self.team_features.get_form_features(
            away_id, 10, "away", as_of_date=as_of_date, elo_ratings=_elo, preload_cache=_cache)

        features.update(self._prefix_dict(home_form_all, "home_overall_"))
        features.update(self._prefix_dict(home_form_home, "home_home_"))
        features.update(self._prefix_dict(away_form_all, "away_overall_"))
        features.update(self._prefix_dict(away_form_away, "away_away_"))

        # Yield event loop so other coroutines (other fixtures) can interleave.
        # Without this, asyncio.gather(concurrency=5) is effectively sequential
        # because synchronous DB calls never yield control.
        await _asyncio.sleep(0)

        # 2. H2H features
        h2h = self.h2h_features.get_h2h_features(
            home_id, away_id, as_of_date=as_of_date, preload_cache=_cache)
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
        from datetime import date as _date, datetime as _datetime
        _effective_date = _standings_date if _standings_date is not None else _date.today()
        _standings_date = _effective_date.replace(day=1)
        # Preload-cache rows store match_date as datetime.datetime (from SQLAlchemy).
        # Normalize _standings_date to datetime so the preload-path comparison
        # `m["match_date"] < as_of_date` doesn't raise TypeError (datetime vs date).
        if not isinstance(_standings_date, _datetime):
            _standings_date = _datetime(_standings_date.year, _standings_date.month, 1)
        home_pos = self.team_features.get_league_position(home_id, league, as_of_date=_standings_date, preload_cache=_cache)
        away_pos = self.team_features.get_league_position(away_id, league, as_of_date=_standings_date, preload_cache=_cache)
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
        home_intl = self.team_features.get_international_form(home_id, as_of_date=as_of_date, preload_cache=_cache)
        away_intl = self.team_features.get_international_form(away_id, as_of_date=as_of_date, preload_cache=_cache)
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
        home_mom = self.team_features.get_momentum_indicators(home_id, as_of_date=as_of_date, preload_cache=_cache)
        away_mom = self.team_features.get_momentum_indicators(away_id, as_of_date=as_of_date, preload_cache=_cache)
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
        _meta2 = (self._preload_cache or {}).get("match_meta", {}).get(match_id)
        if _meta2 is not None:
            _match_date = _meta2["match_date"]
            _home_id = _meta2["home_team_id"]
            _away_id = _meta2["away_team_id"]
            _venue = _meta2["venue"]
        else:
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
            weather = self._get_weather_features(_venue, _match_date, league=league)
            features.update(weather)

        # 15. WC / national-tournament features (group-stage form, knockout flag).
        # Resolved from the match_meta preload when available; falls back to DB.
        _meta_wc = (self._preload_cache or {}).get("match_meta", {}).get(match_id)
        _wc_round = _meta_wc["round"] if _meta_wc else None
        _wc_season = _meta_wc["season"] if _meta_wc else None
        if _wc_round is None or _wc_season is None:
            try:
                with self.db.get_session() as _wsess:
                    _wm = _wsess.get(Match, match_id)
                    if _wm:
                        _wc_round = _wm.round
                        _wc_season = _wm.season
            except Exception:
                pass
        wc_feat = self._get_wc_tournament_features(
            home_id, away_id, league,
            round_name=_wc_round,
            season=_wc_season or "",
            match_date=_match_date,
        )
        features.update(wc_feat)

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

        matches_data = _pc.team_rows(
            self._preload_cache, team_id, limit=num_matches,
            predicate=lambda m: (
                m["home_xg"] is not None
                and (as_of_date is None or m["match_date"] < as_of_date)
                and (venue != "home" or m["home_team_id"] == team_id)
                and (venue != "away" or m["away_team_id"] == team_id)
            ),
        )
        if matches_data is not None:
            if matches_data:
                xg_for_list = []
                xg_against_list = []
                goals_for_list = []
                for m in matches_data:
                    is_home = m["home_team_id"] == team_id
                    if is_home:
                        xg_for_list.append(m["home_xg"] or 0)
                        xg_against_list.append(m["away_xg"] or 0)
                        goals_for_list.append(m["home_goals"] or 0)
                    else:
                        xg_for_list.append(m["away_xg"] or 0)
                        xg_against_list.append(m["home_xg"] or 0)
                        goals_for_list.append(m["away_goals"] or 0)
                xg_avg = sum(xg_for_list) / len(xg_for_list)
                xg_against_avg = sum(xg_against_list) / len(xg_against_list)
                goals_avg = sum(goals_for_list) / len(goals_for_list)
                return {
                    "xg_avg": round(xg_avg, 3),
                    "xg_against_avg": round(xg_against_avg, 3),
                    "xg_overperformance": round(goals_avg - xg_avg, 3),
                    "xg_matches": len(xg_for_list),
                }
            return empty

        with self.db.get_session() as session:
            query = session.query(
                Match.home_team_id, Match.home_xg, Match.away_xg,
                Match.home_goals, Match.away_goals,
            ).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
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

        # Dedicated referee scope. The old code scanned team_history — i.e. only
        # the two teams playing this fixture — so a referee's "last 30 matches"
        # was computed from however many of those two teams' games they happened
        # to officiate, and the live query's 365-day window was not applied
        # either. Both are now handled by _preload_referee_history, and a slice
        # it cannot prove exact returns None so the live query below runs.
        matches_data = _pc.referee_rows(
            self._preload_cache, referee, limit=30,
            predicate=lambda m: (
                m["home_goals"] is not None
                and (as_of_date is None or m["match_date"] < as_of_date)
            ),
        )
        if matches_data is not None:
            if matches_data:
                cards_list = []
                yellow_list = []
                red_list = []
                fouls_total = 0
                fouls_matches = 0
                goals_list = []
                over25_count = 0
                for m in matches_data:
                    yc = (m["home_yellow_cards"] or 0) + (m["away_yellow_cards"] or 0)
                    rc = (m["home_red_cards"] or 0) + (m["away_red_cards"] or 0)
                    cards_list.append(yc + rc)
                    yellow_list.append(yc)
                    red_list.append(rc)
                    total_goals = (m["home_goals"] or 0) + (m["away_goals"] or 0)
                    goals_list.append(total_goals)
                    if total_goals > 2.5:
                        over25_count += 1
                    hf = m["home_fouls"] or 0
                    af = m["away_fouls"] or 0
                    if hf > 0 or af > 0:
                        fouls_total += hf + af
                        fouls_matches += 1
                n = len(matches_data)
                return {
                    "referee_cards_per_match_avg": round(sum(cards_list) / n, 2),
                    "referee_fouls_per_match_avg": round(fouls_total / fouls_matches, 2) if fouls_matches else 0.0,
                    "referee_goals_per_match_avg": round(sum(goals_list) / n, 2),
                    "referee_over25_rate": round(over25_count / n, 3),
                    "referee_avg_yellow_cards": round(sum(yellow_list) / n, 2),
                    "referee_avg_red_cards": round(sum(red_list) / n, 2),
                    "referee_matches": n,
                }
            return empty

        with self.db.get_session() as session:
            _ref_cutoff = _date.today() - timedelta(days=365)
            query = session.query(
                Match.home_yellow_cards, Match.away_yellow_cards,
                Match.home_red_cards, Match.away_red_cards,
                Match.home_goals, Match.away_goals,
                Match.home_fouls, Match.away_fouls,
            ).filter(
                Match.referee == referee,
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
                Match.match_date >= _ref_cutoff,
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

        NOT a preferred-bookmaker lookup. Each market's probability is the
        PER-OUTCOME MEDIAN across every book whose overround falls inside the
        market's declared band (src.data.market_spec.OVERROUND_3WAY / _2WAY).
        A book outside the band is DROPPED, not normalised — normalising a
        two-way pair plus a genuine draw leg produces a plausible-looking
        distribution of the wrong shape.

        This docstring used to read "the preferred bookmaker order is: Bet365 →
        Pinnacle → any". That was replaced in Stage 4 and the sentence was left
        behind. On 2026-08-25 it caused Stage 18 to be halted on the false
        finding that the model's primary input was 92% contaminated. See the
        guard-design notes: a stale docstring is a definition read as an
        occurrence, and it survives greps for the thing it misdescribes.
        """
        defaults = {
            # 1X2
            "home_implied_prob": 1/3,
            "draw_implied_prob": 1/3,
            "away_implied_prob": 1/3,
            "bookmaker_available": 0,
            "bookmaker_consensus_books": 0,
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
            bk_data: dict = defaultdict(dict)
            _cached_odds = (self._preload_cache or {}).get("odds", {}).get(match_id)
            if _cached_odds is not None:
                _markets = {"1X2", "over_under", "btts", "team_goals"}
                for row in _cached_odds:
                    if row["market_type"] in _markets:
                        bk_data[(row["market_type"], row["bookmaker"])][row["selection"]] = row["odds_value"]
                if not bk_data:
                    return defaults
            else:
                with self.db.get_session() as session:
                    rows = session.query(
                        Odds.market_type, Odds.bookmaker,
                        Odds.selection, Odds.odds_value,
                    ).filter(
                        Odds.match_id == match_id,
                        Odds.market_type.in_(["1X2", "over_under", "btts", "team_goals"]),
                    ).all()

                    if not rows:
                        return defaults

                    for row in rows:
                        bk_data[(row.market_type, row.bookmaker)][row.selection] = row.odds_value

            result = dict(defaults)

            # Consensus de-vigging.
            #
            # This used to de-vig ONE bookmaker, chosen Bet365 -> Pinnacle -> any.
            # Two problems, both measured on 2026-08-07:
            #
            #  1. Bet365's stored 1X2 was corrupt for all 2,486 matches in the
            #     database (median overround 1.3524 — see the "Home/Away" note in
            #     apifootball_scraper.BET_TYPE_MAP). Being first in the priority
            #     order, it was the source of home/draw/away_implied_prob for
            #     nearly every fixture, and those features feed both ML training
            #     and the 60% bookmaker blend.
            #  2. The probability came from one book while the EV in
            #     value_calculator used the MEDIAN price across all books. The
            #     median price sat 8.3% above the reference book's on average
            #     (p90 +36.9%), and every point of that gap entered claimed EV as
            #     if it were edge.
            #
            # Taking the per-outcome median of de-vigged probabilities across
            # every book that prices the market fixes both: one bad book can no
            # longer dominate, and the probability now comes from the same
            # cross-book consensus as the price it will be compared against.
            #
            # An overround plausibility gate runs first, so a book whose market
            # does not sum to a believable figure is excluded outright rather
            # than averaged in.

            def _devig(od, keys, lo, hi):
                """De-vig one book's market. None when a leg is missing or the
                overround is outside [lo, hi] — an implausible book is dropped,
                not silently normalised into a plausible-looking answer."""
                prices = []
                for alts in keys:
                    v = next((od[k] for k in alts if od.get(k)), None)
                    if not v or v <= 1.0:
                        return None
                    prices.append(v)
                inv = [1.0 / p for p in prices]
                overround = sum(inv)
                if not (lo <= overround <= hi):
                    return None
                return [i / overround for i in inv]

            def _consensus(market_type, keys, lo, hi):
                """Median de-vigged probability per outcome across all books."""
                per_book = []
                for (mt, _bk), od in bk_data.items():
                    if mt != market_type:
                        continue
                    probs = _devig(od, keys, lo, hi)
                    if probs:
                        per_book.append(probs)
                if not per_book:
                    return None
                cols = list(zip(*per_book))
                med = []
                for col in cols:
                    s = sorted(col)
                    n = len(s)
                    med.append(s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)
                # Medians taken per outcome need not sum to 1 — renormalise.
                total = sum(med)
                if total <= 0:
                    return None
                return [round(v / total, 4) for v in med], len(per_book)

            # Plausible overround bands. A 3-way book runs ~1.02-1.15 and a 2-way
            # ~1.01-1.12; the ceilings are deliberately generous so that only
            # genuinely broken markets (the 1.35 above) are rejected.
            from src.data.market_spec import (
                OVERROUND_3WAY as _OR3, OVERROUND_2WAY as _OR2)

            # ── 1X2 ──────────────────────────────────────────────────────────
            r = _consensus(
                "1X2",
                [("Home", "Home Win"), ("Draw",), ("Away", "Away Win")],
                *_OR3,
            )
            if r:
                (h, d, a), n_books = r
                result["home_implied_prob"] = h
                result["draw_implied_prob"] = d
                result["away_implied_prob"] = a
                result["bookmaker_available"] = 1
                result["bookmaker_consensus_books"] = n_books

            # ── Over/Under ────────────────────────────────────────────────────
            r25 = _consensus("over_under", [("Over 2.5",), ("Under 2.5",)], *_OR2)
            if r25:
                (o, u), _ = r25
                result["over25_implied_prob"] = o
                result["under25_implied_prob"] = u
                result["goals_bookmaker_available"] = 1
            r15 = _consensus("over_under", [("Over 1.5",), ("Under 1.5",)], *_OR2)
            if r15:
                (o, u), _ = r15
                result["over15_implied_prob"] = o
                result["under15_implied_prob"] = u
                result["goals_bookmaker_available"] = 1

            # ── BTTS ─────────────────────────────────────────────────────────
            rb = _consensus("btts", [("Yes", "BTTS Yes"), ("No", "BTTS No")], *_OR2)
            if rb:
                (y, n) = rb[0]
                result["btts_yes_implied_prob"] = y
                result["btts_no_implied_prob"] = n
                result["btts_bookmaker_available"] = 1

            # ── Team goal lines ───────────────────────────────────────────────
            rh = _consensus(
                "team_goals", [("Home Over 1.5",), ("Home Under 1.5",)], *_OR2)
            if rh:
                result["home_over15_implied_prob"] = rh[0][0]
                result["team_goals_bookmaker_available"] = 1
            ra = _consensus(
                "team_goals", [("Away Over 1.5",), ("Away Under 1.5",)], *_OR2)
            if ra:
                result["away_over15_implied_prob"] = ra[0][0]
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
            _cached_odds = (self._preload_cache or {}).get("odds", {}).get(match_id)
            if _cached_odds is not None:
                rows = [
                    r for r in _cached_odds
                    if r.get("opening_odds") is not None and r["bookmaker"] != "Flashscore"
                ]
            else:
                with self.db.get_session() as session:
                    db_rows = session.query(
                        Odds.market_type, Odds.selection,
                        Odds.odds_value, Odds.opening_odds, Odds.bookmaker,
                    ).filter(
                        Odds.match_id == match_id,
                        Odds.opening_odds.isnot(None),
                        Odds.bookmaker != "Flashscore",
                    ).all()
                    rows = [
                        {
                            "market_type": r.market_type, "selection": r.selection,
                            "odds_value": r.odds_value, "opening_odds": r.opening_odds,
                            "bookmaker": r.bookmaker,
                        }
                        for r in db_rows
                    ]

            if not rows:
                return defaults

            # Find movements for key selections
            movements = {}
            for row in rows:
                opening = row["opening_odds"]
                current = row["odds_value"]
                if opening and opening > 0 and current > 0:
                    pct_change = (current - opening) / opening
                    key = (row["market_type"], row["selection"])
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

        cache_key = (league, as_of_date)
        if cache_key in self._league_features_cache:
            return self._league_features_cache[cache_key]

        try:
            with self.db.get_session() as session:
                query = session.query(
                    Match.home_goals, Match.away_goals,
                ).filter(
                    Match.league == league,
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
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

    @staticmethod
    def _compute_situational_from_rows(rows: list, team_id: int, match_date) -> dict:
        """Pure function: compute rest/fatigue features from a list of plain
        match-history rows (each a dict with keys home_team_id, away_team_id,
        match_date, home_goals, away_goals, regulation_home_goals,
        regulation_away_goals).

        Extracted from the cached/live branches so both paths share one
        source of truth — previously the same arithmetic lived twice.
        """
        defaults = {
            "rest_days": 7, "midweek_flag": 0,
            "matches_14d": 0, "matches_21d": 0, "matches_30d": 0,
            "fatigue_index": 0.0, "short_rest_count": 0,
        }
        if not rows:
            return defaults

        prev = rows[0]
        delta = (match_date - prev["match_date"]).days
        rest_days = min(delta, 21)
        midweek_flag = 1 if prev["match_date"].weekday() in (1, 2, 3) else 0

        matches_14d = matches_21d = matches_30d = 0
        short_rest_count = 0
        extra_time_recent = 0
        match_dates = []
        for m in rows:
            days_before = (match_date - m["match_date"]).days
            if days_before <= 14:
                matches_14d += 1
            if days_before <= 21:
                matches_21d += 1
            if days_before <= 30:
                matches_30d += 1
            match_dates.append(m["match_date"])
            # Extra time: regulation score recorded and differs from final.
            if (m.get("regulation_home_goals") is not None
                    and m.get("home_goals") is not None
                    and (m["regulation_home_goals"] != m["home_goals"]
                         or m["regulation_away_goals"] != m["away_goals"])):
                if days_before <= 14:
                    extra_time_recent += 1

        for i in range(len(match_dates) - 1):
            gap = (match_dates[i] - match_dates[i + 1]).days
            if gap < 4:
                short_rest_count += 1

        # Composite fatigue index (0-1 scale).
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

    def _get_situational_features(self, team_id: int, match_date) -> dict:
        """Return rest days, midweek flag, and cumulative fatigue index.

        Reads from preload cache when available; otherwise queries the DB
        for the team's last 10 matches before `match_date`. Both branches
        delegate to `_compute_situational_from_rows` for the actual math.
        """
        defaults = {
            "rest_days": 7, "midweek_flag": 0,
            "matches_14d": 0, "matches_21d": 0, "matches_30d": 0,
            "fatigue_index": 0.0, "short_rest_count": 0,
        }
        try:
            recent_cached = _pc.team_rows(
                self._preload_cache, team_id, limit=10,
                predicate=lambda m: m["match_date"] < match_date,
            )
            if recent_cached is not None:
                return self._compute_situational_from_rows(recent_cached, team_id, match_date)

            with self.db.get_session() as session:
                # Fetch last 10 matches (enough to cover 30 days for busy teams)
                recent = session.query(
                    Match.match_date, Match.home_team_id, Match.away_team_id,
                    Match.home_goals, Match.away_goals,
                    Match.regulation_home_goals, Match.regulation_away_goals,
                ).filter(
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
                    Match.match_date < match_date,
                    or_(Match.home_team_id == team_id, Match.away_team_id == team_id),
                ).order_by(Match.match_date.desc()).limit(10).all()
                rows = [
                    {
                        "match_date": m.match_date,
                        "home_team_id": m.home_team_id,
                        "away_team_id": m.away_team_id,
                        "home_goals": m.home_goals,
                        "away_goals": m.away_goals,
                        "regulation_home_goals": m.regulation_home_goals,
                        "regulation_away_goals": m.regulation_away_goals,
                    }
                    for m in recent
                ]
            return self._compute_situational_from_rows(rows, team_id, match_date)
        except Exception as e:
            logger.warning(f"Situational features failed for team {team_id}: {e}")
            return defaults

    # ISO 3166-1 alpha-2 country codes derived from league slug prefix.
    # Used to disambiguate geocoding when a city name exists in multiple countries.
    _LEAGUE_COUNTRY_CODES: dict = {
        "scotland/": "GB", "england/": "GB", "wales/": "GB", "northern-ireland/": "GB",
        "france/": "FR", "germany/": "DE", "italy/": "IT", "spain/": "ES",
        "portugal/": "PT", "netherlands/": "NL", "belgium/": "BE", "turkey/": "TR",
        "greece/": "GR", "norway/": "NO", "sweden/": "SE", "denmark/": "DK",
        "austria/": "AT", "switzerland/": "CH", "poland/": "PL", "romania/": "RO",
        "czech/": "CZ", "russia/": "RU", "ukraine/": "UA", "croatia/": "HR",
        "serbia/": "RS", "hungary/": "HU", "slovakia/": "SK", "bulgaria/": "BG",
        "finland/": "FI", "ireland/": "IE", "scotland/": "GB",
    }

    @classmethod
    def _league_country_code(cls, league: str) -> str | None:
        if not league:
            return None
        for prefix, code in cls._LEAGUE_COUNTRY_CODES.items():
            if league.startswith(prefix):
                return code
        return None

    def _get_weather_features(self, venue, match_date, league: str = None) -> dict:
        """Return weather features for the match venue and date.

        Uses Open-Meteo free API (no key). Returns neutral defaults on failure.
        Can be disabled via models.weather_features_enabled: false in config.
        A league-derived country_code hint is passed to the geocoder to avoid
        false matches (e.g. Dunfermline → Scotland, not Illinois).
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
            country_code = self._league_country_code(league)
            return self._weather_service.get_match_weather(venue, md, country_code=country_code)
        except Exception as exc:
            logger.debug(f"Weather features failed: {exc}")
            return defaults

    def _get_wc_tournament_features(
        self,
        home_id: int,
        away_id: int,
        league: str,
        round_name: str,
        season: str,
        match_date,
    ) -> dict:
        """WC-specific features: group stage standing, tournament form, round stage.

        For group stage matches these features capture how many points / goal
        difference each team has accumulated so far in the tournament — a proxy
        for desperation, confidence, and qualification pressure.

        Round classification:
          - "Group Stage - N" → wc_is_group_stage=1, wc_is_knockout=0
          - All other rounds  → wc_is_group_stage=0, wc_is_knockout=1
        """
        defaults = {
            "wc_is_group_stage": 0, "wc_is_knockout": 0,
            "wc_home_points": 0, "wc_away_points": 0,
            "wc_home_gd": 0, "wc_away_gd": 0,
            "wc_home_matches_played": 0, "wc_away_matches_played": 0,
            "wc_home_goals_for": 0, "wc_away_goals_for": 0,
            "wc_points_diff": 0, "wc_gd_diff": 0,
        }

        try:
            from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
            if league not in NATIONAL_TEAM_LEAGUES:
                return defaults

            # Determine round stage.
            rn = (round_name or "").lower()
            is_group = "group stage" in rn
            defaults["wc_is_group_stage"] = int(is_group)
            defaults["wc_is_knockout"] = int(not is_group)

            # Query all completed WC matches for this season played BEFORE this match.
            # Extract primitive tuples INSIDE the session — Match instances become
            # detached once the `with` block closes and lazy attribute access then
            # raises DetachedInstanceError (silently falling back to zeros).
            cutoff = match_date if match_date else None
            with self.db.get_session() as session:
                from sqlalchemy import or_ as _or
                q = session.query(
                    Match.home_team_id, Match.away_team_id,
                    Match.home_goals, Match.away_goals,
                ).filter(
                    Match.league == league,
                    Match.season == season,
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    Match.training_exclusion_reason.is_(None),  # s5.3 — learns/measures
                    _or(
                        Match.home_team_id.in_([home_id, away_id]),
                        Match.away_team_id.in_([home_id, away_id]),
                    ),
                )
                if cutoff:
                    q = q.filter(Match.match_date < cutoff)
                past_rows = [
                    (r[0], r[1], r[2], r[3]) for r in q.all()
                ]

            def _team_stats(team_id):
                pts = gd = gf = mp = 0
                for h_tid, a_tid, hg, ag in past_rows:
                    if h_tid == team_id:
                        mp += 1
                        gf += hg
                        gd += hg - ag
                        pts += 3 if hg > ag else (1 if hg == ag else 0)
                    elif a_tid == team_id:
                        mp += 1
                        gf += ag
                        gd += ag - hg
                        pts += 3 if ag > hg else (1 if ag == hg else 0)
                return pts, gd, gf, mp

            h_pts, h_gd, h_gf, h_mp = _team_stats(home_id)
            a_pts, a_gd, a_gf, a_mp = _team_stats(away_id)

            return {
                "wc_is_group_stage": int(is_group),
                "wc_is_knockout": int(not is_group),
                "wc_home_points": h_pts,
                "wc_away_points": a_pts,
                "wc_home_gd": h_gd,
                "wc_away_gd": a_gd,
                "wc_home_matches_played": h_mp,
                "wc_away_matches_played": a_mp,
                "wc_home_goals_for": h_gf,
                "wc_away_goals_for": a_gf,
                "wc_points_diff": h_pts - a_pts,
                "wc_gd_diff": h_gd - a_gd,
            }
        except Exception as exc:
            logger.debug(f"WC tournament features failed: {exc}")
            return defaults

    def _prefix_dict(self, d: dict, prefix: str) -> dict:
        """Add a prefix to all dictionary keys."""
        return {f"{prefix}{k}": v for k, v in d.items()}
