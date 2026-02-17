"""API-Football scraper for xG, advanced stats, fixture data, and odds.

Uses the direct API-Football endpoint (v3.football.api-sports.io).
Free tier: 100 requests/day, seasons 2022-2024.
Includes real bookmaker odds via the /odds endpoint.
"""

import asyncio
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Match, Team, Odds
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

API_FOOTBALL_BASE = "https://v3.football.api-sports.io"

# Map our internal league keys to API-Football league IDs
LEAGUE_ID_MAP = {
    "england/premier-league": 39,
    "england/championship": 40,
    "spain/laliga": 140,
    "spain/laliga2": 141,
    "germany/bundesliga": 78,
    "germany/2-bundesliga": 79,
    "italy/serie-a": 135,
    "italy/serie-b": 136,
    "france/ligue-1": 61,
    "france/ligue-2": 62,
    "netherlands/eredivisie": 88,
    "portugal/primeira-liga": 94,
    "belgium/jupiler-pro-league": 144,
    "turkey/super-lig": 203,
    "scotland/premiership": 179,
    "austria/bundesliga": 218,
    "switzerland/super-league": 207,
    "greece/super-league": 197,
    "denmark/superliga": 120,
    "norway/eliteserien": 103,
    "sweden/allsvenskan": 113,
    "finland/veikkausliiga": 244,
    "poland/ekstraklasa": 106,
    "romania/liga-1": 283,
    "champions-league": 2,
    "europa-league": 3,
    "europa-conference-league": 848,
}

# Reverse map: API-Football league ID -> our internal league key
ID_TO_LEAGUE = {v: k for k, v in LEAGUE_ID_MAP.items()}

# Leagues to fetch odds for first (highest priority)
PRIORITY_LEAGUES = [
    "england/premier-league", "spain/laliga", "germany/bundesliga",
    "italy/serie-a", "france/ligue-1", "champions-league",
    "europa-league", "netherlands/eredivisie", "portugal/primeira-liga",
    "belgium/jupiler-pro-league", "turkey/super-lig", "scotland/premiership",
]

# API-Football bet type IDs -> our market_type and selection mapping
BET_TYPE_MAP = {
    "Match Winner": {
        "market_type": "1X2",
        "selections": {"Home": "Home", "Draw": "Draw", "Away": "Away"},
    },
    "Home/Away": {
        "market_type": "1X2",
        "selections": {"Home": "Home", "Draw": "Draw", "Away": "Away"},
    },
    "Goals Over/Under": {
        "market_type": "over_under",
        "selections": {
            "Over 1.5": "Over 1.5", "Under 1.5": "Under 1.5",
            "Over 2.5": "Over 2.5", "Under 2.5": "Under 2.5",
            "Over 3.5": "Over 3.5", "Under 3.5": "Under 3.5",
        },
    },
    "Both Teams Score": {
        "market_type": "btts",
        "selections": {"Yes": "Yes", "No": "No"},
    },
}


class APIFootballScraper(BaseScraper):
    """Fetches xG, match statistics, fixtures, and odds from API-Football."""

    # Request budget allocation (out of 100/day)
    BUDGET_RESULTS = 4
    BUDGET_FIXTURES = 2
    BUDGET_XG = 15
    BUDGET_ODDS = 70
    BUDGET_RESERVE = 9

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = (
            os.environ.get("API_FOOTBALL_KEY")
            or self.config.get("data_sources.apifootball_key", "")
        )
        self.enabled = bool(self.api_key)
        self.db = get_db()
        self._requests_today = 0
        self._daily_limit = 100  # Free tier

    async def _api_get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make an authenticated GET request to API-Football.

        Uses BaseScraper.fetch_json for retry + circuit breaker support.
        """
        if self._requests_today >= self._daily_limit:
            logger.warning("API-Football daily request limit reached, skipping")
            return None

        url = f"{API_FOOTBALL_BASE}{endpoint}"
        headers = {"x-apisports-key": self.api_key}

        try:
            data = await self.fetch_json(url, params=params, headers=headers)
            self._requests_today += 1
            errors = data.get("errors", {})
            if errors:
                if "rateLimit" in errors:
                    logger.debug("API-Football rate limited, waiting 12s")
                    await asyncio.sleep(12)
                else:
                    logger.error(f"API-Football errors: {errors}")
                return None
            # Pace requests to stay under 10/min free-tier limit
            await asyncio.sleep(7)
            return data
        except Exception as e:
            self._requests_today += 1
            logger.error(f"API-Football request failed: {e}")
            return None

    async def update(self):
        """Run the full API-Football update: recent results + today/tomorrow fixtures + xG."""
        if not self.enabled:
            logger.warning("API-Football key not set. Skipping.")
            return

        logger.info("Starting API-Football update")

        # 1. Fetch yesterday's results (so settle can work)
        await self.fetch_recent_results(days_back=2)

        # 2. Fetch today's and tomorrow's fixtures
        today = date.today()
        tomorrow = today + timedelta(days=1)
        await self._fetch_fixtures_by_date(today)
        await self._fetch_fixtures_by_date(tomorrow)

        # 3. Backfill xG for recent completed matches that don't have it
        await self._backfill_xg(days_back=3)

        # 4. Fetch real bookmaker odds for upcoming fixtures
        await self.fetch_upcoming_odds()

        logger.info(f"API-Football update complete ({self._requests_today} requests used)")

    async def fetch_recent_results(self, days_back: int = 3):
        """Fetch results for recent past days to update scores in the DB.

        Uses 1 API request per day. Free tier only allows yesterday + today,
        so older dates are silently skipped on error.
        """
        if not self.enabled:
            return

        today = date.today()
        for d in range(1, days_back + 1):
            past_date = today - timedelta(days=d)
            try:
                await self._fetch_fixtures_by_date(past_date)
            except Exception:
                break  # Free plan date limit reached

    async def _fetch_fixtures_by_date(self, target_date: date):
        """Fetch all fixtures for a given date and upsert into DB."""
        date_str = target_date.strftime("%Y-%m-%d")
        data = await self._api_get("/fixtures", {"date": date_str})
        if not data:
            return

        fixtures = data.get("response", [])
        logger.info(f"API-Football: {len(fixtures)} fixtures on {date_str}")

        created = 0
        updated = 0

        for fix in fixtures:
            league_id = fix.get("league", {}).get("id")
            if league_id not in ID_TO_LEAGUE:
                continue  # Skip leagues we don't track

            league_key = ID_TO_LEAGUE[league_id]
            fixture_id = fix.get("fixture", {}).get("id")
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")
            match_ts = fix.get("fixture", {}).get("timestamp")
            status_short = fix.get("fixture", {}).get("status", {}).get("short", "")

            if not home_name or not away_name:
                continue

            match_dt = datetime.utcfromtimestamp(match_ts) if match_ts else None
            if not match_dt:
                continue

            # Get or create teams (returns IDs)
            home_team_id = self._get_or_create_team_id(home_name, league_key)
            away_team_id = self._get_or_create_team_id(away_name, league_key)

            # Check for existing match
            match_id = self._find_match_id(home_team_id, away_team_id, match_dt)

            goals = fix.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            is_finished = status_short in ("FT", "AET", "PEN")
            is_fixture = status_short in ("NS", "TBD", "")

            if match_id:
                # Update existing match
                with self.db.get_session() as session:
                    match = session.get(Match, match_id)
                    if match:
                        match.apifootball_id = fixture_id
                        if is_finished and home_goals is not None:
                            match.home_goals = home_goals
                            match.away_goals = away_goals
                            match.is_fixture = False
                        session.commit()
                updated += 1
            else:
                # Create new match
                with self.db.get_session() as session:
                    match = Match(
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        match_date=match_dt,
                        league=league_key,
                        season=self._get_season(match_dt),
                        is_fixture=is_fixture,
                        apifootball_id=fixture_id,
                    )
                    if is_finished and home_goals is not None:
                        match.home_goals = home_goals
                        match.away_goals = away_goals
                        match.is_fixture = False
                    session.add(match)
                    session.commit()
                created += 1

        logger.info(f"API-Football fixtures {date_str}: {created} created, {updated} updated")

    async def _backfill_xg(self, days_back: int = 3):
        """Fetch xG and stats for recent matches that don't have xG data yet."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            matches = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.home_xg.is_(None),
                Match.match_date >= cutoff,
                Match.apifootball_id.isnot(None),
            ).limit(20).all()

            if not matches:
                logger.debug("No matches need xG backfill")
                return

            logger.info(f"Backfilling xG for {len(matches)} recent matches")
            match_data = [(m.id, m.apifootball_id) for m in matches]

        for match_id, fixture_id in match_data:
            if self._requests_today >= self._daily_limit - 5:
                logger.warning("Approaching API limit, stopping xG backfill")
                break

            stats = await self._fetch_fixture_stats(fixture_id)
            if stats:
                self._update_match_stats(match_id, stats)

    async def _fetch_fixture_stats(self, fixture_id: int) -> Optional[Dict]:
        """Fetch detailed statistics for a single fixture."""
        data = await self._api_get("/fixtures/statistics", {"fixture": fixture_id})
        if not data:
            return None

        response = data.get("response", [])
        if len(response) < 2:
            return None

        result = {"home": {}, "away": {}}
        for i, side in enumerate(["home", "away"]):
            team_stats = response[i].get("statistics", [])
            for stat in team_stats:
                stat_type = stat.get("type", "")
                stat_value = stat.get("value")
                if stat_type == "expected_goals" and stat_value is not None:
                    try:
                        result[side]["xg"] = float(stat_value)
                    except (ValueError, TypeError):
                        pass
                elif stat_type == "Total Shots":
                    result[side]["shots"] = _safe_int(stat_value)
                elif stat_type == "Shots on Goal":
                    result[side]["shots_on_target"] = _safe_int(stat_value)
                elif stat_type == "Ball Possession":
                    result[side]["possession"] = _parse_pct(stat_value)
                elif stat_type == "Corner Kicks":
                    result[side]["corners"] = _safe_int(stat_value)
                elif stat_type == "Fouls":
                    result[side]["fouls"] = _safe_int(stat_value)
                elif stat_type == "Yellow Cards":
                    result[side]["yellow_cards"] = _safe_int(stat_value)
                elif stat_type == "Red Cards":
                    result[side]["red_cards"] = _safe_int(stat_value)

        return result

    def _update_match_stats(self, match_id: int, stats: Dict):
        """Write xG and advanced stats back to the Match row."""
        home = stats.get("home", {})
        away = stats.get("away", {})

        with self.db.get_session() as session:
            match = session.get(Match, match_id)
            if not match:
                return

            # xG
            if "xg" in home:
                match.home_xg = home["xg"]
            if "xg" in away:
                match.away_xg = away["xg"]

            # Fill other stats if they're missing
            if match.home_shots is None and "shots" in home:
                match.home_shots = home["shots"]
            if match.away_shots is None and "shots" in away:
                match.away_shots = away["shots"]
            if match.home_shots_on_target is None and "shots_on_target" in home:
                match.home_shots_on_target = home["shots_on_target"]
            if match.away_shots_on_target is None and "shots_on_target" in away:
                match.away_shots_on_target = away["shots_on_target"]
            if match.home_possession is None and "possession" in home:
                match.home_possession = home["possession"]
            if match.away_possession is None and "possession" in away:
                match.away_possession = away["possession"]
            if match.home_corners is None and "corners" in home:
                match.home_corners = home["corners"]
            if match.away_corners is None and "corners" in away:
                match.away_corners = away["corners"]
            if match.home_fouls is None and "fouls" in home:
                match.home_fouls = home["fouls"]
            if match.away_fouls is None and "fouls" in away:
                match.away_fouls = away["fouls"]
            if match.home_yellow_cards is None and "yellow_cards" in home:
                match.home_yellow_cards = home["yellow_cards"]
            if match.away_yellow_cards is None and "yellow_cards" in away:
                match.away_yellow_cards = away["yellow_cards"]
            if match.home_red_cards is None and "red_cards" in home:
                match.home_red_cards = home["red_cards"]
            if match.away_red_cards is None and "red_cards" in away:
                match.away_red_cards = away["red_cards"]

            session.commit()
            logger.debug(f"Updated stats for match {match_id} (xG: {home.get('xg')}-{away.get('xg')})")

    def _get_or_create_team_id(self, name: str, league: str) -> int:
        """Find existing team by name or create a new one. Returns team ID."""
        with self.db.get_session() as session:
            team = session.query(Team).filter_by(name=name).first()
            if team:
                return team.id

            # Try fuzzy match (common name variations)
            team = session.query(Team).filter(
                Team.name.ilike(f"%{name}%")
            ).first()
            if team:
                return team.id

            # Create new team
            country = league.split("/")[0].title() if "/" in league else ""
            team = Team(name=name, league=league, country=country)
            session.add(team)
            session.commit()
            logger.info(f"Created new team: {name} ({league})")
            return team.id

    def _find_match_id(self, home_id: int, away_id: int, match_dt: datetime) -> Optional[int]:
        """Find an existing match within a 2-day window. Returns match ID or None."""
        with self.db.get_session() as session:
            window = timedelta(days=1)
            match = session.query(Match).filter(
                Match.home_team_id == home_id,
                Match.away_team_id == away_id,
                Match.match_date >= match_dt - window,
                Match.match_date <= match_dt + window,
            ).first()
            return match.id if match else None

    def _get_season(self, dt: datetime) -> str:
        """Determine the season string from a match date."""
        year = dt.year
        month = dt.month
        if month >= 7:
            return f"{year % 100:02d}{(year + 1) % 100:02d}"
        else:
            return f"{(year - 1) % 100:02d}{year % 100:02d}"

    # ---- Odds fetching ----

    async def fetch_upcoming_odds(self):
        """Fetch real bookmaker odds for upcoming fixtures from API-Football.

        Prioritizes top leagues and respects daily request budget.
        """
        if not self.enabled:
            return 0

        odds_budget = self._daily_limit - self._requests_today - self.BUDGET_RESERVE
        odds_budget = min(odds_budget, self.BUDGET_ODDS)

        if odds_budget <= 0:
            logger.warning("No API budget remaining for odds fetching")
            return 0

        # Get upcoming fixtures that have an apifootball_id
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() + timedelta(days=2)
            upcoming = session.query(Match).filter(
                Match.is_fixture == True,
                Match.apifootball_id.isnot(None),
                Match.match_date <= cutoff,
            ).all()

            if not upcoming:
                logger.debug("No upcoming fixtures need odds")
                return 0

            # Build list of (match_id, apifootball_id, league) tuples
            fixture_list = [
                (m.id, m.apifootball_id, m.league or "")
                for m in upcoming
            ]

        # Sort by league priority
        def league_priority(item):
            league = item[2]
            try:
                return PRIORITY_LEAGUES.index(league)
            except ValueError:
                return len(PRIORITY_LEAGUES)

        fixture_list.sort(key=league_priority)

        # Check which fixtures already have odds
        with self.db.get_session() as session:
            fixtures_with_odds = set()
            for match_id, _, _ in fixture_list:
                has_odds = session.query(Odds).filter_by(match_id=match_id).first()
                if has_odds:
                    fixtures_with_odds.add(match_id)

        # Filter to fixtures without odds
        need_odds = [
            (mid, fid, lg) for mid, fid, lg in fixture_list
            if mid not in fixtures_with_odds
        ]

        if not need_odds:
            logger.debug("All upcoming fixtures already have odds")
            return 0

        fetched = 0
        for match_id, fixture_id, league in need_odds:
            if fetched >= odds_budget:
                logger.info(f"Odds budget exhausted after {fetched} fixtures")
                break

            odds_data = await self._fetch_fixture_odds(fixture_id)
            if odds_data:
                count = self._save_fixture_odds(match_id, odds_data)
                if count > 0:
                    fetched += 1
                    logger.debug(
                        f"Saved {count} odds for fixture {fixture_id} "
                        f"(match {match_id}, {league})"
                    )

        logger.info(f"Fetched odds for {fetched} upcoming fixtures ({self._requests_today} API requests used)")
        return fetched

    async def _fetch_fixture_odds(self, fixture_id: int) -> Optional[List]:
        """Fetch bookmaker odds for a specific fixture."""
        data = await self._api_get("/odds", {"fixture": fixture_id})
        if not data:
            return None
        return data.get("response", [])

    def _save_fixture_odds(self, match_id: int, odds_response: list) -> int:
        """Parse API-Football odds response and save to database.

        Returns count of odds records created.
        """
        count = 0

        with self.db.get_session() as session:
            for entry in odds_response:
                bookmakers = entry.get("bookmakers", [])
                for bookie in bookmakers:
                    bookie_name = bookie.get("name", "Unknown")
                    bets = bookie.get("bets", [])

                    for bet in bets:
                        bet_name = bet.get("name", "")
                        bet_mapping = BET_TYPE_MAP.get(bet_name)
                        if not bet_mapping:
                            continue

                        market_type = bet_mapping["market_type"]
                        selection_map = bet_mapping["selections"]

                        for value in bet.get("values", []):
                            raw_selection = value.get("value", "")
                            odds_str = value.get("odd", "")

                            selection = selection_map.get(raw_selection)
                            if not selection:
                                continue

                            try:
                                odds_value = float(odds_str)
                            except (ValueError, TypeError):
                                continue

                            if odds_value <= 1.0:
                                continue

                            # Skip duplicate
                            existing = session.query(Odds).filter_by(
                                match_id=match_id,
                                bookmaker=bookie_name,
                                market_type=market_type,
                                selection=selection,
                            ).first()

                            if existing:
                                # Update if odds changed
                                if existing.odds_value != odds_value:
                                    existing.odds_value = odds_value
                                    existing.timestamp = datetime.utcnow()
                                continue

                            odds = Odds(
                                match_id=match_id,
                                bookmaker=bookie_name,
                                market_type=market_type,
                                selection=selection,
                                odds_value=odds_value,
                            )
                            session.add(odds)
                            count += 1

            session.commit()

        return count

    # ---- League xG backfill ----

    async def fetch_league_xg(self, league_key: str, season: int = 2024,
                               max_fixtures: int = 30) -> int:
        """Fetch xG for completed fixtures in a specific league/season.

        Useful for bulk backfill. Returns count of matches updated.
        """
        league_id = LEAGUE_ID_MAP.get(league_key)
        if not league_id:
            logger.warning(f"No API-Football league ID for {league_key}")
            return 0

        data = await self._api_get("/fixtures", {
            "league": league_id,
            "season": season,
            "status": "FT",
        })
        if not data:
            return 0

        fixtures = data.get("response", [])
        updated = 0

        for fix in fixtures[:max_fixtures]:
            fixture_id = fix.get("fixture", {}).get("id")
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")

            if not fixture_id or not home_name:
                continue

            # Find matching DB match
            home_team_id = self._get_or_create_team_id(home_name, league_key)
            away_team_id = self._get_or_create_team_id(away_name, league_key)

            match_ts = fix.get("fixture", {}).get("timestamp")
            match_dt = datetime.utcfromtimestamp(match_ts) if match_ts else None
            if not match_dt:
                continue

            match_id = self._find_match_id(home_team_id, away_team_id, match_dt)
            if match_id:
                # Check if xG is missing
                with self.db.get_session() as session:
                    m = session.get(Match, match_id)
                    needs_xg = m and m.home_xg is None

                if needs_xg:
                    with self.db.get_session() as session:
                        m = session.get(Match, match_id)
                        if m:
                            m.apifootball_id = fixture_id
                            session.commit()

                    stats = await self._fetch_fixture_stats(fixture_id)
                    if stats:
                        self._update_match_stats(match_id, stats)
                        updated += 1

            if self._requests_today >= self._daily_limit - 5:
                logger.warning("Approaching API limit, stopping league xG fetch")
                break

        logger.info(f"Updated xG for {updated} matches in {league_key}")
        return updated


def _safe_int(value) -> Optional[int]:
    """Safely convert a stat value to int."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _parse_pct(value) -> Optional[float]:
    """Parse a percentage string like '65%' to float 65.0."""
    if value is None:
        return None
    try:
        return float(str(value).replace("%", ""))
    except (ValueError, TypeError):
        return None
