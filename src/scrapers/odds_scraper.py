"""Odds and scores scraper using The Odds API."""

from datetime import datetime, timedelta
from typing import List, Optional

from sqlalchemy import and_

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Odds, Match, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Map league config names to Odds API sport keys
LEAGUE_TO_SPORT_KEY = {
    # Top 5 + second divisions
    "england/premier-league": "soccer_epl",
    "england/championship": "soccer_efl_champ",
    "spain/laliga": "soccer_spain_la_liga",
    "spain/laliga2": "soccer_spain_segunda_division",
    "germany/bundesliga": "soccer_germany_bundesliga",
    "germany/2-bundesliga": "soccer_germany_bundesliga2",
    "italy/serie-a": "soccer_italy_serie_a",
    "italy/serie-b": "soccer_italy_serie_b",
    "france/ligue-1": "soccer_france_ligue_one",
    "france/ligue-2": "soccer_france_ligue_two",
    # Strong European
    "netherlands/eredivisie": "soccer_netherlands_eredivisie",
    "portugal/primeira-liga": "soccer_portugal_primeira_liga",
    "belgium/jupiler-pro-league": "soccer_belgium_first_div",
    "turkey/super-lig": "soccer_turkey_super_league",
    "scotland/premiership": "soccer_spl",
    "austria/bundesliga": "soccer_austria_bundesliga",
    "switzerland/super-league": "soccer_switzerland_superleague",
    "greece/super-league": "soccer_greece_super_league",
    "denmark/superliga": "soccer_denmark_superliga",
    # Nordic & Eastern European
    "norway/eliteserien": "soccer_norway_eliteserien",
    "sweden/allsvenskan": "soccer_sweden_allsvenskan",
    "finland/veikkausliiga": "soccer_finland_veikkausliiga",
    "poland/ekstraklasa": "soccer_poland_ekstraklasa",
    "romania/liga-1": "soccer_romania_liga_1",
    # European Competitions
    "champions-league": "soccer_uefa_champs_league",
    "europa-league": "soccer_uefa_europa_league",
    "europa-conference-league": "soccer_uefa_europa_conference_league",
}

# Map Odds API market keys to our internal market types
MARKET_MAP = {
    "h2h": "1X2",
    "totals": "over_under",
    "spreads": "asian_handicap",
    "btts": "btts",
}


class OddsScraper(BaseScraper):
    """Collects odds data and match scores from The Odds API."""

    def __init__(self, config=None):
        super().__init__(config)
        self.api_key = self.config.get("data_sources.odds_api_key", "")
        self.enabled = self.config.get("data_sources.odds_api_enabled", False)
        self.leagues = self.config.get("scraping.flashscore_leagues", [])

    async def update(self):
        """Fetch latest odds AND scores for all configured leagues."""
        if not self.enabled or not self.api_key:
            logger.warning("Odds API is disabled or API key not set. Skipping update.")
            return

        logger.info("Starting Odds API update cycle")
        successful = 0
        failed = 0

        for league in self.leagues:
            sport_key = LEAGUE_TO_SPORT_KEY.get(league)
            if not sport_key:
                logger.debug(f"No Odds API mapping for league: {league}")
                continue

            try:
                # Fetch odds (upcoming fixtures)
                await self.fetch_league_odds(sport_key, league)
                # Fetch recent scores/results
                await self.fetch_league_scores(sport_key, league)
                successful += 1
            except Exception as e:
                logger.error(f"Error fetching data for {league}: {e}")
                failed += 1

        logger.info(f"Odds API update complete: {successful} leagues OK, {failed} failed")

    async def fetch_league_odds(self, sport_key: str, league: str, markets: str = "h2h,totals"):
        """Fetch odds for a specific league/sport."""
        url = f"{ODDS_API_BASE}/sports/{sport_key}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": markets,
            "oddsFormat": "decimal",
        }

        data = await self.fetch_json(url, params=params)

        if not data:
            logger.info(f"No odds data for {sport_key}")
            return

        db = get_db()
        odds_count = 0

        with db.get_session() as session:
            for event in data:
                home_team_name = event.get("home_team", "")
                away_team_name = event.get("away_team", "")
                commence_time = event.get("commence_time", "")

                try:
                    match_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    match_date = datetime.now()

                match = self._find_or_create_match(
                    session, home_team_name, away_team_name, match_date, league
                )

                for bookmaker in event.get("bookmakers", []):
                    bookmaker_name = bookmaker.get("title", "unknown")

                    for market in bookmaker.get("markets", []):
                        market_key = market.get("key", "")
                        market_type = MARKET_MAP.get(market_key, market_key)

                        for outcome in market.get("outcomes", []):
                            selection = outcome.get("name", "")
                            price = outcome.get("price", 0.0)
                            point = outcome.get("point")

                            if point is not None:
                                selection = f"{selection} {point}"

                            odds_record = Odds(
                                match_id=match.id,
                                bookmaker=bookmaker_name,
                                market_type=market_type,
                                selection=selection,
                                odds_value=price,
                            )
                            session.add(odds_record)
                            odds_count += 1

        logger.info(f"Saved {odds_count} odds records for {league}")

    async def fetch_league_scores(self, sport_key: str, league: str, days_back: int = 3):
        """Fetch recent match scores/results from the API.

        Args:
            sport_key: The Odds API sport key
            league: Internal league identifier
            days_back: Number of days back to fetch completed results
        """
        url = f"{ODDS_API_BASE}/sports/{sport_key}/scores/"
        params = {
            "apiKey": self.api_key,
            "daysFrom": days_back,
        }

        try:
            data = await self.fetch_json(url, params=params)
        except Exception as e:
            logger.debug(f"Scores endpoint not available for {sport_key}: {e}")
            return

        if not data:
            return

        db = get_db()
        results_count = 0

        with db.get_session() as session:
            for event in data:
                completed = event.get("completed", False)
                if not completed:
                    continue

                home_team_name = event.get("home_team", "")
                away_team_name = event.get("away_team", "")
                commence_time = event.get("commence_time", "")

                try:
                    match_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    match_date = datetime.now()

                # Parse scores
                scores = event.get("scores", [])
                home_goals = None
                away_goals = None

                if scores:
                    for score_entry in scores:
                        name = score_entry.get("name", "")
                        score_val = score_entry.get("score")
                        if score_val is not None:
                            try:
                                if name == home_team_name:
                                    home_goals = int(score_val)
                                elif name == away_team_name:
                                    away_goals = int(score_val)
                            except (ValueError, TypeError):
                                pass

                if home_goals is None or away_goals is None:
                    continue

                # Find or create match and update with result
                match = self._find_or_create_match(
                    session, home_team_name, away_team_name, match_date, league
                )

                match.home_goals = home_goals
                match.away_goals = away_goals
                match.is_fixture = False
                results_count += 1

        if results_count > 0:
            logger.info(f"Updated {results_count} match results for {league}")

    def _find_or_create_match(self, session, home_name: str, away_name: str,
                               match_date: datetime, league: str) -> Match:
        """Find an existing match or create a fixture entry."""
        home_team = session.query(Team).filter_by(name=home_name).first()
        if not home_team:
            home_team = Team(name=home_name, league=league)
            session.add(home_team)
            session.flush()

        away_team = session.query(Team).filter_by(name=away_name).first()
        if not away_team:
            away_team = Team(name=away_name, league=league)
            session.add(away_team)
            session.flush()

        match = session.query(Match).filter(
            and_(
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date.between(
                    match_date - timedelta(hours=24),
                    match_date + timedelta(hours=24),
                ),
            )
        ).first()

        if not match:
            match = Match(
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                match_date=match_date,
                league=league,
                is_fixture=True,
            )
            session.add(match)
            session.flush()

        return match

    async def get_available_sports(self) -> Optional[List[dict]]:
        """List all available sports from the API (useful for debugging)."""
        if not self.api_key:
            return None

        url = f"{ODDS_API_BASE}/sports/"
        params = {"apiKey": self.api_key}
        return await self.fetch_json(url, params=params)
