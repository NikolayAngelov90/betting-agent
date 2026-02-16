"""Historical match data loader from football-data.co.uk CSV files.

Bootstraps the database with seasons of historical results so that
Poisson, Elo, and ML models have enough data for meaningful predictions.

Supports two CSV formats:
  - mmz4281: per-season files for main leagues (columns: HomeTeam, AwayTeam, FTHG, FTAG)
  - new: single all-seasons file for extra leagues (columns: Home, Away, HG, AG, Season)
"""

import csv
import io
from datetime import datetime
from typing import List

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Match, Team
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

# --- Main leagues: per-season CSVs from mmz4281/ directory ---
# URL: https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv
LEAGUE_CSV_MAP = {
    # Top 5 leagues + second divisions
    "england/premier-league": {"code": "E0", "country": "England"},
    "england/championship": {"code": "E1", "country": "England"},
    "spain/laliga": {"code": "SP1", "country": "Spain"},
    "spain/laliga2": {"code": "SP2", "country": "Spain"},
    "germany/bundesliga": {"code": "D1", "country": "Germany"},
    "germany/2-bundesliga": {"code": "D2", "country": "Germany"},
    "italy/serie-a": {"code": "I1", "country": "Italy"},
    "italy/serie-b": {"code": "I2", "country": "Italy"},
    "france/ligue-1": {"code": "F1", "country": "France"},
    "france/ligue-2": {"code": "F2", "country": "France"},
    # Other European leagues (mmz4281 format)
    "netherlands/eredivisie": {"code": "N1", "country": "Netherlands"},
    "portugal/primeira-liga": {"code": "P1", "country": "Portugal"},
    "belgium/jupiler-pro-league": {"code": "B1", "country": "Belgium"},
    "turkey/super-lig": {"code": "T1", "country": "Turkey"},
    "scotland/premiership": {"code": "SC0", "country": "Scotland"},
    "greece/super-league": {"code": "G1", "country": "Greece"},
}

# --- Extra leagues: single all-seasons CSV from /new/ directory ---
# URL: https://www.football-data.co.uk/new/{code}.csv
# Columns: Country, League, Season, Date, Time, Home, Away, HG, AG, Res, ...
EXTRA_LEAGUE_CSV_MAP = {
    "austria/bundesliga": {"code": "AUT", "country": "Austria"},
    "switzerland/super-league": {"code": "SWZ", "country": "Switzerland"},
    "denmark/superliga": {"code": "DNK", "country": "Denmark"},
    "norway/eliteserien": {"code": "NOR", "country": "Norway"},
    "sweden/allsvenskan": {"code": "SWE", "country": "Sweden"},
    "finland/veikkausliiga": {"code": "FIN", "country": "Finland"},
    "poland/ekstraklasa": {"code": "POL", "country": "Poland"},
    "romania/liga-1": {"code": "ROU", "country": "Romania"},
}

# Season codes for mmz4281 format (yy/yy format)
SEASONS = ["2425", "2324", "2223"]

# How many recent seasons to keep from /new/ format files
EXTRA_SEASONS_LIMIT = 3


class HistoricalDataLoader(BaseScraper):
    """Loads historical match results from football-data.co.uk CSVs."""

    def __init__(self, config=None):
        super().__init__(config)

    async def update(self):
        """Load historical data for all configured leagues."""
        await self.load_all_leagues()

    async def load_all_leagues(self, seasons: List[str] = None):
        """Load historical results for all mapped leagues."""
        seasons = seasons or SEASONS
        total_loaded = 0

        # Main leagues (mmz4281 per-season format)
        for league, info in LEAGUE_CSV_MAP.items():
            for season in seasons:
                try:
                    count = await self.load_league_season(league, info["code"], season)
                    total_loaded += count
                except Exception as e:
                    logger.debug(f"No data for {league} season {season}: {e}")

        # Extra leagues (/new/ all-seasons format)
        for league, info in EXTRA_LEAGUE_CSV_MAP.items():
            try:
                count = await self.load_extra_league(league, info["code"])
                total_loaded += count
            except Exception as e:
                logger.debug(f"No data for extra league {league}: {e}")

        logger.info(f"Historical data loading complete: {total_loaded} matches loaded")
        return total_loaded

    async def load_league_season(self, league: str, league_code: str,
                                  season: str) -> int:
        """Load one league-season CSV (mmz4281 format) into the database."""
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

        try:
            csv_text = await self.fetch(url)
        except Exception as e:
            logger.debug(f"Could not fetch {url}: {e}")
            return 0

        if not csv_text or len(csv_text) < 50:
            return 0

        return self._parse_and_save(csv_text, league, season)

    async def load_extra_league(self, league: str, league_code: str) -> int:
        """Load an extra league CSV (/new/ format) with all seasons in one file.

        Filters to only the most recent EXTRA_SEASONS_LIMIT seasons.
        """
        url = f"https://www.football-data.co.uk/new/{league_code}.csv"

        try:
            csv_text = await self.fetch(url)
        except Exception as e:
            logger.debug(f"Could not fetch {url}: {e}")
            return 0

        if not csv_text or len(csv_text) < 50:
            return 0

        return self._parse_and_save_extra(csv_text, league)

    def _parse_and_save(self, csv_text: str, league: str, season: str) -> int:
        """Parse mmz4281 format CSV and save matches to database."""
        db = get_db()
        reader = csv.DictReader(io.StringIO(csv_text))
        count = 0

        with db.get_session() as session:
            for row in reader:
                try:
                    home_name = row.get("HomeTeam", "").strip()
                    away_name = row.get("AwayTeam", "").strip()
                    fthg = row.get("FTHG", "")  # Full Time Home Goals
                    ftag = row.get("FTAG", "")  # Full Time Away Goals
                    date_str = row.get("Date", "")

                    if not home_name or not away_name or not fthg or not ftag:
                        continue

                    home_goals = int(fthg)
                    away_goals = int(ftag)
                    match_date = self._parse_date(date_str)

                    if not match_date:
                        continue

                    saved = self._save_match(
                        session, home_name, away_name, home_goals, away_goals,
                        match_date, league,
                    )
                    if saved:
                        count += 1

                except (ValueError, KeyError):
                    continue

        if count > 0:
            logger.info(f"Loaded {count} historical matches for {league} ({season})")
        return count

    def _parse_and_save_extra(self, csv_text: str, league: str) -> int:
        """Parse /new/ format CSV (all seasons in one file) and save recent matches."""
        db = get_db()
        reader = csv.DictReader(io.StringIO(csv_text))
        count = 0

        # Collect all unique seasons to determine the most recent ones
        rows = []
        seasons_seen = set()
        for row in reader:
            season_str = row.get("Season", "").strip()
            if season_str:
                seasons_seen.add(season_str)
                rows.append(row)

        # Sort seasons and keep only the most recent N
        sorted_seasons = sorted(seasons_seen, reverse=True)[:EXTRA_SEASONS_LIMIT]
        recent_seasons = set(sorted_seasons)

        with db.get_session() as session:
            for row in rows:
                try:
                    season_str = row.get("Season", "").strip()
                    if season_str not in recent_seasons:
                        continue

                    home_name = row.get("Home", "").strip()
                    away_name = row.get("Away", "").strip()
                    hg = row.get("HG", "")
                    ag = row.get("AG", "")
                    date_str = row.get("Date", "")

                    if not home_name or not away_name or not hg or not ag:
                        continue

                    home_goals = int(hg)
                    away_goals = int(ag)
                    match_date = self._parse_date(date_str)

                    if not match_date:
                        continue

                    saved = self._save_match(
                        session, home_name, away_name, home_goals, away_goals,
                        match_date, league,
                    )
                    if saved:
                        count += 1

                except (ValueError, KeyError):
                    continue

        if count > 0:
            logger.info(f"Loaded {count} historical matches for {league} (last {EXTRA_SEASONS_LIMIT} seasons)")
        return count

    def _save_match(self, session, home_name: str, away_name: str,
                    home_goals: int, away_goals: int,
                    match_date: datetime, league: str) -> bool:
        """Save a single match to database. Returns True if saved/updated."""
        from sqlalchemy import and_
        from datetime import timedelta

        # Find or create teams
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

        # Check for existing match (avoid duplicates)
        existing = session.query(Match).filter(
            and_(
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.match_date.between(
                    match_date - timedelta(hours=24),
                    match_date + timedelta(hours=24),
                ),
            )
        ).first()

        if existing:
            if existing.is_fixture:
                existing.home_goals = home_goals
                existing.away_goals = away_goals
                existing.is_fixture = False
                return True
            return False

        match = Match(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_date,
            league=league,
            home_goals=home_goals,
            away_goals=away_goals,
            is_fixture=False,
        )
        session.add(match)
        return True

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from CSV (handles multiple formats)."""
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        return None
