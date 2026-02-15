"""Historical match data loader from football-data.co.uk CSV files.

Bootstraps the database with seasons of historical results so that
Poisson, Elo, and ML models have enough data for meaningful predictions.
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

# football-data.co.uk CSV URLs for current and previous seasons
# Format: https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv
LEAGUE_CSV_MAP = {
    "england/premier-league": {"code": "E0", "country": "England"},
    "spain/laliga": {"code": "SP1", "country": "Spain"},
    "germany/bundesliga": {"code": "D1", "country": "Germany"},
    "italy/serie-a": {"code": "I1", "country": "Italy"},
    "france/ligue-1": {"code": "F1", "country": "France"},
    "netherlands/eredivisie": {"code": "N1", "country": "Netherlands"},
    "portugal/primeira-liga": {"code": "P1", "country": "Portugal"},
    "belgium/jupiler-pro-league": {"code": "B1", "country": "Belgium"},
    "turkey/super-lig": {"code": "T1", "country": "Turkey"},
    "scotland/premiership": {"code": "SC0", "country": "Scotland"},
    "austria/bundesliga": {"code": "AUT", "country": "Austria"},
    "switzerland/super-league": {"code": "SWZ", "country": "Switzerland"},
    "greece/super-league": {"code": "G1", "country": "Greece"},
    "denmark/superliga": {"code": "DNK", "country": "Denmark"},
}

# Season codes for football-data.co.uk (yy/yy format)
SEASONS = ["2425", "2324", "2223"]


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

        for league, info in LEAGUE_CSV_MAP.items():
            for season in seasons:
                try:
                    count = await self.load_league_season(league, info["code"], season)
                    total_loaded += count
                except Exception as e:
                    logger.debug(f"No data for {league} season {season}: {e}")

        logger.info(f"Historical data loading complete: {total_loaded} matches loaded")
        return total_loaded

    async def load_league_season(self, league: str, league_code: str,
                                  season: str) -> int:
        """Load one league-season CSV into the database.

        Args:
            league: Internal league identifier
            league_code: football-data.co.uk league code
            season: Season code (e.g., '2425')

        Returns:
            Number of matches loaded
        """
        url = f"https://www.football-data.co.uk/mmz4281/{season}/{league_code}.csv"

        try:
            csv_text = await self.fetch(url)
        except Exception as e:
            logger.debug(f"Could not fetch {url}: {e}")
            return 0

        if not csv_text or len(csv_text) < 50:
            return 0

        return self._parse_and_save(csv_text, league, season)

    def _parse_and_save(self, csv_text: str, league: str, season: str) -> int:
        """Parse CSV text and save matches to database."""
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
                    from sqlalchemy import and_
                    from datetime import timedelta
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
                        # Update if it was a fixture stub
                        if existing.is_fixture:
                            existing.home_goals = home_goals
                            existing.away_goals = away_goals
                            existing.is_fixture = False
                            count += 1
                        continue

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
                    count += 1

                except (ValueError, KeyError) as e:
                    continue

        if count > 0:
            logger.info(f"Loaded {count} historical matches for {league} ({season})")
        return count

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from CSV (handles multiple formats)."""
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        return None
