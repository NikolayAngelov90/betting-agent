"""Historical match data loader from football-data.co.uk CSV files.

Bootstraps the database with seasons of historical results so that
Poisson, Elo, and ML models have enough data for meaningful predictions.

Supports two CSV formats:
  - mmz4281: per-season files for main leagues (columns: HomeTeam, AwayTeam, FTHG, FTAG)
  - new: single all-seasons file for extra leagues (columns: Home, Away, HG, AG, Season)

Also extracts historical bookmaker odds from the CSVs (Bet365, Pinnacle, etc.)
and stores them as Odds records for real-data backtesting and model training.
"""

import csv
import io
import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Match, Team, Odds
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

# --- Main leagues: per-season CSVs from mmz4281/ directory ---
# URL: https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv
LEAGUE_CSV_MAP = {
    # Top 5 leagues + second divisions
    "england/premier-league": {"code": "E0", "country": "England"},
    "england/championship": {"code": "E1", "country": "England"},
    "england/league-one": {"code": "E2", "country": "England"},
    "england/league-two": {"code": "E3", "country": "England"},
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

# Cache file tracking when each league/season was last loaded
CACHE_FILE = Path("data/historical_load_cache.json")

# Past seasons (2324, 2223) are nearly immutable — refresh once every 60 days
PAST_SEASON_REFRESH_DAYS = 60
# Current season and extra leagues get new rows weekly — refresh every 7 days
CURRENT_SEASON_REFRESH_DAYS = 7

# Mapping CSV column names to (bookmaker, market_type, selection)
# These columns are available in football-data.co.uk mmz4281 CSVs
ODDS_COLUMN_MAP = {
    # 1X2 (match result) odds
    "B365H": ("Bet365", "1X2", "Home"),
    "B365D": ("Bet365", "1X2", "Draw"),
    "B365A": ("Bet365", "1X2", "Away"),
    "PSH":   ("Pinnacle", "1X2", "Home"),
    "PSD":   ("Pinnacle", "1X2", "Draw"),
    "PSA":   ("Pinnacle", "1X2", "Away"),
    "MaxH":  ("MarketMax", "1X2", "Home"),
    "MaxD":  ("MarketMax", "1X2", "Draw"),
    "MaxA":  ("MarketMax", "1X2", "Away"),
    "AvgH":  ("MarketAvg", "1X2", "Home"),
    "AvgD":  ("MarketAvg", "1X2", "Draw"),
    "AvgA":  ("MarketAvg", "1X2", "Away"),
    # Over/Under 2.5 goals
    "B365>2.5": ("Bet365", "over_under", "Over 2.5"),
    "B365<2.5": ("Bet365", "over_under", "Under 2.5"),
    "P>2.5":    ("Pinnacle", "over_under", "Over 2.5"),
    "P<2.5":    ("Pinnacle", "over_under", "Under 2.5"),
    "Max>2.5":  ("MarketMax", "over_under", "Over 2.5"),
    "Max<2.5":  ("MarketMax", "over_under", "Under 2.5"),
    "Avg>2.5":  ("MarketAvg", "over_under", "Over 2.5"),
    "Avg<2.5":  ("MarketAvg", "over_under", "Under 2.5"),
}

# Extra leagues (/new/ CSVs) use different column names for odds
EXTRA_ODDS_COLUMN_MAP = {
    "AvgH": ("MarketAvg", "1X2", "Home"),
    "AvgD": ("MarketAvg", "1X2", "Draw"),
    "AvgA": ("MarketAvg", "1X2", "Away"),
    "Avg>2.5": ("MarketAvg", "over_under", "Over 2.5"),
    "Avg<2.5": ("MarketAvg", "over_under", "Under 2.5"),
}


class HistoricalDataLoader(BaseScraper):
    """Loads historical match results and odds from football-data.co.uk CSVs."""

    def __init__(self, config=None):
        super().__init__(config)

    # ------------------------------------------------------------------
    # Load cache helpers
    # ------------------------------------------------------------------

    def _load_cache(self) -> dict:
        """Return {cache_key: iso_date_string} from disk, or {} on any error."""
        if CACHE_FILE.exists():
            try:
                return json.loads(CACHE_FILE.read_text())
            except Exception:
                pass
        return {}

    def _save_cache(self, cache: dict):
        """Persist the cache dict to disk."""
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(cache, indent=2))
        except Exception as e:
            logger.debug(f"Could not save historical load cache: {e}")

    def _is_fresh(self, cache: dict, key: str, days: int) -> bool:
        """Return True if *key* was loaded within *days* days (i.e. safe to skip)."""
        last_loaded = cache.get(key)
        if not last_loaded:
            return False
        try:
            last_dt = datetime.strptime(last_loaded, "%Y-%m-%d").date()
            return (date.today() - last_dt).days < days
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self):
        """Load historical data for all configured leagues."""
        await self.load_all_leagues()

    async def load_all_leagues(self, seasons: List[str] = None):
        """Load historical results for all mapped leagues.

        Uses a file-based cache (data/historical_load_cache.json) to skip
        CSVs that were already fetched recently:
          - Past seasons (2324, 2223): re-fetch at most every 60 days
          - Current season + extra leagues: re-fetch at most every 7 days

        When the DB already has substantial historical data (e.g. migrated from
        SQLite to Neon), past seasons are skipped entirely to avoid thousands of
        per-row existence checks over the network.
        """
        seasons = seasons or SEASONS
        current_season = seasons[0]  # Most recent season code (e.g. "2425")
        total_loaded = 0
        cache = self._load_cache()
        skipped = 0

        # Fast check: if DB already has plenty of matches, skip past seasons.
        # This prevents re-processing thousands of rows via slow network queries
        # when historical data was already migrated (e.g. SQLite → Neon).
        db = get_db()
        with db.get_session() as session:
            match_count = session.query(Match).filter(
                Match.home_goals.isnot(None)
            ).count()
        skip_past = match_count > 10000
        if skip_past:
            logger.info(
                f"Historical loader: {match_count:,} matches in DB — "
                f"skipping past seasons, only refreshing current ({current_season})"
            )

        # Main leagues (mmz4281 per-season format)
        for league, info in LEAGUE_CSV_MAP.items():
            for season in seasons:
                if skip_past and season != current_season:
                    skipped += 1
                    continue
                refresh_days = (
                    CURRENT_SEASON_REFRESH_DAYS
                    if season == current_season
                    else PAST_SEASON_REFRESH_DAYS
                )
                cache_key = f"{league}/{season}"
                if self._is_fresh(cache, cache_key, refresh_days):
                    skipped += 1
                    logger.debug(
                        f"Skipping {league} {season}: loaded within {refresh_days}d"
                    )
                    continue
                try:
                    count = await self.load_league_season(league, info["code"], season)
                    total_loaded += count
                    cache[cache_key] = date.today().isoformat()
                    self._save_cache(cache)
                except Exception as e:
                    logger.debug(f"No data for {league} season {season}: {e}")

        # Extra leagues (/new/ all-seasons format)
        for league, info in EXTRA_LEAGUE_CSV_MAP.items():
            cache_key = f"extra/{league}"
            if self._is_fresh(cache, cache_key, CURRENT_SEASON_REFRESH_DAYS):
                skipped += 1
                logger.debug(
                    f"Skipping extra league {league}: loaded within {CURRENT_SEASON_REFRESH_DAYS}d"
                )
                continue
            try:
                count = await self.load_extra_league(league, info["code"])
                total_loaded += count
                cache[cache_key] = date.today().isoformat()
                self._save_cache(cache)
            except Exception as e:
                logger.debug(f"No data for extra league {league}: {e}")

        if skipped:
            logger.info(f"Historical loader: skipped {skipped} already-fresh CSV(s)")
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
        """Parse mmz4281 format CSV and save matches + odds to database."""
        db = get_db()
        reader = csv.DictReader(io.StringIO(csv_text))
        count = 0
        odds_count = 0

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

                    match_id = self._save_match(
                        session, home_name, away_name, home_goals, away_goals,
                        match_date, league,
                    )
                    if match_id:
                        count += 1
                        odds_count += self._extract_odds_from_row(
                            session, row, match_id, ODDS_COLUMN_MAP,
                        )

                except (ValueError, KeyError):
                    continue

        if count > 0:
            logger.info(
                f"Loaded {count} historical matches + {odds_count} odds "
                f"for {league} ({season})"
            )
        return count

    def _parse_and_save_extra(self, csv_text: str, league: str) -> int:
        """Parse /new/ format CSV (all seasons in one file) and save recent matches."""
        db = get_db()
        reader = csv.DictReader(io.StringIO(csv_text))
        count = 0
        odds_count = 0

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

                    match_id = self._save_match(
                        session, home_name, away_name, home_goals, away_goals,
                        match_date, league,
                    )
                    if match_id:
                        count += 1
                        odds_count += self._extract_odds_from_row(
                            session, row, match_id, EXTRA_ODDS_COLUMN_MAP,
                        )

                except (ValueError, KeyError):
                    continue

        if count > 0:
            logger.info(
                f"Loaded {count} historical matches + {odds_count} odds "
                f"for {league} (last {EXTRA_SEASONS_LIMIT} seasons)"
            )
        return count

    def _save_match(self, session, home_name: str, away_name: str,
                    home_goals: int, away_goals: int,
                    match_date: datetime, league: str) -> Optional[int]:
        """Save a single match to database. Returns match ID if saved/updated, None otherwise."""
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
                return existing.id
            # Match already exists — still return ID so odds can be added
            return existing.id

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
        session.flush()
        return match.id

    def _extract_odds_from_row(self, session, row: dict, match_id: int,
                                column_map: dict) -> int:
        """Extract bookmaker odds from a CSV row and create Odds records.

        Returns count of odds records created.
        """
        count = 0

        for csv_col, (bookmaker, market_type, selection) in column_map.items():
            raw_val = row.get(csv_col, "").strip()
            if not raw_val:
                continue

            try:
                odds_value = float(raw_val)
            except (ValueError, TypeError):
                continue

            if odds_value <= 1.0:
                continue  # Invalid odds

            # Check for existing odds (avoid duplicates on re-runs)
            existing = session.query(Odds).filter_by(
                match_id=match_id,
                bookmaker=bookmaker,
                market_type=market_type,
                selection=selection,
            ).first()

            if existing:
                continue

            odds = Odds(
                match_id=match_id,
                bookmaker=bookmaker,
                market_type=market_type,
                selection=selection,
                odds_value=odds_value,
            )
            session.add(odds)
            count += 1

        return count

    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from CSV (handles multiple formats)."""
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except (ValueError, AttributeError):
                continue
        return None
