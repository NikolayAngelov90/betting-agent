"""football-data.org API client — free fixtures, results, and standings.

Free tier: 10 calls/minute, **no daily limit**, 9 relevant competitions:
  PL  - Premier League       CL  - Champions League
  BL1 - Bundesliga           DED - Eredivisie
  PD  - La Liga              PPL - Primeira Liga
  SA  - Serie A              ELC - Championship
  FL1 - Ligue 1

Register for a free API key at https://www.football-data.org/

Used as a reliable, quota-free supplement for top-league fixture/result data.
Supplements Flashscore (Selenium-based) which can fail on Chrome crashes.
API-Football quota is freed for rich odds markets (Over/Under 1.5, Team Goals).
"""

import asyncio
import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import httpx

from src.data.database import get_db
from src.data.models import Match, Team
from src.utils.logger import get_logger

logger = get_logger()

BASE_URL = "https://api.football-data.org/v4"

# football-data.org competition code → our internal league key
COMPETITION_MAP: Dict[str, str] = {
    "PL":  "england/premier-league",
    "ELC": "england/championship",
    "PD":  "spain/laliga",
    "BL1": "germany/bundesliga",
    "SA":  "italy/serie-a",
    "FL1": "france/ligue-1",
    "CL":  "europe/champions-league",
    "DED": "netherlands/eredivisie",
    "PPL": "portugal/primeira-liga",
}

# football-data.org full team name → name used in our DB (from football-data.co.uk CSVs)
TEAM_NAME_MAP: Dict[str, str] = {
    # England — Premier League
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa",
    "West Ham United FC": "West Ham",
    "Brighton & Hove Albion FC": "Brighton",
    "Wolverhampton Wanderers FC": "Wolves",
    "Fulham FC": "Fulham",
    "Brentford FC": "Brentford",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Nottingham Forest FC": "Nott'm Forest",
    "Leicester City FC": "Leicester",
    "Burnley FC": "Burnley",
    "Sheffield United FC": "Sheffield United",
    "Luton Town FC": "Luton",
    "Ipswich Town FC": "Ipswich",
    "AFC Bournemouth": "Bournemouth",
    "Southampton FC": "Southampton",
    # England — Championship
    "Leeds United FC": "Leeds",
    "Sunderland AFC": "Sunderland",
    "Middlesbrough FC": "Middlesbrough",
    "Norwich City FC": "Norwich",
    "Coventry City FC": "Coventry",
    "West Bromwich Albion FC": "West Brom",
    "Preston North End FC": "Preston",
    "Swansea City AFC": "Swansea",
    "Hull City AFC": "Hull",
    "Blackburn Rovers FC": "Blackburn",
    "Bristol City FC": "Bristol City",
    "Stoke City FC": "Stoke",
    "Queens Park Rangers FC": "QPR",
    "Millwall FC": "Millwall",
    "Cardiff City FC": "Cardiff",
    "Birmingham City FC": "Birmingham",
    "Derby County FC": "Derby",
    "Watford FC": "Watford",
    "Plymouth Argyle FC": "Plymouth",
    "Sheffield Wednesday FC": "Sheff Wed",
    "Peterborough United FC": "Peterborough",
    "Portsmouth FC": "Portsmouth",
    "Oxford United FC": "Oxford",
    "Hannover 96": "Hannover",
    "SG Dynamo Dresden": "Dynamo Dresden",
    # Germany — Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "RB Leipzig": "RB Leipzig",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "VfB Stuttgart": "Stuttgart",
    "SC Freiburg": "Freiburg",
    "Borussia Mönchengladbach": "M'gladbach",
    "1899 Hoffenheim": "Hoffenheim",
    "FC Augsburg": "Augsburg",
    "SV Werder Bremen": "Werder Bremen",
    "1. FC Union Berlin": "Union Berlin",
    "1. FSV Mainz 05": "Mainz",
    "VfL Wolfsburg": "Wolfsburg",
    "FC St. Pauli 1910": "St. Pauli",
    "Holstein Kiel": "Holstein Kiel",
    "1. FC Heidenheim 1846": "Heidenheim",
    "VfL Bochum 1848": "Bochum",
    # Spain — La Liga
    "Real Madrid CF": "Real Madrid",
    "FC Barcelona": "Barcelona",
    "Club Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Real Sociedad de Fútbol": "Sociedad",
    "Villarreal CF": "Villarreal",
    "Real Betis Balompié": "Betis",
    "RC Celta de Vigo": "Celta",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Getafe CF": "Getafe",
    "UD Las Palmas": "Las Palmas",
    "Rayo Vallecano de Madrid": "Vallecano",
    "CA Osasuna": "Osasuna",
    "Deportivo Alavés": "Alaves",
    "Girona FC": "Girona",
    "RCD Mallorca": "Mallorca",
    "RCD Espanyol de Barcelona": "Espanyol",
    "CD Leganés": "Leganes",
    # Italy — Serie A
    "FC Internazionale Milano": "Inter",
    "AC Milan": "Milan",
    "Juventus FC": "Juventus",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "ACF Fiorentina": "Fiorentina",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "Torino FC": "Torino",
    "US Sassuolo Calcio": "Sassuolo",
    "Empoli FC": "Empoli",
    "Hellas Verona FC": "Verona",
    "Udinese Calcio": "Udinese",
    "Genoa CFC": "Genoa",
    "Frosinone Calcio": "Frosinone",
    "Cagliari Calcio": "Cagliari",
    "AC Monza": "Monza",
    "Venezia FC": "Venezia",
    "Como 1907": "Como",
    # France — Ligue 1
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "Stade Rennais FC 1901": "Rennes",
    "RC Strasbourg Alsace": "Strasbourg",
    "Stade Brestois 29": "Brest",
    "Toulouse FC": "Toulouse",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "Angers SCO": "Angers",
    "Le Havre AC": "Le Havre",
    "Stade de Reims": "Reims",
    "FC Lorient": "Lorient",
    "AJ Auxerre": "Auxerre",
    "Saint-Étienne": "St Etienne",
    # Netherlands — Eredivisie
    "Ajax Amsterdam": "Ajax",
    "PSV": "PSV",
    "Feyenoord Rotterdam": "Feyenoord",
    "AZ Alkmaar": "AZ Alkmaar",
    "FC Utrecht": "Utrecht",
    "FC Twente": "Twente",
    "Go Ahead Eagles": "Go Ahead Eag",
    "NEC Nijmegen": "Nijmegen",
    "Sparta Rotterdam": "Sp Rotterdam",
    "SC Heerenveen": "Heerenveen",
    "Willem II": "Willem II",
    "FC Groningen": "Groningen",
    "NAC Breda": "NAC Breda",
    "RKC Waalwijk": "Waalwijk",
    "PEC Zwolle": "PEC Zwolle",
    "Almere City FC": "Almere City",
    # Portugal — Primeira Liga
    "SL Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting Lisbon",
    "SC Braga": "Sp Braga",
    "Estoril Praia": "Estoril",
    "Gil Vicente FC": "Gil Vicente",
    "Moreirense FC": "Moreirense",
    "Vitória SC": "Vitoria SC",
    "Rio Ave FC": "Rio Ave",
    "CD Nacional": "Nacional",
    "SC Farense": "Farense",
    "GD Chaves": "Chaves",
    "Casa Pia AC": "Casa Pia",
    "AVS Futebol SAD": "AVS",
}

# Suffixes to strip when doing fuzzy name matching
_STRIP_SUFFIXES = (
    " fc", " afc", " sc", " cf", " ac", " bc", " bsc", " kv",
    " 1846", " 1909", " 1901", " 1848", " 05", " 04",
)


def _normalize(name: str) -> str:
    """Lower-case, strip common suffixes for fuzzy comparison."""
    n = name.lower().strip()
    for s in _STRIP_SUFFIXES:
        if n.endswith(s):
            n = n[: -len(s)].strip()
    return n


def _names_match(a: str, b: str) -> bool:
    """Return True if two team names refer to the same team."""
    if not a or not b:
        return False
    na, nb = _normalize(a), _normalize(b)
    return na == nb or na in nb or nb in na


class FootballDataOrgScraper:
    """Fetches fixtures and results from football-data.org (free, no daily limit).

    Rate limit: 10 calls/minute (enforced via _min_interval sleep).
    Covers 9 top European competitions on the free tier.

    Primary use-cases:
    1. update_results()  — update scores for recently finished matches (settlement)
    2. sync_fixtures()   — ensure today's/tomorrow's fixtures are in the DB
    """

    _min_interval = 6.1  # 10 calls/min

    def __init__(self, config=None):
        self.api_key = (
            os.environ.get("FOOTBALL_DATA_ORG_KEY", "").strip()
            or (config or {}).get("data_sources", {}).get("footballdataorg_key", "")
        )
        self.enabled = bool(self.api_key)
        self.db = get_db()
        self._last_call_at = 0.0

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    async def _get(self, path: str, params: dict = None) -> Optional[dict]:
        if not self.enabled:
            return None
        now = asyncio.get_event_loop().time()
        wait = self._min_interval - (now - self._last_call_at)
        if wait > 0:
            await asyncio.sleep(wait)
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{BASE_URL}{path}",
                    params=params or {},
                    headers={"X-Auth-Token": self.api_key},
                )
            self._last_call_at = asyncio.get_event_loop().time()
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                logger.warning("football-data.org rate limit hit — sleeping 60s")
                await asyncio.sleep(60)
            else:
                logger.debug(f"football-data.org {path}: HTTP {resp.status_code}")
        except Exception as e:
            logger.debug(f"football-data.org request error: {type(e).__name__}: {e}")
        return None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    async def fetch_matches(self, target_date: date) -> List[dict]:
        """Fetch all free-tier matches on *target_date* (1 API call)."""
        data = await self._get("/matches", {"date": target_date.isoformat()})
        if not data:
            return []
        return data.get("matches", [])

    # ------------------------------------------------------------------
    # Settlement enrichment
    # ------------------------------------------------------------------

    async def update_results(self, days_back: int = 2) -> int:
        """Fetch finished match scores and update any unscored DB matches.

        Calls the API once per day (days_back+1 total calls).
        Returns total number of match records updated.
        """
        if not self.enabled:
            return 0

        total = 0
        for i in range(days_back + 1):
            target = date.today() - timedelta(days=i)
            matches = await self.fetch_matches(target)
            for m in matches:
                if m.get("status") != "FINISHED":
                    continue
                ft = m.get("score", {}).get("fullTime", {})
                home_goals = ft.get("home")
                away_goals = ft.get("away")
                if home_goals is None or away_goals is None:
                    continue
                comp_code = m.get("competition", {}).get("code", "")
                league = COMPETITION_MAP.get(comp_code)
                if not league:
                    continue
                home_name = TEAM_NAME_MAP.get(
                    m.get("homeTeam", {}).get("name", ""),
                    m.get("homeTeam", {}).get("name", ""),
                )
                away_name = TEAM_NAME_MAP.get(
                    m.get("awayTeam", {}).get("name", ""),
                    m.get("awayTeam", {}).get("name", ""),
                )
                updated = self._apply_score(
                    league, home_name, away_name, target, home_goals, away_goals
                )
                total += updated

        if total:
            logger.info(f"football-data.org: updated {total} match scores")
        return total

    def _apply_score(
        self,
        league: str,
        home_name: str,
        away_name: str,
        match_date: date,
        home_goals: int,
        away_goals: int,
    ) -> int:
        """Find an unscored DB match and write the result. Returns 1 if updated."""
        date_start = datetime.combine(match_date, datetime.min.time())
        date_end = date_start + timedelta(days=1)

        with self.db.get_session() as session:
            candidates = (
                session.query(Match)
                .filter(
                    Match.league == league,
                    Match.match_date >= date_start,
                    Match.match_date < date_end,
                    Match.home_goals.is_(None),
                )
                .all()
            )
            for match in candidates:
                ht = session.get(Team, match.home_team_id)
                at = session.get(Team, match.away_team_id)
                if not ht or not at:
                    continue
                if _names_match(home_name, ht.name) and _names_match(away_name, at.name):
                    match.home_goals = home_goals
                    match.away_goals = away_goals
                    match.is_fixture = False
                    session.commit()
                    logger.debug(
                        f"FDO score: {ht.name} {home_goals}-{away_goals} {at.name} "
                        f"({league})"
                    )
                    return 1
        return 0

    # ------------------------------------------------------------------
    # Fixture sync
    # ------------------------------------------------------------------

    async def sync_fixtures(self, days_ahead: int = 1) -> int:
        """Ensure today's (and tomorrow's) fixtures for top leagues are in the DB.

        Creates Team and Match records if they don't already exist.
        Returns count of new fixtures added.
        """
        if not self.enabled:
            return 0

        added = 0
        for i in range(days_ahead + 1):
            target = date.today() + timedelta(days=i)
            matches = await self.fetch_matches(target)
            for m in matches:
                if m.get("status") not in ("SCHEDULED", "TIMED"):
                    continue
                comp_code = m.get("competition", {}).get("code", "")
                league = COMPETITION_MAP.get(comp_code)
                if not league:
                    continue
                home_raw = m.get("homeTeam", {}).get("name", "")
                away_raw = m.get("awayTeam", {}).get("name", "")
                home_name = TEAM_NAME_MAP.get(home_raw, home_raw)
                away_name = TEAM_NAME_MAP.get(away_raw, away_raw)
                utc_str = m.get("utcDate", "")
                try:
                    match_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                    match_dt = match_dt.replace(tzinfo=None)  # store as naive UTC
                except Exception:
                    continue

                if self._ensure_fixture(league, home_name, away_name, match_dt):
                    added += 1

        if added:
            logger.info(f"football-data.org: added {added} new fixtures to DB")
        return added

    @staticmethod
    def _find_team_by_prefix(session, name: str):
        """Return a Team whose stored name is a prefix of *name* (or vice-versa).

        Flashscore sometimes stores truncated team names (e.g. "Go Ahead Eag"
        instead of "Go Ahead Eagles").  When FDO provides the full name we try a
        prefix match so we don't create a duplicate Team record.

        Only applied when the name is longer than 8 characters to avoid false
        positives on short names.
        """
        if len(name) <= 8:
            return None
        # FDO name starts with DB name (DB name is a truncated prefix)
        candidate = session.query(Team).filter(
            Team.name.ilike(f"{name[:12]}%")
        ).first()
        if candidate:
            return candidate
        # DB name starts with FDO name (FDO name is shorter)
        candidate = session.query(Team).filter(
            Team.name.ilike(f"{name}%")
        ).first()
        return candidate

    def _ensure_fixture(
        self,
        league: str,
        home_name: str,
        away_name: str,
        match_dt: datetime,
    ) -> bool:
        """Create Match + Team records if this fixture isn't in the DB yet.

        Returns True if a new record was created.
        """
        date_start = datetime.combine(match_dt.date(), datetime.min.time())
        date_end = date_start + timedelta(days=1)

        with self.db.get_session() as session:
            # Check if already exists — use fuzzy matching to avoid duplicates
            # when Flashscore and FDO use different name variants for the same team
            candidates = (
                session.query(Match)
                .filter(
                    Match.league == league,
                    Match.match_date >= date_start,
                    Match.match_date < date_end,
                )
                .all()
            )
            for cand in candidates:
                ht = session.get(Team, cand.home_team_id)
                if ht and _names_match(home_name, ht.name):
                    return False

            # Get or create home team
            home_team = session.query(Team).filter_by(name=home_name).first()
            if not home_team:
                # Prefix fallback — Flashscore may store truncated names
                # e.g. "Go Ahead Eag" in DB vs "Go Ahead Eagles" from FDO
                home_team = self._find_team_by_prefix(session, home_name)
            if not home_team:
                home_team = Team(name=home_name)
                session.add(home_team)
                session.flush()

            # Get or create away team
            away_team = session.query(Team).filter_by(name=away_name).first()
            if not away_team:
                away_team = self._find_team_by_prefix(session, away_name)
            if not away_team:
                away_team = Team(name=away_name)
                session.add(away_team)
                session.flush()

            match = Match(
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                match_date=match_dt,
                league=league,
                is_fixture=True,
            )
            session.add(match)
            session.commit()
            logger.debug(
                f"FDO fixture: {home_name} vs {away_name} "
                f"{match_dt.strftime('%Y-%m-%d %H:%M')} ({league})"
            )
            return True

    # ------------------------------------------------------------------
    # Historical season backfill
    # ------------------------------------------------------------------

    async def backfill_historical_seasons(self, seasons: List[int] = None) -> int:
        """Bulk-fetch all finished matches for all 9 covered competitions across seasons.

        football-data.org has no daily limit so this runs without touching the
        API-Football quota. Each competition × season = 1 API call (6s rate limit).
        9 competitions × 3 seasons = 27 calls ≈ 3 minutes total.

        Saves any match not already in the DB, creating Team records as needed.
        Returns total number of new match records created.
        """
        if not self.enabled:
            return 0
        if seasons is None:
            seasons = [2023, 2024, 2025]

        total_saved = 0
        for code, league in COMPETITION_MAP.items():
            for season in seasons:
                data = await self._get(
                    f"/competitions/{code}/matches",
                    {"season": season, "status": "FINISHED"},
                )
                if not data:
                    continue
                matches = data.get("matches", [])
                saved = 0
                for m in matches:
                    ft = m.get("score", {}).get("fullTime", {})
                    home_goals = ft.get("home")
                    away_goals = ft.get("away")
                    if home_goals is None or away_goals is None:
                        continue
                    home_raw = m.get("homeTeam", {}).get("name", "")
                    away_raw = m.get("awayTeam", {}).get("name", "")
                    home_name = TEAM_NAME_MAP.get(home_raw, home_raw)
                    away_name = TEAM_NAME_MAP.get(away_raw, away_raw)
                    utc_str = m.get("utcDate", "")
                    try:
                        match_dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
                        match_dt = match_dt.replace(tzinfo=None)
                    except Exception:
                        continue
                    if self._ensure_historical_match(
                        league, home_name, away_name, match_dt, home_goals, away_goals, str(season)
                    ):
                        saved += 1

                if saved:
                    logger.info(f"FDO backfill {code} {season}: {saved} new matches saved")
                else:
                    logger.debug(f"FDO backfill {code} {season}: {len(matches)} matches already in DB")
                total_saved += saved

        logger.info(f"FDO historical backfill complete: {total_saved} new matches saved")
        return total_saved

    def _ensure_historical_match(
        self,
        league: str,
        home_name: str,
        away_name: str,
        match_dt: datetime,
        home_goals: int,
        away_goals: int,
        season: str,
    ) -> bool:
        """Create a completed match record if it doesn't already exist.

        Also creates Team records for any team not yet in the DB.
        Returns True if a new record was created.
        """
        date_start = datetime.combine(match_dt.date(), datetime.min.time())
        date_end = date_start + timedelta(days=1)

        with self.db.get_session() as session:
            # Check if already exists by home team name + date
            existing = (
                session.query(Match)
                .join(Team, Match.home_team_id == Team.id)
                .filter(
                    Match.league == league,
                    Match.match_date >= date_start,
                    Match.match_date < date_end,
                    Team.name == home_name,
                )
                .first()
            )
            if existing:
                # Update score if this was a fixture stub without goals
                if existing.home_goals is None:
                    existing.home_goals = home_goals
                    existing.away_goals = away_goals
                    existing.is_fixture = False
                    session.commit()
                return False

            # Get or create home team
            home_team = session.query(Team).filter_by(name=home_name).first()
            if not home_team:
                home_team = self._find_team_by_prefix(session, home_name)
            if not home_team:
                home_team = Team(name=home_name)
                session.add(home_team)
                session.flush()

            # Get or create away team
            away_team = session.query(Team).filter_by(name=away_name).first()
            if not away_team:
                away_team = self._find_team_by_prefix(session, away_name)
            if not away_team:
                away_team = Team(name=away_name)
                session.add(away_team)
                session.flush()

            match = Match(
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                match_date=match_dt,
                league=league,
                season=season,
                is_fixture=False,
                home_goals=home_goals,
                away_goals=away_goals,
            )
            session.add(match)
            session.commit()
            return True
