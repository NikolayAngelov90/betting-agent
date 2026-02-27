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

# Map API-Football team names to our historical database names.
# API-Football often uses full official names while football-data.co.uk uses short names.
TEAM_NAME_ALIASES: Dict[str, str] = {
    # England
    "Bayer Leverkusen": "Leverkusen",
    "Borussia Dortmund": "Dortmund",
    "Borussia Monchengladbach": "M'gladbach",
    "VfB Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "FC Augsburg": "Augsburg",
    "1899 Hoffenheim": "Hoffenheim",
    "SC Freiburg": "Freiburg",
    "1. FC Heidenheim 1846": "Heidenheim",
    "1. FC Union Berlin": "Union Berlin",
    # Spain
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Deportivo Alaves": "Alaves",
    "Celta Vigo": "Celta",
    "Celta de Vigo": "Celta",
    "Rayo Vallecano": "Vallecano",
    # Italy
    "AC Milan": "Milan",
    "Inter": "Inter",
    "Hellas Verona": "Verona",
    # France
    "Paris Saint Germain": "Paris SG",
    "Olympique Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "RC Lens": "Lens",
    "Stade Rennais": "Rennes",
    "RC Strasbourg Alsace": "Strasbourg",
    # Belgium
    "Club Brugge KV": "Club Brugge",
    "RSC Anderlecht": "Anderlecht",
    "Standard Liege": "Standard",
    "KAA Gent": "Gent",
    # Netherlands
    "Feyenoord": "Feyenoord",
    "PSV Eindhoven": "PSV",
    # Portugal
    "Sporting CP": "Sporting Lisbon",
    "FC Porto": "Porto",
    "SL Benfica": "Benfica",
    "SC Braga": "Sp Braga",
    # Greece
    "Olympiakos Piraeus": "Olympiakos",
    "Panathinaikos": "Panathinaikos",
    # Turkey
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Besiktas": "Besiktas",
    # England (short names in football-data.co.uk)
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Wolverhampton Wanderers": "Wolves",
    "West Ham United": "West Ham",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Tottenham Hotspur": "Tottenham",
    "Leicester City": "Leicester",
    "Brighton & Hove Albion": "Brighton",
    # Scotland
    "Glasgow Rangers": "Rangers",
    "Celtic": "Celtic",
    # England — Championship abbreviations
    "Sheffield Wednesday": "Sheff Wed",
    "Sheffield United": "Sheffield Utd",
    "Queens Park Rangers": "QPR",
    "West Bromwich Albion": "West Brom",
    "Middlesbrough": "Middlesbrough",
    "Swansea City": "Swansea",
    "Coventry City": "Coventry",
    "Norwich City": "Norwich",
    # Eastern Europe
    "FK Crvena Zvezda": "Crvena Zvezda",
    "Dinamo Zagreb": "Dinamo Zagreb",
    "Ferencvarosi TC": "Ferencvaros",
    # Switzerland
    "FC Winterthur": "Winterthur",
    "FC ST. Gallen": "St. Gallen",
    "FC Luzern": "Luzern",
    "FC Basel 1893": "Basel",
    "FC Zurich": "Zurich",
    "BSC Young Boys": "Young Boys",
    "Servette FC": "Servette",
    "FC Lugano": "Lugano",
    # Other
    "FC Salzburg": "Salzburg",
    "Sturm Graz": "Sturm Graz",
    "Stade Brestois 29": "Brest",
    # England — League One (football-data.co.uk uses bare city/nickname)
    "Mansfield Town": "Mansfield",
    "Burton Albion": "Burton",
    "Exeter City": "Exeter",
    "Stockport County": "Stockport",
    "Wycombe Wanderers": "Wycombe",
    "Shrewsbury Town": "Shrewsbury",
    "Peterborough United": "Peterborough",
    "Lincoln City": "Lincoln",
    "Cambridge United": "Cambridge",
    "Charlton Athletic": "Charlton",
    "Wigan Athletic": "Wigan",
    "Blackpool": "Blackpool",
    "Bolton Wanderers": "Bolton",
    "Bristol Rovers": "Bristol Rov",
    "Rotherham United": "Rotherham",
    "Huddersfield Town": "Huddersfield",
    "Port Vale": "Port Vale",
    "Stevenage": "Stevenage",
    "Reading": "Reading",
    "Birmingham City": "Birmingham",
    "Barnsley": "Barnsley",
    "Wrexham": "Wrexham",
    # England — League Two (football-data.co.uk short names)
    "Swindon Town": "Swindon",
    "Accrington Stanley": "Accrington",
    "Harrogate Town": "Harrogate",
    "Salford City": "Salford",
    "Newport County": "Newport",
    "Tranmere Rovers": "Tranmere",
    "Grimsby Town": "Grimsby",
    "Colchester United": "Colchester",
    "Notts County": "Notts Co",
    "Bradford City": "Bradford",
    "Barrow": "Barrow",
    "Gillingham": "Gillingham",
    "MK Dons": "MK Dons",
    "Crawley Town": "Crawley",
    "AFC Wimbledon": "Wimbledon",
    "Fleetwood Town": "Fleetwood",
    "Crewe Alexandra": "Crewe",
    "Doncaster Rovers": "Doncaster",
    "Forest Green Rovers": "Forest Green",
    "Sutton United": "Sutton",
    # Netherlands
    "NEC Nijmegen": "Nijmegen",
    # Romania
    "CFR 1907 Cluj": "CFR Cluj",
    # Norway
    "HamKam": "Ham-Kam",
    # Turkey (with special characters)
    "Fenerbahçe": "Fenerbahce",
    "Galatasaray SK": "Galatasaray",
    "Beşiktaş JK": "Besiktas",
    "Trabzonspor": "Trabzon",
    "Alanyaspor": "Alanyaspor",
}

# Map our internal league keys to API-Football league IDs
LEAGUE_ID_MAP = {
    "england/premier-league": 39,
    "england/championship": 40,
    "england/league-one": 41,
    "england/league-two": 42,
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
    "europe/champions-league": 2,
    "europe/europa-league": 3,
    "europe/europa-conference-league": 848,
}

# Reverse map: API-Football league ID -> our internal league key
ID_TO_LEAGUE = {v: k for k, v in LEAGUE_ID_MAP.items()}

# Leagues to fetch odds for first (highest priority)
PRIORITY_LEAGUES = [
    "england/premier-league", "spain/laliga", "germany/bundesliga",
    "italy/serie-a", "france/ligue-1", "europe/champions-league",
    "europe/europa-league", "europe/europa-conference-league",
    "netherlands/eredivisie", "portugal/primeira-liga",
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
    # Team goal line markets — confirmed API-Football bet names (verified from live API)
    "Total - Home": {
        "market_type": "team_goals",
        "selections": {
            "Over 0.5": "Home Over 0.5", "Under 0.5": "Home Under 0.5",
            "Over 1.5": "Home Over 1.5", "Under 1.5": "Home Under 1.5",
            "Over 2.5": "Home Over 2.5", "Under 2.5": "Home Under 2.5",
        },
    },
    "Total - Away": {
        "market_type": "team_goals",
        "selections": {
            "Over 0.5": "Away Over 0.5", "Under 0.5": "Away Under 0.5",
            "Over 1.5": "Away Over 1.5", "Under 1.5": "Away Under 1.5",
            "Over 2.5": "Away Over 2.5", "Under 2.5": "Away Under 2.5",
            "Over 3.5": "Away Over 3.5", "Under 3.5": "Away Under 3.5",
        },
    },
}


class APIFootballScraper(BaseScraper):
    """Fetches xG, match statistics, fixtures, and odds from API-Football."""

    # Request budget allocation (out of 100/day free tier).
    #
    # Strategy: redirect the bulk of the budget from xG backfill (which the
    # Poisson model ignores entirely — it computes its own expected goals from
    # attack/defense ratings) to rich odds markets that the value calculator
    # can directly use (Over/Under 1.5, Team Goals Home/Away, etc.).
    #
    # football-data.org (free, no daily limit) now covers fixtures + results
    # for 9 top leagues, so API-Football results/fixture calls are secondary.
    BUDGET_RESULTS = 4    # keep: settlement for leagues not in football-data.org
    BUDGET_FIXTURES = 2   # keep: fixture ids needed for odds lookup
    BUDGET_XG = 1         # was 5 — Poisson ignores DB xG; 1 request keeps backfill alive
    BUDGET_ODDS = 80      # was 0  — re-enabled: Over/Under 1.5, Team Goals, BTTS
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
        self._quota_exhausted = False  # Set True on first quota error; skips all further calls
        self._logged_unknown_bets: set = set()  # Suppress repeated unknown bet type logs

    async def _api_get(self, endpoint: str, params: dict = None) -> Optional[dict]:
        """Make an authenticated GET request to API-Football.

        Uses BaseScraper.fetch_json for retry + circuit breaker support.
        """
        if self._quota_exhausted:
            return None
        if self._requests_today >= self._daily_limit:
            logger.warning("API-Football daily request limit reached, skipping")
            self._quota_exhausted = True
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
                elif "requests" in errors:
                    logger.warning("API-Football daily quota exhausted — skipping all further API calls")
                    self._quota_exhausted = True
                elif "plan" in str(errors).lower():
                    # Free-tier date restriction — expected, not an error
                    logger.debug(f"API-Football plan restriction: {errors}")
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
        """Run the full API-Football update: yesterday results + today fixtures + xG + odds.

        Only fetches today's fixtures and odds (the CI pipeline runs daily, so
        tomorrow's matches will be fetched when tomorrow's run executes).
        """
        if not self.enabled:
            logger.warning("API-Football key not set. Skipping.")
            return

        logger.info("Starting API-Football update")

        # 1. Fetch yesterday's results (free plan only allows yesterday+today+tomorrow)
        await self.fetch_recent_results(days_back=1)

        # 2. Fetch today's fixtures only (CI runs daily — tomorrow handled next run)
        today = date.today()
        await self._fetch_fixtures_by_date(today)

        # 3. Backfill xG for recent completed matches that don't have it (7 days back)
        await self._backfill_xg()

        # 4. Fetch real bookmaker odds for today's fixtures
        await self.fetch_upcoming_odds()

        # 5. Backfill historical match data for low-coverage pick teams (uses leftover budget)
        remaining = max(0, self._daily_limit - self._requests_today - self.BUDGET_RESERVE)
        backfill_budget = min(remaining, 50)
        if backfill_budget > 2:
            await self.backfill_team_history(max_budget=backfill_budget)

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
            home_api_id = fix.get("teams", {}).get("home", {}).get("id")
            away_api_id = fix.get("teams", {}).get("away", {}).get("id")
            match_ts = fix.get("fixture", {}).get("timestamp")
            status_short = fix.get("fixture", {}).get("status", {}).get("short", "")
            referee = fix.get("fixture", {}).get("referee") or ""

            if not home_name or not away_name:
                continue

            match_dt = datetime.utcfromtimestamp(match_ts) if match_ts else None
            if not match_dt:
                continue

            # Get or create teams — also saves API-Football team IDs for future backfill
            home_team_id = self._get_or_create_team_id(
                home_name, league_key, apifootball_team_id=home_api_id
            )
            away_team_id = self._get_or_create_team_id(
                away_name, league_key, apifootball_team_id=away_api_id
            )

            # Check for existing match — first by apifootball_id (most reliable),
            # then by team DB IDs, then fuzzy name search.
            match_id = None
            with self.db.get_session() as session:
                existing = session.query(Match).filter(
                    Match.apifootball_id == fixture_id
                ).first()
                if existing:
                    match_id = existing.id

            if match_id is None:
                match_id = self._find_match_id(home_team_id, away_team_id, match_dt)

            # Fallback: team-ID match failed (name mismatch created a different team
            # record, or fixture came from Flashscore/FDO under a slightly different name).
            # Search by league + date + fuzzy team name to avoid creating a duplicate.
            if match_id is None:
                match_id = self._find_match_by_date_league(
                    league_key, match_dt, home_name, away_name
                )
                if match_id:
                    logger.debug(
                        f"API-Football fuzzy-linked {home_name} vs {away_name} "
                        f"({league_key}) to existing fixture {match_id}"
                    )

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
                        if referee and not match.referee:
                            match.referee = referee
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
                        referee=referee or None,
                    )
                    if is_finished and home_goals is not None:
                        match.home_goals = home_goals
                        match.away_goals = away_goals
                        match.is_fixture = False
                    session.add(match)
                    session.commit()
                created += 1

        logger.info(f"API-Football fixtures {date_str}: {created} created, {updated} updated")

    async def _backfill_xg(self, days_back: int = 7):
        """Fetch xG and stats for recent matches that don't have xG data yet."""
        with self.db.get_session() as session:
            cutoff = datetime.utcnow() - timedelta(days=days_back)
            matches = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.home_xg.is_(None),
                Match.match_date >= cutoff,
                Match.apifootball_id.isnot(None),
            ).limit(self.BUDGET_XG).all()

            if not matches:
                logger.debug("No matches need xG backfill")
                return

            logger.info(f"Backfilling xG for {len(matches)} recent matches")
            match_data = [(m.id, m.apifootball_id) for m in matches]

        for match_id, fixture_id in match_data:
            if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
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

    # Leagues where teams are stored under their domestic league, not the competition.
    # Searching by league="europe/champions-league" would return nothing.
    _INTERNATIONAL_LEAGUES = frozenset({
        "europe/champions-league",
        "europe/europa-league",
        "europe/europa-conference-league",
    })

    @staticmethod
    def _names_similar(a: str, b: str) -> bool:
        """Return True when two team names plausibly refer to the same club.

        Uses multi-token prefix matching so abbreviations like "Sheff Wed" match
        "Sheffield Wednesday", while "Manchester City" does NOT match "Manchester United".

        Algorithm:
        1. Exact / substring match (fast path).
        2. Tokenise both names (min 3 chars, strip club-type tokens like fc/sc/afc).
        3. For each token in the shorter list, check if any token in the longer list
           starts with it OR it starts with that token (prefix match).
        4. Require ≥ 70 % of the shorter list's tokens to match.
        """
        if not a or not b:
            return False
        na, nb = a.lower().strip(), b.lower().strip()
        if na == nb or na in nb or nb in na:
            return True

        # Tokens that indicate club type only — ignore them
        _type = {"fc", "sc", "sk", "afc", "sfc", "cf", "bk", "fk",
                 "ac", "as", "cd", "ad", "sv", "rc", "kv", "rsc", "bsc"}

        def tok(s):
            return [w.rstrip(".") for w in s.split()
                    if len(w.rstrip(".")) >= 3 and w.rstrip(".") not in _type]

        wa, wb = tok(na), tok(nb)
        if not wa or not wb:
            return False

        # Compare against the shorter token list to handle abbreviated names
        sh, lo = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
        matches = sum(
            1 for t in sh if any(l.startswith(t) or t.startswith(l) for l in lo)
        )
        return matches / len(sh) >= 0.7

    def _get_or_create_team_id(self, name: str, league: str,
                               apifootball_team_id: int = None) -> int:
        """Find existing team by name or create a new one. Returns team ID.

        When apifootball_team_id is provided, it is stored on the team record
        so that backfill_team_history() can fetch historical seasons later.
        """
        with self.db.get_session() as session:
            def _save_api_id(team):
                if apifootball_team_id and not team.apifootball_team_id:
                    team.apifootball_team_id = apifootball_team_id
                    session.commit()

            # 1. Exact match
            team = session.query(Team).filter_by(name=name).first()
            if team:
                _save_api_id(team)
                return team.id

            # 2. Check alias map (API-Football name -> historical name)
            alias = TEAM_NAME_ALIASES.get(name)
            if alias:
                team = session.query(Team).filter_by(name=alias).first()
                if team:
                    _save_api_id(team)
                    return team.id

            # 3. Forward fuzzy: DB names that contain the API name
            team = session.query(Team).filter(
                Team.name.ilike(f"%{name}%")
            ).first()
            if team:
                _save_api_id(team)
                return team.id

            # 4. Reverse fuzzy: API name contains a DB team name (for short DB names)
            #    e.g. "Bayer Leverkusen" contains "Leverkusen"
            #    For CL/EL/ECL, teams are stored under domestic leagues — search all teams.
            name_lower = name.lower()
            if league in self._INTERNATIONAL_LEAGUES:
                # Search all teams (CL teams stored under domestic league keys)
                candidates = session.query(Team).filter(
                    Team.league.notin_(self._INTERNATIONAL_LEAGUES)
                ).all()
            else:
                candidates = session.query(Team).filter_by(league=league).all()

            for candidate in candidates:
                cname = candidate.name
                if len(cname) >= 5 and cname.lower() in name_lower:
                    _save_api_id(candidate)
                    return candidate.id

            # 4b. _names_similar fallback for same candidate pool
            for candidate in candidates:
                if self._names_similar(name, candidate.name):
                    _save_api_id(candidate)
                    return candidate.id

            # 5. Try matching without common prefixes/suffixes (FC, SK, etc.)
            stripped = name.replace("FC ", "").replace("SK ", "").replace("SC ", "").replace("AC ", "").replace("AS ", "").replace("RC ", "").replace("KV ", "").replace("CF ", "").strip()
            if stripped != name:
                team = session.query(Team).filter(
                    Team.name.ilike(f"%{stripped}%")
                ).first()
                if team:
                    _save_api_id(team)
                    return team.id

            # 6. Create new team
            country = league.split("/")[0].title() if "/" in league else ""
            team = Team(name=name, league=league, country=country,
                        apifootball_team_id=apifootball_team_id)
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

    def _find_match_by_date_league(
        self,
        league: str,
        match_dt: datetime,
        home_name_api: str,
        away_name_api: str,
    ) -> Optional[int]:
        """Fallback fixture lookup by league + date + team name similarity.

        Used when team-ID matching fails (e.g. name mismatch created a different team
        record, or the fixture was created by Flashscore/FDO under a slightly different
        name). Only considers fixtures that don't already have an apifootball_id so we
        never hijack an already-linked row.
        """
        window = timedelta(hours=26)
        with self.db.get_session() as session:
            candidates = (
                session.query(Match)
                .filter(
                    Match.league == league,
                    Match.match_date >= match_dt - window,
                    Match.match_date <= match_dt + window,
                    Match.apifootball_id.is_(None),
                )
                .all()
            )
            for m in candidates:
                ht = session.get(Team, m.home_team_id)
                at = session.get(Team, m.away_team_id)
                if not ht or not at:
                    continue
                if self._names_similar(home_name_api, ht.name) and self._names_similar(
                    away_name_api, at.name
                ):
                    return m.id
        return None

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
        """Fetch real bookmaker odds for today's fixtures from API-Football.

        Only fetches odds for today's matches (CI pipeline runs daily).
        Prioritizes top leagues and respects daily request budget.
        """
        if not self.enabled:
            return 0

        odds_budget = self._daily_limit - self._requests_today - self.BUDGET_RESERVE
        odds_budget = min(odds_budget, self.BUDGET_ODDS)

        if odds_budget <= 0:
            logger.warning("No API budget remaining for odds fetching")
            return 0

        # Get today's fixtures that have an apifootball_id
        with self.db.get_session() as session:
            today_start = datetime.combine(date.today(), datetime.min.time())
            today_end = today_start + timedelta(days=1)
            upcoming = session.query(Match).filter(
                Match.is_fixture == True,
                Match.apifootball_id.isnot(None),
                Match.match_date >= today_start,
                Match.match_date < today_end,
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

        # Always fetch odds for all today's fixtures so newly added bet types
        # (e.g. team goal markets) are picked up even if 1X2 odds already exist.
        need_odds = fixture_list

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
                            if bet_name not in self._logged_unknown_bets:
                                logger.debug(f"[Odds] Skipping unsupported bet type: '{bet_name}'")
                                self._logged_unknown_bets.add(bet_name)
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

    # ---- Historical data backfill ----

    async def backfill_team_history(self, min_matches: int = 20,
                                    seasons: List[int] = None,
                                    max_budget: int = 30,
                                    min_remaining_budget: int = 25):
        """Fetch historical match data for low-coverage teams in the database.

        Covers ALL teams with fewer than min_matches completed results, not just
        teams in today's fixtures. This ensures teams playing next week get
        backfilled now rather than on the day of the match.

        Priority: teams with the fewest matches are processed first.
        Each team × season costs 1 API request. Results saved as completed matches.
        """
        if seasons is None:
            seasons = [2023, 2024, 2025]

        # Guard: skip backfill if the remaining daily budget is too low.
        # When a second run happens on the same day (manual trigger), the first
        # run already consumed most of the 100 req/day quota.  Backfill needs at
        # least min_remaining_budget requests to be useful; below that threshold
        # we log a warning and return early to preserve budget for odds/fixtures.
        remaining = self._daily_limit - self._requests_today
        if remaining < min_remaining_budget:
            logger.warning(
                f"API-Football backfill skipped: only {remaining} requests remaining "
                f"(threshold={min_remaining_budget}). Run again tomorrow for full backfill."
            )
            return

        # Collect ALL team IDs in the DB that have upcoming fixtures (within 14 days)
        # plus any team that has ever appeared as low-coverage in a pick analysis.
        # Using a 14-day window ensures teams for next week's fixtures are covered.
        with self.db.get_session() as session:
            cutoff = datetime.combine(date.today(), datetime.min.time()) + timedelta(days=14)
            upcoming = session.query(Match).filter(
                Match.is_fixture == True,
                Match.match_date >= datetime.combine(date.today(), datetime.min.time()),
                Match.match_date <= cutoff,
            ).all()

            team_ids = set()
            for m in upcoming:
                team_ids.add(m.home_team_id)
                team_ids.add(m.away_team_id)

        if not team_ids:
            return

        # Find teams with insufficient historical data
        from sqlalchemy import or_ as _or
        low_coverage = []
        with self.db.get_session() as session:
            for team_id in team_ids:
                count = session.query(Match).filter(
                    Match.is_fixture == False,
                    Match.home_goals.isnot(None),
                    _or(Match.home_team_id == team_id, Match.away_team_id == team_id),
                ).count()

                if count < min_matches:
                    team = session.get(Team, team_id)
                    if team:
                        low_coverage.append((
                            team_id,
                            team.apifootball_team_id,
                            team.name,
                            count,
                        ))

        if not low_coverage:
            logger.debug("All fixture teams have sufficient historical data (>= %d matches)", min_matches)
            return

        # Sort by least data first so we prioritise the most data-starved teams
        low_coverage.sort(key=lambda x: x[3])
        logger.info(
            f"Historical backfill: {len(low_coverage)} low-coverage teams "
            f"(< {min_matches} matches): "
            + ", ".join(f"{n}({c})" for _, _, n, c in low_coverage[:5])
        )

        # Resolve missing API team IDs from linked fixtures before backfilling.
        # When a team was first created from a fixture fetch but the team ID wasn't
        # persisted (e.g. schema added later), we can recover it by re-fetching
        # the fixture and reading the home/away team IDs from the response.
        missing_api_id = [
            (tid, name, cnt) for tid, api_id, name, cnt in low_coverage if not api_id
        ]
        if missing_api_id:
            resolved = {}  # team_db_id -> api_team_id
            fixture_cache: dict = {}
            with self.db.get_session() as session:
                for tid, name, _ in missing_api_id:
                    fix_row = session.query(Match).filter(
                        _or(Match.home_team_id == tid, Match.away_team_id == tid),
                        Match.apifootball_id.isnot(None),
                    ).first()
                    if not fix_row:
                        continue
                    fix_api_id = fix_row.apifootball_id
                    is_home = fix_row.home_team_id == tid

                    if fix_api_id not in fixture_cache:
                        used = self._requests_today - (self._requests_today - len(fixture_cache))
                        if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
                            break
                        data = await self._api_get("/fixtures", {"id": fix_api_id})
                        fixture_cache[fix_api_id] = (
                            data.get("response", [])[0] if data and data.get("response") else None
                        )

                    fix_data = fixture_cache.get(fix_api_id)
                    if not fix_data:
                        continue

                    side_key = "home" if is_home else "away"
                    api_team_id_val = fix_data.get("teams", {}).get(side_key, {}).get("id")
                    if api_team_id_val:
                        resolved[tid] = api_team_id_val

            # Persist resolved IDs
            if resolved:
                with self.db.get_session() as session:
                    for tid, api_id in resolved.items():
                        team = session.get(Team, tid)
                        if team and not team.apifootball_team_id:
                            team.apifootball_team_id = api_id
                    session.commit()
                logger.info(f"Resolved API team IDs for {len(resolved)} teams: "
                            + ", ".join(str(i) for i in resolved.keys()))

                # Rebuild low_coverage with resolved IDs
                low_coverage = [
                    (tid, resolved.get(api_id if api_id else tid, api_id), name, cnt)
                    if not api_id else (tid, api_id, name, cnt)
                    for tid, api_id, name, cnt in low_coverage
                ]
                # Re-read from DB to get updated api_team_id
                with self.db.get_session() as session:
                    low_coverage = []
                    for team_id in team_ids:
                        count = session.query(Match).filter(
                            Match.is_fixture == False,
                            Match.home_goals.isnot(None),
                            _or(Match.home_team_id == team_id, Match.away_team_id == team_id),
                        ).count()
                        if count < min_matches:
                            team = session.get(Team, team_id)
                            if team:
                                low_coverage.append((
                                    team_id, team.apifootball_team_id, team.name, count,
                                ))
                    low_coverage.sort(key=lambda x: x[3])

        # Second-pass: for teams still missing apifootball_team_id, try /teams?search=name.
        # This covers teams created via Flashscore/FDO that were never in an API-Football
        # fixture response. Once resolved, IDs are persisted so this cost is one-time per team.
        # Hard-capped at 10 searches/day to preserve odds budget.
        still_missing = [
            (tid, name, cnt) for tid, api_id, name, cnt in low_coverage if not api_id
        ]
        if still_missing:
            search_resolved = {}  # db_team_id -> apifootball_team_id
            for tid, name, cnt in still_missing[:10]:
                if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
                    break
                # Strip apostrophes/special chars that break the API search
                # e.g. "FC Twente '65" → "FC Twente 65"
                import re as _re
                search_name = _re.sub(r"['\u2018\u2019\u201a\u201b\u2032]", "", name).strip()
                data = await self._api_get("/teams", {"search": search_name})
                if not data:
                    continue
                teams_resp = data.get("response", [])
                if teams_resp:
                    api_team_id_val = teams_resp[0].get("team", {}).get("id")
                    if api_team_id_val:
                        search_resolved[tid] = api_team_id_val
            if search_resolved:
                with self.db.get_session() as session:
                    for tid, api_id in search_resolved.items():
                        team = session.get(Team, tid)
                        if team and not team.apifootball_team_id:
                            team.apifootball_team_id = api_id
                    session.commit()
                # Patch low_coverage tuples so the backfill loop can use the resolved IDs
                id_map = {tid: api_id for tid, api_id in search_resolved.items()}
                low_coverage = [
                    (tid, id_map.get(tid, api_id), name, cnt)
                    for tid, api_id, name, cnt in low_coverage
                ]
                resolved_names = [name for tid, name, _ in still_missing if tid in search_resolved]
                logger.info(
                    f"Resolved {len(search_resolved)} team IDs via /teams?search: "
                    + ", ".join(resolved_names)
                )

        requests_before = self._requests_today

        for team_id, api_team_id, team_name, current_count in low_coverage:
            used = self._requests_today - requests_before
            if used >= max_budget:
                logger.info(f"Backfill budget exhausted ({used} requests used)")
                break

            if not api_team_id:
                logger.debug(f"No API-Football team ID for {team_name} — skipping backfill")
                continue

            logger.info(f"Backfilling history for {team_name} ({current_count} matches currently)")

            for season in seasons:
                used = self._requests_today - requests_before
                if used >= max_budget:
                    break

                data = await self._api_get("/fixtures", {
                    "team": api_team_id,
                    "season": season,
                    "status": "FT",
                })
                if not data:
                    continue

                fixtures = data.get("response", [])
                saved = 0

                for fix in fixtures:
                    league_id = fix.get("league", {}).get("id")
                    league_key = ID_TO_LEAGUE.get(league_id)
                    if not league_key:
                        continue  # Skip leagues we don't track

                    home_name = fix.get("teams", {}).get("home", {}).get("name", "")
                    away_name = fix.get("teams", {}).get("away", {}).get("name", "")
                    home_api_id = fix.get("teams", {}).get("home", {}).get("id")
                    away_api_id = fix.get("teams", {}).get("away", {}).get("id")
                    match_ts = fix.get("fixture", {}).get("timestamp")
                    fixture_api_id = fix.get("fixture", {}).get("id")
                    ref = fix.get("fixture", {}).get("referee") or ""

                    if not home_name or not away_name or not match_ts:
                        continue

                    goals = fix.get("goals", {})
                    home_goals = goals.get("home")
                    away_goals = goals.get("away")
                    if home_goals is None or away_goals is None:
                        continue

                    match_dt = datetime.utcfromtimestamp(match_ts)
                    h_id = self._get_or_create_team_id(
                        home_name, league_key, apifootball_team_id=home_api_id
                    )
                    a_id = self._get_or_create_team_id(
                        away_name, league_key, apifootball_team_id=away_api_id
                    )

                    if self._find_match_id(h_id, a_id, match_dt):
                        continue  # Already in DB

                    with self.db.get_session() as session:
                        match = Match(
                            home_team_id=h_id,
                            away_team_id=a_id,
                            match_date=match_dt,
                            league=league_key,
                            season=str(season),
                            is_fixture=False,
                            apifootball_id=fixture_api_id,
                            home_goals=home_goals,
                            away_goals=away_goals,
                            referee=ref or None,
                        )
                        session.add(match)
                        session.commit()
                        saved += 1

                if saved > 0:
                    logger.info(f"Saved {saved} historical matches for {team_name} (season {season})")

        total_used = self._requests_today - requests_before
        logger.info(f"Historical backfill complete ({total_used} API requests used)")

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
