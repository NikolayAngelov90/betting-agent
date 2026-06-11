"""API-Football scraper for xG, advanced stats, fixture data, and odds.

Uses the direct API-Football endpoint (v3.football.api-sports.io).
Free tier: 100 requests/day, seasons 2022-2024.
Includes real bookmaker odds via the /odds endpoint.
"""

import asyncio
import os
from datetime import datetime, date, timedelta, timezone
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
    "Paris Saint-Germain": "Paris SG",
    "Paris Saint Germain": "Paris SG",
    "PSG": "Paris SG",
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
    "Vitoria Guimaraes": "Guimaraes",
    "Vitória SC": "Guimaraes",
    "Vitoria SC": "Guimaraes",
    "FC Alverca": "FC Alverca",
    # Netherlands (dedup variants)
    "Fortuna Sittard": "For Sittard",
    "Sittard": "For Sittard",
    "Telstar": "Telstar 1963",
    # Poland
    "Bruk-Bet Termalica Nieciecza": "Termalica B-B.",
    "Nieciecza": "Termalica B-B.",
    # Germany (dedup)
    "Borussia Mönchengladbach": "M'gladbach",
    "Borussia Monchengladbach": "M'gladbach",
    "FC Bayern München": "Bayern Munich",
    "Bayern München": "Bayern Munich",
    # England (dedup)
    "Sheffield Wednesday": "Sheff Wed",
    "Oxford United": "Oxford",
    # France (dedup)
    "AS Saint-Étienne": "St Etienne",
    "Saint-Etienne": "St Etienne",
    "Saint Etienne": "St Etienne",
    # Greece
    "Olympiakos Piraeus": "Olympiakos",
    "Panathinaikos": "Panathinaikos",
    "Larisa": "AEL Larissa",
    "AEL Larissa FC": "AEL Larissa",
    "Larissa": "AEL Larissa",
    # Turkey
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Besiktas": "Besiktas",
    "Kasimpasa": "Kasimpasa",
    "Kasımpaşa": "Kasimpasa",
    "Kasimpasa SK": "Kasimpasa",
    "Istanbul Basaksehir": "Basaksehir",
    "İstanbul Başakşehir": "Basaksehir",
    "Basaksehir FK": "Basaksehir",
    "Sivasspor": "Sivasspor",
    "Konyaspor": "Konyaspor",
    "Antalyaspor": "Antalyaspor",
    "Gaziantep FK": "Gaziantep",
    "Rizespor": "Rizespor",
    "Caykur Rizespor": "Rizespor",
    "Hatayspor": "Hatayspor",
    "Kayserispor": "Kayserispor",
    "Adana Demirspor": "Adana Demirspor",
    "Samsunspor": "Samsunspor",
    "Pendikspor": "Pendikspor",
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
    # Turkey (with special characters + variants)
    "Fenerbahçe": "Fenerbahce",
    "Galatasaray SK": "Galatasaray",
    "Beşiktaş JK": "Besiktas",
    "Trabzonspor": "Trabzon",
    "Alanyaspor": "Alanyaspor",
    "Fatih Karagumruk": "Karagumruk",
    "Fatih Karagümrük": "Karagumruk",
    # England — additional variants
    "Accrington ST": "Accrington",
    "Charlton Athletic FC": "Charlton",
    "Cambridge Utd": "Cambridge",
    "Milton Keynes Dons": "MK Dons",
    "Nottingham Forest": "Nott'm Forest",
    # Netherlands
    "G.A. Eagles": "Go Ahead Eagles",
    # Austria
    "WSG Wattens": "Tirol",
    "WSG Tirol": "Tirol",
    # Belgium
    "Royale Union SG": "St. Gilloise",
    "Union St.-Gilloise": "St. Gilloise",
    "Union SG": "St. Gilloise",
    # Portugal
    "AFS": "AVS",
    # Germany
    "VfL Bochum": "Bochum",
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
    # "netherlands/eerste-divisie": 131,  # ID 131 = Argentina B Metropolitana, not NL Eerste Divisie
    "portugal/primeira-liga": 94,
    "belgium/jupiler-pro-league": 144,
    "turkey/super-lig": 203,
    "scotland/premiership": 179,
    "austria/bundesliga": 218,
    "switzerland/super-league": 207,
    "greece/super-league": 197,
    "denmark/superliga": 119,
    "norway/eliteserien": 103,
    "sweden/allsvenskan": 113,
    "finland/veikkausliiga": 244,
    "poland/ekstraklasa": 106,
    "romania/liga-1": 283,
    "europe/champions-league": 2,
    "europe/europa-league": 3,
    "europe/europa-conference-league": 848,
    # International — national team competitions
    "world/fifa-world-cup": 1,
    # National-team competitions used for historical backfill (training data
    # for WC predictions). Not in flashscore_leagues, so the daily fixture
    # fetch ignores them (_tracked_league_ids filter); only --backfill-wc
    # pulls them via backfill_competition_season().
    "world/friendlies": 10,
    "world/wc-qualification-africa": 29,
    "world/wc-qualification-asia": 30,
    "world/wc-qualification-concacaf": 31,
    "world/wc-qualification-europe": 32,
    "world/wc-qualification-oceania": 33,
    "world/wc-qualification-south-america": 34,
    # Continental tournaments — within the free plan's 2022-2024 season window.
    # Critical for the WC hosts (Mexico/USA/Canada play no qualifiers).
    "world/euro-championship": 4,
    "world/uefa-nations-league": 5,
    "world/africa-cup-of-nations": 6,
    "world/asian-cup": 7,
    "world/copa-america": 9,
    "world/gold-cup": 22,
    "world/concacaf-nations-league": 536,
}

# Reverse map: API-Football league ID -> our internal league key
ID_TO_LEAGUE = {v: k for k, v in LEAGUE_ID_MAP.items()}

# Leagues to fetch odds for first (highest priority)
PRIORITY_LEAGUES = [
    # International tournaments (highest priority when active)
    "world/fifa-world-cup",
    # Top 5 + European competitions
    "england/premier-league", "spain/laliga", "germany/bundesliga",
    "italy/serie-a", "france/ligue-1", "europe/champions-league",
    "europe/europa-league", "europe/europa-conference-league",
    # Strong European leagues
    "netherlands/eredivisie", "portugal/primeira-liga",
    "belgium/jupiler-pro-league", "turkey/super-lig", "scotland/premiership",
    # Lower divisions & smaller leagues
    "england/championship", "england/league-one", "england/league-two",
    "spain/laliga2", "germany/2-bundesliga", "italy/serie-b", "france/ligue-2",
    "austria/bundesliga", "switzerland/super-league", "greece/super-league",
    "denmark/superliga", "norway/eliteserien", "sweden/allsvenskan",
    "poland/ekstraklasa", "romania/liga-1",
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
    # Static total: 4+2+50+20+9 = 85.  Remaining 15 shared dynamically
    # between injuries (1 per fixture) and targeted backfill.
    # BUDGET_XG is a ceiling only — actual usage is capped by remaining daily quota.
    BUDGET_RESULTS = 4    # settlement for leagues not in football-data.org
    BUDGET_FIXTURES = 2   # fixture ids needed for odds lookup
    BUDGET_XG = 50        # xG backfill — doubled to accelerate coverage; budget-capped at runtime
    BUDGET_ODDS = 20      # odds from top 3 bookmakers (fast DB writes)
    BUDGET_RESERVE = 9    # safety margin
    # Note: no static BUDGET_INJURIES — computed dynamically as
    # min(fixture_count, remaining_after_static_steps) in injury_scraper.
    ODDS_CONCURRENCY = 3  # max concurrent fixture-odds HTTP requests (3×~10s ≈ 170s for 49 fixtures)

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

        # Only process leagues that are configured in flashscore_leagues.
        # Prevents creating fixtures/odds for non-configured leagues that
        # waste API budget and analysis time.
        _configured = set(self.config.get("scraping.flashscore_leagues", []))
        if _configured:
            self._tracked_league_ids = {
                lid for lid, lkey in ID_TO_LEAGUE.items()
                if lkey in _configured
            }
        else:
            self._tracked_league_ids = set(ID_TO_LEAGUE.keys())
        self._quota_exhausted = False  # Set True on first quota error; skips all further calls
        self._plan_restricted = False  # Set True on first plan error; skips season-restricted endpoints
        self._logged_unknown_bets: set = set()  # Suppress repeated unknown bet type logs
        self._today_fixture_count = 0  # Updated after _fetch_fixtures_by_date(today)
        self._rate_limit_retries = 0  # Count 429-triggered retries for telemetry
        self._account_suspended = False  # Set True when API returns account-suspended error
        self._xg_all_failed = False  # Set True when xG backfill attempts all return None

    def remaining_budget(self) -> int:
        """Return remaining API requests available (excluding safety reserve)."""
        return max(0, self._daily_limit - self._requests_today - self.BUDGET_RESERVE)

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

        # Retry loop for rateLimit errors with exponential backoff + jitter.
        # API-Football returns HTTP 200 with {"errors": {"rateLimit": "..."}} instead
        # of HTTP 429, so the base-scraper retry path doesn't catch it — we handle
        # it here and retry the full request up to 3 times before giving up.
        import random as _random
        for _rl_attempt in range(4):  # 1 initial attempt + up to 3 retries
            try:
                data = await self.fetch_json(url, params=params, headers=headers)
                self._requests_today += 1
                errors = data.get("errors", {})
                if errors:
                    if "rateLimit" in errors:
                        self._rate_limit_retries += 1
                        if _rl_attempt < 3:
                            _backoff = min(12 * (2 ** _rl_attempt), 60) + _random.uniform(0, 4)
                            logger.warning(
                                f"API-Football rate limited — retry {_rl_attempt + 1}/3 "
                                f"in {_backoff:.1f}s"
                            )
                            await asyncio.sleep(_backoff)
                            continue  # retry the request
                        logger.warning("API-Football rate limited — all retries exhausted")
                    elif "requests" in errors:
                        logger.warning("API-Football daily quota exhausted — skipping all further API calls")
                        self._quota_exhausted = True
                    elif "plan" in str(errors).lower():
                        # Free-tier season restriction (expected on free plan — not an error).
                        # Logs once at INFO level so it's visible but not alarming.
                        if not self._plan_restricted:
                            logger.info(
                                "API-Football free-plan: current season stats unavailable "
                                "(plan allows 2022-2024 only). "
                                "Skipping restricted endpoints for this run."
                            )
                            self._plan_restricted = True
                    elif "access" in errors:
                        # Account suspended or access revoked — stop all further requests
                        if not self._account_suspended:
                            logger.error(f"API-Football account suspended: {errors}")
                            self._account_suspended = True
                        self._quota_exhausted = True
                    else:
                        logger.error(f"API-Football errors: {errors}")
                    return None
                # Pace requests to stay under 10/min free-tier limit (6s = safe margin)
                await asyncio.sleep(6)
                return data
            except Exception as e:
                self._requests_today += 1
                logger.error(f"API-Football request failed: {e}")
                return None
        return None  # exhausted all rate-limit retries

    async def update(self):
        """Run the full API-Football update: yesterday results + today fixtures + odds + xG.

        Step order is deliberately odds-first so time-sensitive bookmaker odds for
        today's fixtures are prioritised over historical xG backfill on busy days
        (Sat/Sun 80-90 fixtures).  xG uses whatever budget remains after odds and
        the injury-scraper reserve.

        Only fetches today's fixtures and odds (the CI pipeline runs daily, so
        tomorrow's matches will be fetched when tomorrow's run executes).
        """
        if not self.enabled:
            logger.warning("API-Football key not set. Skipping.")
            return

        logger.info("Starting API-Football update")

        # 1. Fetch yesterday's results (free plan only allows yesterday+today+tomorrow)
        await self.fetch_recent_results(days_back=1)

        # 2. Fetch today's fixtures. Also populates self._today_fixture_count.
        today = date.today()
        await self._fetch_fixtures_by_date(today)

        # 3a. WC is hosted in North America (UTC-4 to UTC-7). Late evening matches
        # (e.g. 9 PM ET = 01:00 UTC) fall on the next UTC calendar day. Fetch
        # tomorrow too so those fixtures are in DB before picks are generated.
        _wc_league_id = LEAGUE_ID_MAP.get("world/fifa-world-cup")
        if _wc_league_id and _wc_league_id in self._tracked_league_ids:
            await self._fetch_fixtures_by_date(today + timedelta(days=1))

        # 3. Fetch real bookmaker odds for today's fixtures (time-sensitive — expires today).
        # Budget is dynamic: total remaining minus injury-scraper reserve so we never
        # starve the injury step.  On Sat/Sun with 80+ fixtures this yields ~49 odds
        # requests vs the old fixed cap of 20.
        await self.fetch_upcoming_odds()

        # 4. Backfill xG for historical matches (not time-sensitive — uses leftover budget).
        # On high-fixture weekends the injury reserve leaves 0 for xG; it resumes next
        # weekday.  On quiet weekdays it gets the full BUDGET_XG=25 allocation.
        await self._backfill_xg()

        # 5. Historical backfill — DISABLED for CI speed.
        # backfill_team_history used 40 requests + 30 min on obscure teams (Pescara,
        # Boulogne, Nancy). The 30K+ matches in Neon already provide sufficient
        # coverage for top league teams. Re-enable for local runs if needed.

        logger.info(f"API-Football update complete ({self._requests_today} requests used)")

    async def fetch_lineups(self, fixture_id: int) -> dict:
        """Fetch confirmed starting XIs for a fixture (available ~20-45 min pre-KO).

        Returns a dict keyed by side:
            {
              "home": {"team": str, "formation": str,
                       "start_xi": [{"name", "number", "pos"}],
                       "substitutes": [...], "coach": str},
              "away": {...},
            }
        Empty dict when lineups are not yet published (API returns []).
        """
        if not self.enabled or not fixture_id:
            return {}

        data = await self._api_get("/fixtures/lineups", {"fixture": fixture_id})
        if not data:
            return {}

        resp = data.get("response", [])
        if not resp:
            return {}  # lineups not published yet

        def _players(raw_list):
            out = []
            for entry in raw_list or []:
                p = entry.get("player", {}) if isinstance(entry, dict) else {}
                name = p.get("name")
                if not name:
                    continue
                out.append({
                    "name": name,
                    "number": p.get("number"),
                    "pos": p.get("pos"),
                })
            return out

        result = {}
        # API returns the home team first, away team second (by convention).
        for idx, side in enumerate(("home", "away")):
            if idx >= len(resp):
                break
            block = resp[idx]
            result[side] = {
                "team": (block.get("team") or {}).get("name", ""),
                "formation": block.get("formation") or "",
                "start_xi": _players(block.get("startXI")),
                "substitutes": _players(block.get("substitutes")),
                "coach": (block.get("coach") or {}).get("name", ""),
            }
        return result

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

        # Pre-load all existing apifootball_id → match_id for this date in one query.
        # This avoids a per-fixture DB round-trip (1s Neon latency) for existing fixtures.
        day_start = datetime.combine(target_date, datetime.min.time())
        day_end = day_start + timedelta(days=1)
        with self.db.get_session() as session:
            existing_rows = session.query(Match.id, Match.apifootball_id).filter(
                Match.apifootball_id.isnot(None),
                Match.match_date >= day_start,
                Match.match_date < day_end,
            ).all()
        _afid_to_match_id = {row.apifootball_id: row.id for row in existing_rows}

        for fix in fixtures:
            league_id = fix.get("league", {}).get("id")
            if league_id not in self._tracked_league_ids:
                continue  # Skip leagues not in configured flashscore_leagues

            league_key = ID_TO_LEAGUE[league_id]
            fixture_id = fix.get("fixture", {}).get("id")
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")
            home_api_id = fix.get("teams", {}).get("home", {}).get("id")
            away_api_id = fix.get("teams", {}).get("away", {}).get("id")
            match_ts = fix.get("fixture", {}).get("timestamp")
            status_short = fix.get("fixture", {}).get("status", {}).get("short", "")
            referee = fix.get("fixture", {}).get("referee") or ""
            round_name = fix.get("league", {}).get("round") or None

            if not home_name or not away_name:
                continue

            match_dt = datetime.utcfromtimestamp(match_ts) if match_ts else None
            if not match_dt:
                continue

            goals = fix.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            is_finished = status_short in ("FT", "AET", "PEN")
            is_fixture = status_short in ("NS", "TBD", "")

            score = fix.get("score", {})
            venue_city = (fix.get("fixture", {}).get("venue") or {}).get("city") or ""
            ht_home = score.get("halftime", {}).get("home")
            ht_away = score.get("halftime", {}).get("away")
            reg_home = score.get("fulltime", {}).get("home")
            reg_away = score.get("fulltime", {}).get("away")
            pen_home = score.get("penalty", {}).get("home")
            pen_away = score.get("penalty", {}).get("away")

            # Fast path: fixture already known by apifootball_id (pre-loaded cache).
            # Only update score/referee — skip all team lookups and fuzzy matching.
            if fixture_id in _afid_to_match_id:
                match_id = _afid_to_match_id[fixture_id]
                if is_finished and home_goals is not None:
                    with self.db.get_session() as session:
                        match = session.get(Match, match_id)
                        if match:
                            # Never overwrite a score that was already stored —
                            # the first source to set the score wins.  Flashscore
                            # runs later and will correct if the AF score is wrong;
                            # this guard prevents AF from re-reverting that fix.
                            if (match.home_goals is not None
                                    and (match.home_goals != home_goals
                                         or match.away_goals != away_goals)):
                                logger.debug(
                                    f"API-Football score ignored for match id={match_id}: "
                                    f"DB={match.home_goals}-{match.away_goals} "
                                    f"API={home_goals}-{away_goals} — keeping existing score"
                                )
                            else:
                                match.home_goals = home_goals
                                match.away_goals = away_goals
                            match.is_fixture = False
                            if referee and not match.referee:
                                match.referee = referee
                            if venue_city and not match.venue:
                                match.venue = venue_city
                            if ht_home is not None and match.ht_home_goals is None:
                                match.ht_home_goals = ht_home
                                match.ht_away_goals = ht_away
                            if reg_home is not None and match.regulation_home_goals is None:
                                match.regulation_home_goals = reg_home
                                match.regulation_away_goals = reg_away
                            if pen_home is not None and match.penalty_home_score is None:
                                match.penalty_home_score = pen_home
                                match.penalty_away_score = pen_away
                            session.commit()
                updated += 1
                continue

            # Slow path: new fixture — get/create teams and find/create match record.
            # Get or create teams — also saves API-Football team IDs for future backfill
            home_team_id = self._get_or_create_team_id(
                home_name, league_key, apifootball_team_id=home_api_id
            )
            away_team_id = self._get_or_create_team_id(
                away_name, league_key, apifootball_team_id=away_api_id
            )

            # Check for existing match — by team DB IDs then fuzzy name search
            # (apifootball_id lookup already handled by the pre-loaded cache above).
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

            if match_id:
                # Update existing match (linked by team IDs or fuzzy match)
                with self.db.get_session() as session:
                    match = session.get(Match, match_id)
                    if match:
                        match.apifootball_id = fixture_id
                        if round_name and not match.round:
                            match.round = round_name
                        if referee and not match.referee:
                            match.referee = referee
                        if venue_city and not match.venue:
                            match.venue = venue_city
                        if is_finished and home_goals is not None:
                            # Never overwrite an existing score — first source wins.
                            if not (match.home_goals is not None
                                    and (match.home_goals != home_goals
                                         or match.away_goals != away_goals)):
                                match.home_goals = home_goals
                                match.away_goals = away_goals
                            match.is_fixture = False
                            if ht_home is not None and match.ht_home_goals is None:
                                match.ht_home_goals = ht_home
                                match.ht_away_goals = ht_away
                            if reg_home is not None and match.regulation_home_goals is None:
                                match.regulation_home_goals = reg_home
                                match.regulation_away_goals = reg_away
                            if pen_home is not None and match.penalty_home_score is None:
                                match.penalty_home_score = pen_home
                                match.penalty_away_score = pen_away
                        session.commit()
                _afid_to_match_id[fixture_id] = match_id  # warm cache for future use
                updated += 1
            else:
                # Create new match — log at INFO so we can trace missing fixtures
                logger.info(
                    f"API-Football: creating new fixture {home_name} vs {away_name} "
                    f"({league_key}, {date_str}, afid={fixture_id})"
                )
                with self.db.get_session() as session:
                    match = Match(
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        match_date=match_dt,
                        league=league_key,
                        season=self._season_from_fixture(fix, match_dt, league_key),
                        round=round_name,
                        is_fixture=is_fixture,
                        apifootball_id=fixture_id,
                        referee=referee or None,
                        venue=venue_city or None,
                    )
                    if is_finished and home_goals is not None:
                        match.home_goals = home_goals
                        match.away_goals = away_goals
                        match.is_fixture = False
                        if ht_home is not None:
                            match.ht_home_goals = ht_home
                            match.ht_away_goals = ht_away
                        if reg_home is not None:
                            match.regulation_home_goals = reg_home
                            match.regulation_away_goals = reg_away
                        if pen_home is not None:
                            match.penalty_home_score = pen_home
                            match.penalty_away_score = pen_away
                    session.add(match)
                    session.commit()
                created += 1

        logger.info(f"API-Football fixtures {date_str}: {created} created, {updated} updated")
        if target_date == date.today():
            # Track how many tracked fixtures are on today's slate so odds and
            # xG backfill steps can allocate budget dynamically.
            with self.db.get_session() as session:
                today_start = datetime.combine(target_date, datetime.min.time())
                today_end = today_start + timedelta(days=1)
                self._today_fixture_count = session.query(Match).filter(
                    Match.is_fixture == True,
                    Match.apifootball_id.isnot(None),
                    Match.match_date >= today_start,
                    Match.match_date < today_end,
                ).count()

    async def _backfill_xg(self, days_back: int = 365):
        """Fetch xG and stats for recent matches missing xG OR match stats.

        This is the primary stats enrichment path — much faster than Flashscore
        scraping (~0.5s/match via API vs ~12s/match via browser).

        Each API call has a mandatory 6s rate-limit sleep (free tier: 10/min),
        so 25 matches ≈ 175s, 50 matches ≈ 350s.  Budget is capped at runtime by
        BUDGET_XG and remaining daily quota.  We batch DB writes into a single
        commit at the end to save ~50ms/match of Neon latency.
        """
        # Compute available budget: reserve room for injury-scraper that runs after this.
        injury_reserve = min(40, self._today_fixture_count + 10)
        xg_budget = self._daily_limit - self._requests_today - self.BUDGET_RESERVE - injury_reserve
        xg_budget = max(0, min(xg_budget, self.BUDGET_XG))
        if xg_budget == 0:
            logger.debug(
                f"xG backfill skipped — no budget left after odds+injury reserve "
                f"({self._today_fixture_count} fixtures today)"
            )
            return

        from sqlalchemy import or_
        # API-Football free plan limits stats to a fixed range of seasons.
        # The cutoff used to be hard-coded to 2025-07-01 which silently dropped
        # all xG backfill from that date forward.  We now read it from config
        # (data_sources.apifootball_stats_cutoff, ISO date string), and the
        # _plan_restricted flag set on the first 403/plan error stops further
        # requests automatically — so leaving the cutoff unset is safe.
        cutoff_iso = self.config.get("data_sources.apifootball_stats_cutoff")
        if cutoff_iso:
            try:
                _FREE_PLAN_SEASON_CUTOFF = datetime.fromisoformat(cutoff_iso)
            except Exception:
                _FREE_PLAN_SEASON_CUTOFF = None
        else:
            _FREE_PLAN_SEASON_CUTOFF = None
        with self.db.get_session() as session:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_back)
            base_filters = [
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.apifootball_id.isnot(None),
                Match.match_date >= cutoff,
            ]
            if _FREE_PLAN_SEASON_CUTOFF is not None:
                base_filters.append(Match.match_date < _FREE_PLAN_SEASON_CUTOFF)
            # Count all matches in window and those missing xG (without limit so the
            # "already complete" number is accurate even when missing pool >= budget).
            total_in_window = session.query(Match).filter(*base_filters).count()
            missing_total = session.query(Match).filter(
                *base_filters,
                or_(Match.home_xg.is_(None), Match.home_shots.is_(None)),
            ).count()
            already_complete = total_in_window - missing_total

            # Only fetch matches actually missing xG or stats, capped at budget
            matches = session.query(Match).filter(
                *base_filters,
                or_(Match.home_xg.is_(None), Match.home_shots.is_(None)),
            ).order_by(Match.match_date.desc()).limit(xg_budget).all()

            if not matches:
                logger.info(
                    f"xG backfill: all {total_in_window} recent matches have xG data, skipping"
                )
                return

            needed = len(matches)
            logger.info(
                f"Backfilling xG/stats for {needed} recent matches via API-Football "
                f"({already_complete}/{total_in_window} already complete, "
                f"{missing_total} total missing)"
            )
            match_data = [(m.id, m.apifootball_id) for m in matches]

        # Fetch all stats first, then batch-write to DB in a single commit
        pending_updates: list = []
        for match_id, fixture_id in match_data:
            if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
                logger.warning("Approaching API limit, stopping stats backfill")
                break

            stats = await self._fetch_fixture_stats(fixture_id)
            if stats:
                pending_updates.append((match_id, stats))

        # Batch commit all updates in a single session
        if pending_updates:
            self._batch_update_match_stats(pending_updates)
            logger.info(
                f"xG backfill: {len(pending_updates)}/{needed} matches updated "
                f"({already_complete + len(pending_updates)}/{total_in_window} now complete)"
            )
        elif needed > 0:
            logger.warning(f"xG backfill: {needed} matches needed but 0 updated — all API calls failed")
            self._xg_all_failed = True

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
                elif stat_type == "Goalkeeper Saves":
                    result[side]["saves"] = _safe_int(stat_value)
                elif stat_type == "Offsides":
                    result[side]["offsides"] = _safe_int(stat_value)

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
            if match.home_saves is None and "saves" in home:
                match.home_saves = home["saves"]
            if match.away_saves is None and "saves" in away:
                match.away_saves = away["saves"]
            if match.home_offsides is None and "offsides" in home:
                match.home_offsides = home["offsides"]
            if match.away_offsides is None and "offsides" in away:
                match.away_offsides = away["offsides"]

            session.commit()
            logger.debug(f"Updated stats for match {match_id} (xG: {home.get('xg')}-{away.get('xg')})")

    def _batch_update_match_stats(self, updates: list):
        """Write xG and stats for multiple matches in a single DB commit.

        Saves ~50ms of Neon latency per match vs individual commits.
        """
        with self.db.get_session() as session:
            for match_id, stats in updates:
                match = session.get(Match, match_id)
                if not match:
                    continue
                home = stats.get("home", {})
                away = stats.get("away", {})
                if "xg" in home:
                    match.home_xg = home["xg"]
                if "xg" in away:
                    match.away_xg = away["xg"]
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
                if match.home_saves is None and "saves" in home:
                    match.home_saves = home["saves"]
                if match.away_saves is None and "saves" in away:
                    match.away_saves = away["saves"]
                if match.home_offsides is None and "offsides" in home:
                    match.home_offsides = home["offsides"]
                if match.away_offsides is None and "offsides" in away:
                    match.away_offsides = away["offsides"]
            session.commit()

    async def _fetch_fixture_detail(self, fixture_id: int) -> Optional[Dict]:
        """Fetch venue, halftime, regulation, and penalty scores for a single fixture."""
        data = await self._api_get("/fixtures", {"id": fixture_id})
        if not data:
            return None
        response = data.get("response", [])
        if not response:
            return None
        fix = response[0]
        score = fix.get("score", {})
        venue_city = ((fix.get("fixture") or {}).get("venue") or {}).get("city") or ""
        ht_home = score.get("halftime", {}).get("home")
        ht_away = score.get("halftime", {}).get("away")
        reg_home = score.get("fulltime", {}).get("home")
        reg_away = score.get("fulltime", {}).get("away")
        pen_home = score.get("penalty", {}).get("home")
        pen_away = score.get("penalty", {}).get("away")
        return {
            "venue": venue_city or None,
            "ht_home": ht_home,
            "ht_away": ht_away,
            "reg_home": reg_home,
            "reg_away": reg_away,
            "pen_home": pen_home,
            "pen_away": pen_away,
        }

    def _batch_update_fixture_details(self, updates: list):
        """Write venue/halftime/regulation/penalty for multiple matches in one commit."""
        with self.db.get_session() as session:
            for match_id, detail in updates:
                match = session.get(Match, match_id)
                if not match:
                    continue
                if detail.get("venue") and not match.venue:
                    match.venue = detail["venue"]
                if detail.get("ht_home") is not None and match.ht_home_goals is None:
                    match.ht_home_goals = detail["ht_home"]
                if detail.get("ht_away") is not None and match.ht_away_goals is None:
                    match.ht_away_goals = detail["ht_away"]
                if detail.get("reg_home") is not None and match.regulation_home_goals is None:
                    match.regulation_home_goals = detail["reg_home"]
                if detail.get("reg_away") is not None and match.regulation_away_goals is None:
                    match.regulation_away_goals = detail["reg_away"]
                if detail.get("pen_home") is not None and match.penalty_home_score is None:
                    match.penalty_home_score = detail["pen_home"]
                if detail.get("pen_away") is not None and match.penalty_away_score is None:
                    match.penalty_away_score = detail["pen_away"]
            session.commit()

    async def backfill_match_stats(self, budget: int = 80) -> dict:
        """Bulk backfill statistics and fixture metadata for historical matches.

        Designed for manual use via --backfill-stats only — not called during
        the daily update cycle. Processes two passes within the given budget:

          Pass 1 (60% of budget) — /fixtures/statistics:
            xG, shots, possession, corners, fouls, cards, saves, offsides.
            Targets completed matches with apifootball_id that are missing xG or shots.

          Pass 2 (40% of budget) — /fixtures?id=:
            venue, halftime scores, regulation scores, penalty scores.
            Targets completed matches with apifootball_id that are missing venue or halftime.

        Both passes process newest matches first (most useful for ML training).

        Args:
            budget: Maximum API requests to spend across both passes (default 80).
                    Run on a day with no prior daily update to avoid exceeding the
                    100 req/day free-tier limit.

        Returns:
            {"stats": N, "details": M} — number of matches updated per pass.
        """
        from sqlalchemy import or_

        if not self.enabled:
            logger.warning("API-Football not enabled — backfill-stats skipped")
            return {"stats": 0, "details": 0}

        stats_budget = round(budget * 0.6)
        detail_budget = budget - stats_budget

        # ── Pass 1: statistics ──────────────────────────────────────────────
        with self.db.get_session() as session:
            stats_needed = session.query(Match).filter(
                Match.apifootball_id.isnot(None),
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                or_(Match.home_xg.is_(None), Match.home_shots.is_(None)),
            ).order_by(Match.match_date.desc()).limit(stats_budget).all()
            stats_queue = [(m.id, m.apifootball_id) for m in stats_needed]

        logger.info(
            f"Backfill stats pass: {len(stats_queue)} matches to process "
            f"(budget={stats_budget})"
        )

        stats_done = 0
        pending_stats: list = []
        for match_id, fixture_id in stats_queue:
            if self._requests_today >= self._daily_limit - 2:
                logger.warning(f"API limit approached — stopping stats pass at {stats_done}")
                break
            stats = await self._fetch_fixture_stats(fixture_id)
            if stats:
                pending_stats.append((match_id, stats))
                stats_done += 1
            await asyncio.sleep(0.12)

        if pending_stats:
            self._batch_update_match_stats(pending_stats)
        logger.info(f"Stats pass complete: {stats_done}/{len(stats_queue)} matches updated")

        # ── Pass 2: fixture details ─────────────────────────────────────────
        with self.db.get_session() as session:
            detail_needed = session.query(Match).filter(
                Match.apifootball_id.isnot(None),
                Match.is_fixture == False,
                or_(Match.venue.is_(None), Match.ht_home_goals.is_(None)),
            ).order_by(Match.match_date.desc()).limit(detail_budget).all()
            detail_queue = [(m.id, m.apifootball_id) for m in detail_needed]

        logger.info(
            f"Backfill details pass: {len(detail_queue)} matches to process "
            f"(budget={detail_budget})"
        )

        detail_done = 0
        pending_details: list = []
        for match_id, fixture_id in detail_queue:
            if self._requests_today >= self._daily_limit - 2:
                logger.warning(f"API limit approached — stopping details pass at {detail_done}")
                break
            detail = await self._fetch_fixture_detail(fixture_id)
            if detail:
                pending_details.append((match_id, detail))
                detail_done += 1
            await asyncio.sleep(0.12)

        if pending_details:
            self._batch_update_fixture_details(pending_details)
        logger.info(f"Details pass complete: {detail_done}/{len(detail_queue)} matches updated")

        return {"stats": stats_done, "details": detail_done}

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
            from src.utils.team_names import team_names_similar
            for m in candidates:
                ht = session.get(Team, m.home_team_id)
                at = session.get(Team, m.away_team_id)
                if not ht or not at:
                    continue
                if team_names_similar(home_name_api, ht.name) and team_names_similar(
                    away_name_api, at.name
                ):
                    return m.id
        return None

    def _get_season(self, dt: datetime, league: str = None) -> str:
        """Determine the season string from a match date.

        International tournaments (WC, Euros, etc.) run in a single calendar
        year and API-Football uses that year as the season key — return "2026"
        not "2025/2026" for a June 2026 WC match.
        """
        from src.models.poisson_model import NATIONAL_TEAM_LEAGUES
        if league and league in NATIONAL_TEAM_LEAGUES:
            return str(dt.year)
        year = dt.year
        if dt.month >= 7:
            return f"{year}/{year + 1}"
        else:
            return f"{year - 1}/{year}"

    def _season_from_fixture(self, fix: dict, match_dt: datetime, league_key: str) -> str:
        """Return season string, preferring the API-provided season value."""
        api_season = fix.get("league", {}).get("season")
        if api_season is not None:
            # API-Football returns an integer season year for all competitions.
            # Use it directly — this is more reliable than deriving from date.
            return str(api_season)
        return self._get_season(match_dt, league=league_key)

    # ---- Odds fetching ----

    def _make_odds_semaphore(self, remaining_budget: int, n_fixtures: int) -> asyncio.Semaphore:
        # Serial (cap=1): API-Football free tier allows 10 req/min = 1 every 6 s.
        # Each request already sleeps 6 s after the response, so cap=1 guarantees
        # exactly one in-flight request at a time and eliminates HTTP 429 bursts.
        # cap=2 caused concurrent requests 2 s apart → both within a single 6 s
        # window → rate-limit errors on every run with ≥2 fixtures.
        capacity = max(1, min(1, remaining_budget, n_fixtures))
        return asyncio.Semaphore(capacity)

    async def _fetch_odds_guarded(self, sem, match_id: int, fixture_id: int, league: str) -> tuple:
        async with sem:
            if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
                logger.debug(f"Quota limit reached — skipping odds for fixture {fixture_id}")
                return (match_id, None, league)
            odds_data = await self._fetch_fixture_odds(fixture_id)
            return (match_id, odds_data, league)

    async def fetch_upcoming_odds(self):
        """Fetch real bookmaker odds for today's fixtures from API-Football.

        Only fetches odds for today's matches (CI pipeline runs daily).
        Prioritizes top leagues and respects daily request budget.
        """
        if not self.enabled:
            return 0

        # Dynamic budget: reserve room for the injury-scraper step that runs after us.
        # On Sat/Sun with 80+ fixtures this is much larger than the fixed BUDGET_ODDS=20.
        # injury_reserve = 1 request per fixture (fixture-level) + 10 team fallback, max 40.
        injury_reserve = min(40, self._today_fixture_count + 10)
        odds_budget = self._daily_limit - self._requests_today - self.BUDGET_RESERVE - injury_reserve
        # Allow up to fixture_count odds requests (one per match), min is old fixed cap.
        odds_cap = max(self.BUDGET_ODDS, self._today_fixture_count)
        odds_budget = max(0, min(odds_budget, odds_cap))

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

        # Count existing real (non-Flashscore) odds per fixture so we can
        # prioritise matches that have NO odds yet — critical when budget is
        # limited after multiple runs in the same day.
        with self.db.get_session() as session:
            from src.data.models import Odds
            odds_counts: dict = {}
            for mid, _afid, _lg in fixture_list:
                cnt = session.query(Odds).filter(
                    Odds.match_id == mid,
                    Odds.bookmaker != "Flashscore",
                ).count()
                odds_counts[mid] = cnt

        # Only fetch odds for configured leagues — skip lower divisions that
        # have apifootball_id but aren't in PRIORITY_LEAGUES (prevents
        # wasting budget on league-one/league-two/etc.)
        priority_set = set(PRIORITY_LEAGUES)
        fixture_list = [
            (mid, afid, lg) for mid, afid, lg in fixture_list
            if lg in priority_set
        ]

        if not fixture_list:
            logger.debug("No priority-league fixtures need odds")
            return 0

        # Sort: fixtures with fewest existing odds first, then by league priority
        def sort_key(item):
            mid, _afid, league = item
            existing = odds_counts.get(mid, 0)
            try:
                lp = PRIORITY_LEAGUES.index(league)
            except ValueError:
                lp = len(PRIORITY_LEAGUES)
            return (existing, lp)

        fixture_list.sort(key=sort_key)

        _ODDS_TIME_BUDGET_S = 600  # 10-minute hard cap on odds fetching

        sem = self._make_odds_semaphore(
            min(self.ODDS_CONCURRENCY, odds_budget), len(fixture_list)
        )

        _DISPATCH_DELAY_S = 2.0  # stagger task dispatch to prevent concurrent 429 bursts

        async def _rate_limited_gather():
            tasks = []
            for i, (match_id, fixture_id, league) in enumerate(fixture_list[:odds_budget]):
                if i > 0:
                    await asyncio.sleep(_DISPATCH_DELAY_S)
                tasks.append(
                    asyncio.create_task(
                        self._fetch_odds_guarded(sem, match_id, fixture_id, league)
                    )
                )
            return await asyncio.gather(*tasks, return_exceptions=True)

        _retries_before = getattr(self, "_rate_limit_retries", 0)
        try:
            results = await asyncio.wait_for(
                _rate_limited_gather(),
                timeout=_ODDS_TIME_BUDGET_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"Odds fetch time budget exhausted ({_ODDS_TIME_BUDGET_S // 60} min)"
            )
            results = []
        new_retries = getattr(self, "_rate_limit_retries", 0) - _retries_before
        if new_retries > 0:
            logger.warning(f"API-Football retries: {new_retries} (target: 0)")
        else:
            logger.debug("API-Football retries: 0 (target: 0)")

        fetched = 0
        for result in results:
            if isinstance(result, BaseException):
                logger.warning(f"Odds fetch error (fixture skipped): {result}")
                continue
            match_id, odds_data, league = result
            if odds_data:
                count = self._save_fixture_odds(match_id, odds_data)
                if count > 0:
                    fetched += 1
                    logger.debug(f"Saved {count} odds for match {match_id} ({league})")
                else:
                    logger.debug(
                        f"No top-bookmaker odds for match {match_id} ({league})"
                    )

        logger.info(f"Fetched odds for {fetched} upcoming fixtures ({self._requests_today} API requests used)")
        return fetched

    async def _fetch_fixture_odds(self, fixture_id: int) -> Optional[List]:
        """Fetch bookmaker odds for a specific fixture."""
        data = await self._api_get("/odds", {"fixture": fixture_id})
        if not data:
            return None
        return data.get("response", [])

    # Primary bookmakers (preferred). If none cover a fixture, fall back to
    # the secondary tier so minor leagues still get odds.
    _TOP_BOOKMAKERS = {"Bet365", "1xBet", "Pinnacle"}
    _FALLBACK_BOOKMAKERS = {"Unibet", "Betfair", "William Hill", "Bwin", "Betway"}

    def _save_fixture_odds(self, match_id: int, odds_response: list) -> int:
        """Parse API-Football odds response and save to database.

        Uses primary bookmakers first; falls back to secondary tier if the
        primary set produces zero odds for this fixture.

        Returns count of odds records created.
        """
        count = self._save_odds_from_set(match_id, odds_response, self._TOP_BOOKMAKERS)
        if count == 0:
            count = self._save_odds_from_set(match_id, odds_response, self._FALLBACK_BOOKMAKERS)
            if count > 0:
                logger.debug(f"Match {match_id}: primary bookmakers had no odds, saved {count} from fallback tier")
        return count

    def _save_odds_from_set(self, match_id: int, odds_response: list,
                            allowed_bookmakers: set) -> int:
        """Save odds for a fixture, filtered to the given bookmaker set."""
        count = 0

        with self.db.get_session() as session:
            # Preload all existing Odds rows for this match in one query instead
            # of one SELECT per row — reduces Neon roundtrips from N_rows to 1.
            existing_rows = session.query(Odds).filter_by(match_id=match_id).all()
            existing_index = {
                (r.bookmaker, r.market_type, r.selection): r
                for r in existing_rows
            }

            for entry in odds_response:
                bookmakers = entry.get("bookmakers", [])
                for bookie in bookmakers:
                    bookie_name = bookie.get("name", "Unknown")
                    if bookie_name not in allowed_bookmakers:
                        continue
                    bets = bookie.get("bets", [])

                    for bet in bets:
                        bet_name = bet.get("name", "")
                        bet_mapping = BET_TYPE_MAP.get(bet_name)
                        if not bet_mapping:
                            if bet_name not in self._logged_unknown_bets:
                                logger.trace(f"[Odds] Skipping unsupported bet type: '{bet_name}'")
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

                            key = (bookie_name, market_type, selection)
                            existing = existing_index.get(key)
                            if existing:
                                # Update if odds changed; preserve opening_odds
                                if existing.odds_value != odds_value:
                                    if existing.opening_odds is None:
                                        existing.opening_odds = existing.odds_value
                                    existing.odds_value = odds_value
                                    existing.timestamp = datetime.now(timezone.utc).replace(tzinfo=None)
                                continue

                            odds = Odds(
                                match_id=match_id,
                                bookmaker=bookie_name,
                                market_type=market_type,
                                selection=selection,
                                odds_value=odds_value,
                                opening_odds=odds_value,
                            )
                            session.add(odds)
                            existing_index[key] = odds  # prevent re-insert on dup
                            count += 1

            session.commit()

        return count

    # ---- Historical data backfill ----

    async def backfill_competition_season(
        self, league_key: str, season: int
    ) -> tuple[int, int]:
        """Fetch all fixtures for a competition+season and upsert into DB.

        Unlike _fetch_fixtures_by_date(), this ignores _tracked_league_ids so
        that competition backfills (e.g. WC 2022) always run regardless of the
        current flashscore_leagues config.

        Returns (created, updated) counts.
        """
        league_id = LEAGUE_ID_MAP.get(league_key)
        if league_id is None:
            logger.warning(
                f"backfill_competition_season: unknown league key '{league_key}'"
            )
            return 0, 0

        data = await self._api_get("/fixtures", {"league": league_id, "season": season})
        if not data:
            return 0, 0

        fixtures = data.get("response", [])
        logger.info(
            f"API-Football backfill {league_key} {season}: {len(fixtures)} fixtures"
        )

        # Pre-load existing apifootball_id → match_id for all WC matches in DB.
        with self.db.get_session() as session:
            existing_rows = session.query(Match.id, Match.apifootball_id).filter(
                Match.league == league_key,
                Match.apifootball_id.isnot(None),
            ).all()
        _afid_to_match_id = {row.apifootball_id: row.id for row in existing_rows}

        created = 0
        updated = 0

        for fix in fixtures:
            fixture_id = fix.get("fixture", {}).get("id")
            home_name = fix.get("teams", {}).get("home", {}).get("name", "")
            away_name = fix.get("teams", {}).get("away", {}).get("name", "")
            home_api_id = fix.get("teams", {}).get("home", {}).get("id")
            away_api_id = fix.get("teams", {}).get("away", {}).get("id")
            match_ts = fix.get("fixture", {}).get("timestamp")
            status_short = fix.get("fixture", {}).get("status", {}).get("short", "")
            referee = fix.get("fixture", {}).get("referee") or ""
            round_name = fix.get("league", {}).get("round") or None

            if not home_name or not away_name or not match_ts:
                continue

            match_dt = datetime.utcfromtimestamp(match_ts)
            goals = fix.get("goals", {})
            home_goals = goals.get("home")
            away_goals = goals.get("away")
            is_finished = status_short in ("FT", "AET", "PEN")
            is_fixture = status_short in ("NS", "TBD", "")
            venue_city = (fix.get("fixture", {}).get("venue") or {}).get("city") or ""

            score = fix.get("score", {})
            ht_home = score.get("halftime", {}).get("home")
            ht_away = score.get("halftime", {}).get("away")
            reg_home = score.get("fulltime", {}).get("home")
            reg_away = score.get("fulltime", {}).get("away")
            pen_home = score.get("penalty", {}).get("home")
            pen_away = score.get("penalty", {}).get("away")

            # Fast path: already in DB by apifootball_id.
            if fixture_id in _afid_to_match_id:
                match_id = _afid_to_match_id[fixture_id]
                if is_finished and home_goals is not None:
                    with self.db.get_session() as session:
                        match = session.get(Match, match_id)
                        if match and match.home_goals is None:
                            match.home_goals = home_goals
                            match.away_goals = away_goals
                            match.is_fixture = False
                            if ht_home is not None and match.ht_home_goals is None:
                                match.ht_home_goals = ht_home
                                match.ht_away_goals = ht_away
                            if reg_home is not None and match.regulation_home_goals is None:
                                match.regulation_home_goals = reg_home
                                match.regulation_away_goals = reg_away
                            if pen_home is not None and match.penalty_home_score is None:
                                match.penalty_home_score = pen_home
                                match.penalty_away_score = pen_away
                            session.commit()
                            updated += 1
                continue

            # Slow path: create/find teams and match.
            home_team_id = self._get_or_create_team_id(
                home_name, league_key, apifootball_team_id=home_api_id
            )
            away_team_id = self._get_or_create_team_id(
                away_name, league_key, apifootball_team_id=away_api_id
            )

            match_id = self._find_match_id(home_team_id, away_team_id, match_dt)
            if match_id is None:
                match_id = self._find_match_by_date_league(
                    league_key, match_dt, home_name, away_name
                )

            if match_id:
                with self.db.get_session() as session:
                    match = session.get(Match, match_id)
                    if match:
                        match.apifootball_id = fixture_id
                        if round_name and not match.round:
                            match.round = round_name
                        if referee and not match.referee:
                            match.referee = referee
                        if venue_city and not match.venue:
                            match.venue = venue_city
                        if is_finished and home_goals is not None and match.home_goals is None:
                            match.home_goals = home_goals
                            match.away_goals = away_goals
                            match.is_fixture = False
                            if ht_home is not None and match.ht_home_goals is None:
                                match.ht_home_goals = ht_home
                                match.ht_away_goals = ht_away
                            if reg_home is not None and match.regulation_home_goals is None:
                                match.regulation_home_goals = reg_home
                                match.regulation_away_goals = reg_away
                            if pen_home is not None and match.penalty_home_score is None:
                                match.penalty_home_score = pen_home
                                match.penalty_away_score = pen_away
                        session.commit()
                _afid_to_match_id[fixture_id] = match_id
                updated += 1
            else:
                with self.db.get_session() as session:
                    match = Match(
                        home_team_id=home_team_id,
                        away_team_id=away_team_id,
                        match_date=match_dt,
                        league=league_key,
                        season=self._season_from_fixture(fix, match_dt, league_key),
                        round=round_name,
                        is_fixture=is_fixture,
                        apifootball_id=fixture_id,
                        referee=referee or None,
                        venue=venue_city or None,
                    )
                    if is_finished and home_goals is not None:
                        match.home_goals = home_goals
                        match.away_goals = away_goals
                        match.is_fixture = False
                        if ht_home is not None:
                            match.ht_home_goals = ht_home
                            match.ht_away_goals = ht_away
                        if reg_home is not None:
                            match.regulation_home_goals = reg_home
                            match.regulation_away_goals = reg_away
                        if pen_home is not None:
                            match.penalty_home_score = pen_home
                            match.penalty_away_score = pen_away
                    session.add(match)
                    session.commit()
                created += 1

        logger.info(
            f"API-Football backfill {league_key} {season}: "
            f"{created} created, {updated} updated"
        )
        return created, updated

    async def backfill_team_history(self, min_matches: int = 10,
                                    seasons: Optional[List[int]] = None,
                                    max_budget: int = 30,
                                    min_remaining_budget: int = 25,
                                    target_team_ids: Optional[set] = None):
        """Fetch historical match data for low-coverage teams in the database.

        When target_team_ids is provided, only those teams are considered
        (used by targeted backfill for today's fixtures). Otherwise scans
        all teams with upcoming fixtures within 14 days.

        Args:
            min_matches: Skip teams with at least this many completed matches.
            seasons: API-Football season years to fetch. Defaults to [2022, 2023, 2024].
            max_budget: Maximum API requests to spend on backfill.
            min_remaining_budget: Skip backfill if fewer than this many requests
                remain. Callers that already reserved budget may pass 0.
            target_team_ids: When set, only backfill these team IDs.

        Priority: teams with the fewest matches are processed first.
        Each team × season costs 1 API request.
        """
        if seasons is None:
            seasons = [2022, 2023, 2024]

        # Guard: skip backfill if the remaining daily budget is too low.
        remaining = self.remaining_budget()
        if remaining < min_remaining_budget:
            logger.warning(
                f"API-Football backfill skipped: only {remaining} requests remaining "
                f"(threshold={min_remaining_budget}). Run again tomorrow for full backfill."
            )
            return

        if target_team_ids is not None:
            # Targeted mode: only backfill specific teams (e.g. today's low-coverage)
            team_ids = target_team_ids
        else:
            # Default mode: scan all teams with upcoming fixtures (14-day window)
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
            # Override map for abbreviated DB names → full API-Football search terms
            _SEARCH_OVERRIDES = {
                "Oxford Utd": "Oxford United",
                "Atl. Madrid": "Atletico Madrid",
                "Ath Madrid": "Atletico Madrid",
                "Atl Madrid": "Atletico Madrid",
                "MK Dons": "Milton Keynes Dons",
                "Sheff Wed": "Sheffield Wednesday",
                "Sheffield Utd": "Sheffield United",
                "Nott'm Forest": "Nottingham Forest",
                "Man United": "Manchester United",
                "Man City": "Manchester City",
                "Sp Braga": "Sporting Braga",
                "West Brom": "West Bromwich",
                "Ein Frankfurt": "Eintracht Frankfurt",
                "M'gladbach": "Monchengladbach",
                "Ath Bilbao": "Athletic Bilbao",
                "Paris SG": "Paris Saint Germain",
            }
            for tid, name, cnt in still_missing[:10]:
                if self._requests_today >= self._daily_limit - self.BUDGET_RESERVE:
                    break
                # Use override if available, otherwise sanitize the DB name
                import re as _re
                import unicodedata as _ud
                search_name = _SEARCH_OVERRIDES.get(name)
                if not search_name:
                    # Transliterate accented chars to ASCII (ç→c, ğ→g, ı→i, etc.)
                    search_name = _ud.normalize("NFKD", name).encode("ascii", "ignore").decode()
                    # Keep only alphanumeric and spaces
                    search_name = _re.sub(r"[^a-zA-Z0-9\s]", "", search_name).strip()
                    # Collapse multiple spaces
                    search_name = _re.sub(r"\s+", " ", search_name)
                if len(search_name) < 3:
                    logger.debug(f"Search name too short after sanitizing '{name}' → '{search_name}'")
                    continue
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
