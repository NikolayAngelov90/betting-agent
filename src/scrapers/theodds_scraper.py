"""The Odds API scraper — free-tier supplemental odds source.

Fetches 1X2 and Over/Under odds for leagues that have today's fixtures
in the DB, using https://api.the-odds-api.com/v4.

Free tier: 500 credits/month (1 credit per league request).
"""

import asyncio
import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import aiohttp

from src.data.sql_helpers import id_in
from src.data.database import get_db
from src.data.models import Match, Odds, SavedPick, Team
from src.utils.logger import get_logger, utcnow
from src.data.price_history import record_price as _record_price
from src.scrapers.barren_leagues import (
    BARREN_CONSECUTIVE_THRESHOLD, BARREN_TTL_DAYS, BarrenLeagueCache)

logger = get_logger()

_CREDITS_STATE_PATH = Path("data/models/theodds_credits.json")
_CREDITS_LOW_THRESHOLD = 100   # existing WARNING in betting_agent.py

# Tiered in-scraper warnings (absolute credit counts, out of 500/month free tier)
_CREDITS_TIER_INFO = 60       # ~3 days of daily runs remaining
_CREDITS_TIER_WARNING = 40    # ~2 days remaining
_CREDITS_TIER_CRITICAL = 20   # ~1 day remaining
_CREDITS_GATE_THRESHOLD = 10  # hard skip — not enough for even one league call


def _load_persisted_credits() -> Optional[int]:
    """Return last-known TheOddsAPI remaining credits from state file, or None."""
    try:
        if _CREDITS_STATE_PATH.exists():
            return json.loads(_CREDITS_STATE_PATH.read_text()).get("remaining")
    except Exception:
        pass
    return None


def _persist_credits(remaining: int) -> None:
    """Save current remaining credit count so next run can warn early."""
    try:
        _CREDITS_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CREDITS_STATE_PATH.write_text(
            json.dumps({"remaining": remaining, "updated": date.today().isoformat()})
        )
    except Exception:
        pass

# ---------------------------------------------------------------------------
# League key → The Odds API sport key mapping
# Only leagues that The Odds API supports (romania/liga-1 is excluded).
# ---------------------------------------------------------------------------
LEAGUE_TO_THEODDS_SPORT: Dict[str, str] = {
    # Top 5
    "england/premier-league":          "soccer_epl",
    "spain/laliga":                    "soccer_spain_la_liga",
    "germany/bundesliga":              "soccer_germany_bundesliga",
    "italy/serie-a":                   "soccer_italy_serie_a",
    "france/ligue-1":                  "soccer_france_ligue_one",
    # Strong European
    "netherlands/eredivisie":          "soccer_netherlands_eredivisie",
    "portugal/primeira-liga":          "soccer_portugal_primeira_liga",
    "belgium/jupiler-pro-league":      "soccer_belgium_first_div",
    "turkey/super-lig":                "soccer_turkey_super_league",
    "scotland/premiership":            "soccer_spl",
    # European Competitions
    "europe/champions-league":         "soccer_uefa_champs_league",
    "europe/europa-league":            "soccer_uefa_europa_league",
    "europe/europa-conference-league": "soccer_uefa_europa_conference_league",
    # English lower divisions
    "england/championship":            "soccer_efl_champ",
    "england/league-one":              "soccer_england_league1",
    "england/league-two":              "soccer_england_league2",
    # Second divisions
    "spain/laliga2":                   "soccer_spain_segunda_division",
    "germany/2-bundesliga":            "soccer_germany_bundesliga2",
    "italy/serie-b":                   "soccer_italy_serie_b",
    "france/ligue-2":                  "soccer_france_ligue_two",
    "austria/bundesliga":              "soccer_austria_bundesliga",
    "switzerland/super-league":        "soccer_switzerland_superleague",
    "greece/super-league":             "soccer_greece_super_league",
    "denmark/superliga":               "soccer_denmark_superliga",
    # Nordic / Eastern
    "norway/eliteserien":              "soccer_norway_eliteserien",
    "sweden/allsvenskan":              "soccer_sweden_allsvenskan",
    "poland/ekstraklasa":              "soccer_poland_ekstraklasa",
    # romania/liga-1 intentionally omitted — not available on The Odds API
    # International — national team competitions
    "world/fifa-world-cup":             "soccer_fifa_world_cup",
}

# Bookmaker prefix so these odds rows are distinguishable from API-Football rows
BOOKMAKER_PREFIX = "TheOddsAPI"

BASE_URL = "https://api.the-odds-api.com/v4"


def _normalise(name: str) -> str:
    """Lowercase + strip for fast pre-filter before fuzzy matching."""
    return name.lower().strip()


def _team_names_similar(name_a: str, name_b: str) -> bool:
    """Reuse FlashscoreScraper's fuzzy match logic without importing the class.

    Handles abbreviations, diacritics, prefixes/suffixes, and known aliases.
    Returns True when two names likely refer to the same club.
    """
    import unicodedata
    from difflib import SequenceMatcher

    if name_a == name_b:
        return True

    _ALIASES: Dict[str, str] = {
        # England
        "man city": "manchester city",
        "man united": "manchester united",
        "man utd": "manchester united",
        "spurs": "tottenham hotspur",
        "tottenham": "tottenham hotspur",
        "wolves": "wolverhampton wanderers",
        "wolverhampton": "wolverhampton wanderers",
        "newcastle": "newcastle united",
        "west ham": "west ham united",
        "leicester": "leicester city",
        "brighton": "brighton & hove albion",
        "brighton hove albion": "brighton & hove albion",
        "sheffield utd": "sheffield united",
        "nottingham": "nottingham forest",
        "nott'm forest": "nottingham forest",
        "nott'm f": "nottingham forest",
        # Spain
        "ath bilbao": "athletic bilbao",
        "ath madrid": "atletico madrid",
        "atl. madrid": "atletico madrid",
        "atl madrid": "atletico madrid",
        "atletico de madrid": "atletico madrid",
        "real betis balompie": "real betis",
        "rcd espanyol": "espanyol",
        "deportivo alaves": "alaves",
        # Italy
        "inter milan": "inter",
        "internazionale": "inter",
        "internazionale milano": "inter",
        "ac milan": "milan",
        "juventus fc": "juventus",
        "as roma": "roma",
        "ss lazio": "lazio",
        "acf fiorentina": "fiorentina",
        "hellas verona": "verona",
        # Germany
        "borussia m'gladbach": "borussia monchengladbach",
        "m'gladbach": "monchengladbach",
        "bayer leverkusen": "leverkusen",
        "hertha bsc": "hertha",
        "vfb stuttgart": "stuttgart",
        "tsg hoffenheim": "hoffenheim",
        "sc freiburg": "freiburg",
        "rb leipzig": "leipzig",
        "eintracht frankfurt": "frankfurt",
        "1. fc koln": "koln",
        "1. fc union berlin": "union berlin",
        "sv werder bremen": "werder bremen",
        # France
        "paris sg": "paris saint-germain",
        "psg": "paris saint-germain",
        "paris saint germain": "paris saint-germain",
        "olympique de marseille": "marseille",
        "olympique lyonnais": "lyon",
        "as monaco fc": "monaco",
        "ogc nice": "nice",
        "stade rennais": "rennes",
        "stade brestois": "brest",
        "stade de reims": "reims",
        # Netherlands
        "psv eindhoven": "psv",
        "ajax amsterdam": "ajax",
        "az alkmaar": "az",
        "feyenoord rotterdam": "feyenoord",
        "fc twente": "twente",
        "sc heerenveen": "heerenveen",
        # Portugal
        "sporting cp": "sporting",
        "sporting clube de portugal": "sporting",
        "sl benfica": "benfica",
        "fc porto": "porto",
        "sc braga": "braga",
        # Turkey
        "galatasaray sk": "galatasaray",
        "fenerbahce sk": "fenerbahce",
        "besiktas jk": "besiktas",
        "trabzonspor as": "trabzonspor",
        # Greece
        "olympiakos": "olympiacos piraeus",
        "olympiacos": "olympiacos piraeus",
        "olympiakos piraeus": "olympiacos piraeus",
        "paok salonika": "paok",
        "paok thessaloniki": "paok",
        # Romania
        "din. bucuresti": "dinamo bucuresti",
        "din bucuresti": "dinamo bucuresti",
        "fcsb": "steaua bucuresti",
        "fc steaua bucuresti": "steaua bucuresti",
        # European cups (common TheOdds API variants)
        "rb salzburg": "salzburg",
        "red bull salzburg": "salzburg",
        "shakhtar donetsk": "shakhtar",
        "bsc young boys": "young boys",
        "fc celtic": "celtic",
        "celtic fc": "celtic",
        "rangers fc": "rangers",
        "sporting lisbon": "sporting",
        "benfica": "benfica",
        "porto": "porto",
        # Belgium — Jupiler Pro League
        "union saint-gilloise": "st. gilloise",
        "royale union saint-gilloise": "st. gilloise",
        "union sg": "st. gilloise",
        # Standard Liège: TheOddsAPI sends "Standard Liège" → canon "standard liege";
        # DB may store bare "Standard" → canon "standard". Token match {"standard","liege"}
        # vs {"standard"} = 0.5 < 0.7 threshold → no match. Alias collapses all forms.
        "standard liege": "standard",
        "r. standard de liege": "standard",
        "r standard de liege": "standard",
        "standard de liege": "standard",
        # Further Belgium aliases for completeness
        "rsc anderlecht": "anderlecht",
        "royale union sportive anderlecht": "anderlecht",
        "club brugge kv": "club brugge",
        "krc genk": "genk",
        "kaa gent": "gent",
        "oh leuven": "oud-heverlee leuven",
        # Poland — Ekstraklasa (TheOdds API uses short name; DB stores API-Football display name)
        "nieciecza": "termalica b-b.",
        "bruk-bet termalica nieciecza": "termalica b-b.",
        # Portugal — TheOddsAPI uses "Vitória SC" for Vitória SC Guimarães; DB stores "Guimaraes"
        "vitoria sc": "guimaraes",
        # Greece — DB stores short names; TheOddsAPI sends city-qualified names
        "ofi crete": "ofi",
        "ofi crete fc": "ofi",
        "panserraikos fc": "panserraikos",
        "panserraikos fc thessaloniki": "panserraikos",
        # Netherlands — DB stores "PSV Eindhoven"; TheOdds may send bare "PSV"
        "psv": "psv eindhoven",
    }

    def _strip_accents(s: str) -> str:
        return "".join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def _canon(n: str) -> str:
        n = _strip_accents(n.lower().strip())
        n = _ALIASES.get(n, n)
        for suffix in (" fc", " cf", " sc", " ac", " fk", " sk", " bk",
                       " city", " town", " united", " utd", " rovers",
                       " wanderers", " athletic", " albion", " hotspur"):
            if n.endswith(suffix):
                n = n[: -len(suffix)].strip()
                break
        for prefix in ("fc ", "cf ", "sc ", "ac ", "fk ", "sk ", "bk ",
                       "as ", "ss ", "us ", "rc ", "cd ", "ud ", "sd "):
            if n.startswith(prefix):
                n = n[len(prefix):].strip()
                break
        return n

    c_a, c_b = _canon(name_a), _canon(name_b)
    if c_a == c_b:
        return True
    if SequenceMatcher(None, c_a, c_b).ratio() >= 0.75:
        return True

    # Token-based partial match
    t_a, t_b = set(c_a.split()), set(c_b.split())
    shorter, longer = (t_a, t_b) if len(t_a) <= len(t_b) else (t_b, t_a)
    if not shorter:
        return False

    def _tok_match(ta: str, tb: str) -> bool:
        if ta == tb:
            return True
        if ta in tb or tb in ta:
            return True
        return SequenceMatcher(None, ta, tb).ratio() >= 0.75

    matches = sum(
        1 for tok in shorter
        if any(_tok_match(tok, long_tok) for long_tok in longer)
    )
    return matches / len(shorter) >= 0.7


class TheOddsScraper:
    """Fetches odds from The Odds API for leagues with today's fixtures only.

    Budget: 1 API credit per league call. With ~28 supported leagues and 500
    monthly credits, this comfortably covers daily runs even if all leagues
    have fixtures simultaneously.
    """

    def __init__(self, config=None):
        from src.utils.config import get_config
        self.config = config or get_config()
        self.api_key: str = os.environ.get("ODDS_API_KEY", "")
        self.db = get_db()
        self._remaining_requests: Optional[int] = None
        self._used_requests: Optional[int] = None
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _leagues_with_today_fixtures(self) -> List[str]:
        """Return internal league keys that have at least one fixture today."""
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = today_start + timedelta(days=1)

        leagues_with_fixtures: Set[str] = set()
        with self.db.get_session() as session:
            rows = (
                session.query(Match.league)
                .filter(
                    Match.is_fixture == True,
                    Match.match_date >= today_start,
                    Match.match_date < today_end,
                )
                .distinct()
                .all()
            )
            leagues_with_fixtures = {r.league for r in rows}

        # Only keep leagues that map to The Odds API sport keys
        supported = [
            league
            for league in leagues_with_fixtures
            if league in LEAGUE_TO_THEODDS_SPORT
        ]
        unsupported = sorted(leagues_with_fixtures - set(supported))
        if unsupported:
            logger.info(
                f"TheOddsAPI: {len(supported)} leagues with today's fixtures "
                f"(skipping {len(unsupported)} unsupported: {', '.join(unsupported)})"
            )
        else:
            logger.info(
                f"TheOddsAPI: {len(supported)} leagues with today's fixtures"
            )
        return supported

    def _get_today_fixtures(self, league: str) -> List[Dict]:
        """Return today's DB fixtures for a league with team names."""
        today_start = datetime.combine(date.today(), datetime.min.time())
        today_end = today_start + timedelta(days=1)

        fixtures = []
        with self.db.get_session() as session:
            matches = (
                session.query(Match)
                .filter(
                    Match.league == league,
                    Match.is_fixture == True,
                    Match.match_date >= today_start,
                    Match.match_date < today_end,
                )
                .all()
            )
            for m in matches:
                home_team = session.get(Team, m.home_team_id)
                away_team = session.get(Team, m.away_team_id)
                if home_team and away_team:
                    fixtures.append({
                        "match_id": m.id,
                        "home_name": home_team.name,
                        "away_name": away_team.name,
                        "match_date": m.match_date,
                    })
        return fixtures

    async def _fetch_league_odds(self, sport_key: str) -> Optional[List[Dict]]:
        """Call The Odds API for a single sport key.

        Returns the raw list of game objects or None on error.
        Tracks remaining credits from response headers.
        """
        if not self.api_key:
            logger.warning("TheOddsAPI: ODDS_API_KEY not set — skipping")
            return None

        url = f"{BASE_URL}/sports/{sport_key}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        }

        session = await self._get_session()
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 401:
                    logger.error("TheOddsAPI: invalid API key (401)")
                    return None
                if resp.status == 422:
                    logger.debug(f"TheOddsAPI: sport key '{sport_key}' not found (422)")
                    return None
                if resp.status == 429:
                    logger.warning("TheOddsAPI: quota exhausted (429)")
                    return None
                resp.raise_for_status()

                # Track quota from response headers
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    self._remaining_requests = int(remaining)
                    _persist_credits(self._remaining_requests)
                    _r = self._remaining_requests
                    if _r <= _CREDITS_TIER_CRITICAL:
                        logger.critical(
                            f"TheOddsAPI CRITICAL: only {_r} credits remaining "
                            f"— suspend non-essential league calls or add credits"
                        )
                    elif _r <= _CREDITS_TIER_WARNING:
                        logger.warning(
                            f"TheOddsAPI WARNING: {_r} credits remaining (~2 days left)"
                        )
                    elif _r <= _CREDITS_TIER_INFO:
                        logger.info(
                            f"TheOddsAPI: {_r} credits remaining (~3 days left)"
                        )
                if used is not None:
                    self._used_requests = int(used)

                data = await resp.json()
                logger.debug(
                    f"TheOddsAPI: fetched {len(data)} games for '{sport_key}' "
                    f"(credits remaining: {self._remaining_requests})"
                )
                return data
        except aiohttp.ClientError as e:
            logger.warning(f"TheOddsAPI: request failed for '{sport_key}': {e}")
            return None

    def _find_matching_fixture(
        self, api_home: str, api_away: str, db_fixtures: List[Dict]
    ) -> Optional[int]:
        """Return match_id of the DB fixture that matches the API game's team names."""
        for fix in db_fixtures:
            if (_team_names_similar(api_home, fix["home_name"]) and
                    _team_names_similar(api_away, fix["away_name"])):
                return fix["match_id"]
        return None

    def _save_game_odds(self, game: Dict, match_id: int, saved: List[str],
                        session=None, existing_index: Optional[dict] = None) -> int:
        """Parse one API game object and upsert odds rows into DB.

        When `session` is provided the caller owns the transaction and is
        responsible for committing.  When None (legacy), opens its own session
        and commits once per game (not once per row as before).

        `existing_index` (keyed by (match_id, bm, market, selection)) is passed
        to _upsert_odds to avoid per-row SELECT queries when provided.

        Returns count of odds rows written.
        """
        written = 0
        bookmakers = game.get("bookmakers", [])
        if not bookmakers:
            return 0

        def _process(sess):
            nonlocal written
            for bm in bookmakers:
                bm_key = f"{BOOKMAKER_PREFIX}-{bm['key']}"
                for market in bm.get("markets", []):
                    market_key = market["key"]  # "h2h" or "totals"

                    if market_key == "h2h":
                        home_name = game.get("home_team", "")
                        away_name = game.get("away_team", "")
                        for outcome in market.get("outcomes", []):
                            oname = outcome.get("name", "")
                            oprice = outcome.get("price")
                            if oprice is None:
                                continue
                            if _team_names_similar(oname, home_name):
                                selection = "Home"
                            elif oname.lower() == "draw":
                                selection = "Draw"
                            elif _team_names_similar(oname, away_name):
                                selection = "Away"
                            else:
                                continue
                            written += self._upsert_odds(
                                sess, match_id, bm_key, "1X2", selection, oprice,
                                existing_index=existing_index,
                            )

                    elif market_key == "totals":
                        for outcome in market.get("outcomes", []):
                            direction = outcome.get("name", "")
                            point = outcome.get("point")
                            oprice = outcome.get("price")
                            if direction not in ("Over", "Under") or point is None or oprice is None:
                                continue
                            selection = f"{direction} {point}"
                            written += self._upsert_odds(
                                sess, match_id, bm_key, "over_under", selection, oprice,
                                existing_index=existing_index,
                            )

        if session is not None:
            _process(session)
        else:
            # Legacy path: own session, single commit per game (not per row)
            with self.db.get_session() as sess:
                try:
                    _process(sess)
                    sess.commit()
                except Exception as e:
                    sess.rollback()
                    logger.debug(f"TheOddsAPI game save failed for match_id={match_id}: {e}")
                    return 0

        return written

    def _upsert_odds(
        self,
        session,
        match_id: int,
        bookmaker: str,
        market_type: str,
        selection: str,
        odds_value: float,
        existing_index: Optional[dict] = None,
    ) -> int:
        """Insert-or-update a single Odds row. Returns 1 if written, 0 otherwise.

        When `existing_index` is provided (keyed by (match_id, bookmaker,
        market_type, selection)) the existence check is done in-memory instead
        of firing a SELECT per row — reduces Neon roundtrips from N_rows to 0.

        Does NOT commit — caller is responsible for a single commit per game/league.
        """
        try:
            key = (match_id, bookmaker, market_type, selection)
            if existing_index is not None:
                existing = existing_index.get(key)
            else:
                existing = (
                    session.query(Odds)
                    .filter_by(
                        match_id=match_id,
                        bookmaker=bookmaker,
                        market_type=market_type,
                        selection=selection,
                    )
                    .first()
                )
            _now = datetime.now(timezone.utc).replace(tzinfo=None)
            if existing:
                existing.odds_value = odds_value
                existing.timestamp = _now
            else:
                new_row = Odds(
                    match_id=match_id,
                    bookmaker=bookmaker,
                    market_type=market_type,
                    selection=selection,
                    odds_value=odds_value,
                    opening_odds=odds_value,
                    first_seen_at=_now,
                )
                session.add(new_row)
                if existing_index is not None:
                    existing_index[key] = new_row  # prevent re-insert on dup

            # Stage 18 C1: append the observation to the price PATH. Every
            # write, insert or update — an update is exactly the observation
            # that overwrite semantics used to destroy. Fails open.
            _record_price(session, match_id=match_id, bookmaker=bookmaker,
                          market_type=market_type, selection=selection,
                          odds_value=odds_value, observed_at=_now)
            return 1
        except Exception as e:
            logger.debug(f"TheOddsAPI upsert failed ({bookmaker} / {market_type} / {selection}): {e}")
            return 0

    async def update(self) -> int:
        """Fetch odds for all leagues with today's fixtures.

        Performance optimisations vs the old sequential approach:
        - All HTTP requests are fired concurrently (asyncio.gather).
        - DB writes use one session per league with a single commit, instead
          of one session and one commit per odds row (the old hot path had
          90+ commits per game × 14 leagues = ~1,000+ Neon roundtrips).

        Returns total count of odds rows written.
        """
        if not self.api_key:
            logger.info("TheOddsAPI: ODDS_API_KEY not configured — skipping")
            return 0

        # Hard gate: if last-known credits are below the gate threshold, skip
        # entirely rather than burning the last few credits and going silent.
        _persisted = _load_persisted_credits()
        if _persisted is not None and _persisted <= _CREDITS_GATE_THRESHOLD:
            logger.warning(
                f"TheOddsAPI: skipping update — only {_persisted} credits remain "
                f"(gate threshold: {_CREDITS_GATE_THRESHOLD}). "
                f"Add credits or wait for monthly reset."
            )
            return 0

        leagues = self._leagues_with_today_fixtures()
        if not leagues:
            logger.info("TheOddsAPI: no leagues with today's fixtures — skipping")
            return 0

        # Gather DB fixtures for all leagues up front (sequential — fast, no HTTP)
        league_fixtures: Dict[str, List[Dict]] = {}
        for league in leagues:
            fixtures = self._get_today_fixtures(league)
            if fixtures:
                league_fixtures[league] = fixtures
            else:
                logger.debug(f"TheOddsAPI: no DB fixtures for '{league}', skipping API call")

        if not league_fixtures:
            return 0

        return await self._fetch_and_persist(league_fixtures)

    async def _fetch_and_persist(self, league_fixtures: Dict[str, List[Dict]],
                                 quota=None) -> int:
        """Fetch odds for the given leagues and persist them.

        Extracted from ``update()`` so the Stage 6 imminent-fixture refresh
        can reuse the identical fetch/match/persist path. Splitting it was the
        alternative to duplicating ~140 lines of team-name matching and upsert
        logic, which would have drifted.

        Args:
            league_fixtures: internal league key -> DB fixtures to match against.
            quota: optional OddsApiQuota. When supplied, budget for exactly
                len(league_fixtures) requests is CLAIMED BEFORE any HTTP call,
                and the league set is truncated to what the budget grants.
        """
        if not league_fixtures:
            return 0

        if quota is not None:
            wanted = list(league_fixtures)
            granted = quota.claim_requests(len(wanted))
            if granted <= 0:
                logger.warning(
                    "TheOddsAPI: monthly credit budget exhausted — making NO "
                    "requests this run. " + quota.describe())
                return 0
            if granted < len(wanted):
                dropped = wanted[granted:]
                league_fixtures = {l: league_fixtures[l] for l in wanted[:granted]}
                logger.warning(
                    f"TheOddsAPI: budget allows {granted}/{len(wanted)} league "
                    f"request(s); skipping {dropped} this run")

        # Fire all HTTP requests concurrently — reduces wall time from
        # 14× single-request latency to ~1× single-request latency.
        sport_keys = [LEAGUE_TO_THEODDS_SPORT[l] for l in league_fixtures]
        raw_results = await asyncio.gather(
            *[self._fetch_league_odds(sk) for sk in sport_keys],
            return_exceptions=True,
        )
        league_games: Dict[str, Optional[List[Dict]]] = {}
        for league, result in zip(league_fixtures, raw_results):
            if isinstance(result, Exception):
                logger.warning(f"TheOddsAPI: fetch failed for '{league}': {result}")
                league_games[league] = None
            else:
                league_games[league] = result

        # Per-league outcome, recorded because the aggregate hid it. The
        # attribution log in refresh_imminent used to derive `result=` from the
        # TOTAL rows written across the whole batch, so a batch where one league
        # returned rows and three returned nothing emitted four `result=ok`
        # lines. Every no_rows figure ever read out of those logs is therefore a
        # LOWER bound on the waste, including Stage 15's.
        self._last_league_outcomes = {
            lg: ("error" if games is None else ("no_rows" if not games else "ok"))
            for lg, games in league_games.items()
        }

        total_written = 0
        matched_games = 0
        unmatched_games = 0
        unmatched_details: List[str] = []

        # The window used to decide whether an API game is one we care about.
        # Derived from the fixtures the caller passed in rather than hard-coded
        # to "today": the Stage 6 imminent refresh can run at 23:00 UTC and
        # legitimately want a 00:30 kickoff, which a today-only window drops.
        # For update(), whose fixtures are all today, this is the same window.
        _fixture_kos = [
            f["match_date"] for fixes in league_fixtures.values() for f in fixes
            if f.get("match_date")
        ]
        if _fixture_kos:
            window_start = min(_fixture_kos) - timedelta(hours=1)
            window_end = max(_fixture_kos) + timedelta(hours=1)
        else:
            window_start = datetime.combine(date.today(), datetime.min.time())
            window_end = window_start + timedelta(days=1)

        for league, games in league_games.items():
            if not games:
                continue

            db_fixtures = league_fixtures[league]
            match_ids = [fix["match_id"] for fix in db_fixtures]
            league_unmatched = 0
            saved_match_ids: List[str] = []

            # One DB session for the entire league — single commit at the end
            with self.db.get_session() as session:
                try:
                    # Preload ALL existing odds for this league's matches in one
                    # query instead of one SELECT per row — reduces Neon roundtrips
                    # from N_odds_rows to 1 per league.
                    existing_rows = session.query(Odds).filter(
                        id_in(session, Odds.match_id, match_ids)
                    ).all()
                    existing_index = {
                        (r.match_id, r.bookmaker, r.market_type, r.selection): r
                        for r in existing_rows
                    }

                    for game in games:
                        api_home = game.get("home_team", "")
                        api_away = game.get("away_team", "")

                        commence_raw = game.get("commence_time", "")
                        is_today = False
                        if commence_raw:
                            try:
                                from datetime import timezone as _tz
                                ct = datetime.fromisoformat(commence_raw.replace("Z", "+00:00"))
                                ct_naive = ct.astimezone(_tz.utc).replace(tzinfo=None)
                                is_today = window_start <= ct_naive < window_end
                            except Exception:
                                is_today = True

                        match_id = self._find_matching_fixture(api_home, api_away, db_fixtures)

                        if match_id is None:
                            if is_today:
                                unmatched_games += 1
                                league_unmatched += 1
                                unmatched_details.append(
                                    f"  [{league}] '{api_home}' vs '{api_away}'"
                                )
                            continue

                        matched_games += 1
                        n = self._save_game_odds(
                            game, match_id, saved_match_ids,
                            session=session, existing_index=existing_index,
                        )
                        total_written += n
                        logger.debug(
                            f"TheOddsAPI: {api_home} vs {api_away} → match_id={match_id}, "
                            f"{n} odds rows staged"
                        )

                    session.commit()
                except Exception as e:
                    session.rollback()
                    logger.warning(f"TheOddsAPI: DB write failed for league '{league}': {e}")

            if league_unmatched:
                logger.debug(f"TheOddsAPI: {league_unmatched} unmatched in '{league}'")

            # Cooperative yield — lets asyncio.wait_for(timeout=300) fire between
            # leagues rather than being blocked by the synchronous DB write loop.
            await asyncio.sleep(0)

        logger.info(
            f"TheOddsAPI update complete: {total_written} odds rows written, "
            f"{matched_games} games matched, {unmatched_games} unmatched "
            f"(credits remaining: {self._remaining_requests})"
        )
        if unmatched_details:
            logger.warning(
                f"TheOddsAPI: {unmatched_games} games could not be matched to DB fixtures "
                f"(team name mismatch) — add aliases to fix:\n" + "\n".join(unmatched_details)
            )
        return total_written

    # ------------------------------------------------------------------------
    # Stage 6 — imminent-fixture refresh
    # ------------------------------------------------------------------------

    def _imminent_league_fixtures(self, window_minutes: int, now=None,
                                  require_pending_pick: bool = True):
        """Leagues whose fixtures kick off inside the window, most urgent first.

        Returns ``(league_fixtures, skip_reasons)``.

        A league is worth a credit only when refreshing it can actually produce
        a closing line. Measured on production history: ~3.6 mapped leagues per
        day carry a pending pick, against ~7.2 with any fixture and 27
        configured. That gap is the entire Stage 6 saving.

        Egress: two projected queries over the handful of fixtures in the
        window, using the existing ``ix_match_fixture_date`` index — not
        ``SELECT matches.*`` across a whole day, and no per-fixture team lookup.
        """
        now = now or utcnow()
        horizon = now + timedelta(minutes=window_minutes)
        skips = {}

        with self.db.get_session() as session:
            rows = (
                session.query(
                    Match.id, Match.league, Match.match_date,
                    Team.name.label("home_name"),
                )
                .join(Team, Team.id == Match.home_team_id)
                .filter(
                    Match.is_fixture == True,  # noqa: E712
                    Match.match_date > now,
                    Match.match_date <= horizon,
                )
                .all()
            )
            if not rows:
                return {}, skips

            match_ids = [r.id for r in rows]
            away = dict(
                session.query(Match.id, Team.name)
                .join(Team, Team.id == Match.away_team_id)
                .filter(id_in(session, Match.id, match_ids))
                .all()
            )
            pending = {
                mid for (mid,) in session.query(SavedPick.match_id).filter(
                    id_in(session, SavedPick.match_id, match_ids),
                    SavedPick.closing_odds.is_(None),
                    SavedPick.closing_capture_status == "pending",
                ).distinct().all()
            }

        by_league = {}
        for r in rows:
            league = r.league or ""
            if league not in LEAGUE_TO_THEODDS_SPORT:
                skips.setdefault(league, "not mapped to a The Odds API sport key")
                continue
            if require_pending_pick and r.id not in pending:
                skips.setdefault(league, "no pending pick awaiting a closing line")
                continue
            by_league.setdefault(league, []).append({
                "match_id": r.id,
                "home_name": r.home_name,
                "away_name": away.get(r.id, ""),
                "match_date": r.match_date,
            })

        # A league that qualified must not also be reported as skipped.
        for league in by_league:
            skips.pop(league, None)

        # Soonest kickoff first, so a truncated budget keeps the urgent ones.
        ordered = sorted(by_league.items(),
                         key=lambda kv: min(f["match_date"] for f in kv[1]))
        return dict(ordered), skips

    def _leagues_refreshed_since(self, leagues, since):
        """Leagues whose Odds-API rows already carry a timestamp >= ``since``.

        Deduplication is derived from the odds table, not a state file: it is
        authoritative, survives loss of the GitHub Actions cache, and cannot
        disagree with what was actually stored. One projected aggregate query.
        """
        if not leagues:
            return set()
        from sqlalchemy import func as _func

        with self.db.get_session() as session:
            rows = (
                session.query(Match.league, _func.max(Odds.timestamp))
                .join(Odds, Odds.match_id == Match.id)
                .filter(
                    id_in(session, Match.league, list(leagues)),
                    Odds.bookmaker.like(BOOKMAKER_PREFIX + "%"),
                    Odds.timestamp >= since,
                )
                .group_by(Match.league)
                .all()
            )
        return {lg for lg, ts in rows if ts is not None}

    async def refresh_imminent(self, window_minutes: int = 120,
                               min_interval_minutes: int = 180,
                               quota=None,
                               require_pending_pick: bool = True,
                               dry_run: bool = False,
                               now=None) -> dict:
        """Refresh odds only for leagues with imminent fixtures worth a credit.

        This never widens what counts as a valid closing line — it only changes
        WHICH odds are fresh enough for ``capture_closing_lines`` to accept.

        ``now`` is injectable so the selection and dedup windows can be tested
        deterministically instead of against the wall clock.
        """
        from src.data.odds_quota import CREDITS_PER_REQUEST, credits_for

        now = now or utcnow()
        plan = {
            "now": now.isoformat(),
            "window_minutes": window_minutes,
            "min_interval_minutes": min_interval_minutes,
            "candidates": [], "requested": [], "skipped": {},
            "credits_estimated": 0, "credits_claimed": 0,
            "odds_written": 0, "dry_run": dry_run,
        }
        if not self.api_key:
            plan["skipped"]["*"] = "ODDS_API_KEY not configured"
            logger.info("TheOddsAPI: ODDS_API_KEY not configured — skipping")
            return plan

        league_fixtures, skips = self._imminent_league_fixtures(
            window_minutes, now=now, require_pending_pick=require_pending_pick)
        plan["skipped"].update(skips)
        plan["candidates"] = [
            {"league": lg, "fixtures": len(fx),
             "next_kickoff": min(x["match_date"] for x in fx).isoformat()}
            for lg, fx in league_fixtures.items()
        ]

        # A league refreshed inside the interval is already fresh enough;
        # spending again buys nothing.
        if league_fixtures and min_interval_minutes > 0:
            since = now - timedelta(minutes=min_interval_minutes)
            recent = self._leagues_refreshed_since(list(league_fixtures), since)
            for league in list(league_fixtures):
                if league in recent:
                    plan["skipped"][league] = (
                        f"refreshed within the last {min_interval_minutes} min")
                    league_fixtures.pop(league)

        # L2: a league that returned an empty event list three runs running is
        # not priced by the provider. Requesting it again cannot produce an
        # observation, so declining costs no coverage. The exclusion expires.
        barren = BarrenLeagueCache()
        if league_fixtures:
            for league in list(league_fixtures):
                if barren.should_skip(league):
                    plan["skipped"][league] = (
                        "no odds returned on the last "
                        f"{BARREN_CONSECUTIVE_THRESHOLD} attempts "
                        f"(retried after {BARREN_TTL_DAYS}d)")
                    league_fixtures.pop(league)

        plan["requested"] = list(league_fixtures)
        plan["credits_estimated"] = credits_for(len(league_fixtures))

        if not league_fixtures:
            logger.info(
                f"TheOddsAPI imminent refresh: nothing to do (window "
                f"{window_minutes} min) — {len(plan['skipped'])} league(s) skipped")
            return plan

        if dry_run:
            logger.info(
                f"TheOddsAPI imminent refresh [DRY RUN]: would request "
                f"{len(league_fixtures)} league(s) "
                f"({plan['credits_estimated']} credits): {list(league_fixtures)}")
            return plan

        before = getattr(quota, "spent_this_run", 0) if quota else 0
        written = await self._fetch_and_persist(league_fixtures, quota=quota)
        after = getattr(quota, "spent_this_run", 0) if quota else 0
        plan["odds_written"] = written
        outcomes = getattr(self, "_last_league_outcomes", {})
        for league, outcome in outcomes.items():
            # An exception is not evidence the league is unpriced — a timeout
            # says nothing about the provider's catalogue. Only a clean empty
            # response counts toward exclusion.
            if outcome in ("ok", "no_rows"):
                barren.record(league, empty=(outcome == "no_rows"))
        barren.save()
        plan["barren"] = barren.describe()
        plan["credits_claimed"] = (after - before) if quota else plan["credits_estimated"]
        plan["credits_remaining_header"] = self._remaining_requests

        # Reconcile against the provider's own counter. The ledger only knows
        # about spend that went through it; the API counts everything on the
        # key. Measured 2026-08-10: provider 95 used, ledger 0. Doing this after
        # the requests rather than before costs one cycle of staleness and saves
        # a probe call — the response headers we already have carry the truth.
        if quota is not None and self._used_requests is not None:
            plan["credits_used_provider"] = self._used_requests
            plan["credits_used_ledger"] = quota.reconcile(self._used_requests)

        # Per-league attribution (section 7E): every refresh decision is
        # traceable to a date, league, estimated cost, outcome and reason.
        # Emitted as one structured line per league so a CI log can be grepped
        # into a ledger without parsing prose.
        for league in plan["requested"]:
            logger.info(
                f"ODDS_REFRESH date={now:%Y-%m-%d} time={now:%H:%M} "
                f"league={league} sport_key={LEAGUE_TO_THEODDS_SPORT.get(league)} "
                f"requests=1 est_credits={CREDITS_PER_REQUEST} "
                f"result={outcomes.get(league, 'ok' if written else 'no_rows')} "
                f"reason=imminent_pending_pick")
        for league, why in plan["skipped"].items():
            logger.info(
                f"ODDS_REFRESH date={now:%Y-%m-%d} time={now:%H:%M} "
                f"league={league or '-'} requests=0 est_credits=0 "
                f"result=skipped reason={why.replace(' ', '_')}")

        logger.info(
            f"TheOddsAPI imminent refresh: {written} odds rows from "
            f"{len(league_fixtures)} league(s), {plan['credits_claimed']} credits "
            f"claimed (header remaining: {self._remaining_requests})")
        return plan
