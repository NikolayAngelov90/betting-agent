"""Flashscore scraper for match results, fixtures, and statistics.

Selector reference (current Flashscore layout as of 2025-2026):
  Listing page  (flashscore.com/football/{league}/results/):
    Match rows:   .event__match.event__match--static.event__match--twoLine
    Match URL:    a.eventRowLink  (href = match detail URL)
    Team names:   .event__participant--home / .event__participant--away
    Scores:       .event__score  (index 0=home, index 1=away)
    Date/time:    .event__time
    Load more:    [data-testid="wcl-buttonLink"]

  Match detail page  (flashscore.com/match/{id}/#/match-summary):
    Home team:    .duelParticipant__home .participant__participantName
    Away team:    .duelParticipant__away .participant__participantName
    Start time:   .duelParticipant__startTime
    Scores:       .detailScore__wrapper span:not(.detailScore__divider)
    Reg. time:    .detailScore__fullTime
    Penalties:    [data-testid="wcl-scores-overline-02"] with text "penalties"

  Match info box  (referee, venue, capacity):
    Container:    div[data-testid='wcl-summaryMatchInformation'] > div
    Key/value:    even-indexed divs = labels, odd-indexed divs = values

  Statistics sub-page  (match detail URL + /summary/stats/0/):
    Stat row:     div[data-testid='wcl-statistics']
    Stat name:    div[data-testid='wcl-statistics-category']
    Home value:   div[data-testid='wcl-statistics-value'] > strong  (index 0)
    Away value:   div[data-testid='wcl-statistics-value'] > strong  (index 1)

  Useful statistics for betting:
    Ball Possession, Total Shots, Shots on Target, Blocked Shots,
    Corner Kicks, Yellow Cards, Red Cards, Free Kicks, Offsides,
    Goalkeeper Saves, Dangerous Attacks, Expected Goals (xG)

Note: Flashscore uses Cloudflare anti-bot. Selenium is often blocked.
      The scraper is kept as a best-effort fallback; primary data comes
      from API-Football (apifootball_scraper.py).
"""

from datetime import datetime, timedelta
from typing import List, Optional

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import asyncio
import concurrent.futures

from src.scrapers.base_scraper import BaseScraper
from src.data.models import Team, Match
from src.data.database import get_db
from src.utils.logger import get_logger

logger = get_logger()

FLASHSCORE_BASE_URL = "https://www.flashscore.com"

# Statistics that are useful for our betting models, mapped from the
# Flashscore label (lowercased) to Match model column names.
STAT_MAP = {
    # Possession & shots
    "ball possession":        ("home_possession",        "away_possession"),
    "total shots":            ("home_shots",             "away_shots"),
    "shots on target":        ("home_shots_on_target",   "away_shots_on_target"),
    "shots on goal":          ("home_shots_on_target",   "away_shots_on_target"),
    "corner kicks":           ("home_corners",           "away_corners"),
    "corners":                ("home_corners",           "away_corners"),
    "fouls":                  ("home_fouls",             "away_fouls"),
    "yellow cards":           ("home_yellow_cards",      "away_yellow_cards"),
    "red cards":              ("home_red_cards",         "away_red_cards"),
    # Extended stats (added 2025)
    "goalkeeper saves":       ("home_saves",             "away_saves"),
    "saves":                  ("home_saves",             "away_saves"),
    "offsides":               ("home_offsides",          "away_offsides"),
    "free kicks":             ("home_free_kicks",        "away_free_kicks"),
    "dangerous attacks":      ("home_dangerous_attacks", "away_dangerous_attacks"),
    "expected goals":         ("home_xg",                "away_xg"),
    "xg":                     ("home_xg",                "away_xg"),
    "expected goals (xg)":    ("home_xg",                "away_xg"),
}


class FlashscoreScraper(BaseScraper):
    """Scrapes match data, fixtures, results, and statistics from Flashscore."""

    def __init__(self, config=None):
        super().__init__(config)
        self.leagues = self.config.get("scraping.flashscore_leagues", [])
        self.headless = self.config.get("scraping.headless", True)
        self._driver = None
        self._chrome_failed = False

    def _get_driver(self):
        """Create a Selenium Chrome driver. Returns None if Chrome unavailable."""
        if self._chrome_failed:
            return None
        if self._driver is None:
            try:
                options = Options()
                if self.headless:
                    options.add_argument("--headless=new")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-extensions")
                options.add_argument(
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(webdriver.Chrome, options=options)
                    self._driver = future.result(timeout=15)
                self._driver.implicitly_wait(10)
                self._driver.set_page_load_timeout(30)
            except Exception as e:
                logger.warning(f"Chrome/Selenium not available: {e}")
                self._chrome_failed = True
                self._driver = None
        return self._driver

    def close_driver(self):
        """Close the Selenium driver."""
        if self._driver:
            self._driver.quit()
            self._driver = None

    async def update(self):
        """Run full update: results + fixtures for all configured leagues."""
        logger.info(f"Starting Flashscore update cycle for {len(self.leagues)} leagues")
        for league in self.leagues:
            try:
                logger.info(f"[Flashscore] Scraping: {league}")
                await asyncio.wait_for(self.scrape_league_results(league), timeout=60)
                await asyncio.wait_for(self.scrape_league_fixtures(league), timeout=60)
                await self._rate_limit()
            except asyncio.TimeoutError:
                logger.warning(f"[Flashscore] Timeout on {league}, skipping to next")
            except Exception as e:
                logger.error(f"[Flashscore] Error scraping {league}: {e}")
        logger.info("Flashscore update cycle complete")

    async def scrape_league_results(self, league: str, num_pages: int = 1):
        """Scrape recent match results for a league.

        After saving basic scores, fetches extended statistics (shots, possession,
        dangerous attacks, saves, offsides, referee, etc.) for recently completed
        matches that are missing that data.  Limited to last 7 days and 10 matches
        per call to avoid overloading Flashscore / triggering Cloudflare.
        """
        url = f"{FLASHSCORE_BASE_URL}/football/{league}/results/"
        logger.info(f"Scraping results: {league}")

        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(None, self._scrape_results_page, url)

        db = get_db()
        # Save basic scores and collect (match_id, match_url, match_date) for
        # matches that have a detail URL so we can enrich them below.
        saved_with_urls = []
        with db.get_session() as session:
            for match_data in matches:
                saved = self._save_match(session, match_data, league, is_fixture=False)
                if saved and match_data.get("match_url"):
                    session.flush()  # ensure PK is assigned before we exit
                    saved_with_urls.append((
                        saved.id,
                        match_data["match_url"],
                        match_data.get("match_date", datetime.now()),
                    ))
        # session committed here

        # Enrich recent matches (last 7 days) that lack extended stats.
        cutoff = datetime.now() - timedelta(days=7)
        stats_count = 0
        max_detail_scrapes = 10  # cap per league run to limit Cloudflare exposure
        for match_id, match_url, match_date in saved_with_urls:
            if stats_count >= max_detail_scrapes:
                break
            if isinstance(match_date, datetime) and match_date < cutoff:
                continue
            # Skip if extended stats already present
            with db.get_session() as session:
                m = session.get(Match, match_id)
                if m is None or m.home_shots is not None:
                    continue

            stats = await self.scrape_match_statistics(match_url)
            if stats:
                with db.get_session() as session:
                    m = session.get(Match, match_id)
                    if m:
                        self._apply_stats_to_match(session, m, stats)
                        stats_count += 1
            # Small delay to reduce Cloudflare detection risk
            await asyncio.sleep(1)

        if stats_count:
            logger.info(f"[Flashscore] Extended stats saved for {stats_count} {league} matches")
        logger.info(f"Scraped {len(matches)} results from {league}")
        return matches

    async def scrape_league_fixtures(self, league: str):
        """Scrape upcoming fixtures for a league."""
        url = f"{FLASHSCORE_BASE_URL}/football/{league}/fixtures/"
        logger.info(f"Scraping fixtures: {league}")

        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(None, self._scrape_fixtures_page, url)

        db = get_db()
        with db.get_session() as session:
            for match_data in matches:
                self._save_match(session, match_data, league, is_fixture=True)

        logger.info(f"Scraped {len(matches)} fixtures from {league}")
        return matches

    def _click_load_more(self, driver):
        """Click 'Load more' until all matches are visible on the listing page."""
        LOAD_MORE_SELECTOR = '[data-testid="wcl-buttonLink"]'
        MATCH_SELECTOR = ".event__match.event__match--static.event__match--twoLine"
        MAX_EMPTY_CYCLES = 4
        empty_cycles = 0

        while True:
            try:
                count_before = len(driver.find_elements(By.CSS_SELECTOR, MATCH_SELECTOR))
                btn = driver.find_elements(By.CSS_SELECTOR, LOAD_MORE_SELECTOR)
                if not btn:
                    break
                btn[0].click()
                import time; time.sleep(0.7)
                count_after = len(driver.find_elements(By.CSS_SELECTOR, MATCH_SELECTOR))
                if count_after == count_before:
                    empty_cycles += 1
                    if empty_cycles >= MAX_EMPTY_CYCLES:
                        break
                else:
                    empty_cycles = 0
            except Exception:
                break

    def _scrape_results_page(self, url: str) -> List[dict]:
        """Scrape a results page using Selenium."""
        driver = self._get_driver()
        if not driver:
            logger.debug("Flashscore: Chrome not available, skipping results scrape")
            return []
        matches = []

        try:
            driver.get(url)
            # Wait for first match row
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )
            self._click_load_more(driver)

            match_elements = driver.find_elements(
                By.CSS_SELECTOR,
                ".event__match.event__match--static.event__match--twoLine"
            )
            if not match_elements:
                # Fall back to generic selector if the specific one returns nothing
                match_elements = driver.find_elements(By.CLASS_NAME, "event__match")

            for el in match_elements:
                try:
                    match_data = self._parse_match_element(el)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.debug(f"Failed to parse match element: {e}")

        except Exception as e:
            logger.error(f"Error loading results page {url}: {e}")

        return matches

    def _scrape_fixtures_page(self, url: str) -> List[dict]:
        """Scrape a fixtures page using Selenium."""
        driver = self._get_driver()
        if not driver:
            return []
        matches = []

        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )
            self._click_load_more(driver)

            match_elements = driver.find_elements(
                By.CSS_SELECTOR,
                ".event__match.event__match--static.event__match--twoLine"
            )
            if not match_elements:
                match_elements = driver.find_elements(By.CLASS_NAME, "event__match")

            for el in match_elements:
                try:
                    match_data = self._parse_fixture_element(el)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.debug(f"Failed to parse fixture element: {e}")

        except Exception as e:
            logger.error(f"Error loading fixtures page {url}: {e}")

        return matches

    def _parse_match_element(self, element) -> Optional[dict]:
        """Parse a completed match row from the listing page."""
        try:
            # Team names — try new selectors first, fall back to old
            try:
                home_team = element.find_element(
                    By.CSS_SELECTOR,
                    ".duelParticipant__home .participant__participantName"
                ).text.strip()
                away_team = element.find_element(
                    By.CSS_SELECTOR,
                    ".duelParticipant__away .participant__participantName"
                ).text.strip()
            except Exception:
                home_team = element.find_element(
                    By.CLASS_NAME, "event__participant--home"
                ).text.strip()
                away_team = element.find_element(
                    By.CLASS_NAME, "event__participant--away"
                ).text.strip()

            # Scores
            scores = element.find_elements(By.CLASS_NAME, "event__score")
            if len(scores) >= 2:
                home_goals = int(scores[0].text.strip())
                away_goals = int(scores[1].text.strip())
            else:
                return None

            # Date/time — try new selector, fall back to old
            match_date = datetime.now()
            for time_selector in [".duelParticipant__startTime", "event__time"]:
                try:
                    by = By.CSS_SELECTOR if time_selector.startswith(".") else By.CLASS_NAME
                    time_el = element.find_elements(by, time_selector)
                    if time_el:
                        date_text = time_el[0].text.strip()
                        try:
                            match_date = datetime.strptime(date_text, "%d.%m.%Y %H:%M")
                        except ValueError:
                            pass
                        break
                except Exception:
                    continue

            # Match detail URL (for fetching extended stats later)
            match_url = None
            try:
                link = element.find_element(By.CSS_SELECTOR, "a.eventRowLink")
                match_url = link.get_attribute("href")
            except Exception:
                pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "match_date": match_date,
                "match_url": match_url,
            }
        except Exception:
            return None

    def _parse_fixture_element(self, element) -> Optional[dict]:
        """Parse an upcoming fixture row from the listing page."""
        try:
            try:
                home_team = element.find_element(
                    By.CSS_SELECTOR,
                    ".duelParticipant__home .participant__participantName"
                ).text.strip()
                away_team = element.find_element(
                    By.CSS_SELECTOR,
                    ".duelParticipant__away .participant__participantName"
                ).text.strip()
            except Exception:
                home_team = element.find_element(
                    By.CLASS_NAME, "event__participant--home"
                ).text.strip()
                away_team = element.find_element(
                    By.CLASS_NAME, "event__participant--away"
                ).text.strip()

            match_date = datetime.now() + timedelta(days=1)
            for time_selector in [".duelParticipant__startTime", "event__time"]:
                try:
                    by = By.CSS_SELECTOR if time_selector.startswith(".") else By.CLASS_NAME
                    time_el = element.find_elements(by, time_selector)
                    if time_el:
                        date_text = time_el[0].text.strip()
                        try:
                            match_date = datetime.strptime(date_text, "%d.%m.%Y %H:%M")
                        except ValueError:
                            pass
                        break
                except Exception:
                    continue

            match_url = None
            try:
                link = element.find_element(By.CSS_SELECTOR, "a.eventRowLink")
                match_url = link.get_attribute("href")
            except Exception:
                pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "match_url": match_url,
            }
        except Exception:
            return None

    async def scrape_match_statistics(self, match_url: str) -> Optional[dict]:
        """Scrape detailed statistics + match info for a specific match URL."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._scrape_match_detail, match_url)

    def _scrape_match_detail(self, url: str) -> Optional[dict]:
        """Scrape match detail page for stats, referee, venue (runs in executor)."""
        driver = self._get_driver()
        if not driver:
            return None
        result = {}

        try:
            # --- Statistics sub-page ---
            stats_url = url.rstrip("/") + "/summary/stats/0/"
            driver.get(stats_url)

            # Try new data-testid selectors first
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div[data-testid='wcl-statistics']")
                    )
                )
                stat_rows = driver.find_elements(
                    By.CSS_SELECTOR, "div[data-testid='wcl-statistics']"
                )
                for row in stat_rows:
                    try:
                        label = row.find_element(
                            By.CSS_SELECTOR, "div[data-testid='wcl-statistics-category']"
                        ).text.strip().lower()
                        values = row.find_elements(
                            By.CSS_SELECTOR, "div[data-testid='wcl-statistics-value'] > strong"
                        )
                        if len(values) >= 2:
                            self._apply_stat(result, label,
                                             values[0].text.strip(),
                                             values[1].text.strip())
                    except Exception:
                        continue

            except Exception:
                # Fall back to old class-based selectors
                try:
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "stat__row"))
                    )
                    stat_rows = driver.find_elements(By.CLASS_NAME, "stat__row")
                    for row in stat_rows:
                        try:
                            label = row.find_element(
                                By.CLASS_NAME, "stat__categoryName"
                            ).text.strip().lower()
                            home_els = row.find_elements(By.CLASS_NAME, "stat__homeValue")
                            away_els = row.find_elements(By.CLASS_NAME, "stat__awayValue")
                            if home_els and away_els:
                                self._apply_stat(result, label,
                                                 home_els[0].text.strip(),
                                                 away_els[0].text.strip())
                        except Exception:
                            continue
                except Exception:
                    pass

            # --- Match information (referee, venue, capacity) ---
            try:
                driver.get(url.rstrip("/") + "/")
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "div[data-testid='wcl-summaryMatchInformation']")
                    )
                )
                info_container = driver.find_element(
                    By.CSS_SELECTOR, "div[data-testid='wcl-summaryMatchInformation']"
                )
                info_divs = info_container.find_elements(By.XPATH, "./div")
                # Even-indexed = labels, odd-indexed = values
                for i in range(0, len(info_divs) - 1, 2):
                    try:
                        key = info_divs[i].text.strip().lower()
                        val = info_divs[i + 1].text.strip()
                        if "referee" in key:
                            result["referee"] = val
                        elif "venue" in key or "stadium" in key or "ground" in key:
                            result["venue"] = val
                        elif "capacity" in key:
                            try:
                                result["venue_capacity"] = int(val.replace(",", "").replace(".", ""))
                            except ValueError:
                                pass
                    except Exception:
                        continue
            except Exception:
                pass

            # --- Regulation time / penalty score ---
            try:
                reg_el = driver.find_elements(By.CLASS_NAME, "detailScore__fullTime")
                if reg_el:
                    reg_text = reg_el[0].text.strip().replace("(", "").replace(")", "")
                    parts = reg_text.split(":")
                    if len(parts) == 2:
                        result["regulation_home_goals"] = int(parts[0].strip())
                        result["regulation_away_goals"] = int(parts[1].strip())
            except Exception:
                pass

        except Exception as e:
            logger.error(f"Error scraping match detail {url}: {e}")
            return None

        return result if result else None

    def _apply_stat(self, result: dict, label: str, home_raw: str, away_raw: str):
        """Parse a stat label/value pair and store in result dict."""
        if label not in STAT_MAP:
            return
        home_key, away_key = STAT_MAP[label]
        try:
            home_val = float(home_raw.replace("%", "").strip())
            away_val = float(away_raw.replace("%", "").strip())
            result[home_key] = home_val
            result[away_key] = away_val
        except ValueError:
            pass

    def _apply_stats_to_match(self, session, match: Match, stats: dict):
        """Apply a stats dict returned by _scrape_match_detail to a Match record.

        The stats dict uses Match column names as keys (e.g. 'home_shots',
        'away_possession', 'referee', 'venue', 'venue_capacity',
        'regulation_home_goals', 'regulation_away_goals').
        Unknown keys are silently ignored.
        """
        direct_fields = {
            "referee", "venue", "venue_capacity",
            "regulation_home_goals", "regulation_away_goals",
        }
        for key, value in stats.items():
            if key in direct_fields or hasattr(match, key):
                try:
                    setattr(match, key, value)
                except Exception:
                    pass

    def _get_or_create_team(self, session, team_name: str, league: str) -> Team:
        """Get existing team or create a new one."""
        team = session.query(Team).filter_by(name=team_name).first()
        if not team:
            team = Team(name=team_name, league=league)
            session.add(team)
            session.flush()
        return team

    def _save_match(self, session, match_data: dict, league: str, is_fixture: bool):
        """Save a match to the database, avoiding duplicates."""
        home_team = self._get_or_create_team(session, match_data["home_team"], league)
        away_team = self._get_or_create_team(session, match_data["away_team"], league)

        existing = session.query(Match).filter_by(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_data["match_date"],
        ).first()

        if existing:
            if existing.is_fixture and not is_fixture and "home_goals" in match_data:
                existing.home_goals = match_data.get("home_goals")
                existing.away_goals = match_data.get("away_goals")
                existing.is_fixture = False
            return existing

        match = Match(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_data["match_date"],
            league=league,
            home_goals=match_data.get("home_goals"),
            away_goals=match_data.get("away_goals"),
            is_fixture=is_fixture,
        )
        session.add(match)
        return match

    async def get_h2h_data(self, team1_name: str, team2_name: str, limit: int = 10) -> List[Match]:
        """Get head-to-head data between two teams from the database."""
        db = get_db()
        with db.get_session() as session:
            team1 = session.query(Team).filter_by(name=team1_name).first()
            team2 = session.query(Team).filter_by(name=team2_name).first()

            if not team1 or not team2:
                return []

            matches = session.query(Match).filter(
                Match.is_fixture == False,
                (
                    ((Match.home_team_id == team1.id) & (Match.away_team_id == team2.id)) |
                    ((Match.home_team_id == team2.id) & (Match.away_team_id == team1.id))
                )
            ).order_by(Match.match_date.desc()).limit(limit).all()

            return matches
