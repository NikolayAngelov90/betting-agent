"""Flashscore scraper for match results, fixtures, and statistics."""

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
                # Use a thread with timeout to prevent hanging on Chrome startup
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

        Args:
            league: League path (e.g. 'england/premier-league')
            num_pages: Number of result pages to scrape
        """
        url = f"{FLASHSCORE_BASE_URL}/football/{league}/results/"
        logger.info(f"Scraping results: {league}")

        loop = asyncio.get_event_loop()
        matches = await loop.run_in_executor(None, self._scrape_results_page, url)

        db = get_db()
        with db.get_session() as session:
            for match_data in matches:
                self._save_match(session, match_data, league, is_fixture=False)

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

    def _scrape_results_page(self, url: str) -> List[dict]:
        """Scrape a results page using Selenium (runs in thread executor)."""
        driver = self._get_driver()
        if not driver:
            logger.debug("Flashscore: Chrome not available, skipping results scrape")
            return []
        matches = []

        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )

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
        """Scrape a fixtures page using Selenium (runs in thread executor)."""
        driver = self._get_driver()
        matches = []

        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )

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
        """Parse a completed match element into a data dictionary."""
        try:
            home_team = element.find_element(By.CLASS_NAME, "event__participant--home").text.strip()
            away_team = element.find_element(By.CLASS_NAME, "event__participant--away").text.strip()

            scores = element.find_elements(By.CLASS_NAME, "event__score")
            if len(scores) >= 2:
                home_goals = int(scores[0].text.strip())
                away_goals = int(scores[1].text.strip())
            else:
                return None

            # Try to get match time/date
            time_el = element.find_elements(By.CLASS_NAME, "event__time")
            match_date = datetime.now()
            if time_el:
                date_text = time_el[0].text.strip()
                try:
                    match_date = datetime.strptime(date_text, "%d.%m.%Y %H:%M")
                except ValueError:
                    pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "match_date": match_date,
            }
        except Exception:
            return None

    def _parse_fixture_element(self, element) -> Optional[dict]:
        """Parse an upcoming fixture element into a data dictionary."""
        try:
            home_team = element.find_element(By.CLASS_NAME, "event__participant--home").text.strip()
            away_team = element.find_element(By.CLASS_NAME, "event__participant--away").text.strip()

            time_el = element.find_elements(By.CLASS_NAME, "event__time")
            match_date = datetime.now() + timedelta(days=1)
            if time_el:
                date_text = time_el[0].text.strip()
                try:
                    match_date = datetime.strptime(date_text, "%d.%m.%Y %H:%M")
                except ValueError:
                    pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
            }
        except Exception:
            return None

    async def scrape_match_statistics(self, match_url: str) -> Optional[dict]:
        """Scrape detailed statistics for a specific match."""
        logger.info(f"Scraping match stats: {match_url}")

        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(None, self._scrape_stats_page, match_url)
        return stats

    def _scrape_stats_page(self, url: str) -> Optional[dict]:
        """Scrape match statistics page (runs in thread executor)."""
        driver = self._get_driver()
        stats = {}

        try:
            stats_url = url if url.endswith("/match-statistics/") else url.rstrip("/") + "/#/match-summary/match-statistics"
            driver.get(stats_url)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "stat__row"))
            )

            stat_rows = driver.find_elements(By.CLASS_NAME, "stat__row")
            for row in stat_rows:
                try:
                    label = row.find_element(By.CLASS_NAME, "stat__categoryName").text.strip().lower()
                    values = row.find_elements(By.CLASS_NAME, "stat__homeValue") + \
                             row.find_elements(By.CLASS_NAME, "stat__awayValue")

                    if len(values) >= 2:
                        home_val = values[0].text.strip().replace("%", "")
                        away_val = values[1].text.strip().replace("%", "")

                        stat_map = {
                            "ball possession": ("home_possession", "away_possession"),
                            "shots on target": ("home_shots_on_target", "away_shots_on_target"),
                            "shots": ("home_shots", "away_shots"),
                            "corner kicks": ("home_corners", "away_corners"),
                            "fouls": ("home_fouls", "away_fouls"),
                            "yellow cards": ("home_yellow_cards", "away_yellow_cards"),
                            "red cards": ("home_red_cards", "away_red_cards"),
                        }

                        if label in stat_map:
                            home_key, away_key = stat_map[label]
                            try:
                                stats[home_key] = float(home_val)
                                stats[away_key] = float(away_val)
                            except ValueError:
                                pass
                except Exception:
                    continue

        except Exception as e:
            logger.error(f"Error scraping stats from {url}: {e}")
            return None

        return stats if stats else None

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

        # Check for existing match
        existing = session.query(Match).filter_by(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_data["match_date"],
        ).first()

        if existing:
            # Update result if it was a fixture and now has results
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
