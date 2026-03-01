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

import re
import time
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urlparse, parse_qs

from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import asyncio
import concurrent.futures

try:
    import undetected_chromedriver as uc
    _UC_AVAILABLE = True
except ImportError:
    # Fallback to standard selenium if undetected-chromedriver not installed
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    _UC_AVAILABLE = False

try:
    import camoufox as _camoufox_mod  # noqa: F401
    _CF_AVAILABLE = True
except ImportError:
    _CF_AVAILABLE = False

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
        # camoufox (Firefox + anti-fingerprinting) — primary Cloudflare bypass
        self._cf_browser = None    # playwright sync Browser (camoufox)
        self._cf_ctx_mgr = None   # camoufox context manager handle
        self._cf_failed = False   # True after an unrecoverable camoufox error

    def _get_driver(self):
        """Create a Chrome driver. Returns None if Chrome unavailable.

        Uses undetected-chromedriver when available to bypass Cloudflare's
        bot detection (removes navigator.webdriver flag and other signals).
        Falls back to standard Selenium if the package is not installed.

        When a virtual display is available (DISPLAY env var set by Xvfb in CI),
        Chrome runs in headed mode — Cloudflare detects headless markers far more
        reliably than headed Chrome, so Xvfb + headed is the primary bypass.
        """
        import os as _os
        if self._chrome_failed:
            return None
        if self._driver is None:
            try:
                if _UC_AVAILABLE:
                    options = uc.ChromeOptions()
                else:
                    options = Options()  # type: ignore[assignment]

                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--disable-extensions")
                # Realistic 1080p viewport — headless default (640×480) is a
                # known Cloudflare fingerprint signal.
                options.add_argument("--window-size=1920,1080")
                options.add_argument(
                    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
                )
                # page_load_strategy='none': driver.get() returns immediately
                # without waiting for the browser load event. Flashscore is a
                # Cloudflare-protected SPA whose load event sometimes never fires,
                # causing driver.get() to hang for the full timeout. With 'none'
                # we rely entirely on explicit WebDriverWait for specific elements.
                options.page_load_strategy = "none"

                # If a virtual display (Xvfb) is running, disable headless mode.
                # Headed Chrome on Xvfb is indistinguishable from a real desktop
                # browser to Cloudflare — the primary bypass for odds comparison pages.
                _has_virtual_display = bool(_os.environ.get("DISPLAY"))
                _run_headless = self.headless and not _has_virtual_display
                if _has_virtual_display:
                    logger.info("Xvfb detected — running Chrome in headed mode (Cloudflare bypass)")

                if _UC_AVAILABLE:
                    import platform as _platform
                    import subprocess as _subprocess
                    import re as _re
                    # headless= passed directly to uc.Chrome is stealthier than
                    # the --headless flag in options (uc uses a different code path).
                    # use_subprocess=True avoids zombie processes in CI (Linux only —
                    # on Windows it causes Chrome to exit immediately).
                    _use_subprocess = _platform.system() != "Windows"

                    # Detect installed Chrome binary path and major version so uc
                    # downloads the matching ChromeDriver AND launches the exact
                    # same binary (prevents "ChromeDriver X supports Chrome Y"
                    # when multiple Chrome versions exist on the runner).
                    import shutil as _shutil
                    _chrome_version = None
                    _chrome_binary = None
                    for _cmd in (
                        ["google-chrome", "--version"],
                        ["google-chrome-stable", "--version"],
                        ["chromium-browser", "--version"],
                        ["chromium", "--version"],
                    ):
                        try:
                            _out = _subprocess.run(
                                _cmd, capture_output=True, text=True, timeout=5
                            ).stdout
                            _m = _re.search(r"(\d+)\.", _out)
                            if _m:
                                _chrome_version = int(_m.group(1))
                                _chrome_binary = _shutil.which(_cmd[0])
                                break
                        except Exception:
                            continue

                    # Pin binary_location so ChromeDriver uses the exact same
                    # Chrome build that version_main was detected from.
                    if _chrome_binary:
                        options.binary_location = _chrome_binary

                    def _create_uc():
                        kwargs = dict(
                            options=options,
                            headless=_run_headless,
                            use_subprocess=_use_subprocess,
                        )
                        if _chrome_version:
                            kwargs["version_main"] = _chrome_version
                        return uc.Chrome(**kwargs)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(_create_uc)
                        self._driver = future.result(timeout=30)
                else:
                    if _run_headless:
                        options.add_argument("--headless=new")
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(webdriver.Chrome, options=options)  # type: ignore[attr-defined]
                        self._driver = future.result(timeout=15)

                self._driver.implicitly_wait(0)  # explicit waits only
            except Exception as e:
                logger.warning(f"Chrome/Selenium not available: {e}")
                self._chrome_failed = True
                self._driver = None
        return self._driver

    def _get_camoufox_browser(self):
        """Lazy-init a persistent camoufox Firefox browser (Cloudflare bypass).

        Returns a Playwright sync Browser on success, None if unavailable.
        The browser is kept alive across requests so Cloudflare cookies
        (cf_clearance, __cf_bm) persist between pages within a session.
        """
        import os as _os
        if not _CF_AVAILABLE or self._cf_failed:
            return None
        if self._cf_browser is not None:
            return self._cf_browser
        try:
            from camoufox.sync_api import Camoufox
            # Worker-thread fix: when called from run_in_executor, the worker thread
            # inherits the parent's running-loop reference. Playwright's sync API checks
            # asyncio._get_running_loop() and raises "Playwright Sync API inside the
            # asyncio loop" even though this thread is NOT actually running the loop.
            # Fix: clear the running-loop marker on this thread, then set a fresh
            # (non-running) event loop so get_event_loop() doesn't error.
            import asyncio as _asyncio
            try:
                # Clear the "running loop" flag inherited from the parent thread
                _asyncio._set_running_loop(None)
                _asyncio.set_event_loop(_asyncio.new_event_loop())
            except Exception:
                pass
            # headed (headless=False) with Xvfb is the most realistic profile;
            # headless=True still works but risks detection on Cloudflare JS challenge.
            _has_display = bool(_os.environ.get("DISPLAY"))
            mgr = Camoufox(
                headless=(not _has_display),
                os="windows",   # Windows UA fingerprint — most common desktop platform
            )
            self._cf_browser = mgr.__enter__()
            self._cf_ctx_mgr = mgr
            logger.info(
                f"camoufox Firefox browser started "
                f"(headless={not _has_display}, Cloudflare bypass active)"
            )
        except Exception as e:
            logger.warning(f"camoufox failed to start: {e} — will use Selenium fallback")
            self._cf_failed = True
            self._cf_browser = None
        return self._cf_browser

    def _close_camoufox(self):
        """Shut down the camoufox Firefox browser and reset state."""
        if self._cf_ctx_mgr is not None:
            try:
                self._cf_ctx_mgr.__exit__(None, None, None)
            except Exception:
                pass
        self._cf_browser = None
        self._cf_ctx_mgr = None
        self._cf_failed = False

    def _cf_fetch_html(self, url: str, wait_selector: str,
                       timeout_ms: int = 20000) -> Optional[str]:
        """Load *url* with camoufox Firefox and return the page HTML.

        Waits up to *timeout_ms* ms for *wait_selector* to appear before
        returning the page source.  Returns None if camoufox is unavailable
        or the page load fails (caller should fall back to Selenium).
        """
        browser = self._get_camoufox_browser()
        if browser is None:
            return None
        try:
            page = browser.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                try:
                    page.wait_for_selector(wait_selector, timeout=timeout_ms)
                except Exception:
                    pass  # selector may not appear if Cloudflare still blocks
                html = page.content()
            finally:
                page.close()
            return html
        except Exception as e:
            logger.warning(
                f"camoufox page load failed ({type(e).__name__}) for {url} "
                f"— marking camoufox as failed, falling back to Selenium"
            )
            self._close_camoufox()
            self._cf_failed = True
            return None

    def close_driver(self):
        """Close the Selenium driver and camoufox browser; reset failure flags."""
        if self._driver:
            try:
                self._driver.quit()
            except Exception:
                pass
            self._driver = None
        # Reset so next league can attempt a fresh Chrome session
        self._chrome_failed = False
        # Also close camoufox — caller can re-open a fresh browser next time
        self._close_camoufox()

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

    async def scrape_league_results(self, league: str, num_pages: int = 1,
                                    skip_stats: bool = False):
        """Scrape recent match results for a league.

        After saving basic scores, fetches extended statistics (shots, possession,
        dangerous attacks, saves, offsides, referee, etc.) for recently completed
        matches that are missing that data.  Limited to last 7 days and 10 matches
        per call to avoid overloading Flashscore / triggering Cloudflare.

        Args:
            skip_stats: When True, skip the per-match detail page enrichment.
                        Scores are still saved; only extended stats are omitted.
                        Use in time-sensitive contexts (e.g. daily_update).
        """
        url = f"{FLASHSCORE_BASE_URL}/football/{league}/results/"
        logger.info(f"Scraping results: {league}")

        loop = asyncio.get_event_loop()
        # When skip_stats=True we only need today's/yesterday's results for
        # settlement — skip load_more to avoid the indefinite click loop that
        # causes Premier League to always exceed the timeout.
        matches = await loop.run_in_executor(
            None, self._scrape_results_page, url, not skip_stats
        )

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

        if skip_stats:
            logger.info(f"Scraped {len(matches)} results from {league} (stats skipped)")
            return matches

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

    async def enrich_recent_match_stats(self, days_back: int = 7, max_matches: int = 50):
        """Backfill per-match stats (shots, possession, corners, etc.) for recent
        completed matches that are missing extended stats.

        Runs a single cross-league pass capped at `max_matches` total so the
        per-league timeout in the results loop doesn't prevent stat collection.
        Matches are processed most-recent-first so today's games get stats first.
        """
        db = get_db()
        cutoff = datetime.now() - timedelta(days=days_back)
        with db.get_session() as session:
            matches_needing_stats = session.query(Match).filter(
                Match.is_fixture == False,
                Match.home_goals.isnot(None),
                Match.home_shots.is_(None),
                Match.flashscore_id.isnot(None),
                Match.match_date >= cutoff,
            ).order_by(Match.match_date.desc()).limit(max_matches).all()
            pending = [(m.id, m.flashscore_id) for m in matches_needing_stats]

        if not pending:
            logger.debug("No matches need Flashscore stats enrichment")
            return 0

        total = len(pending)
        logger.info(f"Enriching stats for {total} recent matches (cross-league pass)")
        enriched = 0
        consecutive_failures = 0
        _CF_ABORT = 4  # abort if Cloudflare blocks this many matches in a row

        for n, (match_id, fs_id) in enumerate(pending, 1):
            if consecutive_failures >= _CF_ABORT:
                logger.warning(
                    f"Stats enrichment: {consecutive_failures} consecutive failures "
                    f"(Cloudflare?) — aborting early after {n - 1}/{total} matches"
                )
                break

            logger.info(f"[Stats {n}/{total}] Scraping match {match_id} (id={fs_id})")
            match_url = f"{FLASHSCORE_BASE_URL}/match/{fs_id}/"
            try:
                # Per-match 60s timeout — if Chrome hangs on a Cloudflare challenge
                # we detect it quickly instead of burning the full 900s budget.
                stats = await asyncio.wait_for(
                    self.scrape_match_statistics(match_url, stats_only=True),
                    timeout=60,
                )
            except asyncio.TimeoutError:
                logger.warning(f"[Stats {n}/{total}] Timeout after 60s for {fs_id}")
                consecutive_failures += 1
                continue
            except Exception as exc:
                logger.warning(f"[Stats {n}/{total}] Error for {fs_id}: {exc}")
                consecutive_failures += 1
                continue

            if stats:
                with db.get_session() as session:
                    m = session.get(Match, match_id)
                    if m:
                        self._apply_stats_to_match(session, m, stats)
                        enriched += 1
                consecutive_failures = 0  # reset on success
            else:
                logger.debug(f"[Stats {n}/{total}] No stats returned for {fs_id}")
                consecutive_failures += 1

            await asyncio.sleep(1)

        logger.info(f"Flashscore stats enriched: {enriched}/{total} matches")
        return enriched

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

    def _scrape_results_page(self, url: str, load_more: bool = True) -> List[dict]:
        """Scrape a results page using Selenium.

        Args:
            load_more: When False, skip clicking 'Load more' and only parse the
                       matches already visible on the first page (~10-20 most recent).
                       Use when only the latest results are needed (e.g. settlement).
        """
        driver = self._get_driver()
        if not driver:
            logger.debug("Flashscore: Chrome not available, skipping results scrape")
            return []
        matches = []

        try:
            driver.get(url)
            # With page_load_strategy='none', driver.get() returns immediately.
            # Wait up to 20s for the first match row to be rendered by JS.
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )
            if load_more:
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
            # Always reset the driver on any page-load failure: Chrome may have
            # crashed (empty Message:), timed out, or lost its window. Resetting
            # ensures the next league starts with a fresh Chrome instance instead
            # of inheriting a broken session that causes cascading failures.
            self._driver = None
            logger.warning(f"Results page failed ({type(e).__name__}) for {url} — retrying once")
            try:
                driver2 = self._get_driver()
                if driver2:
                    driver2.get(url)
                    WebDriverWait(driver2, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
                    )
                    for el in driver2.find_elements(By.CLASS_NAME, "event__match"):
                        try:
                            m = self._parse_result_element(el)
                            if m:
                                matches.append(m)
                        except Exception:
                            pass
                    logger.info(f"Results page retry OK: {len(matches)} results from {url}")
            except Exception as e2:
                logger.warning(f"Results page retry also failed for {url}: {type(e2).__name__} — skipping")
                self._driver = None

        return matches

    def _scrape_fixtures_page(self, url: str, max_days_ahead: int = 7) -> List[dict]:
        """Scrape a fixtures page using Selenium.

        Only loads and saves fixtures within `max_days_ahead` days to avoid
        clicking through the entire season (which causes timeouts and stores
        hundreds of irrelevant far-future matches).
        """
        driver = self._get_driver()
        if not driver:
            return []
        matches = []

        try:
            driver.get(url)
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
            )
            self._click_load_more(driver)

            match_elements = driver.find_elements(
                By.CSS_SELECTOR,
                ".event__match.event__match--static.event__match--twoLine"
            )
            if not match_elements:
                match_elements = driver.find_elements(By.CLASS_NAME, "event__match")

            cutoff = datetime.now() + timedelta(days=max_days_ahead)
            for el in match_elements:
                try:
                    match_data = self._parse_fixture_element(el)
                    if match_data:
                        if match_data.get("match_date") and match_data["match_date"] > cutoff:
                            continue  # skip far-future fixtures
                        matches.append(match_data)
                except Exception as e:
                    logger.debug(f"Failed to parse fixture element: {e}")

        except Exception as e:
            self._driver = None
            logger.warning(f"Fixtures page failed ({type(e).__name__}) for {url} — retrying once")
            try:
                driver2 = self._get_driver()
                if driver2:
                    driver2.get(url)
                    WebDriverWait(driver2, 20).until(
                        EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
                    )
                    cutoff = datetime.now() + timedelta(days=max_days_ahead)
                    for el in driver2.find_elements(By.CLASS_NAME, "event__match"):
                        try:
                            m = self._parse_fixture_element(el)
                            if m and (not m.get("match_date") or m["match_date"] <= cutoff):
                                matches.append(m)
                        except Exception:
                            pass
                    logger.info(f"Fixtures page retry OK: {len(matches)} fixtures from {url}")
            except Exception as e2:
                logger.warning(f"Fixtures page retry also failed for {url}: {type(e2).__name__} — skipping")
                self._driver = None

        return matches

    @staticmethod
    def _parse_team_names(element):
        """Extract home/away team names from a match row element.

        Flashscore has iterated through several DOM layouts; this method tries
        selectors in preference order (newest → oldest) so we survive future
        minor CSS-class renames without a full rewrite.
        """
        # 2026 layout: event__homeParticipant / event__awayParticipant
        try:
            home = element.find_element(By.CLASS_NAME, "event__homeParticipant").text.strip()
            away = element.find_element(By.CLASS_NAME, "event__awayParticipant").text.strip()
            if home and away:
                return home, away
        except Exception:
            pass
        # 2024-2025 layout: duelParticipant
        try:
            home = element.find_element(
                By.CSS_SELECTOR, ".duelParticipant__home .participant__participantName"
            ).text.strip()
            away = element.find_element(
                By.CSS_SELECTOR, ".duelParticipant__away .participant__participantName"
            ).text.strip()
            if home and away:
                return home, away
        except Exception:
            pass
        # Oldest layout: event__participant--home/away
        home = element.find_element(By.CLASS_NAME, "event__participant--home").text.strip()
        away = element.find_element(By.CLASS_NAME, "event__participant--away").text.strip()
        return home, away

    @staticmethod
    def _parse_match_date(element, default: datetime, is_result: bool = False) -> datetime:
        """Extract match date/time from a row element, returning *default* on failure.

        Args:
            element: Selenium WebElement for the match row.
            default: Fallback datetime if parsing fails.
            is_result: True when parsing a completed match (results page).
                       In that case, if the parsed date falls more than 60 days
                       in the future it is almost certainly a year-parse error
                       (e.g. Dec 2025 stored as Dec 2026) and we subtract one year.
        """
        for selector, by in [
            ("event__time", By.CLASS_NAME),
            (".duelParticipant__startTime", By.CSS_SELECTOR),
        ]:
            try:
                time_el = element.find_elements(by, selector)
                if not time_el:
                    continue
                date_text = time_el[0].text.strip()
                # "18.02.2026 22:00" (full) or "18.02. 22:00" (no year)
                for fmt in ("%d.%m.%Y %H:%M", "%d.%m. %H:%M"):
                    try:
                        dt = datetime.strptime(date_text, fmt)
                        if fmt == "%d.%m. %H:%M":
                            dt = dt.replace(year=datetime.now().year)
                            # For results: if parsed date is >60 days in the future
                            # it's a year-off error (e.g. Dec 2025 → wrongly Dec 2026).
                            if is_result and dt > datetime.now() + timedelta(days=60):
                                dt = dt.replace(year=dt.year - 1)
                        return dt
                    except ValueError:
                        continue
            except Exception:
                continue
        return default

    def _parse_match_element(self, element) -> Optional[dict]:
        """Parse a completed match row from the listing page."""
        try:
            home_team, away_team = self._parse_team_names(element)

            # Scores — event__score class still present in 2026 layout
            scores = element.find_elements(By.CLASS_NAME, "event__score")
            if len(scores) >= 2:
                home_goals = int(scores[0].text.strip())
                away_goals = int(scores[1].text.strip())
            else:
                return None  # not a completed result (e.g. header row)

            match_date = self._parse_match_date(
                element, default=datetime.now(), is_result=True
            )

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
            home_team, away_team = self._parse_team_names(element)

            match_date = self._parse_match_date(
                element, default=datetime.now() + timedelta(days=1)
            )

            match_url = None
            flashscore_id = None
            try:
                link = element.find_element(By.CSS_SELECTOR, "a.eventRowLink")
                match_url = link.get_attribute("href")
                # Extract short Flashscore match ID from ?mid=XXXXXXXX or event element id g_1_XXXXXXXX
                if match_url:
                    qs = parse_qs(urlparse(match_url).query)
                    if "mid" in qs:
                        flashscore_id = qs["mid"][0]
                if not flashscore_id:
                    # Try the event element's id attribute: "g_1_G8MZEpbl"
                    el_id = element.get_attribute("id") or ""
                    m = re.search(r'g_\d+_([A-Za-z0-9]+)', el_id)
                    if m:
                        flashscore_id = m.group(1)
            except Exception:
                pass

            return {
                "home_team": home_team,
                "away_team": away_team,
                "match_date": match_date,
                "match_url": match_url,
                "flashscore_id": flashscore_id,
            }
        except Exception:
            return None

    async def scrape_match_statistics(self, match_url: str, stats_only: bool = False) -> Optional[dict]:
        """Scrape detailed statistics + match info for a specific match URL.

        Tries camoufox (Firefox + anti-fingerprinting) first for better Cloudflare
        bypass, then falls back to Selenium.
        stats_only=True skips the second page load (referee/venue/capacity), cutting
        per-match time from ~22s to ~12s.  Use this for bulk enrichment passes.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._scrape_match_detail, match_url, stats_only)

    def _scrape_match_detail_cf(self, url: str, stats_only: bool = False) -> Optional[dict]:
        """Scrape match stats using camoufox + BeautifulSoup (Cloudflare bypass).

        Uses the same CSS selectors as the Selenium path but parses page HTML
        with BS4 instead of Selenium element APIs.
        """
        result = {}

        # --- Statistics sub-page ---
        stats_url = url.rstrip("/") + "/summary/stats/0/"
        html = self._cf_fetch_html(
            stats_url,
            "[data-testid='wcl-statistics'],[class*='stat__row']",
            timeout_ms=20000,
        )
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")

        # New (2025+) data-testid selectors
        stat_rows = soup.select("[data-testid='wcl-statistics']")
        for row in stat_rows:
            label_el = row.select_one("[data-testid='wcl-statistics-category']")
            val_els = row.select("[data-testid='wcl-statistics-value'] > strong")
            if label_el and len(val_els) >= 2:
                self._apply_stat(
                    result,
                    label_el.get_text(strip=True).lower(),
                    val_els[0].get_text(strip=True),
                    val_els[1].get_text(strip=True),
                )

        # Fallback to older class-based selectors
        if not result:
            for row in soup.select(".stat__row"):
                label_el = row.select_one(".stat__categoryName")
                home_el = row.select_one(".stat__homeValue")
                away_el = row.select_one(".stat__awayValue")
                if label_el and home_el and away_el:
                    self._apply_stat(
                        result,
                        label_el.get_text(strip=True).lower(),
                        home_el.get_text(strip=True),
                        away_el.get_text(strip=True),
                    )

        if stats_only:
            return result if result else None

        # --- Match info (referee, venue, capacity) ---
        info_html = self._cf_fetch_html(
            url.rstrip("/") + "/",
            "[data-testid='wcl-summaryMatchInformation']",
            timeout_ms=15000,
        )
        if info_html:
            soup2 = BeautifulSoup(info_html, "lxml")
            container = soup2.select_one("[data-testid='wcl-summaryMatchInformation']")
            if container:
                divs = list(container.children)
                divs = [d for d in divs if hasattr(d, "get_text")]
                for i in range(0, len(divs) - 1, 2):
                    key = divs[i].get_text(strip=True).lower()
                    val = divs[i + 1].get_text(strip=True)
                    if "referee" in key:
                        result["referee"] = val
                    elif "venue" in key or "stadium" in key or "ground" in key:
                        result["venue"] = val
                    elif "capacity" in key:
                        try:
                            result["venue_capacity"] = int(
                                val.replace(",", "").replace(".", "")
                            )
                        except ValueError:
                            pass

        return result if result else None

    def _scrape_match_detail(self, url: str, stats_only: bool = False) -> Optional[dict]:
        """Scrape match detail page for stats, referee, venue (runs in executor).

        Tries camoufox (Firefox + anti-fingerprinting) first for Cloudflare bypass,
        then falls back to Selenium/undetected-chromedriver.
        stats_only=True skips the referee/venue page — saves ~10s per match.
        """
        # --- camoufox path (primary: Firefox + anti-fingerprinting) ---
        if _CF_AVAILABLE and not self._cf_failed:
            result = self._scrape_match_detail_cf(url, stats_only)
            if result:
                return result
            logger.debug(f"camoufox stats empty for {url} — trying Selenium fallback")

        # --- Selenium fallback ---
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
            # Skipped in stats_only mode to halve per-match scrape time.
            if stats_only:
                return result if result else None

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

    # ------------------------------------------------------------------
    # Odds scraping
    # ------------------------------------------------------------------

    def _parse_odds_from_soup(self, soup: BeautifulSoup, market: str) -> dict:
        """Parse odds from page HTML (used by the camoufox path)."""
        result = {}
        spans = soup.select("[data-testid='wcl-oddsValue']")
        values = []
        for span in spans:
            try:
                values.append(float(span.get_text(strip=True)))
            except ValueError:
                pass

        if market == "home-draw-away":
            if len(values) >= 3:
                home_vals, draw_vals, away_vals = [], [], []
                for i in range(0, len(values) - 2, 3):
                    home_vals.append(values[i])
                    draw_vals.append(values[i + 1])
                    away_vals.append(values[i + 2])
                if home_vals:
                    result["Home Win"] = max(home_vals)
                if draw_vals:
                    result["Draw"] = max(draw_vals)
                if away_vals:
                    result["Away Win"] = max(away_vals)

        elif market == "over-under":
            # Walk UP from each wcl-oddsValue span to find its line-specific
            # section.  A line section is the closest ancestor where exactly
            # one "Over X.5" string appears in its text (avoids catching
            # multi-line parent containers that hold all lines at once).
            # Values come in pairs per bookmaker row: (Over, Under, Over, Under,
            # ...) so even indices = Over, odd indices = Under for each line.
            from collections import defaultdict as _dd
            line_buckets: dict = _dd(list)

            for span in soup.find_all(attrs={"data-testid": "wcl-oddsValue"}):
                try:
                    val = float(span.get_text(strip=True))
                except ValueError:
                    continue

                node = span.parent
                assigned = None
                while node and node.name not in ("body", "html", "[document]"):
                    over_hits = re.findall(r"\bOver (\d+\.5)\b",
                                          node.get_text(separator=" ", strip=True))
                    if len(over_hits) == 1:
                        assigned = over_hits[0]
                        break
                    if len(over_hits) > 1:
                        break  # too high in the tree — stop
                    node = node.parent

                if assigned:
                    line_buckets[assigned].append(val)

            for line, vals in line_buckets.items():
                if len(vals) < 2:
                    continue
                over_vals  = [v for v in vals[0::2] if 1.01 < v < 20]
                under_vals = [v for v in vals[1::2] if 1.01 < v < 20]
                if over_vals:
                    result[f"Over {line}"]  = max(over_vals)
                if under_vals:
                    result[f"Under {line}"] = max(under_vals)

        elif market == "btts":
            if len(values) >= 2:
                yes_vals, no_vals = [], []
                for i in range(0, len(values) - 1, 2):
                    yes_vals.append(values[i])
                    no_vals.append(values[i + 1])
                ft_yes = [v for v in yes_vals if v < 2.5]
                if ft_yes:
                    result["BTTS Yes"] = max(ft_yes)
                elif yes_vals:
                    result["BTTS Yes"] = min(yes_vals)
                if no_vals:
                    result["BTTS No"] = max(no_vals)

        return result

    def _scrape_odds_page_cf(self, flashscore_id: str, market: str = "home-draw-away") -> dict:
        """Scrape odds using camoufox Firefox (Cloudflare bypass).

        Loads the Flashscore odds-comparison page with camoufox, then parses
        the result HTML with BeautifulSoup.  Falls back gracefully (returns {})
        if camoufox is not available or the page load fails.
        """
        market_path = {
            "home-draw-away": "home-draw-away/full-time",
            "over-under": "over-under/full-time",
            "btts": "both-teams-to-score/full-time",
        }.get(market, market)

        # Use market-specific wait selectors.
        # [data-testid='wcl-oddsValue'] fires on the initial 1X2 content
        # before the SPA has navigated to the O/U or BTTS tab — we'd capture
        # the wrong HTML.  Waiting for content that only exists on each tab
        # ensures we have the right page before parsing.
        wait_selector = {
            "home-draw-away": "[data-testid='wcl-oddsValue']",
            "over-under":     "text=Over 0.5",
            "btts":           "text=Both Teams",
        }.get(market, "[data-testid='wcl-oddsValue']")

        urls = [
            f"https://www.flashscore.com/match/{flashscore_id}/#/odds-comparison/{market_path}",
            f"https://www.flashscore.com/match/{flashscore_id}/odds-comparison/{market_path}/",
        ]
        for url in urls:
            html = self._cf_fetch_html(url, wait_selector, timeout_ms=12000)
            if html:
                soup = BeautifulSoup(html, "lxml")
                result = self._parse_odds_from_soup(soup, market)
                if result:
                    return result
                # Got HTML but parsing returned nothing — both URLs go through the
                # same SPA router so the second URL would return identical content.
                # Stop early to avoid wasting another 8s.
                break
        return {}

    def _scrape_odds_page(self, flashscore_id: str, market: str = "home-draw-away") -> dict:
        """Scrape bookmaker odds from the Flashscore odds-comparison page.

        Tries camoufox (Firefox + anti-fingerprinting) first, then Selenium.

        Args:
            flashscore_id: 8-char Flashscore match ID (e.g. "G8MZEpbl")
            market: URL segment — "home-draw-away", "over-under", "both-teams-to-score"

        Returns:
            Dict mapping selection names to best available odds (float).
            Empty dict if scraping fails.
        """
        # --- camoufox path (primary) ---
        if _CF_AVAILABLE and not self._cf_failed:
            result = self._scrape_odds_page_cf(flashscore_id, market)
            if result:
                return result
            logger.debug(
                f"camoufox odds empty for {flashscore_id} {market} — trying Selenium"
            )

        # --- Selenium fallback ---
        driver = self._get_driver()
        if not driver:
            return {}

        market_path = {
            "home-draw-away": "home-draw-away/full-time",
            "over-under": "over-under/full-time",
            "btts": "both-teams-to-score/full-time",
        }.get(market, market)

        # Try hash-based URL first (standard Flashscore SPA routing),
        # then fall back to the non-hash path in case of a frontend update.
        urls_to_try = [
            f"https://www.flashscore.com/match/{flashscore_id}/#/odds-comparison/{market_path}",
            f"https://www.flashscore.com/match/{flashscore_id}/odds-comparison/{market_path}/",
        ]
        result = {}
        tried_reset = False
        for attempt, url in enumerate(urls_to_try):
            try:
                driver.get(url)
                # Wait for odds values to appear (longer timeout to handle slow JS routing)
                WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "[data-testid='wcl-oddsValue']")
                    )
                )
                time.sleep(1)
                break  # success — stop trying alternative URLs
            except Exception:
                if attempt == 0 and not tried_reset:
                    # First URL failed — reset Chrome session (Cloudflare may have
                    # flagged the long-running session) and retry with fresh driver.
                    tried_reset = True
                    try:
                        self.close_driver()
                    except Exception:
                        pass
                    driver = self._get_driver()
                    if not driver:
                        return {}
                    continue
                elif attempt < len(urls_to_try) - 1:
                    continue  # try next URL format
                else:
                    return result  # all attempts failed

        try:
            spans = driver.find_elements(By.CSS_SELECTOR, "[data-testid='wcl-oddsValue']")
            values = []
            for span in spans:
                try:
                    val = float(span.text.strip())
                    values.append(val)
                except ValueError:
                    pass

            if market == "home-draw-away":
                # Values arrive in groups of 3: Home, Draw, Away per bookmaker row.
                # Take the max across all bookmakers for each position.
                if len(values) >= 3:
                    home_vals, draw_vals, away_vals = [], [], []
                    for i in range(0, len(values) - 2, 3):
                        home_vals.append(values[i])
                        draw_vals.append(values[i + 1])
                        away_vals.append(values[i + 2])
                    result["Home Win"] = max(home_vals)
                    result["Draw"] = max(draw_vals)
                    result["Away Win"] = max(away_vals)

            elif market == "over-under":
                # Over/Under page lists lines like "Over 0.5 / Under 0.5 / Over 1.5 / ..."
                # The row header text tells us the line.  Values come in pairs: Over, Under.
                # We collect labeled rows by looking at the wcl-oddsRow structure.
                result = self._parse_ou_odds(driver)

            elif market == "btts":
                # The BTTS page shows FT, 1H and 2H sections; values arrive as
                # interleaved pairs (Yes, No) across ALL sections.
                # FT BTTS Yes is always < 2.5 (typical 1.65–2.10).
                # 1H BTTS Yes is always >= 2.5 (typical 3.50–5.00).
                # FT BTTS No is the highest No value across all sections.
                # Strategy: filter Yes candidates to < 2.5 to exclude 1H/2H
                # contamination; take max of No directly (FT No is always highest).
                if len(values) >= 2:
                    yes_vals, no_vals = [], []
                    for i in range(0, len(values) - 1, 2):
                        yes_vals.append(values[i])
                        no_vals.append(values[i + 1])
                    ft_yes = [v for v in yes_vals if v < 2.5]
                    if ft_yes:
                        result["BTTS Yes"] = max(ft_yes)
                    elif yes_vals:
                        # Fallback: defensive/low-scoring match where FT Yes > 2.5
                        result["BTTS Yes"] = min(yes_vals)
                    if no_vals:
                        result["BTTS No"] = max(no_vals)

        except Exception as e:
            if "no such window" in str(e).lower() or "target window already closed" in str(e).lower():
                self._driver = None
            logger.debug(f"Odds scrape failed for {flashscore_id} {market}: {e}")

        return result

    def _parse_ou_odds(self, driver) -> dict:
        """Parse Over/Under odds from the loaded odds-comparison page."""
        try:
            soup = BeautifulSoup(driver.page_source, "lxml")
            return self._parse_odds_from_soup(soup, "over-under")
        except Exception as e:
            logger.debug(f"O/U parsing error: {e}")
            return {}

    async def scrape_and_save_odds(self, match_id: int, flashscore_id: str,
                                     markets: tuple = ("home-draw-away", "over-under", "btts")):
        """Scrape Flashscore odds for the specified markets and save to the Odds table.

        Args:
            match_id: Database match ID.
            flashscore_id: 8-char Flashscore match ID.
            markets: Tuple of market slugs to scrape. Defaults to all three.
                     Pass ("home-draw-away",) to only scrape 1X2 (faster).
        """
        from src.data.database import get_db
        from src.data.models import Odds

        db = get_db()

        loop = asyncio.get_event_loop()

        def _scrape_all():
            combined = {}
            for mkt in markets:
                try:
                    combined.update(self._scrape_odds_page(flashscore_id, mkt))
                except Exception as exc:
                    logger.debug(f"Market {mkt} scrape error for {flashscore_id}: {exc}")
            return combined

        all_odds = await loop.run_in_executor(None, _scrape_all)

        if not all_odds:
            logger.debug(f"No odds found for flashscore_id={flashscore_id} (Cloudflare block or no market data)")
            return

        # Market type mapping for storage
        market_for_selection = {
            "Home Win": "1X2", "Draw": "1X2", "Away Win": "1X2",
            "BTTS Yes": "btts", "BTTS No": "btts",
        }
        # Over/Under selections
        for key in all_odds:
            if key.startswith("Over ") or key.startswith("Under "):
                market_for_selection[key] = "over_under"

        with db.get_session() as session:
            # Remove stale Flashscore odds for this match first
            session.query(Odds).filter_by(match_id=match_id, bookmaker="Flashscore").delete()
            for selection, odds_value in all_odds.items():
                if odds_value <= 1.0:
                    continue
                # Filter out non-goals markets scraped by accident (corners, shots, etc.)
                # Standard goal lines go up to 4.5 at most; anything higher is a
                # different market (corners, booking points, etc.) and must be discarded.
                if selection.startswith("Over ") or selection.startswith("Under "):
                    try:
                        line = float(selection.split()[1])
                        if line > 4.5:
                            logger.debug(f"Discarding non-goals line {selection} (line {line} > 4.5)")
                            continue
                    except (IndexError, ValueError):
                        pass
                market_type = market_for_selection.get(selection, "other")
                session.add(Odds(
                    match_id=match_id,
                    bookmaker="Flashscore",
                    market_type=market_type,
                    selection=selection,
                    odds_value=odds_value,
                ))
            session.commit()

        logger.info(
            f"Saved {len(all_odds)} odds for match {match_id} "
            f"(flashscore_id={flashscore_id}): {list(all_odds.keys())}"
        )

    def _get_or_create_team(self, session, team_name: str, league: str) -> Team:
        """Get existing team or create a new one."""
        team = session.query(Team).filter_by(name=team_name).first()
        if not team:
            team = Team(name=team_name, league=league)
            session.add(team)
            session.flush()
        return team

    # Known cross-source name aliases (lowercase canonical form).
    # Maps variant spellings to a shared canonical string so that alias-based
    # pairs compare equal after normalisation.
    # NOTE: only add entries that cannot be resolved by the token/prefix logic
    # below (e.g. completely different abbreviation styles).
    _NAME_ALIASES: dict = {
        # Atletico Madrid (API-Football uses "Ath", Flashscore uses "Atl.")
        "ath madrid": "atletico madrid",
        "atl. madrid": "atletico madrid",
        "atl madrid": "atletico madrid",
        # PSG (API-Football: "Paris SG", Flashscore: "PSG")
        "paris sg": "paris saint-germain",
        "psg": "paris saint-germain",
        # Olympiakos / Olympiacos (different Latin transliterations)
        "olympiakos": "olympiacos piraeus",
        "olympiacos": "olympiacos piraeus",
        "olympiakos piraeus": "olympiacos piraeus",
        # Dinamo Bucharest abbreviations ("Din." prefix)
        "din. bucuresti": "dinamo bucuresti",
        "din bucuresti": "dinamo bucuresti",
    }

    @staticmethod
    def _team_names_similar(name_a: str, name_b: str) -> bool:
        """Return True if two team names likely refer to the same team.

        Handles common differences between data sources:
        - Abbreviations:    "Man City" vs "Manchester City"
        - Short names:      "Oxford" vs "Oxford Utd" / "Oxford United"
        - Prefixes:         "SK Rapid" vs "Rapid Vienna"
        - Suffixes:         "Bradford" vs "Bradford City"
        - Diacritics:       "München" vs "Munich", "Castellón" vs "Castellon"
        - Known aliases:    "Ath Madrid" vs "Atl. Madrid", "PSG" vs "Paris SG"
        - Spelling variants:"Olympiakos" vs "Olympiacos"
        """
        import unicodedata
        from difflib import SequenceMatcher

        if name_a == name_b:
            return True

        def _norm(s: str) -> str:
            s = s.lower().strip()
            # Apply known aliases before further normalisation
            if s in FlashscoreScraper._NAME_ALIASES:
                s = FlashscoreScraper._NAME_ALIASES[s]
            # Strip diacritics: ü→u, ó→o, ą→a, ñ→n …
            s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
            return s

        a, b = _norm(name_a), _norm(name_b)
        if a == b:
            return True

        # Strip common club-type tags that differ between sources
        _STRIP = {"fc", "sc", "sk", "afc", "sfc", "cf", "bk", "fk", "ac", "as", "cd", "ad"}

        def _tokens(n):
            return [t.rstrip(".") for t in n.split() if t.rstrip(".") not in _STRIP]

        ta, tb = _tokens(a), _tokens(b)
        if not ta or not tb:
            return False

        shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

        def _tok_match(t1: str, t2: str) -> bool:
            # Exact prefix match (handles "Man"/"Manchester", "Stockport"/"Stockport County")
            if t2.startswith(t1) or t1.startswith(t2):
                return True
            # Fuzzy match: catches diacritics stripped unevenly, k/c spelling variants
            return SequenceMatcher(None, t1, t2).ratio() >= 0.75

        matches = sum(
            1 for tok in shorter
            if any(_tok_match(tok, long_tok) for long_tok in longer)
        )
        return matches / len(shorter) >= 0.7

    def _save_match(self, session, match_data: dict, league: str, is_fixture: bool):
        """Save a match to the database, avoiding duplicates.

        Deduplication strategy (in order of priority):
        1. Exact match: same home/away team IDs + exact timestamp.
        2. Flashscore-ID match: any existing record with the same flashscore_id.
        3. Fuzzy match: same league, same is_fixture flag, match time within
           ±4 hours, and home/away team names pass _team_names_similar().
           This catches UTC-vs-local-time duplicates (±2 h) and abbreviated
           team names from different data sources.
        """
        home_team = self._get_or_create_team(session, match_data["home_team"], league)
        away_team = self._get_or_create_team(session, match_data["away_team"], league)
        match_date = match_data["match_date"]

        def _apply_update(existing):
            """Sync mutable fields onto an existing Match record."""
            if existing.is_fixture and not is_fixture and "home_goals" in match_data:
                existing.home_goals = match_data.get("home_goals")
                existing.away_goals = match_data.get("away_goals")
                existing.is_fixture = False
            if not existing.flashscore_id and match_data.get("flashscore_id"):
                existing.flashscore_id = match_data["flashscore_id"]
            return existing

        # 1. Exact match
        existing = session.query(Match).filter_by(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_date,
        ).first()
        if existing:
            return _apply_update(existing)

        # 2. Flashscore-ID match (data came from Flashscore with a known ID)
        if match_data.get("flashscore_id"):
            existing = session.query(Match).filter_by(
                flashscore_id=match_data["flashscore_id"]
            ).first()
            if existing:
                return _apply_update(existing)

        # 3. Fuzzy match: ±4 h window + similar team names.
        #    ±4 h catches UTC vs UTC+2 local-time discrepancies (2 h) with margin.
        #    We intentionally do NOT filter by is_fixture so that Flashscore results
        #    can update existing fixture records from other sources (API-Football, FDO).
        window = timedelta(hours=4)
        candidates = session.query(Match).filter(
            Match.league == league,
            Match.match_date >= match_date - window,
            Match.match_date <= match_date + window,
        ).all()

        for candidate in candidates:
            c_home = session.get(Team, candidate.home_team_id)
            c_away = session.get(Team, candidate.away_team_id)
            if c_home and c_away:
                if (self._team_names_similar(c_home.name, match_data["home_team"]) and
                        self._team_names_similar(c_away.name, match_data["away_team"])):
                    logger.debug(
                        f"Fuzzy-merged '{match_data['home_team']} vs {match_data['away_team']}'"
                        f" into existing id={candidate.id}"
                        f" ('{c_home.name} vs {c_away.name}', Δt="
                        f"{abs((match_date - candidate.match_date).total_seconds()/3600):.1f}h)"
                    )
                    return _apply_update(candidate)

        # 4. No match found — create a new record
        match = Match(
            home_team_id=home_team.id,
            away_team_id=away_team.id,
            match_date=match_date,
            league=league,
            home_goals=match_data.get("home_goals"),
            away_goals=match_data.get("away_goals"),
            is_fixture=is_fixture,
            flashscore_id=match_data.get("flashscore_id"),
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
