"""Base scraper with shared functionality."""

import asyncio
import random
from abc import ABC, abstractmethod

import aiohttp

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


class BaseScraper(ABC):
    """Base class for all scrapers with rate limiting and error handling."""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.delay = self.config.get("scraping.request_delay", 3)
        self._session = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }
            self._session = aiohttp.ClientSession(headers=headers)
        return self._session

    async def _rate_limit(self):
        """Apply rate limiting with slight randomization."""
        jitter = random.uniform(0.5, 1.5)
        await asyncio.sleep(self.delay * jitter)

    async def fetch(self, url: str, params: dict = None) -> str:
        """Fetch a URL with rate limiting and error handling.

        Returns:
            Response text content.
        """
        await self._rate_limit()
        session = await self._get_session()

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                text = await resp.text()
                logger.debug(f"Fetched {url} — status {resp.status}")
                return text
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise

    async def fetch_json(self, url: str, params: dict = None) -> dict:
        """Fetch a URL and parse JSON response."""
        await self._rate_limit()
        session = await self._get_session()

        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                resp.raise_for_status()
                data = await resp.json()
                logger.debug(f"Fetched JSON from {url} — status {resp.status}")
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch JSON from {url}: {e}")
            raise

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def update(self):
        """Run the scraper update cycle."""
        pass
