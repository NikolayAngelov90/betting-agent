"""Base scraper with shared functionality."""

import asyncio
import random
import time
from abc import ABC, abstractmethod

import aiohttp

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger()


class CircuitBreaker:
    """Simple circuit breaker to avoid hammering failing APIs.

    States:
        CLOSED  — requests pass through normally
        OPEN    — requests are immediately rejected (API assumed down)
        HALF_OPEN — one probe request allowed to test recovery
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0,
                 name: str = ""):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name or "circuit"
        self._state = self.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - self._last_failure_time >= self.recovery_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def record_success(self):
        self._failure_count = 0
        if self._state == self.HALF_OPEN:
            logger.info(f"Circuit breaker [{self.name}] recovered → CLOSED")
        self._state = self.CLOSED

    def record_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.monotonic()
        if self._failure_count >= self.failure_threshold:
            self._state = self.OPEN
            logger.warning(
                f"Circuit breaker [{self.name}] OPEN after {self._failure_count} failures "
                f"(recovery in {self.recovery_timeout}s)"
            )

    def allow_request(self) -> bool:
        state = self.state
        if state == self.CLOSED:
            return True
        if state == self.HALF_OPEN:
            return True  # allow one probe
        return False


class BaseScraper(ABC):
    """Base class for all scrapers with rate limiting, retry, and circuit breaker."""

    # Default retry settings (can be overridden per subclass)
    MAX_RETRIES = 3
    RETRY_BASE_DELAY = 2.0   # seconds, doubles each retry
    RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

    def __init__(self, config=None):
        self.config = config or get_config()
        self.delay = self.config.get("scraping.request_delay", 3)
        self._session = None
        self._circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=120.0,
            name=self.__class__.__name__,
        )

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

    async def fetch(self, url: str, params: dict = None, headers: dict = None) -> str:
        """Fetch a URL with rate limiting, retry, and circuit breaker.

        Returns:
            Response text content.
        """
        return await self._request(url, params=params, headers=headers, as_json=False)

    async def fetch_json(self, url: str, params: dict = None, headers: dict = None) -> dict:
        """Fetch a URL and parse JSON response with retry and circuit breaker."""
        return await self._request(url, params=params, headers=headers, as_json=True)

    async def _request(self, url: str, *, params: dict = None, headers: dict = None,
                       as_json: bool = False):
        """Internal request method with retry + exponential backoff + circuit breaker."""
        if not self._circuit.allow_request():
            logger.warning(f"Circuit OPEN for {self.__class__.__name__}, skipping {url}")
            raise ConnectionError(
                f"Circuit breaker open for {self.__class__.__name__} — API assumed down"
            )

        last_exc = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            await self._rate_limit()
            session = await self._get_session()

            try:
                async with session.get(
                    url, params=params, headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    # Retry on transient HTTP errors
                    if resp.status in self.RETRY_STATUS_CODES and attempt < self.MAX_RETRIES:
                        retry_after = float(resp.headers.get("Retry-After", 0))
                        backoff = max(
                            retry_after,
                            self.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
                        )
                        logger.warning(
                            f"HTTP {resp.status} from {url} — retry {attempt}/{self.MAX_RETRIES} "
                            f"in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                        continue

                    resp.raise_for_status()

                    self._circuit.record_success()

                    if as_json:
                        data = await resp.json()
                        logger.debug(f"Fetched JSON from {url} — status {resp.status}")
                        return data
                    else:
                        text = await resp.text()
                        logger.debug(f"Fetched {url} — status {resp.status}")
                        return text

            except aiohttp.ClientResponseError as e:
                # Non-retryable HTTP errors (auth, not found, etc.) — fail immediately
                if e.status not in self.RETRY_STATUS_CODES:
                    self._circuit.record_failure()
                    logger.error(f"HTTP {e.status} from {url}: {e.message}")
                    raise
                last_exc = e
                if attempt < self.MAX_RETRIES:
                    backoff = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Request to {url} failed ({e}) — retry {attempt}/{self.MAX_RETRIES} "
                        f"in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                else:
                    self._circuit.record_failure()
                    logger.error(f"Request to {url} failed after {self.MAX_RETRIES} retries: {e}")
                    raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                if attempt < self.MAX_RETRIES:
                    backoff = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"Request to {url} failed ({e}) — retry {attempt}/{self.MAX_RETRIES} "
                        f"in {backoff:.1f}s"
                    )
                    await asyncio.sleep(backoff)
                else:
                    self._circuit.record_failure()
                    logger.error(f"Request to {url} failed after {self.MAX_RETRIES} retries: {e}")
                    raise

        # Should not reach here, but safety net
        self._circuit.record_failure()
        raise last_exc or RuntimeError(f"Request to {url} failed")

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    @abstractmethod
    async def update(self):
        """Run the scraper update cycle."""
        pass
