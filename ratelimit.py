"""ratelimit.py — shared async/sync rate-limiter for all API clients.

Provides:
    RateLimiter        — async token-bucket for aiohttp sessions (MAL, AniList)
    SyncRateLimiter    — sync token-bucket for spotipy (Spotify)
    AniPlaylistLimiter — inter-scrape delay guard for Playwright scraper

Usage
-----
    # Async (MAL, AniList)
    limiter = RateLimiter(per_second=1.0, name="MAL")
    await limiter.acquire()          # blocks until a slot is available
    resp = await session.get(url)

    # Sync (Spotify)
    limiter = SyncRateLimiter(per_second=5.0, name="Spotify")
    limiter.acquire()
    result = spotify_client.search(...)

    # AniPlaylist scrape throttle
    limiter = AniPlaylistLimiter(per_scrape=0.5)   # 0.5 scrapes/s = 2 s gap
    await limiter.acquire()
    result = await scrape(query)

Rate-limit defaults (can be overridden via env vars or constructor args):
    MAL         1 req/s   — conservative; MAL enforces ~3 req/s but docs say 1
    AniList     1 req/s   — public limit is ~90 req/min with 429 back-off
    Spotify     5 req/s   — spotipy handles 429s but we stay below the ceiling
    AniPlaylist 0.5 scrapes/s (= 2 s between full scrapes) — Playwright overhead
                           already pads naturally; this is a safety floor
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

logger = logging.getLogger(__name__)

# Default rates — edit here to tune globally, or override per-client in main.py
MAL_DEFAULT_RPS: float = 1.0  # MAL docs: ~3 req/s; 1.0 is conservative
ANILIST_DEFAULT_RPS: float = 1.0  # AniList: ~90 req/min; 1.0 ≈ 60 req/min
SPOTIFY_DEFAULT_RPS: float = 5.0  # spotipy handles 429s; 5.0 stays well under
ANIPLAYLIST_DEFAULT_RPS: float = 0.5  # 2 s floor between scrapes

# Default burst budgets — number of requests allowed with no delay at startup
MAL_DEFAULT_BURST: int = 1
ANILIST_DEFAULT_BURST: int = 1
SPOTIFY_DEFAULT_BURST: int = 1
ANIPLAYLIST_DEFAULT_BURST: int = 1


# ---------------------------------------------------------------------------
# Async rate limiter (MAL, AniList)
# ---------------------------------------------------------------------------


class RateLimiter:
    """Async token-bucket rate limiter.

    A single asyncio.Lock serialises callers so no two requests can slip
    through simultaneously even under high concurrency.  The delay is
    computed from the *actual* time of the last release, not the scheduled
    time, so accumulated slack is never gifted to the next caller.

    Args:
        per_second: Maximum sustained request rate (requests / second).
                    Fractional values are fine: 0.5 → one request every 2 s.
        name:       Label included in debug log lines.
        burst:      Number of requests allowed without delay at startup
                    (default 1 — no burst).  Useful when an API allows a
                    small initial burst before the steady-state window kicks in.
    """

    def __init__(
        self,
        per_second: float = 1.0,
        *,
        name: str = "unnamed",
        burst: int = 1,
    ) -> None:
        if per_second <= 0:
            raise ValueError(f"per_second must be > 0, got {per_second!r}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst!r}")

        self.per_second = per_second
        self.name = name
        self.burst = burst
        self._min_interval = 1.0 / per_second
        self._lock = asyncio.Lock()
        self._last_release: float = 0.0
        self._burst_remaining: int = burst

        logger.info(
            "RateLimiter[%s] created — %.3f req/s  interval=%.3fs  burst=%d",
            name,
            per_second,
            self._min_interval,
            burst,
        )

    async def acquire(self) -> None:
        """Wait until a request slot is available, then return."""
        async with self._lock:
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                self._last_release = asyncio.get_running_loop().time()
                logger.debug(
                    "RateLimiter[%s] burst slot used (%d remaining)",
                    self.name,
                    self._burst_remaining,
                )
                return

            now = asyncio.get_running_loop().time()
            wait = self._min_interval - (now - self._last_release)
            if wait > 0:
                logger.debug("RateLimiter[%s] sleeping %.3fs", self.name, wait)
                await asyncio.sleep(wait)
            self._last_release = asyncio.get_running_loop().time()

    # ------------------------------------------------------------------
    # Convenience: wrap an aiohttp session method
    # ------------------------------------------------------------------

    async def get(self, session: object, *args: object, **kwargs: object) -> object:
        """Acquire a slot then call ``session.get(*args, **kwargs)``."""
        await self.acquire()
        return await session.get(*args, **kwargs)  # type: ignore[attr-defined]

    async def post(self, session: object, *args: object, **kwargs: object) -> object:
        """Acquire a slot then call ``session.post(*args, **kwargs)``."""
        await self.acquire()
        return await session.post(*args, **kwargs)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Sync rate limiter (Spotify / spotipy)
# ---------------------------------------------------------------------------


class SyncRateLimiter:
    """Thread-safe synchronous token-bucket rate limiter for spotipy.

    Identical semantics to :class:`RateLimiter` but uses ``threading.Lock``
    and ``time.monotonic`` so it works outside an async event loop.

    Args:
        per_second: Maximum sustained request rate.
        name:       Label for log lines.
        burst:      Burst budget (see :class:`RateLimiter`).
    """

    def __init__(
        self,
        per_second: float = 5.0,
        *,
        name: str = "unnamed-sync",
        burst: int = 1,
    ) -> None:
        if per_second <= 0:
            raise ValueError(f"per_second must be > 0, got {per_second!r}")

        self.per_second = per_second
        self.name = name
        self.burst = burst
        self._min_interval = 1.0 / per_second
        self._lock = threading.Lock()
        self._last_release: float = 0.0
        self._burst_remaining: int = burst

        logger.info(
            "SyncRateLimiter[%s] created — %.3f req/s  interval=%.3fs  burst=%d",
            name,
            per_second,
            self._min_interval,
            burst,
        )

    def acquire(self) -> None:
        """Block until a request slot is available."""
        with self._lock:
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                self._last_release = time.monotonic()
                return

            now = time.monotonic()
            wait = self._min_interval - (now - self._last_release)
            if wait > 0:
                logger.debug("SyncRateLimiter[%s] sleeping %.3fs", self.name, wait)
                time.sleep(wait)
            self._last_release = time.monotonic()


# ---------------------------------------------------------------------------
# AniPlaylist scrape throttle
# ---------------------------------------------------------------------------


class AniPlaylistLimiter:
    """Async rate limiter for AniPlaylist *scrape calls* (not HTTP requests).

    AniPlaylist is scraped via Playwright, so there are no HTTP sessions to
    throttle directly.  This limiter sits one level up and enforces a minimum
    gap between full ``scrape()`` invocations so the site isn't hammered.

    The Playwright overhead (browser launch + page load + scroll) already
    takes several seconds per scrape; this limiter adds an *additional*
    floor when that natural delay is shorter than ``min_gap_s``.

    Args:
        per_scrape: Maximum scrapes per second (default 0.5 → 2 s gap).
                    Set to e.g. 1.0 for faster scraping when experimenting.
        name:       Label for log lines.
    """

    def __init__(
        self,
        per_scrape: float = ANIPLAYLIST_DEFAULT_RPS,
        *,
        name: str = "AniPlaylist",
        burst: int = ANIPLAYLIST_DEFAULT_BURST,
    ) -> None:
        if per_scrape <= 0:
            raise ValueError(f"per_scrape must be > 0, got {per_scrape!r}")
        self._inner = RateLimiter(per_second=per_scrape, name=name, burst=burst)
        logger.info(
            "AniPlaylistLimiter[%s] created — %.3f scrapes/s  interval=%.3fs  burst=%d",
            name,
            per_scrape,
            1.0 / per_scrape,
            burst,
        )

    async def acquire(self) -> None:
        """Wait until the next scrape slot is available."""
        logger.debug("AniPlaylistLimiter[%s] acquiring scrape slot", self.name)
        await self._inner.acquire()
        logger.debug("AniPlaylistLimiter[%s] scrape slot acquired", self.name)

    @property
    def per_scrape(self) -> float:
        return self._inner.per_second

    @property
    def name(self) -> str:
        return self._inner.name
