from __future__ import annotations

import asyncio
import logging
import random
import threading
import time

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global Default Configuration Budgets
# ---------------------------------------------------------------------------
DEFAULTS = {
    "MAL": {"per_second": 1.0, "burst": 1, "jitter_min": 0.0, "jitter_max": 0.0},
    "AniList": {"per_second": 1.0, "burst": 1, "jitter_min": 0.0, "jitter_max": 0.0},
    "Spotify": {"per_second": 2.0, "burst": 1, "jitter_min": 0.0, "jitter_max": 0.0},
    "Algolia": {"per_second": 0.25, "burst": 1, "jitter_min": 0.1, "jitter_max": 1.0},
    "AniPlaylist": {
        "per_second": 0.5,
        "burst": 1,
        "jitter_min": 0.0,
        "jitter_max": 0.0,
    },
}


# ---------------------------------------------------------------------------
# Unified Rate Limiter Class
# ---------------------------------------------------------------------------
class RateLimiter:
    """Unified Thread-safe and Async-safe token-bucket rate limiter.

    Uses a single threading.Lock and time.monotonic() to accurately control
    execution speed across both synchronous threads and asynchronous loops.
    """

    def __init__(
        self,
        per_second: float = 1.0,
        *,
        name: str = "unnamed",
        burst: int = 1,
        jitter_min: float = 0.0,
        jitter_max: float = 0.0,
    ) -> None:
        if per_second <= 0:
            raise ValueError(f"per_second must be > 0, got {per_second!r}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst!r}")

        self.per_second = per_second
        self.name = name
        self.burst = burst
        self.jitter_min = jitter_min
        self.jitter_max = jitter_max
        self._min_interval = 1.0 / per_second

        self._lock = threading.Lock()
        self._last_release: float = 0.0
        self._burst_remaining: int = burst

        logger.info(
            "RateLimiter[%s] created — %.3f req/s  interval=%.3fs  burst=%d  jitter=[%.3f, %.3f]",
            name,
            per_second,
            self._min_interval,
            burst,
            jitter_min,
            jitter_max,
        )

    @classmethod
    def from_preset(cls, name: str) -> RateLimiter:
        """Convenience constructor to build a limiter using predefined target defaults."""
        config = DEFAULTS.get(
            name, {"per_second": 1.0, "burst": 1, "jitter_min": 0.0, "jitter_max": 0.0}
        )
        return cls(name=name, **config)

    def _compute_wait(self) -> float:
        """Internal helper to calculate wait time and safely update timestamps."""
        with self._lock:
            if self._burst_remaining > 0:
                self._burst_remaining -= 1
                self._last_release = time.monotonic()
                return 0.0

            now = time.monotonic()
            wait = self._min_interval - (now - self._last_release)
            if self.jitter_max > 0:
                wait += random.uniform(self.jitter_min, self.jitter_max)

            if wait > 0:
                self._last_release = now + wait
            else:
                self._last_release = now
                wait = 0.0
            return wait

    def acquire(self) -> None:
        """Block the current thread until a request slot is available (Synchronous)."""
        wait = self._compute_wait()
        if wait > 0:
            logger.debug("RateLimiter[%s] sync sleeping %.3fs", self.name, wait)
            time.sleep(wait)

    async def acquire_async(self) -> None:
        """Yield control to the event loop until a slot is available (Asynchronous)."""
        wait = self._compute_wait()
        if wait > 0:
            logger.debug("RateLimiter[%s] async sleeping %.3fs", self.name, wait)
            await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # aiohttp Session Wrappers
    # ------------------------------------------------------------------
    async def get(self, session: object, *args: object, **kwargs: object) -> object:
        """Acquire an async slot then call ``session.get(*args, **kwargs)``."""
        await self.acquire_async()
        return await session.get(*args, **kwargs)  # type: ignore[attr-defined]

    async def post(self, session: object, *args: object, **kwargs: object) -> object:
        """Acquire an async slot then call ``session.post(*args, **kwargs)``."""
        await self.acquire_async()
        return await session.post(*args, **kwargs)  # type: ignore[attr-defined]
