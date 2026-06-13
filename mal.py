"""MAL API client."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from rich.progress import Progress

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RateLimitedSession:
    def __init__(self, per_second: float = 1.0):
        self.per_second = per_second
        self._session = aiohttp.ClientSession()
        self._lock = asyncio.Lock()
        self._last_request = 0.0

    async def get(self, *args, **kwargs):
        async with self._lock:
            now = asyncio.get_running_loop().time()
            if self._last_request:
                delay = (1.0 / self.per_second) - (now - self._last_request)
                if delay > 0:
                    logger.debug("Rate limit: sleeping %.3fs", delay)
                    await asyncio.sleep(delay)
            self._last_request = asyncio.get_running_loop().time()
        return await self._session.get(*args, **kwargs)

    async def close(self):
        logger.debug("Closing MAL HTTP session")
        await self._session.close()


@dataclass(slots=True)
class MALAnimeEntry:
    mal_id: int
    title: str
    alternative_titles: dict[str, Any] | None = None
    related_anime: list[dict[str, Any]] | None = None
    status: str | None = None
    score: float | None = None
    num_episodes_watched: int | None = None
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class MALClient:
    client_id: str
    username: str
    access_token: str | None = None
    base_url: str = "https://api.myanimelist.net/v2"
    per_second: float = 1.0
    session: RateLimitedSession = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.session = RateLimitedSession(per_second=self.per_second)
        logger.debug(
            "MALClient initialized — user=%r  rate_limit=%.2f/s  authenticated=%s",
            self.username,
            self.per_second,
            bool(self.access_token),
        )

    def _headers(self) -> dict[str, str]:
        headers = {"X-MAL-CLIENT-ID": self.client_id}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def get_anime_details(self, anime_id: int) -> dict[str, Any]:
        logger.debug("Fetching anime details for MAL ID %d", anime_id)
        url = f"{self.base_url}/anime/{anime_id}"
        return await self._get_json(
            url,
            params={"fields": "title,alternative_titles,related_anime"},
        )

    async def _get_json(
        self,
        url: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        logger.debug("MAL GET %s params=%s", url, params)

        last_exc = None

        for attempt in range(6):  # you can raise this if you want even more persistence
            try:
                resp = await self.session.get(
                    url,
                    headers=self._headers(),
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=30),
                )

                # -------------------------
                # RETRYABLE HTTP RESPONSES
                # -------------------------
                if resp.status in {429, 500, 502, 503, 504}:
                    text = await resp.text()

                    # scale delay with your rate limit
                    base_delay = max(1.5, 6.0 / self.per_second)

                    # exponential backoff (NO CAP)
                    delay = base_delay * (3**attempt)

                    # jitter (prevents sync retry waves)
                    delay += random.uniform(base_delay * 0.5, base_delay * 1.5)

                    logger.warning(
                        "MAL retryable HTTP %d for %s (attempt %d/6), sleeping %.2fs",
                        resp.status,
                        url,
                        attempt + 1,
                        delay,
                    )

                    await asyncio.sleep(delay)
                    last_exc = RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    continue

                # -------------------------
                # FATAL ERRORS (NO RETRY)
                # -------------------------
                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(
                        "MAL request failed (%d) for %s: %s",
                        resp.status,
                        url,
                        text[:500],
                    )
                    raise RuntimeError(f"MAL request failed ({resp.status})")

                return json.loads(await resp.text())

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                base_delay = max(1.5, 6.0 / self.per_second)
                delay = base_delay * (3**attempt)
                delay += random.uniform(base_delay * 0.5, base_delay * 1.5)

                logger.warning(
                    "MAL network error %s (attempt %d/6), sleeping %.2fs",
                    exc,
                    attempt + 1,
                    delay,
                )

                await asyncio.sleep(delay)
                last_exc = exc

        raise RuntimeError(f"MAL request failed after retries") from last_exc

    async def list_user_anime(
        self,
        status: str | None = None,
        *,
        progress: Progress,
    ) -> list[MALAnimeEntry]:
        """Fetch the user's anime list.

        Args:
            status: Optional MAL watch-status filter.
            progress: A *started* Rich Progress instance owned by the caller.
                      A task will be added and advanced; the caller retains
                      ownership and must not stop the Progress here.
        """
        url = f"{self.base_url}/users/{self.username}/animelist"
        params: dict[str, object] = {
            "fields": "list_status,alternative_titles",
            "limit": 1000,
        }
        if status:
            params["status"] = status

        logger.info(
            "Fetching MAL anime list for user=%r status=%s", self.username, status
        )

        entries: list[MALAnimeEntry] = []
        offset = 0
        task = progress.add_task("MAL", total=None)
        page = 0

        while True:
            payload = await self._get_json(url, {**params, "offset": offset})
            page += 1
            page_entries = payload.get("data", [])
            logger.debug(
                "MAL list page %d (offset=%d) — %d entry(ies)",
                page,
                offset,
                len(page_entries),
            )

            for item in page_entries:
                node = item.get("node", {})
                list_status = item.get("list_status", {})
                entries.append(
                    MALAnimeEntry(
                        mal_id=int(node.get("id")),
                        title=str(node.get("title") or ""),
                        alternative_titles=node.get("alternative_titles"),
                        status=list_status.get("status"),
                        score=(
                            float(list_status["score"])
                            if list_status.get("score") is not None
                            else None
                        ),
                        num_episodes_watched=(
                            int(list_status["num_episodes_watched"])
                            if list_status.get("num_episodes_watched") is not None
                            else None
                        ),
                        raw=item,
                    )
                )
                progress.advance(task)

            paging = payload.get("paging", {})
            if not paging.get("next"):
                break
            offset += int(params["limit"])

        progress.update(task, total=len(entries))
        logger.info(
            "Fetched %d MAL anime entry(ies) for user=%r", len(entries), self.username
        )
        return entries

    async def close(self) -> None:
        await self.session.close()
