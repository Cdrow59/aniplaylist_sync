"""AniList API client (GraphQL)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from ratelimit import (
    ANILIST_DEFAULT_BURST,
    ANILIST_DEFAULT_JITTER_MAX,
    ANILIST_DEFAULT_JITTER_MIN,
    ANILIST_DEFAULT_RPS,
    RateLimiter,
)

logger = logging.getLogger(__name__)

ANILIST_API = "https://graphql.anilist.co"
ANILIST_AUTH_URL = "https://anilist.co/api/v2/oauth/token"

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class RateLimitedSession:
    """aiohttp session wrapped with a :class:`~ratelimit.RateLimiter`."""

    def __init__(
        self,
        per_second: float = ANILIST_DEFAULT_RPS,
        burst: int = ANILIST_DEFAULT_BURST,
        jitter_min: float = ANILIST_DEFAULT_JITTER_MIN,
        jitter_max: float = ANILIST_DEFAULT_JITTER_MAX,
    ) -> None:
        self._session = aiohttp.ClientSession()
        self._limiter = RateLimiter(
            per_second=per_second,
            name="AniList",
            burst=burst,
            jitter_min=jitter_min,
            jitter_max=jitter_max,
        )

    async def post(self, *args: object, **kwargs: object) -> object:
        await self._limiter.acquire()
        return await self._session.post(*args, **kwargs)  # type: ignore[arg-type]

    async def close(self) -> None:
        logger.debug("Closing AniList HTTP session")
        await self._session.close()


@dataclass
class AniListAnimeEntry:
    anilist_id: int
    title: str
    native_title: str | None = None
    english_title: str | None = None
    synonyms: list[str] | None = None
    status: str | None = None
    score: float | None = None
    num_episodes_watched: int | None = None
    raw: dict[str, Any] | None = None

    @property
    def mal_id(self) -> int:
        return self.anilist_id

    @property
    def alternative_titles(self) -> dict[str, object]:
        alt: dict[str, object] = {}
        if self.english_title:
            alt["en"] = self.english_title
        if self.native_title:
            alt["ja"] = self.native_title
        if self.synonyms:
            alt["synonyms"] = self.synonyms
        return alt


@dataclass(slots=True)
class AniListClient:
    username: str
    client_id: str | None = None
    client_secret: str | None = None
    redirect_uri: str | None = None
    api_url: str = ANILIST_API
    auth_url: str = ANILIST_AUTH_URL
    per_second: float = 1.0
    session: RateLimitedSession | None = field(init=False, repr=False, default=None)
    _token: str | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        logger.debug(
            "AniListClient initialized — user=%r  rate_limit=%.2f/s  credentials=%s",
            self.username,
            self.per_second,
            bool(self.client_id and self.client_secret),
        )

    @classmethod
    def from_env(cls, username: str | None = None, **kwargs) -> "AniListClient":
        """Create an AniListClient from environment variables.

        Reads ``ANILIST_CLIENT_ID``, ``ANILIST_CLIENT_SECRET``,
        ``ANILIST_USERNAME`` (overridden by *username*), and optionally
        ``ANILIST_REDIRECT_URI`` / ``ANILIST_RATE_LIMIT_PER_SECOND``.
        Raises ``RuntimeError`` if required variables are missing.
        """
        client_id = os.getenv("ANILIST_CLIENT_ID", "").strip()
        client_secret = os.getenv("ANILIST_CLIENT_SECRET", "").strip()
        missing = [
            n
            for n, v in [
                ("ANILIST_CLIENT_ID", client_id),
                ("ANILIST_CLIENT_SECRET", client_secret),
            ]
            if not v
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )
        resolved_username = username or os.getenv("ANILIST_USERNAME", "").strip() or ""
        if not resolved_username:
            raise RuntimeError(
                "AniList username required — pass username= or set ANILIST_USERNAME"
            )
        redirect_uri = os.getenv("ANILIST_REDIRECT_URI") or None
        per_second_raw = os.getenv("ANILIST_RATE_LIMIT_PER_SECOND")
        per_second = float(per_second_raw) if per_second_raw else 1.0
        return cls(
            username=resolved_username,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            per_second=per_second,
            **kwargs,
        )

    def _get_session(self) -> RateLimitedSession:
        if self.session is None:
            self.session = RateLimitedSession(per_second=self.per_second)
        return self.session

    async def _ensure_token(self) -> str:
        if self._token:
            return self._token
        if not self.client_id or not self.client_secret:
            raise RuntimeError("AniList client_id and client_secret are required")
        logger.debug("Requesting AniList OAuth2 client credentials token")
        payload = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        resp = await self._get_session().post(
            self.auth_url,
            data=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        )
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(
                f"AniList token request failed ({resp.status}): {text[:200]}"
            )
        data = json.loads(await resp.text())
        self._token = data["access_token"]
        logger.debug("AniList OAuth2 token acquired")
        return self._token

    async def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        token = await self._ensure_token()
        headers["Authorization"] = f"Bearer {token}"
        return headers

    async def _graphql(
        self,
        query: str,
        variables: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, object] = {"query": query}
        if variables:
            payload["variables"] = variables

        body = json.dumps(payload, ensure_ascii=False)
        logger.debug("AniList GraphQL query=%.200s variables=%s", query, variables)

        last_exc = None

        for attempt in range(6):
            try:
                resp = await self._get_session().post(
                    self.api_url,
                    headers=await self._headers(),
                    data=body,
                    timeout=aiohttp.ClientTimeout(total=30),
                )

                if resp.status in {429, 500, 502, 503, 504}:
                    text = await resp.text()
                    base_delay = max(1.5, 6.0 / self.per_second)
                    delay = base_delay * (3**attempt)
                    delay += random.uniform(base_delay * 0.5, base_delay * 1.5)
                    logger.warning(
                        "AniList retryable HTTP %d (attempt %d/6), sleeping %.2fs",
                        resp.status,
                        attempt + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    last_exc = RuntimeError(f"HTTP {resp.status}: {text[:200]}")
                    continue

                if resp.status >= 400:
                    text = await resp.text()
                    logger.error(
                        "AniList request failed (%d): %s",
                        resp.status,
                        text[:500],
                    )
                    raise RuntimeError(f"AniList request failed ({resp.status})")

                data: dict[str, Any] = json.loads(await resp.text())

                if "errors" in data:
                    errors = data["errors"]
                    msg = "; ".join(e.get("message", str(e)) for e in errors)
                    logger.warning("AniList GraphQL errors: %s", msg)

                    status_codes = {e.get("status") for e in errors if e.get("status")}
                    if status_codes & {429, 500, 502, 503, 504}:
                        base_delay = max(1.5, 6.0 / self.per_second)
                        delay = base_delay * (3**attempt)
                        delay += random.uniform(base_delay * 0.5, base_delay * 1.5)
                        logger.warning(
                            "AniList retryable GraphQL error (attempt %d/6), sleeping %.2fs",
                            attempt + 1,
                            delay,
                        )
                        await asyncio.sleep(delay)
                        last_exc = RuntimeError(f"GraphQL errors: {msg}")
                        continue

                    raise RuntimeError(f"AniList GraphQL error: {msg}")

                return data

            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                base_delay = max(1.5, 6.0 / self.per_second)
                delay = base_delay * (3**attempt)
                delay += random.uniform(base_delay * 0.5, base_delay * 1.5)
                logger.warning(
                    "AniList network error %s (attempt %d/6), sleeping %.2fs",
                    exc,
                    attempt + 1,
                    delay,
                )
                await asyncio.sleep(delay)
                last_exc = exc

        raise RuntimeError("AniList request failed after retries") from last_exc

    async def get_anime_details(
        self,
        anime_id: int,
    ) -> dict[str, Any]:
        logger.debug("Fetching anime details for AniList ID %d", anime_id)
        query = """
        query ($id: Int) {
            Media(id: $id, type: ANIME) {
                id
                title {
                    romaji
                    english
                    native
                }
                synonyms
                relations {
                    edges {
                        node { id }
                        relationType
                    }
                }
            }
        }
        """
        result = await self._graphql(query, {"id": anime_id})
        media: dict[str, Any] | None = result.get("data", {}).get("Media")
        if media is None:
            raise RuntimeError(f"AniList anime {anime_id} not found")

        title_data = media.get("title", {}) or {}

        alternative_titles: dict[str, Any] = {}
        if title_data.get("english"):
            alternative_titles["en"] = title_data["english"]
        if title_data.get("native"):
            alternative_titles["ja"] = title_data["native"]
        if media.get("synonyms"):
            alternative_titles["synonyms"] = media["synonyms"]

        related_anime: list[dict[str, Any]] = []
        for edge in (media.get("relations") or {}).get("edges") or []:
            node = edge.get("node") or {}
            relation_type = edge.get("relationType", "")
            related_anime.append(
                {"node": {"id": int(node["id"])}, "relation_type": relation_type}
            )

        return {
            "id": int(media["id"]),
            "title": str(title_data.get("romaji") or ""),
            "alternative_titles": alternative_titles,
            "related_anime": related_anime,
        }

    async def list_user_anime(
        self,
        status: str | None = None,
        *,
        progress: Any = None,
    ) -> list[AniListAnimeEntry]:
        logger.info(
            "Fetching AniList anime list for user=%r status=%s",
            self.username,
            status,
        )

        query = """
        query ($userName: String, $status: MediaListStatus, $page: Int) {
            Page(page: $page, perPage: 50) {
                pageInfo { hasNextPage }
                mediaList(userName: $userName, status: $status, type: ANIME) {
                    media {
                        id
                        title {
                            romaji
                            english
                            native
                        }
                        synonyms
                    }
                    status
                    score
                    progress
                }
            }
        }
        """

        entries: list[AniListAnimeEntry] = []
        page = 1
        has_next = True

        task = progress.add_task("AniList", total=None) if progress else None

        variables: dict[str, object] = {"userName": self.username, "page": page}
        if status:
            variables["status"] = status

        while has_next:
            variables["page"] = page
            result = await self._graphql(query, variables)
            page_data = result.get("data", {}).get("Page", {})
            page_info = page_data.get("pageInfo", {})
            has_next = bool(page_info.get("hasNextPage"))

            for item in page_data.get("mediaList", []):
                media = item.get("media", {}) or {}
                title_data = media.get("title", {}) or {}
                entries.append(
                    AniListAnimeEntry(
                        anilist_id=int(media.get("id")),
                        title=str(title_data.get("romaji") or ""),
                        native_title=title_data.get("native"),
                        english_title=title_data.get("english"),
                        synonyms=media.get("synonyms"),
                        status=item.get("status"),
                        score=(
                            float(item["score"])
                            if item.get("score") is not None
                            else None
                        ),
                        num_episodes_watched=(
                            int(item["progress"])
                            if item.get("progress") is not None
                            else None
                        ),
                        raw=item,
                    )
                )
                if task is not None:
                    progress.advance(task)

            page += 1

        if task is not None:
            progress.update(task, total=len(entries))

        logger.info(
            "Fetched %d AniList anime entry(ies) for user=%r",
            len(entries),
            self.username,
        )
        return entries

    async def close(self) -> None:
        if self.session is not None:
            await self.session.close()
