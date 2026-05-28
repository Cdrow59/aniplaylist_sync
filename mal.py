from __future__ import annotations

import asyncio
import aiohttp
import json
from dataclasses import dataclass, field
from typing import Any

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# ---------------------------------------------------------------------------
# Simple rate-limited session
# ---------------------------------------------------------------------------


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
                    await asyncio.sleep(delay)

            self._last_request = asyncio.get_running_loop().time()

        return await self._session.get(*args, **kwargs)

    async def close(self):
        await self._session.close()


# ---------------------------------------------------------------------------
# MAL models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# MAL client
# ---------------------------------------------------------------------------


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

    def _headers(self) -> dict[str, str]:
        headers = {"X-MAL-CLIENT-ID": self.client_id}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    async def get_anime_details(self, anime_id: int) -> dict[str, Any]:
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
        resp = await self.session.get(
            url,
            headers=self._headers(),
            params=params,
            timeout=aiohttp.ClientTimeout(total=30),
        )

        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"MAL request failed ({resp.status}): {text[:500]}")

        return json.loads(await resp.text())

    async def list_user_anime(self, status: str | None = None) -> list[MALAnimeEntry]:
        user_path = self.username or "@me"

        url = f"{self.base_url}/users/{user_path}/animelist"

        params: dict[str, object] = {
            "fields": "list_status,alternative_titles",
            "limit": 1000,
        }
        if status:
            params["status"] = status

        entries: list[MALAnimeEntry] = []
        offset = 0

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("MAL", total=None)

            while True:
                payload = await self._get_json(
                    url,
                    {**params, "offset": offset},
                )

                for item in payload.get("data", []):
                    node = item.get("node", {})
                    status = item.get("list_status", {})

                    entries.append(
                        MALAnimeEntry(
                            mal_id=int(node.get("id")),
                            title=str(node.get("title") or ""),
                            alternative_titles=node.get("alternative_titles"),
                            status=status.get("status"),
                            score=(
                                float(status["score"])
                                if status.get("score") is not None
                                else None
                            ),
                            num_episodes_watched=(
                                int(status["num_episodes_watched"])
                                if status.get("num_episodes_watched") is not None
                                else None
                            ),
                            raw=item,
                        )
                    )

                    progress.advance(task)

                paging = payload.get("paging", {})
                next_url = paging.get("next")

                if not next_url:
                    break

                offset += int(params["limit"])

            progress.update(task, total=len(entries))

        return entries

    async def close(self) -> None:
        await self.session.close()


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


async def main():
    client = MALClient(
        client_id="YOUR_CLIENT_ID",
        access_token="YOUR_ACCESS_TOKEN",
        username="@me",
    )

    try:
        anime = await client.list_user_anime()
        print(f"Found {len(anime)} entries")

        for entry in anime[:10]:
            print(entry.title)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
