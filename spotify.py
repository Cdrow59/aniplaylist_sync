"""Spotify playlist creation helpers for AniPlaylist sync."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
import spotipy
from ratelimit import SPOTIFY_DEFAULT_BURST, SPOTIFY_DEFAULT_JITTER_MAX, SPOTIFY_DEFAULT_JITTER_MIN, SPOTIFY_DEFAULT_RPS, SyncRateLimiter
from rich.progress import Progress
from spotipy.oauth2 import SpotifyOAuth

logger = logging.getLogger(__name__)

SPOTIFY_PLAYLIST_LIMIT = 9999
# ---------------------------------------------------------------------------
# ENV
# ---------------------------------------------------------------------------


def _read_spotify_env(name: str) -> str:
    value = os.getenv(name)
    if value and value.strip():
        return value.strip()

    if name.startswith("SPOTIPY_"):
        fallback = name.replace("SPOTIPY_", "SPOTIFY_", 1)
        val = os.getenv(fallback)
        if val and val.strip():
            return val.strip()

    if name.startswith("SPOTIFY_"):
        fallback = name.replace("SPOTIFY_", "SPOTIPY_", 1)
        val = os.getenv(fallback)
        if val and val.strip():
            return val.strip()

    raise RuntimeError(f"Missing required environment variable: {name}")


# ---------------------------------------------------------------------------
# RATE LIMITER
# ---------------------------------------------------------------------------


class RateLimitedSpotifyClient:
    """Wraps a spotipy.Spotify instance with per-second rate limiting.

    Uses :class:`~ratelimit.SyncRateLimiter` internally so all rate-limit
    logic lives in one place.  The actual spotipy calls run in a thread pool
    via ``asyncio.to_thread`` so the event loop is never blocked.
    """

    def __init__(
        self,
        client: Any,
        per_second: float = SPOTIFY_DEFAULT_RPS,
        burst: int = SPOTIFY_DEFAULT_BURST,
        jitter_min: float = SPOTIFY_DEFAULT_JITTER_MIN,
        jitter_max: float = SPOTIFY_DEFAULT_JITTER_MAX,
    ) -> None:
        self._client = client
        self._limiter = SyncRateLimiter(
            per_second=per_second, name="Spotify", burst=burst,
            jitter_min=jitter_min, jitter_max=jitter_max,
        )

    @classmethod
    def from_env(cls, **kwargs) -> "RateLimitedSpotifyClient":
        """Create a RateLimitedSpotifyClient from environment variables.

        Reads ``SPOTIFY_CLIENT_ID`` / ``SPOTIPY_CLIENT_ID``,
        ``SPOTIFY_CLIENT_SECRET`` / ``SPOTIPY_CLIENT_SECRET``, and
        ``SPOTIFY_REDIRECT_URI`` / ``SPOTIPY_REDIRECT_URI``.
        Builds the spotipy OAuth2 auth manager and wraps it.
        Raises ``RuntimeError`` if any required variable is missing.
        """
        auth = SpotifyOAuth(
            client_id=_read_spotify_env("SPOTIPY_CLIENT_ID"),
            client_secret=_read_spotify_env("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=_read_spotify_env("SPOTIPY_REDIRECT_URI"),
            scope="playlist-modify-private playlist-modify-public",
        )
        return cls(spotipy.Spotify(auth_manager=auth), **kwargs)

    async def _call(self, fn, *args, **kwargs):
        # Acquire inside the thread so the event loop isn't blocked by sleep.
        def _run():
            self._limiter.acquire()
            return fn(*args, **kwargs)

        return await asyncio.to_thread(_run)

    async def current_user(self):
        return await self._call(self._client.current_user)

    async def post_playlist(self, payload: dict) -> dict:
        return await self._call(self._client._post, "me/playlists", payload=payload)

    async def playlist_add_items(self, pid: str, uris: list[str]):
        return await self._call(self._client.playlist_add_items, pid, uris)

    async def album_tracks(self, rid: str, limit: int, offset: int) -> dict:
        return await self._call(
            self._client.album_tracks, rid, limit=limit, offset=offset
        )


# ---------------------------------------------------------------------------
# SPOTIFY LINK PARSING
# ---------------------------------------------------------------------------


def spotify_link_kind_and_id(link: str) -> tuple[str | None, str | None]:
    link = (link or "").strip()
    if not link:
        return None, None

    if link.startswith("spotify:track:"):
        return "track", link.split(":")[-1]

    if link.startswith("spotify:album:"):
        return "album", link.split(":")[-1]

    m = re.search(r"/track/([A-Za-z0-9]+)", link)
    if m:
        return "track", m.group(1)

    m = re.search(r"/album/([A-Za-z0-9]+)", link)
    if m:
        return "album", m.group(1)

    return None, None


def spotify_link_to_track_uri(link: str) -> str | None:
    kind, rid = spotify_link_kind_and_id(link)
    if kind != "track" or not rid:
        return None
    return f"spotify:track:{rid}"


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _chunked(values: list[str], size: int):
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _media_priority(song_type: str | None) -> int:
    t = (song_type or "").lower()
    if t == "op":
        return 0
    if t == "ed":
        return 1
    if t == "ost":
        return 2
    return 3


def _safe_seq(seq: int | None) -> int:
    return seq if seq is not None else 10**9


# ---------------------------------------------------------------------------
# DB FETCH
# ---------------------------------------------------------------------------


async def fetch_result_links(db_path: Path) -> list[tuple[int, str, str, int | None]]:
    """
    Returns:
        (mal_id, spotify_link, song_type, sequence)
    """
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("""
            SELECT mal_id, spotify_link, song_type, sequence
            FROM results
            WHERE spotify_link IS NOT NULL AND TRIM(spotify_link) <> ''
            ORDER BY id
        """) as cursor:
            rows = await cursor.fetchall()

    return [
        (int(mal_id), str(link).strip(), song_type, sequence)
        for mal_id, link, song_type, sequence in rows
        if str(link).strip()
    ]


async def fetch_playlist_links_for_mal_ids(
    db_path: Path, mal_ids: Iterable[int]
) -> list[tuple[int, str, str, int | None]]:
    unique_ids = sorted({int(i) for i in mal_ids})
    if not unique_ids:
        return []

    placeholders = ",".join("?" for _ in unique_ids)

    query = f"""
        SELECT mal_id, spotify_link, song_type, sequence
        FROM results
        WHERE mal_id IN ({placeholders})
          AND spotify_link IS NOT NULL
          AND TRIM(spotify_link) <> ''
        ORDER BY id
    """

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(query, unique_ids) as cursor:
            rows = await cursor.fetchall()

    return [
        (int(mal_id), str(link).strip(), song_type, sequence)
        for mal_id, link, song_type, sequence in rows
        if str(link).strip()
    ]


# ---------------------------------------------------------------------------
# SERIES
# ---------------------------------------------------------------------------


async def fetch_series_playlist_sources(db_path: Path):
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("""
            SELECT series_name, member_ids_json
            FROM series
            ORDER BY series_name COLLATE NOCASE
        """) as cursor:
            rows = await cursor.fetchall()

    out = []
    for name, json_ids in rows:
        try:
            ids = [int(x) for x in json.loads(json_ids)]
        except Exception:
            continue
        if name and ids:
            out.append((name.strip(), ids))
    return out


# ---------------------------------------------------------------------------
# SPOTIFY RESOLUTION
# ---------------------------------------------------------------------------


async def resolve_spotify_link_to_track_uris(
    client: RateLimitedSpotifyClient, link: str
) -> list[str]:
    kind, rid = spotify_link_kind_and_id(link)
    if not kind or not rid:
        return []

    if kind == "track":
        uri = spotify_link_to_track_uri(link)
        return [uri] if uri else []

    if kind == "album":
        try:
            out = []
            offset = 0
            while True:
                page = await client.album_tracks(rid, limit=50, offset=offset)
                for item in page.get("items", []):
                    uri = item.get("uri")
                    if uri:
                        out.append(uri)
                if not page.get("next"):
                    break
                offset += 50
            return out
        except Exception:
            return []

    return []


# ---------------------------------------------------------------------------
# PLAYLIST CREATION
# ---------------------------------------------------------------------------


async def create_spotify_playlist(
    client: RateLimitedSpotifyClient,
    user_id: str,
    name: str,
    entries: list[tuple[int, str, str, int | None]],
) -> None:

    sorted_entries = sorted(
        entries,
        key=lambda x: (
            x[0],
            _media_priority(x[2]),
            _safe_seq(x[3]),
        ),
    )

    resolved: list[str] = []
    for _mal_id, link, _type, _seq in sorted_entries:
        resolved.extend(await resolve_spotify_link_to_track_uris(client, link))

    uris = _unique(resolved)

    if not uris:
        logger.warning("No tracks for %s", name)
        return

    chunks = list(_chunked(uris, SPOTIFY_PLAYLIST_LIMIT))

    for idx, chunk in enumerate(chunks, start=1):
        playlist_name = name if len(chunks) == 1 else f"{name} (Part {idx})"

        playlist = await client.post_playlist(
            {
                "name": playlist_name,
                "public": True,
                "description": "Created by aniplaylist_sync",
            }
        )

        pid = playlist["id"]

        for batch in _chunked(chunk, 100):
            await client.playlist_add_items(pid, batch)

        logger.info(
            "Created playlist %s with %d tracks",
            playlist_name,
            len(chunk),
        )


# ---------------------------------------------------------------------------
# MAIN STAGE
# ---------------------------------------------------------------------------


async def run_spotify_stage(
    db_path: Path,
    *,
    megaplaylist: bool,
    progress: Progress,
) -> None:

    client = RateLimitedSpotifyClient.from_env()
    user_id = (await client.current_user())["id"]

    if megaplaylist:
        sources = [("AniPlaylist Megaplaylist", await fetch_result_links(db_path))]
    else:
        sources = []
        for name, ids in await fetch_series_playlist_sources(db_path):
            entries = await fetch_playlist_links_for_mal_ids(db_path, ids)
            sources.append((name, entries))

    task = progress.add_task("Spotify", total=len(sources))

    for name, entries in sources:
        await create_spotify_playlist(client, user_id, name, entries)
        progress.advance(task)
