"""Spotify playlist creation helpers for AniPlaylist sync."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
import aiohttp
from ratelimit import RateLimiter
from rich.progress import Progress

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
# SPOTIFY CLIENT
# ---------------------------------------------------------------------------

_SPOTIFY_BASE = "https://api.spotify.com/v1"
_SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
_SPOTIFY_SCOPE = "playlist-modify-private playlist-modify-public"


class SpotifyClient:
    """Native async Spotify API client — aiohttp + Authorization Code Flow.

    Token refresh is handled transparently: the access token is refreshed
    automatically when it expires or when a 401 is received mid-flight.
    Spotify occasionally rotates the refresh token; when it does, a warning
    is logged so you can update ``SPOTIFY_REFRESH_TOKEN`` in your ``.env``.

    Env vars read by :meth:`from_env`::

        SPOTIFY_CLIENT_ID     (or SPOTIPY_CLIENT_ID)
        SPOTIFY_CLIENT_SECRET (or SPOTIPY_CLIENT_SECRET)
        SPOTIFY_REFRESH_TOKEN — obtained once via :func:`run_auth_flow`

    Usage::

        client = SpotifyClient.from_env()
        me = await client.current_user()
        await client.close()

    Or as an async context manager::

        async with SpotifyClient.from_env() as client:
            me = await client.current_user()
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._access_token: str | None = None
        self._token_expiry: float = 0.0
        self._token_lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        self._limiter = RateLimiter.from_preset("Spotify")
        logger.debug("SpotifyClient initialised")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "SpotifyClient":
        """Create a SpotifyClient from environment variables.

        Reads ``SPOTIFY_CLIENT_ID``, ``SPOTIFY_CLIENT_SECRET``, and
        ``SPOTIFY_REFRESH_TOKEN`` (with ``SPOTIPY_`` prefix fallbacks for the
        first two).  Run :func:`run_auth_flow` once to obtain the refresh token.
        """
        return cls(
            client_id=_read_spotify_env("SPOTIPY_CLIENT_ID"),
            client_secret=_read_spotify_env("SPOTIPY_CLIENT_SECRET"),
            refresh_token=_read_spotify_env("SPOTIFY_REFRESH_TOKEN"),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        logger.debug("SpotifyClient closing HTTP session")
        if self._session and not self._session.closed:
            await self._session.close()

    async def __aenter__(self) -> "SpotifyClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _basic_auth(self) -> str:
        raw = f"{self._client_id}:{self._client_secret}"
        return base64.b64encode(raw.encode()).decode()

    async def _ensure_token(self) -> str:
        """Return a valid access token, refreshing if needed."""
        async with self._token_lock:
            if self._access_token and time.monotonic() < self._token_expiry - 30:
                return self._access_token

            logger.debug("Refreshing Spotify access token")
            async with self._get_session().post(
                _SPOTIFY_TOKEN_URL,
                headers={
                    "Authorization": f"Basic {self._basic_auth()}",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._refresh_token,
                },
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Spotify token refresh failed ({resp.status}): {text[:200]}"
                    )
                data = await resp.json()

            self._access_token = data["access_token"]
            expires_in = int(data.get("expires_in", 3600))
            self._token_expiry = time.monotonic() + expires_in

            # Spotify occasionally rotates the refresh token
            if "refresh_token" in data:
                self._refresh_token = data["refresh_token"]
                logger.warning(
                    "Spotify issued a new refresh token — update SPOTIFY_REFRESH_TOKEN in .env: %s",
                    data["refresh_token"],
                )

            logger.debug("Spotify token refreshed, expires in %ds", expires_in)
            return self._access_token

    # ------------------------------------------------------------------
    # Core request helper
    # ------------------------------------------------------------------

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict | None = None,
        params: dict | None = None,
    ) -> Any:
        await self._limiter.acquire_async()
        token = await self._ensure_token()
        url = f"{_SPOTIFY_BASE}/{path.lstrip('/')}"

        for attempt in range(5):
            async with self._get_session().request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json=json_body,
                params=params,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                # 204 No Content — success with no body (e.g. playlist_add_items)
                if resp.status == 204:
                    return None

                if resp.status == 401:
                    # Token expired mid-flight; force refresh and retry
                    logger.warning(
                        "Spotify 401 on %s %s — forcing token refresh (attempt %d/5)",
                        method, path, attempt + 1,
                    )
                    async with self._token_lock:
                        self._access_token = None
                    token = await self._ensure_token()
                    continue

                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 1))
                    logger.warning(
                        "Spotify 429 on %s %s — retrying in %.1fs (attempt %d/5)",
                        method, path, retry_after, attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status in {500, 502, 503, 504}:
                    delay = 2.0 * (2 ** attempt)
                    logger.warning(
                        "Spotify HTTP %d on %s %s — retrying in %.1fs (attempt %d/5)",
                        resp.status, method, path, delay, attempt + 1,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(
                        f"Spotify {method} /{path} failed ({resp.status}): {text[:200]}"
                    )

                return await resp.json()

        raise RuntimeError(f"Spotify {method} /{path} failed after 5 attempts")

    # ------------------------------------------------------------------
    # Public API  (same signatures as the old RateLimitedSpotifyClient)
    # ------------------------------------------------------------------

    async def current_user(self) -> dict:
        return await self._request("GET", "me")

    async def post_playlist(self, payload: dict) -> dict:
        return await self._request("POST", "me/playlists", json_body=payload)

    async def playlist_add_items(self, pid: str, uris: list[str]) -> None:
        await self._request("POST", f"playlists/{pid}/tracks", json_body={"uris": uris})

    async def album_tracks(self, rid: str, limit: int, offset: int) -> dict:
        return await self._request(
            "GET", f"albums/{rid}/tracks", params={"limit": limit, "offset": offset}
        )


# ---------------------------------------------------------------------------
# ONE-TIME AUTH FLOW HELPER
# ---------------------------------------------------------------------------


def run_auth_flow(
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
) -> str:
    """Interactive helper to obtain a refresh token via Authorization Code Flow.

    Run this once from the CLI::

        python -c "from spotify import run_auth_flow; run_auth_flow()"

    Follow the printed URL, paste the redirected URL back, and copy the
    ``SPOTIFY_REFRESH_TOKEN`` value printed at the end into your ``.env``.
    """
    import urllib.request

    client_id = client_id or _read_spotify_env("SPOTIPY_CLIENT_ID")
    client_secret = client_secret or _read_spotify_env("SPOTIPY_CLIENT_SECRET")
    redirect_uri = redirect_uri or _read_spotify_env("SPOTIPY_REDIRECT_URI")

    params = urllib.parse.urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": _SPOTIFY_SCOPE,
    })
    auth_url = f"{_SPOTIFY_AUTH_URL}?{params}"

    print("\n── Spotify Auth Flow ──────────────────────────────────────")
    print("Open this URL in your browser and authorise the app:\n")
    print(f"  {auth_url}\n")
    print("After redirecting, paste the full redirect URL here:")
    redirect_response = input("> ").strip()

    # Extract the code from the redirected URL
    parsed = urllib.parse.urlparse(redirect_response)
    code = urllib.parse.parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        raise RuntimeError(f"No 'code' found in redirect URL: {redirect_response!r}")

    # Exchange code for tokens
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        _SPOTIFY_TOKEN_URL,
        data=urllib.parse.urlencode({
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
        }).encode(),
        headers={
            "Authorization": f"Basic {credentials}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(f"No refresh_token in response: {data}")

    print("\n── Success! Add this to your .env ─────────────────────────")
    print(f"SPOTIFY_REFRESH_TOKEN={refresh_token}\n")
    return refresh_token


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
    client: SpotifyClient, link: str
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
    client: SpotifyClient,
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

    client = SpotifyClient.from_env()
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
