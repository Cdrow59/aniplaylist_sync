"""Spotify playlist creation helpers for AniPlaylist sync."""

from __future__ import annotations

import asyncio
import base64
import csv
import html
import json
import logging
import os
import re
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable

import aiohttp
import aiosqlite
from rich.progress import Progress

from ratelimit import RateLimiter

logger = logging.getLogger(__name__)

SPOTIFY_PLAYLIST_LIMIT = 9999

# ---------------------------------------------------------------------------
# SPOTIFY CLIENT
# ---------------------------------------------------------------------------

_SPOTIFY_BASE = "https://api.spotify.com/v1"
_SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
_SPOTIFY_AUTH_URL = "https://accounts.spotify.com/authorize"
_SPOTIFY_SCOPE = "playlist-modify-private playlist-modify-public"


# ---------------------------------------------------------------------------
# SONG TYPE FILTERING
# ---------------------------------------------------------------------------

# Song types to include by default (case-insensitive). Applies to every
# playlist unless overridden below in SERIES_SONG_TYPE_FILTER. Set to
# ``None`` to disable filtering entirely (include every song type,
# including entries with a missing/blank song_type).
DEFAULT_SONG_TYPE_FILTER: set[str] | None = {"op", "ed"}

# Per-series overrides, keyed by the exact series name as it appears in the
# `series` table / playlist name (or "AniPlaylist Megaplaylist" for the
# megaplaylist run). A series not listed here falls back to
# DEFAULT_SONG_TYPE_FILTER. Set a series's value to ``None`` to disable
# filtering for just that series (include all song types for it).
#
# These overrides are applied PER ENTRY based on each track's own series —
# they work the same whether you're building individual per-series
# playlists or one combined megaplaylist (see ``series_lookup`` in
# :func:`create_spotify_playlist`).
#
# Example:
#   SERIES_SONG_TYPE_FILTER = {
#       "Attack on Titan": {"op", "ed"},   # no OSTs for this one
#       "Cowboy Bebop": None,              # include everything
#   }
SERIES_SONG_TYPE_FILTER: dict[str, set[str] | None] = {
    "Naruto": None,
    "One Piece": None,
    "Bleach": None,
    "Fairy Tail": None,
}


def spotify_token_env_key(spotify_user: str | None = None) -> str:
    """Return the ``.env`` key holding the refresh token for ``spotify_user``.

    With no user given, this is the original single-account
    ``SPOTIFY_REFRESH_TOKEN``. When ``--spotify-user NAME`` is passed on the
    CLI, each account gets its own key (``SPOTIFY_REFRESH_TOKEN_NAME``), so
    multiple Spotify accounts — each added as a user on the app in the
    Spotify Developer Dashboard — can be authorised once and reused by name.
    """
    if not spotify_user:
        return "SPOTIFY_REFRESH_TOKEN"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", spotify_user).strip("_").upper()
    return f"SPOTIFY_REFRESH_TOKEN_{slug}"


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
        token_env_key: str = "SPOTIFY_REFRESH_TOKEN",
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._token_env_key = token_env_key
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
    def from_env(cls, spotify_user: str | None = None) -> "SpotifyClient":
        """Create a SpotifyClient from environment variables.

        Reads ``SPOTIFY_CLIENT_ID``, ``SPOTIFY_CLIENT_SECRET``, and the
        refresh token. With no ``spotify_user``, the refresh token comes
        from ``SPOTIFY_REFRESH_TOKEN``; pass ``spotify_user`` to instead use
        that account's own ``SPOTIFY_REFRESH_TOKEN_<USER>`` key (see
        :func:`spotify_token_env_key`), letting multiple Spotify accounts —
        each added as a user on the app in the Spotify Developer Dashboard —
        share one ``.env``.  Run :func:`run_auth_flow` once per account to
        obtain its refresh token.
        """
        token_env_key = spotify_token_env_key(spotify_user)
        return cls(
            client_id=os.getenv("SPOTIFY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
            refresh_token=os.getenv(token_env_key),
            token_env_key=token_env_key,
        )

    @classmethod
    async def from_env_async(cls, spotify_user: str | None = None) -> "SpotifyClient":
        """Create a SpotifyClient from environment variables.

        Unlike :meth:`from_env`, this will automatically launch the
        browser-based authorisation flow (see :func:`run_auth_flow_async`)
        if the refresh token is missing — no manual script run or
        copy/pasting required.

        ``spotify_user`` selects which account's refresh token to use/store,
        via :func:`spotify_token_env_key`. This lets ``--spotify-user NAME``
        run the sync against a different Spotify account than the default,
        as long as that account has been added as a user on the app in the
        Spotify Developer Dashboard (required while the app is in
        Development Mode).
        """
        client_id = os.getenv("SPOTIFY_CLIENT_ID")
        client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
        token_env_key = spotify_token_env_key(spotify_user)
        refresh_token = os.getenv(token_env_key)

        if not refresh_token:
            logger.info(
                "No %s found — opening browser for Spotify authorisation%s",
                token_env_key,
                f" (user: {spotify_user})" if spotify_user else "",
            )
            refresh_token = await run_auth_flow_async(
                client_id, client_secret, redirect_uri, token_env_key=token_env_key
            )

        return cls(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            token_env_key=token_env_key,
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
                    "Spotify issued a new refresh token — update %s in .env: %s",
                    self._token_env_key,
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
                        method,
                        path,
                        attempt + 1,
                    )
                    async with self._token_lock:
                        self._access_token = None
                    token = await self._ensure_token()
                    continue

                if resp.status == 429:
                    retry_after = float(resp.headers.get("Retry-After", 1))
                    logger.warning(
                        "Spotify 429 on %s %s — retrying in %.1fs (attempt %d/5)",
                        method,
                        path,
                        retry_after,
                        attempt + 1,
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if resp.status in {500, 502, 503, 504}:
                    delay = 2.0 * (2**attempt)
                    logger.warning(
                        "Spotify HTTP %d on %s %s — retrying in %.1fs (attempt %d/5)",
                        resp.status,
                        method,
                        path,
                        delay,
                        attempt + 1,
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

    client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI")

    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": _SPOTIFY_SCOPE,
        }
    )
    auth_url = f"{_SPOTIFY_AUTH_URL}?{params}"

    logger.info("Spotify Auth Flow — open this URL and authorise the app: %s", auth_url)
    redirect_response = input(
        "After authorising, paste the full redirect URL here:\n> "
    ).strip()

    # Extract the code from the redirected URL
    parsed = urllib.parse.urlparse(redirect_response)
    code = urllib.parse.parse_qs(parsed.query).get("code", [None])[0]
    if not code:
        raise RuntimeError(f"No 'code' found in redirect URL: {redirect_response!r}")

    # Exchange code for tokens
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        _SPOTIFY_TOKEN_URL,
        data=urllib.parse.urlencode(
            {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri,
            }
        ).encode(),
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

    logger.info(
        "Spotify authorisation succeeded — SPOTIFY_REFRESH_TOKEN=%s", refresh_token
    )
    return refresh_token


# ---------------------------------------------------------------------------
# FULLY AUTOMATIC BROWSER AUTH FLOW (no copy/paste required)
# ---------------------------------------------------------------------------


_SUCCESS_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spotify connected</title>
<style>
  html, body {
    height: 100%;
    margin: 0;
    background: #121212;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .card {
    text-align: center;
    padding: 48px 40px;
  }
  .check {
    width: 72px;
    height: 72px;
    margin: 0 auto 24px;
    border-radius: 50%;
    background: #1DB954;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .check svg { width: 36px; height: 36px; }
  h1 { font-size: 22px; font-weight: 700; margin: 0 0 8px; }
  p { font-size: 15px; color: #b3b3b3; margin: 0; }
</style>
</head>
<body>
  <div class="card">
    <div class="check">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M5 13l4 4L19 7" stroke="#121212" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <h1>Spotify connected</h1>
    <p id="msg">This tab will close automatically&hellip;</p>
  </div>
  <script>
    setTimeout(function () {
      window.close();
      setTimeout(function () {
        document.getElementById('msg').textContent = 'You can close this tab now.';
      }, 400);
    }, 900);
  </script>
</body>
</html>
"""

_ERROR_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Spotify authorisation failed</title>
<style>
  html, body {{
    height: 100%;
    margin: 0;
    background: #121212;
    color: #ffffff;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .card {{ text-align: center; padding: 48px 40px; max-width: 420px; }}
  .x {{
    width: 72px;
    height: 72px;
    margin: 0 auto 24px;
    border-radius: 50%;
    background: #e91429;
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .x svg {{ width: 32px; height: 32px; }}
  h1 {{ font-size: 22px; font-weight: 700; margin: 0 0 8px; }}
  p {{ font-size: 15px; color: #b3b3b3; margin: 0; }}
  code {{ color: #e0e0e0; }}
</style>
</head>
<body>
  <div class="card">
    <div class="x">
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M6 6l12 12M18 6L6 18" stroke="#121212" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <h1>Authorisation failed</h1>
    <p><code>{error}</code></p>
    <p style="margin-top:12px;">You can close this tab and try again.</p>
  </div>
</body>
</html>
"""


class _AuthCallbackHandler(BaseHTTPRequestHandler):
    """Tiny one-shot HTTP handler that captures the Spotify redirect."""

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        self.server.auth_code = qs.get("code", [None])[0]
        self.server.auth_error = qs.get("error", [None])[0]

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if self.server.auth_code:
            body = _SUCCESS_PAGE
        else:
            safe_error = html.escape(self.server.auth_error or "unknown_error")
            body = _ERROR_PAGE.format(error=safe_error)
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass  # silence default request logging


def _persist_refresh_token(
    refresh_token: str, token_env_key: str = "SPOTIFY_REFRESH_TOKEN"
) -> None:
    """Write the new refresh token into .env so future runs skip the browser step."""
    try:
        from dotenv import set_key

        env_path = Path(__file__).with_name(".env")
        if env_path.exists():
            set_key(str(env_path), token_env_key, refresh_token)
            logger.info("Saved %s to %s", token_env_key, env_path)
        else:
            logger.warning(
                "No .env file found next to spotify.py — add this manually: " "%s=%s",
                token_env_key,
                refresh_token,
            )
    except Exception:
        logger.warning(
            "Could not auto-save %s to .env — add manually: %s",
            token_env_key,
            refresh_token,
        )


async def run_auth_flow_async(
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    timeout: float = 180.0,
    token_env_key: str = "SPOTIFY_REFRESH_TOKEN",
) -> str:
    """Fully automatic browser-based auth flow — no copy/pasting required.

    Spins up a one-shot local HTTP server on ``redirect_uri``, opens the
    Spotify authorisation page in the default browser, waits for the
    redirect to land, exchanges the code for tokens, and saves the new
    refresh token straight into ``.env`` under ``token_env_key`` (defaults
    to ``SPOTIFY_REFRESH_TOKEN``; pass a per-user key from
    :func:`spotify_token_env_key` to authorise a specific account).

    The browser prompt authorises whichever Spotify account is currently
    logged in on this machine — to authorise a *different* account, log
    that account into open.spotify.com/Spotify first (or use a private
    browser window), and make sure it has been added as a user on the app
    in the Spotify Developer Dashboard, which is required while the app is
    in Development Mode.

    ``redirect_uri`` must already be registered as a Redirect URI on the
    app in the Spotify Developer Dashboard.
    """
    import webbrowser

    client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
    redirect_uri = redirect_uri or os.getenv("SPOTIFY_REDIRECT_URI")

    if not client_id or not client_secret or not redirect_uri:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI "
            "must all be set (in .env or passed explicitly) to authorise."
        )

    parsed_redirect = urllib.parse.urlparse(redirect_uri)
    host = parsed_redirect.hostname or "127.0.0.1"
    port = parsed_redirect.port or 80

    server = HTTPServer((host, port), _AuthCallbackHandler)
    server.auth_code = None
    server.auth_error = None
    server.timeout = (
        timeout  # makes handle_request() return after this many idle seconds
    )

    params = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "scope": _SPOTIFY_SCOPE,
        }
    )
    auth_url = f"{_SPOTIFY_AUTH_URL}?{params}"

    logger.info("Opening browser for Spotify authorisation: %s", auth_url)
    webbrowser.open(auth_url, new=1, autoraise=False)

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, server.handle_request)
    server.server_close()

    if server.auth_code is None and server.auth_error is None:
        raise RuntimeError(
            f"Timed out after {timeout:.0f}s waiting for Spotify authorisation in browser"
        )

    if server.auth_error:
        raise RuntimeError(f"Spotify authorisation failed: {server.auth_error}")
    if not server.auth_code:
        raise RuntimeError("No authorisation code received from Spotify redirect")

    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            _SPOTIFY_TOKEN_URL,
            headers={
                "Authorization": f"Basic {credentials}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={
                "grant_type": "authorization_code",
                "code": server.auth_code,
                "redirect_uri": redirect_uri,
            },
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(
                    f"Spotify token exchange failed ({resp.status}): {text[:200]}"
                )
            data = await resp.json()

    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise RuntimeError(f"No refresh_token in token exchange response: {data}")

    _persist_refresh_token(refresh_token, token_env_key=token_env_key)
    logger.info("Spotify authorisation complete")
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


def _allowed_song_types(series_name: str | None) -> set[str] | None:
    """Resolve the allowed song types for a given series name.

    Returns a lowercase set of allowed song types, or ``None`` if all types
    are allowed (no filtering).
    """
    if series_name is not None and series_name in SERIES_SONG_TYPE_FILTER:
        allowed = SERIES_SONG_TYPE_FILTER[series_name]
    else:
        allowed = DEFAULT_SONG_TYPE_FILTER
    return {t.lower() for t in allowed} if allowed is not None else None


def _entry_passes_song_type_filter(
    song_type: str | None, allowed: set[str] | None
) -> bool:
    if allowed is None:
        return True
    return (song_type or "").lower() in allowed


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


async def fetch_mal_id_to_series(db_path: Path) -> dict[int, str]:
    """Map each ``mal_id`` to its series name.

    Used in megaplaylist mode so per-series ``SERIES_SONG_TYPE_FILTER``
    overrides can still be applied per-entry, even though all entries are
    combined into a single playlist with no series-specific ``name``.
    """
    mapping: dict[int, str] = {}
    for name, ids in await fetch_series_playlist_sources(db_path):
        for mal_id in ids:
            mapping[mal_id] = name
    return mapping


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
    series_lookup: dict[int, str] | None = None,
) -> list[dict[str, object]]:
    """Create (one or more) Spotify playlists from ``entries``.

    ``series_lookup`` maps ``mal_id -> series_name`` and, when provided, is
    used to resolve the song-type filter *per entry* via that entry's own
    series — this is what makes ``SERIES_SONG_TYPE_FILTER`` overrides work
    correctly in megaplaylist mode, where ``name`` is a fixed label
    ("AniPlaylist Megaplaylist") rather than an actual series name. When
    ``series_lookup`` is ``None`` (per-series mode), the filter falls back
    to resolving against ``name`` directly, same as before.
    """

    def _entry_allowed(mal_id: int, song_type: str | None) -> bool:
        if series_lookup is not None:
            series_name = series_lookup.get(mal_id, name)
        else:
            series_name = name
        allowed_types = _allowed_song_types(series_name)
        return _entry_passes_song_type_filter(song_type, allowed_types)

    filtered_entries = [e for e in entries if _entry_allowed(e[0], e[2])]

    if len(filtered_entries) != len(entries):
        logger.info(
            "[%s] song-type filter: %d/%d entries kept",
            name,
            len(filtered_entries),
            len(entries),
        )

    sorted_entries = sorted(
        filtered_entries,
        key=lambda x: (
            x[0],
            _media_priority(x[2]),
            _safe_seq(x[3]),
        ),
    )

    resolved: list[str] = []
    total_entries = len(sorted_entries)
    for idx, (mal_id, link, song_type, _seq) in enumerate(sorted_entries, start=1):
        uris_for_entry = await resolve_spotify_link_to_track_uris(client, link)
        resolved.extend(uris_for_entry)
        logger.info(
            "[%s] (%d/%d) mal_id=%s type=%s -> %d track(s) resolved",
            name,
            idx,
            total_entries,
            mal_id,
            song_type or "?",
            len(uris_for_entry),
        )

    uris = _unique(resolved)

    if not uris:
        logger.warning("No tracks for %s", name)
        return []

    chunks = list(_chunked(uris, SPOTIFY_PLAYLIST_LIMIT))

    created: list[dict[str, object]] = []
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

        label = f"[playlist:{pid} '{playlist_name}']"
        logger.info(
            "Created %s with %d tracks",
            label,
            len(chunk),
        )

        created.append({"id": pid, "name": playlist_name, "length": len(chunk)})

    return created


_PLAYLIST_CSV_HEADER = ["playlist_id", "name", "length"]


class _PlaylistCsvWriter:
    """Writes playlist id/name/length rows incrementally to
    ``output/output_{username}.csv`` as each playlist is created, so
    progress isn't lost if a run is interrupted partway through.
    """

    def __init__(self, username: str) -> None:
        self.path = Path("output") / f"output_{username}.csv"
        self._file = None
        self._writer = None
        self.count = 0

    def __enter__(self) -> "_PlaylistCsvWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(_PLAYLIST_CSV_HEADER)
        self._file.flush()
        return self

    def write_records(self, records: list[dict[str, object]]) -> None:
        if not records or self._writer is None:
            return
        for r in records:
            self._writer.writerow([r["id"], r["name"], r["length"]])
        self._file.flush()
        self.count += len(records)

    def __exit__(self, *_: object) -> None:
        if self._file is not None:
            self._file.close()
        logger.info("Wrote %d playlist record(s) to %s", self.count, self.path)


# ---------------------------------------------------------------------------
# MAIN STAGE
# ---------------------------------------------------------------------------


async def run_spotify_stage(
    db_path: Path,
    *,
    megaplaylist: bool,
    progress: Progress,
    username: str | None = None,
    spotify_user: str | None = None,
) -> None:
    """Run the Spotify playlist-creation stage.

    ``spotify_user`` selects which Spotify account to authorise/act as via
    ``--spotify-user NAME`` on the CLI (see :func:`spotify_token_env_key`).
    That account must be added as a user on the app in the Spotify
    Developer Dashboard while the app is in Development Mode. Leave it
    unset to keep using the single default account (``SPOTIFY_REFRESH_TOKEN``).
    """

    client = await SpotifyClient.from_env_async(spotify_user=spotify_user)

    try:
        try:
            user_id = (await client.current_user())["id"]
        except RuntimeError as exc:
            # Stored refresh token is invalid/expired/revoked — re-authorise
            # in the browser automatically rather than failing out.
            msg = str(exc)
            if "Spotify token refresh failed" in msg or " 400" in msg or " 401" in msg:
                logger.warning(
                    "%s appears invalid or expired — re-authorising in browser",
                    client._token_env_key,
                )
                new_refresh_token = await run_auth_flow_async(
                    client._client_id,
                    client._client_secret,
                    os.getenv("SPOTIFY_REDIRECT_URI"),
                    token_env_key=client._token_env_key,
                )
                client._refresh_token = new_refresh_token
                client._access_token = None
                user_id = (await client.current_user())["id"]
            else:
                raise

        # series_lookup (mal_id -> series name) lets per-entry filtering
        # apply SERIES_SONG_TYPE_FILTER overrides correctly in megaplaylist
        # mode, where sources is a single combined batch with no per-entry
        # series name otherwise available.
        if megaplaylist:
            series_lookup = await fetch_mal_id_to_series(db_path)
            sources = [("AniPlaylist Megaplaylist", await fetch_result_links(db_path))]
        else:
            series_lookup = None
            sources = []
            for name, ids in await fetch_series_playlist_sources(db_path):
                entries = await fetch_playlist_links_for_mal_ids(db_path, ids)
                sources.append((name, entries))

        task = progress.add_task("Spotify", total=len(sources))

        if username:
            csv_writer_cm = _PlaylistCsvWriter(username)
        else:
            csv_writer_cm = None
            if sources:
                logger.warning("No username provided — skipping playlist CSV output")

        if csv_writer_cm is not None:
            with csv_writer_cm as csv_writer:
                for name, entries in sources:
                    created = await create_spotify_playlist(
                        client, user_id, name, entries, series_lookup=series_lookup
                    )
                    csv_writer.write_records(created)
                    progress.advance(task)
        else:
            for name, entries in sources:
                await create_spotify_playlist(
                    client, user_id, name, entries, series_lookup=series_lookup
                )
                progress.advance(task)
    finally:
        await client.close()
