"""
scraper.py — AniPlaylist Algolia API client.

Credentials are read from the environment (or a .env file loaded by cli.py):

    ALGOLIA_APP_ID   — Algolia application ID
    ALGOLIA_API_KEY  — Algolia search-only API key

Public API
----------
    client = AlgoliaClient.from_env()
    result = await client.scrape(query)

    # or pass credentials explicitly:
    client = AlgoliaClient(app_id="P4B7...", api_key="cd90...")

ScrapeResult:
    {
        "query":             str,
        "results":           list[ResultItem],
        "raw_html_snapshot": str,   # always "" — kept for API compat
    }

ResultItem:
    {
        "basic_data": BasicData,
        "portal_data": dict | None  # only present during Pass 2
    }

BasicData:
    {
        "anime_title":        str,
        "song_type_raw":      str,       # first token of song_key e.g. "OP", "ED"
        "song_type_sequence": int | None # number after the type token e.g. 1 for "OP1"
        "title_raw":          str,
        "artist_values":      list[str],
        "spotify_link":       str | None,
        "source_index":       int,
        "unreleased":         bool,
    }

Algolia hit field mapping
-------------------------
    hit["anime_titles"]      list[str]  — [0] is the primary English title
    hit["song_key"]          str        — e.g. "OP1", "ED4", "OP EP 4, 6"
    hit["song_type_short"]   str        — fallback e.g. "OP"
    hit["song_type"]         str        — fallback e.g. "Opening"
    hit["titles"]            list[str]  — song titles; [0] is primary
    hit["display_artists"]   list[str]  — ready-to-use artist name list
    hit["artists"]           list[dict] — each has "names": list[str]
    hit["links"]             list[dict] — each has "platform" and "link"
    hit["unreleased"]        int        — 0 = released, 1 = unreleased
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import TypedDict

import aiohttp
from ratelimit import ALGOLIA_DEFAULT_BURST, ALGOLIA_DEFAULT_JITTER_MAX, ALGOLIA_DEFAULT_JITTER_MIN, ALGOLIA_DEFAULT_RPS, AlgoliaLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry constants
# ---------------------------------------------------------------------------

_RETRY_ATTEMPTS = 4  # max attempts for 5xx server errors
_RETRY_BASE_DELAY = 8  # seconds — doubled on each retry
_RETRY_MAX_DELAY = 60.0  # cap so we never wait longer than this

# ---------------------------------------------------------------------------
# Algolia constants
# ---------------------------------------------------------------------------

_INDEX = "songs_prod"
_HITS_PER_PAGE = 100
_FACETS = [
    "links.label",
    "links.link_markets",
    "platforms",
    "season",
    "song_type",
    "status",
]
_USER_AGENT = (
    "Mozilla/5.0 (Linux; Android 15; Pixel 9) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Mobile Safari/537.36"
)

# ---------------------------------------------------------------------------
# TypedDicts  (parser.py imports these)
# ---------------------------------------------------------------------------


class BasicData(TypedDict):
    anime_title: str
    song_type_raw: str  # first token of song_key, e.g. "OP", "ED", "IN"
    song_type_sequence: int | None  # integer following the type token, e.g. 1 for "OP1"
    title_raw: str
    artist_values: list[str]
    spotify_link: str | None
    source_index: int
    unreleased: bool


class ResultItem(TypedDict, total=False):
    basic_data: BasicData  # always present
    portal_data: dict | None  # only present during Pass 2


class ScrapeResult(TypedDict):
    query: str
    results: list[ResultItem]
    raw_html_snapshot: str  # always "" — kept for API compat


# ---------------------------------------------------------------------------
# AlgoliaClient
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AlgoliaClient:
    """Async client for the Algolia search index that backs aniplaylist.com.

    Args:
        app_id:     Algolia application ID (``ALGOLIA_APP_ID`` env var).
        api_key:    Algolia search-only API key (``ALGOLIA_API_KEY`` env var).
        per_second: Maximum requests per second (default: ALGOLIA_DEFAULT_RPS).
        burst:      Initial burst budget (default: ALGOLIA_DEFAULT_BURST).

    Typical usage::

        client = AlgoliaClient.from_env()
        result = await client.scrape("Fullmetal Alchemist")
        await client.close()

    Or as an async context manager::

        async with AlgoliaClient.from_env() as client:
            result = await client.scrape("Fullmetal Alchemist")
    """

    app_id: str
    api_key: str
    per_second: float = ALGOLIA_DEFAULT_RPS
    burst: int = ALGOLIA_DEFAULT_BURST
    jitter_min: float = ALGOLIA_DEFAULT_JITTER_MIN
    jitter_max: float = ALGOLIA_DEFAULT_JITTER_MAX

    _limiter: AlgoliaLimiter = field(init=False, repr=False)
    _session: aiohttp.ClientSession = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._limiter = AlgoliaLimiter(per_second=self.per_second, burst=self.burst, jitter_min=self.jitter_min, jitter_max=self.jitter_max)
        self._session = aiohttp.ClientSession()
        logger.debug(
            "AlgoliaClient initialised — app_id=%r  rate=%.1f/s  burst=%d",
            self.app_id,
            self.per_second,
            self.burst,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        *,
        per_second: float = ALGOLIA_DEFAULT_RPS,
        burst: int = ALGOLIA_DEFAULT_BURST,
    ) -> "AlgoliaClient":
        """Create a client from ``ALGOLIA_APP_ID`` / ``ALGOLIA_API_KEY`` env vars.

        Raises ``RuntimeError`` if either variable is missing.
        """
        app_id = os.getenv("ALGOLIA_APP_ID")
        api_key = os.getenv("ALGOLIA_API_KEY")
        missing = [
            name
            for name, val in [("ALGOLIA_APP_ID", app_id), ("ALGOLIA_API_KEY", api_key)]
            if not val
        ]
        if missing:
            raise RuntimeError(
                f"Missing required environment variable(s): {', '.join(missing)}"
            )
        return cls(app_id=app_id, api_key=api_key, per_second=per_second, burst=burst)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        logger.debug("AlgoliaClient closing HTTP session")
        await self._session.close()

    async def __aenter__(self) -> "AlgoliaClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _url(self) -> str:
        return (
            f"https://{self.app_id}-dsn.algolia.net/1/indexes/*/queries"
            f"?x-algolia-agent=Algolia%20for%20JavaScript%20(4.26.0)%3B%20Browser%20(lite)"
            f"&x-algolia-api-key={self.api_key}"
            f"&x-algolia-application-id={self.app_id}"
        )

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Referer": "https://aniplaylist.com/",
            "Origin": "https://aniplaylist.com",
            "User-Agent": _USER_AGENT,
        }

    def _build_payload(self, query: str, page: int) -> dict:
        params = "&".join(
            [
                f"query={urllib.parse.quote(query)}",
                "analytics=true",
                "clickAnalytics=true",
                "distinct=true",
                "enablePersonalization=false",
                f"facets={urllib.parse.quote(json.dumps(_FACETS))}",
                "highlightPostTag=__/ais-highlight__",
                "highlightPreTag=__ais-highlight__",
                f"hitsPerPage={_HITS_PER_PAGE}",
                "maxValuesPerFacet=250",
                f"page={page}",
                "userToken=anonymous-python-client",
            ]
        )
        return {"requests": [{"indexName": _INDEX, "params": params}]}

    async def _fetch_page(
        self,
        query: str,
        page: int,
        mal_label: str,
        raw_log: list[dict] | None = None,
    ) -> dict:
        """Fetch one Algolia page with retry/backoff on transient errors.

        HTTP responses handled by status range:
          2xx — success
          429 — rate limited: retry indefinitely with backoff (Retry-After respected)
          5xx — server error: retry up to _RETRY_ATTEMPTS times with backoff
          4xx (other) — client error: raise immediately

        If *raw_log* is provided, a dict is appended for every attempt with the
        full response envelope (status, headers, body, timing).
        """
        import time as _time

        payload = self._build_payload(query, page)

        attempt = 0
        while True:
            await self._limiter.acquire()
            logger.debug(
                "%s Algolia page=%d query=%r attempt=%d",
                mal_label,
                page,
                query,
                attempt,
            )

            t0 = _time.monotonic()
            async with self._session.post(
                self._url(), headers=self._headers(), json=payload
            ) as resp:
                status = resp.status
                elapsed_ms = round((_time.monotonic() - t0) * 1000, 1)
                resp_headers = dict(resp.headers)

                if 200 <= status < 300:
                    body = await resp.json()
                    if raw_log is not None:
                        raw_log.append({
                            "page": page,
                            "attempt": attempt,
                            "status": status,
                            "elapsed_ms": elapsed_ms,
                            "headers": resp_headers,
                            "body": body,
                            "error": None,
                        })
                    return body["results"][0]

                text = await resp.text()
                try:
                    body_raw = json.loads(text)
                except Exception:
                    body_raw = text

                if raw_log is not None:
                    raw_log.append({
                        "page": page,
                        "attempt": attempt,
                        "status": status,
                        "elapsed_ms": elapsed_ms,
                        "headers": resp_headers,
                        "body": body_raw,
                        "error": f"HTTP {status}",
                    })

                if status == 429:
                    # Rate limited — retry indefinitely, honouring Retry-After
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after is not None:
                        try:
                            delay = min(float(retry_after), _RETRY_MAX_DELAY)
                        except ValueError:
                            delay = min(
                                _RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY
                            )
                    else:
                        delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                    logger.warning(
                        "%s Algolia 429 on page=%d — retrying in %.1fs (attempt %d): %s",
                        mal_label,
                        page,
                        delay,
                        attempt + 1,
                        text[:200],
                    )
                    await asyncio.sleep(delay)
                    attempt += 1
                    continue

                if status >= 500:
                    # Transient server error — retry up to _RETRY_ATTEMPTS times
                    if attempt < _RETRY_ATTEMPTS - 1:
                        delay = min(_RETRY_BASE_DELAY * (2**attempt), _RETRY_MAX_DELAY)
                        logger.warning(
                            "%s Algolia HTTP %d on page=%d — retrying in %.1fs "
                            "(attempt %d/%d): %s",
                            mal_label,
                            status,
                            page,
                            delay,
                            attempt + 1,
                            _RETRY_ATTEMPTS,
                            text[:200],
                        )
                        await asyncio.sleep(delay)
                        attempt += 1
                        continue
                    logger.error(
                        "%s Algolia HTTP %d on page=%d — all %d attempts exhausted: %s",
                        mal_label,
                        status,
                        page,
                        _RETRY_ATTEMPTS,
                        text[:200],
                    )
                else:
                    # 4xx (non-429) — non-retryable
                    logger.error(
                        "%s Algolia HTTP %d on page=%d (non-retryable): %s",
                        mal_label,
                        status,
                        page,
                        text[:200],
                    )
                raise RuntimeError(
                    f"Algolia HTTP {status} on page {page}: {text[:200]}"
                )

    @staticmethod
    def _hit_to_result_item(
        hit: dict, idx: int, include_alternates: bool
    ) -> ResultItem:
        """Map a raw Algolia hit directly to a ResultItem."""

        # anime_title
        anime_titles: list[str] = hit.get("anime_titles") or []
        anime_title = anime_titles[0] if anime_titles else ""

        # song_type_raw + song_type_sequence
        # song_key e.g. "OP1", "ED4", "OP EP 4, 6" — first token minus trailing
        # digits is the type; first integer anywhere is the sequence number.
        song_key: str = (
            hit.get("song_key")
            or hit.get("song_type_short")
            or hit.get("song_type")
            or ""
        )
        tokens = song_key.split()
        song_type_raw = re.sub(r"\d+$", "", tokens[0]).upper() if tokens else ""
        seq_match = re.search(r"\d+", song_key)
        song_type_sequence: int | None = int(seq_match.group()) if seq_match else None

        # title_raw
        titles: list[str] = hit.get("titles") or []
        title_raw = titles[0] if titles else ""

        # artist_values — prefer display_artists; fall back to artists[*].names[0]
        artist_values: list[str] = hit.get("display_artists") or []
        if not artist_values:
            for artist in hit.get("artists") or []:
                names: list[str] = artist.get("names") or []
                if names:
                    artist_values.append(names[0])

        # spotify_link — first link whose platform is "spotify"
        spotify_link: str | None = None
        for link in hit.get("links") or []:
            if (link.get("platform") or "").lower() == "spotify" and link.get("link"):
                spotify_link = link["link"]
                break

        # unreleased
        unreleased = bool(hit.get("unreleased", 0))

        item = ResultItem(
            basic_data=BasicData(
                anime_title=anime_title,
                song_type_raw=song_type_raw,
                song_type_sequence=song_type_sequence,
                title_raw=title_raw,
                artist_values=artist_values,
                spotify_link=spotify_link,
                source_index=idx,
                unreleased=unreleased,
            )
        )

        # Pass 2 only: inject alternate anime titles as portal synonyms
        if include_alternates:
            item["portal_data"] = {"synonyms": anime_titles[1:]}

        return item

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def scrape(
        self,
        query: str,
        headless: bool = True,  # accepted for API compat; ignored
        fetch_portal_indices: set[int] | None = None,  # accepted; ignored
        mal_label: str = "",
        raw_log: list[dict] | None = None,
    ) -> ScrapeResult:
        """Search AniPlaylist via Algolia for *query*.

        Args:
            query:                Search term (anime title, song name, …).
            headless:             Ignored. Kept for drop-in compatibility.
            fetch_portal_indices: When not None, alternate anime titles are
                                  injected as portal synonyms (Pass 2 behaviour).
            mal_label:            Optional "[MAL:id 'title']" prefix for log lines.
            raw_log:              If provided, each HTTP attempt's full response
                                  envelope (status, headers, body, timing) is
                                  appended to this list.
        """
        logger.info("%s AlgoliaClient.scrape() — query=%r", mal_label, query)

        all_hits: list[dict] = []

        # Fetch page 0 first to learn total page count
        result0 = await self._fetch_page(query, 0, mal_label, raw_log)
        nb_pages: int = result0.get("nbPages", 1)
        nb_hits: int = result0.get("nbHits", 0)
        all_hits.extend(result0.get("hits", []))

        logger.info(
            "%s Algolia: %d hits across %d page(s)",
            mal_label,
            nb_hits,
            nb_pages,
        )

        # Fetch remaining pages concurrently (still rate-limited per request)
        if nb_pages > 1:
            tasks = [self._fetch_page(query, p, mal_label, raw_log) for p in range(1, nb_pages)]
            pages = await asyncio.gather(*tasks, return_exceptions=True)
            for i, page_result in enumerate(pages, start=1):
                if isinstance(page_result, Exception):
                    logger.warning(
                        "%s Algolia page %d failed: %s",
                        mal_label,
                        i,
                        page_result,
                    )
                else:
                    all_hits.extend(page_result.get("hits", []))

        logger.info("%s Algolia: fetched %d total hits", mal_label, len(all_hits))

        include_alternates = fetch_portal_indices is not None
        results = [
            self._hit_to_result_item(hit, idx, include_alternates)
            for idx, hit in enumerate(all_hits)
        ]

        return ScrapeResult(query=query, results=results, raw_html_snapshot="")


# ---------------------------------------------------------------------------
# Module-level compat shim — lets main.py keep `from scraper import scrape`
# until it's updated to use AlgoliaClient directly.
# ---------------------------------------------------------------------------

_default_client: AlgoliaClient | None = None


async def scrape(
    query: str,
    headless: bool = True,
    fetch_portal_indices: set[int] | None = None,
    mal_label: str = "",
) -> ScrapeResult:
    """Module-level shim: delegates to a lazily-created AlgoliaClient.from_env()."""
    global _default_client
    if _default_client is None:
        _default_client = AlgoliaClient.from_env()
    return await _default_client.scrape(
        query,
        headless=headless,
        fetch_portal_indices=fetch_portal_indices,
        mal_label=mal_label,
    )
