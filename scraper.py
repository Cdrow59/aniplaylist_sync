"""
scraper.py — AniPlaylist Algolia API client.

Replaces the former Playwright-based scraper with direct calls to the
Algolia search index that powers aniplaylist.com.  The public TypedDicts
(BasicData, ResultItem, ScrapeResult) and the scrape() signature are
preserved so parser.py and main.py need no changes.

Public API
----------
    result = await scrape(query)

ScrapeResult:
    {
        "query":             str,
        "results":           list[ResultItem],
        "raw_html_snapshot": str,   # always "" — kept for API compat
    }

ResultItem:
    {
        "basic_data": BasicData,
        # portal_data is never present — Algolia gives us everything directly
    }

BasicData:
    {
        "anime_title":   str,
        "song_type_raw": str,
        "title_raw":     str,
        "artist_values": list[str],
        "spotify_link":  str | None,
        "source_index":  int,
        "unreleased":    bool,
    }

Algolia field mapping
---------------------
The index is ``songs_prod``.  Known hit fields (adjust if the schema
changes):

    hit["anime"]           → anime_title
    hit["song_type"]       → song_type_raw  (e.g. "OP1", "ED3")
    hit["title"]           → title_raw
    hit["artists"]         → artist_values  (list[str])
    hit["links"]           → list of {label, url} dicts; Spotify link extracted
    hit["status"]          → "unreleased" → unreleased=True

If a field is missing the code falls back gracefully (empty string / []).
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict

import aiohttp

from ratelimit import AlgoliaLimiter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Algolia constants
# ---------------------------------------------------------------------------

_APP_ID = "P4B7HT5P18"
_API_KEY = "cd90c9c918df8b42327310ade1f599bd"
_INDEX   = "songs_prod"
_URL     = (
    f"https://{_APP_ID}-dsn.algolia.net/1/indexes/*/queries"
    f"?x-algolia-agent=Algolia%20for%20JavaScript%20(4.26.0)%3B%20Browser%20(lite)"
    f"&x-algolia-api-key={_API_KEY}"
    f"&x-algolia-application-id={_APP_ID}"
)
_HEADERS = {
    "Content-Type": "application/json",
    "Referer":      "https://aniplaylist.com/",
    "Origin":       "https://aniplaylist.com",
    "User-Agent": (
        "Mozilla/5.0 (Linux; Android 15; Pixel 9) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/149.0.0.0 Mobile Safari/537.36"
    ),
}

_HITS_PER_PAGE = 100
_FACETS = ["links.label", "links.link_markets", "platforms", "season", "song_type", "status"]

# ---------------------------------------------------------------------------
# Module-level rate limiter (shared across all scrape() calls)
# ---------------------------------------------------------------------------

_limiter: AlgoliaLimiter | None = None


def _get_limiter() -> AlgoliaLimiter:
    global _limiter
    if _limiter is None:
        _limiter = AlgoliaLimiter()
    return _limiter


# ---------------------------------------------------------------------------
# TypedDicts  (identical to former scraper.py — parser.py imports these)
# ---------------------------------------------------------------------------


class BasicData(TypedDict):
    anime_title:   str
    song_type_raw: str
    title_raw:     str
    artist_values: list[str]
    spotify_link:  str | None
    source_index:  int
    unreleased:    bool


class ResultItem(TypedDict, total=False):
    basic_data:  BasicData   # always present
    portal_data: dict | None # never set by this client; kept for type compat


class ScrapeResult(TypedDict):
    query:             str
    results:           list[ResultItem]
    raw_html_snapshot: str   # always "" — kept for API compat


# ---------------------------------------------------------------------------
# Algolia request helpers
# ---------------------------------------------------------------------------


def _build_payload(query: str, page: int) -> dict:
    import json, urllib.parse
    params = "&".join([
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
    ])
    return {"requests": [{"indexName": _INDEX, "params": params}]}


def _extract_spotify(links: list[dict]) -> str | None:
    """Return the Spotify URL from a hit's links list, or None."""
    for link in links:
        label = (link.get("label") or "").lower()
        if "spotify" in label:
            return link.get("url") or link.get("link") or None
    return None


def _hit_to_basic(hit: dict, index: int) -> BasicData:
    """Map a raw Algolia hit dict to a BasicData TypedDict."""
    links: list[dict] = hit.get("links") or []

    # artists may be a list of strings or a list of dicts with a "name" key
    raw_artists = hit.get("artists") or []
    if raw_artists and isinstance(raw_artists[0], dict):
        artist_values = [a.get("name", "") for a in raw_artists if a.get("name")]
    else:
        artist_values = [str(a) for a in raw_artists if a]

    status: str = (hit.get("status") or "").lower()
    unreleased = status in {"unreleased", "not_released", "upcoming"}

    return BasicData(
        anime_title   = hit.get("anime") or hit.get("anime_title") or "",
        song_type_raw = hit.get("song_type") or "",
        title_raw     = hit.get("title") or hit.get("name") or "",
        artist_values = artist_values,
        spotify_link  = _extract_spotify(links),
        source_index  = index,
        unreleased    = unreleased,
    )


# ---------------------------------------------------------------------------
# Core fetch — single page
# ---------------------------------------------------------------------------


async def _fetch_page(
    session: aiohttp.ClientSession,
    query: str,
    page: int,
    mal_label: str,
) -> dict:
    """Fetch one page from Algolia and return the raw result dict."""
    limiter = _get_limiter()
    await limiter.acquire()

    payload = _build_payload(query, page)
    logger.debug("%s Algolia page=%d query=%r", mal_label, page, query)

    async with session.post(_URL, headers=_HEADERS, json=payload) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(
                f"Algolia returned HTTP {resp.status} on page {page}: {text[:200]}"
            )
        data = await resp.json()

    return data["results"][0]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def scrape(
    query: str,
    headless: bool = True,          # accepted for API compat; ignored
    fetch_portal_indices: set[int] | None = None,  # accepted; ignored
    mal_label: str = "",
) -> ScrapeResult:
    """
    Search AniPlaylist via the Algolia API for *query*.

    Parameters mirror the former Playwright-based scrape() so callers in
    main.py need no changes.  ``headless`` and ``fetch_portal_indices`` are
    accepted but have no effect — the API returns all data directly without
    portal dialogs.

    Parameters
    ----------
    query : str
        Search term (anime title, song name, …).
    headless : bool
        Ignored.  Kept for drop-in compatibility.
    fetch_portal_indices : set[int] | None
        Ignored.  Kept for drop-in compatibility.
    mal_label : str
        Optional "[MAL:id 'title']" prefix for log lines.

    Returns
    -------
    ScrapeResult
    """
    logger.info("%s scrape() called via Algolia API — query=%r", mal_label, query)

    all_hits: list[dict] = []

    async with aiohttp.ClientSession() as session:
        # Fetch page 0 first to learn total page count
        result0 = await _fetch_page(session, query, 0, mal_label)
        nb_pages: int = result0.get("nbPages", 1)
        nb_hits:  int = result0.get("nbHits", 0)
        all_hits.extend(result0.get("hits", []))

        logger.info(
            "%s Algolia: %d hits across %d page(s)",
            mal_label, nb_hits, nb_pages,
        )

        # Fetch remaining pages concurrently (still rate-limited per request)
        if nb_pages > 1:
            tasks = [
                _fetch_page(session, query, p, mal_label)
                for p in range(1, nb_pages)
            ]
            pages = await asyncio.gather(*tasks, return_exceptions=True)
            for i, page_result in enumerate(pages, start=1):
                if isinstance(page_result, Exception):
                    logger.warning(
                        "%s Algolia page %d failed: %s", mal_label, i, page_result
                    )
                else:
                    all_hits.extend(page_result.get("hits", []))

    logger.info("%s Algolia: fetched %d total hits", mal_label, len(all_hits))

    results: list[ResultItem] = [
        ResultItem(basic_data=_hit_to_basic(hit, idx))
        for idx, hit in enumerate(all_hits)
    ]

    return ScrapeResult(
        query=query,
        results=results,
        raw_html_snapshot="",   # no HTML — kept for API compat
    )
