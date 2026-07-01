"""
parser.py — Convert ScrapeResult into SearchResult objects.

Public API
----------
    records = parse(scrape_result, exact_titles=["Naruto", ...])

portal_data key presence:
    absent       — portal was never requested for this result
                   → advanced_* fields left at defaults
    {synonyms}   — alternate anime titles returned by Algolia directly;
                   synonym matching applied
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from aniplaylist import BasicData, ResultItem, ScrapeResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SearchResult:
    anime_title: str
    song_type: str
    sequence: int | None
    title: str
    artists: list[str]
    spotify_link: str | None
    matched_query: bool
    source_index: int | None = None
    advanced_attempted: bool = False
    advanced_synonyms: list[str] | None = None
    advanced_matched_synonym: str | None = None
    advanced_error: str | None = None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u02bc": "'",
    "\uff07": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u201f": '"',
    "\uff02": '"',
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2015": "-",
    "\u2212": "-",
    "\ufe58": "-",
    "\ufe63": "-",
    "\uff0d": "-",
    "\uff5e": "~",
    "\u301c": "~",
    "\u223c": "~",
    "\uff01": "!",
    "\ufe57": "!",
    "\uff1f": "?",
    "\ufe56": "?",
    "\uff1a": ":",
    "\uff1b": ";",
    "\uff0c": ",",
    "\uff0e": ".",
    "\u3000": " ",
}

_SUFFIX_RE = re.compile(r"\s*\((tv|ona|ova|movie|special)\)\s*$", re.IGNORECASE)
_PUNCT_RE = re.compile("[" + re.escape("".join(_PUNCT_REPLACEMENTS)) + "]")
_WHITESPACE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[^\w]+")


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    return _WHITESPACE.sub(" ", text).strip().casefold()


def normalize_for_match(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = strip_media_suffixes(text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_REPLACEMENTS[m.group()], text)
    return _WHITESPACE.sub(" ", text).strip()


def exact_key_set(titles: Iterable[str]) -> set[str]:
    return {normalize_for_match(t) for t in titles if t and t.strip()}


def strip_media_suffixes(value: str) -> str:
    return _SUFFIX_RE.sub("", value).strip()


def dedup(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        v = v.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def parse_song_type(basic: "BasicData") -> tuple[str, int | None]:
    """Return (song_type, sequence) using the pre-parsed scraper fields."""
    kind = normalize_text(basic["song_type_raw"])
    seq = basic.get("song_type_sequence")  # type: ignore[typeddict-item]
    return kind, seq


# ---------------------------------------------------------------------------
# Synonym scoring
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> set[str]:
    """Split a normalized string into a set of non-empty word tokens."""
    return {t for t in _TOKEN_RE.split(text) if t}


def _synonym_score(synonym: str, query_tokens: set[str]) -> tuple[int, int]:
    """
    Score a synonym for preference when multiple synonyms match exact_keys.

    Returns (overlap, length) — both higher is better, sort descending.
    """
    norm = normalize_for_match(synonym)
    overlap = len(_tokenize(norm) & query_tokens)
    return overlap, len(norm)


def _best_matching_synonym(
    synonyms: list[str],
    exact_keys: set[str],
    query: str,
) -> str | None:
    """
    Return the synonym from *synonyms* that normalizes to a key in *exact_keys*,
    preferring the one with the most token overlap with *query*.
    """
    query_tokens = _tokenize(normalize_for_match(query))
    candidates = [s for s in synonyms if normalize_for_match(s) in exact_keys]
    if not candidates:
        return None
    return max(candidates, key=lambda s: _synonym_score(s, query_tokens))


# ---------------------------------------------------------------------------
# Per-item conversion
# ---------------------------------------------------------------------------


def _build_result(
    basic: BasicData,
    exact_keys: set[str],
    portal: dict | None,
    portal_requested: bool,
    query: str,
) -> SearchResult | None:
    """
    Parameters
    ----------
    portal_requested : bool
        True  — portal_data key was present in the ResultItem (may be None or dict).
        False — portal was never fetched for this result; skip advanced logic.
    query : str
        The search string submitted to Algolia. Used to score synonyms.
    """
    if basic["unreleased"]:
        return None

    title = normalize_text(basic["title_raw"])
    if not title:
        return None

    anime_title = basic["anime_title"]
    song_type, sequence = parse_song_type(basic)
    artists = dedup(basic["artist_values"])
    spotify_link = basic["spotify_link"] or None
    source_index = basic["source_index"]

    matched_query = bool(anime_title and normalize_for_match(anime_title) in exact_keys)
    advanced_attempted = False
    advanced_synonyms: list[str] | None = None
    advanced_matched_synonym: str | None = None
    advanced_error: str | None = None

    if not matched_query and portal_requested and portal:
        synonyms: list[str] = portal.get("synonyms") or []
        if synonyms:
            advanced_attempted = True
            advanced_synonyms = synonyms
            hit = _best_matching_synonym(synonyms, exact_keys, query)
            if hit is not None:
                matched_query = True
                advanced_matched_synonym = hit

    return SearchResult(
        anime_title=anime_title,
        song_type=song_type,
        sequence=sequence,
        title=title,
        artists=artists,
        spotify_link=spotify_link,
        matched_query=matched_query,
        source_index=source_index,
        advanced_attempted=advanced_attempted,
        advanced_synonyms=advanced_synonyms,
        advanced_matched_synonym=advanced_matched_synonym,
        advanced_error=advanced_error,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse(
    scrape_result: ScrapeResult,
    exact_titles: Iterable[str] | None = None,
) -> list[SearchResult]:
    """
    Convert a ScrapeResult into a flat list of SearchResult objects.

    Parameters
    ----------
    scrape_result : ScrapeResult
        Output of ``AlgoliaClient.scrape()``.
    exact_titles : iterable of str, optional
        Candidate anime titles used to determine ``matched_query``.
        Defaults to the original search query when omitted.
    """
    query = scrape_result["query"]
    candidates = list(exact_titles) if exact_titles is not None else [query]
    exact_keys = exact_key_set(candidates)

    results: list[SearchResult] = []
    dropped = 0

    for item in scrape_result["results"]:
        portal_requested = "portal_data" in item
        portal = item.get("portal_data")

        record = _build_result(
            basic=item["basic_data"],
            exact_keys=exact_keys,
            portal=portal,
            portal_requested=portal_requested,
            query=query,
        )
        if record is None:
            dropped += 1
        else:
            results.append(record)

    logger.debug(
        "parse(): query=%r  in=%d  kept=%d  dropped=%d  matched=%d",
        query,
        len(scrape_result["results"]),
        len(results),
        dropped,
        sum(1 for r in results if r.matched_query),
    )
    return results
