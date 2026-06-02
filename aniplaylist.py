"""AniPlaylist search helpers."""

from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from playwright.async_api import Browser, BrowserContext
from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)

BASE_URLS = (
    "https://aniplaylist.com/",
    "https://www.aniplaylist.com/",
)
SEARCH_INPUT_SELECTOR = "#songSearch"
RESULTS_CONTAINER_SELECTOR = "div.ais-InfiniteHits"
RESULTS_STATS_SELECTOR = "div.ais-Stats.mt-1.text-white > strong"
RESULT_CARD_SELECTOR = (
    "div.relative.h-full.bg-white.rounded-lg.shadow-card.overflow-hidden.flex.flex-row"
)
ANIME_TITLE_SELECTOR = "div.relative.bg-gray-200.block.xl\\:hidden > div"
SONG_TYPE_SELECTOR = "div.xl\\:flex-1.my-2.md\\:my-0.lg\\:my-2.xl\\:my-0 > div.flex.xl\\:block.flex-wrap.xl\\:flex-nowrap > div.mr-1.flex.items-center > div > span.inline-block.xl\\:hidden"
SONG_TITLE_SELECTOR = "div.xl\\:flex-1.my-2.md\\:my-0.lg\\:my-2.xl\\:my-0 > div.flex.xl\\:block.flex-wrap.xl\\:flex-nowrap > div.text-sm.min-w-0.font-normal.flex.items-center.xl\\:flex-none.xl\\:flex-start.xl\\:text-lg.xl\\:mt-2 > span"
ARTISTS_CONTAINER_SELECTOR = "div.text-sm.xl\\:mt-1"
ARTIST_SELECTOR = "span.text-sgreen"
SPOTIFY_LINK_SELECTOR = r"div.flex-initial.xl\:mt-2 a[aria-label*='Spotify']"
UNRELEASED_NOTICE_TEXT = "Not yet released on streaming platforms."
DB_PATH = Path("aniplaylist.sqlite3")
ADVANCED_INFO_DIALOG_SELECTOR = "div[id^='headlessui-dialog-panel-']"
ADVANCED_INFO_SYNONYMS_SELECTOR = (
    "div.mt-4.flex.flex-col-reverse.sm\\:flex-row.w-full > "
    "div.w-full.md\\:w-2\\/3 > div:nth-child(1) > span"
)

TITLE_PUNCT_TRANSLATION = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "ʼ": "'",
        "＇": "'",
        "‛": "'",
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "＂": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        "―": "-",
        "−": "-",
        "﹘": "-",
        "﹣": "-",
        "－": "-",
        "～": "~",
        "〜": "~",
        "∼": "~",
        "！": "!",
        "﹗": "!",
        "？": "?",
        "﹖": "?",
        "：": ":",
        "；": ";",
        "，": ",",
        "．": ".",
    }
)


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


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def normalize_title_for_match(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold().strip()
    text = text.translate(TITLE_PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


# `normalize_anime_query` removed — use `normalize_title_for_match` or `normalize_text`
# depending on whether punctuation should be canonicalized or preserved.


def parse_song_type(raw_label: str) -> tuple[str, int | None]:
    label = normalize_text(raw_label)
    sequence_match = re.search(r"(\d+)$", label)
    sequence = int(sequence_match.group(1)) if sequence_match else None
    song_type = re.sub(r"\d+$", "", label).strip()
    return song_type, sequence


def unique_non_empty(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    collected: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        collected.append(cleaned)
    return collected


def exact_title_candidates_lower(titles: Iterable[str]) -> set[str]:
    return {
        normalize_title_for_match(title) for title in titles if title and title.strip()
    }


async def parse_results(
    page,
    query: str,
    exact_titles: Iterable[str] | None = None,
) -> list[SearchResult]:
    exact_title_keys = exact_title_candidates_lower(exact_titles or [query])
    cards = page.locator(f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}")
    raw_results = await cards.evaluate_all(f"""
        (elements) => elements.map((card) => {{
            const textContent = (selector) => {{
                const element = card.querySelector(selector);
                return element ? element.textContent.trim() : "";
            }};

            const unreleased = Array.from(card.querySelectorAll('p')).some(
                (element) => element.textContent.trim() === {UNRELEASED_NOTICE_TEXT!r}
            );
            const spotifyElement = card.querySelector({SPOTIFY_LINK_SELECTOR!r});

            return {{
                unreleased,
                anime_title: textContent({ANIME_TITLE_SELECTOR!r}),
                song_type_raw: textContent({SONG_TYPE_SELECTOR!r}),
                title_raw: textContent({SONG_TITLE_SELECTOR!r}),
                artist_values: Array.from(card.querySelectorAll({ARTIST_SELECTOR!r})).map(
                    (element) => element.textContent.trim()
                ),
                spotify_link: spotifyElement ? new URL(spotifyElement.getAttribute('href'), document.baseURI).href : null,
            }};
        }})
        """)

    results: list[SearchResult] = []
    for index, raw_result in enumerate(raw_results):
        if raw_result["unreleased"]:
            continue

        anime_title = raw_result["anime_title"]
        song_type, sequence = parse_song_type(raw_result["song_type_raw"])
        title = normalize_text(raw_result["title_raw"])
        artists = unique_non_empty(raw_result["artist_values"])
        spotify_link = raw_result["spotify_link"]
        matched_query = (
            normalize_title_for_match(anime_title) in exact_title_keys
            if anime_title
            else False
        )

        results.append(
            SearchResult(
                anime_title=anime_title,
                song_type=song_type,
                sequence=sequence,
                title=title,
                artists=artists,
                spotify_link=spotify_link,
                matched_query=matched_query,
                source_index=index,
            )
        )

    return results


def split_synonyms(raw_value: str) -> list[str]:
    return unique_non_empty(
        part for part in (item.strip() for item in raw_value.split(","))
    )


async def apply_advanced_synonym_matches(
    page, results: list[SearchResult], exact_titles: Iterable[str]
) -> None:
    exact_title_keys = exact_title_candidates_lower(exact_titles)
    cards = page.locator(f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}")

    for result in results:
        if result.matched_query or result.source_index is None:
            continue

        result.advanced_attempted = True
        card = cards.nth(result.source_index)
        advanced_button = card.locator("div.cursor-pointer i").first
        if not await advanced_button.count():
            advanced_button = card.locator("div.cursor-pointer").first

        try:
            await card.hover(timeout=2000)
        except Exception:
            pass

        try:
            await advanced_button.click(force=True, timeout=5000)
        except Exception as exc:
            result.advanced_error = str(exc)
            continue

        dialog = page.locator(ADVANCED_INFO_DIALOG_SELECTOR).first
        try:
            await dialog.wait_for(state="visible", timeout=5000)
            synonyms_locator = dialog.locator(ADVANCED_INFO_SYNONYMS_SELECTOR).first
            synonyms_text = (await synonyms_locator.text_content() or "").strip()
        except Exception as exc:
            result.advanced_error = str(exc)
            synonyms_text = ""
        finally:
            try:
                await page.keyboard.press("Escape")
                await page.wait_for_timeout(100)
            except Exception:
                pass

        if not synonyms_text:
            result.advanced_synonyms = []
            continue

        synonyms = split_synonyms(synonyms_text)
        result.advanced_synonyms = synonyms
        if not synonyms:
            continue

        matched_synonym = next(
            (
                synonym
                for synonym in synonyms
                if normalize_title_for_match(synonym) in exact_title_keys
            ),
            None,
        )
        if matched_synonym is not None:
            result.matched_query = True
            result.advanced_matched_synonym = matched_synonym


async def load_all_result_cards(page, timeout_ms: int = 20000) -> None:
    cards = page.locator(f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}")
    previous_count = await cards.count()
    stable_rounds = 0
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)

    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
    await page.wait_for_timeout(1000)

    while asyncio.get_running_loop().time() < deadline:
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(500)

        current_count = await cards.count()
        if current_count == previous_count:
            stable_rounds += 1
            if stable_rounds >= 2:
                break
        else:
            previous_count = current_count
            stable_rounds = 0


class BrowserPool:
    """Thread-safe browser context pool for efficient reuse and resource cleanup."""

    def __init__(self, max_concurrent: int = 1):
        """Initialize browser pool.

        Args:
            max_concurrent: Maximum concurrent browser contexts
        """
        self.max_concurrent = max_concurrent
        self._browser: Browser | None = None
        self._playwright = None
        self._contexts: list[BrowserContext] = []
        self._lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the browser pool (must be called before use)."""
        if self._initialized:
            return

        async with self._lock:
            if self._initialized:  # Double-check
                return

            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._initialized = True
            logger.info("Browser pool initialized")

    async def acquire_context(self) -> BrowserContext:
        """Acquire a browser context from the pool."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            if len(self._contexts) < self.max_concurrent:
                context = await self._browser.new_context(
                    viewport={"width": 1200, "height": 1200}
                )
                self._contexts.append(context)
                logger.debug(f"Created new context, pool size: {len(self._contexts)}")
                return context

        # Wait for a context to become available
        await asyncio.sleep(0.5)
        return await self.acquire_context()

    async def release_context(self, context: BrowserContext) -> None:
        """Release a browser context back to the pool."""
        if context in self._contexts:
            await context.close()
            self._contexts.remove(context)
            logger.debug(f"Released context, pool size: {len(self._contexts)}")

    async def close(self) -> None:
        """Close all contexts and the browser."""
        async with self._lock:
            for context in self._contexts[:]:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")
                self._contexts.remove(context)

            if self._browser:
                try:
                    await self._browser.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
                self._browser = None

            if self._playwright:
                await self._playwright.stop()
                self._playwright = None

            self._initialized = False
            logger.info("Browser pool closed")


# Global browser pool instance
_browser_pool: BrowserPool | None = None


async def get_browser_pool() -> BrowserPool:
    """Get the global browser pool instance."""
    global _browser_pool
    if _browser_pool is None:
        _browser_pool = BrowserPool(max_concurrent=1)
        await _browser_pool.initialize()
    return _browser_pool


async def close_browser_pool() -> None:
    """Close the global browser pool."""
    global _browser_pool
    if _browser_pool:
        await _browser_pool.close()
        _browser_pool = None


async def search_aniplaylist(
    query: str,
    headless: bool = True,
    exact_titles: Iterable[str] | None = None,
    advanced_fallback: bool = False,
    max_retries: int = 2,
) -> list[SearchResult]:
    """Search AniPlaylist with browser pooling and automatic retries.

    Args:
        query: Search query string
        headless: Whether to run browser in headless mode
        exact_titles: Titles to match exactly
        advanced_fallback: Whether to use advanced fallback search
        max_retries: Maximum number of retry attempts

    Returns:
        List of SearchResult objects

    Raises:
        RuntimeError: If search fails after all retries
    """
    exact_titles = tuple(exact_titles or (query,))

    for attempt in range(max_retries + 1):
        try:
            return await _search_aniplaylist_impl(
                query=query,
                headless=headless,
                exact_titles=exact_titles,
                advanced_fallback=advanced_fallback,
            )
        except Exception as e:
            if attempt < max_retries:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s, etc.
                logger.warning(
                    f"Search failed (attempt {attempt + 1}/{max_retries + 1}), "
                    f"retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Search failed after {max_retries + 1} attempts: {e}")
                raise RuntimeError(
                    f"Failed to search AniPlaylist for '{query}' after {max_retries + 1} attempts: {e}"
                ) from e


async def _search_aniplaylist_impl(
    query: str,
    headless: bool = True,
    exact_titles: Iterable[str] | None = None,
    advanced_fallback: bool = False,
) -> list[SearchResult]:
    """Internal implementation of search with browser pool reuse.

    This is called by search_aniplaylist with automatic retries.
    """
    pool = await get_browser_pool()
    context = await pool.acquire_context()
    page = None

    try:
        page = await context.new_page()
        logger.debug(f"Searching AniPlaylist for: {query}")

        last_error: Exception | None = None
        for base_url in BASE_URLS:
            try:
                await page.goto(base_url, wait_until="domcontentloaded", timeout=15000)
                break
            except Exception as error:
                last_error = error
                logger.debug(f"Failed to load {base_url}: {error}")
        else:
            assert last_error is not None
            raise last_error

        previous_stats_text = ""
        previous_stats = page.locator(RESULTS_STATS_SELECTOR)
        if await previous_stats.count():
            previous_stats_text = (await previous_stats.first.inner_text()).strip()

        # Accept cookie/consent dialogs
        for label in ("Accept all", "Accept", "Reject non-essential"):
            try:
                button = page.get_by_role("button", name=label)
                if await button.count():
                    await button.first.click(timeout=1500)
                    break
            except Exception as e:
                logger.debug(f"Could not click '{label}' button: {e}")
                continue

        search_box = page.locator(SEARCH_INPUT_SELECTOR)
        await search_box.wait_for(state="visible", timeout=15000)

        # Prefer filling the input when enabled; wait briefly for enablement.
        try:
            await page.wait_for_function(
                "(selector) => { const el = document.querySelector(selector); return !!el && !el.disabled; }",
                arg=SEARCH_INPUT_SELECTOR,
                timeout=5000,
            )
            await search_box.fill(query)
            await search_box.press("Enter")
        except PlaywrightTimeoutError:
            logger.debug("Search box enablement timed out, using fallback input method")
            # If the input remains disabled (site behavior / race), set the value
            # directly via DOM and dispatch events so the site reacts as if typed.
            await page.evaluate(
                """
                ({ selector, value }) => {
                    const element = document.querySelector(selector);
                    if (!element) {
                        throw new Error(`Search input not found: ${selector}`);
                    }
                    element.removeAttribute('disabled');
                    element.focus();
                    element.value = value;
                    element.dispatchEvent(new Event('input', { bubbles: true }));
                    element.dispatchEvent(new Event('change', { bubbles: true }));
                    element.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
                    element.dispatchEvent(new KeyboardEvent('keypress', { key: 'Enter', bubbles: true }));
                    element.dispatchEvent(new KeyboardEvent('keyup', { key: 'Enter', bubbles: true }));
                }
                """,
                {"selector": SEARCH_INPUT_SELECTOR, "value": query},
            )
            # allow site to process the injected events
            await page.wait_for_timeout(250)

        try:
            await page.wait_for_function(
                r"""
                ({ statsSelector, previousStatsText }) => {
                    const stats = document.querySelector(statsSelector);
                    if (!stats) {
                        return false;
                    }

                    const currentStatsText = stats.textContent.trim();
                    if (!currentStatsText) {
                        return false;
                    }

                    if (previousStatsText && currentStatsText === previousStatsText) {
                        return false;
                    }

                    return /\d+/.test(currentStatsText);
                }
                """,
                arg={
                    "statsSelector": RESULTS_STATS_SELECTOR,
                    "previousStatsText": previous_stats_text,
                },
                timeout=20000,
            )
        except PlaywrightTimeoutError:
            logger.debug(
                f"Results stats timeout for query '{query}', returning partial results"
            )
            return await parse_results(page, query, exact_titles=exact_titles)

        await load_all_result_cards(page)
        results = await parse_results(page, query, exact_titles=exact_titles)

        if advanced_fallback and exact_titles is not None and results:
            if not any(result.matched_query for result in results):
                logger.debug(f"Using advanced fallback for query '{query}'")
                await apply_advanced_synonym_matches(page, results, exact_titles)

        logger.debug(f"Search completed for '{query}': {len(results)} results found")
        return results
    finally:
        try:
            if page:
                await page.close()
        except Exception as e:
            logger.warning(f"Error closing page: {e}")

        try:
            await pool.release_context(context)
        except Exception as e:
            logger.warning(f"Error releasing context: {e}")
