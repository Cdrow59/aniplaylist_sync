"""AniPlaylist search helpers."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

from playwright.async_api import TimeoutError as PlaywrightTimeoutError
from playwright.async_api import async_playwright

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


@dataclass(slots=True)
class SearchResult:
    anime_title: str
    song_type: str
    sequence: int | None
    title: str
    artists: list[str]
    spotify_link: str | None
    matched_query: bool


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def normalize_anime_query(query: str) -> str:
    value = normalize_text(query)
    for character in ("'", '"', "`", "‘", "’", "“", "”"):
        value = value.replace(character, "")
    return re.sub(r"[^\w\s;:-]+", "", value)


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
    return {normalize_text(title) for title in titles if title and title.strip()}


async def parse_results(
    page,
    query: str,
    exact_titles: Iterable[str] | None = None,
) -> list[SearchResult]:
    exact_title_keys = exact_title_candidates_lower(exact_titles or [query])
    results: list[SearchResult] = []

    cards = page.locator(f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}")
    card_count = await cards.count()

    for index in range(card_count):
        card = cards.nth(index)

        if await card.locator("p", has_text=UNRELEASED_NOTICE_TEXT).count():
            continue

        anime_title = ""
        song_type = ""
        sequence: int | None = None
        title = ""
        artists: list[str] = []
        spotify_link: str | None = None

        anime_title_loc = card.locator(ANIME_TITLE_SELECTOR)
        if await anime_title_loc.count():
            anime_title = (await anime_title_loc.first.inner_text()).strip()

        song_type_loc = card.locator(SONG_TYPE_SELECTOR)
        if await song_type_loc.count():
            song_type, sequence = parse_song_type(
                (await song_type_loc.first.inner_text()).strip()
            )

        title_loc = card.locator(SONG_TITLE_SELECTOR)
        if await title_loc.count():
            raw_title = (await title_loc.first.inner_text()).strip()
            title = normalize_text(raw_title)

        artists_container = card.locator(ARTISTS_CONTAINER_SELECTOR)
        if await artists_container.count():
            artist_nodes = artists_container.first.locator(ARTIST_SELECTOR)
            artist_count = await artist_nodes.count()
            artist_values: list[str] = []
            for artist_index in range(artist_count):
                artist_values.append(
                    (await artist_nodes.nth(artist_index).inner_text()).strip()
                )
            artists = unique_non_empty(artist_values)

        spotify_link_loc = card.locator(SPOTIFY_LINK_SELECTOR)
        if await spotify_link_loc.count():
            href = await spotify_link_loc.first.get_attribute("href")
            if href:
                spotify_link = urljoin(page.url, href)

        matched_query = (
            normalize_text(anime_title) in exact_title_keys if anime_title else False
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
            )
        )

    return results


async def load_all_result_cards(page, timeout_ms: int = 20000) -> None:
    cards = page.locator(f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}")
    previous_count = await cards.count()
    stable_rounds = 0
    deadline = asyncio.get_running_loop().time() + (timeout_ms / 1000)

    while asyncio.get_running_loop().time() < deadline:
        if previous_count == 0:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(750)
            current_count = await cards.count()
            if current_count > 0:
                previous_count = current_count
                stable_rounds = 0
            continue

        await cards.last.scroll_into_view_if_needed()
        await page.wait_for_timeout(750)

        current_count = await cards.count()
        if current_count == previous_count:
            stable_rounds += 1
            if stable_rounds >= 2:
                break
        else:
            previous_count = current_count
            stable_rounds = 0

    await page.evaluate("window.scrollTo(0, 0)")


async def search_aniplaylist(
    query: str,
    headless: bool = True,
    exact_titles: Iterable[str] | None = None,
) -> list[SearchResult]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(viewport={"width": 1200, "height": 1200})
        try:
            last_error: Exception | None = None
            for base_url in BASE_URLS:
                try:
                    await page.goto(base_url, wait_until="domcontentloaded")
                    break
                except Exception as error:
                    last_error = error
            else:
                assert last_error is not None
                raise last_error

            previous_stats_text = ""
            previous_stats = page.locator(RESULTS_STATS_SELECTOR)
            if await previous_stats.count():
                previous_stats_text = (await previous_stats.first.inner_text()).strip()

            for label in ("Accept all", "Accept", "Reject non-essential"):
                button = page.get_by_role("button", name=label)
                if await button.count():
                    await button.first.click(timeout=1500)
                    break

            search_box = page.locator(SEARCH_INPUT_SELECTOR)
            await search_box.wait_for(state="visible", timeout=15000)
            await search_box.click()
            if await search_box.is_enabled():
                await search_box.fill(query)
            else:
                await page.evaluate(
                    """
                    ({ selector, value }) => {
                        const element = document.querySelector(selector);
                        if (!element) {
                            throw new Error(`Search input not found: ${selector}`);
                        }
                        element.removeAttribute('disabled');
                        element.value = value;
                        element.dispatchEvent(new Event('input', { bubbles: true }));
                        element.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    """,
                    {"selector": SEARCH_INPUT_SELECTOR, "value": query},
                )
            await search_box.press("Enter")

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
                return await parse_results(page, query, exact_titles=exact_titles)

            await load_all_result_cards(page)
            return await parse_results(page, query, exact_titles=exact_titles)
        finally:
            if browser.is_connected():
                try:
                    await browser.close()
                except Exception:
                    pass
