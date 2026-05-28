"""AniPlaylist search bot.

Searches AniPlaylist, extracts result cards, filters by an exact anime-title match,
and stores query/output history in SQLite.

Usage:
    python aniplaylist_bot.py "Steins;Gate"
    python aniplaylist_bot.py "Steins;Gate" --db aniplaylist.sqlite3 --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin

import aiosqlite
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


@dataclass
class SearchResult:
    anime_title: str
    song_type: str
    sequence: int | None
    title: str
    artists: list[str]
    spotify_link: Optional[str]
    matched_query: bool


async def init_db(db_path: Path) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mal_id INTEGER,
                query TEXT NOT NULL,
                native_title TEXT,
                english_title TEXT,
                japanese_title TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id INTEGER NOT NULL,
                mal_id INTEGER,
                native_title TEXT,
                english_title TEXT,
                japanese_title TEXT,
                anime_title TEXT NOT NULL,
                song_type TEXT,
                sequence INTEGER,
                title TEXT,
                artists_json TEXT,
                spotify_link TEXT,
                matched_query INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(search_id) REFERENCES searches(id)
            )
            """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS failed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mal_id INTEGER,
                query TEXT NOT NULL,
                native_title TEXT,
                english_title TEXT,
                japanese_title TEXT,
                status TEXT,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS series (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                series_name TEXT NOT NULL,
                member_ids_json TEXT NOT NULL,
                member_count INTEGER NOT NULL,
                representative_mal_id INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """)
        cursor = await db.execute("PRAGMA table_info(searches)")
        existing_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        for column_name in (
            "mal_id",
            "native_title",
            "english_title",
            "japanese_title",
        ):
            if column_name not in existing_columns:
                column_type = "INTEGER" if column_name == "mal_id" else "TEXT"
                await db.execute(
                    f"ALTER TABLE searches ADD COLUMN {column_name} {column_type}"
                )

        cursor = await db.execute("PRAGMA table_info(results)")
        existing_result_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        for column_name in (
            "mal_id",
            "native_title",
            "english_title",
            "japanese_title",
            "sequence",
        ):
            if column_name not in existing_result_columns:
                column_type = (
                    "INTEGER" if column_name in {"mal_id", "sequence"} else "TEXT"
                )
                await db.execute(
                    f"ALTER TABLE results ADD COLUMN {column_name} {column_type}"
                )

        cursor = await db.execute("PRAGMA table_info(failed)")
        existing_failed_columns = {row[1] for row in await cursor.fetchall()}
        await cursor.close()

        for column_name in (
            "mal_id",
            "native_title",
            "english_title",
            "japanese_title",
            "status",
        ):
            if column_name not in existing_failed_columns:
                column_type = "INTEGER" if column_name == "mal_id" else "TEXT"
                await db.execute(
                    f"ALTER TABLE failed ADD COLUMN {column_name} {column_type}"
                )

        await db.commit()


async def save_run(
    db_path: Path,
    query: str,
    results: list[SearchResult],
    mal_id: int | None = None,
    native_title: str | None = None,
    english_title: str | None = None,
    japanese_title: str | None = None,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO searches(
                mal_id,
                query,
                native_title,
                english_title,
                japanese_title
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (mal_id, query, native_title, english_title, japanese_title),
        )
        search_id = cursor.lastrowid
        await cursor.close()

        for result in results:
            await db.execute(
                """
                INSERT INTO results(
                    search_id,
                    mal_id,
                    native_title,
                    english_title,
                    japanese_title,
                    anime_title,
                    song_type,
                    sequence,
                    title,
                    artists_json,
                    spotify_link,
                    matched_query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    search_id,
                    mal_id,
                    native_title,
                    english_title,
                    japanese_title,
                    result.anime_title,
                    result.song_type,
                    result.sequence,
                    result.title,
                    json.dumps(result.artists, ensure_ascii=False),
                    result.spotify_link,
                    1 if result.matched_query else 0,
                ),
            )
        await db.commit()


async def save_failure(
    db_path: Path,
    query: str,
    reason: str,
    mal_id: int | None = None,
    native_title: str | None = None,
    english_title: str | None = None,
    japanese_title: str | None = None,
    status: str | None = None,
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO failed(
                mal_id,
                query,
                native_title,
                english_title,
                japanese_title,
                status,
                reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mal_id,
                query,
                native_title,
                english_title,
                japanese_title,
                status,
                reason,
            ),
        )
        await db.commit()


def normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip().casefold()
    return value


def normalize_anime_query(query: str) -> str:
    value = normalize_text(query)
    for character in ("'", '"', "`", "‘", "’", "“", "”"):
        value = value.replace(character, "")
    value = re.sub(r"[^\w\s;:-]+", "", value)
    return value


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
    return {title.strip().lower() for title in titles if title and title.strip()}


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
        spotify_link: Optional[str] = None

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
                raw_artist = (await artist_nodes.nth(artist_index).inner_text()).strip()
                artist_values.append(normalize_text(raw_artist))
            artists = unique_non_empty(artist_values)

        spotify_anchor = card.locator(SPOTIFY_LINK_SELECTOR)
        if await spotify_anchor.count():
            href = await spotify_anchor.first.get_attribute("href")
            if href:
                spotify_link = urljoin(page.url, href)

        if not anime_title or not title:
            continue

        matched_query = anime_title.strip().lower() in exact_title_keys
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

            previous_first_title = ""
            previous_stats_text = ""
            previous_cards = page.locator(
                f"{RESULTS_CONTAINER_SELECTOR} {RESULT_CARD_SELECTOR}"
            )
            if await previous_cards.count():
                previous_title_loc = previous_cards.first.locator(ANIME_TITLE_SELECTOR)
                if await previous_title_loc.count():
                    previous_first_title = (
                        await previous_title_loc.first.inner_text()
                    ).strip()

            previous_stats = page.locator(RESULTS_STATS_SELECTOR)
            if await previous_stats.count():
                previous_stats_text = (await previous_stats.first.inner_text()).strip()

            # Some sessions show a cookie banner; ignore if it is not present.
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
                    """
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

                        return /\\d+/.test(currentStatsText);
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


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search AniPlaylist and extract result cards."
    )
    parser.add_argument("query", help="Anime title to search for")
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--json", action="store_true", help="Print results as JSON")
    parser.add_argument(
        "--no-exact-filter",
        action="store_true",
        help="Keep non-exact anime-title matches",
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run browser in headed mode"
    )
    args = parser.parse_args()

    await init_db(args.db)
    results = await search_aniplaylist(args.query, headless=not args.headed)

    if not args.no_exact_filter:
        filtered = [item for item in results if item.matched_query]
        # If no exact match is found, keep the raw results so the caller can inspect them.
        results = filtered or results

    await save_run(args.db, args.query, results)

    if args.json:
        print(
            json.dumps([asdict(item) for item in results], ensure_ascii=False, indent=2)
        )
        return

    if not results:
        print("No results found.")
        return

    for index, item in enumerate(results, start=1):
        print(f"[{index}] anime_title: {item.anime_title}")
        print(f"    type: {item.song_type}")
        print(f"    title: {item.title}")
        print(f"    artists: {item.artists}")
        print(f"    spotify_link: {item.spotify_link}")
        print(f"    matched_query: {item.matched_query}")


if __name__ == "__main__":
    asyncio.run(main())
