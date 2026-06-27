"""
scraper.py — AniPlaylist scraper module.

Public API
----------
    # Phase 1 only (static card data, no portal clicks):
    result = await scrape(query, headless=True)

    # Phase 1 + portal extraction for specific card indices:
    result = await scrape(query, headless=True, fetch_portal_indices={2, 5, 7})

ScrapeResult:
    {
        "query":             str,
        "results":           list[ResultItem],
        "raw_html_snapshot": str,
    }

ResultItem:
    {
        "basic_data":  BasicData,
        "portal_data": dict | None,
            # None  — extraction attempted, all retries failed
            # {}    — no info button found on this card
            # {...} — synonyms (and optional error) extracted
            # not present in key when fetch_portals was False for this card
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
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict
from urllib.parse import quote

from playwright.async_api import (
    Browser,
    BrowserContext,
)
from playwright.async_api import Error as PWError
from playwright.async_api import (
    Page,
)
from playwright.async_api import TimeoutError as PWTimeout
from playwright.async_api import (
    async_playwright,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Selectors
# ---------------------------------------------------------------------------

BASE_URLS = (
    "https://aniplaylist.com/",
    "https://www.aniplaylist.com/",
)
SEARCH_INPUT_SEL = "#songSearch"
RESULTS_CONTAINER_SEL = "div.ais-InfiniteHits"
RESULTS_STATS_SEL = "div.ais-Stats.mt-1.text-white > strong"
RESULT_CARD_SEL = (
    "div.relative.h-full.bg-white.rounded-lg"
    ".shadow-card.overflow-hidden.flex.flex-row"
)
ANIME_TITLE_SEL = "div.relative.bg-gray-200.block.xl\\:hidden > div"
SONG_TYPE_SEL = (
    "div.xl\\:flex-1.my-2.md\\:my-0.lg\\:my-2.xl\\:my-0 > "
    "div.flex.xl\\:block.flex-wrap.xl\\:flex-nowrap > "
    "div.mr-1.flex.items-center > div > "
    "span.inline-block.xl\\:hidden"
)
SONG_TITLE_SEL = (
    "div.xl\\:flex-1.my-2.md\\:my-0.lg\\:my-2.xl\\:my-0 > "
    "div.flex.xl\\:block.flex-wrap.xl\\:flex-nowrap > "
    "div.text-sm.min-w-0.font-normal.flex.items-center"
    ".xl\\:flex-none.xl\\:flex-start.xl\\:text-lg.xl\\:mt-2 > span"
)
ARTIST_SEL = "span.text-sgreen"
SPOTIFY_LINK_SEL = r"div.flex-initial.xl\:mt-2 a[aria-label*='Spotify']"
UNRELEASED_TEXT = "Not yet released on streaming platforms."

# Dialog
PORTAL_SEL = "div[data-headlessui-state='open']"
DIALOG_PANEL_SEL = "div[id^='headlessui-dialog-panel-']"
DIALOG_CLOSE_BTN_SEL = "div[id^='headlessui-dialog-panel-'] > button"
SYNONYMS_SEL = "div[id^='headlessui-dialog-panel-'] span.text-xs.italic"
INFO_BTN_SEL = "div.absolute.inset-0.cursor-pointer[tabindex='0'] span.absolute"
INFO_BTN_FALLBACK_SEL = "div.absolute.inset-0.cursor-pointer[tabindex='0']"

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

SCROLL_PAUSE_S = 0.7
SCROLL_MAX_ITER = 60
SCROLL_STABLE_REPS = 3

STATS_TIMEOUT_MS = 20_000
PORTAL_TIMEOUT_MS = 8_000
PORTAL_STABLE_S = 0.4
PORTAL_CLOSE_S = 0.4
CARD_HOVER_S = 0.25
RETRY_COUNT = 3
RETRY_BACKOFF_S = 1.0


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------


class BasicData(TypedDict):
    anime_title: str
    song_type_raw: str
    title_raw: str
    artist_values: list[str]
    spotify_link: str | None
    source_index: int
    unreleased: bool


class ResultItem(TypedDict, total=False):
    basic_data: BasicData  # always present
    portal_data: dict | None  # present only when portal was requested


class ScrapeResult(TypedDict):
    query: str
    results: list[ResultItem]
    raw_html_snapshot: str


# ---------------------------------------------------------------------------
# Phase 1 — navigate, search, scroll
# ---------------------------------------------------------------------------


async def _type_query_and_wait(page: Page, query: str) -> None:
    previous_stats_text = ""
    stats_loc = page.locator(RESULTS_STATS_SEL)
    if await stats_loc.count():
        previous_stats_text = (await stats_loc.first.inner_text()).strip()

    for label in ("Accept all", "Accept", "Reject non-essential"):
        try:
            btn = page.get_by_role("button", name=label)
            if await btn.count():
                await btn.first.click(timeout=1500)
                break
        except Exception:
            pass

    search_box = page.locator(SEARCH_INPUT_SEL)
    await search_box.wait_for(state="visible", timeout=15_000)

    try:
        await page.wait_for_function(
            "(sel) => { const el = document.querySelector(sel); return !!el && !el.disabled; }",
            arg=SEARCH_INPUT_SEL,
            timeout=5_000,
        )
        await search_box.fill(query)
        await search_box.press("Enter")
    except PWTimeout:
        logger.debug("Search box enablement timed out — using DOM injection fallback")
        await page.evaluate(
            """
            ({ selector, value }) => {
                const el = document.querySelector(selector);
                if (!el) throw new Error("Search input not found: " + selector);
                el.removeAttribute("disabled");
                el.focus();
                el.value = value;
                el.dispatchEvent(new Event("input",  { bubbles: true }));
                el.dispatchEvent(new Event("change", { bubbles: true }));
                ["keydown", "keypress", "keyup"].forEach(type =>
                    el.dispatchEvent(new KeyboardEvent(type, { key: "Enter", bubbles: true }))
                );
            }
            """,
            {"selector": SEARCH_INPUT_SEL, "value": query},
        )
        await asyncio.sleep(0.25)

    # Wait for stats to reflect the new query
    try:
        await page.wait_for_function(
            r"""
            ({ statsSelector, previousStatsText }) => {
                const el = document.querySelector(statsSelector);
                if (!el) return false;
                const text = el.textContent.trim();
                if (!text) return false;
                if (previousStatsText && text === previousStatsText) return false;
                return /\d+/.test(text);
            }
            """,
            arg={
                "statsSelector": RESULTS_STATS_SEL,
                "previousStatsText": previous_stats_text,
            },
            timeout=STATS_TIMEOUT_MS,
        )
    except PWTimeout:
        logger.debug("Stats timeout for '%s' — continuing with partial results", query)
        return

    # Wait for DOM cards to be consistent with the new stats count.
    # 0 stats → wait until 0 cards (old cards flushed).
    # >0 stats → wait until at least 1 card is present and count <= expected
    #            (guards against reading a stale prior result set).
    try:
        full_card_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
        await page.wait_for_function(
            r"""
            ({ statsSel, cardSel }) => {
                const statsEl = document.querySelector(statsSel);
                if (!statsEl) return false;
                const match = statsEl.textContent.match(/(\d+)/);
                if (!match) return false;
                const expected = parseInt(match[1], 10);
                const actual = document.querySelectorAll(cardSel).length;
                if (expected === 0) return actual === 0;
                return actual > 0 && actual <= expected;
            }
            """,
            arg={"statsSel": RESULTS_STATS_SEL, "cardSel": full_card_sel},
            timeout=STATS_TIMEOUT_MS,
        )
        logger.debug("Results settled for '%s'", query)
    except PWTimeout:
        logger.debug(
            "Result-settle timeout for '%s' — continuing with whatever is in DOM", query
        )


async def _scroll_until_stable(page: Page) -> list[BasicData]:
    full_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"

    seen: dict[tuple, BasicData] = {}
    stable = 0
    previous_count = 0

    for i in range(SCROLL_MAX_ITER):
        cards = page.locator(full_sel)

        raw_cards = await cards.evaluate_all(f"""
        (cards) => cards.map((card, index) => {{
            const text = (sel) => {{
                const el = card.querySelector(sel);
                return el ? el.textContent.trim() : "";
            }};

            const spotifyEl =
                card.querySelector({SPOTIFY_LINK_SEL!r});

            return {{
                anime_title: text({ANIME_TITLE_SEL!r}),
                song_type_raw: text({SONG_TYPE_SEL!r}),
                title_raw: text({SONG_TITLE_SEL!r}),

                artist_values:
                    Array.from(
                        card.querySelectorAll({ARTIST_SEL!r})
                    ).map(el => el.textContent.trim()),

                spotify_link:
                    spotifyEl
                    ? new URL(
                        spotifyEl.getAttribute("href"),
                        document.baseURI
                      ).href
                    : null,

                unreleased:
                    Array.from(card.querySelectorAll("p"))
                    .some(
                        el =>
                        el.textContent.trim() ===
                        {UNRELEASED_TEXT!r}
                    )
            }};
        }})
        """)

        for index, raw in enumerate(raw_cards):
            item = BasicData(
                **raw,
                source_index=len(seen),
            )

            key = (
                item["anime_title"],
                item["title_raw"],
                item["spotify_link"],
            )

            seen[key] = item

        logger.debug(
            "Scroll %d — DOM=%d collected=%d",
            i,
            len(raw_cards),
            len(seen),
        )

        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await asyncio.sleep(SCROLL_PAUSE_S)

        if len(seen) == previous_count:
            stable += 1
            if stable >= SCROLL_STABLE_REPS:
                logger.info(
                    "Scroll stable at %d cards after %d steps",
                    len(seen),
                    i + 1,
                )
                break
        else:
            previous_count = len(seen)
            stable = 0

    return list(seen.values())


# ---------------------------------------------------------------------------
# Phase 2 — portal close (guaranteed) + extraction
# ---------------------------------------------------------------------------


async def _force_close_portal(page: Page, ctx: str = "") -> None:
    """
    Unconditionally close any open headlessui dialog.
    Guaranteed to leave the page in a stable, portal-free state.

    Close priority:
      1. Known close button: first <button> inside the dialog panel
      2. Escape key
      3. Backdrop click (to the left of the panel)

    Each strategy is attempted in turn and verified; if one succeeds we stop.
    A final verification wait ensures the portal is gone before returning.
    """
    # Strategy 1 — known close button (confirmed selector)
    close_btn = await page.query_selector(DIALOG_CLOSE_BTN_SEL)
    if close_btn:
        try:
            await close_btn.click(timeout=2_000)
            await asyncio.sleep(PORTAL_CLOSE_S)
        except Exception:
            pass

    if not await page.query_selector(PORTAL_SEL):
        return

    # Strategy 2 — Escape key
    await page.keyboard.press("Escape")
    await asyncio.sleep(PORTAL_CLOSE_S)

    if not await page.query_selector(PORTAL_SEL):
        return

    # Strategy 3 — backdrop click (left of panel)
    panel = await page.query_selector(DIALOG_PANEL_SEL)
    if panel:
        box = await panel.bounding_box()
        if box:
            await page.mouse.click(
                max(0.0, box["x"] - 50),
                box["y"] + box["height"] / 2,
            )
            await asyncio.sleep(PORTAL_CLOSE_S)

    if await page.query_selector(PORTAL_SEL):
        logger.warning(
            "%s Portal still present after all close strategies — page may be stuck",
            ctx,
        )
        return

    # Final: wait for portal to disappear from DOM entirely
    try:
        await page.wait_for_selector(PORTAL_SEL, state="hidden", timeout=3_000)
    except PWTimeout:
        pass


async def _extract_synonyms(page: Page) -> tuple[list[str], str | None]:
    try:
        await page.wait_for_selector(
            DIALOG_PANEL_SEL, state="visible", timeout=PORTAL_TIMEOUT_MS
        )
    except PWTimeout:
        return [], "dialog panel never became visible"

    await asyncio.sleep(PORTAL_STABLE_S)

    try:
        raw = (await page.locator(SYNONYMS_SEL).nth(0).text_content() or "").strip()
    except Exception as exc:
        return [], f"synonym text_content failed: {exc}"

    if not raw:
        return [], None

    return [p.strip() for p in raw.split(",") if p.strip()], None


async def _extract_portal_for_card(
    page: Page,
    index: int,
    ctx: str = "",
) -> dict | None:
    """
    Open the info dialog for the card at *index*, extract portal data, then
    close it.

    Uses a lazy Locator (re-evaluated on each interaction) so virtual-scroll
    rendering is triggered automatically and stale-handle errors are avoided.

    The close is performed in a `finally` block so the portal is always
    dismissed before returning — even on timeout or exception.

    Returns:
        {}                        — info button not found on this card
        {"synonyms": [...], ...}  — successful extraction
        None                      — all retries exhausted
    """
    full_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
    # Locator is lazy — re-evaluates on each interaction and scrolling into
    # view triggers virtual-scroll rendering, so index is always valid.
    card_loc = page.locator(full_sel).nth(index)

    for attempt in range(1, RETRY_COUNT + 1):
        portal_opened = False
        try:
            await card_loc.scroll_into_view_if_needed()

            try:
                await card_loc.hover(timeout=2_000)
            except Exception:
                pass
            await asyncio.sleep(CARD_HOVER_S)

            info_loc = card_loc.locator(INFO_BTN_SEL).first
            if not await info_loc.count():
                info_loc = card_loc.locator(INFO_BTN_FALLBACK_SEL).first
            if not await info_loc.count():
                logger.debug("%s Card %d — no info button found", ctx, index)
                return {}

            await info_loc.scroll_into_view_if_needed()
            await info_loc.click(timeout=2_000)
            await page.wait_for_selector(
                PORTAL_SEL, state="visible", timeout=PORTAL_TIMEOUT_MS
            )
            portal_opened = True

            synonyms, error = await _extract_synonyms(page)
            portal_data: dict = {"synonyms": synonyms}
            if error:
                portal_data["error"] = error

            logger.debug(
                "%s Card %d — portal OK attempt %d, synonyms=%s",
                ctx,
                index,
                attempt,
                synonyms,
            )
            return portal_data

        except PWTimeout:
            logger.warning(
                "%s Card %d — timeout attempt %d/%d", ctx, index, attempt, RETRY_COUNT
            )
        except Exception as exc:
            logger.warning(
                "%s Card %d — error attempt %d/%d: %s",
                ctx,
                index,
                attempt,
                RETRY_COUNT,
                exc,
            )

        finally:
            if portal_opened or await page.query_selector(PORTAL_SEL):
                await _force_close_portal(page, ctx=ctx)

        await asyncio.sleep(RETRY_BACKOFF_S * attempt)

    logger.error(
        "%s Card %d — all %d retries exhausted; portal_data=None",
        ctx,
        index,
        RETRY_COUNT,
    )
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def scrape(
    query: str,
    headless: bool = True,
    fetch_portal_indices: set[int] | None = None,
    mal_label: str = "",
) -> ScrapeResult:
    """
    Scrape AniPlaylist for *query*.

    Parameters
    ----------
    query : str
        Search term passed to aniplaylist.com.
    headless : bool
        True  → headless Chromium (default).
        False → visible browser (debugging).
    fetch_portal_indices : set[int] | None
        When None (default) — Phase 2 is skipped entirely; portal_data is
        absent from all ResultItems.
        When a set — only cards at those source_index positions have their
        portal opened and extracted.  All other cards get no portal_data key.
        Pass an empty set to skip all portals explicitly.
    mal_label : str
        Optional "[MAL:id 'title']" prefix included in every log line
        emitted during this scrape call. Pass from main.py for traceability.

    Returns
    -------
    ScrapeResult
    """

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        context: BrowserContext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1200, "height": 1200},
        )
        page: Page = await context.new_page()
        raw_html_snapshot = ""
        results: list[ResultItem] = []

        try:
            # ── Phase 1 ──────────────────────────────────────────────────
            # Navigate directly to the search URL so results are scoped to
            # this query from page load — no stale-card race from typing
            # into the search box on the homepage.
            search_url = f"https://aniplaylist.com/{quote(query, safe='')}"
            last_err: Exception | None = None
            for url in (
                search_url,
                f"https://www.aniplaylist.com/{quote(query, safe='')}",
            ):
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
            if last_err:
                raise last_err

            # Wait for stats to appear and cards to settle
            try:
                await page.wait_for_function(
                    r"""
                    ({ statsSel, cardSel }) => {
                        const statsEl = document.querySelector(statsSel);
                        if (!statsEl) return false;
                        const match = statsEl.textContent.match(/(\d+)/);
                        if (!match) return false;
                        const expected = parseInt(match[1], 10);
                        if (expected === 0) return true;
                        return document.querySelectorAll(cardSel).length > 0;
                    }
                    """,
                    arg={
                        "statsSel": RESULTS_STATS_SEL,
                        "cardSel": f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}",
                    },
                    timeout=STATS_TIMEOUT_MS,
                )
            except PWTimeout:
                logger.debug(
                    "%s Stats/card timeout for '%s' — continuing", mal_label, query
                )

            raw_html_snapshot = await page.content()

            basic_data_list = await _scroll_until_stable(page)
            logger.info(
                "%s Phase 1 complete — %d cards", mal_label, len(basic_data_list)
            )

            for basic in basic_data_list:
                results.append(ResultItem(basic_data=basic))

            # ── Phase 2 — only when indices requested ────────────────────
            if fetch_portal_indices is None:
                logger.debug(
                    "%s Phase 2 skipped — fetch_portal_indices not set", mal_label
                )
            else:
                targets = sorted(
                    idx for idx in fetch_portal_indices if idx < len(results)
                )
                logger.info(
                    "%s Phase 2 — fetching portals for %d/%d cards",
                    mal_label,
                    len(targets),
                    len(results),
                )
                for idx in targets:
                    card_title = (
                        results[idx]["basic_data"].get("anime_title") or f"card {idx}"
                    )
                    logger.info(
                        "%s Portal %d/%d — card_title='%s'",
                        mal_label,
                        idx + 1,
                        len(results),
                        card_title,
                    )
                    results[idx]["portal_data"] = await _extract_portal_for_card(
                        page, idx, ctx=mal_label
                    )

                logger.info("%s Phase 2 complete", mal_label)

        finally:
            try:
                await page.close()
            except PWError as e:
                logger.error("Encountered Playwright error during page close: %s", e)
            try:
                await browser.close()
            except PWError as e:
                logger.error("Encountered Playwright error during browser close: %s", e)

    return ScrapeResult(
        query=query,
        results=results,
        raw_html_snapshot=raw_html_snapshot,
    )
