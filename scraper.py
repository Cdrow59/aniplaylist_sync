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
            # key absent when fetch_portals was False for this card
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
import random
import time
from enum import Enum, auto
from typing import TypedDict

from playwright.async_api import (
    Browser,
    BrowserContext,
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

NO_RESULTS_SEL = "div.my-5 > p"
NO_RESULTS_MARKER = "Sorry, we couldn't find any song or album for your query."

# Dialog
PORTAL_SEL = "div[data-headlessui-state='open']"
DIALOG_PANEL_SEL = "div[id^='headlessui-dialog-panel-']"
DIALOG_CLOSE_BTN_SEL = "div[id^='headlessui-dialog-panel-'] > button"
SYNONYMS_SEL = "div[id^='headlessui-dialog-panel-'] span.text-xs.italic"
INFO_BTN_SEL = "div.cursor-pointer i"
INFO_BTN_FALLBACK_SEL = "div.cursor-pointer"

# ---------------------------------------------------------------------------
# Timing  — all values deliberately "human-paced"
# ---------------------------------------------------------------------------

# Scroll: Algolia's virtual-scroll loader needs time to fire and hydrate.
# 0.7s was causing the scraper to outrun the network requests.
SCROLL_PAUSE_S = 1.4  # was 0.7 — give Algolia time to actually load

SCROLL_MAX_ITER = 60
SCROLL_STABLE_REPS = 3

# How long to wait for the stats counter to appear after submitting a query.
# Algolia can be slow on cold cache; bumped from 20s → 30s.
STATS_TIMEOUT_MS = 30_000  # was 20_000

# Card quiesce: Algolia fires the stats update before the DOM is hydrated.
# 8s was too tight on slow connections; 14s gives React time to render.
CARDS_QUIESCE_TIMEOUT_MS = 14_000  # was 8_000
CARDS_QUIESCE_POLL_MS = 400  # was 200 — less aggressive polling

# Portal dialogs: HeadlessUI transitions take ~150-300ms on real sites.
# Previous values (0.4s) were firing the synonym scrape during the animation.
PORTAL_TIMEOUT_MS = 12_000  # was 8_000 — more room for slow responses
PORTAL_STABLE_S = 0.9  # was 0.4 — wait for dialog animation to finish
PORTAL_CLOSE_S = 0.8  # was 0.4 — wait for close animation

# Card hover: some cards lazy-load content on hover; 0.25s wasn't enough.
CARD_HOVER_S = 0.6  # was 0.25

# Retry policy for Phase 1.
SCRAPE_MAX_ATTEMPTS = 4
SCRAPE_BACKOFF_BASE_S = 4.0

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

# Minimum seconds between the end of one browser session and the start of the
# next.  Prevents hammering AniPlaylist's Algolia endpoint back-to-back.
INTER_SESSION_DELAY_MIN_S = 2.0
INTER_SESSION_DELAY_MAX_S = 4.5

# Minimum gap between opening individual portals inside Phase 2.
# Mimics a human pausing before clicking the next card's info button.
INTER_PORTAL_DELAY_MIN_S = 0.8
INTER_PORTAL_DELAY_MAX_S = 2.2

# Delay injected before submitting the search query, simulating a human
# landing on the page and then typing/pasting.
PRE_SEARCH_DELAY_MIN_S = 0.6
PRE_SEARCH_DELAY_MAX_S = 1.8

# Delay after page navigation before we start interacting with anything.
POST_NAV_DELAY_MIN_S = 0.8
POST_NAV_DELAY_MAX_S = 2.0

# Delay after filling the search box and before pressing Enter.
# Simulates a human glancing at what they typed before submitting.
POST_FILL_DELAY_MIN_S = 0.5
POST_FILL_DELAY_MAX_S = 1.4

# Delay after pressing Enter — simulates a human waiting to see the page react
# before doing anything else (e.g. scrolling, clicking).
POST_ENTER_DELAY_MIN_S = 0.8
POST_ENTER_DELAY_MAX_S = 1.8

# Jitter added to each scroll pause so the scroll rhythm doesn't look robotic.
# Applied as: actual_pause = SCROLL_PAUSE_S + uniform(SCROLL_JITTER_MIN_S, SCROLL_JITTER_MAX_S)
SCROLL_JITTER_MIN_S = -0.2
SCROLL_JITTER_MAX_S = 0.5
SCROLL_FLOOR_S = 0.9  # absolute minimum pause per scroll step

# Pause after scroll_into_view — gives the card time to fully enter the
# viewport and trigger any lazy-load before we hover or click.
POST_SCROLL_INTO_VIEW_MIN_S = 0.4
POST_SCROLL_INTO_VIEW_MAX_S = 0.9

# Jitter on portal close waits so close timing isn't perfectly mechanical.
PORTAL_CLOSE_JITTER_MIN_S = 0.0
PORTAL_CLOSE_JITTER_MAX_S = 0.4

# Jitter on portal stable wait (after dialog opens, before reading content).
PORTAL_STABLE_JITTER_MIN_S = 0.0
PORTAL_STABLE_JITTER_MAX_S = 0.3

# Jitter added to retry backoff sleeps to avoid thundering-herd on concurrent
# workers.  Applied as: actual_backoff = base_backoff + uniform(min, max)
RETRY_JITTER_MIN_S = 0.5
RETRY_JITTER_MAX_S = 2.5

# Track the wall-clock time the last browser session finished so we can
# enforce INTER_SESSION_DELAY even across Phase 1 → Phase 2 boundaries.
_last_session_end: float = 0.0


async def _inter_session_pause(label: str = "") -> None:
    """
    Sleep long enough that at least INTER_SESSION_DELAY_MIN_S has elapsed
    since the previous browser session ended, then add random extra jitter.

    This is the primary rate-limit guard: it fires before every browser.launch()
    call, including Phase 2 re-navigation, so AniPlaylist never sees two
    Algolia search requests back-to-back.
    """
    global _last_session_end
    elapsed = time.monotonic() - _last_session_end
    base_gap = random.uniform(INTER_SESSION_DELAY_MIN_S, INTER_SESSION_DELAY_MAX_S)
    wait = max(0.0, base_gap - elapsed)
    if wait > 0:
        logger.debug(
            "%s Rate-limit pause: %.2fs (elapsed since last session: %.2fs)",
            label,
            wait,
            elapsed,
        )
        await asyncio.sleep(wait)


def _record_session_end() -> None:
    global _last_session_end
    _last_session_end = time.monotonic()


async def _human_pause(
    min_s: float,
    max_s: float,
    label: str = "",
) -> None:
    """Sleep for a random duration in [min_s, max_s] to mimic human timing."""
    delay = random.uniform(min_s, max_s)
    logger.debug("%s Human pause: %.2fs", label, delay)
    await asyncio.sleep(delay)


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
# Internal outcome enum for Phase 1
# ---------------------------------------------------------------------------


class _Phase1Outcome(Enum):
    SUCCESS = auto()  # got ≥1 card belonging to this query
    ZERO_RESULTS = auto()  # genuine empty search; do not retry
    TRANSIENT = auto()  # timing/rate-limit glitch; retry with backoff


# ---------------------------------------------------------------------------
# Phase 1 helpers
# ---------------------------------------------------------------------------


async def _dismiss_cookie_banner(page: Page) -> None:
    for label in ("Accept all", "Accept", "Reject non-essential"):
        try:
            btn = page.get_by_role("button", name=label)
            if await btn.count():
                await btn.first.click(timeout=1_500)
                return
        except Exception:
            pass


async def _submit_query(page: Page, query: str, mal_label: str = "") -> None:
    """Fill the search box and submit; falls back to DOM injection."""
    search_box = page.locator(SEARCH_INPUT_SEL)
    await search_box.wait_for(state="visible", timeout=15_000)

    # Human pause: simulate reading the page before typing
    await _human_pause(PRE_SEARCH_DELAY_MIN_S, PRE_SEARCH_DELAY_MAX_S, mal_label)

    try:
        await page.wait_for_function(
            "(sel) => { const el = document.querySelector(sel); return !!el && !el.disabled; }",
            arg=SEARCH_INPUT_SEL,
            timeout=5_000,
        )
        await search_box.fill(query)
        # Pause after filling — simulate a human glancing at what they typed
        await _human_pause(POST_FILL_DELAY_MIN_S, POST_FILL_DELAY_MAX_S, mal_label)
        await search_box.press("Enter")
        # Pause after Enter — let the page begin reacting before we do anything
        await _human_pause(POST_ENTER_DELAY_MIN_S, POST_ENTER_DELAY_MAX_S, mal_label)
    except PWTimeout:
        logger.debug(
            "%s Search box enablement timed out — using DOM injection fallback",
            mal_label,
        )
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
        await _human_pause(POST_ENTER_DELAY_MIN_S, POST_ENTER_DELAY_MAX_S, mal_label)


async def _wait_for_stats_change(page: Page, previous_text: str) -> bool:
    """
    Wait for the stats counter to show a number that differs from
    *previous_text*.  Returns True on success, False on timeout.
    """
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
                "previousStatsText": previous_text,
            },
            timeout=STATS_TIMEOUT_MS,
        )
        return True
    except PWTimeout:
        return False


async def _is_zero_results_page(page: Page) -> bool:
    """Return True when AniPlaylist's 'no results' banner is visible."""
    try:
        loc = page.locator(NO_RESULTS_SEL)
        if await loc.count():
            text = (await loc.first.inner_text()).strip()
            if NO_RESULTS_MARKER in text:
                return True
    except Exception:
        pass
    return False


async def _wait_for_cards_quiesce(page: Page, query: str, mal_label: str) -> bool:
    """
    After the stats counter updates, the card DOM may still be showing the
    previous query's results while Algolia hydrates the new ones.  Poll until
    at least one card is present and its anime-title text is non-empty,
    indicating the new result set has landed.

    We intentionally do NOT check whether the card title matches the query —
    zero-relevant-result pages (e.g. obscure titles) are valid and will have
    cards whose titles differ from the query.  We only care that the DOM has
    settled from a loading state into a stable rendered state.

    Returns True when stable, False on timeout.
    """
    full_card_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
    deadline_ms = CARDS_QUIESCE_TIMEOUT_MS
    elapsed_ms = 0

    while elapsed_ms < deadline_ms:
        try:
            count = await page.locator(full_card_sel).count()
            if count > 0:
                # Confirm the first card has rendered its anime-title text —
                # an empty string means the card shell exists but Algolia
                # hasn't populated it yet.
                first_title = await (
                    page.locator(full_card_sel)
                    .nth(0)
                    .locator(ANIME_TITLE_SEL)
                    .inner_text(timeout=500)
                )
                if first_title.strip():
                    logger.debug(
                        "%s Cards quiesced — %d card(s), first title=%r",
                        mal_label,
                        count,
                        first_title.strip(),
                    )
                    return True
        except Exception:
            pass

        await asyncio.sleep(CARDS_QUIESCE_POLL_MS / 1000)
        elapsed_ms += CARDS_QUIESCE_POLL_MS

    logger.debug(
        "%s Cards did not quiesce within %dms for query %r",
        mal_label,
        CARDS_QUIESCE_TIMEOUT_MS,
        query,
    )
    return False


async def _scroll_until_stable(page: Page, mal_label: str = "") -> None:
    full_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
    prev_count = 0
    stable = 0

    for i in range(SCROLL_MAX_ITER):
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        # Randomise the pause so scroll rhythm doesn't look robotic
        pause = SCROLL_PAUSE_S + random.uniform(
            SCROLL_JITTER_MIN_S, SCROLL_JITTER_MAX_S
        )
        await asyncio.sleep(max(SCROLL_FLOOR_S, pause))

        count = await page.locator(full_sel).count()
        logger.debug("%s Scroll %d — cards: %d", mal_label, i, count)

        if count == prev_count:
            stable += 1
            if stable >= SCROLL_STABLE_REPS:
                logger.debug(
                    "%s Scroll stable at %d cards after %d steps",
                    mal_label,
                    count,
                    i + 1,
                )
                break
        else:
            stable = 0
            prev_count = count


async def _extract_basic_data_all(page: Page) -> list[BasicData]:
    full_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
    raw: list[dict] = await page.locator(full_sel).evaluate_all(f"""
        (cards) => cards.map((card, index) => {{
            const text = (sel) => {{
                const el = card.querySelector(sel);
                return el ? el.textContent.trim() : "";
            }};
            const unreleased = Array.from(card.querySelectorAll("p")).some(
                el => el.textContent.trim() === {UNRELEASED_TEXT!r}
            );
            const spotifyEl = card.querySelector({SPOTIFY_LINK_SEL!r});
            return {{
                anime_title:   text({ANIME_TITLE_SEL!r}),
                song_type_raw: text({SONG_TYPE_SEL!r}),
                title_raw:     text({SONG_TITLE_SEL!r}),
                artist_values: Array.from(card.querySelectorAll({ARTIST_SEL!r}))
                                    .map(el => el.textContent.trim()),
                spotify_link:  spotifyEl
                               ? new URL(spotifyEl.getAttribute("href"), document.baseURI).href
                               : null,
                source_index:  index,
                unreleased,
            }};
        }})
        """)
    return [BasicData(**r) for r in raw]


# ---------------------------------------------------------------------------
# Phase 1 — single attempt (fresh browser per call)
# ---------------------------------------------------------------------------


async def _phase1_attempt(
    query: str,
    headless: bool,
    mal_label: str,
) -> tuple[_Phase1Outcome, list[BasicData], str]:
    """
    Run one full Phase 1 attempt in a fresh browser.

    Returns
    -------
    outcome : _Phase1Outcome
    basic_data_list : list[BasicData]  (non-empty only on SUCCESS)
    raw_html : str
    """
    # Rate limit: enforce minimum gap since the last browser session closed.
    await _inter_session_pause(mal_label)

    async with async_playwright() as pw:
        browser: Browser = await pw.chromium.launch(headless=headless)
        context: BrowserContext = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page: Page = await context.new_page()
        raw_html = ""

        try:
            # Navigate — try both base URLs
            last_nav_err: Exception | None = None
            for base_url in BASE_URLS:
                try:
                    await page.goto(
                        base_url, wait_until="domcontentloaded", timeout=20_000
                    )
                    last_nav_err = None
                    break
                except Exception as exc:
                    last_nav_err = exc
            if last_nav_err:
                logger.warning("%s Navigation failed: %s", mal_label, last_nav_err)
                return _Phase1Outcome.TRANSIENT, [], ""

            # Human pause: simulate reading the page after it loads
            await _human_pause(POST_NAV_DELAY_MIN_S, POST_NAV_DELAY_MAX_S, mal_label)

            await _dismiss_cookie_banner(page)

            # Record what the stats counter says *before* we submit so we can
            # detect when it changes to this query's result count.
            stats_loc = page.locator(RESULTS_STATS_SEL)
            previous_stats_text = ""
            if await stats_loc.count():
                previous_stats_text = (await stats_loc.first.inner_text()).strip()

            await _submit_query(page, query, mal_label)

            # ── Wait for stats counter to reflect the new query ──────────
            stats_updated = await _wait_for_stats_change(page, previous_stats_text)

            if not stats_updated:
                # Stats timed out.  Check for zero-results banner before
                # giving up — AniPlaylist sometimes skips the counter on empty
                # searches and jumps straight to the banner.
                raw_html = await page.content()
                if await _is_zero_results_page(page):
                    logger.info(
                        "%s Zero-results banner detected (stats timeout path) for query %r",
                        mal_label,
                        query,
                    )
                    return _Phase1Outcome.ZERO_RESULTS, [], raw_html

                logger.warning(
                    "%s Stats counter did not update for query %r — likely transient",
                    mal_label,
                    query,
                )
                return _Phase1Outcome.TRANSIENT, [], raw_html

            # ── Wait for card DOM to reflect the new query ───────────────
            cards_ready = await _wait_for_cards_quiesce(page, query, mal_label)

            if not cards_ready:
                raw_html = await page.content()
                if await _is_zero_results_page(page):
                    logger.info(
                        "%s Zero-results banner detected (quiesce timeout path) for query %r",
                        mal_label,
                        query,
                    )
                    return _Phase1Outcome.ZERO_RESULTS, [], raw_html

                logger.warning(
                    "%s Card DOM did not quiesce for query %r — treating as transient",
                    mal_label,
                    query,
                )
                return _Phase1Outcome.TRANSIENT, [], raw_html

            # ── Scroll to load all virtual-scroll pages ──────────────────
            await _scroll_until_stable(page, mal_label)
            raw_html = await page.content()

            # ── Final zero-results check (banner may appear post-scroll) ─
            if await _is_zero_results_page(page):
                logger.info(
                    "%s Zero-results banner detected post-scroll for query %r",
                    mal_label,
                    query,
                )
                return _Phase1Outcome.ZERO_RESULTS, [], raw_html

            basic_data_list = await _extract_basic_data_all(page)

            if not basic_data_list:
                logger.warning(
                    "%s Extraction returned 0 cards despite quiescence for query %r",
                    mal_label,
                    query,
                )
                return _Phase1Outcome.TRANSIENT, [], raw_html

            return _Phase1Outcome.SUCCESS, basic_data_list, raw_html

        finally:
            await page.close()
            await browser.close()
            _record_session_end()


# ---------------------------------------------------------------------------
# Phase 1 — retry wrapper
# ---------------------------------------------------------------------------


async def _run_phase1(
    query: str,
    headless: bool,
    mal_label: str,
) -> tuple[list[BasicData], str]:
    """
    Run Phase 1 with retries on TRANSIENT outcomes.

    ZERO_RESULTS exits immediately (no retry).
    SUCCESS exits immediately.
    TRANSIENT retries up to SCRAPE_MAX_ATTEMPTS times with exponential backoff.

    Each attempt uses a completely fresh Playwright browser+context+page, so
    no session state, cookies, or Algolia JS state carries over.

    Returns
    -------
    basic_data_list : list[BasicData]  (empty on zero-results or total failure)
    raw_html        : str
    """
    last_raw_html = ""

    for attempt in range(1, SCRAPE_MAX_ATTEMPTS + 1):
        logger.info(
            "%s Phase 1 attempt %d/%d for query %r",
            mal_label,
            attempt,
            SCRAPE_MAX_ATTEMPTS,
            query,
        )

        outcome, basic_data_list, raw_html = await _phase1_attempt(
            query, headless, mal_label
        )

        if raw_html:
            last_raw_html = raw_html

        if outcome is _Phase1Outcome.SUCCESS:
            logger.info(
                "%s Phase 1 succeeded on attempt %d — %d card(s)",
                mal_label,
                attempt,
                len(basic_data_list),
            )
            return basic_data_list, raw_html

        if outcome is _Phase1Outcome.ZERO_RESULTS:
            logger.info(
                "%s Phase 1 — genuine zero results for query %r (attempt %d)",
                mal_label,
                query,
                attempt,
            )
            return [], raw_html

        # TRANSIENT — decide whether to retry
        if attempt >= SCRAPE_MAX_ATTEMPTS:
            logger.error(
                "%s Phase 1 — all %d attempts exhausted for query %r; giving up",
                mal_label,
                SCRAPE_MAX_ATTEMPTS,
                query,
            )
            return [], last_raw_html

        backoff = SCRAPE_BACKOFF_BASE_S * (2 ** (attempt - 1))
        # Add jitter so concurrent workers don't thunderherd on retry
        backoff += random.uniform(RETRY_JITTER_MIN_S, RETRY_JITTER_MAX_S)
        logger.warning(
            "%s Phase 1 attempt %d/%d transient failure — retrying in %.1fs",
            mal_label,
            attempt,
            SCRAPE_MAX_ATTEMPTS,
            backoff,
        )
        await asyncio.sleep(backoff)

    return [], last_raw_html


# ---------------------------------------------------------------------------
# Phase 2 — portal close (guaranteed) + extraction
# ---------------------------------------------------------------------------


async def _force_close_portal(page: Page, ctx: str = "") -> None:
    """
    Unconditionally close any open headlessui dialog.

    Close priority:
      1. Known close button (first <button> inside the dialog panel)
      2. Escape key
      3. Backdrop click (to the left of the panel)
    """
    close_btn = await page.query_selector(DIALOG_CLOSE_BTN_SEL)
    if close_btn:
        try:
            await close_btn.click(timeout=2_000)
            await asyncio.sleep(
                PORTAL_CLOSE_S
                + random.uniform(PORTAL_CLOSE_JITTER_MIN_S, PORTAL_CLOSE_JITTER_MAX_S)
            )
        except Exception:
            pass

    if not await page.query_selector(PORTAL_SEL):
        return

    await page.keyboard.press("Escape")
    await asyncio.sleep(
        PORTAL_CLOSE_S
        + random.uniform(PORTAL_CLOSE_JITTER_MIN_S, PORTAL_CLOSE_JITTER_MAX_S)
    )

    if not await page.query_selector(PORTAL_SEL):
        return

    panel = await page.query_selector(DIALOG_PANEL_SEL)
    if panel:
        box = await panel.bounding_box()
        if box:
            await page.mouse.click(
                max(0.0, box["x"] - 50),
                box["y"] + box["height"] / 2,
            )
            await asyncio.sleep(
                PORTAL_CLOSE_S
                + random.uniform(PORTAL_CLOSE_JITTER_MIN_S, PORTAL_CLOSE_JITTER_MAX_S)
            )

    if await page.query_selector(PORTAL_SEL):
        logger.warning(
            "%s Portal still present after all close strategies — page may be stuck",
            ctx,
        )
        return

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

    # Wait for dialog animation to fully complete before reading content
    await asyncio.sleep(
        PORTAL_STABLE_S
        + random.uniform(PORTAL_STABLE_JITTER_MIN_S, PORTAL_STABLE_JITTER_MAX_S)
    )

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
    Open the info dialog for the card at *index*, extract portal data, close it.

    Returns:
        {}                        — info button not found on this card
        {"synonyms": [...], ...}  — successful extraction
        None                      — all retries exhausted
    """
    full_sel = f"{RESULTS_CONTAINER_SEL} {RESULT_CARD_SEL}"
    card_loc = page.locator(full_sel).nth(index)

    for attempt in range(1, SCRAPE_MAX_ATTEMPTS + 1):
        portal_opened = False
        try:
            await card_loc.scroll_into_view_if_needed()
            # Give the card time to fully enter the viewport and lazy-load
            await _human_pause(
                POST_SCROLL_INTO_VIEW_MIN_S, POST_SCROLL_INTO_VIEW_MAX_S, ctx
            )

            try:
                await card_loc.hover(timeout=3_000)
            except Exception:
                pass
            # Hold the hover long enough for any hover-triggered content to load
            await asyncio.sleep(CARD_HOVER_S)

            info_loc = card_loc.locator(INFO_BTN_SEL).first
            if not await info_loc.count():
                info_loc = card_loc.locator(INFO_BTN_FALLBACK_SEL).first
            if not await info_loc.count():
                logger.debug("%s Card %d — no info button found", ctx, index)
                return {}

            await info_loc.click(force=True)
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
                "%s Card %d — timeout attempt %d/%d",
                ctx,
                index,
                attempt,
                SCRAPE_MAX_ATTEMPTS,
            )
        except Exception as exc:
            logger.warning(
                "%s Card %d — error attempt %d/%d: %s",
                ctx,
                index,
                attempt,
                SCRAPE_MAX_ATTEMPTS,
                exc,
            )

        finally:
            if portal_opened or await page.query_selector(PORTAL_SEL):
                await _force_close_portal(page, ctx=ctx)

        backoff = SCRAPE_BACKOFF_BASE_S * (2 ** (attempt - 1))
        backoff += random.uniform(RETRY_JITTER_MIN_S, RETRY_JITTER_MAX_S)
        await asyncio.sleep(backoff)

    logger.error(
        "%s Card %d — all %d retries exhausted; portal_data=None",
        ctx,
        index,
        SCRAPE_MAX_ATTEMPTS,
    )
    return None


# ---------------------------------------------------------------------------
# Phase 2 wrapper — runs inside the *same* browser that Phase 1 used
# ---------------------------------------------------------------------------


async def _run_phase2(
    page: Page,
    results: list[ResultItem],
    fetch_portal_indices: set[int],
    mal_label: str,
) -> None:
    """
    Mutates *results* in-place: sets portal_data on targeted indices.

    Inserts a human-paced delay between each portal extraction so the site
    doesn't see a machine-speed sequence of dialog open/close events.
    """
    targets = sorted(idx for idx in fetch_portal_indices if idx < len(results))
    logger.info(
        "%s Phase 2 — fetching portals for %d/%d card(s)",
        mal_label,
        len(targets),
        len(results),
    )

    for pos, idx in enumerate(targets):
        # Inter-portal delay: pause between cards like a human reading each one.
        # Skip before the very first card (no prior card to pause after).
        if pos > 0:
            await _human_pause(
                INTER_PORTAL_DELAY_MIN_S, INTER_PORTAL_DELAY_MAX_S, mal_label
            )

        card_title = results[idx]["basic_data"].get("anime_title") or f"card {idx}"
        logger.info(
            "%s Portal %d/%d — card_title=%r",
            mal_label,
            idx + 1,
            len(results),
            card_title,
        )
        results[idx]["portal_data"] = await _extract_portal_for_card(
            page, idx, ctx=mal_label
        )

    logger.info("%s Phase 2 complete", mal_label)


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

    Phase 1 is retried up to ``SCRAPE_MAX_ATTEMPTS`` times with exponential
    backoff when the outcome is TRANSIENT (stats timeout with no cards and no
    zero-results banner, or navigation failure).  Each retry launches a brand-
    new browser process so no JS/cookie/Algolia session state bleeds over.

    A genuine zero-results page exits immediately without retrying.

    Phase 2 (portal extraction) runs in the same browser instance that
    completed Phase 1, avoiding an extra page load.

    Parameters
    ----------
    query : str
    headless : bool
    fetch_portal_indices : set[int] | None
        None  — Phase 2 skipped; portal_data absent from all ResultItems.
        set   — only cards at those source_index positions get portal extracted.
    mal_label : str
        Optional "[MAL:id 'title']" prefix for log lines.

    Returns
    -------
    ScrapeResult
    """
    # Phase 1: retry loop with fresh browser per attempt.
    basic_data_list, raw_html_snapshot = await _run_phase1(query, headless, mal_label)

    # Build ResultItems from Phase 1 data.
    results: list[ResultItem] = [
        ResultItem(basic_data=basic) for basic in basic_data_list
    ]

    # Phase 2: portal extraction.
    # Runs in a fresh browser (separate from Phase 1) since Phase 1 already
    # closed its browser.  We re-navigate to the same search so the cards are
    # in the same positions as Phase 1 found them.
    if fetch_portal_indices is None:
        logger.debug("%s Phase 2 skipped — fetch_portal_indices not set", mal_label)
    elif not results:
        logger.debug(
            "%s Phase 2 skipped — no Phase 1 results to open portals for", mal_label
        )
    else:
        # Rate limit: enforce gap between Phase 1 session end and Phase 2 start.
        await _inter_session_pause(mal_label)

        async with async_playwright() as pw:
            browser: Browser = await pw.chromium.launch(headless=headless)
            context: BrowserContext = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 900},
            )
            page: Page = await context.new_page()
            try:
                last_nav_err = None
                for base_url in BASE_URLS:
                    try:
                        await page.goto(
                            base_url, wait_until="domcontentloaded", timeout=20_000
                        )
                        last_nav_err = None
                        break
                    except Exception as exc:
                        last_nav_err = exc
                if last_nav_err:
                    logger.error(
                        "%s Phase 2 navigation failed: %s — portal extraction skipped",
                        mal_label,
                        last_nav_err,
                    )
                else:
                    # Human pause after page load before interacting
                    await _human_pause(
                        POST_NAV_DELAY_MIN_S, POST_NAV_DELAY_MAX_S, mal_label
                    )
                    await _dismiss_cookie_banner(page)
                    await _submit_query(page, query, mal_label)
                    await _wait_for_stats_change(page, "")
                    await _wait_for_cards_quiesce(page, query, mal_label)
                    await _scroll_until_stable(page, mal_label)
                    await _run_phase2(page, results, fetch_portal_indices, mal_label)
            finally:
                await page.close()
                await browser.close()
                _record_session_end()

    return ScrapeResult(
        query=query,
        results=results,
        raw_html_snapshot=raw_html_snapshot,
    )
