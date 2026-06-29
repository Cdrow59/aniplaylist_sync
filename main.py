"""Main orchestration and stages for MAL -> AniPlaylist sync."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from parser import SearchResult, parse
from pathlib import Path

import rich
from playwright.async_api import Error as PlaywrightError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, Prompt

from anilist import AniListClient
from db import (
    DB_PATH,
    FAILURE_NO_MATCH,
    FAILURE_NOT_FOUND,
    FAILURE_SCRAPE_ERROR,
    init_db,
    save_failure,
    save_run,
)
from logging_config import console
from mal import MALAnimeEntry, MALClient
from scraper import scrape
from series import discover_series, save_series_clusters

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = DB_PATH
MAL_STATUS_ALIASES = {
    "complete": "completed",
    "completed": "completed",
    "watching": "watching",
    "on_hold": "on_hold",
    "dropped": "dropped",
    "plan_to_watch": "plan_to_watch",
}

ANILIST_STATUS_MAP = {
    "completed": "COMPLETED",
    "complete": "COMPLETED",
    "watching": "CURRENT",
    "on_hold": "PAUSED",
    "dropped": "DROPPED",
    "plan_to_watch": "PLANNING",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_env(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name, fallback)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def _add_title_candidate(target: list[str], value: object | None) -> None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned and cleaned not in target:
            target.append(cleaned)
    elif isinstance(value, list):
        for item in value:
            _add_title_candidate(target, item)


def candidate_titles(entry: MALAnimeEntry) -> list[str]:
    candidates: list[str] = []
    _add_title_candidate(candidates, entry.title)
    alt = entry.alternative_titles or {}
    if isinstance(alt, dict):
        _add_title_candidate(candidates, alt.get("en"))
    if isinstance(alt, dict):
        _add_title_candidate(candidates, alt.get("ja"))
        _add_title_candidate(candidates, alt.get("synonyms"))
    return candidates


def exact_title_candidates(entry: MALAnimeEntry) -> list[str]:
    titles: list[str] = []
    _add_title_candidate(titles, entry.title)
    alt = entry.alternative_titles or {}
    if isinstance(alt, dict):
        _add_title_candidate(titles, alt.get("en"))
    if isinstance(alt, dict):
        _add_title_candidate(titles, alt.get("ja"))
        _add_title_candidate(titles, alt.get("synonyms"))
    return titles


def title_metadata(entry: MALAnimeEntry) -> tuple[str | None, str | None, str | None]:
    def clean(value: object | None) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        return None

    native_title = clean(entry.title)
    alt = entry.alternative_titles or {}
    english_title = clean(alt.get("en")) if isinstance(alt, dict) else None
    japanese_title = clean(alt.get("ja")) if isinstance(alt, dict) else None
    return native_title, english_title, japanese_title


def normalize_status(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in MAL_STATUS_ALIASES:
        raise ValueError(
            "Invalid status. Use one of: complete, completed, watching, "
            "on_hold, dropped, plan_to_watch"
        )
    return MAL_STATUS_ALIASES[normalized]


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


async def run_series_stage(
    db_path: Path,
    client: MALClient,
    mal_entries: list[MALAnimeEntry],
    *,
    progress: Progress,
) -> None:
    seed_ids = [entry.mal_id for entry in mal_entries]
    if not seed_ids:
        logger.info("Series stage skipped — no MAL entries to seed discovery")
        return
    discovery = await discover_series(client, seed_ids, progress=progress)
    await save_series_clusters(db_path, discovery.clusters)


async def _search_one(
    title_query: str,
    exact_titles: list[str],
    exact_filter: bool,
    headed: bool,
    *,
    mal_id: int,
    mal_title: str,
    save_html: bool,
    allow_pass2: bool = True,
) -> tuple[list[SearchResult], list[dict], bool]:
    ctx = f"[MAL:{mal_id} '{mal_title}']"
    attempt_logs: list[dict] = []

    # ── Pass 1: static data only ──────────────────────────────────────────
    logger.info("%s Pass 1 — querying '%s'", ctx, title_query)
    try:
        scrape_result = await scrape(title_query, headless=not headed, mal_label=ctx)
    except (PlaywrightError, RuntimeError, ValueError, OSError) as exc:
        logger.warning("%s Pass 1 failed for query '%s': %s", ctx, title_query, exc)
        attempt_logs.append(
            {
                "pass": 1,
                "query": title_query,
                "result_count": 0,
                "matched_count": 0,
                "error": str(exc),
            }
        )
        return [], attempt_logs, False

    raw_results = parse(scrape_result, exact_titles=exact_titles, save_html=save_html)
    matched = [r for r in raw_results if r.matched_query]

    logger.info(
        "%s Pass 1 done — query='%s'  cards=%d  matched=%d",
        ctx,
        title_query,
        len(raw_results),
        len(matched),
    )
    attempt_logs.append(
        {
            "pass": 1,
            "query": title_query,
            "result_count": len(raw_results),
            "matched_count": len(matched),
            "error": None,
        }
    )

    if matched or not exact_filter:
        results = matched if exact_filter else raw_results
        if results:
            logger.debug(
                "%s Pass 1 matched %d result(s) — skipping portal phase",
                ctx,
                len(results),
            )
        return results, attempt_logs, True

    # ── Pass 2: open portals for unmatched cards only ─────────────────────
    unmatched_indices = {
        r.source_index
        for r in raw_results
        if r.source_index is not None and not r.matched_query
    }

    if not unmatched_indices:
        logger.info("%s Pass 1 returned no cards — nothing to escalate", ctx)
        return [], attempt_logs, True

    if not allow_pass2:
        logger.debug(
            "%s Pass 2 deferred — %d unmatched card(s), will retry after all Pass 1 candidates exhausted",
            ctx,
            len(unmatched_indices),
        )
        return [], attempt_logs, True

    logger.warning(
        "%s Pass 1 found no title match across %d card(s) — "
        "escalating to Pass 2 (portal synonym check) for indices %s",
        ctx,
        len(raw_results),
        sorted(unmatched_indices),
    )

    try:
        scrape_result2 = await scrape(
            title_query,
            headless=not headed,
            fetch_portal_indices=unmatched_indices,
            mal_label=ctx,
        )
    except (PlaywrightError, RuntimeError, ValueError, OSError) as exc:
        logger.error(
            "%s Pass 2 scrape failed for query '%s': %s", ctx, title_query, exc
        )
        attempt_logs.append(
            {
                "pass": 2,
                "query": title_query,
                "result_count": 0,
                "matched_count": 0,
                "error": str(exc),
            }
        )
        return [], attempt_logs, True

    raw_results2 = parse(scrape_result2, exact_titles=exact_titles, save_html=save_html)
    matched2 = [r for r in raw_results2 if r.matched_query]

    advanced_card_checks = [
        {
            "source_index": r.source_index,
            "anime_title": r.anime_title,
            "matched_query": r.matched_query,
            "matched_synonym": r.advanced_matched_synonym,
            "synonyms": r.advanced_synonyms or [],
            "advanced_error": r.advanced_error,
        }
        for r in raw_results2
        if r.advanced_attempted
    ]

    if matched2:
        for r in raw_results2:
            if r.matched_query and r.advanced_matched_synonym:
                logger.info(
                    "%s Pass 2 matched via synonym '%s' on card %s ('%s')",
                    ctx,
                    r.advanced_matched_synonym,
                    r.source_index,
                    r.anime_title,
                )
        logger.info(
            "%s Pass 2 done — query='%s'  portals_opened=%d  matched=%d",
            ctx,
            title_query,
            len(unmatched_indices),
            len(matched2),
        )
    else:
        for check in advanced_card_checks:
            if check["advanced_error"]:
                logger.warning(
                    "%s Pass 2 card %s ('%s') portal error: %s",
                    ctx,
                    check["source_index"],
                    check["anime_title"],
                    check["advanced_error"],
                )
            elif check["synonyms"]:
                logger.debug(
                    "%s Pass 2 card %s ('%s') synonyms checked: %s — no match",
                    ctx,
                    check["source_index"],
                    check["anime_title"],
                    check["synonyms"],
                )
            else:
                logger.debug(
                    "%s Pass 2 card %s ('%s') — no synonyms in portal",
                    ctx,
                    check["source_index"],
                    check["anime_title"],
                )
        logger.warning(
            "%s Pass 2 exhausted — query='%s'  portals_opened=%d  no match found",
            ctx,
            title_query,
            len(unmatched_indices),
        )

    attempt_logs.append(
        {
            "pass": 2,
            "query": title_query,
            "result_count": len(raw_results2),
            "matched_count": len(matched2),
            "advanced_card_checks": advanced_card_checks,
            "error": None,
        }
    )

    results = matched2 if exact_filter else raw_results2
    return results, attempt_logs, True


async def run_aniplaylist_stage(
    db_path: Path,
    mal_entries: list[MALAnimeEntry],
    *,
    headed: bool,
    exact_filter: bool,
    aniplaylist_delay: float,
    emit_json: bool,
    save_html: bool,
    progress: Progress,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    task_id = progress.add_task("AniPlaylist", total=len(mal_entries))

    total = len(mal_entries)
    logger.info("AniPlaylist stage starting — %d title(s) to process", total)
    for n, entry in enumerate(mal_entries, 1):
        ctx = f"[MAL:{entry.mal_id} '{entry.title}']"
        logger.info("%s Starting (%d/%d)", ctx, n, total)

        titles_to_try = candidate_titles(entry)
        native_title, english_title, japanese_title = title_metadata(entry)
        exact_titles = exact_title_candidates(entry)
        results: list[SearchResult] = []
        used_query = titles_to_try[0] if titles_to_try else entry.title
        all_attempt_logs: list[dict] = []
        any_scrape_succeeded = False
        total_cards_seen = 0

        for title_query in titles_to_try:
            results, attempt_logs, succeeded = await _search_one(
                title_query,
                exact_titles,
                exact_filter,
                headed,
                mal_id=entry.mal_id,
                mal_title=entry.title,
                allow_pass2=False,
                save_html=save_html,
            )
            all_attempt_logs.extend(attempt_logs)
            total_cards_seen += sum(log.get("result_count", 0) for log in attempt_logs)

            if not succeeded:
                logger.warning("%s Query '%s' failed", ctx, title_query)
                continue

            any_scrape_succeeded = True
            used_query = title_query

            if results:
                break
            else:
                logger.debug(
                    "%s Query '%s' returned cards but no match — trying next title candidate",
                    ctx,
                    title_query,
                )

        if not results:
            logger.info(
                "%s All Pass 1 candidates exhausted — retrying with portal synonym checks",
                ctx,
            )
            for title_query in titles_to_try:
                results, attempt_logs, succeeded = await _search_one(
                    title_query,
                    exact_titles,
                    exact_filter,
                    headed,
                    mal_id=entry.mal_id,
                    mal_title=entry.title,
                    allow_pass2=True,
                    save_html=save_html,
                )
                all_attempt_logs.extend(attempt_logs)
                total_cards_seen += sum(
                    log.get("result_count", 0) for log in attempt_logs
                )

                if not succeeded:
                    logger.warning("%s Query '%s' failed", ctx, title_query)
                    continue

                any_scrape_succeeded = True
                used_query = title_query

                if results:
                    break
                else:
                    logger.debug(
                        "%s Query '%s' portal check found no match — trying next title candidate",
                        ctx,
                        title_query,
                    )

        if results:
            matched_count = sum(1 for r in results if r.matched_query)
            logger.debug(
                "%s Done — query='%s'  results=%d  matched=%d",
                ctx,
                used_query,
                len(results),
                matched_count,
            )
        else:
            if not any_scrape_succeeded:
                failure_type = FAILURE_SCRAPE_ERROR
            elif total_cards_seen == 0:
                failure_type = FAILURE_NOT_FOUND
            else:
                failure_type = FAILURE_NO_MATCH

            logger.warning(
                "%s No results — failure_type=%r  cards_seen=%d  tried=%s",
                ctx,
                failure_type,
                total_cards_seen,
                titles_to_try,
            )
            await save_failure(
                db_path,
                failure_type=failure_type,
                tried_queries=titles_to_try,
                cards_seen=total_cards_seen,
                mal_id=entry.mal_id,
                native_title=native_title,
                english_title=english_title,
                japanese_title=japanese_title,
                mal_status=entry.status,
                attempt_log=(
                    all_attempt_logs if failure_type != FAILURE_NOT_FOUND else None
                ),
            )

        await save_run(
            db_path,
            used_query,
            results,
            mal_id=entry.mal_id,
            native_title=native_title,
            english_title=english_title,
            japanese_title=japanese_title,
        )

        summary.append(
            {
                "mal_title": entry.title,
                "used_query": used_query,
                "result_count": len(results),
                "matched": any(r.matched_query for r in results),
            }
        )

        if aniplaylist_delay > 0:
            await asyncio.sleep(aniplaylist_delay)

        progress.advance(task_id)

    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


async def run(args) -> None:
    timeout_seconds = args.timeout or None
    logger.info(
        f"Starting sync with timeout of {timeout_seconds}s"
        if timeout_seconds
        else f"Starting sync without timeout."
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            await _run_impl(args)
    except asyncio.TimeoutError:
        logger.error(f"Sync operation timed out after {timeout_seconds}s")
        raise
    else:
        logger.info("Sync completed successfully")


def create_progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


async def run_spotify_stage_if_needed(
    args,
    progress: Progress,
) -> None:
    if args.dry_run:
        logger.info("Dry run enabled — skipping Spotify stage")
        return

    from spotify import run_spotify_stage

    if args.confirm:
        if not Confirm.ask("\nRun Spotify?", console=console):
            logger.info("Spotify stage skipped — user declined confirmation")
            return

    await run_spotify_stage(
        args.db,
        megaplaylist=bool(args.megaplaylist),
        progress=progress,
    )


async def run_cached_mode(args) -> None:
    logger.info("Cached mode — skipping MAL fetch and AniPlaylist scrape")

    with create_progress() as progress:
        await run_spotify_stage_if_needed(
            args,
            progress,
        )


def create_client(args) -> tuple[
    MALClient | AniListClient,
    str,
    str | None,
]:
    if args.anilist:
        client = AniListClient(
            username=args.username,
            client_id=read_env("ANILIST_CLIENT_ID"),
            client_secret=read_env("ANILIST_CLIENT_SECRET"),
            redirect_uri=os.getenv("ANILIST_REDIRECT_URI"),
        )

        status = (
            ANILIST_STATUS_MAP.get(normalize_status(args.status))
            if args.status
            else None
        )

        return client, "AniList", status

    client = MALClient(
        client_id=read_env("MAL_CLIENT_ID"),
        username=args.username or "@me",
        redirect_uri=os.getenv("MAL_REDIRECT_URI"),
    )

    status = normalize_status(args.status) if args.status else None

    return client, "MAL", status


async def run_main_pipeline(args) -> list[dict[str, object]]:
    client, source_label, status = create_client(args)

    try:
        with create_progress() as progress:
            entries = await client.list_user_anime(
                status=status,
                progress=progress,
            )

            if args.limit and args.limit > 0:
                logger.info(
                    "Limiting %s entries from %d to %d",
                    source_label,
                    len(entries),
                    args.limit,
                )
                entries = entries[: args.limit]

            if not entries:
                logger.error(
                    "No %s anime entries found!!",
                    source_label,
                )
                return []

            logger.info(
                "Proceeding with %d %s entries",
                len(entries),
                source_label,
            )

            delay = (
                args.aniplaylist_delay
                if args.aniplaylist_delay is not None
                else float(
                    read_env(
                        "ANIPLAYLIST_DELAY_SECONDS",
                        "60",
                    )
                )
            )

            series_task = asyncio.create_task(
                run_series_stage(
                    args.db,
                    client,
                    entries,
                    progress=progress,
                )
            )

            aniplaylist_task = asyncio.create_task(
                run_aniplaylist_stage(
                    args.db,
                    entries,
                    headed=args.headed,
                    exact_filter=not args.no_exact_filter,
                    aniplaylist_delay=delay,
                    emit_json=args.json,
                    save_html=args.save_html,
                    progress=progress,
                )
            )

            summary, _ = await asyncio.gather(
                aniplaylist_task,
                series_task,
            )

            logger.info(
                "AniPlaylist stage complete — %d/%d matched",
                sum(1 for item in summary if item.get("matched")),
                len(summary),
            )

            return summary

    finally:
        try:
            await client.close()
        except Exception as exc:
            logger.warning(
                "Error closing %s client: %s",
                source_label,
                exc,
            )


async def _run_impl(args) -> None:
    await init_db(args.db)

    if getattr(args, "cached", False):
        await run_cached_mode(args)
        return

    summary = await run_main_pipeline(args)

    # Progress context is fully DEAD here
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.dry_run:
        logger.info("Dry run enabled — skipping Spotify stage")
        return

    if args.confirm:
        if not Confirm.ask("Run Spotify?", console=console):
            logger.info("Spotify stage skipped — user declined confirmation")
            return

    with create_progress() as progress:
        from spotify import run_spotify_stage

        await run_spotify_stage(
            args.db,
            megaplaylist=bool(args.megaplaylist),
            progress=progress,
        )
