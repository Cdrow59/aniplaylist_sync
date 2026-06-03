"""Main orchestration and stages for MAL -> AniPlaylist sync.

This module owns the single Rich Progress instance for the entire run and
passes it down to every sub-stage.  No sub-module creates its own Progress.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

from playwright.async_api import Error as PlaywrightError
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from aniplaylist import SearchResult, close_browser_pool, search_aniplaylist
from db import DB_PATH, init_db, save_failure, save_run
from mal import MALAnimeEntry, MALClient
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
    alt = entry.alternative_titles or {}
    if isinstance(alt, dict):
        _add_title_candidate(candidates, alt.get("en"))
    _add_title_candidate(candidates, entry.title)
    if isinstance(alt, dict):
        _add_title_candidate(candidates, alt.get("ja"))
        _add_title_candidate(candidates, alt.get("synonyms"))
    return candidates


def exact_title_candidates(entry: MALAnimeEntry) -> list[str]:
    titles: list[str] = []
    alt = entry.alternative_titles or {}
    if isinstance(alt, dict):
        _add_title_candidate(titles, alt.get("en"))
    _add_title_candidate(titles, entry.title)
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
            "Invalid status. Use one of: complete, completed, watching, on_hold, dropped, plan_to_watch"
        )
    return MAL_STATUS_ALIASES[normalized]


# ---------------------------------------------------------------------------
# Stage helpers – all accept an already-started Progress from the caller
# ---------------------------------------------------------------------------


async def run_series_stage(
    db_path: Path,
    client: MALClient,
    mal_entries: list[MALAnimeEntry],
    *,
    progress: Progress,
) -> None:
    """Discover series clusters and persist them.  Progress is owned by caller."""
    seed_ids = [entry.mal_id for entry in mal_entries]
    if not seed_ids:
        return
    discovery = await discover_series(client, seed_ids, progress=progress)
    await save_series_clusters(db_path, discovery.clusters)


async def run_aniplaylist_stage(
    db_path: Path,
    mal_entries: list[MALAnimeEntry],
    *,
    headed: bool,
    exact_filter: bool,
    aniplaylist_delay: float,
    emit_json: bool,
    progress: Progress,
) -> list[dict[str, object]]:
    """Search AniPlaylist for every MAL entry and persist results.
    Progress is owned by caller."""
    summary: list[dict[str, object]] = []
    task_id = progress.add_task("AniPlaylist", total=len(mal_entries))

    for entry in mal_entries:
        titles_to_try = candidate_titles(entry)
        native_title, english_title, japanese_title = title_metadata(entry)
        exact_titles = exact_title_candidates(entry)
        results: list[SearchResult] = []
        used_query = titles_to_try[0] if titles_to_try else entry.title
        attempt_logs: list[dict[str, object]] = []
        last_error_reason: str | None = None
        search_succeeded = False

        for title_query in titles_to_try:
            try:
                raw_results = await search_aniplaylist(
                    title_query,
                    headless=not headed,
                    exact_titles=exact_titles,
                )
            except (PlaywrightError, RuntimeError, ValueError, OSError) as exc:
                last_error_reason = str(exc)
                attempt_logs.append(
                    {
                        "mode": "simple",
                        "query": title_query,
                        "result_count": 0,
                        "matched_count": 0,
                        "error": last_error_reason,
                    }
                )
                continue

            search_succeeded = True
            used_query = title_query
            matched_results = [r for r in raw_results if r.matched_query]
            attempt_logs.append(
                {
                    "mode": "simple",
                    "query": title_query,
                    "result_count": len(raw_results),
                    "matched_count": len(matched_results),
                    "error": None,
                }
            )

            if exact_filter:
                results = matched_results
                if results:
                    break
                continue

            results = raw_results
            if results:
                break

        if not results and exact_filter and search_succeeded:
            logger.warning(
                f"Advanced AniPlaylist fallback triggered for: {entry.mal_id}"
            )

            for title_query in titles_to_try:
                try:
                    raw_results = await search_aniplaylist(
                        title_query,
                        headless=not headed,
                        exact_titles=exact_titles,
                        advanced_fallback=True,
                    )
                except (PlaywrightError, RuntimeError, ValueError, OSError) as exc:
                    last_error_reason = str(exc)
                    attempt_logs.append(
                        {
                            "mode": "advanced",
                            "query": title_query,
                            "result_count": 0,
                            "matched_count": 0,
                            "error": last_error_reason,
                        }
                    )
                    continue

                search_succeeded = True
                used_query = title_query
                matched_results = [r for r in raw_results if r.matched_query]
                advanced_card_checks = [
                    {
                        "source_index": r.source_index,
                        "anime_title": r.anime_title,
                        "matched_query": r.matched_query,
                        "matched_synonym": r.advanced_matched_synonym,
                        "synonyms": r.advanced_synonyms or [],
                        "advanced_error": r.advanced_error,
                    }
                    for r in raw_results
                    if r.advanced_attempted
                ]
                attempt_logs.append(
                    {
                        "mode": "advanced",
                        "query": title_query,
                        "result_count": len(raw_results),
                        "matched_count": len(matched_results),
                        "advanced_card_checks": advanced_card_checks,
                        "error": None,
                    }
                )

                results = matched_results if exact_filter else raw_results
                if results:
                    break

        if not results:
            failure_query = json.dumps(attempt_logs, ensure_ascii=False)
            failure_reason = (
                "No AniPlaylist results matched"
                if search_succeeded
                else (last_error_reason or "AniPlaylist search failed")
            )
            await save_failure(
                db_path,
                failure_query,
                reason=failure_reason,
                mal_id=entry.mal_id,
                native_title=native_title,
                english_title=english_title,
                japanese_title=japanese_title,
                status=entry.status,
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
    """Main entry point with timeout protection and resource cleanup."""
    timeout_seconds = getattr(args, "timeout", None) or None

    logger.info(f"Starting sync with timeout of {timeout_seconds}s")
    try:
        async with asyncio.timeout(timeout_seconds):
            await _run_impl(args)
    except asyncio.TimeoutError:
        logger.error(f"Sync operation timed out after {timeout_seconds}s")
        raise
    finally:
        try:
            await close_browser_pool()
        except Exception as e:
            logger.warning(f"Error closing browser pool: {e}")


async def _run_impl(args) -> None:
    """Internal implementation – owns the single Progress for the whole run."""
    client_id = args.client_id or read_env("MAL_CLIENT_ID")
    access_token = args.access_token or os.getenv("MAL_ACCESS_TOKEN")
    username = args.username or read_env("MAL_USERNAME", "@me")
    rate_limit = float(read_env("MAL_RATE_LIMIT_PER_SECOND", "1"))
    aniplaylist_delay = (
        args.aniplaylist_delay
        if args.aniplaylist_delay is not None
        else float(read_env("ANIPLAYLIST_DELAY_SECONDS", "1"))
    )
    mal_status = normalize_status(args.status) if args.status else None

    await init_db(args.db)

    client = MALClient(
        client_id=client_id,
        access_token=access_token,
        username=username,
        per_second=rate_limit,
    )

    try:
        summary: list[dict[str, object]] = []

        # One Progress instance for the entire run – every stage adds its own
        # tasks to this shared instance; none of them start or stop it.
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            mal_entries = await client.list_user_anime(
                status=mal_status,
                progress=progress,
            )

            if args.limit and args.limit > 0:
                mal_entries = mal_entries[: args.limit]

            if not mal_entries:
                logger.error("No MAL anime entries found!!")
                return

            # Run series discovery and AniPlaylist search concurrently.
            # Both stages receive the same live Progress and add their own tasks.
            series_task = asyncio.create_task(
                run_series_stage(
                    args.db,
                    client,
                    mal_entries,
                    progress=progress,
                )
            )
            aniplaylist_task = asyncio.create_task(
                run_aniplaylist_stage(
                    args.db,
                    mal_entries,
                    headed=args.headed,
                    exact_filter=not args.no_exact_filter,
                    aniplaylist_delay=aniplaylist_delay,
                    emit_json=args.json,
                    progress=progress,
                )
            )

            summary, _ = await asyncio.gather(aniplaylist_task, series_task)

            if not args.dry_run:
                from spotify import run_spotify_stage

                await run_spotify_stage(
                    args.db,
                    megaplaylist=bool(args.megaplaylist),
                    progress=progress,
                )
    finally:
        try:
            await client.close()
        except Exception as e:
            logger.warning(f"Error closing MAL client: {e}")

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
