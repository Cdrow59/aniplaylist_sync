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

from db import DB_PATH, init_db, save_failure, save_run
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
    allow_pass2: bool = True,
) -> tuple[list[SearchResult], list[dict], bool]:
    """
    Run one scrape+parse cycle for *title_query*.

    Pass 1 — Phase 1 only (no portal clicks).
    If exact_filter is on, no cards matched by title, and allow_pass2 is
    True, run Pass 2 — re-scrape with portals opened only for unmatched cards.

    Returns (results, attempt_log_entries, search_succeeded).
    """
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

    raw_results = parse(scrape_result, exact_titles=exact_titles)
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

    # Matched on title alone — no portal needed
    if matched or not exact_filter:
        results = matched if exact_filter else raw_results
        if results:
            logger.info(
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

    raw_results2 = parse(scrape_result2, exact_titles=exact_titles)
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
    progress: Progress,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    task_id = progress.add_task("AniPlaylist", total=len(mal_entries))

    total = len(mal_entries)
    for n, entry in enumerate(mal_entries, 1):
        ctx = f"[MAL:{entry.mal_id} '{entry.title}']"
        logger.info("%s Starting (%d/%d)", ctx, n, total)

        titles_to_try = candidate_titles(entry)
        native_title, english_title, japanese_title = title_metadata(entry)
        exact_titles = exact_title_candidates(entry)
        results: list[SearchResult] = []
        used_query = titles_to_try[0] if titles_to_try else entry.title
        all_attempt_logs: list[dict] = []
        last_error_reason: str | None = None
        search_succeeded = False

        # ── Loop 1: exhaust all title candidates with Pass 1 only ────────
        # Portals are expensive — try every title variant via simple search
        # before falling back to portal synonym checks.
        for title_query in titles_to_try:
            results, attempt_logs, succeeded = await _search_one(
                title_query,
                exact_titles,
                exact_filter,
                headed,
                mal_id=entry.mal_id,
                mal_title=entry.title,
                allow_pass2=False,
            )
            all_attempt_logs.extend(attempt_logs)

            if not succeeded:
                last_error_reason = next(
                    (e["error"] for e in reversed(attempt_logs) if e.get("error")), None
                )
                logger.warning(
                    "%s Query '%s' failed: %s", ctx, title_query, last_error_reason
                )
                continue

            search_succeeded = True
            used_query = title_query

            if results:
                break
            else:
                logger.debug(
                    "%s Query '%s' returned cards but no match — trying next title candidate",
                    ctx,
                    title_query,
                )

        # ── Loop 2: if still no match, retry all candidates with portals ─
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
                )
                all_attempt_logs.extend(attempt_logs)

                if not succeeded:
                    last_error_reason = next(
                        (e["error"] for e in reversed(attempt_logs) if e.get("error")),
                        None,
                    )
                    logger.warning(
                        "%s Query '%s' failed: %s", ctx, title_query, last_error_reason
                    )
                    continue

                search_succeeded = True
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
            logger.info(
                "%s Done — query='%s'  results=%d  matched=%d",
                ctx,
                used_query,
                len(results),
                matched_count,
            )
        else:
            failure_query = json.dumps(all_attempt_logs, ensure_ascii=False)
            failure_reason = (
                "No AniPlaylist results matched"
                if search_succeeded
                else (last_error_reason or "AniPlaylist search failed")
            )
            logger.warning(
                "%s No results — reason: %s  (tried %d candidate(s): %s)",
                ctx,
                failure_reason,
                len(titles_to_try),
                titles_to_try,
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
    timeout_seconds = getattr(args, "timeout", None) or None
    logger.info(f"Starting sync with timeout of {timeout_seconds}s")
    try:
        async with asyncio.timeout(timeout_seconds):
            await _run_impl(args)
    except asyncio.TimeoutError:
        logger.error(f"Sync operation timed out after {timeout_seconds}s")
        raise


async def _run_impl(args) -> None:
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

            series_task = asyncio.create_task(
                run_series_stage(args.db, client, mal_entries, progress=progress)
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

                if args.confirm:
                    if Confirm.ask("Run Spotify?"):
                        await run_spotify_stage(
                            args.db,
                            megaplaylist=bool(args.megaplaylist),
                            progress=progress,
                        )

                else:
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
