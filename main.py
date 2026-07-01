"""Main orchestration and stages for MAL -> AniPlaylist sync."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from parser import SearchResult, parse
from pathlib import Path

import rich
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm

from anilist import AniListClient
from aniplaylist import AlgoliaClient, ScrapeResult
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
from series import discover_series, save_series_clusters
from spotify import run_spotify_stage

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
    *,
    algolia: AlgoliaClient,
    mal_id: int,
    mal_title: str,
    allow_pass2: bool = True,
    raw_log: list[dict] | None = None,
) -> tuple[list[SearchResult], list[dict], bool]:
    ctx = f"[MAL:{mal_id} '{mal_title}']"
    attempt_logs: list[dict] = []

    # ── Pass 1: fetch all hits, match on primary anime title ─────────────
    logger.info("%s Searching — query=%r", ctx, title_query)
    try:
        scrape_result = await algolia.scrape(
            title_query, mal_label=ctx, raw_log=raw_log
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.warning("%s Search failed for query=%r: %s", ctx, title_query, exc)
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
        "%s Search done — query=%r  hits=%d  matched=%d",
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
        results = [r for r in results if r.spotify_link]
        if results:
            logger.debug(
                "%s Matched %d result(s) on primary title — done",
                ctx,
                len(results),
            )
        return results, attempt_logs, True

    # ── Pass 2: re-fetch with alternate titles as synonyms ────────────────
    unmatched_indices = {
        r.source_index
        for r in raw_results
        if r.source_index is not None and not r.matched_query
    }

    if not unmatched_indices:
        logger.info("%s No hits returned — nothing to escalate", ctx)
        return [], attempt_logs, True

    if not allow_pass2:
        logger.debug(
            "%s Synonym check deferred — %d unmatched hit(s), will retry after all queries exhausted",
            ctx,
            len(unmatched_indices),
        )
        return [], attempt_logs, True

    logger.warning(
        "%s No title match across %d hit(s) — escalating to synonym check for indices %s",
        ctx,
        len(raw_results),
        sorted(unmatched_indices),
    )

    try:
        scrape_result2 = await algolia.scrape(
            title_query,
            fetch_portal_indices=unmatched_indices,
            mal_label=ctx,
            raw_log=raw_log,
        )
    except (RuntimeError, ValueError, OSError) as exc:
        logger.error("%s Synonym fetch failed for query=%r: %s", ctx, title_query, exc)
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

    if matched2:
        for r in matched2:
            if r.advanced_matched_synonym:
                logger.info(
                    "%s Synonym match — hit=%r synonym=%r index=%s",
                    ctx,
                    r.anime_title,
                    r.advanced_matched_synonym,
                    r.source_index,
                )
        logger.info(
            "%s Synonym check done — query=%r  matched=%d",
            ctx,
            title_query,
            len(matched2),
        )
    else:
        for r in raw_results2:
            if r.advanced_attempted and r.advanced_synonyms:
                logger.debug(
                    "%s Index %s synonyms checked — no match: %s",
                    ctx,
                    r.source_index,
                    r.advanced_synonyms,
                )
        logger.warning(
            "%s Synonym check exhausted — query=%r  no match",
            ctx,
            title_query,
        )

    attempt_logs.append(
        {
            "pass": 2,
            "query": title_query,
            "result_count": len(raw_results2),
            "matched_count": len(matched2),
            "error": None,
        }
    )

    results = matched2 if exact_filter else raw_results2
    results = [r for r in results if r.spotify_link]
    return results, attempt_logs, True


def _write_raw_log(
    raw_dir: Path,
    mal_id: int,
    title: str,
    raw_log: list[dict],
) -> None:
    """Write the full HTTP response log for one MAL entry to debug/raw/."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60]
    path = raw_dir / f"{mal_id}_{safe}.json"
    payload = {
        "mal_id": mal_id,
        "mal_title": title,
        "requests": raw_log,
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug("Wrote raw HTTP log — %s", path)
    except (OSError, TypeError) as exc:
        logger.warning("Failed to write raw log for MAL:%d: %s", mal_id, exc)


def _result_to_dict(r: SearchResult) -> dict:
    return {
        "anime_title": r.anime_title,
        "song_type": r.song_type,
        "sequence": r.sequence,
        "title": r.title,
        "artists": r.artists,
        "spotify_link": r.spotify_link,
        "matched_query": r.matched_query,
        "source_index": r.source_index,
        "advanced_attempted": r.advanced_attempted,
        "advanced_synonyms": r.advanced_synonyms,
        "advanced_matched_synonym": r.advanced_matched_synonym,
        "advanced_error": r.advanced_error,
    }


def _write_entry_json(
    json_dir: Path,
    mal_id: int,
    title: str,
    used_query: str,
    results: list[SearchResult],
    attempt_logs: list[dict],
) -> None:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)[:60]
    path = json_dir / f"{mal_id}_{safe}.json"
    payload = {
        "mal_id": mal_id,
        "mal_title": title,
        "used_query": used_query,
        "attempt_logs": attempt_logs,
        "results": [_result_to_dict(r) for r in results],
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.debug("Wrote JSON — %s", path)
    except OSError as exc:
        logger.warning("Failed to write JSON for MAL:%d: %s", mal_id, exc)


async def run_aniplaylist_stage(
    db_path: Path,
    mal_entries: list[MALAnimeEntry],
    *,
    algolia: AlgoliaClient,
    exact_filter: bool,
    emit_json: bool,
    emit_raw: bool,
    progress: Progress,
) -> list[dict[str, object]]:
    json_dir: Path | None = None
    if emit_json:
        json_dir = Path("debug/json")
        json_dir.mkdir(parents=True, exist_ok=True)

    raw_dir: Path | None = None
    if emit_raw:
        raw_dir = Path("debug/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)

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
        total_hits_seen = 0
        entry_raw_log: list[dict] = [] if emit_raw else None  # type: ignore[assignment]

        for title_query in titles_to_try:
            results, attempt_logs, succeeded = await _search_one(
                title_query,
                exact_titles,
                exact_filter,
                algolia=algolia,
                mal_id=entry.mal_id,
                mal_title=entry.title,
                allow_pass2=False,
                raw_log=entry_raw_log,
            )
            all_attempt_logs.extend(attempt_logs)
            total_hits_seen += sum(log.get("result_count", 0) for log in attempt_logs)

            if not succeeded:
                logger.warning("%s Query %r failed — skipping", ctx, title_query)
                continue

            any_scrape_succeeded = True
            used_query = title_query

            if results:
                break
            else:
                logger.debug(
                    "%s Query %r returned hits but no match — trying next candidate",
                    ctx,
                    title_query,
                )

        if not results:
            logger.info(
                "%s All queries exhausted — retrying with synonym checks",
                ctx,
            )
            for title_query in titles_to_try:
                results, attempt_logs, succeeded = await _search_one(
                    title_query,
                    exact_titles,
                    exact_filter,
                    algolia=algolia,
                    mal_id=entry.mal_id,
                    mal_title=entry.title,
                    allow_pass2=True,
                    raw_log=entry_raw_log,
                )
                all_attempt_logs.extend(attempt_logs)
                total_hits_seen += sum(
                    log.get("result_count", 0) for log in attempt_logs
                )

                if not succeeded:
                    logger.warning("%s Query %r failed — skipping", ctx, title_query)
                    continue

                any_scrape_succeeded = True
                used_query = title_query

                if results:
                    break
                else:
                    logger.debug(
                        "%s Query %r synonym check found no match — trying next candidate",
                        ctx,
                        title_query,
                    )

        if results:
            matched_count = sum(1 for r in results if r.matched_query)
            logger.debug(
                "%s Done — query=%r  results=%d  matched=%d",
                ctx,
                used_query,
                len(results),
                matched_count,
            )
        else:
            if not any_scrape_succeeded:
                failure_type = FAILURE_SCRAPE_ERROR
            elif total_hits_seen == 0:
                failure_type = FAILURE_NOT_FOUND
            else:
                failure_type = FAILURE_NO_MATCH

            logger.warning(
                "%s No results — failure_type=%r  hits_seen=%d  tried=%s",
                ctx,
                failure_type,
                total_hits_seen,
                titles_to_try,
            )
            await save_failure(
                db_path,
                failure_type=failure_type,
                tried_queries=titles_to_try,
                cards_seen=total_hits_seen,
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

        if emit_json and json_dir is not None:
            _write_entry_json(
                json_dir,
                mal_id=entry.mal_id,
                title=entry.title,
                used_query=used_query,
                results=results,
                attempt_logs=all_attempt_logs,
            )

        if emit_raw and raw_dir is not None and entry_raw_log:
            _write_raw_log(raw_dir, entry.mal_id, entry.title, entry_raw_log)

        summary.append(
            {
                "mal_title": entry.title,
                "used_query": used_query,
                "result_count": len(results),
                "matched": any(r.matched_query for r in results),
            }
        )

        progress.advance(task_id)

    return summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


async def run(args) -> None:
    timeout_seconds = args.timeout or None
    logger.info(
        "Starting sync — timeout=%s",
        f"{timeout_seconds}s" if timeout_seconds else "none",
    )
    try:
        async with asyncio.timeout(timeout_seconds):
            await _run_impl(args)
    except asyncio.TimeoutError:
        logger.error("Sync timed out after %ss", timeout_seconds)
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
        logger.info("Dry run — skipping Spotify stage")
        return

    if args.confirm:
        if not Confirm.ask("\nRun Spotify?", console=console):
            logger.info("Spotify stage skipped — user declined")
            return

    await run_spotify_stage(
        args.db,
        megaplaylist=bool(args.megaplaylist),
        progress=progress,
        username=args.username,
    )


async def run_cached_mode(args) -> None:
    logger.info("Cached mode — skipping MAL fetch and AniPlaylist scrape")

    with create_progress() as progress:
        await run_spotify_stage_if_needed(args, progress)


def create_client(args) -> tuple[
    MALClient | AniListClient,
    str,
    str | None,
]:
    if args.anilist:
        client = AniListClient.from_env(username=args.username)
        status = (
            ANILIST_STATUS_MAP.get(normalize_status(args.status))
            if args.status
            else None
        )
        return client, "AniList", status

    client = MALClient.from_env(username=args.username)
    status = normalize_status(args.status) if args.status else None

    return client, "MAL", status


async def run_main_pipeline(args) -> list[dict[str, object]]:
    client, source_label, status = create_client(args)
    algolia = AlgoliaClient.from_env()

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
                logger.error("No %s anime entries found", source_label)
                return []

            logger.info("Proceeding with %d %s entries", len(entries), source_label)

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
                    algolia=algolia,
                    exact_filter=not args.no_exact_filter,
                    emit_json=args.json,
                    emit_raw=args.raw,
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
            logger.warning("Error closing %s client: %s", source_label, exc)
        try:
            await algolia.close()
        except Exception as exc:
            logger.warning("Error closing AlgoliaClient: %s", exc)


async def _run_impl(args) -> None:
    await init_db(args.db)

    if getattr(args, "cached", False):
        await run_cached_mode(args)
        return

    summary = await run_main_pipeline(args)

    if args.json:
        logger.info("JSON output written to debug/json/")

    if args.dry_run:
        logger.info("Dry run — skipping Spotify stage")
        return

    if args.confirm:
        if not Confirm.ask("Run Spotify?", console=console):
            logger.info("Spotify stage skipped — user declined")
            return

    with create_progress() as progress:
        await run_spotify_stage(
            args.db,
            megaplaylist=bool(args.megaplaylist),
            progress=progress,
            username=args.username,
        )
