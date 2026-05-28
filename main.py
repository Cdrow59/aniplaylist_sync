"""Main orchestration and stages for MAL -> AniPlaylist sync.

This module contains the core logic and helper functions. `cli.py` is a
thin entrypoint that parses arguments and calls `run(args)` below.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from aniplaylist import SearchResult, search_aniplaylist
from db import DB_PATH, init_db, save_failure, save_run
from mal import MALAnimeEntry, MALClient
from series import discover_series, save_series_clusters

DEFAULT_DB_PATH = DB_PATH
MAL_STATUS_ALIASES = {
    "complete": "completed",
    "completed": "completed",
    "watching": "watching",
    "on_hold": "on_hold",
    "dropped": "dropped",
    "plan_to_watch": "plan_to_watch",
}


def read_env(name: str, fallback: str | None = None) -> str:
    value = os.getenv(name, fallback)
    if value is None or not value.strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value.strip()


def candidate_titles(entry: MALAnimeEntry) -> list[str]:
    candidates: list[str] = []

    def add_candidate(value: object | None) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

    alternative_titles = entry.alternative_titles or {}
    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("en"))

    add_candidate(entry.title)

    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("ja"))

    return candidates


def exact_title_candidates(entry: MALAnimeEntry) -> list[str]:
    titles: list[str] = []

    def add_candidate(value: object | None) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in titles:
                titles.append(cleaned)

    alternative_titles = entry.alternative_titles or {}
    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("en"))

    add_candidate(entry.title)

    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("ja"))

    return titles


def title_metadata(entry: MALAnimeEntry) -> tuple[str | None, str | None, str | None]:
    native_title = entry.title or None
    alternative_titles = entry.alternative_titles or {}

    english_title = None
    japanese_title = None
    if isinstance(alternative_titles, dict):
        english_title = alternative_titles.get("en")
        japanese_title = alternative_titles.get("ja")

    def clean(value: object | None) -> str | None:
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
        return None

    return clean(native_title), clean(english_title), clean(japanese_title)


def normalize_status(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in MAL_STATUS_ALIASES:
        raise ValueError(
            "Invalid status. Use one of: complete, completed, watching, on_hold, dropped, plan_to_watch"
        )
    return MAL_STATUS_ALIASES[normalized]


async def run_series_stage(
    db_path: Path, client: MALClient, mal_entries: list[MALAnimeEntry]
) -> None:
    seed_ids = [entry.mal_id for entry in mal_entries]
    if not seed_ids:
        return

    discovery = await discover_series(client, seed_ids)
    await save_series_clusters(db_path, discovery.clusters)


async def run_aniplaylist_stage(
    db_path: Path,
    mal_entries: list[MALAnimeEntry],
    *,
    headed: bool,
    exact_filter: bool,
    aniplaylist_delay: float,
    emit_json: bool,
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task_id = progress.add_task("AniPlaylist", total=len(mal_entries))

        for entry in mal_entries:
            titles_to_try = candidate_titles(entry)
            native_title, english_title, japanese_title = title_metadata(entry)
            exact_titles = exact_title_candidates(entry)
            results: list[SearchResult] = []
            used_query = titles_to_try[0] if titles_to_try else entry.title

            for title_query in titles_to_try:
                try:
                    results = await search_aniplaylist(
                        title_query,
                        headless=not headed,
                        exact_titles=exact_titles,
                    )
                except Exception as exc:
                    await save_failure(
                        db_path,
                        title_query,
                        reason=str(exc),
                        mal_id=entry.mal_id,
                        native_title=native_title,
                        english_title=english_title,
                        japanese_title=japanese_title,
                        status=entry.status,
                    )
                    continue

                used_query = title_query
                if results:
                    break

            if not exact_filter:
                filtered = [result for result in results if result.matched_query]
                results = filtered or results

            if not results:
                await save_failure(
                    db_path,
                    used_query,
                    reason="No AniPlaylist results matched",
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
                    "matched": any(result.matched_query for result in results),
                }
            )

            if aniplaylist_delay > 0:
                await asyncio.sleep(aniplaylist_delay)

            progress.advance(task_id)

    if emit_json:
        return summary

    return summary


async def run(args) -> None:
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
        mal_entries = await client.list_user_anime(status=mal_status)

        if args.limit and args.limit > 0:
            mal_entries = mal_entries[: args.limit]

        if not mal_entries:
            print("No MAL anime entries found.")
            return

        series_task = asyncio.create_task(
            run_series_stage(args.db, client, mal_entries)
        )
        aniplaylist_task = asyncio.create_task(
            run_aniplaylist_stage(
                args.db,
                mal_entries,
                headed=args.headed,
                exact_filter=not args.no_exact_filter,
                aniplaylist_delay=aniplaylist_delay,
                emit_json=args.json,
            )
        )

        summary, _ = await asyncio.gather(aniplaylist_task, series_task)
    finally:
        await client.close()

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # allow running the orchestrator directly for debugging
    import argparse

    parser = argparse.ArgumentParser()
    # mimic the CLI arguments so this module can be run directly
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--username", default=None)
    parser.add_argument("--client-id", default=None)
    parser.add_argument("--access-token", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-exact-filter", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--aniplaylist-delay", type=float, default=None)
    parser.add_argument("--status", default=None)
    args = parser.parse_args()
    asyncio.run(run(args))
