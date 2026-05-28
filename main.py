"""Batch MAL -> AniPlaylist linker.

This script pulls a user's MAL anime list, uses each anime title as an AniPlaylist
search query, and stores the query/results in SQLite.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from aniplaylist_bot import (
    DB_PATH,
    SearchResult,
    init_db,
    save_failure,
    save_run,
    search_aniplaylist,
)
from mal import MALClient
from series_discovery import discover_series, save_series_clusters

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)

DEFAULT_DB_PATH = DB_PATH
DEFAULT_CONCURRENCY = 1
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


def chunked(items: Iterable[tuple[int, str]], size: int) -> list[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    batches: list[list[tuple[int, str]]] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def candidate_titles(entry) -> list[str]:
    candidates: list[str] = []

    def add_candidate(value: object | None) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in candidates:
                candidates.append(cleaned)

    alternative_titles = getattr(entry, "alternative_titles", None) or {}
    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("en"))

    add_candidate(getattr(entry, "title", None))

    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("ja"))

    return candidates


def exact_title_candidates(entry) -> list[str]:
    titles: list[str] = []

    def add_candidate(value: object | None) -> None:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned and cleaned not in titles:
                titles.append(cleaned)

    alternative_titles = getattr(entry, "alternative_titles", None) or {}
    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("en"))

    add_candidate(getattr(entry, "title", None))

    if isinstance(alternative_titles, dict):
        add_candidate(alternative_titles.get("ja"))

    return titles


def title_metadata(entry) -> tuple[str | None, str | None, str | None]:
    native_title = getattr(entry, "title", None)
    alternative_titles = getattr(entry, "alternative_titles", None) or {}

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


async def process_title(
    db_path: Path, query: str, exact_only: bool
) -> list[SearchResult]:
    results = await search_aniplaylist(query)
    if exact_only:
        filtered = [result for result in results if result.matched_query]
        results = filtered or results
    await save_run(db_path, query, results)
    return results


def normalize_status(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized not in MAL_STATUS_ALIASES:
        raise ValueError(
            "Invalid status. Use one of: complete, completed, watching, on_hold, dropped, plan_to_watch"
        )
    return MAL_STATUS_ALIASES[normalized]


async def run_series_stage(
    db_path: Path, client: MALClient, mal_entries: list[object]
) -> None:
    seed_ids = [entry.mal_id for entry in mal_entries]
    if not seed_ids:
        return

    discovery = await discover_series(client, seed_ids)
    await save_series_clusters(db_path, discovery.clusters)


async def run_aniplaylist_stage(
    db_path: Path,
    mal_entries: list[object],
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
        task = progress.add_task("AniPlaylist", total=len(mal_entries))

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

            progress.advance(task)

    if emit_json:
        return summary

    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a MAL user's anime titles and search each one on AniPlaylist."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path"
    )
    parser.add_argument("--username", default=None, help="MAL username or @me")
    parser.add_argument("--client-id", default=None, help="MAL client ID")
    parser.add_argument("--access-token", default=None, help="MAL access token")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of MAL entries to process",
    )
    parser.add_argument(
        "--no-exact-filter",
        action="store_true",
        help="Keep all AniPlaylist results instead of only exact anime-title matches",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print a JSON summary at the end"
    )
    parser.add_argument(
        "--headed", action="store_true", help="Run Playwright in headed mode"
    )
    parser.add_argument(
        "--aniplaylist-delay",
        type=float,
        default=None,
        help="Seconds to wait between AniPlaylist searches",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Filter MAL anime by status (complete, watching, on_hold, dropped, plan_to_watch)",
    )
    args = parser.parse_args()

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
        return


if __name__ == "__main__":
    asyncio.run(main())
