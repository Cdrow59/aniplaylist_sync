"""SQLite persistence helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import aiosqlite

from aniplaylist import SearchResult

logger = logging.getLogger(__name__)
DB_PATH = Path("aniplaylist.sqlite3")


async def init_db(db_path: Path) -> None:
    """Initialize the SQLite database with required tables.

    Creates tables for searches, results, failed searches, and series data.
    All tables include mal_id for tracking MyAnimeList identifiers.

    Args:
        db_path: Path to the SQLite database file
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
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
        await db.commit()
        logger.info(f"Database initialized at {db_path}")


async def save_run(
    db_path: Path,
    query: str,
    results: Iterable[SearchResult],
    *,
    mal_id: int | None = None,
    native_title: str | None = None,
    english_title: str | None = None,
    japanese_title: str | None = None,
) -> None:
    """Save a successful search with its results to the database.

    Stores search metadata and all matching results. The query field contains
    all search attempt details for auditing purposes.

    Args:
        db_path: Path to the SQLite database file
        query: Search query used (can include JSON attempt details)
        results: Iterable of SearchResult objects
        mal_id: MyAnimeList ID for the anime
        native_title: Native title (typically Japanese)
        english_title: English title if available
        japanese_title: Japanese title if available
    """
    result_rows = list(results)

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
        search_id = int(cursor.lastrowid)

        for result in result_rows:
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
                    int(result.matched_query),
                ),
            )

        await db.commit()
        logger.debug(f"Saved search with MAL ID {mal_id}: {len(result_rows)} results")


async def save_failure(
    db_path: Path,
    query: str,
    *,
    reason: str,
    mal_id: int | None = None,
    native_title: str | None = None,
    english_title: str | None = None,
    japanese_title: str | None = None,
    status: str | None = None,
) -> None:
    """Save a failed search to the database for auditing and recovery.

    Stores complete failure information including the attempt details (query field),
    reason for failure, and MAL metadata.

    Args:
        db_path: Path to the SQLite database file
        query: Search query or JSON with attempt details (for recovery)
        reason: Reason for the failure
        mal_id: MyAnimeList ID for the anime
        native_title: Native title (typically Japanese)
        english_title: English title if available
        japanese_title: Japanese title if available
        status: User's watch status on MAL if available
    """
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
        logger.debug(f"Saved failure for MAL ID {mal_id}: {reason}")
