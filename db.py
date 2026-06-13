"""SQLite persistence helpers."""

from __future__ import annotations

import json
import logging
from parser import SearchResult
from pathlib import Path
from typing import Iterable

import aiosqlite

logger = logging.getLogger(__name__)
DB_PATH = Path("aniplaylist.sqlite3")

# ---------------------------------------------------------------------------
# Failure type constants
# ---------------------------------------------------------------------------

# AniPlaylist returned zero cards for every candidate title tried.
# The title simply isn't indexed on AniPlaylist.
FAILURE_NOT_FOUND = "not_found"

# AniPlaylist returned cards for at least one candidate, but none matched
# any of our exact candidate titles (including portal synonym checks).
FAILURE_NO_MATCH = "no_match"

# Every scrape attempt for every candidate raised an exception or timed out
# before returning any cards.  Likely a network or Playwright issue.
FAILURE_SCRAPE_ERROR = "scrape_error"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


async def init_db(db_path: Path) -> None:
    logger.debug("Initializing database schema at %s", db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")

        # One row per AniPlaylist search that returned results.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS searches (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                mal_id          INTEGER,
                query           TEXT NOT NULL,
                native_title    TEXT,
                english_title   TEXT,
                japanese_title  TEXT,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Individual song results from a search.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id       INTEGER NOT NULL,
                mal_id          INTEGER,
                native_title    TEXT,
                english_title   TEXT,
                japanese_title  TEXT,
                anime_title     TEXT NOT NULL,
                song_type       TEXT,
                sequence        INTEGER,
                title           TEXT,
                artists_json    TEXT,
                spotify_link    TEXT,
                matched_query   INTEGER NOT NULL,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(search_id) REFERENCES searches(id)
            )
        """)

        # One row per MAL entry that produced no usable results.
        #
        # failure_type  — queryable enum: not_found | no_match | scrape_error
        # tried_queries — JSON array of every title string actually submitted
        #                 to AniPlaylist, in order
        # cards_seen    — total cards returned across all attempts (0 for
        #                 not_found and scrape_error; >0 for no_match)
        # attempt_log   — JSON array of per-attempt diagnostic dicts;
        #                 NULL when failure_type is not_found (nothing to debug)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS failed (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                mal_id          INTEGER,
                native_title    TEXT,
                english_title   TEXT,
                japanese_title  TEXT,
                mal_status      TEXT,
                failure_type    TEXT NOT NULL
                                    CHECK(failure_type IN (
                                        'not_found', 'no_match', 'scrape_error'
                                    )),
                tried_queries   TEXT NOT NULL,
                cards_seen      INTEGER NOT NULL DEFAULT 0,
                attempt_log     TEXT,
                created_at      TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.execute("""
            CREATE TABLE IF NOT EXISTS series (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                series_name           TEXT NOT NULL,
                member_ids_json       TEXT NOT NULL,
                member_count          INTEGER NOT NULL,
                representative_mal_id INTEGER,
                created_at            TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.commit()
        logger.info("Database initialized at %s", db_path)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


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
    result_rows = list(results)
    matched_count = sum(1 for r in result_rows if r.matched_query)

    logger.debug(
        "save_run: mal_id=%s query=%r -> %d result(s), %d matched",
        mal_id,
        query,
        len(result_rows),
        matched_count,
    )

    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO searches(
                mal_id, query, native_title, english_title, japanese_title
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (mal_id, query, native_title, english_title, japanese_title),
        )
        search_id = int(cursor.lastrowid)

        for result in result_rows:
            await db.execute(
                """
                INSERT INTO results(
                    search_id, mal_id, native_title, english_title, japanese_title,
                    anime_title, song_type, sequence, title, artists_json,
                    spotify_link, matched_query
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
        logger.debug(
            "save_run: committed search_id=%d (mal_id=%s, %d result(s))",
            search_id,
            mal_id,
            len(result_rows),
        )


async def save_failure(
    db_path: Path,
    *,
    failure_type: str,
    tried_queries: list[str],
    cards_seen: int = 0,
    mal_id: int | None = None,
    native_title: str | None = None,
    english_title: str | None = None,
    japanese_title: str | None = None,
    mal_status: str | None = None,
    attempt_log: list[dict] | None = None,
) -> None:
    """
    Persist a failed AniPlaylist lookup.

    Parameters
    ----------
    failure_type : str
        One of FAILURE_NOT_FOUND, FAILURE_NO_MATCH, FAILURE_SCRAPE_ERROR.
    tried_queries : list[str]
        Every candidate title string actually submitted to AniPlaylist, in order.
    cards_seen : int
        Total result cards observed across all attempts.  Should be 0 for
        not_found and scrape_error, and >0 for no_match.
    attempt_log : list[dict] | None
        Per-attempt structured diagnostics from _search_one().  Pass None for
        not_found failures — there's nothing useful to debug there.
    """
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO failed(
                mal_id, native_title, english_title, japanese_title,
                mal_status, failure_type, tried_queries, cards_seen, attempt_log
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mal_id,
                native_title,
                english_title,
                japanese_title,
                mal_status,
                failure_type,
                json.dumps(tried_queries, ensure_ascii=False),
                cards_seen,
                json.dumps(attempt_log, ensure_ascii=False) if attempt_log else None,
            ),
        )
        await db.commit()
        logger.debug(
            "save_failure: mal_id=%s type=%r cards_seen=%d tried=%s",
            mal_id,
            failure_type,
            cards_seen,
            tried_queries,
        )
