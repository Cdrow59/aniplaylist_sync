"""Thin CLI entrypoint for the MAL -> AniPlaylist sync.

This module only parses arguments and delegates to `main.run(args)`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from db import DB_PATH
from logging_config import setup_logging

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=True)


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a MAL user's anime titles and search each one on AniPlaylist."
    )
    parser.add_argument("--db", type=Path, default=None, help="SQLite database path")
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        required=True,
        help="Username to run the query for.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Does a dry run without interacting with Spotify",
    )
    parser.add_argument(
        "--megaplaylist",
        action="store_true",
        help="Put all results into one Spotify playlist instead of one playlist per series",
    )
    parser.add_argument(
        "--no-exact-filter",
        action="store_true",
        help="Keep all AniPlaylist results instead of only exact anime-title matches",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write per-entry parsed results to debug/json/",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Write raw Algolia HTTP responses to debug/raw/",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="Confirm running spotify"
    )
    parser.add_argument(
        "--spotify-user",
        type=str,
        default=None,
        help=(
            "Run Spotify actions as a different account than the default. "
            "That account must already be added as a user on the app in "
            "the Spotify Developer Dashboard (required while the app is "
            "in Development Mode). Each --spotify-user gets its own saved "
            "refresh token (SPOTIFY_REFRESH_TOKEN_<NAME> in .env), so "
            "switching users doesn't require re-authorising every run."
        ),
    )
    parser.add_argument(
        "--aniplaylist-delay",
        type=float,
        default=None,
        help="Seconds to wait between AniPlaylist searches",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of MAL anime entries processed",
    )
    parser.add_argument(
        "--status",
        default=None,
        help="Filter MAL anime by status (complete, watching, on_hold, dropped, plan_to_watch)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for entire operation (default: 1800)",
    )
    parser.add_argument(
        "--console-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Console Logging level",
    )
    parser.add_argument(
        "--file-level",
        type=str,
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="File Logging level",
    )
    parser.add_argument(
        "--anilist",
        action="store_true",
        help="Use AniList instead of MAL as the anime list source",
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help=(
            "Skip MAL fetching and AniPlaylist scraping entirely; "
            "run the Spotify stage against existing DB data only."
        ),
    )
    return parser


def main() -> None:
    parser = _make_argparser()
    args = parser.parse_args()

    if args.db is None:
        args.db = Path(f"aniplaylist_{args.username}.sqlite3")

    safe_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = Path("debug/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    console_level = getattr(logging, args.console_level, logging.INFO)
    file_level = getattr(logging, args.file_level, logging.DEBUG)
    setup_logging(
        console_level=console_level,
        file_level=file_level,
        log_file=Path(f"debug/logs/aniplaylist_sync_{safe_timestamp}.log"),
    )

    logger = logging.getLogger(__name__)
    logger.debug("Parsed CLI args: %s", vars(args))

    import main as orchestrator

    try:
        asyncio.run(orchestrator.run(args))
    except KeyboardInterrupt:
        logger.info("Sync interrupted by user")
        raise
    except Exception as e:
        logger.error("Sync failed: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
