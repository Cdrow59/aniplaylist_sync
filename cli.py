"""Thin CLI entrypoint for the MAL -> AniPlaylist sync.

This module only parses arguments and delegates to `main.run(args)`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv

from db import DB_PATH
from logging_config import setup_logging

load_dotenv(dotenv_path=Path(__file__).with_name(".env"), override=False)


def _make_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch a MAL user's anime titles and search each one on AniPlaylist."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH, help="SQLite database path")
    parser.add_argument("--username", default="Cdrow", help="MAL username to fetch")
    parser.add_argument("--client-id", default=None, help="MAL client ID")
    parser.add_argument("--access-token", default=None, help="MAL access token")
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
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser


def main() -> None:
    parser = _make_argparser()
    args = parser.parse_args()

    # Initialize logging with specified level
    log_level = getattr(logging, args.log_level, logging.INFO)
    setup_logging(level=log_level, log_file=Path("aniplaylist_sync.log"))

    # delegate to orchestrator in main.py
    import main as orchestrator

    try:
        asyncio.run(orchestrator.run(args))
    except KeyboardInterrupt:
        logger = logging.getLogger(__name__)
        logger.info("Sync interrupted by user")
        raise
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Sync failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
