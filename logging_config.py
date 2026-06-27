"""Structured logging configuration for production monitoring."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

console = Console()


def _handle_uncaught_exception(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_traceback,
) -> None:
    """Log uncaught exceptions (main thread)."""

    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.getLogger("exception").critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def _handle_asyncio_exception(loop, context: dict) -> None:
    """Log uncaught asyncio task exceptions."""

    exception = context.get("exception")
    message = context.get("message", "Unhandled asyncio exception")

    logger = logging.getLogger("asyncio")

    if exception:
        logger.error(
            message,
            exc_info=(
                type(exception),
                exception,
                exception.__traceback__,
            ),
        )
    else:
        logger.error(message)


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure logging with structured format for production.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file for persistent logging
    """

    # Rich console handler (pretty terminal output)
    console_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
    )
    console_handler.setLevel(level)

    # Root logger setup
    root_logger = logging.getLogger()

    # Prevent duplicate handlers on re-init
    root_logger.handlers.clear()

    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)

    # File logging (structured, machine-readable)
    if log_file:
        file_formatter = logging.Formatter(
            fmt="%(module)-10s:%(lineno)-4d | %(asctime)s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(file_formatter)

        root_logger.addHandler(file_handler)

    # Catch uncaught sync exceptions
    sys.excepthook = _handle_uncaught_exception

    # Reduce noise from third-party libs
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)


def configure_asyncio_logging() -> None:
    """Attach asyncio exception handler to current event loop."""

    try:
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(_handle_asyncio_exception)
    except RuntimeError:
        # No running loop yet
        pass


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)
