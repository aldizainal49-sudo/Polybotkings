"""
Structured logging with structlog + rich console output.
"""

import logging
import sys
from datetime import datetime

import structlog

from polybotking.config import settings


def setup_logging() -> None:
    """Initialize structured logging for the bot."""
    log_dir = settings.logs_dir
    # Reserve a daily log filename even if we don't currently write a file
    # handler (kept for forward compatibility).
    _ = log_dir / f"polybotking_{datetime.now().strftime('%Y%m%d')}.log"

    # Resolve Python logging level safely
    level_name = (settings.alerts.log_level or "INFO").upper()
    log_level = getattr(logging, level_name, logging.INFO)

    use_console_renderer = level_name == "DEBUG"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            (
                structlog.dev.ConsoleRenderer()
                if use_console_renderer
                else structlog.processors.JSONRenderer()
            ),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
