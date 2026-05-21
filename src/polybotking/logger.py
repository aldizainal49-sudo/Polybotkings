"""
Structured logging with structlog + rich console output.
"""

import sys
import structlog
from pathlib import Path
from datetime import datetime
from polybotking.config import settings


def setup_logging():
    """Initialize structured logging for the bot."""
    log_dir = settings.logs_dir
    log_file = log_dir / f"polybotking_{datetime.now().strftime('%Y%m%d')}.log"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.alerts.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            structlog.get_level_from_name(settings.alerts.log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(
            file=open(log_file, "a") if log_file else sys.stdout
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a named logger instance."""
    return structlog.get_logger(name)
