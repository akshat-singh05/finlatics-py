"""
Centralized logging configuration for Finlatics-Py.

Provides a factory function that returns a configured logger with:
  - Rotating file handler -> logs/app.log (5MB, 3 backups)
  - Console handler with severity-colored output
  - Timestamps, severity levels, module names
  - Optional JSON formatter (LOG_FORMAT=json)
  - Request-ID context injection via LogContext

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started")

    # With request context (API):
    from utils.logger import LogContext
    with LogContext(request_id="abc-123"):
        logger.info("Processing request")
"""

import os
import sys
import json
import logging
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import contextmanager


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "app.log"

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
LOG_FORMAT_TYPE = os.environ.get("LOG_FORMAT", "text").lower()

_INITIALIZED = False
_context = threading.local()


class LogContext:
    """Thread-local context manager for injecting request metadata into logs."""

    def __init__(self, **kwargs):
        self._data = kwargs

    def __enter__(self):
        for key, value in self._data.items():
            setattr(_context, key, value)
        return self

    def __exit__(self, *args):
        for key in self._data:
            if hasattr(_context, key):
                delattr(_context, key)


class ContextFilter(logging.Filter):
    """Inject thread-local context (e.g. request_id) into log records."""

    def filter(self, record):
        record.request_id = getattr(_context, "request_id", "-")
        return True


class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production environments."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def _ensure_log_dir():
    """Create the logs directory if it doesn't exist."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _setup_root_logger():
    """Configure the root logger with file and console handlers (once)."""
    global _INITIALIZED
    if _INITIALIZED:
        return
    _INITIALIZED = True

    _ensure_log_dir()

    root = logging.getLogger("finlatics")
    root.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Attached to the handlers, not the logger: logger-level filters do not
    # run for records propagated from child loggers (finlatics.<name>).
    context_filter = ContextFilter()

    if LOG_FORMAT_TYPE == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(request_id)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(context_filter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    console_handler.setFormatter(formatter)
    console_handler.addFilter(context_filter)
    root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger with file and console handlers.
    """
    _setup_root_logger()
    return logging.getLogger(f"finlatics.{name}")
