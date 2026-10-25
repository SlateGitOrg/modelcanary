"""Logging configuration shared by the CLI and the gateway."""

from __future__ import annotations

import logging

LOGGER_NAME = "modelcanary"


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the package logger."""
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def configure_logging(level: str = "INFO") -> None:
    """Configure a single stderr handler with a compact key=value format."""
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level.upper())
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s level=%(levelname)s logger=%(name)s %(message)s")
        )
        root.addHandler(handler)
    root.propagate = False
