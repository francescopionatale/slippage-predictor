"""Lightweight logging helpers.

Library code calls :func:`get_logger` and emits via the standard
``logging`` module — it never calls ``print``. CLIs/scripts call
:func:`configure_logging` once at start-up to route those records to the
console at the desired verbosity.
"""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """Return the module logger (no handler attached — configured by the app)."""
    return logging.getLogger(name)


def configure_logging(level: int | str = logging.INFO) -> None:
    """Attach a console handler at ``level`` to the package root logger.

    Idempotent: repeated calls update the level without stacking handlers.
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
    root = logging.getLogger("slippage")
    root.setLevel(level)
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
    else:
        for h in root.handlers:
            h.setLevel(level)
