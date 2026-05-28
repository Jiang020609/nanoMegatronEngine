"""Small logging helper for examples."""

from __future__ import annotations

import logging


def get_logger(name: str = "nano_megatron_engine") -> logging.Logger:
    """Return a basic console logger."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

