"""Single source of truth for log formatting across the project."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_INITIALIZED = False


def get_logger(name: str = "ceng467", level: int = logging.INFO,
                log_file: str | Path | None = None) -> logging.Logger:
    """Return a logger configured once per process."""
    global _INITIALIZED
    logger = logging.getLogger(name)
    if _INITIALIZED:
        return logger

    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    _INITIALIZED = True
    return logger
