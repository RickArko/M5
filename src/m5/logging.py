"""Tiny loguru wrapper so every module logs the same way."""

from __future__ import annotations

import os
import sys

from loguru import logger as _logger


def configure(level: str | None = None) -> None:
    resolved_level = level or os.getenv("LOG_LEVEL") or "INFO"
    _logger.remove()
    _logger.add(
        sys.stderr,
        level=resolved_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


configure()
logger = _logger

__all__ = ["configure", "logger"]
