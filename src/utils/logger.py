"""Loguru logger setup for the application."""

from __future__ import annotations

import os
from typing import Any

from loguru import logger

from config import AppConfig


def init_logger(config: AppConfig) -> Any:
    """Initialize loguru with rotation, formatting, and level from config.

    Returns the configured logger.
    """
    logger.remove()

    log_level = config.log_level.upper()
    logs_dir = os.path.join(config.data_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "app.log")

    # Console sink
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    # File sink with rotation and retention
    logger.add(
        log_path,
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    return logger


