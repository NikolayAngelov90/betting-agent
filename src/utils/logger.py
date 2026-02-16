"""Logging configuration for Football Betting Agent."""

import uuid
from loguru import logger
from pathlib import Path
import sys


def setup_logger(log_level: str = "INFO", log_file: str = "logs/betting_agent.log"):
    """Configure loguru logger.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Remove default handler
    logger.remove()

    # Bind a unique run ID so log lines from one execution can be grouped
    run_id = uuid.uuid4().hex[:8]
    logger.configure(extra={"run_id": run_id})

    # Add console handler with colors
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
    )

    # Add file handler — daily rotation, 30-day retention, compressed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | run={extra[run_id]} - {message}",
        level=log_level,
        rotation="1 day",
        retention="30 days",
        compression="zip",
    )

    logger.info(f"Logger initialized — level={log_level}, run_id={run_id}")


def get_logger():
    """Get logger instance."""
    return logger
