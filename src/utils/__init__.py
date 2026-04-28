"""Utility helpers shared across all pipeline stages.

Provides:
    - Logger setup with consistent format and date format
    - Section progress headers for visible notebook execution
    - UTC timestamp formatting in DD-MM-YYYY HH:MM:SS
    - CSV append-mode helpers for incremental output

Usage:
    from src.utils import setup_logger, log_section, get_timestamp

    logger = setup_logger("kb_pipeline")
    log_section("Loading raw data", current=1, total=7)
    timestamp = get_timestamp()  # "27-04-2026 10:30:22"
"""

from src.utils.io import init_csv_with_headers
from src.utils.logging import log_section, setup_logger
from src.utils.timestamps import get_timestamp

__all__ = [
    "setup_logger",
    "log_section",
    "get_timestamp",
    "init_csv_with_headers",
]