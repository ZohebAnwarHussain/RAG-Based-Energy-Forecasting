"""Timestamp formatting utilities.

Provides a single source of truth for timestamp formatting across the
pipeline. All `generated_at` columns in output CSVs use the format
returned by get_timestamp() — DD-MM-YYYY HH:MM:SS in UTC.

This format is human-readable and unambiguous (no MM/DD vs DD/MM
confusion) and is used consistently in:
    - KB generated_at column
    - Golden dataset generated_at column
    - Request and failure log files
"""

from datetime import datetime, timezone

# Format string used for all timestamp output across the pipeline
TIMESTAMP_FORMAT: str = "%d-%m-%Y %H:%M:%S"


def get_timestamp() -> str:
    """Return the current UTC time in DD-MM-YYYY HH:MM:SS format.

    Uses datetime.now(timezone.utc) instead of the deprecated
    datetime.utcnow() to comply with Python 3.12+ deprecation warnings
    while still producing UTC-anchored timestamps.

    Returns:
        Current UTC timestamp string formatted as DD-MM-YYYY HH:MM:SS.
        Example: "27-04-2026 10:30:22"

    Example:
        >>> get_timestamp()
        '27-04-2026 10:30:22'
    """
    return datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)
