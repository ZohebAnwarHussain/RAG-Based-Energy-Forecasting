"""CSV file handling utilities.

Helpers for the append-mode CSV output pattern used throughout the
pipeline. KB summaries, golden answers, and request logs all use this
pattern — write each new record to disk immediately so a crash or
disconnection loses at most one entry rather than the entire batch.
"""

import csv
from pathlib import Path
from typing import List


def init_csv_with_headers(path: Path, headers: List[str]) -> None:
    """Create a CSV file with column headers if it does not already exist.

    Used for log files and incremental output CSVs. If the file already
    exists from a previous run, it is left untouched so historical
    records are preserved across multiple notebook executions.

    Args:
        path: Filesystem path where the CSV should be created.
        headers: Column names to write as the first row of the CSV.

    Example:
        >>> from pathlib import Path
        >>> init_csv_with_headers(
        ...     Path("logs/requests.csv"),
        ...     ["timestamp", "status", "latency_s"]
        ... )
    """
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file_handle:
        csv.DictWriter(file_handle, fieldnames=headers).writeheader()
