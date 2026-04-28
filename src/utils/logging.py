"""Logging configuration and progress display utilities.

Provides:
    - setup_logger() — configures a named logger with the standard format
    - log_section()  — prints a visible progress header inside notebook output

The standard log format is:
    YYYY-MM-DD HH:MM:SS | LEVEL    | logger_name | message

This format is used consistently across all pipeline stages so that log
output from the KB, golden dataset, retrieval, and RAG modules can be
read in a unified way during long-running notebook executions.
"""

import logging

LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a named logger with the standard pipeline format.

    Idempotent — if the logger already has handlers, it is returned as-is
    rather than duplicating handlers (which would cause duplicate log lines
    in Jupyter notebook re-executions).

    Args:
        name: Logger name. Conventionally the pipeline stage name,
            e.g. "kb_pipeline", "golden_dataset", "retrieval".
        level: Log level threshold. Defaults to logging.INFO.

    Returns:
        Configured logger instance ready for use.

    Example:
        >>> logger = setup_logger("kb_pipeline")
        >>> logger.info("Starting Knowledge Base generation.")
        2026-04-27 10:30:22 | INFO     | kb_pipeline | Starting Knowledge Base generation.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers when notebooks are re-run
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


def log_section(title: str, current: int, total: int) -> None:
    """Print a clearly visible section header showing pipeline progress.

    Used inside notebook orchestration cells to make it obvious which
    pipeline step is currently executing. The progress bar is a snapshot
    at the moment the section starts — historical markers in the cell
    output, not a live updating bar.

    Args:
        title: Section name to display in the header.
        current: Current step number (1-indexed).
        total: Total number of steps in this phase.

    Example:
        >>> log_section("Computing daily statistics", 2, 7)

        ============================================================
          STEP 2/7  [██░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 28%
          Computing daily statistics
        ============================================================
    """
    bar = "█" * current + "░" * (total - current)
    pct = int(current / total * 100)
    width = 60
    print(f"\n{'=' * width}")
    print(f"  STEP {current}/{total}  [{bar}] {pct}%")
    print(f"  {title}")
    print(f"{'=' * width}\n")
