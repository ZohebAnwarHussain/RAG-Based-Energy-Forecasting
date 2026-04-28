"""Gemini API integration and resumable batch summary generation.

Three components:

    configure_gemini_kb()  Initialise and return a configured Gemini client.
                          Reads API key from environment (.env), Colab
                          Secrets, or interactive prompt as a fallback chain.

    call_gemini()          Send one prompt to Gemini and return the generated
                          text. Handles transient errors with exponential
                          back-off retry. Logs every request and permanent
                          failure to CSV log files for audit and recovery.

    generate_summaries()   Iterate over a prompt-input DataFrame, calling
                          call_gemini() per row. Appends each successful
                          result to the output CSV immediately so the
                          pipeline can resume from any interruption point
                          without losing already-generated summaries.

The output CSV schema is defined by SUMMARY_CSV_COLUMNS, with generated_at
in DD-MM-YYYY HH:MM:SS UTC format.
"""

from __future__ import annotations

import csv
import getpass
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import pandas as pd
from google import genai
from google.genai import types
from tqdm.auto import tqdm

from config import (
    KB_MAX_TOKENS,
    KB_MODEL_NAME,
    KB_TEMPERATURE,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
)
from src.knowledge_base.validation import is_valid_summary
from src.utils.io import init_csv_with_headers
from src.utils.timestamps import get_timestamp

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Output CSV schema
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_CSV_COLUMNS: List[str] = [
    "row_id",
    "dataset",
    "granularity",
    "context_json",
    "prompt_text",
    "summary",
    "generated_at",  # DD-MM-YYYY HH:MM:SS (UTC)
]


# ─────────────────────────────────────────────────────────────────────────────
# Log file schemas (used by log_request and log_failure)
# ─────────────────────────────────────────────────────────────────────────────

REQUEST_LOG_HEADERS: List[str] = [
    "timestamp",
    "dataset",
    "granularity",
    "row_id",
    "prompt_tokens_est",
    "status",
    "latency_s",
]

FAILED_ROW_HEADERS: List[str] = [
    "timestamp",
    "dataset",
    "granularity",
    "row_id",
    "error_msg",
]


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Client Configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_gemini_kb(api_key: Optional[str] = None) -> genai.Client:
    """Initialise and return a configured Gemini client for KB generation.

    Resolves the API key in this priority order:
        1. Explicit ``api_key`` argument (for testing)
        2. ``GEMINI_API_KEY`` environment variable (.env)
        3. Google Colab userdata secrets (Colab only)
        4. Interactive ``getpass`` prompt as final fallback

    Args:
        api_key: Optional explicit API key override. Useful for tests.

    Returns:
        Configured genai.Client instance ready to call ``KB_MODEL_NAME``.

    Raises:
        ValueError: If no API key can be resolved from any source.

    Example:
        >>> client = configure_gemini_kb()
        >>> response = client.models.generate_content(
        ...     model=KB_MODEL_NAME, contents="Hello"
        ... )
    """
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            logger.info(
                "Gemini API key loaded from environment variable."
            )

    if not api_key:
        try:
            from google.colab import userdata  # noqa: PLC0415

            api_key = userdata.get("GEMINI_API_KEY")
            if api_key:
                logger.info("Gemini API key loaded from Colab Secrets.")
        except (ImportError, Exception):  # noqa: BLE001
            pass

    if not api_key:
        logger.info(
            "API key not found in environment. Prompting for manual entry."
        )
        api_key = getpass.getpass("Enter your Gemini API key: ")

    if not api_key:
        raise ValueError(
            "No Gemini API key could be resolved. "
            "Add GEMINI_API_KEY to your .env file or Colab Secrets."
        )

    client = genai.Client(api_key=api_key)
    logger.info(
        "Gemini KB client configured for model '%s'.", KB_MODEL_NAME
    )
    return client


# Generation config used by every call_gemini() invocation.
# Low temperature ensures factual, reproducible summaries — required for
# thesis evaluation reproducibility.
KB_GENERATION_CONFIG = types.GenerateContentConfig(
    temperature=KB_TEMPERATURE,
    max_output_tokens=KB_MAX_TOKENS,
)


# ─────────────────────────────────────────────────────────────────────────────
# Logging Utilities
# ─────────────────────────────────────────────────────────────────────────────

def log_request(
    log_path: Path,
    dataset: str,
    granularity: str,
    row_id: str,
    prompt_tokens_est: int,
    status: str,
    latency_s: float,
) -> None:
    """Append a single API call record to the request log CSV.

    Called after every Gemini API call (success or failure) to maintain
    a full audit trail. Token count is estimated from prompt length as
    ``len(prompt) // 4`` which approximates GPT-style tokenisation.

    Args:
        log_path: Path to gemini_requests_log.csv.
        dataset: Source dataset, e.g. 'gefcom' or 'household'.
        granularity: Summary granularity, e.g. 'daily', 'weekly'.
        row_id: Unique prompt row identifier.
        prompt_tokens_est: Estimated token count (len(prompt) // 4).
        status: Outcome — 'success' or 'error_attempt{n}'.
        latency_s: Wall-clock duration of the API call in seconds.
    """
    record = {
        "timestamp":         get_timestamp(),
        "dataset":           dataset,
        "granularity":       granularity,
        "row_id":            row_id,
        "prompt_tokens_est": prompt_tokens_est,
        "status":            status,
        "latency_s":         round(latency_s, 3),
    }
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=REQUEST_LOG_HEADERS).writerow(record)


def log_failure(
    log_path: Path,
    dataset: str,
    granularity: str,
    row_id: str,
    error_msg: str,
) -> None:
    """Append a permanently-failed row record to the failure log CSV.

    Called only when a row exhausts all retry attempts without success.
    Failure log is used by the retry mechanism to re-attempt failed
    rows after the API recovers, without re-processing successful ones.

    Args:
        log_path: Path to failed_rows.csv.
        dataset: Source dataset identifier.
        granularity: Summary granularity.
        row_id: Unique identifier of the row that failed permanently.
        error_msg: Exception message from the final failed attempt.
    """
    record = {
        "timestamp":   get_timestamp(),
        "dataset":     dataset,
        "granularity": granularity,
        "row_id":      row_id,
        "error_msg":   error_msg,
    }
    with log_path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=FAILED_ROW_HEADERS).writerow(record)


# ─────────────────────────────────────────────────────────────────────────────
# Core API Call with Retry
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini(
    client: genai.Client,
    prompt: str,
    row_id: str,
    dataset: str,
    granularity: str,
    request_log_path: Path,
    failure_log_path: Path,
    max_retries: int = MAX_RETRIES,
    retry_backoff: float = RETRY_BACKOFF_SECONDS,
) -> Optional[str]:
    """Send a single prompt to Gemini and return the generated text.

    Implements exponential back-off retry to handle transient server
    errors (e.g. 503 UNAVAILABLE during free-tier high-load periods).
    Wait time doubles on each retry attempt: first retry waits
    ``retry_backoff`` seconds, second waits ``2 * retry_backoff``, etc.

    If all retry attempts are exhausted, the row is logged to
    ``failure_log_path`` for later review and possible re-attempt.

    Args:
        client: Configured genai.Client from configure_gemini_kb().
        prompt: Full prompt string to submit. Should include both the
            template instructions and filled-in statistics.
        row_id: Unique row identifier for logging and audit.
        dataset: Dataset name for logging.
        granularity: Summary type for logging.
        request_log_path: Path to gemini_requests_log.csv.
        failure_log_path: Path to failed_rows.csv.
        max_retries: Maximum retry attempts before permanent failure.
        retry_backoff: Base wait seconds before first retry. Doubles
            on subsequent attempts.

    Returns:
        Generated summary text (stripped) on success, or None if all
        retries failed.
    """
    prompt_tokens_est = len(prompt) // 4

    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=KB_MODEL_NAME,
                contents=prompt,
                config=KB_GENERATION_CONFIG,
            )
            latency = time.perf_counter() - t0
            summary = response.text.strip()
            log_request(
                request_log_path,
                dataset, granularity, row_id,
                prompt_tokens_est, "success", latency,
            )
            logger.info("Generated '%s' in %.2fs.", row_id, latency)
            return summary

        except Exception as exc:  # noqa: BLE001
            latency  = time.perf_counter() - t0
            err_msg  = str(exc)
            log_request(
                request_log_path,
                dataset, granularity, row_id,
                prompt_tokens_est, f"error_attempt{attempt}", latency,
            )
            logger.warning(
                "Attempt %d/%d failed for '%s': %s",
                attempt, max_retries, row_id, err_msg,
            )

            if attempt < max_retries:
                wait = retry_backoff * (2 ** (attempt - 1))
                logger.info("Waiting %.0f s before retry...", wait)
                time.sleep(wait)
            else:
                log_failure(
                    failure_log_path,
                    dataset, granularity, row_id, err_msg,
                )
                logger.error("Permanently failed for '%s'.", row_id)

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Resumable Batch Generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_summaries(
    client: genai.Client,
    prompt_df: pd.DataFrame,
    output_path: Path,
    request_log_path: Path,
    failure_log_path: Path,
    request_delay: float = REQUEST_DELAY_SECONDS,
) -> pd.DataFrame:
    """Generate Gemini summaries for all rows in a prompt-input DataFrame.

    Iterates over ``prompt_df`` and calls call_gemini() for each row.
    Supports seamless resume after interruption by loading any existing
    output file at startup and skipping rows whose row_id is already
    present. Each successful result is written to ``output_path``
    immediately so at most one summary is lost in the event of a crash.

    Quality check — every generated summary passes through
    is_valid_summary() before being saved. Summaries that are too short
    or contain refusal phrases are dropped silently and the row is left
    pending for retry.

    Args:
        client: Configured genai.Client instance.
        prompt_df: DataFrame with at minimum columns
            [row_id, dataset, granularity, context_json, prompt_text].
        output_path: CSV path where successful results are appended.
            File is created with headers if it does not exist.
        request_log_path: Path to gemini_requests_log.csv.
        failure_log_path: Path to failed_rows.csv.
        request_delay: Seconds to sleep between API calls. Default
            REQUEST_DELAY_SECONDS (4.5s) keeps below 15 RPM free-tier limit.

    Returns:
        Complete DataFrame of all summaries in output_path, including
        rows generated in previous runs and this run.

    Example:
        >>> result_df = generate_summaries(
        ...     client, gefcom_daily_prompts,
        ...     PATHS["summaries_csv"] / "gefcom_daily_summaries.csv",
        ...     PATHS["logs"] / "gemini_requests_log.csv",
        ...     PATHS["logs"] / "failed_rows.csv",
        ... )
        >>> len(result_df)
        50
    """
    # ── Initialise output CSV with headers if needed ──────────────────────────
    if output_path.exists():
        existing      = pd.read_csv(output_path)
        completed_ids = set(existing["row_id"].tolist())
        logger.info(
            "Resuming '%s': %d rows already completed.",
            output_path.name, len(completed_ids),
        )
    else:
        existing      = pd.DataFrame(columns=SUMMARY_CSV_COLUMNS)
        completed_ids: set = set()
        existing.to_csv(output_path, index=False)

    # Initialise log files if not yet created
    init_csv_with_headers(request_log_path, REQUEST_LOG_HEADERS)
    init_csv_with_headers(failure_log_path, FAILED_ROW_HEADERS)

    # Filter out already-completed rows
    pending = prompt_df[
        ~prompt_df["row_id"].isin(completed_ids)
    ].reset_index(drop=True)
    logger.info(
        "%d rows pending for '%s'.", len(pending), output_path.name
    )

    for _, row in tqdm(
        pending.iterrows(),
        total=len(pending),
        desc=output_path.stem,
    ):
        summary = call_gemini(
            client=client,
            prompt=row["prompt_text"],
            row_id=row["row_id"],
            dataset=row["dataset"],
            granularity=row["granularity"],
            request_log_path=request_log_path,
            failure_log_path=failure_log_path,
        )

        if summary is not None and is_valid_summary(summary):
            record = {
                "row_id":       row["row_id"],
                "dataset":      row["dataset"],
                "granularity":  row["granularity"],
                "context_json": row["context_json"],
                "prompt_text":  row["prompt_text"],
                "summary":      summary,
                "generated_at": get_timestamp(),
            }
            pd.DataFrame([record]).to_csv(
                output_path, mode="a", header=False, index=False
            )
        elif summary is not None:
            logger.warning(
                "Summary for '%s' failed quality check — skipping.",
                row["row_id"],
            )

        time.sleep(request_delay)

    result_df = pd.read_csv(output_path)
    logger.info(
        "Generation complete for '%s': %d total summaries.",
        output_path.name, len(result_df),
    )
    return result_df
