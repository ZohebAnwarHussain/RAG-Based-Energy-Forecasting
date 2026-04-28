"""Data and summary validation utilities for the KB pipeline.

Two validation functions are provided:

    validate_aggregates() — runs on aggregate DataFrames before they
        are saved or used for prompt building. Removes rows with zero
        or null primary metric values that would produce misleading
        summaries (e.g. "Zone 4 recorded a mean load of 0.0 MW").

    is_valid_summary() — runs on Gemini-generated summary text before
        it is appended to the output CSV. Rejects summaries that are
        too short or contain AI refusal phrases.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


# Primary metric column candidates checked by validate_aggregates(),
# in priority order. The first match wins.
PRIMARY_COLUMN_CANDIDATES: List[str] = [
    "load_mean",                  # GEFCom daily, system_level daily
    "weekly_mean",                # GEFCom weekly
    "monthly_mean",               # GEFCom monthly
    "seasonal_mean",              # GEFCom seasonal
    "Global_active_power_mean",   # Household daily/weekly/monthly
    "yearly_mean",                # Household yearly
]


# Refusal phrases that indicate the LLM declined to generate a summary
# rather than producing valid output. Detected case-insensitively.
REFUSAL_PHRASES: List[str] = [
    "i cannot",
    "i'm unable",
    "as an ai",
    "i don't have",
    "i am unable",
    "i apologize",
    "i cannot provide",
]


def validate_aggregates(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Remove rows with zero or null primary metric values.

    Prevents invalid statistics (zero load, null power) from being
    passed to prompt templates, which would produce misleading
    summaries like 'Zone 4 recorded a mean load of 0.0 MW'.

    Detects the primary column by checking PRIMARY_COLUMN_CANDIDATES
    in order — the first column found in the DataFrame is used as
    the validation target.

    Args:
        df: Aggregate DataFrame to validate.
        label: Human-readable label for log messages,
            e.g. "gefcom_daily" or "household_monthly".

    Returns:
        Cleaned DataFrame with invalid rows removed. Returns the
        input unchanged if no primary column can be detected.

    Example:
        >>> clean = validate_aggregates(gefcom_daily, "gefcom_daily")
        Validation passed for 'gefcom_daily': all 1588 rows valid.
    """
    primary = next(
        (col for col in PRIMARY_COLUMN_CANDIDATES if col in df.columns),
        None,
    )
    if primary is None:
        logger.warning(
            "No primary column found for validation in '%s'. "
            "Available columns: %s",
            label, list(df.columns)[:5],
        )
        return df

    before = len(df)
    df = df[df[primary].notna() & (df[primary] > 0)].reset_index(drop=True)
    removed = before - len(df)

    if removed > 0:
        logger.warning(
            "Validation removed %d invalid rows from '%s' "
            "(zero or null %s).",
            removed, label, primary,
        )
    else:
        logger.info(
            "Validation passed for '%s': all %d rows valid.",
            label, before,
        )

    return df


def is_valid_summary(summary: str, min_words: int = 30) -> bool:
    """Check that a generated summary meets minimum quality criteria.

    Rejects summaries that are too short, empty, or contain AI refusal
    phrases. Called inside generate_summaries() before appending each
    Gemini result to the output CSV.

    Args:
        summary: Generated summary text from Gemini.
        min_words: Minimum acceptable word count. Summaries shorter
            than this are likely incomplete or refused. Default 30.

    Returns:
        True if the summary passes all quality checks, False otherwise.

    Example:
        >>> is_valid_summary("This is too short.")
        False
        >>> is_valid_summary("I cannot help with that request.")
        False
    """
    if not summary or len(summary.strip()) == 0:
        return False

    word_count = len(summary.split())
    if word_count < min_words:
        logger.warning(
            "Summary rejected: too short (%d words, minimum %d).",
            word_count, min_words,
        )
        return False

    summary_lower = summary.lower()
    for phrase in REFUSAL_PHRASES:
        if phrase in summary_lower:
            logger.warning(
                "Summary rejected: contains refusal phrase '%s'.",
                phrase,
            )
            return False

    return True
