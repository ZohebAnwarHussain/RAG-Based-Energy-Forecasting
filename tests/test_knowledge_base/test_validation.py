"""Unit tests for src/knowledge_base/validation.py.

Verifies validate_aggregates() removes invalid rows and
is_valid_summary() rejects short/refusal summaries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.knowledge_base.validation import (
    validate_aggregates,
    is_valid_summary,
    PRIMARY_COLUMN_CANDIDATES,
    REFUSAL_PHRASES,
)


# -- validate_aggregates -------------------------------------------------------

def test_validate_keeps_valid_rows():
    """Rows with positive load_mean are retained."""
    df = pd.DataFrame({"load_mean": [100.0, 200.0, 300.0]})
    result = validate_aggregates(df, "test")
    assert len(result) == 3


def test_validate_removes_zero_rows():
    """Rows with load_mean == 0 are removed."""
    df = pd.DataFrame({"load_mean": [100.0, 0.0, 300.0]})
    result = validate_aggregates(df, "test")
    assert len(result) == 2
    assert 0.0 not in result["load_mean"].values


def test_validate_removes_null_rows():
    """Rows with null primary column are removed."""
    df = pd.DataFrame({"load_mean": [100.0, None, 300.0]})
    result = validate_aggregates(df, "test")
    assert len(result) == 2


def test_validate_detects_weekly_mean():
    """Uses weekly_mean when load_mean is absent."""
    df = pd.DataFrame({"weekly_mean": [50.0, 0.0, 75.0]})
    result = validate_aggregates(df, "test")
    assert len(result) == 2


def test_validate_detects_household_column():
    """Uses Global_active_power_mean for household data."""
    df = pd.DataFrame({"Global_active_power_mean": [1.5, 0.0, 2.3]})
    result = validate_aggregates(df, "test")
    assert len(result) == 2


def test_validate_returns_unchanged_no_primary():
    """Returns input unchanged when no primary column is found."""
    df = pd.DataFrame({"other_col": [1, 2, 3]})
    result = validate_aggregates(df, "test")
    assert len(result) == 3


def test_validate_resets_index():
    """Returned DataFrame has a clean reset index."""
    df = pd.DataFrame({"load_mean": [100.0, 0.0, 300.0]})
    result = validate_aggregates(df, "test")
    assert list(result.index) == [0, 1]


# -- is_valid_summary -----------------------------------------------------------

def test_valid_summary_passes():
    """A 30+ word summary without refusal phrases passes."""
    summary = " ".join(["word"] * 35)
    assert is_valid_summary(summary) is True


def test_short_summary_fails():
    """Summary under min_words is rejected."""
    assert is_valid_summary("Too short.") is False


def test_empty_summary_fails():
    """Empty string is rejected."""
    assert is_valid_summary("") is False


def test_whitespace_only_fails():
    """Whitespace-only string is rejected."""
    assert is_valid_summary("   ") is False


def test_refusal_phrase_fails():
    """Summary containing an AI refusal phrase is rejected."""
    summary = "I cannot provide this information. " + " ".join(["word"] * 30)
    assert is_valid_summary(summary) is False


def test_refusal_case_insensitive():
    """Refusal detection is case-insensitive."""
    summary = "As An AI language model I " + " ".join(["word"] * 30)
    assert is_valid_summary(summary) is False


def test_custom_min_words():
    """Custom min_words threshold is respected."""
    summary = " ".join(["word"] * 10)
    assert is_valid_summary(summary, min_words=5) is True
    assert is_valid_summary(summary, min_words=15) is False


def test_primary_column_candidates_not_empty():
    """PRIMARY_COLUMN_CANDIDATES must have at least one entry."""
    assert len(PRIMARY_COLUMN_CANDIDATES) >= 1


def test_refusal_phrases_not_empty():
    """REFUSAL_PHRASES must have at least one entry."""
    assert len(REFUSAL_PHRASES) >= 1
