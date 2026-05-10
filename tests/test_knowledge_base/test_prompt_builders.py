"""Unit tests for src/knowledge_base/prompt_builders.py.

Tests _safe_float helper and verifies prompt builders produce
DataFrames with the correct schema (PROMPT_CSV_COLUMNS).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest

from src.knowledge_base.prompt_builders import (
    _safe_float,
    build_gefcom_daily_prompts,
    build_household_yearly_prompts,
    PROMPT_CSV_COLUMNS,
)


# -- _safe_float ---------------------------------------------------------------

def test_safe_float_returns_value():
    """Returns the float value when column exists and is not null."""
    row = pd.Series({"load_std": 72.6})
    assert _safe_float(row, "load_std") == 72.6


def test_safe_float_returns_default_on_missing():
    """Returns default when column is absent."""
    row = pd.Series({"other": 1.0})
    assert _safe_float(row, "load_std") == 0.0


def test_safe_float_returns_default_on_nan():
    """Returns default when column value is NaN."""
    row = pd.Series({"load_std": float("nan")})
    assert _safe_float(row, "load_std") == 0.0


def test_safe_float_custom_default():
    """Returns custom default value when specified."""
    row = pd.Series({"other": 1.0})
    assert _safe_float(row, "missing_col", default=-1.0) == -1.0


# -- PROMPT_CSV_COLUMNS --------------------------------------------------------

def test_prompt_csv_columns_has_5():
    """Schema must define exactly 5 columns."""
    assert len(PROMPT_CSV_COLUMNS) == 5


def test_prompt_csv_columns_has_required():
    """Schema must include row_id, dataset, granularity, context_json, prompt_text."""
    required = {"row_id", "dataset", "granularity", "context_json", "prompt_text"}
    assert set(PROMPT_CSV_COLUMNS) == required


# -- build_gefcom_daily_prompts ------------------------------------------------

@pytest.fixture
def gefcom_daily_agg():
    """Minimal GEFCom daily aggregate DataFrame."""
    return pd.DataFrame({
        "zone_id": [1, 1, 2, 2],
        "date": pd.to_datetime(["2005-01-01", "2005-01-02", "2005-01-01", "2005-01-02"]),
        "load_mean": [100.0, 110.0, 200.0, 210.0],
        "load_min": [80.0, 90.0, 180.0, 190.0],
        "load_max": [120.0, 130.0, 220.0, 230.0],
        "load_std": [10.0, 11.0, 12.0, 13.0],
        "load_sum": [2400.0, 2640.0, 4800.0, 5040.0],
        "obs_count": [24, 24, 24, 24],
        "dow": ["Saturday", "Sunday", "Saturday", "Sunday"],
    })


def test_gefcom_daily_returns_dataframe(gefcom_daily_agg):
    """Builder must return a DataFrame."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert isinstance(result, pd.DataFrame)


def test_gefcom_daily_has_correct_columns(gefcom_daily_agg):
    """Output must have exactly PROMPT_CSV_COLUMNS."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert list(result.columns) == PROMPT_CSV_COLUMNS


def test_gefcom_daily_row_ids_unique(gefcom_daily_agg):
    """All row_id values must be unique."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert result["row_id"].is_unique


def test_gefcom_daily_dataset_is_gefcom(gefcom_daily_agg):
    """Dataset column must be 'gefcom' for all rows."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert (result["dataset"] == "gefcom").all()


def test_gefcom_daily_granularity_is_daily(gefcom_daily_agg):
    """Granularity column must be 'daily' for all rows."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert (result["granularity"] == "daily").all()


def test_gefcom_daily_prompt_text_not_empty(gefcom_daily_agg):
    """Prompt text must not be empty."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=4)
    assert (result["prompt_text"].str.len() > 0).all()


def test_gefcom_daily_respects_limit(gefcom_daily_agg):
    """Output must not exceed the specified limit."""
    result = build_gefcom_daily_prompts(gefcom_daily_agg, limit=2)
    assert len(result) <= 2


# -- build_household_yearly_prompts -------------------------------------------

@pytest.fixture
def household_yearly_agg():
    """Minimal household yearly aggregate DataFrame."""
    return pd.DataFrame({
        "year": [2007, 2008, 2009],
        "yearly_mean": [1.2, 1.3, 1.1],
        "yearly_min": [0.5, 0.6, 0.4],
        "yearly_max": [2.0, 2.1, 1.9],
        "yearly_std": [0.3, 0.35, 0.28],
        "peak_season": ["Winter", "Winter", "Autumn"],
        "Sub_metering_1_mean": [5.0, 5.5, 4.8],
        "Sub_metering_2_mean": [8.0, 8.5, 7.8],
        "Sub_metering_3_mean": [12.0, 12.5, 11.8],
    })


def test_household_yearly_returns_dataframe(household_yearly_agg):
    """Builder must return a DataFrame."""
    result = build_household_yearly_prompts(household_yearly_agg)
    assert isinstance(result, pd.DataFrame)


def test_household_yearly_has_correct_columns(household_yearly_agg):
    """Output must have exactly PROMPT_CSV_COLUMNS."""
    result = build_household_yearly_prompts(household_yearly_agg)
    assert list(result.columns) == PROMPT_CSV_COLUMNS


def test_household_yearly_dataset_is_household(household_yearly_agg):
    """Dataset column must be 'household'."""
    result = build_household_yearly_prompts(household_yearly_agg)
    assert (result["dataset"] == "household").all()


def test_household_yearly_granularity_is_yearly(household_yearly_agg):
    """Granularity column must be 'yearly'."""
    result = build_household_yearly_prompts(household_yearly_agg)
    assert (result["granularity"] == "yearly").all()


def test_household_yearly_row_id_contains_year(household_yearly_agg):
    """Row IDs must contain the year value."""
    result = build_household_yearly_prompts(household_yearly_agg)
    assert all("2007" in rid or "2008" in rid or "2009" in rid for rid in result["row_id"])
