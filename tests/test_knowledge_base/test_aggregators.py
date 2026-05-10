"""Unit tests for src/knowledge_base/aggregators.py.

Tests the statistical aggregation functions using small synthetic
DataFrames. No real data files are read.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.knowledge_base.aggregators import (
    compute_gefcom_daily_stats,
    compute_gefcom_weekly_stats,
    compute_gefcom_monthly_stats,
    compute_gefcom_seasonal_stats,
    compute_gefcom_system_level,
    clean_household,
    aggregate_household,
    HOUSEHOLD_NUMERIC_COLS,
    SUBMETER_COLS,
)


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def gefcom_long():
    """Minimal long-format GEFCom DataFrame: 2 zones, 48 hourly records each."""
    rows = []
    for zone in [1, 2]:
        for day_offset in range(2):
            for hour in range(24):
                rows.append({
                    "zone_id": zone,
                    "datetime": pd.Timestamp("2005-01-01") + pd.Timedelta(days=day_offset, hours=hour),
                    "load_mw": 100.0 + zone * 10 + hour + day_offset * 5,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def household_raw():
    """Minimal household DataFrame: 48 records across 2 days."""
    rows = []
    for hour in range(48):
        rows.append({
            "datetime": pd.Timestamp("2007-01-01") + pd.Timedelta(hours=hour),
            "Global_active_power": 1.0 + hour * 0.01,
            "Global_reactive_power": 0.1,
            "Voltage": 230.0 + hour * 0.1,
            "Global_intensity": 4.0,
            "Sub_metering_1": 5.0,
            "Sub_metering_2": 8.0,
            "Sub_metering_3": 12.0,
        })
    return pd.DataFrame(rows)


# -- GEFCom daily stats --------------------------------------------------------

def test_daily_stats_has_expected_columns(gefcom_long):
    """Daily stats must include load_mean, load_min, load_max, load_std."""
    result = compute_gefcom_daily_stats(gefcom_long)
    for col in ["zone_id", "date", "load_mean", "load_min", "load_max", "load_std"]:
        assert col in result.columns, f"Missing column: {col}"


def test_daily_stats_row_count(gefcom_long):
    """Should produce 2 zones x 2 days = 4 rows."""
    result = compute_gefcom_daily_stats(gefcom_long)
    assert len(result) == 4


def test_daily_stats_load_mean_positive(gefcom_long):
    """All load_mean values must be positive."""
    result = compute_gefcom_daily_stats(gefcom_long)
    assert (result["load_mean"] > 0).all()


def test_daily_stats_has_year_month_dow(gefcom_long):
    """Daily stats must include derived year, month, dow columns."""
    result = compute_gefcom_daily_stats(gefcom_long)
    assert "year" in result.columns
    assert "month" in result.columns
    assert "dow" in result.columns


# -- GEFCom weekly stats -------------------------------------------------------

def test_weekly_stats_has_iso_columns(gefcom_long):
    """Weekly stats must include iso_year and iso_week."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    result = compute_gefcom_weekly_stats(daily)
    assert "iso_year" in result.columns
    assert "iso_week" in result.columns


def test_weekly_stats_has_weekly_mean(gefcom_long):
    """Weekly stats must include weekly_mean."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    result = compute_gefcom_weekly_stats(daily)
    assert "weekly_mean" in result.columns
    assert (result["weekly_mean"] > 0).all()


# -- GEFCom monthly stats -----------------------------------------------------

def test_monthly_stats_has_monthly_mean(gefcom_long):
    """Monthly stats must include monthly_mean."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    result = compute_gefcom_monthly_stats(daily)
    assert "monthly_mean" in result.columns


# -- GEFCom seasonal stats -----------------------------------------------------

def test_seasonal_stats_has_season_column(gefcom_long):
    """Seasonal stats must include a season column."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    result = compute_gefcom_seasonal_stats(daily)
    assert "season" in result.columns


def test_seasonal_stats_january_is_winter(gefcom_long):
    """January data should map to Winter season."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    result = compute_gefcom_seasonal_stats(daily)
    assert "Winter" in result["season"].values


# -- GEFCom system level -------------------------------------------------------

def test_system_level_returns_dict(gefcom_long):
    """System level must return a dict with daily/weekly/monthly keys."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    weekly = compute_gefcom_weekly_stats(daily)
    monthly = compute_gefcom_monthly_stats(daily)
    result = compute_gefcom_system_level(daily, weekly, monthly)
    assert isinstance(result, dict)
    assert set(result.keys()) == {"daily", "weekly", "monthly"}


def test_system_level_zone_is_system(gefcom_long):
    """System-level rows must have zone_id = 'system'."""
    daily = compute_gefcom_daily_stats(gefcom_long)
    weekly = compute_gefcom_weekly_stats(daily)
    monthly = compute_gefcom_monthly_stats(daily)
    result = compute_gefcom_system_level(daily, weekly, monthly)
    if not result["daily"].empty:
        assert (result["daily"]["zone_id"] == "system").all()


# -- Household clean -----------------------------------------------------------

def test_clean_household_removes_nan(household_raw):
    """clean_household removes rows with NaN Global_active_power."""
    df = household_raw.copy()
    df.loc[0, "Global_active_power"] = None
    result = clean_household(df)
    assert len(result) == len(household_raw) - 1


def test_clean_household_sorts_by_datetime(household_raw):
    """Result must be sorted by datetime ascending."""
    df = household_raw.sample(frac=1, random_state=0)  # shuffle
    result = clean_household(df)
    assert result["datetime"].is_monotonic_increasing


def test_clean_household_coerces_types(household_raw):
    """Numeric columns must be float64 after cleaning."""
    df = household_raw.copy()
    df["Global_active_power"] = df["Global_active_power"].astype(str)
    result = clean_household(df)
    assert result["Global_active_power"].dtype == np.float64


# -- Household aggregate -------------------------------------------------------

def test_aggregate_household_daily(household_raw):
    """Daily aggregation should produce 2 rows (2 days of data)."""
    cleaned = clean_household(household_raw)
    result = aggregate_household(cleaned, "D", "daily")
    assert len(result) == 2


def test_aggregate_household_has_mean_columns(household_raw):
    """Aggregated DataFrame must have _mean columns."""
    cleaned = clean_household(household_raw)
    result = aggregate_household(cleaned, "D", "daily")
    assert "Global_active_power_mean" in result.columns


def test_aggregate_household_has_period_start(household_raw):
    """Aggregated DataFrame must have period_start column."""
    cleaned = clean_household(household_raw)
    result = aggregate_household(cleaned, "D", "daily")
    assert "period_start" in result.columns
