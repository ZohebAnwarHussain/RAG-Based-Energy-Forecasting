"""Statistical aggregation functions for both datasets.

This module contains all functions that transform raw time-series data
into summary statistics suitable for prompt building. Two dataset
sections are clearly separated:

    GEFCom2012 (5 functions):
        reshape_gefcom_load          Wide → long format conversion
        compute_gefcom_daily_stats   Hourly → daily per zone
        compute_gefcom_weekly_stats  Daily → ISO-week per zone
        compute_gefcom_monthly_stats Daily → calendar month per zone
        compute_gefcom_seasonal_stats Daily → meteorological season per zone
        compute_gefcom_system_level  Synthetic Zone 21 (sum of all 20 zones)

    UCI Household (4 functions):
        clean_household              Remove invalid rows, sort by datetime
        aggregate_household          Resample to D/W/ME frequency
        compute_household_appliance  Sub-metering shares by granularity
        compute_household_yearly     Annual aggregates + peak season
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from config import SEASON_MAP

logger = logging.getLogger(__name__)


# Numeric columns in the UCI household dataset
HOUSEHOLD_NUMERIC_COLS: List[str] = [
    "Global_active_power",    # Total active power (kW)
    "Global_reactive_power",  # Reactive power (kVAR)
    "Voltage",                # Mains voltage (V) — typically ~230V
    "Global_intensity",       # Current intensity (A)
    "Sub_metering_1",         # Kitchen: dishwasher, oven, microwave (Wh)
    "Sub_metering_2",         # Laundry: washer, dryer, fridge (Wh)
    "Sub_metering_3",         # Electric water heater + AC (Wh)
]

# Sub-metering columns used for appliance-level analysis
SUBMETER_COLS: List[str] = [
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


# ─────────────────────────────────────────────────────────────────────────────
# GEFCom Aggregation Functions
# ─────────────────────────────────────────────────────────────────────────────

def reshape_gefcom_load(load_df: pd.DataFrame) -> pd.DataFrame:
    """Convert GEFCom load_history from wide to long format.

    The raw GEFCom file stores 24 hourly load values as separate columns
    (h1 through h24) on each row. This function melts those columns into
    rows, creating one record per zone-hour with a proper datetime.

    Wide format (one row per zone-day)::

        zone_id | year | month | day | h1  | h2  | ... | h24
        4       | 2004 | 1     | 1   | 444 | 420 | ... | 512

    Long format (one row per zone-hour)::

        zone_id | datetime            | load_mw
        4       | 2004-01-01 00:00:00 | 444.0
        4       | 2004-01-01 01:00:00 | 420.0

    Args:
        load_df: Raw load_history DataFrame in wide format.

    Returns:
        Long-format DataFrame with columns
        ``['zone_id', 'datetime', 'load_mw']``. Rows with non-numeric
        or missing load values are dropped.
    """
    logger.info("Reshaping GEFCom load history: %d rows input.", len(load_df))

    # Normalise column names: lowercase + strip whitespace
    load_df = load_df.copy()
    load_df.columns = [c.lower().strip() for c in load_df.columns]

    # Identify the 24 hourly load columns (h1, h2, ..., h24)
    hour_cols = [
        c for c in load_df.columns
        if c.startswith("h") and c[1:].isdigit()
    ]
    id_vars = [c for c in load_df.columns if c not in hour_cols]

    logger.info(
        "Found %d hour columns and %d id columns.",
        len(hour_cols), len(id_vars),
    )

    # Melt wide → long: each h1-h24 column becomes a separate row
    long_df = load_df.melt(
        id_vars=id_vars,
        value_vars=hour_cols,
        var_name="hour_label",
        value_name="load_mw",
    )

    # Extract hour as integer: h1 → hour 0 (midnight), h24 → hour 23
    long_df["hour"] = (
        long_df["hour_label"].str.extract(r"(\d+)").astype(int) - 1
    )

    # Build a proper datetime from year/month/day + hour offset
    try:
        long_df["datetime"] = (
            pd.to_datetime(long_df[["year", "month", "day"]])
            + pd.to_timedelta(long_df["hour"], unit="h")
        )
    except KeyError:
        logger.warning(
            "Could not find year/month/day columns — datetime set to NaT."
        )
        long_df["datetime"] = pd.NaT

    # Identify the zone column (first column containing 'zone')
    zone_col = next((c for c in id_vars if "zone" in c), id_vars[0])

    result = (
        long_df[[zone_col, "datetime", "load_mw"]]
        .rename(columns={zone_col: "zone_id"})
    )
    result["load_mw"] = pd.to_numeric(result["load_mw"], errors="coerce")
    result = result.dropna(subset=["load_mw"]).reset_index(drop=True)

    logger.info(
        "Reshape complete: %d hourly records across all zones.", len(result)
    )
    return result


def compute_gefcom_daily_stats(long_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly GEFCom load to daily statistics per zone.

    Args:
        long_df: Long-format DataFrame from reshape_gefcom_load() with
            columns ``['zone_id', 'datetime', 'load_mw']``.

    Returns:
        DataFrame with one row per zone-day and columns:
        ``zone_id``, ``date``, ``load_mean``, ``load_min``, ``load_max``,
        ``load_std``, ``load_sum``, ``obs_count``, ``year``, ``month``,
        ``dow``.
    """
    logger.info("Computing daily statistics for all GEFCom zones.")
    df = long_df.copy()
    df["date"] = df["datetime"].dt.date

    daily = (
        df.groupby(["zone_id", "date"])["load_mw"]
        .agg(
            load_mean="mean",
            load_min="min",
            load_max="max",
            load_std="std",
            load_sum="sum",
            obs_count="count",
        )
        .reset_index()
    )

    daily["date"]  = pd.to_datetime(daily["date"])
    daily["year"]  = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["dow"]   = daily["date"].dt.day_name()

    logger.info("Daily stats complete: %d zone-day records.", len(daily))
    return daily


def compute_gefcom_weekly_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GEFCom daily statistics to ISO-week level per zone.

    Args:
        daily_df: Daily DataFrame from compute_gefcom_daily_stats().

    Returns:
        DataFrame with one row per zone-week and columns:
        ``zone_id``, ``iso_year``, ``iso_week``, ``weekly_mean``,
        ``weekly_min``, ``weekly_max``, ``weekly_std``.
    """
    logger.info("Computing weekly statistics for all GEFCom zones.")
    df = daily_df.copy()
    df["iso_year"] = df["date"].dt.isocalendar().year.astype(int)
    df["iso_week"] = df["date"].dt.isocalendar().week.astype(int)

    weekly = (
        df.groupby(["zone_id", "iso_year", "iso_week"])["load_mean"]
        .agg(
            weekly_mean="mean",
            weekly_min="min",
            weekly_max="max",
            weekly_std="std",
        )
        .reset_index()
    )
    logger.info("Weekly stats complete: %d zone-week records.", len(weekly))
    return weekly


def compute_gefcom_monthly_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GEFCom daily statistics to calendar-month level per zone.

    Args:
        daily_df: Daily DataFrame from compute_gefcom_daily_stats().

    Returns:
        DataFrame with one row per zone-month and columns:
        ``zone_id``, ``year``, ``month``, ``monthly_mean``,
        ``monthly_min``, ``monthly_max``, ``monthly_std``.
    """
    logger.info("Computing monthly statistics for all GEFCom zones.")
    monthly = (
        daily_df.groupby(["zone_id", "year", "month"])["load_mean"]
        .agg(
            monthly_mean="mean",
            monthly_min="min",
            monthly_max="max",
            monthly_std="std",
        )
        .reset_index()
    )
    logger.info(
        "Monthly stats complete: %d zone-month records.", len(monthly)
    )
    return monthly


def compute_gefcom_seasonal_stats(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate GEFCom daily statistics to meteorological season per zone.

    Season definitions (meteorological)::
        Winter : December, January, February
        Spring : March, April, May
        Summer : June, July, August
        Autumn : September, October, November

    Args:
        daily_df: Daily DataFrame from compute_gefcom_daily_stats().

    Returns:
        DataFrame with one row per zone-season-year and columns:
        ``zone_id``, ``year``, ``season``, ``seasonal_mean``,
        ``seasonal_min``, ``seasonal_max``, ``seasonal_std``,
        ``day_count``.
    """
    logger.info("Computing seasonal statistics for all GEFCom zones.")
    df = daily_df.copy()
    df["season"] = df["date"].dt.month.map(SEASON_MAP)

    seasonal = (
        df.groupby(["zone_id", "year", "season"])["load_mean"]
        .agg(
            seasonal_mean="mean",
            seasonal_min="min",
            seasonal_max="max",
            seasonal_std="std",
            day_count="count",
        )
        .reset_index()
    )
    logger.info(
        "Seasonal stats complete: %d zone-season-year records.",
        len(seasonal),
    )
    return seasonal


def compute_gefcom_system_level(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Compute synthetic system-level aggregates by summing all zones.

    The Kaggle GEFCom2012 download does not include Zone 21, which in
    the original competition represented the system-level sum of all
    20 zones. This function reconstructs the equivalent system-level
    view by summing load statistics across all available zones.

    Aggregation rules:
        - load_mean, load_sum, load_min, load_max → summed
        - load_std → averaged (mean of zone-level stds approximates
          system-level variability)

    Args:
        daily_df:   GEFCom daily aggregates (all 20 zones).
        weekly_df:  GEFCom weekly aggregates (all 20 zones).
        monthly_df: GEFCom monthly aggregates (all 20 zones).

    Returns:
        Dict with keys ``'daily'``, ``'weekly'``, ``'monthly'`` mapping
        to system-level DataFrames. Each has ``zone_id = "system"``.
    """
    logger.info(
        "Computing synthetic system-level aggregates "
        "(sum across all 20 zones — equivalent to Zone 21)."
    )
    result: Dict[str, pd.DataFrame] = {}

    # ── Daily system-level ───────────────────────────────────────────────────
    if not daily_df.empty:
        sys_daily = (
            daily_df.groupby("date")
            .agg(
                load_mean=("load_mean", "sum"),
                load_min=("load_min",   "sum"),
                load_max=("load_max",   "sum"),
                load_std=("load_std",   "mean"),
                load_sum=("load_sum",   "sum"),
                obs_count=("obs_count", "mean"),
            )
            .reset_index()
        )
        sys_daily["zone_id"] = "system"
        sys_daily["date"]    = pd.to_datetime(sys_daily["date"])
        sys_daily["year"]    = sys_daily["date"].dt.year
        sys_daily["month"]   = sys_daily["date"].dt.month
        sys_daily["dow"]     = sys_daily["date"].dt.day_name()
        result["daily"] = sys_daily
        logger.info("System-level daily: %d records.", len(sys_daily))
    else:
        result["daily"] = pd.DataFrame()

    # ── Weekly system-level ──────────────────────────────────────────────────
    if not weekly_df.empty:
        sys_weekly = (
            weekly_df.groupby(["iso_year", "iso_week"])
            .agg(
                weekly_mean=("weekly_mean", "sum"),
                weekly_min=("weekly_min",   "sum"),
                weekly_max=("weekly_max",   "sum"),
                weekly_std=("weekly_std",   "mean"),
            )
            .reset_index()
        )
        sys_weekly["zone_id"] = "system"
        result["weekly"] = sys_weekly
        logger.info("System-level weekly: %d records.", len(sys_weekly))
    else:
        result["weekly"] = pd.DataFrame()

    # ── Monthly system-level ─────────────────────────────────────────────────
    if not monthly_df.empty:
        sys_monthly = (
            monthly_df.groupby(["year", "month"])
            .agg(
                monthly_mean=("monthly_mean", "sum"),
                monthly_min=("monthly_min",   "sum"),
                monthly_max=("monthly_max",   "sum"),
                monthly_std=("monthly_std",   "mean"),
            )
            .reset_index()
        )
        sys_monthly["zone_id"] = "system"
        result["monthly"] = sys_monthly
        logger.info("System-level monthly: %d records.", len(sys_monthly))
    else:
        result["monthly"] = pd.DataFrame()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Household Aggregation Functions
# ─────────────────────────────────────────────────────────────────────────────

def clean_household(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw household power consumption DataFrame.

    Three operations:
        1. Type coercion — convert all measurement columns to float64,
           replacing any non-numeric values with NaN.
        2. Row removal — drop rows where Global_active_power is NaN.
        3. Sorting — sort by datetime ascending (required for resample).

    Args:
        df: Raw household DataFrame from load_household_data() with
            a ``datetime`` column.

    Returns:
        Cleaned DataFrame sorted by datetime, with no NaN values in
        the primary measurement column.
    """
    logger.info("Cleaning household data: %d rows before.", len(df))
    df = df.copy()

    for col in HOUSEHOLD_NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    primary_col = "Global_active_power"
    if primary_col in df.columns:
        before  = len(df)
        df      = df.dropna(subset=[primary_col])
        dropped = before - len(df)
        if dropped > 0:
            logger.info(
                "Removed %d rows with missing %s values.",
                dropped, primary_col,
            )

    df = df.sort_values("datetime").reset_index(drop=True)
    logger.info("Household cleaning complete: %d rows after.", len(df))
    return df


def aggregate_household(
    df: pd.DataFrame,
    freq: str,
    label: str,
) -> pd.DataFrame:
    """Aggregate household minute-level data to a specified time frequency.

    Computes mean/min/max/std for all seven numeric measurement columns
    at the requested frequency.

    Args:
        df: Cleaned household DataFrame from clean_household().
        freq: Pandas offset alias defining the aggregation frequency.
            Common values:
                - ``'D'`` daily
                - ``'W'`` weekly (period ends Sunday)
                - ``'ME'`` month-end
        label: Human-readable frequency name for log messages.

    Returns:
        Aggregated DataFrame with ``period_start`` timestamp column and
        one column per statistic per measurement (e.g.
        ``Global_active_power_mean``, ``Global_active_power_std``).
    """
    logger.info(
        "Aggregating household data at '%s' (%s) frequency.", freq, label
    )
    df = df.copy().set_index("datetime")

    numeric_cols = [c for c in HOUSEHOLD_NUMERIC_COLS if c in df.columns]

    agg_dict: Dict[str, Any] = {}
    for col in numeric_cols:
        for stat in ["mean", "min", "max", "std"]:
            agg_dict[f"{col}_{stat}"] = (col, stat)

    result = (
        df[numeric_cols]
        .resample(freq)
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"datetime": "period_start"})
    )

    primary_mean = f"{numeric_cols[0]}_mean"
    result = result.dropna(subset=[primary_mean]).reset_index(drop=True)

    logger.info(
        "%s aggregation complete: %d records.",
        label.capitalize(), len(result),
    )
    return result


def compute_household_appliance(
    daily_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Build appliance-focused DataFrames with sub-meter percentage shares.

    Extracts sub-metering columns from the daily, weekly, and monthly
    aggregate DataFrames and computes the percentage share of each
    sub-meter relative to total sub-metering consumption::

        sm1_share_pct = Sub_metering_1_mean / total_submetering_mean * 100

    Args:
        daily_df: Household daily aggregates from aggregate_household().
        weekly_df: Household weekly aggregates.
        monthly_df: Household monthly aggregates.

    Returns:
        Dict with keys ``'daily'``, ``'weekly'``, ``'monthly'`` mapping
        to appliance DataFrames including share percentages.
    """
    logger.info("Computing household appliance sub-metering aggregates.")
    result: Dict[str, pd.DataFrame] = {}

    for label, df in [
        ("daily",   daily_df),
        ("weekly",  weekly_df),
        ("monthly", monthly_df),
    ]:
        if df.empty:
            result[label] = pd.DataFrame()
            continue

        sm_mean_cols = [
            f"{c}_mean" for c in SUBMETER_COLS
            if f"{c}_mean" in df.columns
        ]
        sm_all_cols = [
            f"{c}_{stat}"
            for c in SUBMETER_COLS
            for stat in ["mean", "min", "max", "std"]
            if f"{c}_{stat}" in df.columns
        ]

        keep_cols = ["period_start"] + sm_all_cols
        if "Global_active_power_mean" in df.columns:
            keep_cols.append("Global_active_power_mean")

        appliance_df = df[keep_cols].copy()

        if sm_mean_cols:
            appliance_df["total_submetering_mean"] = (
                appliance_df[sm_mean_cols].sum(axis=1)
            )
            for col in sm_mean_cols:
                share_col = (
                    col.replace("_mean", "")
                       .replace("Sub_metering_", "sm")
                    + "_share_pct"
                )
                appliance_df[share_col] = (
                    appliance_df[col]
                    .div(
                        appliance_df["total_submetering_mean"]
                        .replace(0, np.nan)
                    )
                    .mul(100)
                    .round(1)
                )

        result[label] = appliance_df
        logger.info(
            "Appliance %s DataFrame: %d rows, %d columns.",
            label, len(appliance_df), len(appliance_df.columns),
        )

    return result


def compute_household_yearly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate household monthly statistics to calendar-year level.

    Adds a peak_season column identifying the season with the highest
    average monthly power for each year, plus annual mean sub-metering
    values for appliance context.

    Args:
        monthly_df: Household monthly aggregates from
            aggregate_household() with freq='ME'.

    Returns:
        DataFrame with one row per calendar year and columns:
        ``year``, ``yearly_mean``, ``yearly_min``, ``yearly_max``,
        ``yearly_std``, ``peak_season``,
        ``Sub_metering_1_mean``, ``Sub_metering_2_mean``,
        ``Sub_metering_3_mean``.
    """
    if monthly_df.empty:
        logger.warning(
            "Monthly DataFrame is empty — cannot compute yearly aggregates."
        )
        return pd.DataFrame()

    logger.info("Computing household yearly aggregates.")
    df = monthly_df.copy()
    df["year"]   = pd.to_datetime(df["period_start"]).dt.year
    df["month"]  = pd.to_datetime(df["period_start"]).dt.month
    df["season"] = df["month"].map(SEASON_MAP)

    gap_col = "Global_active_power_mean"
    if gap_col not in df.columns:
        logger.warning(
            "'%s' not found — cannot compute yearly aggregates.", gap_col
        )
        return pd.DataFrame()

    yearly = (
        df.groupby("year")[gap_col]
        .agg(
            yearly_mean="mean",
            yearly_min="min",
            yearly_max="max",
            yearly_std="std",
        )
        .reset_index()
    )

    # Identify peak season per year
    seasonal_means = (
        df.groupby(["year", "season"])[gap_col]
        .mean()
        .reset_index()
    )
    peak_season = (
        seasonal_means
        .loc[seasonal_means.groupby("year")[gap_col].idxmax()]
        [["year", "season"]]
        .rename(columns={"season": "peak_season"})
        .reset_index(drop=True)
    )
    yearly = yearly.merge(peak_season, on="year", how="left")

    # Annual mean sub-metering values
    sm_mean_cols = [
        f"{c}_mean" for c in SUBMETER_COLS
        if f"{c}_mean" in df.columns
    ]
    if sm_mean_cols:
        sm_yearly = df.groupby("year")[sm_mean_cols].mean().reset_index()
        yearly    = yearly.merge(sm_yearly, on="year", how="left")

    logger.info("Yearly aggregates complete: %d rows.", len(yearly))
    return yearly
