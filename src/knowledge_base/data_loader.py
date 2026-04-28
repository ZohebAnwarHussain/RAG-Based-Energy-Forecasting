"""Raw dataset loading utilities for the Knowledge Base pipeline.

Two datasets are loaded:

    GEFCom2012 Load Forecasting
        Hourly electricity load data for 20 US utility zones from
        2004 to 2008. Numeric values use comma thousands separators
        in the raw file (e.g. "16,853") which are handled via
        thousands="," in pd.read_csv.

    UCI Household Power Consumption
        One-minute interval power measurements for a single French
        household from 2006 to 2010. Original file uses semicolons
        as delimiters and '?' as missing-value sentinel — both
        handled automatically by load_household_data().
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


def verify_data_paths(gefcom_path: Path, household_path: Path) -> None:
    """Verify that raw data directories exist before attempting to load.

    Provides a clear error message if either path is missing, rather
    than letting a cryptic FileNotFoundError surface deep inside the
    pipeline. Called once at the start of the KB pipeline.

    Args:
        gefcom_path: Filesystem path to the directory containing
            GEFCom CSV files (e.g. data/gefcom/).
        household_path: Filesystem path to the directory containing
            the UCI household CSV file (e.g. data/household/).

    Raises:
        AssertionError: If either directory does not exist. The error
            message includes the missing path and a hint to check the
            BASE_PATH setting in .env.
    """
    assert gefcom_path.exists(), (
        f"GEFCom data directory not found at {gefcom_path}. "
        "Check your BASE_PATH setting in .env."
    )
    assert household_path.exists(), (
        f"Household data directory not found at {household_path}. "
        "Check your BASE_PATH setting in .env."
    )
    logger.info("Data directories verified and accessible.")


def load_gefcom_data(raw_path: Path) -> Dict[str, pd.DataFrame]:
    """Load all GEFCom2012 CSV files from the given directory.

    Scans the directory for every CSV file and loads each one into a
    pandas DataFrame, storing them in a dictionary keyed by the file
    stem (filename without extension, lowercased).

    The ``thousands=","`` parameter is critical for GEFCom because
    load values are formatted with comma thousands separators
    (e.g. ``"16,853"`` for 16853 MW). Without this parameter, pandas
    would read these as strings rather than integers, causing the
    reshape step to silently produce zero valid records.

    Args:
        raw_path: Path to the directory containing GEFCom CSV files.

    Returns:
        Dictionary mapping lowercase file stem to DataFrame.
        Example: ``{"load_history": <DataFrame>, ...}``

    Note:
        Files that cannot be loaded are skipped with a warning
        rather than raising — pipeline continues with whichever
        files are readable.
    """
    frames: Dict[str, pd.DataFrame] = {}

    for csv_file in sorted(raw_path.glob("*.csv")):
        try:
            df = pd.read_csv(
                csv_file,
                low_memory=False,
                thousands=",",  # Handle "16,853" → 16853
            )
            frames[csv_file.stem.lower()] = df
            logger.info(
                "Loaded GEFCom file '%-30s' — %6d rows × %3d cols",
                csv_file.name, df.shape[0], df.shape[1],
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load '%s': %s", csv_file.name, exc)

    if not frames:
        logger.warning("No CSV files found in %s.", raw_path)

    return frames


def load_household_data(raw_path: Path) -> pd.DataFrame:
    """Load the UCI Individual Household Electric Power Consumption dataset.

    Handles two non-standard aspects of the original UCI file:

        1. Delimiter — original file uses semicolons (``;``) rather
           than commas. Auto-detected from the first line so the
           function works with both the original UCI format and
           comma-delimited exports.
        2. Missing values — original file uses ``'?'`` as the
           missing-value sentinel rather than empty cells. Replaced
           with NaN during loading via ``na_values=['?']``.

    After loading, the separate ``Date`` and ``Time`` columns are
    combined into a single timezone-naive ``datetime`` column,
    required for all subsequent time-based resampling operations.

    Args:
        raw_path: Path to the directory containing the household
            power consumption CSV file.

    Returns:
        DataFrame with a ``datetime`` column and seven numeric
        measurement columns:
        ``Global_active_power``, ``Global_reactive_power``,
        ``Voltage``, ``Global_intensity``, ``Sub_metering_1``,
        ``Sub_metering_2``, ``Sub_metering_3``.

    Raises:
        FileNotFoundError: If no CSV file is found in ``raw_path``.

    Example:
        >>> df = load_household_data(Path("data/household"))
        >>> "datetime" in df.columns
        True
    """
    csv_files = sorted(raw_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {raw_path}. "
            "Place household_power_consumption.csv in data/household/."
        )

    csv_path = csv_files[0]

    # Auto-detect delimiter from first line of the file
    with csv_path.open("r", encoding="utf-8") as file_handle:
        first_line = file_handle.readline()
    sep = ";" if ";" in first_line else ","
    logger.info("Detected delimiter: '%s' in '%s'", sep, csv_path.name)

    df = pd.read_csv(
        csv_path,
        sep=sep,
        na_values=["?"],  # Handle UCI missing-value sentinel
        low_memory=False,
    )
    logger.info(
        "Loaded household file '%-35s' — %7d rows × %2d cols",
        csv_path.name, df.shape[0], df.shape[1],
    )

    # Combine Date and Time columns into a single datetime column
    # UCI file stores dates as DD/MM/YYYY (dayfirst=True)
    date_col = next((c for c in df.columns if c.lower() == "date"), None)
    time_col = next((c for c in df.columns if c.lower() == "time"), None)

    if date_col and time_col:
        df["datetime"] = pd.to_datetime(
            df[date_col] + " " + df[time_col],
            dayfirst=True,
            errors="coerce",
        )
        df = df.drop(columns=[date_col, time_col])
        invalid_dt = df["datetime"].isna().sum()
        if invalid_dt > 0:
            logger.warning(
                "%d rows had unparseable datetime values and will be NaT.",
                invalid_dt,
            )
        logger.info("Combined 'Date' and 'Time' into 'datetime' column.")

    return df
