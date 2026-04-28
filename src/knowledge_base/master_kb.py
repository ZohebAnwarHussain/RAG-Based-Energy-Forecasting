"""Master Knowledge Base CSV builder.

Merges all 10 individual summary DataFrames into the single master KB
CSV consumed by all downstream pipeline stages (embedding, retrieval,
RAG generation, evaluation).

In addition to merging, this module:

    1. Deduplicates by row_id
    2. Extracts metadata columns (zone_id, year, month, season) from
       the context_json field for efficient filtering during retrieval
    3. Builds parent_id links for hierarchical retrieval — daily rows
       link to their weekly parent, weekly rows to monthly parent
    4. Assigns a sequential kb_id integer for stable referencing
    5. Enforces a fixed column order for downstream code stability
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List

import pandas as pd

logger = logging.getLogger(__name__)


# Final column order for the master KB CSV
MASTER_KB_COLUMNS: List[str] = [
    "kb_id",
    "row_id",
    "dataset",
    "granularity",
    "zone_id",
    "year",
    "month",
    "season",
    "parent_id",
    "summary",
    "context_json",
    "generated_at",
]


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def _safe_int(val: Any) -> int:
    """Convert a value to int safely, handling float strings.

    Pandas often stores numeric IDs as float strings like '17.0' when
    they were read from CSV with mixed-type columns. Direct int() on
    '17.0' raises ValueError. This helper goes via float() first.

    Args:
        val: Value to convert. Can be int, float, str, or numeric string.

    Returns:
        Integer representation of the input value.

    Example:
        >>> _safe_int("17.0")
        17
        >>> _safe_int(17.5)
        17
    """
    return int(float(str(val)))


def _extract_context_field(
    context_json_str: str,
    field: str,
    default: str = "",
) -> str:
    """Extract a single field from a context_json string.

    Parses the JSON string stored in the context_json column and
    returns the value of the requested field as a string. Returns
    ``default`` if the field is absent or the JSON cannot be parsed.

    Args:
        context_json_str: JSON string from the context_json column.
        field: Key name to extract from the parsed JSON object.
        default: Value to return when the field is missing or parse fails.

    Returns:
        String value of the requested field, or ``default``.

    Example:
        >>> _extract_context_field('{"zone_id": "4"}', "zone_id")
        '4'
    """
    try:
        return str(json.loads(context_json_str).get(field, default))
    except (json.JSONDecodeError, TypeError):
        return default


def _build_parent_id(row: pd.Series) -> str:
    """Construct the parent_id for a KB row based on its granularity.

    Implements the parent-child linking required by the hierarchical
    retrieval pipeline (Pipeline 3):

        - daily rows  → link to their ISO-week weekly parent
        - weekly rows → link to their calendar-month monthly parent
        - All other granularities (monthly, seasonal, system_level,
          appliance, yearly) have no parent — return empty string

    Args:
        row: One row from the master KB DataFrame, including columns
            granularity, dataset, context_json, and row_id.

    Returns:
        Parent row_id string if a parent exists, else ''.
    """
    granularity = row.get("granularity", "")
    dataset     = row.get("dataset", "")

    try:
        ctx = json.loads(row.get("context_json", "{}"))
    except (json.JSONDecodeError, TypeError):
        return ""

    # ── GEFCom daily → weekly parent ─────────────────────────────────────────
    if granularity == "daily" and dataset == "gefcom":
        zone_id = ctx.get("zone_id", "")
        try:
            date  = pd.to_datetime(ctx.get("date", ""))
            iso_w = date.isocalendar().week
            iso_y = date.isocalendar().year
            return f"gefcom_weekly_{zone_id}_W{iso_w}_{iso_y}"
        except Exception:  # noqa: BLE001
            return ""

    # ── GEFCom weekly → monthly parent ───────────────────────────────────────
    if granularity == "weekly" and dataset == "gefcom":
        zone_id  = ctx.get("zone_id", "")
        iso_year = ctx.get("iso_year", "")
        iso_week = ctx.get("iso_week", 1)
        try:
            # Use Thursday of the ISO week (day 4) as a stable midpoint
            # for mapping ISO week → calendar month
            approx_date = pd.Timestamp.fromisocalendar(
                _safe_int(iso_year),
                _safe_int(iso_week),
                4,
            )
            month_name = approx_date.strftime("%B")
            return (
                f"gefcom_monthly_{zone_id}_"
                f"{month_name}_{_safe_int(iso_year)}"
            )
        except Exception:  # noqa: BLE001
            return ""

    # ── Household daily → weekly parent ──────────────────────────────────────
    if granularity == "daily" and dataset == "household":
        date_str = ctx.get("period_start", "")[:10]
        try:
            date     = pd.to_datetime(date_str)
            # Weekly periods end on Sunday in the household resampling
            week_end = date + pd.offsets.Week(weekday=6)
            return f"household_weekly_{week_end.strftime('%Y-%m-%d')}"
        except Exception:  # noqa: BLE001
            return ""

    # ── Household weekly → monthly parent ────────────────────────────────────
    if granularity == "weekly" and dataset == "household":
        date_str = ctx.get("period_start", "")[:10]
        try:
            date      = pd.to_datetime(date_str)
            # Monthly periods use month-end timestamps (freq='ME')
            month_end = (
                date + pd.offsets.MonthEnd(0)
            ).strftime("%Y-%m-%d")
            return f"household_monthly_{month_end}"
        except Exception:  # noqa: BLE001
            return ""

    # No parent for monthly, seasonal, system_level, appliance, yearly
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Master KB Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_master_knowledge_base(
    summary_dfs: List[pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """Merge all summary DataFrames into the master knowledge base CSV.

    Performs five steps:
        1. Concatenate all 10 summary DataFrames (skipping any empty)
        2. Deduplicate by row_id
        3. Extract metadata columns (zone_id, year, month, season)
           from context_json for filtered retrieval
        4. Build parent_id links via _build_parent_id() for the
           hierarchical retrieval pipeline
        5. Assign sequential kb_id integers and enforce column order

    The resulting CSV is the single file consumed by all downstream
    stages — embedding, indexing, retrieval, RAG generation, evaluation.

    Args:
        summary_dfs: List of summary DataFrames, one per summary type.
            Empty DataFrames are silently skipped.
        output_path: Destination CSV path for the combined master file
            (typically combined_master_summaries.csv).

    Returns:
        Master KB DataFrame with MASTER_KB_COLUMNS schema, written
        to output_path.

    Example:
        >>> master = build_master_knowledge_base(
        ...     [gefcom_daily_summaries, household_daily_summaries, ...],
        ...     PATHS["summaries_csv"] / "combined_master_summaries.csv",
        ... )
        >>> len(master)
        480
    """
    valid = [df for df in summary_dfs if df is not None and not df.empty]
    if not valid:
        logger.warning("No summary DataFrames provided — master KB is empty.")
        return pd.DataFrame()

    # Step 1: Concatenate
    master = pd.concat(valid, ignore_index=True)
    before = len(master)

    # Step 2: Deduplicate by row_id
    master = master.drop_duplicates(subset=["row_id"]).reset_index(drop=True)
    logger.info(
        "Merged %d rows, removed %d duplicates → %d unique entries.",
        before, before - len(master), len(master),
    )

    # Step 3: Extract metadata columns from context_json
    logger.info("Extracting metadata columns from context_json...")
    master["zone_id"] = master["context_json"].apply(
        lambda x: _extract_context_field(x, "zone_id")
    )
    master["year"] = master["context_json"].apply(
        lambda x: _extract_context_field(x, "year")
    )
    master["month"] = master["context_json"].apply(
        lambda x: _extract_context_field(x, "month")
    )
    master["season"] = master["context_json"].apply(
        lambda x: _extract_context_field(x, "season")
    )

    # Step 4: Build parent_id links for hierarchical retrieval
    logger.info("Building parent_id links for hierarchical retrieval...")
    master["parent_id"] = master.apply(_build_parent_id, axis=1)
    linked = (master["parent_id"] != "").sum()
    logger.info("parent_id assigned: %d rows linked to a parent.", linked)

    # Step 5: Sequential kb_id and enforced column order
    master.insert(0, "kb_id", range(1, len(master) + 1))
    master = master[[c for c in MASTER_KB_COLUMNS if c in master.columns]]
    master.to_csv(output_path, index=False)

    logger.info(
        "Master KB saved: %d entries → %s", len(master), output_path
    )
    return master
