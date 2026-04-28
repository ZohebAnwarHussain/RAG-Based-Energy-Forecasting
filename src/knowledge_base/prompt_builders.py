"""Prompt-input row builders for all 10 KB summary types.

Each builder takes a statistical aggregate DataFrame, applies stratified
sampling to limit the pilot size, and produces a DataFrame of prompt-input
rows ready to be sent to Gemini for summary generation.

Output schema for all builders is defined by ``PROMPT_CSV_COLUMNS``:

    row_id        Unique identifier for the prompt row
    dataset       'gefcom' or 'household'
    granularity   'daily', 'weekly', 'monthly', 'seasonal', etc.
    context_json  Raw statistics as JSON (stored for traceability)
    prompt_text   Full prompt text sent to Gemini

The 10 builders are split into two groups (5 GEFCom + 5 household)
mirroring the prompt template organisation in prompt_templates.py.
"""

from __future__ import annotations

import calendar
import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from src.knowledge_base.prompt_templates import (
    GEFCOM_DAILY_TEMPLATE,
    GEFCOM_MONTHLY_TEMPLATE,
    GEFCOM_SEASONAL_TEMPLATE,
    GEFCOM_SYSTEM_LEVEL_TEMPLATE,
    GEFCOM_WEEKLY_TEMPLATE,
    HOUSEHOLD_APPLIANCE_TEMPLATE,
    HOUSEHOLD_DAILY_TEMPLATE,
    HOUSEHOLD_MONTHLY_TEMPLATE,
    HOUSEHOLD_WEEKLY_TEMPLATE,
    HOUSEHOLD_YEARLY_TEMPLATE,
)
from src.knowledge_base.sampling import stratified_sample

logger = logging.getLogger(__name__)


# Standard column schema for all prompt-input CSVs
PROMPT_CSV_COLUMNS: List[str] = [
    "row_id",       # Unique identifier
    "dataset",      # 'gefcom' or 'household'
    "granularity",  # 'daily', 'weekly', 'monthly', etc.
    "context_json", # Raw statistics as JSON (traceability)
    "prompt_text",  # Full prompt sent to Gemini
]


def _safe_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    """Safely retrieve a float value from a pandas Series row.

    Returns the value at ``col`` if the column exists and is not null.
    Returns ``default`` otherwise. Used by all prompt builders to
    prevent KeyError and ValueError when aggregate DataFrames have
    missing or optional columns (e.g. ``load_std`` can be NaN for
    single-row groups).

    Args:
        row: Pandas Series representing one row of an aggregate DataFrame.
        col: Column name to retrieve.
        default: Fallback value when the column is absent or null.

    Returns:
        Float value from the row, or ``default``.

    Example:
        >>> _safe_float(row, "load_std")    # returns 0.0 if NaN
        72.6
    """
    return (
        float(row[col])
        if col in row.index and pd.notna(row[col])
        else default
    )


# ─────────────────────────────────────────────────────────────────────────────
# GEFCom Prompt Builders (5)
# ─────────────────────────────────────────────────────────────────────────────

def build_gefcom_daily_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for GEFCom daily zone summaries.

    Args:
        df: GEFCom daily aggregates from compute_gefcom_daily_stats().
        limit: Maximum rows. Uses stratified sampling across zone_id.
            Set to None for a full run.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    sample = stratified_sample(df, limit, stratify_col="zone_id")
    logger.info("Building %d GEFCom daily prompts.", len(sample))

    for _, r in sample.iterrows():
        prompt = GEFCOM_DAILY_TEMPLATE.format(
            zone_id=r["zone_id"],
            date=str(r["date"])[:10],
            dow=r.get("dow", "N/A"),
            load_mean=r["load_mean"],
            load_min=r["load_min"],
            load_max=r["load_max"],
            load_std=_safe_float(r, "load_std"),
            load_sum=r["load_sum"],
            obs_count=int(r["obs_count"]),
        )
        rows.append({
            "row_id":       f"gefcom_daily_{r['zone_id']}_{str(r['date'])[:10]}",
            "dataset":      "gefcom",
            "granularity":  "daily",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  prompt,
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_gefcom_weekly_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for GEFCom weekly zone summaries.

    Args:
        df: GEFCom weekly aggregates from compute_gefcom_weekly_stats().
        limit: Maximum rows. Uses stratified sampling across zone_id.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    sample = stratified_sample(df, limit, stratify_col="zone_id")
    logger.info("Building %d GEFCom weekly prompts.", len(sample))

    for _, r in sample.iterrows():
        prompt = GEFCOM_WEEKLY_TEMPLATE.format(
            zone_id=r["zone_id"],
            iso_week=r["iso_week"],
            iso_year=r["iso_year"],
            weekly_mean=r["weekly_mean"],
            weekly_min=r["weekly_min"],
            weekly_max=r["weekly_max"],
            weekly_std=_safe_float(r, "weekly_std"),
        )
        rows.append({
            "row_id": (
                f"gefcom_weekly_{r['zone_id']}_"
                f"W{r['iso_week']}_{r['iso_year']}"
            ),
            "dataset":      "gefcom",
            "granularity":  "weekly",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  prompt,
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_gefcom_monthly_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for GEFCom monthly zone summaries.

    Args:
        df: GEFCom monthly aggregates from compute_gefcom_monthly_stats().
        limit: Maximum rows. Uses stratified sampling across zone_id.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    sample = stratified_sample(df, limit, stratify_col="zone_id")
    logger.info("Building %d GEFCom monthly prompts.", len(sample))

    for _, r in sample.iterrows():
        month_name = calendar.month_name[int(r["month"])]
        prompt = GEFCOM_MONTHLY_TEMPLATE.format(
            zone_id=r["zone_id"],
            month_name=month_name,
            year=int(r["year"]),
            monthly_mean=r["monthly_mean"],
            monthly_min=r["monthly_min"],
            monthly_max=r["monthly_max"],
            monthly_std=_safe_float(r, "monthly_std"),
        )
        rows.append({
            "row_id": (
                f"gefcom_monthly_{r['zone_id']}_"
                f"{month_name}_{int(r['year'])}"
            ),
            "dataset":      "gefcom",
            "granularity":  "monthly",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  prompt,
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_gefcom_seasonal_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for GEFCom seasonal zone summaries.

    Args:
        df: GEFCom seasonal aggregates from compute_gefcom_seasonal_stats().
        limit: Maximum rows. Uses stratified sampling across zone_id.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    sample = stratified_sample(df, limit, stratify_col="zone_id")
    logger.info("Building %d GEFCom seasonal prompts.", len(sample))

    for _, r in sample.iterrows():
        prompt = GEFCOM_SEASONAL_TEMPLATE.format(
            zone_id=r["zone_id"],
            season=r["season"],
            year=int(r["year"]),
            seasonal_mean=r["seasonal_mean"],
            seasonal_min=r["seasonal_min"],
            seasonal_max=r["seasonal_max"],
            seasonal_std=_safe_float(r, "seasonal_std"),
            day_count=int(r["day_count"]),
        )
        rows.append({
            "row_id": (
                f"gefcom_seasonal_{r['zone_id']}_"
                f"{r['season']}_{int(r['year'])}"
            ),
            "dataset":      "gefcom",
            "granularity":  "seasonal",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  prompt,
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_gefcom_system_level_prompts(
    system_dfs: Dict[str, pd.DataFrame],
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for GEFCom synthetic system-level summaries.

    Generates prompts for daily, weekly, and monthly system-level aggregates,
    distributing the limit equally across the three granularities.

    Args:
        system_dfs: Dict mapping granularity → system-level DataFrame as
            produced by compute_gefcom_system_level().
        limit: Maximum total rows across all three granularities.

    Returns:
        Combined DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    per_gran = (limit // 3) if limit else None

    # ── Daily system-level ───────────────────────────────────────────────────
    df = system_dfs.get("daily", pd.DataFrame())
    sample = df.head(per_gran) if per_gran else df
    logger.info("Building %d system-level daily prompts.", len(sample))
    for _, r in sample.iterrows():
        date_str = str(r.get("date", ""))[:10]
        rows.append({
            "row_id":       f"gefcom_system_level_daily_{date_str}",
            "dataset":      "gefcom",
            "granularity":  "system_level",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  GEFCOM_SYSTEM_LEVEL_TEMPLATE.format(
                date_label=f"{date_str} ({r.get('dow', 'N/A')})",
                load_mean=_safe_float(r, "load_mean"),
                load_min=_safe_float(r, "load_min"),
                load_max=_safe_float(r, "load_max"),
                load_std=_safe_float(r, "load_std"),
                load_sum=_safe_float(r, "load_sum"),
                granularity="daily",
            ),
        })

    # ── Weekly system-level ──────────────────────────────────────────────────
    df = system_dfs.get("weekly", pd.DataFrame())
    sample = df.head(per_gran) if per_gran else df
    logger.info("Building %d system-level weekly prompts.", len(sample))
    for _, r in sample.iterrows():
        rows.append({
            "row_id": (
                f"gefcom_system_level_weekly_"
                f"W{r.get('iso_week')}_{r.get('iso_year')}"
            ),
            "dataset":      "gefcom",
            "granularity":  "system_level",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  GEFCOM_SYSTEM_LEVEL_TEMPLATE.format(
                date_label=(
                    f"ISO Week {r.get('iso_week', 'N/A')} of "
                    f"{r.get('iso_year', 'N/A')}"
                ),
                load_mean=_safe_float(r, "weekly_mean"),
                load_min=_safe_float(r, "weekly_min"),
                load_max=_safe_float(r, "weekly_max"),
                load_std=_safe_float(r, "weekly_std"),
                load_sum=_safe_float(r, "weekly_mean") * 7,
                granularity="weekly",
            ),
        })

    # ── Monthly system-level ─────────────────────────────────────────────────
    df = system_dfs.get("monthly", pd.DataFrame())
    sample = df.head(per_gran) if per_gran else df
    logger.info("Building %d system-level monthly prompts.", len(sample))
    for _, r in sample.iterrows():
        month_name = calendar.month_name[int(r.get("month", 1))]
        rows.append({
            "row_id": (
                f"gefcom_system_level_monthly_"
                f"{month_name}_{int(r.get('year', 0))}"
            ),
            "dataset":      "gefcom",
            "granularity":  "system_level",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  GEFCOM_SYSTEM_LEVEL_TEMPLATE.format(
                date_label=f"{month_name} {int(r.get('year', 0))}",
                load_mean=_safe_float(r, "monthly_mean"),
                load_min=_safe_float(r, "monthly_min"),
                load_max=_safe_float(r, "monthly_max"),
                load_std=_safe_float(r, "monthly_std"),
                load_sum=_safe_float(r, "monthly_mean") * 30,
                granularity="monthly",
            ),
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


# ─────────────────────────────────────────────────────────────────────────────
# Household Prompt Builders (5)
# ─────────────────────────────────────────────────────────────────────────────

def build_household_daily_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for household daily summaries.

    Args:
        df: Household daily aggregates from aggregate_household().
        limit: Maximum rows. Uses stratified sampling across year.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    df_with_year = df.copy()
    df_with_year["_year"] = pd.to_datetime(
        df_with_year["period_start"]
    ).dt.year
    sample = stratified_sample(df_with_year, limit, stratify_col="_year")
    logger.info("Building %d household daily prompts.", len(sample))

    for _, r in sample.iterrows():
        date_str = str(r["period_start"])[:10]
        rows.append({
            "row_id":       f"household_daily_{date_str}",
            "dataset":      "household",
            "granularity":  "daily",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  HOUSEHOLD_DAILY_TEMPLATE.format(
                date=date_str,
                gap_mean=_safe_float(r, "Global_active_power_mean"),
                gap_min=_safe_float(r, "Global_active_power_min"),
                gap_max=_safe_float(r, "Global_active_power_max"),
                volt_mean=_safe_float(r, "Voltage_mean"),
                gi_mean=_safe_float(r, "Global_intensity_mean"),
                sm1_mean=_safe_float(r, "Sub_metering_1_mean"),
                sm2_mean=_safe_float(r, "Sub_metering_2_mean"),
                sm3_mean=_safe_float(r, "Sub_metering_3_mean"),
            ),
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_household_weekly_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for household weekly summaries.

    Args:
        df: Household weekly aggregates from aggregate_household().
        limit: Maximum rows. Uses stratified sampling across year.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    df_with_year = df.copy()
    df_with_year["_year"] = pd.to_datetime(
        df_with_year["period_start"]
    ).dt.year
    sample = stratified_sample(df_with_year, limit, stratify_col="_year")
    logger.info("Building %d household weekly prompts.", len(sample))

    for _, r in sample.iterrows():
        date_str = str(r["period_start"])[:10]
        rows.append({
            "row_id":       f"household_weekly_{date_str}",
            "dataset":      "household",
            "granularity":  "weekly",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  HOUSEHOLD_WEEKLY_TEMPLATE.format(
                period_start=date_str,
                gap_mean=_safe_float(r, "Global_active_power_mean"),
                gap_min=_safe_float(r, "Global_active_power_min"),
                gap_max=_safe_float(r, "Global_active_power_max"),
                sm1_mean=_safe_float(r, "Sub_metering_1_mean"),
                sm2_mean=_safe_float(r, "Sub_metering_2_mean"),
                sm3_mean=_safe_float(r, "Sub_metering_3_mean"),
            ),
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_household_monthly_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for household monthly summaries.

    Args:
        df: Household monthly aggregates from aggregate_household().
        limit: Maximum rows. Uses stratified sampling across year.

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    df_with_year = df.copy()
    df_with_year["_year"] = pd.to_datetime(
        df_with_year["period_start"]
    ).dt.year
    sample = stratified_sample(df_with_year, limit, stratify_col="_year")
    logger.info("Building %d household monthly prompts.", len(sample))

    for _, r in sample.iterrows():
        date_str = str(r["period_start"])[:10]
        month_year = pd.to_datetime(r["period_start"]).strftime("%B %Y")
        rows.append({
            "row_id":       f"household_monthly_{date_str}",
            "dataset":      "household",
            "granularity":  "monthly",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  HOUSEHOLD_MONTHLY_TEMPLATE.format(
                month_year=month_year,
                gap_mean=_safe_float(r, "Global_active_power_mean"),
                gap_min=_safe_float(r, "Global_active_power_min"),
                gap_max=_safe_float(r, "Global_active_power_max"),
                gap_std=_safe_float(r, "Global_active_power_std"),
                sm1_mean=_safe_float(r, "Sub_metering_1_mean"),
                sm2_mean=_safe_float(r, "Sub_metering_2_mean"),
                sm3_mean=_safe_float(r, "Sub_metering_3_mean"),
            ),
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_household_appliance_prompts(
    appliance_dfs: Dict[str, pd.DataFrame],
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for household appliance sub-metering summaries.

    Combines daily, weekly, and monthly appliance DataFrames, distributing
    the row limit equally across the three granularities.

    Args:
        appliance_dfs: Dict from compute_household_appliance() mapping
            granularity → appliance DataFrame.
        limit: Maximum total rows across all three granularities.

    Returns:
        Combined DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    per_gran = (limit // 3) if limit else None

    for label, df in appliance_dfs.items():
        if df.empty:
            continue
        df_with_year = df.copy()
        df_with_year["_year"] = pd.to_datetime(
            df_with_year["period_start"]
        ).dt.year
        sample = stratified_sample(
            df_with_year, per_gran, stratify_col="_year"
        )
        logger.info(
            "Building %d household appliance %s prompts.",
            len(sample), label,
        )

        for _, r in sample.iterrows():
            date_str = str(r["period_start"])[:10]
            sm1_mean = _safe_float(r, "Sub_metering_1_mean")
            sm2_mean = _safe_float(r, "Sub_metering_2_mean")
            sm3_mean = _safe_float(r, "Sub_metering_3_mean")
            total_sm = (
                _safe_float(r, "total_submetering_mean")
                or (sm1_mean + sm2_mean + sm3_mean)
            )

            def _share(val: float, total: float = total_sm) -> float:
                """Return percentage share of val in total, or 0.0."""
                return round(val / total * 100, 1) if total > 0 else 0.0

            rows.append({
                "row_id":       f"household_appliance_{label}_{date_str}",
                "dataset":      "household",
                "granularity":  "appliance",
                "context_json": json.dumps(
                    {k: str(v) for k, v in r.to_dict().items()}
                ),
                "prompt_text":  HOUSEHOLD_APPLIANCE_TEMPLATE.format(
                    date_label=f"{label} period ending {date_str}",
                    sm1_mean=sm1_mean,
                    sm1_min=_safe_float(r, "Sub_metering_1_min"),
                    sm1_max=_safe_float(r, "Sub_metering_1_max"),
                    sm2_mean=sm2_mean,
                    sm2_min=_safe_float(r, "Sub_metering_2_min"),
                    sm2_max=_safe_float(r, "Sub_metering_2_max"),
                    sm3_mean=sm3_mean,
                    sm3_min=_safe_float(r, "Sub_metering_3_min"),
                    sm3_max=_safe_float(r, "Sub_metering_3_max"),
                    total_sm_mean=total_sm,
                    sm1_share=_share(sm1_mean),
                    sm2_share=_share(sm2_mean),
                    sm3_share=_share(sm3_mean),
                    gap_mean=_safe_float(r, "Global_active_power_mean"),
                ),
            })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)


def build_household_yearly_prompts(
    df: pd.DataFrame,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Build prompt-input rows for household yearly summaries.

    The household dataset spans 2006-2010, so at most 5 rows are generated.
    No stratified sampling is applied — the population is already small.

    Args:
        df: Household yearly aggregates from compute_household_yearly().
        limit: Maximum rows (applied as df.head(limit)).

    Returns:
        DataFrame with PROMPT_CSV_COLUMNS schema.
    """
    rows: List[Dict[str, Any]] = []
    sample = df.head(limit) if limit else df
    logger.info("Building %d household yearly prompts.", len(sample))

    for _, r in sample.iterrows():
        year = int(r["year"])
        rows.append({
            "row_id":       f"household_yearly_{year}",
            "dataset":      "household",
            "granularity":  "yearly",
            "context_json": json.dumps(
                {k: str(v) for k, v in r.to_dict().items()}
            ),
            "prompt_text":  HOUSEHOLD_YEARLY_TEMPLATE.format(
                year=year,
                yearly_mean=r["yearly_mean"],
                yearly_min=r["yearly_min"],
                yearly_max=r["yearly_max"],
                yearly_std=_safe_float(r, "yearly_std"),
                peak_season=r.get("peak_season", "N/A"),
                sm1_mean=_safe_float(r, "Sub_metering_1_mean"),
                sm2_mean=_safe_float(r, "Sub_metering_2_mean"),
                sm3_mean=_safe_float(r, "Sub_metering_3_mean"),
            ),
        })

    return pd.DataFrame(rows, columns=PROMPT_CSV_COLUMNS)
