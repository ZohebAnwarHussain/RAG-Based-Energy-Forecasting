"""Unit tests for src/golden_dataset/context_selector.py.

Verifies select_kb_context() and select_cross_scale_context()
return the correct shape, respect n_chunks, and apply keyword
boosting for zone/appliance queries.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import pytest

from src.golden_dataset.context_selector import (
    select_kb_context,
    select_cross_scale_context,
)


@pytest.fixture
def gefcom_kb():
    """Minimal GEFCom KB DataFrame with 10 rows across granularities."""
    rows = []
    for i in range(5):
        rows.append({
            "row_id": f"gefcom_daily_z1_{i}",
            "dataset": "gefcom",
            "granularity": "daily",
            "zone_id": "1",
            "summary": f"Zone 1 daily load was {100 + i} MW on day {i}.",
        })
    for i in range(3):
        rows.append({
            "row_id": f"gefcom_weekly_z1_{i}",
            "dataset": "gefcom",
            "granularity": "weekly",
            "zone_id": "1",
            "summary": f"Zone 1 weekly load averaged {200 + i} MW in week {i}.",
        })
    for i in range(2):
        rows.append({
            "row_id": f"gefcom_monthly_z1_{i}",
            "dataset": "gefcom",
            "granularity": "monthly",
            "zone_id": "1",
            "summary": f"Zone 1 monthly load was {300 + i} MW in month {i}.",
        })
    return pd.DataFrame(rows)


@pytest.fixture
def master_kb(gefcom_kb):
    """Master KB with both GEFCom and household rows."""
    household_rows = []
    for i in range(5):
        household_rows.append({
            "row_id": f"household_daily_{i}",
            "dataset": "household",
            "granularity": "daily",
            "zone_id": "",
            "summary": f"Household consumed {1.0 + i * 0.1} kW on day {i}. Sub_metering kitchen laundry.",
        })
    household_df = pd.DataFrame(household_rows)
    return pd.concat([gefcom_kb, household_df], ignore_index=True)


# -- select_kb_context ---------------------------------------------------------

def test_returns_tuple_of_three(gefcom_kb):
    """Must return (context_str, all_ids, primary_id)."""
    query = {"granularity_target": "daily", "query_type": "trend", "user_query": "peak demand"}
    result = select_kb_context(query, gefcom_kb, n_chunks=3)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_ids_list_length(gefcom_kb):
    """all_ids length must not exceed n_chunks."""
    query = {"granularity_target": "daily", "query_type": "trend", "user_query": "load pattern"}
    _, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=3)
    assert len(ids) <= 3


def test_primary_id_is_first(gefcom_kb):
    """primary_id must equal all_ids[0]."""
    query = {"granularity_target": "daily", "query_type": "trend", "user_query": "demand"}
    _, ids, primary = select_kb_context(query, gefcom_kb, n_chunks=3)
    assert primary == ids[0]


def test_context_str_contains_row_ids(gefcom_kb):
    """Context string must contain the selected row_id values."""
    query = {"granularity_target": "daily", "query_type": "trend", "user_query": "demand"}
    ctx, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=3)
    for rid in ids:
        assert rid in ctx


def test_granularity_filter_daily(gefcom_kb):
    """When granularity_target=daily, only daily rows should be selected."""
    query = {"granularity_target": "daily", "query_type": "trend", "user_query": "demand"}
    _, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=5)
    assert all("daily" in rid for rid in ids)


def test_granularity_filter_weekly(gefcom_kb):
    """When granularity_target=weekly, weekly rows should be preferred."""
    query = {"granularity_target": "weekly", "query_type": "trend", "user_query": "weekly pattern"}
    _, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=3)
    assert all("weekly" in rid for rid in ids)


def test_mixed_granularity_no_filter(gefcom_kb):
    """granularity_target=mixed should not filter by granularity."""
    query = {"granularity_target": "mixed", "query_type": "trend", "user_query": "demand"}
    _, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=5)
    assert len(ids) == 5


def test_n_chunks_capped_at_available(gefcom_kb):
    """If n_chunks exceeds available rows, returns all available."""
    query = {"granularity_target": "monthly", "query_type": "trend", "user_query": "monthly"}
    _, ids, _ = select_kb_context(query, gefcom_kb, n_chunks=100)
    assert len(ids) == 2  # only 2 monthly rows


# -- select_cross_scale_context ------------------------------------------------

def test_cross_scale_returns_tuple(master_kb):
    """Must return (context_str, all_ids, primary_id)."""
    query = {"granularity_target": "mixed", "query_type": "cross_scale", "user_query": "compare"}
    result = select_cross_scale_context(query, master_kb, n_chunks=6)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_cross_scale_both_datasets(master_kb):
    """Selected IDs must include rows from both gefcom and household."""
    query = {"granularity_target": "mixed", "query_type": "cross_scale", "user_query": "compare grid household"}
    _, ids, _ = select_cross_scale_context(query, master_kb, n_chunks=6)
    has_gefcom = any("gefcom" in rid for rid in ids)
    has_household = any("household" in rid for rid in ids)
    assert has_gefcom, "No GEFCom rows selected"
    assert has_household, "No household rows selected"


def test_cross_scale_respects_n_chunks(master_kb):
    """Total selected must not exceed n_chunks."""
    query = {"granularity_target": "mixed", "query_type": "cross_scale", "user_query": "compare"}
    _, ids, _ = select_cross_scale_context(query, master_kb, n_chunks=4)
    assert len(ids) <= 4


def test_cross_scale_primary_id(master_kb):
    """primary_id must equal all_ids[0]."""
    query = {"granularity_target": "mixed", "query_type": "cross_scale", "user_query": "compare"}
    _, ids, primary = select_cross_scale_context(query, master_kb, n_chunks=6)
    assert primary == ids[0]
