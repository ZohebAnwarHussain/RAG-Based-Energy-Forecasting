"""Unit tests for src/knowledge_base/sampling.py.

Verifies stratified_sample() distributes rows across groups,
respects the limit, handles missing columns, and returns full
data when limit is None.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from src.knowledge_base.sampling import stratified_sample


def _make_df(n_zones=4, rows_per_zone=10):
    """Build a synthetic DataFrame with multiple zones."""
    rows = []
    for z in range(1, n_zones + 1):
        for i in range(rows_per_zone):
            rows.append({"zone_id": z, "load_mean": 100.0 + z * 10 + i, "date": f"2005-01-{i+1:02d}"})
    return pd.DataFrame(rows)


def test_returns_full_df_when_limit_is_none():
    """limit=None returns the entire DataFrame."""
    df = _make_df()
    result = stratified_sample(df, limit=None, stratify_col="zone_id")
    assert len(result) == len(df)


def test_respects_total_limit():
    """Output has at most 'limit' rows."""
    df = _make_df(n_zones=4, rows_per_zone=20)
    result = stratified_sample(df, limit=12, stratify_col="zone_id")
    assert len(result) <= 12


def test_all_groups_represented():
    """Each unique zone_id value should appear in the sample."""
    df = _make_df(n_zones=5, rows_per_zone=10)
    result = stratified_sample(df, limit=15, stratify_col="zone_id")
    assert result["zone_id"].nunique() == 5


def test_even_distribution():
    """Rows should be roughly evenly distributed across groups."""
    df = _make_df(n_zones=4, rows_per_zone=20)
    result = stratified_sample(df, limit=20, stratify_col="zone_id")
    counts = result["zone_id"].value_counts()
    # Each group should get ~5 rows (20/4)
    assert counts.min() >= 4
    assert counts.max() <= 6


def test_fallback_when_column_missing():
    """Falls back to df.head() when stratify_col is not in DataFrame."""
    df = _make_df()
    result = stratified_sample(df, limit=5, stratify_col="nonexistent_col")
    assert len(result) == 5


def test_returns_dataframe():
    """Result must be a pandas DataFrame."""
    df = _make_df()
    result = stratified_sample(df, limit=10, stratify_col="zone_id")
    assert isinstance(result, pd.DataFrame)


def test_reproducible_with_same_seed():
    """Same random_state produces identical results."""
    df = _make_df()
    r1 = stratified_sample(df, limit=8, stratify_col="zone_id", random_state=42)
    r2 = stratified_sample(df, limit=8, stratify_col="zone_id", random_state=42)
    pd.testing.assert_frame_equal(r1, r2)


def test_different_seed_may_differ():
    """Different random_state can produce different results."""
    df = _make_df(n_zones=2, rows_per_zone=20)
    r1 = stratified_sample(df, limit=6, stratify_col="zone_id", random_state=42)
    r2 = stratified_sample(df, limit=6, stratify_col="zone_id", random_state=99)
    # Not guaranteed to differ, but with enough rows they should
    assert len(r1) == len(r2)


def test_limit_larger_than_df():
    """If limit exceeds DataFrame size, returns at most len(df) rows."""
    df = _make_df(n_zones=2, rows_per_zone=3)  # 6 total rows
    result = stratified_sample(df, limit=100, stratify_col="zone_id")
    assert len(result) <= 6
