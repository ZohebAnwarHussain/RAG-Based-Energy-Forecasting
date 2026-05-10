"""Unit tests for src/knowledge_base/chunk_builder.py.

Verifies that build_enriched_chunk_text() produces headers with
the correct semantic keywords for different granularities, datasets,
and context_json values. All tests use synthetic row dicts.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.knowledge_base.chunk_builder import build_enriched_chunk_text


# -- GEFCom daily chunk -------------------------------------------------------

def test_gefcom_daily_has_zone(gefcom_daily_row):
    """GEFCom daily chunk must include Zone identifier."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "Zone 1" in result


def test_gefcom_daily_has_dataset_keywords(gefcom_daily_row):
    """GEFCom chunks must include grid/load/demand keywords."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "GEFCom" in result
    assert "load" in result.lower()


def test_gefcom_daily_has_granularity(gefcom_daily_row):
    """Chunk must mention its granularity level."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "daily" in result.lower()


def test_gefcom_daily_has_season(gefcom_daily_row):
    """January chunk must include 'winter' season keyword."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "winter" in result.lower()


def test_gefcom_daily_has_month_name(gefcom_daily_row):
    """January chunk must include month name."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "January" in result


def test_gefcom_daily_weekday_flag(gefcom_daily_row):
    """dow=0 (Monday) should include 'weekday' keyword."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "weekday" in result.lower()


def test_gefcom_daily_preserves_summary(gefcom_daily_row):
    """Original summary text must appear after the header."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    assert "Zone 1 saw 12345.6 MW" in result


def test_gefcom_daily_header_before_summary(gefcom_daily_row):
    """Header line must come before the summary text."""
    result = build_enriched_chunk_text(gefcom_daily_row)
    lines = result.split("\n")
    assert len(lines) >= 2
    # Header is line 0, summary starts at line 1+
    assert "Zone 1 saw 12345.6 MW" not in lines[0]


# -- GEFCom weekly chunk ------------------------------------------------------

def test_gefcom_weekly_has_week_keywords(gefcom_weekly_row):
    """Weekly chunk must include temporal keywords for early-year week."""
    result = build_enriched_chunk_text(gefcom_weekly_row)
    assert "weekly" in result.lower()
    # ISO week 1 should trigger "early year" keywords
    assert "early year" in result.lower() or "first quarter" in result.lower()


# -- Household chunk -----------------------------------------------------------

def test_household_has_residential_keywords(household_daily_row):
    """Household chunks must include residential/appliance keywords."""
    result = build_enriched_chunk_text(household_daily_row)
    assert "household" in result.lower()
    assert "residential" in result.lower() or "consumption" in result.lower()


def test_household_no_zone(household_daily_row):
    """Household chunks should not include a Zone identifier."""
    result = build_enriched_chunk_text(household_daily_row)
    assert "Zone" not in result.split("\n")[0]


# -- Seasonal chunk ------------------------------------------------------------

def test_seasonal_chunk_has_season_keywords():
    """Seasonal chunk must include season comparison keywords."""
    row = {
        "row_id": "gefcom_seasonal_z1_summer",
        "dataset": "gefcom",
        "granularity": "seasonal",
        "zone_id": "1",
        "context_json": '{"season": "Summer"}',
        "summary": "Zone 1 summer peak was 20000 MW.",
    }
    result = build_enriched_chunk_text(row)
    assert "Summer" in result
    assert "seasonal" in result.lower()


# -- Appliance chunk -----------------------------------------------------------

def test_appliance_chunk_has_device_keywords():
    """Appliance chunk must include sub-metering/device keywords."""
    row = {
        "row_id": "household_appliance_kitchen",
        "dataset": "household",
        "granularity": "appliance",
        "zone_id": "",
        "context_json": '{"Sub_metering_1_mean": 1.5}',
        "summary": "Kitchen appliances consumed 1.5 kW average.",
    }
    result = build_enriched_chunk_text(row)
    assert "appliance" in result.lower()
    assert "sub-metering" in result.lower() or "device" in result.lower()


# -- Edge cases ----------------------------------------------------------------

def test_empty_summary():
    """Empty summary should still produce a header."""
    row = {
        "row_id": "test",
        "dataset": "gefcom",
        "granularity": "daily",
        "zone_id": "1",
        "context_json": "{}",
        "summary": "",
    }
    result = build_enriched_chunk_text(row)
    assert len(result.strip()) > 0


def test_invalid_context_json():
    """Invalid context_json should not raise an error."""
    row = {
        "row_id": "test",
        "dataset": "gefcom",
        "granularity": "daily",
        "zone_id": "1",
        "context_json": "not valid json",
        "summary": "Some summary.",
    }
    result = build_enriched_chunk_text(row)
    assert "Some summary." in result
