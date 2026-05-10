"""
Shared fixtures for all unit tests.

All fixtures use synthetic in-memory data -- no raw CSV files, no FAISS
index, no API keys are read. Tests must run in any environment without
network access.
"""
import sys
from pathlib import Path

# Ensure project root is importable regardless of how pytest is invoked
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# -- Minimal KB row dicts (mimic combined_master_summaries.csv rows) ----------

@pytest.fixture
def gefcom_daily_row():
    """One GEFCom daily summary row as a dict."""
    return {
        "row_id": "gefcom_daily_z1_20050101",
        "dataset": "gefcom",
        "granularity": "daily",
        "zone_id": "1",
        "context_json": '{"month": 1, "dow": 0, "load_mean": 12345.6}',
        "summary": "Zone 1 saw 12345.6 MW average load on January 1 2005.",
        "parent_id": "gefcom_weekly_z1_2005w01",
        "source": "gefcom_daily_z1_20050101",
    }


@pytest.fixture
def gefcom_weekly_row():
    """One GEFCom weekly summary row."""
    return {
        "row_id": "gefcom_weekly_z1_2005w01",
        "dataset": "gefcom",
        "granularity": "weekly",
        "zone_id": "1",
        "context_json": '{"iso_week": 1}',
        "summary": "Zone 1 weekly load in ISO week 1 of 2005 averaged 13000 MW.",
        "parent_id": "gefcom_monthly_z1_200501",
        "source": "gefcom_weekly_z1_2005w01",
    }


@pytest.fixture
def household_daily_row():
    """One household daily summary row."""
    return {
        "row_id": "household_daily_20070101",
        "dataset": "household",
        "granularity": "daily",
        "zone_id": "",
        "context_json": '{"month": 1, "dow": 0, "Global_active_power_mean": 1.234}',
        "summary": "Household consumed 1.234 kW average on January 1 2007.",
        "parent_id": "household_weekly_2007w01",
        "source": "household_daily_20070101",
    }


@pytest.fixture
def sample_kb_rows(gefcom_daily_row, gefcom_weekly_row, household_daily_row):
    """Three-row minimal KB for retrieval tests."""
    return [gefcom_daily_row, gefcom_weekly_row, household_daily_row]


# -- Retrieved doc fixtures ---------------------------------------------------

@pytest.fixture
def sample_retrieved_docs():
    """Three retrieved docs as returned by DenseRetriever.retrieve_with_scores()."""
    return [
        {
            "row_id": "gefcom_daily_z1_20050101",
            "score": 0.85,
            "dataset": "gefcom",
            "granularity": "daily",
            "page_content": "Zone 1 saw 12345.6 MW average load.",
        },
        {
            "row_id": "gefcom_weekly_z1_2005w01",
            "score": 0.72,
            "dataset": "gefcom",
            "granularity": "weekly",
            "page_content": "Zone 1 weekly load averaged 13000 MW.",
        },
        {
            "row_id": "household_daily_20070101",
            "score": 0.55,
            "dataset": "household",
            "granularity": "daily",
            "page_content": "Household consumed 1.234 kW average.",
        },
    ]


# -- Golden dataset query fixture --------------------------------------------

@pytest.fixture
def sample_golden_query():
    """One golden dataset query row as a dict."""
    return {
        "golden_id": "GQ_001",
        "user_query": "What was the peak demand in Zone 1 during winter 2005?",
        "question": "What was the peak demand in Zone 1 during winter 2005?",
        "reference_answer": "Zone 1 peak demand in winter 2005 was approximately 15000 MW.",
        "ground_truth": "Zone 1 peak demand in winter 2005 was approximately 15000 MW.",
        "expected_summary_ids": '["gefcom_daily_z1_20050101", "gefcom_weekly_z1_2005w01"]',
        "answer_must_include": '["Zone 1", "MW"]',
        "answer_must_not_include": '["household", "I think"]',
        "dataset_source": "gefcom",
        "query_type": "trend",
        "difficulty_level": "Easy",
    }
