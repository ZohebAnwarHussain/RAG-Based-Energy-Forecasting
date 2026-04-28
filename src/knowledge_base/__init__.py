"""Knowledge Base generation pipeline.

This package contains all logic for Phase 1 of the thesis pipeline —
transforming raw time-series data into Gemini-generated natural language
summaries suitable for RAG retrieval.

Pipeline flow:
    Raw CSV Data
        → load_gefcom_data() / load_household_data()
        → reshape_gefcom_load() / clean_household()
        → compute_*_stats() / aggregate_household()
        → validate_aggregates()
        → stratified_sample()
        → build_*_prompts()
        → generate_summaries() (Gemini 3 Flash)
        → build_master_knowledge_base() (combined CSV)

All functions are pure relocations from `01_kb_generation_v2.ipynb` —
identical logic, no algorithm changes.

Usage:
    from src.knowledge_base import (
        load_gefcom_data,
        load_household_data,
        compute_gefcom_daily_stats,
        validate_aggregates,
        stratified_sample,
        build_gefcom_daily_prompts,
        configure_gemini_kb,
        generate_summaries,
        build_master_knowledge_base,
    )
"""

# Data loading
from src.knowledge_base.data_loader import (
    load_gefcom_data,
    load_household_data,
    verify_data_paths,
)

# Aggregation
from src.knowledge_base.aggregators import (
    aggregate_household,
    clean_household,
    compute_gefcom_daily_stats,
    compute_gefcom_monthly_stats,
    compute_gefcom_seasonal_stats,
    compute_gefcom_system_level,
    compute_gefcom_weekly_stats,
    compute_household_appliance,
    compute_household_yearly,
    reshape_gefcom_load,
)

# Validation and sampling
from src.knowledge_base.sampling import stratified_sample
from src.knowledge_base.validation import (
    is_valid_summary,
    validate_aggregates,
)

# Prompt building
from src.knowledge_base.prompt_builders import (
    PROMPT_CSV_COLUMNS,
    build_gefcom_daily_prompts,
    build_gefcom_monthly_prompts,
    build_gefcom_seasonal_prompts,
    build_gefcom_system_level_prompts,
    build_gefcom_weekly_prompts,
    build_household_appliance_prompts,
    build_household_daily_prompts,
    build_household_monthly_prompts,
    build_household_weekly_prompts,
    build_household_yearly_prompts,
)

# Generation
from src.knowledge_base.generation import (
    SUMMARY_CSV_COLUMNS,
    call_gemini,
    configure_gemini_kb,
    generate_summaries,
)

# Master KB builder
from src.knowledge_base.master_kb import (
    MASTER_KB_COLUMNS,
    build_master_knowledge_base,
)

__all__ = [
    # Data loading
    "verify_data_paths",
    "load_gefcom_data",
    "load_household_data",
    # GEFCom aggregation
    "reshape_gefcom_load",
    "compute_gefcom_daily_stats",
    "compute_gefcom_weekly_stats",
    "compute_gefcom_monthly_stats",
    "compute_gefcom_seasonal_stats",
    "compute_gefcom_system_level",
    # Household aggregation
    "clean_household",
    "aggregate_household",
    "compute_household_appliance",
    "compute_household_yearly",
    # Validation and sampling
    "validate_aggregates",
    "is_valid_summary",
    "stratified_sample",
    # Prompt building
    "build_gefcom_daily_prompts",
    "build_gefcom_weekly_prompts",
    "build_gefcom_monthly_prompts",
    "build_gefcom_seasonal_prompts",
    "build_gefcom_system_level_prompts",
    "build_household_daily_prompts",
    "build_household_weekly_prompts",
    "build_household_monthly_prompts",
    "build_household_appliance_prompts",
    "build_household_yearly_prompts",
    "PROMPT_CSV_COLUMNS",
    # Generation
    "configure_gemini_kb",
    "call_gemini",
    "generate_summaries",
    "SUMMARY_CSV_COLUMNS",
    # Master KB builder
    "build_master_knowledge_base",
    "MASTER_KB_COLUMNS",
]
