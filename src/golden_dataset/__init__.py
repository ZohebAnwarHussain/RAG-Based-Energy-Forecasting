"""Golden dataset generation pipeline.

This package contains all logic for Phase 2 of the thesis pipeline —
generating a 50-query evaluation dataset with Gemini 2.5 Flash reference
answers grounded in KB context.

Pipeline flow:
    combined_master_summaries.csv (KB)
        → load_kb_summaries()
        → select_kb_context() / select_cross_scale_context()
        → call_gemini_golden() (Gemini 2.5 Flash)
        → generate_golden_dataset() (per dataset source)
        → build_combined_golden_dataset() (master CSV)

Model independence:
    KB summaries       → gemini-3-flash-preview  (Gemini 3)
    Reference answers  → gemini-2.5-flash         (Gemini 2.5)
    RAG generation     → llama-3.3-70b-versatile  (Llama / Groq)

The primary evaluation independence boundary is between Gemini
(reference answers) and Llama (RAG-generated answers), ensuring
RAGAS scores reflect genuine cross-model evaluation.

Usage:
    from src.golden_dataset import (
        load_kb_summaries,
        select_kb_context,
        select_cross_scale_context,
        configure_gemini_golden,
        generate_golden_dataset,
        build_combined_golden_dataset,
        GEFCOM_QUERIES,
        HOUSEHOLD_QUERIES,
        CROSS_SCALE_QUERIES,
    )
"""

from src.golden_dataset.generator import (
    GOLDEN_CSV_COLUMNS,
    build_combined_golden_dataset,
    call_gemini_golden,
    configure_gemini_golden,
    generate_golden_dataset,
)
from src.golden_dataset.kb_loader import load_kb_summaries
from src.golden_dataset.context_selector import (
    select_cross_scale_context,
    select_kb_context,
)
from src.golden_dataset.query_bank import (
    CROSS_SCALE_QUERIES,
    GEFCOM_QUERIES,
    HOUSEHOLD_QUERIES,
)

__all__ = [
    # KB loading
    "load_kb_summaries",
    # Context selection
    "select_kb_context",
    "select_cross_scale_context",
    # Query banks
    "GEFCOM_QUERIES",
    "HOUSEHOLD_QUERIES",
    "CROSS_SCALE_QUERIES",
    # Generation
    "configure_gemini_golden",
    "call_gemini_golden",
    "generate_golden_dataset",
    "build_combined_golden_dataset",
    "GOLDEN_CSV_COLUMNS",
]
