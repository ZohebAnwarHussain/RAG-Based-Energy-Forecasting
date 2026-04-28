"""Centralised configuration for the LJMU thesis RAG pipeline.

All hardcoded values, model names, and pipeline constants are defined
in this package. Import from here rather than redefining values inside
notebooks or src modules.

Usage:
    from config import (
        PATHS,
        BASE,
        KB_MODEL_NAME,
        GOLDEN_MODEL_NAME,
        RAG_MODEL_NAME,
        EMBEDDING_MODEL_NAME,
        MAX_SUMMARIES_PER_TYPE,
        SEASON_MAP,
    )
"""

from config.models import (
    EMBEDDING_MODEL_NAME,
    GOLDEN_MAX_TOKENS,
    GOLDEN_MODEL_NAME,
    GOLDEN_TEMPERATURE,
    KB_MAX_TOKENS,
    KB_MODEL_NAME,
    KB_TEMPERATURE,
    RAG_MAX_TOKENS,
    RAG_MODEL_NAME,
    RAG_TEMPERATURE,
)
from config.paths import BASE, PATHS
from config.pipeline import (
    MAX_RETRIES,
    MAX_SUMMARIES_PER_TYPE,
    REQUEST_DELAY_SECONDS,
    RETRY_BACKOFF_SECONDS,
    SEASON_MAP,
)

__all__ = [
    # Paths
    "BASE",
    "PATHS",
    # Models
    "KB_MODEL_NAME",
    "KB_TEMPERATURE",
    "KB_MAX_TOKENS",
    "GOLDEN_MODEL_NAME",
    "GOLDEN_TEMPERATURE",
    "GOLDEN_MAX_TOKENS",
    "RAG_MODEL_NAME",
    "RAG_TEMPERATURE",
    "RAG_MAX_TOKENS",
    "EMBEDDING_MODEL_NAME",
    # Pipeline
    "MAX_SUMMARIES_PER_TYPE",
    "REQUEST_DELAY_SECONDS",
    "MAX_RETRIES",
    "RETRY_BACKOFF_SECONDS",
    "SEASON_MAP",
]
