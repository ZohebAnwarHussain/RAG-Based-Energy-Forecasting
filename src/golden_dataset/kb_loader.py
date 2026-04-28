"""Knowledge Base loader for golden dataset context selection.

Loads the master KB CSV produced by Phase 1 into a pandas DataFrame.
The loaded summaries serve two roles in the golden dataset pipeline:

    1. Context — selected chunks are passed to Gemini 2.5 Flash as
       the evidence base for generating reference answers. The model
       is instructed to ground its answer strictly on these chunks.

    2. Ground truth — the selected chunk row_ids are stored as
       expected_summary_ids in the golden dataset, forming the
       ground truth for RAGAS Context Precision and Context Recall.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_kb_summaries(
    summaries_path: Path,
    dataset_filter: Optional[str] = None,
) -> pd.DataFrame:
    """Load generated KB summaries from the master CSV.

    Args:
        summaries_path: Path to the directory containing
            ``combined_master_summaries.csv``. Typically
            ``PATHS["summaries_csv"]``.
        dataset_filter: Optional dataset name to restrict results.
            Pass ``'gefcom'`` or ``'household'`` to return only rows
            from that dataset. If None, all rows are returned.

    Returns:
        DataFrame of KB summaries with all metadata columns including
        ``kb_id``, ``row_id``, ``dataset``, ``granularity``,
        ``zone_id``, ``year``, ``season``, ``parent_id``, ``summary``.

    Raises:
        FileNotFoundError: If ``combined_master_summaries.csv`` does
            not exist. Run ``notebooks/01_kb_generation.ipynb`` first.

    Example:
        >>> master_kb = load_kb_summaries(PATHS["summaries_csv"])
        >>> gefcom_kb = load_kb_summaries(
        ...     PATHS["summaries_csv"], dataset_filter="gefcom"
        ... )
        >>> len(master_kb) > len(gefcom_kb)
        True
    """
    master_path = summaries_path / "combined_master_summaries.csv"
    if not master_path.exists():
        raise FileNotFoundError(
            f"Master KB not found at {master_path}. "
            "Run notebooks/01_kb_generation.ipynb first to generate it."
        )

    df = pd.read_csv(master_path)
    logger.info("Loaded master KB: %d entries.", len(df))

    if dataset_filter:
        df = df[df["dataset"] == dataset_filter].reset_index(drop=True)
        logger.info(
            "Filtered to '%s': %d entries.", dataset_filter, len(df)
        )

    return df
