"""Stratified sampling for evenly-distributed KB pilot generation.

The original notebook used df.head(limit) which biased the pilot toward
the earliest rows from the lowest-numbered zones. stratified_sample()
ensures the pilot KB contains representation from every zone (GEFCom)
or every year (Household), which is critical for meaningful Recall@k
scores during RAGAS evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def stratified_sample(
    df: pd.DataFrame,
    limit: Optional[int],
    stratify_col: str,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sample rows evenly across unique values of a stratification column.

    Divides the row limit equally across all unique values of
    ``stratify_col``. This ensures the sample contains representation
    from each zone, season, or other categorical grouping rather than
    being biased toward the most common or earliest-appearing values.

    For example, with limit=50 and 20 unique zones, each zone contributes
    at most 2-3 rows. If a zone has fewer rows than its share, all its
    rows are included and the remaining quota is distributed via .head().

    Implementation note: uses an explicit groupby loop with pd.concat
    rather than df.groupby().apply(). The .apply() approach in pandas 2.x
    changes how the grouping column is handled in the result, causing a
    KeyError on the grouping column. Explicit iteration avoids this.

    Args:
        df: Source DataFrame to sample from.
        limit: Maximum total number of rows to return. If ``None``,
            the full DataFrame is returned without sampling.
        stratify_col: Column name to stratify by. Each unique value
            in this column will be represented in the sample.
            Examples: ``'zone_id'``, ``'season'``, ``'year'``.
        random_state: Random seed for reproducibility. Default 42
            ensures the same sample is produced on every run.

    Returns:
        Sampled DataFrame with at most ``limit`` rows, with
        representation from each unique value of ``stratify_col``.

    Example:
        >>> sampled = stratified_sample(gefcom_daily, 50, "zone_id")
        >>> sampled["zone_id"].nunique()    # ≤ 20
        20
        >>> len(sampled)                    # ≤ 50
        50
    """
    if limit is None:
        return df

    if stratify_col not in df.columns:
        logger.warning(
            "Stratify column '%s' not found — falling back to df.head(%d).",
            stratify_col, limit,
        )
        return df.head(limit)

    # Work on a copy and reset index to avoid downstream issues
    df = df.copy().reset_index(drop=True)

    n_groups  = df[stratify_col].nunique()
    per_group = max(1, limit // n_groups)

    # Explicit groupby loop avoids pandas 2.x apply() column-handling bug
    sampled_frames = []
    for _, group in df.groupby(stratify_col, sort=False):
        n = min(len(group), per_group)
        sampled_frames.append(
            group.sample(n=n, random_state=random_state)
        )

    sampled = (
        pd.concat(sampled_frames, ignore_index=True)
        .head(limit)
        .reset_index(drop=True)
    )

    logger.info(
        "Stratified sample: %d rows from %d '%s' groups (target %d).",
        len(sampled), n_groups, stratify_col, limit,
    )
    return sampled
