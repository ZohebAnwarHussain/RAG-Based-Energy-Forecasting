"""KB context selection for golden dataset reference answer generation.

Two selectors are provided:

    select_kb_context()          Single-dataset queries (GEFCom or household)
    select_cross_scale_context() Cross-scale queries (both datasets combined)

Selection strategy:
    1. Filter by granularity_target (daily / weekly / monthly / mixed)
    2. Boost rows matching zone or appliance keywords for hybrid queries
    3. Sample n_chunks rows randomly with a fixed seed for reproducibility

The selected chunks become:
    - expected_context_summary   Text passed to Gemini 2.5 Flash for generation
    - expected_summary_ids        Row IDs stored as Context Recall ground truth
    - expected_primary_summary_id First row ID used as MRR ground truth
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def select_kb_context(
    query_meta: Dict[str, Any],
    kb_df: pd.DataFrame,
    n_chunks: int = 5,
) -> Tuple[str, List[str], str]:
    """Select relevant KB chunks for a single-dataset query.

    Filters by ``granularity_target`` first, then boosts rows matching
    zone or appliance keywords for hybrid-targeted queries. Returns
    a fixed random sample of ``n_chunks`` rows.

    Args:
        query_meta: Query metadata dict containing at minimum:
            ``granularity_target`` (str), ``query_type`` (str),
            ``user_query`` (str).
        kb_df: KB DataFrame for the relevant dataset (gefcom or household).
            Should be pre-filtered to one dataset for single-dataset queries.
        n_chunks: Number of KB chunks to include as context. Default 5.

    Returns:
        Tuple of:
            - context_str (str): Formatted context string with each chunk
              labelled by row_id, dataset, and granularity.
            - all_ids (List[str]): All selected row_ids — used as
              expected_summary_ids (Context Recall ground truth).
            - primary_id (str): First selected row_id — used as
              expected_primary_summary_id (MRR ground truth).

    Example:
        >>> ctx, ids, primary = select_kb_context(query_meta, gefcom_kb)
        >>> len(ids) <= 5
        True
    """
    granularity = query_meta.get("granularity_target", "mixed")
    query_type  = query_meta.get("query_type", "")
    query_text  = query_meta.get("user_query", "").lower()

    df = kb_df.copy()

    # ── Granularity filter ────────────────────────────────────────────────────
    if granularity in ["daily", "weekly", "monthly"]:
        preferred = df[df["granularity"] == granularity]
        if preferred.empty:
            preferred = df
    else:
        preferred = df

    # ── Keyword boosting for zone_specific and appliance queries ──────────────
    # These query types rely on specific identifiers that keyword search should
    # surface. Without boosting, random sampling may miss the most relevant rows.
    if query_type in ["zone_specific", "appliance"]:
        boost_kws: List[str] = []
        if any(
            k in query_text
            for k in ["zone 21", "system-level", "system level"]
        ):
            boost_kws = ["system"]
        if any(
            k in query_text
            for k in [
                "sub-meter", "sub_meter", "kitchen",
                "laundry", "hvac", "water heat",
            ]
        ):
            boost_kws = ["Sub_metering"]
        if boost_kws:
            mask = preferred["summary"].str.contains(
                "|".join(boost_kws), case=False, na=False
            )
            boosted = preferred[mask]
            if len(boosted) >= 2:
                preferred = boosted

    n        = min(n_chunks, len(preferred))
    selected = preferred.sample(n=n, random_state=42)

    context_parts = [
        f"[{r['row_id']}] ({r['dataset']}/{r['granularity']}):\n{r['summary']}"
        for _, r in selected.iterrows()
    ]
    context_str = "\n\n".join(context_parts)
    all_ids     = selected["row_id"].tolist()
    primary_id  = all_ids[0] if all_ids else ""

    return context_str, all_ids, primary_id


def select_cross_scale_context(
    query_meta: Dict[str, Any],
    master_kb_df: pd.DataFrame,
    n_chunks: int = 6,
) -> Tuple[str, List[str], str]:
    """Select KB context from both datasets for cross-scale queries.

    Splits ``n_chunks`` equally between GEFCom and household KB entries
    to ensure multi-scale evidence is present in the reference answer.
    This is critical for cross-scale queries which must draw evidence
    from both datasets simultaneously — a cross-scale answer grounded
    only in one dataset would fail the thesis evaluation criteria.

    Args:
        query_meta: Query metadata dict. Used to filter by
            ``granularity_target`` within each dataset.
        master_kb_df: Full master KB DataFrame containing both GEFCom
            and household rows. Use the unfiltered master KB here,
            not a dataset-filtered subset.
        n_chunks: Total chunks to include. Split as ``n_chunks // 2``
            per dataset. Default 6 gives 3 GEFCom + 3 household.

    Returns:
        Tuple of (context_str, all_ids, primary_id) following the
        same convention as select_kb_context().

    Example:
        >>> ctx, ids, primary = select_cross_scale_context(
        ...     query_meta, master_kb
        ... )
        >>> # Both datasets represented in context
        >>> any("gefcom" in i for i in ids)
        True
        >>> any("household" in i for i in ids)
        True
    """
    half        = n_chunks // 2
    granularity = query_meta.get("granularity_target", "mixed")

    def _sample_dataset(dataset_name: str) -> pd.DataFrame:
        """Sample chunks from one dataset, filtered by granularity."""
        sub = master_kb_df[master_kb_df["dataset"] == dataset_name]
        if granularity not in ["mixed", "cross_scale"]:
            filtered = sub[sub["granularity"] == granularity]
            if not filtered.empty:
                sub = filtered
        n = min(half, len(sub))
        return sub.sample(n=n, random_state=42) if n > 0 else pd.DataFrame()

    selected = pd.concat(
        [_sample_dataset("gefcom"), _sample_dataset("household")],
        ignore_index=True,
    )

    context_parts = [
        f"[{r['row_id']}] ({r['dataset']}/{r['granularity']}):\n{r['summary']}"
        for _, r in selected.iterrows()
    ]
    context_str = "\n\n".join(context_parts)
    all_ids     = selected["row_id"].tolist()
    primary_id  = all_ids[0] if all_ids else ""

    return context_str, all_ids, primary_id
