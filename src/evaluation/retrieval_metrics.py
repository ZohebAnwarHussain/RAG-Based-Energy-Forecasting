"""Retrieval quality metrics.

Computes standard information retrieval metrics for each query-pipeline
combination by comparing retrieved document IDs against the golden
dataset's expected_summary_ids.

Metrics implemented:

    Recall@k
        Fraction of expected documents that appear in the top-k retrieved.
        recall@k = |retrieved ∩ expected| / |expected|
        High recall means the retriever finds most of the relevant chunks.

    Precision@k
        Fraction of retrieved documents that are in the expected set.
        precision@k = |retrieved ∩ expected| / |retrieved|
        High precision means retrieved chunks are mostly relevant.

    MRR (Mean Reciprocal Rank)
        Reciprocal of the rank position of the first relevant document.
        MRR = 1 / rank_of_first_match
        High MRR means the most relevant document appears early.

    nDCG (normalised Discounted Cumulative Gain)
        Measures ranking quality — rewards relevant documents appearing
        higher in the result list. Uses binary relevance (1 if in
        expected set, 0 otherwise).
        nDCG = DCG / ideal_DCG

All metrics are computed per query and then averaged across queries
to produce pipeline-level scores.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def _recall_at_k(
    retrieved: List[str],
    expected: List[str],
) -> float:
    """Compute Recall@k for a single query.

    Args:
        retrieved: List of retrieved document row_ids (ordered by rank).
        expected: List of expected document row_ids from golden dataset.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not expected:
        return 0.0
    retrieved_set = set(retrieved)
    expected_set  = set(expected)
    hits = len(retrieved_set & expected_set)
    return hits / len(expected_set)


def _precision_at_k(
    retrieved: List[str],
    expected: List[str],
) -> float:
    """Compute Precision@k for a single query.

    Args:
        retrieved: List of retrieved document row_ids.
        expected: List of expected document row_ids.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not retrieved:
        return 0.0
    retrieved_set = set(retrieved)
    expected_set  = set(expected)
    hits = len(retrieved_set & expected_set)
    return hits / len(retrieved_set)


def _mrr(
    retrieved: List[str],
    expected: List[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query.

    Args:
        retrieved: List of retrieved document row_ids (ordered by rank).
        expected: List of expected document row_ids.

    Returns:
        Float between 0.0 and 1.0. Returns 0.0 if no expected document
        appears in the retrieved list.
    """
    expected_set = set(expected)
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in expected_set:
            return 1.0 / rank
    return 0.0


def _ndcg(
    retrieved: List[str],
    expected: List[str],
) -> float:
    """Compute normalised Discounted Cumulative Gain for a single query.

    Uses binary relevance: 1 if document is in expected set, 0 otherwise.

    Args:
        retrieved: List of retrieved document row_ids (ordered by rank).
        expected: List of expected document row_ids.

    Returns:
        Float between 0.0 and 1.0.
    """
    if not expected or not retrieved:
        return 0.0

    expected_set = set(expected)

    # DCG: sum of relevance / log2(rank + 1)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved, 1):
        rel = 1.0 if doc_id in expected_set else 0.0
        dcg += rel / math.log2(rank + 1)

    # Ideal DCG: all relevant docs at the top
    n_relevant = min(len(expected_set), len(retrieved))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_relevant))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_retrieval_metrics(
    retrieval_results_df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    """Compute all retrieval metrics for every query-pipeline combination.

    Reads the retrieval results CSV (from Phase 4), computes Recall@k,
    Precision@k, MRR, and nDCG for each row, and saves the results.

    Args:
        retrieval_results_df: DataFrame from Phase 4 with columns:
            golden_id, pipeline, expected_ids (JSON), retrieved_ids (JSON).
        output_path: CSV path to save per-query metric scores.

    Returns:
        DataFrame with all original columns plus:
        recall_at_k, precision_at_k, mrr, ndcg.

    Example:
        >>> metrics_df = compute_retrieval_metrics(results_df, output_path)
        >>> metrics_df.groupby("pipeline")[["recall_at_k", "mrr"]].mean()
    """
    logger.info(
        "Computing retrieval metrics for %d query-pipeline combinations.",
        len(retrieval_results_df),
    )

    metrics: List[Dict[str, Any]] = []

    for _, row in retrieval_results_df.iterrows():
        retrieved = json.loads(row["retrieved_ids"])
        expected  = json.loads(row["expected_ids"])

        metrics.append({
            "golden_id":     row["golden_id"],
            "pipeline":      row["pipeline"],
            "recall_at_k":   round(_recall_at_k(retrieved, expected), 4),
            "precision_at_k": round(_precision_at_k(retrieved, expected), 4),
            "mrr":           round(_mrr(retrieved, expected), 4),
            "ndcg":          round(_ndcg(retrieved, expected), 4),
        })

    metrics_df = pd.DataFrame(metrics)

    # Merge with original results for complete output
    result = retrieval_results_df.merge(
        metrics_df, on=["golden_id", "pipeline"], how="left"
    )
    result.to_csv(output_path, index=False)

    # Log pipeline-level averages
    print("\n" + "=" * 60)
    print("  RETRIEVAL METRICS BY PIPELINE")
    print("=" * 60)
    summary = result.groupby("pipeline")[
        ["recall_at_k", "precision_at_k", "mrr", "ndcg"]
    ].mean()
    print(summary.round(4).to_string())
    print("=" * 60)

    logger.info(
        "Retrieval metrics saved to %s.", output_path
    )
    return result
