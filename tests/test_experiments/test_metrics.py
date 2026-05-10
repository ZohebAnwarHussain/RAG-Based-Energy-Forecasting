"""Unit tests for src/experiments/metrics.py.

Tests the shared metric functions used across all experiments.
The compute_retrieval_metrics() and compute_retrieval_metrics_with_content_fallback()
functions are tested here with synthetic data. The embedding-based functions
(compute_answer_relevance, compute_semantic_similarity, etc.) require the
sentence-transformers model; tests for those are integration tests that
verify return types and value ranges rather than exact values.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.experiments.metrics import (
    compute_retrieval_metrics,
    compute_retrieval_metrics_with_content_fallback,
    _split_sentences,
)


# -- compute_retrieval_metrics (ID-only matching) ----------------------------

def test_retrieval_metrics_perfect_match():
    """All metrics should be maximal when retrieved == relevant at top positions."""
    result = compute_retrieval_metrics(
        retrieved_ids=["a", "b", "c"],
        relevant_ids=["a", "b", "c"],
        k=3,
    )
    assert result["recall_at_k"] == 1.0
    assert result["precision_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert result["ndcg_at_k"] == 1.0
    assert result["relevant_retrieved"] == 3


def test_retrieval_metrics_no_match():
    """All metrics should be zero when no overlap."""
    result = compute_retrieval_metrics(
        retrieved_ids=["x", "y", "z"],
        relevant_ids=["a", "b"],
        k=3,
    )
    assert result["recall_at_k"] == 0.0
    assert result["precision_at_k"] == 0.0
    assert result["mrr"] == 0.0
    assert result["ndcg_at_k"] == 0.0
    assert result["relevant_retrieved"] == 0


def test_retrieval_metrics_partial_match():
    """Partial overlap gives intermediate metric values."""
    result = compute_retrieval_metrics(
        retrieved_ids=["a", "x", "b"],
        relevant_ids=["a", "b", "c"],
        k=3,
    )
    assert abs(result["recall_at_k"] - 2 / 3) < 1e-4
    assert abs(result["precision_at_k"] - 2 / 3) < 1e-4
    assert result["mrr"] == 1.0  # 'a' at rank 1
    assert result["relevant_retrieved"] == 2


def test_retrieval_metrics_empty_relevant():
    """Metrics should handle empty relevant set gracefully."""
    result = compute_retrieval_metrics(
        retrieved_ids=["a", "b"],
        relevant_ids=[],
        k=2,
    )
    assert result["recall_at_k"] == 0.0
    assert result["relevant_available"] == 0


def test_retrieval_metrics_returns_expected_keys():
    """Result dict must contain all expected metric keys."""
    result = compute_retrieval_metrics(["a"], ["a"], k=1)
    expected_keys = {
        "recall_at_k", "precision_at_k", "mrr", "ndcg_at_k",
        "relevant_available", "relevant_retrieved",
    }
    assert set(result.keys()) == expected_keys


# -- compute_retrieval_metrics_with_content_fallback -------------------------

def test_content_fallback_id_match_takes_priority():
    """When IDs match, content fallback is not needed."""
    result = compute_retrieval_metrics_with_content_fallback(
        retrieved_ids=["a", "b"],
        relevant_ids=["a", "b"],
        k=2,
        retrieved_texts=["text1", "text2"],
        relevant_texts=["different1", "different2"],
    )
    assert result["recall_at_k"] == 1.0


def test_content_fallback_text_overlap_hit():
    """High text overlap should count as a hit when ID does not match."""
    result = compute_retrieval_metrics_with_content_fallback(
        retrieved_ids=["x"],
        relevant_ids=["a"],
        k=1,
        retrieved_texts=["Zone 1 daily load was 12345.6 MW on January 1 2005"],
        relevant_texts=["Zone 1 daily load was 12345.6 MW on January 1 2005"],
        content_threshold=0.55,
    )
    assert result["recall_at_k"] == 1.0


def test_content_fallback_low_overlap_miss():
    """Low text overlap should not count as a hit."""
    result = compute_retrieval_metrics_with_content_fallback(
        retrieved_ids=["x"],
        relevant_ids=["a"],
        k=1,
        retrieved_texts=["Household consumed 1.234 kW average."],
        relevant_texts=["Zone 1 grid demand was 15000 MW peak winter."],
        content_threshold=0.55,
    )
    assert result["recall_at_k"] == 0.0


def test_content_fallback_none_texts():
    """When texts are None, falls back to ID-only matching."""
    result = compute_retrieval_metrics_with_content_fallback(
        retrieved_ids=["x"],
        relevant_ids=["a"],
        k=1,
        retrieved_texts=None,
        relevant_texts=None,
    )
    assert result["recall_at_k"] == 0.0


# -- _split_sentences ---------------------------------------------------------

def test_split_sentences_basic():
    """Sentences are split on period-space boundaries."""
    sentences = _split_sentences("First sentence here. Second sentence here. Third one here.")
    assert len(sentences) == 3


def test_split_sentences_filters_short():
    """Very short fragments (< 3 words) are filtered out."""
    sentences = _split_sentences("OK. This is a real sentence.")
    # "OK" has 1 word, should be filtered
    assert len(sentences) == 1


def test_split_sentences_empty():
    """Empty input returns empty list."""
    assert _split_sentences("") == []
